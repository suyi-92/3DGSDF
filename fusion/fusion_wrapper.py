from __future__ import annotations

"""
Fusion wrapper combining Gaussian Splatting and NeuS.

联合融合包装器，整合 3DGS 和 NeuS 的训练与交互。
"""

from pathlib import Path
from typing import Any, Callable, Dict, Optional

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

from .common import APIRegistry, ExchangeBus, SceneSpec, MutableHandle
from .data_service import DataService
from .gaussian_adapter import GaussianSplattingAdapter
from .neus_adapter import NeuSAdapter


class FusionWrapper:
    """
    High-level orchestrator that combines adapters, API registry, and exchange bus.

    高层调度器，组合适配器、API 注册表与交换总线。
    """

    def __init__(
        self,
        spec: SceneSpec,
        gaussian_repo: Path,
        neus_repo: Path,
        data_service: DataService,
        gaussian_cfg: Dict[str, Any],
        neus_cfg: Dict[str, Any],
        fusion_cfg: Optional[Dict[str, Any]] = None,
    ):
        """
        Wire up adapters with shared data service and configuration.

        将两个适配器与共享数据服务及配置绑定。
        """
        self.spec = spec
        self.registry = APIRegistry()
        self.bus = ExchangeBus()
        self.data_service = data_service
        self.fusion_cfg = fusion_cfg or {}
        depth_cfg = self.fusion_cfg.get("depth_guidance", {})
        sdf_cfg = self.fusion_cfg.get("sdf_guidance", {})
        geom_cfg = self.fusion_cfg.get("geom_loss", {})
        self.depth_guidance_cfg = {
            "k": float(depth_cfg.get("k", 3.0)),
            "min_near": float(depth_cfg.get("min_near", 0.01)),
            "max_far": float(depth_cfg.get("max_far", 100.0)),
            "max_age": int(
                depth_cfg.get("max_age", 1000)
            ),  # 增加默认缓存有效期至 1000 步
            "loss_weight": float(depth_cfg.get("loss_weight", 1.0)),
        }
        self.sdf_guidance_cfg = {
            "sigma": float(sdf_cfg.get("sigma", 0.5)),
            "omega_g": float(sdf_cfg.get("omega_g", 1.0)),
            "omega_p": float(sdf_cfg.get("omega_p", 0.5)),
            "tau_g": float(sdf_cfg.get("tau_g", 1e-3)),
            "tau_p": float(sdf_cfg.get("tau_p", 0.01)),
        }
        self.geom_loss_cfg = {
            "depth_w": float(geom_cfg.get("depth_w", 1.0)),
            "normal_w": float(geom_cfg.get("normal_w", 0.1)),
            "eps": float(geom_cfg.get("eps", 1e-6)),
        }
        self.depth_cache: Dict[Any, Dict[str, Any]] = {}
        self._depth_cache_hits = 0
        self._depth_cache_requests = 0
        self.bus.subscribe("gaussian.render_outputs", self._on_gaussian_render)
        self.gaussian = GaussianSplattingAdapter(
            gaussian_repo, self.registry, self.bus, data_service, gaussian_cfg
        )
        self.neus = NeuSAdapter(
            neus_repo, self.registry, self.bus, data_service, neus_cfg
        )
        self.neus.depth_cache = self.depth_cache
        self.neus.depth_guidance_cfg = self.depth_guidance_cfg
        self.neus.sdf_guidance_cfg = self.sdf_guidance_cfg
        self.neus.geom_loss_cfg = self.geom_loss_cfg
        self._joint_iter = 0

        # Statistics tracking for monitoring and debugging
        self._stats = {
            "depth_cache_size": 0,
            "depth_hit_rate": 0.0,
            "avg_sampling_window": 0.0,
            "densify_count": 0,
            "prune_count": 0,
            "geom_loss_depth": 0.0,
            "geom_loss_normal": 0.0,
            "last_neus_ray_batch": 0,
            "last_gaussian_ray_batch": 0,
        }

    def bootstrap(self):
        """
        Initialize both adapters.

        同时初始化两个适配器。
        """
        self.gaussian.bootstrap(self.spec)
        neus_data_root = Path(self.spec.shared_workspace) / "neus_data"
        neus_conf_root = Path(self.spec.shared_workspace) / "neus_confs"
        scene_label = Path(self.spec.scene_name).name
        neus_data = self.data_service.materialize_neus_scene(
            neus_data_root, scene_label
        )
        neus_conf_root.mkdir(parents=True, exist_ok=True)
        conf_override = self._prepare_neus_conf(neus_data["scene_dir"], neus_conf_root)
        self.spec.neus_conf_path = str(conf_override)
        self.spec.neus_case = scene_label
        self.neus.bootstrap(self.spec)

    def _on_gaussian_render(self, payload: Dict[str, Any]):
        """
        Cache gaussian-rendered depth/normal maps for cross-model guidance.

        缓存 3DGS 渲染的深度/法线，供 NeuS 采样引导使用。
        """

        if not isinstance(payload, dict):
            return

        # Extract primary key and alternative keys
        key = payload.get("camera_id") or payload.get("image_name")
        if key is None:
            return

        entry = {
            "depth": payload.get("depth"),
            "normal": payload.get("normal"),
            "iteration": payload.get("iteration", 0),
        }

        # Store under primary key
        self.depth_cache[key] = entry

        # CRITICAL: Also store under alternative keys for maximum compatibility
        alternative_keys = payload.get("alternative_keys", [])
        for alt_key in alternative_keys:
            if alt_key is not None and alt_key != key:
                self.depth_cache[alt_key] = entry

        # DEBUG: Log cache updates periodically
        iteration = payload.get("iteration", 0)
        if iteration <= 5 or iteration % 100 == 0:
            all_keys = [key] + alternative_keys
            print(
                f"  [Cache Update] Stored depth/normal under keys: {all_keys}, cache_size={len(self.depth_cache)}"
            )

    def _prepare_neus_conf(self, data_dir: Path, conf_dir: Path) -> Path:
        """
        Create a scene-specific NeuS config pointing to the generated dataset.

        生成指向新数据目录的 NeuS 配置副本。
        """
        base_conf = Path(self.spec.neus_conf_path)
        text = base_conf.read_text()
        text = text.replace("CASE_NAME", self.spec.scene_name)
        data_override = (
            f'\ndataset.data_dir = "{data_dir.as_posix()}"\n'
            f'general.base_exp_dir = "{(Path(self.spec.shared_workspace) / "neus_exp").as_posix()}"\n'
        )
        out_path = conf_dir / f"{self.spec.scene_name}.conf"
        out_path.write_text(text + "\n" + data_override)
        return out_path

    def mutable(self, model: str, component: str) -> MutableHandle:
        """
        Forward mutable handle request to the designated adapter.

        将可变句柄请求转发给指定适配器。
        """
        adapter = self.gaussian if model == "gaussian" else self.neus
        if not hasattr(adapter, "mutable"):
            raise AttributeError(f"Adapter '{model}' does not expose mutability.")
        return adapter.mutable(component)

    def register_api(self, name: str, func: Callable[..., Any], description: str):
        """
        Register a fusion-level API shortcut.

        在融合层注册新的 API 入口。
        """
        self.registry.register(name, func, description)

    def describe_apis(self) -> Dict[str, str]:
        """
        Return descriptions for all registered APIs.

        返回所有接口的描述信息。
        """
        return self.registry.describe()

    def neus_to_gaussian(self, mesh_resolution=512, threshold=0.0):
        """
        Export NeuS mesh then import it into the Gaussian model.

        将 NeuS 网格导出并导入到高斯模型中。
        """
        mesh_path = self.neus.export_mesh(mesh_resolution, threshold)
        self.gaussian.import_surface(mesh_path)

    def gaussian_to_neus(self, camera_spec) -> torch.Tensor:
        """
        Render with Gaussians and feed results back to NeuS.

        使用高斯渲染并将结果反馈给 NeuS。
        """
        render = self.gaussian.render(camera_spec)
        self.neus.inject_supervision({"render": render, "camera": camera_spec})
        return render

    def _sdf_guided_gaussian_update(self):
        """
        Use NeuS SDF predictions to guide Gaussian densify/prune.
        Follows the pattern from train.py lines 164-174.

        利用 NeuS 的 SDF 结果对高斯进行生长/修剪引导。
        遵循 train.py 第 164-174 行的模式。
        """
        if torch is None:
            return  # 环境未安装 torch，无法执行
        if (
            getattr(self.neus, "runner", None) is None
            or getattr(self.gaussian, "gaussians", None) is None
            or getattr(self.gaussian, "scene", None) is None
        ):
            return  # 任一模块未初始化完成时直接跳过

        gaussians = self.gaussian.gaussians  # 取出高斯模型实例
        opt = getattr(self.gaussian, "_opt", None)  # 训练配置
        iteration = getattr(self.gaussian, "_iteration", 0)  # 当前迭代步数

        # CRITICAL: Only apply SDF-guided densify in proper window and with proper frequency
        # Standard 3DGS densifies every 100 steps from iteration 500 to 15000
        if opt is None:
            return  # 若无优化器配置则不执行

        # Check densification window
        if iteration < 500 or iteration >= opt.densify_until_iter:
            return  # 不在允许的迭代区间内

        # REDUCE FREQUENCY: Only apply every 100 steps (matching standard 3DGS)
        if iteration % 100 != 0:
            return  # 控制频率，与标准 3DGS 保持一致

        with torch.no_grad():  # 推理阶段不记录梯度，避免额外开销
            xyz = gaussians.get_xyz  # 读取高斯中心坐标
            if xyz is None or xyz.numel() == 0:
                return  # 没有有效点则跳过

            # Evaluate SDF at gaussian centers
            sdf = self.neus.evaluate_sdf(xyz)  # 在高斯中心查询 NeuS SDF
            if not isinstance(sdf, torch.Tensor):
                return  # 未返回张量则不继续

            # Compute μ(s) = exp(-s²/(2σ²)) to weight points near the surface
            sigma = self.sdf_guidance_cfg.get("sigma", 0.5)
            mu = torch.exp(
                -((sdf**2) / (2 * max(sigma, 1e-6) ** 2))
            )  # 计算靠近表面的权重

            # Ensure mu is 1D tensor [N]
            if mu.ndim > 1:
                mu = mu.squeeze()  # 压缩维度确保形状正确

            # Get accumulated gradients (from train.py line 167: add_densification_stats)
            grad_accum = getattr(
                gaussians, "xyz_gradient_accum", None
            )  # 累积的坐标梯度
            denom = getattr(gaussians, "denom", None)  # 归一化因子

            # CRITICAL: Must have valid gradient accumulation
            if grad_accum is None or denom is None or denom.sum() == 0:
                # No valid gradients yet, skip this iteration
                return

            # Normalize accumulated gradients
            grads = grad_accum / (denom + 1e-9)  # 得到平均梯度

            # Ensure gradient shape is [N, 3]
            if grads.ndim == 2 and grads.shape[-1] != 3:
                grads = torch.norm(grads, dim=-1, keepdim=True).expand(-1, 3)
            elif grads.ndim != 2:
                # Handle unexpected gradient shapes
                grads = (
                    grads.reshape(-1, 3)
                    if grads.numel() >= xyz.shape[0] * 3
                    else mu.unsqueeze(-1).expand(-1, 3)
                )

            # Compute enhanced gradient: ε_g = ∇g + ω_g * μ(s)
            # REDUCED omega_g to prevent over-densification
            omega_g = self.sdf_guidance_cfg.get("omega_g", 0.3)  # Reduced from 1.0
            # Ensure mu broadcasting works correctly: [N] -> [N, 1] -> [N, 3]
            mu_expanded = mu.unsqueeze(-1).expand_as(grads)  # 扩展 mu 方便与梯度广播
            # ONLY enhance gradients near surface (mu > 0.5)
            eps_g = grads + omega_g * mu_expanded * (mu > 0.5).float().unsqueeze(
                -1
            )  # 只在靠近表面时强化梯度

            # Densify: follow train.py lines 169-171
            tau_g = self.sdf_guidance_cfg.get(
                "tau_g", 0.0002
            )  # Aligned with densify_grad_threshold，densify 梯度阈值
            size_threshold = (
                20 if opt and iteration > opt.opacity_reset_interval else None
            )  # 根据迭代步决定屏幕尺寸阈值

            # Use the official densify_and_prune method from GaussianModel
            # This handles both densification (clone/split) and pruning in one call
            if hasattr(gaussians, "densify_and_prune"):
                # Calculate gradient magnitudes for densification criterion
                grad_magnitude = torch.norm(
                    eps_g, dim=-1, keepdim=True
                )  # 梯度模长用于阈值判断

                # INCREASED min opacity threshold for more aggressive pruning
                min_opacity = self.sdf_guidance_cfg.get(
                    "tau_p", 0.01
                )  # Increased from 0.005，提升剪枝力度

                # Track counts before operation
                num_before = gaussians.get_xyz.shape[0]  # 记录操作前点数

                # Get radii from max_radii2D (already tracked by GaussianModel)
                radii = getattr(gaussians, "max_radii2D", None)
                if radii is None:
                    # Fallback: create dummy radii if not available
                    radii = torch.zeros(num_before, device=xyz.device)

                # Call official densify_and_prune
                gaussians.densify_and_prune(
                    max_grad=tau_g,
                    min_opacity=min_opacity,
                    extent=self.gaussian.scene.cameras_extent,
                    max_screen_size=size_threshold,
                    radii=radii,
                )

                # Update statistics
                num_after = gaussians.get_xyz.shape[0]  # 操作后的点数
                if num_after > num_before:
                    self._stats["densify_count"] += num_after - num_before  # 统计新增点
                elif num_after < num_before:
                    self._stats["prune_count"] += num_before - num_after  # 统计剪枝点

            else:
                # Fallback: separate densify and prune
                if torch.any(torch.norm(eps_g, dim=-1) > tau_g) and hasattr(
                    gaussians, "densify_and_clone"
                ):
                    num_before = gaussians.get_xyz.shape[0]  # 记录操作前点数
                    gaussians.densify_and_clone(
                        eps_g, tau_g, self.gaussian.scene.cameras_extent
                    )
                    self._stats["densify_count"] += (
                        gaussians.get_xyz.shape[0] - num_before  # 累加新增的高斯数
                    )

                # Prune condition: ε_p = σ_a - ω_p * (1 - μ(s))
                opacity = gaussians.get_opacity  # 当前不透明度
                if opacity.ndim == 1:
                    opacity = opacity[:, None]  # 扩展维度与 mu 对齐
                omega_p = self.sdf_guidance_cfg.get("omega_p", 0.5)
                eps_p = opacity - omega_p * (1 - mu.unsqueeze(-1))  # 计算剪枝指标
                prune_mask = eps_p.squeeze() < self.sdf_guidance_cfg.get(
                    "tau_p", 0.005
                )  # 低于阈值的点将被剪枝

                if torch.any(prune_mask) and hasattr(gaussians, "prune_points"):
                    self._stats[
                        "prune_count"
                    ] += prune_mask.sum().item()  # 记录剪枝数量
                    gaussians.prune_points(prune_mask)  # 执行剪枝操作

    def get_statistics(self) -> Dict[str, Any]:
        """
        Return current fusion statistics for monitoring.

        返回当前融合统计信息供监控。
        """
        stats = self._stats.copy()

        # Update depth cache statistics
        stats["depth_cache_size"] = len(self.depth_cache)

        # Calculate depth guidance hit rate
        if self.neus._depth_requests > 0:
            stats["depth_hit_rate"] = self.neus._depth_hits / self.neus._depth_requests
        else:
            stats["depth_hit_rate"] = 0.0

        # Add model-specific stats
        if hasattr(self.gaussian, "gaussians") and self.gaussian.gaussians:
            stats["num_gaussians"] = self.gaussian.gaussians.get_xyz.shape[0]
        if hasattr(self.neus, "runner") and self.neus.runner:
            stats["neus_iteration"] = self.neus.runner.iter_step

        return stats

    def print_statistics(self, interval: int = 100):
        """
        Print fusion statistics every N iterations.

        每 N 步打印融合统计信息。
        """
        if self._joint_iter % interval == 0:
            stats = self.get_statistics()
            print(f"\n=== Fusion Statistics (iter {self._joint_iter}) ===")
            print(f"Depth Cache: {stats['depth_cache_size']} entries")
            print(f"Depth Hit Rate: {stats['depth_hit_rate']:.2%}")
            print(f"Num Gaussians: {stats.get('num_gaussians', 'N/A')}")
            print(f"Densify/Prune: +{stats['densify_count']} / -{stats['prune_count']}")
            print(f"NeuS Iter: {stats.get('neus_iteration', 'N/A')}")
            print(
                "Ray Batches (NeuS/GS): "
                f"{stats.get('last_neus_ray_batch', 0)} / {stats.get('last_gaussian_ray_batch', 0)}"
            )
            print("=" * 50)

    def joint_step(
        self,
        mesh_every=100,
        log_every=100,
        callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ):
        """
        Run one fused iteration coordinating NeuS and GS training.
        Follows patterns from train.py and exp_runner.py.

        执行一次融合迭代：NeuS 训练、可选网格同步、再训练 GS。
        """
        self._joint_iter += 1

        # Step 0: Fetch shared ray batches for NeuS / Gaussian from the data service
        sampling_cfg = self.fusion_cfg.get("ray_sampling", {})
        neus_sampling_cfg = (
            sampling_cfg.get("neus", sampling_cfg)
            if isinstance(sampling_cfg, dict)
            else {}
        )
        gaussian_sampling_cfg = (
            sampling_cfg.get("gaussian", sampling_cfg)
            if isinstance(sampling_cfg, dict)
            else {}
        )

        neus_batch_size = neus_sampling_cfg.get(
            "batch_size", getattr(getattr(self.neus, "runner", None), "batch_size", None)
        )
        gaussian_batch_size = gaussian_sampling_cfg.get("batch_size", None)
        if gaussian_batch_size is None:
            gaussian_batch_size = neus_batch_size

        neus_ray_batch = None
        gaussian_ray_batch = None

        if neus_batch_size:
            try:
                neus_kwargs = {k: v for k, v in neus_sampling_cfg.items() if k != "batch_size"}
                neus_ray_batch = self.data_service.sample_rays(
                    neus_batch_size, consumer="neus", **neus_kwargs
                )
                self._stats["last_neus_ray_batch"] = neus_batch_size
            except Exception as e:
                print(f"[Fusion] NeuS ray sampling failed (fallback to adapter sampler): {e}")
                self._stats["last_neus_ray_batch"] = 0

        if gaussian_batch_size:
            try:
                gs_kwargs = {
                    k: v for k, v in gaussian_sampling_cfg.items() if k != "batch_size"
                }
                gaussian_ray_batch = self.data_service.sample_rays(
                    gaussian_batch_size, consumer="gaussian", **gs_kwargs
                )
                self._stats["last_gaussian_ray_batch"] = gaussian_batch_size
            except Exception as e:
                print(
                    f"[Fusion] Gaussian ray sampling failed (fallback to adapter sampler): {e}"
                )
                self._stats["last_gaussian_ray_batch"] = 0

        # Step 1: Train NeuS with depth-guided sampling + geometric supervision
        neus_state = self.neus.train_step(ray_batch=neus_ray_batch)

        # Step 2: Periodically export NeuS mesh and import to Gaussians
        if mesh_every > 0 and neus_state.iteration % mesh_every == 0:
            try:
                self.neus_to_gaussian()
            except Exception as e:
                print(f"Warning: mesh sync failed at iter {self._joint_iter}: {e}")

        # Step 3: Train Gaussian Splatting (publishes depth/normal via bus)
        gaussian_state = self.gaussian.train_step(ray_batch=gaussian_ray_batch)

        # Step 4: Apply SDF-guided densify/prune
        self._sdf_guided_gaussian_update()

        # Step 5: Print statistics
        if log_every > 0:
            self.print_statistics(interval=log_every)

        # Build payload for callback
        payload = {
            "neus": neus_state,
            "gaussian": gaussian_state,
            "fusion_step": self._joint_iter,
            "statistics": self.get_statistics(),
        }

        if callback:
            callback(payload)

        return payload
