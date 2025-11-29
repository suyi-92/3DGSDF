from __future__ import annotations

"""
NeuS adapter for fusion wrapper.

NeuS 包装器模块，提供 SDF 训练和评估接口。
"""

import sys
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

from .common import (
    AdapterBase,
    APIRegistry,
    ExchangeBus,
    SceneSpec,
    NeuSIterationState,
    MutableHandle,
    RayBatch,
)
from .data_service import DataService


class NeuSAdapter(AdapterBase):
    """
    Wraps NeuS Runner with the same API-style interface.

    使用相同的 API 风格接口封装 NeuS Runner。
    """

    def __init__(
        self,
        repo_root: Path,
        registry: APIRegistry,
        bus: ExchangeBus,
        data_service: DataService,
        config: Dict[str, Any],
    ):
        """
        Load NeuS runner and keep configuration references.

        加载 NeuS Runner 并保存配置信息。
        """
        super().__init__("neus", registry, bus, data_service)
        self.repo_root = Path(repo_root)
        if str(self.repo_root) not in sys.path:
            sys.path.append(str(self.repo_root))

        from NeuS.exp_runner import Runner

        self.Runner = Runner
        self.config = config
        self.runner: Optional[Runner] = None
        self.depth_cache: Dict[Any, Dict[str, Any]] = {}
        self.depth_guidance_cfg: Dict[str, Any] = {}
        self.sdf_guidance_cfg: Dict[str, Any] = {}
        self.geom_loss_cfg: Dict[str, Any] = {}
        self._depth_hits = 0
        self._depth_requests = 0

    def bootstrap(self, spec: SceneSpec):
        """
        Instantiate NeuS Runner using provided configuration.

        基于配置初始化 NeuS Runner。
        """
        if torch is None:
            raise RuntimeError("PyTorch is required for NeuSAdapter.")
        conf_path = str(spec.neus_conf_path)
        self.runner = self.Runner(
            conf_path,
            mode=self.config.get("mode", "train"),
            case=spec.neus_case,
            is_continue=self.config.get("resume", False),
        )

        self.register_api(
            "train_step",
            self.train_step,
            "执行一步 NeuS 优化。",
        )
        self.register_api(
            "export_surface",
            self.export_mesh,
            "通过 marching cubes 提取 NeuS 网格。",
        )
        self.register_api(
            "evaluate_sdf",
            self.evaluate_sdf,
            "在给定点评估 NeuS SDF（无梯度）。",
        )
        self.register_api(
            "inject_supervision",
            self.inject_supervision,
            "提供额外的监督（例如 Gaussian 渲染）。",
        )

    def load_checkpoint(self, ckpt_path: Path):
        """
        Load NeuS checkpoint from a .pth file recorded during prewarm.

        Args:
            ckpt_path: Full path to ckpt_xxxxxx.pth.
        """

        if torch is None or self.runner is None:
            raise RuntimeError("NeuSAdapter not bootstrapped.")

        ckpt_path = Path(ckpt_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"NeuS checkpoint not found: {ckpt_path}")

        # Ensure runner.base_exp_dir points to checkpoint parent
        checkpoints_dir = ckpt_path.parent
        base_exp_dir = checkpoints_dir.parent
        self.runner.base_exp_dir = str(base_exp_dir)
        self.runner.load_checkpoint(ckpt_path.name)

    def mutable(self, attr: str) -> MutableHandle:
        """
        Return a handle to mutate an internal NeuS attribute.

        返回可用于修改 NeuS 内部属性的句柄。
        """
        if not hasattr(self.runner, attr):
            raise AttributeError(attr)
        return MutableHandle(getattr(self.runner, attr))

    def _camera_key_from_idx(self, idx: int) -> list:
        """
        Generate multiple possible cache keys for a given camera index.
        Returns a list of candidate keys to try in priority order.

        为给定相机索引生成多个可能的缓存键。
        """
        ds = getattr(self.runner, "dataset", None)
        keys = [idx]  # Always try numeric index first

        if ds and hasattr(ds, "images_lis"):
            try:
                from pathlib import Path

                image_path = ds.images_lis[idx]

                # Strategy 1: Stem only (e.g., "000001")
                stem = Path(image_path).stem
                keys.append(stem)

                # Strategy 2: Full filename (e.g., "000001.png")
                name = Path(image_path).name
                keys.append(name)

                # Strategy 3: Try to convert stem to int if numeric
                try:
                    numeric_stem = int(stem)
                    if numeric_stem != idx:
                        keys.append(numeric_stem)
                except ValueError:
                    pass

            except Exception:
                pass

        return keys

    def _sample_rays_with_pixels(
        self, idx: int, batch_size: int
    ) -> Tuple[Any, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Sample rays together with pixel coordinates (stays on the dataset device).

        为当前视角采样射线并返回对应的像素坐标（保持在数据集统一设备上）
        """
        if torch is None or self.runner is None:
            raise RuntimeError("PyTorch is required for NeuSAdapter.")
        ds = self.runner.dataset

        # TODO: 可以做更复杂的采样器，这里先用最简单的随机采样
        try:
            # 以内参的 device 作为统一设备（一般是 cuda:0）
            device = ds.intrinsics_all_inv.device

            # 1. 在统一 device 上采样像素坐标
            pixels_x = torch.randint(low=0, high=ds.W, size=[batch_size], device=device)
            pixels_y = torch.randint(low=0, high=ds.H, size=[batch_size], device=device)

            # 2. 从 image / mask 中取颜色和 mask
            color = ds.images[idx].to(device)[(pixels_y, pixels_x)]
            mask = ds.masks[idx].to(device)[(pixels_y, pixels_x)]

            # 3. 像素坐标 -> 归一化相机坐标
            p = torch.stack(
                [pixels_x, pixels_y, torch.ones_like(pixels_y, device=device)], dim=-1
            ).float()
            p = torch.matmul(
                ds.intrinsics_all_inv[idx, None, :3, :3].to(device), p[:, :, None]
            ).squeeze(-1)

            # 4. 得到射线方向（世界坐标）
            rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)
            rays_v = torch.matmul(
                ds.pose_all[idx, None, :3, :3].to(device), rays_v[:, :, None]
            ).squeeze(-1)

            # 5. 射线起点（世界坐标）
            rays_o = ds.pose_all[idx, None, :3, 3].to(device).expand_as(rays_v)

            # 6. 最终拼接，全部在同一个 device 上，不再来回 .cpu() / .cuda()
            data = torch.cat([rays_o, rays_v, color, mask[:, :1]], dim=-1)
            return data, pixels_x, pixels_y
        except Exception:
            data = ds.gen_random_rays_at(idx, batch_size)
            return data, None, None

    def _override_near_far_with_depth(
        self,
        idx: int,
        rays_o: torch.Tensor,
        rays_d: torch.Tensor,
        pixels_x: Optional[torch.Tensor],
        pixels_y: Optional[torch.Tensor],
        near: torch.Tensor,
        far: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply GS depth-guided adaptive sampling window.
        Follows README section 1: GS → SDF depth-guided sampling.

        应用 GS 深度引导的自适应采样窗口。
        """
        if torch is None or self.depth_cache is None:
            return near, far

        self._depth_requests += 1

        # Try multiple candidate keys (now returns a list)
        candidates = self._camera_key_from_idx(idx)
        cache_entry = None
        matched_key = None

        # DEBUG: Log lookup attempt periodically
        debug_mode = (
            getattr(self.runner, "iter_step", 0) <= 5
            or getattr(self.runner, "iter_step", 0) % 100 == 0
        )

        if debug_mode:
            print(
                f"  [NeuS Lookup] Trying keys: {candidates}, cache_keys={list(self.depth_cache.keys())[:5]}..."
            )

        for key in candidates:
            if key in self.depth_cache:
                cache_entry = self.depth_cache[key]
                matched_key = key
                break

        if cache_entry is None:
            if debug_mode:
                print(
                    f"  [NeuS Lookup] ❌ MISS - No cache entry found for keys {candidates}"
                )
            return near, far

        if debug_mode:
            print(f"  [NeuS Lookup] ✓ HIT - Found cache entry with key={matched_key}")

        # 检查缓存新鲜度 (max_age 步数)
        # Check cache freshness (max_age steps)
        max_age = self.depth_guidance_cfg.get("max_age", None)
        if max_age is not None and isinstance(max_age, (int, float)):
            step = cache_entry.get("iteration")
            if (
                isinstance(step, (int, float))
                and getattr(self.runner, "iter_step", None) is not None
            ):
                age = self.runner.iter_step - step
                if age > max_age:
                    if debug_mode:
                        print(
                            f"  [NeuS Lookup] ⚠ STALE - Cache age {age} > max_age {max_age}"
                        )
                    # 宽容策略：如果缓存稍微过期但没有更好的选择，仍然使用它（可选）
                    # 这里我们严格遵守 max_age，但在 fusion_wrapper 中会增大默认值
                    return near, far

        depth = cache_entry.get("depth")
        if depth is None or pixels_x is None or pixels_y is None:
            if debug_mode:
                print(
                    f"  [NeuS Lookup] ⚠ NO DATA - depth={depth is not None}, pixels={pixels_x is not None}"
                )
            return near, far

        depth_tensor = (
            depth if isinstance(depth, torch.Tensor) else torch.as_tensor(depth)
        )

        # 修复：处理深度张量的维度问题 [C, H, W] -> [H, W]
        if depth_tensor.ndim == 3:
            depth_tensor = depth_tensor.squeeze(0)

        try:
            # 在像素位置采样深度
            # Sample depth at pixel locations
            sampled_depth = depth_tensor[(pixels_y.cpu(), pixels_x.cpu())]
        except Exception as e:
            if debug_mode:
                print(
                    f"  [NeuS Lookup] ⚠ SAMPLE FAILED - {e} (Shape: {depth_tensor.shape})"
                )
            return near, far

        if sampled_depth.ndim == 1:
            sampled_depth = sampled_depth[:, None]
        sampled_depth = sampled_depth.to(rays_o.device)

        try:
            # Compute 3D centers: center = o + D * v
            center = rays_o + sampled_depth * rays_d

            # Evaluate SDF at centers (without gradient)
            sdf = self.evaluate_sdf(center)
            if sdf is None:
                if debug_mode:
                    print(f"  [NeuS Lookup] ⚠ SDF FAILED")
                return near, far

        except Exception as e:
            if debug_mode:
                print(f"  [NeuS Lookup] ⚠ SDF EXCEPTION - {e}")
            return near, far

        # Adaptive window: near = D - k*|s|, far = D + k*|s|
        k = self.depth_guidance_cfg.get("k", 3.0)
        delta = k * sdf.abs()

        # Clamp to safe range
        min_near = self.depth_guidance_cfg.get("min_near", 0.01)
        max_far = self.depth_guidance_cfg.get("max_far", 100.0)

        near_new = torch.clamp(sampled_depth - delta, min=min_near)
        far_new = torch.clamp(sampled_depth + delta, max=max_far)

        self._depth_hits += 1

        if debug_mode:
            print(
                f"  [NeuS Lookup] ✓ SUCCESS - Applied depth guidance, hit_rate={self._depth_hits}/{self._depth_requests}"
            )

        return near_new, far_new

    def _sample_gs_depth_normal(
        self,
        idx: int,
        pixels_x: Optional[torch.Tensor],
        pixels_y: Optional[torch.Tensor],
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Fetch depth/normal supervision samples from cached Gaussian results.

        The function matches the current camera index to possible cache keys
        (index, filename stem, etc.) and, if a cache entry exists, extracts the
        depth and normal values at the given pixel coordinates. When either the
        cache or pixel indices are unavailable, it gracefully returns `None` for
        both outputs to indicate missing supervision.

        从缓存的 Gaussian 结果中获取深度和法线监督样本。

        该函数会将当前的相机索引映射到可能的缓存键（索引、文件名 stem 等），
        如果找到对应的缓存条目，就会根据给定的像素坐标提取深度和法线值。
        当缓存缺失或像素索引不可用时，会优雅地返回 `None` 表示缺少监督。
        """

        if pixels_x is None or pixels_y is None:
            return None, None

        # Try multiple candidate keys
        candidates = self._camera_key_from_idx(idx)
        cache_entry = None
        for key in candidates:
            if key in self.depth_cache:
                cache_entry = self.depth_cache[key]
                break

        if cache_entry is None:
            return None, None

        depth = cache_entry.get("depth")
        normal = cache_entry.get("normal")
        try:
            depth_tensor = (
                depth if isinstance(depth, torch.Tensor) else torch.as_tensor(depth)
            )
            depth_samples = depth_tensor[(pixels_y.cpu(), pixels_x.cpu())]
        except Exception:
            depth_samples = None
        normal_samples = None
        if normal is not None:
            try:
                normal_tensor = (
                    normal
                    if isinstance(normal, torch.Tensor)
                    else torch.as_tensor(normal)
                )
                normal_samples = normal_tensor[(pixels_y.cpu(), pixels_x.cpu())]
            except Exception:
                normal_samples = None
        return depth_samples, normal_samples

    def train_step(
        self,
        ray_batch: Optional[RayBatch] = None,
        callback: Optional[Callable[[NeuSIterationState], None]] = None,
    ):
        """
        Execute one NeuS optimization step and optionally report metrics.
        Follows NeuS/exp_runner.py train() loop pattern (lines 109-216).

        执行一次 NeuS 优化,并可通过回调汇报指标。
        """
        if torch is None:
            raise RuntimeError("PyTorch is required for NeuSAdapter.")
        r = self.runner

        # 更新学习率调度（保持与 exp_runner 训练循环一致）
        r.update_learning_rate()

        # 获取当前要采样的图像索引（按照乱序列表轮询）
        image_perm = r.get_image_perm()
        idx = image_perm[r.iter_step % len(image_perm)]

        # 在该视角下采样光线：优先使用外部采样器提供的 batch
        if ray_batch is not None:
            if self.runner and torch.is_tensor(ray_batch.origins):
                device = self.runner.device
            else:
                device = getattr(r, "device", None)
            rays_o = torch.as_tensor(ray_batch.origins, device=device)
            rays_d = torch.as_tensor(ray_batch.directions, device=device)
            true_rgb = torch.as_tensor(ray_batch.colors, device=device)
            meta = ray_batch.meta or {}
            mask = torch.as_tensor(
                meta.get("mask", torch.ones_like(true_rgb[..., :1])), device=device
            )
            idx = int(meta.get("image_idx", idx))
            pixels_x = meta.get("pixels_x")
            pixels_y = meta.get("pixels_y")
            if isinstance(pixels_x, torch.Tensor):
                pixels_x = pixels_x.to(device)
            if isinstance(pixels_y, torch.Tensor):
                pixels_y = pixels_y.to(device)
            if r.iter_step <= 5 or r.iter_step % 100 == 0:
                print(
                    f"[NeuS] Using external ray batch of size {rays_o.shape[0]} at iter {r.iter_step}"
                )
        else:
            data, pixels_x, pixels_y = self._sample_rays_with_pixels(idx, r.batch_size)
            rays_o, rays_d, true_rgb, mask = (
                data[:, :3],
                data[:, 3:6],
                data[:, 6:9],
                data[:, 9:10],
            )

        # 根据球边界计算该批光线的近远平面
        meta = ray_batch.meta if ray_batch is not None else None
        if meta is not None and (
            meta.get("near") is not None and meta.get("far") is not None
        ):
            near = torch.as_tensor(meta["near"], device=r.device)
            far = torch.as_tensor(meta["far"], device=r.device)
        else:
            near, far = r.dataset.near_far_from_sphere(rays_o, rays_d)

        # 如果有 GS 深度缓存，利用深度引导动态收缩采样窗口
        near, far = self._override_near_far_with_depth(
            idx, rays_o, rays_d, pixels_x, pixels_y, near, far
        )

        # 运行 NeuS 渲染器，得到颜色、权重、梯度等前向结果
        background = torch.ones([1, 3], device=r.device) if r.use_white_bkgd else None
        z_vals_override = None
        if meta is not None and meta.get("z_vals") is not None:
            z_vals_override = torch.as_tensor(meta["z_vals"], device=r.device)
            if z_vals_override.ndim == 1:
                z_vals_override = z_vals_override[None, :].expand(rays_o.shape[0], -1)

        render_out = r.renderer.render(
            rays_o,
            rays_d,
            near,
            far,
            z_vals=z_vals_override,
            cos_anneal_ratio=r.get_cos_anneal_ratio(),
            background_rgb=background,
        )

        color_fine = render_out["color_fine"]
        s_val = render_out["s_val"]
        cdf_fine = render_out["cdf_fine"]
        weight_sum = render_out["weight_sum"]
        weight_max = render_out["weight_max"]
        gradients = render_out["gradients"]
        weights = render_out["weights"]
        mid_z_vals = render_out["mid_z_vals"]
        gradient_error = render_out["gradient_error"]
        inside_sphere = render_out["inside_sphere"]

        # 检查 weights 形状以避免潜在错误
        if weights.shape[-1] != mid_z_vals.shape[-1]:
            weights = weights[..., : mid_z_vals.shape[-1]]

        # 以权重加权的 mid_z 作为渲染深度，用于几何监督
        mu_z = (weights * mid_z_vals).sum(dim=-1, keepdim=True)

        # 用采样权重加权梯度并归一化，得到表面法线估计
        weighted_grad = (gradients * weights[..., None]).sum(dim=1)
        weighted_normal = torch.nn.functional.normalize(
            weighted_grad, dim=-1, eps=self.geom_loss_cfg.get("eps", 1e-6)
        )

        # 从 GS 深度缓存中抽取对应像素的深度/法线，作为外部几何监督
        depth_gt, normal_gt = self._sample_gs_depth_normal(idx, pixels_x, pixels_y)

        # 颜色重建误差：可选择使用掩码权重
        if r.mask_weight > 0.0:
            mask = (mask > 0.5).float()
        else:
            mask = torch.ones_like(mask)
        mask_sum = mask.sum() + 1e-5

        color_error = (color_fine - true_rgb) * mask
        color_loss = (
            torch.nn.functional.l1_loss(
                color_error, torch.zeros_like(color_error), reduction="sum"
            )
            / mask_sum
        )

        # Eikonal 约束：梯度范数逼近 1
        eikonal_loss = ((torch.norm(gradients, dim=-1) - 1.0) ** 2).mean()

        # 掩码损失：权重和与掩码的 BCE
        mask_loss = torch.nn.functional.binary_cross_entropy(
            weight_sum.clip(1e-3, 1.0 - 1e-3), mask
        )

        # 几何监督损失：深度一致性与法线一致性
        geom_loss = 0.0
        depth_loss_val = 0.0
        normal_loss_val = 0.0

        if depth_gt is not None:
            depth_gt = depth_gt.to(mu_z.device)
            if depth_gt.ndim == 1:
                depth_gt = depth_gt[:, None]
            depth_loss_val = torch.mean(torch.abs(mu_z - depth_gt))
            geom_loss = (
                geom_loss + self.geom_loss_cfg.get("depth_w", 1.0) * depth_loss_val
            )

        if normal_gt is not None:
            normal_gt = normal_gt.to(weighted_normal.device)
            # Normal consistency: 1 - cos(n_gs, n_sdf)
            normal_loss_val = torch.mean(
                1.0
                - torch.sum(
                    weighted_normal
                    * torch.nn.functional.normalize(
                        normal_gt, dim=-1, eps=self.geom_loss_cfg.get("eps", 1e-6)
                    ),
                    dim=-1,
                    keepdim=True,
                )
            )
            geom_loss = (
                geom_loss + self.geom_loss_cfg.get("normal_w", 0.1) * normal_loss_val
            )

        # 总损失：颜色 + Eikonal + 掩码 + 几何监督
        loss = (
            color_loss
            + r.igr_weight * eikonal_loss
            + r.mask_weight * mask_loss
            + geom_loss
        )

        # 反向传播并更新参数
        r.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        r.optimizer.step()

        # 递增迭代步数计数器
        r.iter_step += 1

        # 打包当前训练状态并回调/广播给外部
        state = NeuSIterationState(
            iteration=r.iter_step,
            loss=float(loss.detach().cpu()),
            color_loss=float(color_loss.detach().cpu()),
            eikonal_loss=float(eikonal_loss.detach().cpu()),
            lr=r.optimizer.param_groups[0]["lr"],
        )
        if callback:
            callback(state)
        self.bus.publish("neus.train_step", state)
        return state

    def export_mesh(self, resolution=256, threshold=0.0) -> Path:
        """
        Run marching cubes to extract a NeuS mesh and return its path.

        执行 marching cubes 导出 NeuS 网格并返回路径。
        """
        r = self.runner
        r.validate_mesh(world_space=True, resolution=resolution, threshold=threshold)
        mesh_path = Path(r.base_exp_dir) / "meshes" / f"{r.iter_step:08d}.ply"
        self.bus.publish("neus.export_surface", mesh_path)
        return mesh_path

    def inject_supervision(self, payload: Dict[str, Any]):
        """
        Placeholder hook that future tools can override to add extra losses.

        预留钩子，供未来工具注入额外监督或损失项。
        """
        self.bus.publish("neus.inject_supervision", payload)

    def evaluate_sdf(self, points: Any):
        """
        Evaluate the NeuS SDF network without tracking gradients.

        无梯度评估 NeuS 的 SDF 网络，供外部模块查询。
        """
        if torch is None:
            raise RuntimeError("PyTorch is required for NeuSAdapter.")
        if self.runner is None:
            raise RuntimeError("NeuS runner not bootstrapped.")
        with torch.no_grad():
            pts = points.to(self.runner.device)
            return self.runner.sdf_network.sdf(pts)
