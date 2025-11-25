from __future__ import annotations

"""
Gaussian Splatting adapter for fusion wrapper.

3DGS 包装器模块，提供训练、渲染和属性访问接口。
"""

import argparse
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

try:
    import trimesh
except ImportError:  # pragma: no cover
    trimesh = None

try:
    import torch
    import torch.nn.functional as F
except ImportError:  # pragma: no cover
    torch = None
    F = None

from .common import (
    AdapterBase,
    APIRegistry,
    ExchangeBus,
    SceneSpec,
    GaussianIterationState,
    MutableHandle,
)
from .data_service import DataService


class GaussianSplattingAdapter(AdapterBase):
    """
    Thin wrapper around the official Gaussian-Splatting training script that exposes
    pluggable APIs for training, rendering, export/import, etc.

    对官方 Gaussian-Splatting 训练脚本的一个轻量封装，提供可插拔的 API，用于训练、渲染、导出/导入等功能。
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
        Load Gaussian-Splatting modules and initialize state holders.

        加载高斯溅射依赖并初始化内部状态。
        """
        super().__init__("gaussian", registry, bus, data_service)
        self.repo_root = Path(repo_root)
        if str(self.repo_root) not in sys.path:
            sys.path.append(str(self.repo_root))

        from gaussian_splatting.gaussian_renderer import render
        from gaussian_splatting.utils.loss_utils import l1_loss, ssim
        from gaussian_splatting.scene import Scene
        from gaussian_splatting.scene.gaussian_model import GaussianModel
        from gaussian_splatting.arguments import (
            ModelParams,
            OptimizationParams,
            PipelineParams,
        )
        from gaussian_splatting.utils.general_utils import get_expon_lr_func

        self.Scene = Scene
        self.GaussianModel = GaussianModel
        self.ModelParams = ModelParams
        self.OptimizationParams = OptimizationParams
        self.PipelineParams = PipelineParams
        self._render = render
        self._l1 = l1_loss
        self._ssim = ssim
        self._lr_schedule = get_expon_lr_func

        self.config = config
        self.scene: Optional[Scene] = None
        self.gaussians: Optional[GaussianModel] = None
        self._background = None
        self._iteration = 0
        self._depth_weight = None
        self._view_stack: List[Any] = []
        self._view_indices: List[int] = []
        self._pipe = None
        self._opt = None
        self._dataset = None

    def bootstrap(self, spec: SceneSpec):
        """
        Instantiate Scene + GaussianModel using assets from the data service.

        使用数据服务提供的资源构建 Scene 与 GaussianModel。
        """
        if torch is None:
            raise RuntimeError("PyTorch is required for GaussianSplattingAdapter.")
        parser = argparse.ArgumentParser(add_help=False)
        # 将配置项注入 argparse 以复用参数解析逻辑
        lp = self.ModelParams(parser)
        op = self.OptimizationParams(parser)
        pp = self.PipelineParams(parser)
        args = parser.parse_args([])

        materialized = self.data_service.materialize_gaussian_scene(
            Path(spec.gaussian_source_path)
        )
        overrides = {
            "source_path": str(materialized["source"]),
            "model_path": spec.gaussian_model_path,
            "white_background": spec.white_background,
            **self.config.get("model", {}),
        }
        for k, v in overrides.items():
            setattr(args, k, v)  # 写入参数到 argparse.Namespace

        dataset = lp.extract(args)
        optim = op.extract(args)
        pipe = pp.extract(args)
        pipe.debug = self.config.get("debug", False)

        self.gaussians = self.GaussianModel(dataset.sh_degree, optim.optimizer_type)
        self.scene = self.Scene(
            dataset, self.gaussians, resolution_scales=list(spec.resolution_scales)
        )
        self.gaussians.training_setup(optim)

        self._background = torch.tensor(
            [1, 1, 1] if dataset.white_background else [0, 0, 0],
            dtype=torch.float32,
            device=spec.device,
        )
        self._depth_weight = self._lr_schedule(
            optim.depth_l1_weight_init,
            optim.depth_l1_weight_final,
            max_steps=optim.iterations,
        )
        self._pipe = pipe
        self._opt = optim
        self._dataset = dataset
        self._reset_view_stack()

        self.register_api(
            "train_step",
            self.train_step,
            "执行一步 Gaussian Splatting 优化并返回指标。",
        )
        self.register_api(
            "render",
            self.render,
            "使用当前 Gaussian 模型渲染指定相机。",
        )
        self.register_api(
            "export_surface",
            self.export_surface,
            "将当前 Gaussian 点云导出为 .ply 文件。",
        )
        self.register_api(
            "import_surface",
            self.import_surface,
            "导入网格/点云以初始化 Gaussians。",
        )

    def mutable(self, component: str) -> MutableHandle:
        """
        Expose internal state for carefully-scoped mutations.

        暴露内部状态供受控修改。
        """
        if not self.scene or not self.gaussians:
            raise RuntimeError("Adapter not bootstrapped yet.")
        targets = {
            "scene": self.scene,
            "gaussians": self.gaussians,
            "optimizer": getattr(self.gaussians, "optimizer", None),
            "exposure_optimizer": getattr(self.gaussians, "exposure_optimizer", None),
            "pipeline": self._pipe,
        }
        if component not in targets or targets[component] is None:
            raise ValueError(
                f"Component '{component}' unavailable in Gaussian adapter."
            )
        return MutableHandle(targets[component])

    def _reset_view_stack(self):
        """
        Refresh the shuffled list of training cameras.

        重建并随机化训练相机列表。
        """
        cameras = self.scene.getTrainCameras().copy()  # 复制以避免修改原始列表
        self._view_stack = cameras
        self._view_indices = list(range(len(cameras)))

    def _pick_view(self):
        """
        Pop a random camera from the stack.

        从栈中随机取出一个相机。
        """
        if not self._view_stack:
            self._reset_view_stack()
        idx = torch.randint(low=0, high=len(self._view_indices), size=(1,)).item()
        cam = self._view_stack.pop(idx)
        self._view_indices.pop(idx)
        return cam

    def _publish_render_outputs(self, cam, render_pkg: Dict[str, Any]):
        """
        Publish depth/normal maps for the current camera to the exchange bus.

        将当前相机的深度/法线发布到总线上,便于 NeuS 引导采样。"""
        if torch is None or not isinstance(render_pkg, dict):
            return
        depth = render_pkg.get("depth")
        normal = render_pkg.get("normal") or render_pkg.get("normals")

        # Extract camera identifiers - CRITICAL: Must match NeuS adapter lookup logic
        camera_name = getattr(cam, "image_name", None)
        camera_id = getattr(cam, "uid", None) or getattr(cam, "colmap_id", None)

        from pathlib import Path

        # UNIFIED KEY STRATEGY: Try multiple key formats for maximum compatibility
        # 1. Image stem (e.g., "000001" from "000001.png")
        # 2. Full image name (e.g., "000001.png")
        # 3. Numeric camera ID
        primary_key = None
        alternative_keys = []

        if camera_name:
            # Primary: stem without extension
            stem_key = Path(camera_name).stem
            primary_key = stem_key
            alternative_keys.append(camera_name)  # Full name as backup

            # Also try numeric conversion if stem is numeric
            try:
                numeric_key = int(stem_key)
                alternative_keys.append(numeric_key)
            except ValueError:
                pass
        elif camera_id is not None:
            primary_key = camera_id
        else:
            # Fallback: use iteration as key
            primary_key = f"iter_{self._iteration}"

        # Build comprehensive payload with camera metadata
        payload: Dict[str, Any] = {
            "camera_id": primary_key,  # PRIMARY: image stem for NeuS compatibility
            "alternative_keys": alternative_keys,  # FALLBACK: additional key formats
            "image_name": camera_name,
            "uid": camera_id,  # SECONDARY: keep original ID for reference
            "iteration": self._iteration,
            "width": getattr(cam, "image_width", None),
            "height": getattr(cam, "image_height", None),
        }

        # DEBUG: Log cache key for first few iterations
        if self._iteration <= 5 or self._iteration % 100 == 0:
            print(
                f"  [GS Cache] Publishing camera_id={primary_key}, alt_keys={alternative_keys}, name={camera_name}"
            )

        # Detach tensors to avoid gradient issues
        if isinstance(depth, torch.Tensor):
            payload["depth"] = depth.detach().cpu()
        elif depth is not None:
            payload["depth"] = depth

        if isinstance(normal, torch.Tensor):
            payload["normal"] = normal.detach().cpu()
        elif normal is not None:
            payload["normal"] = normal

        # Add visibility and render statistics (enhanced)
        visibility_filter = render_pkg.get("visibility_filter")
        if visibility_filter is not None:
            if isinstance(visibility_filter, torch.Tensor):
                payload["num_visible"] = visibility_filter.sum().item()
            else:
                payload["num_visible"] = (
                    sum(visibility_filter)
                    if hasattr(visibility_filter, "__iter__")
                    else None
                )

        # Add depth statistics if available
        if isinstance(depth, torch.Tensor):
            valid_depth = depth[depth > 0]
            if valid_depth.numel() > 0:
                payload["depth_stats"] = {
                    "min": valid_depth.min().item(),
                    "max": valid_depth.max().item(),
                    "mean": valid_depth.mean().item(),
                }

        self.bus.publish("gaussian.render_outputs", payload)

    def train_step(
        self, callback: Optional[Callable[[GaussianIterationState], None]] = None
    ):
        """
        Execute one optimization iteration and optionally invoke callback with metrics.

        执行一次优化迭代，并可通过回调获取指标。
        """
        if torch is None:
            raise RuntimeError("PyTorch is required for GaussianSplattingAdapter.")
        assert self.scene and self.gaussians

        self._iteration += 1
        cam = self._pick_view()
        self.gaussians.update_learning_rate(self._iteration)
        if self._iteration % 1000 == 0:
            self.gaussians.oneupSHdegree()

        bg = (
            torch.rand((3), device=self._background.device)
            if self._opt.random_background
            else self._background
        )

        render_pkg = self._render(
            cam,
            self.gaussians,
            self._pipe,
            bg,
            use_trained_exp=self._dataset.train_test_exp,
        )
        self._publish_render_outputs(cam, render_pkg)
        image = render_pkg["render"]
        gt = cam.original_image.to(image.device)

        if cam.alpha_mask is not None:
            alpha = cam.alpha_mask.to(image.device)
            image = image * alpha
            gt = gt * alpha

        l1 = self._l1(image, gt)
        ssim_val = self._ssim(image, gt)
        loss = (1.0 - self._opt.lambda_dssim) * l1 + self._opt.lambda_dssim * (
            1.0 - ssim_val
        )

        if self._depth_weight(self._iteration) > 0 and cam.depth_reliable:
            inv_depth = render_pkg["depth"]
            mono_inv = cam.invdepthmap.to(inv_depth.device)
            depth_mask = cam.depth_mask.to(inv_depth.device)
            depth_loss = torch.abs(inv_depth - mono_inv) * depth_mask
            loss += self._depth_weight(self._iteration) * depth_loss.mean()

        loss.backward()
        with torch.no_grad():
            visibility_filter = render_pkg.get("visibility_filter")
            radii = render_pkg.get("radii")
            viewspace_points = render_pkg.get("viewspace_points")
            if (
                visibility_filter is not None
                and radii is not None
                and hasattr(self.gaussians, "max_radii2D")
            ):
                self.gaussians.max_radii2D[visibility_filter] = torch.max(
                    self.gaussians.max_radii2D[visibility_filter],
                    radii[visibility_filter],
                )
            if viewspace_points is not None and visibility_filter is not None:
                self.gaussians.add_densification_stats(
                    viewspace_points, visibility_filter
                )

        self.gaussians.exposure_optimizer.step()
        self.gaussians.exposure_optimizer.zero_grad(set_to_none=True)
        self.gaussians.optimizer.step()
        self.gaussians.optimizer.zero_grad(set_to_none=True)

        state = GaussianIterationState(
            iteration=self._iteration,
            loss=float(loss.detach().cpu()),
            l1=float(l1.detach().cpu()),
            ssim=float(ssim_val.detach().cpu()),
            lr_position=self.gaussians.optimizer.param_groups[0]["lr"],
            num_gaussians=self.gaussians.get_xyz.shape[0],
        )
        if callback:
            callback(state)
        self.bus.publish("gaussian.train_step", state)
        return state

    def render(self, camera) -> torch.Tensor:
        """
        Render helper exposed as API + exchange bus event.

        渲染当前高斯模型并在总线上广播结果。
        """
        if torch is None:
            raise RuntimeError("PyTorch is required for GaussianSplattingAdapter.")
        out = self._render(
            camera,
            self.gaussians,
            self._pipe,
            self._background,
            use_trained_exp=self._dataset.train_test_exp,
        )["render"]
        self.bus.publish("gaussian.render", out)
        return out

    def export_surface(self, iteration: Optional[int] = None) -> Path:
        """
        Persist the current Gaussian cloud and return the resulting PLY path.

        保存当前高斯点云并返回生成的 PLY 路径。
        """
        iteration = iteration or self._iteration
        self.scene.save(iteration)
        ply_path = (
            Path(self.scene.model_path)
            / "point_cloud"
            / f"iteration_{iteration}"
            / "point_cloud.ply"
        )
        self.bus.publish("gaussian.export_surface", ply_path)
        return ply_path

    def import_surface(self, mesh_path: Path, sh_degree: int = 3):
        """
        Initialize Gaussians from an external mesh/point cloud.

        从外部网格或点云初始化 Gaussians。
        """
        if trimesh is None:
            raise RuntimeError("trimesh is required to import surfaces.")

        if not Path(mesh_path).exists():
            raise FileNotFoundError(f"Mesh file not found: {mesh_path}")

        try:
            mesh = trimesh.load_mesh(mesh_path)
            samples, face_idx = trimesh.sample.sample_surface(
                mesh, self.config.get("seed_points", 500000)
            )
        except Exception as e:
            raise RuntimeError(f"Failed to sample mesh {mesh_path}: {e}")

        if samples is None or len(samples) == 0:
            raise RuntimeError(f"Sampled 0 points from mesh {mesh_path}")

        # Prepare new parameters on CPU first to avoid partial GPU updates
        # 预先准备所有参数，避免部分更新导致状态不一致
        device = self._background.device

        try:
            # XYZ
            xyz_tensor = torch.from_numpy(samples).to(device, dtype=torch.float32)

            # Colors / Features DC
            # colors is (N, 3) -> features_dc should be (N, 1, 3)
            normals = mesh.face_normals[face_idx]
            colors = normals * 0.5 + 0.5
            features_dc_tensor = torch.from_numpy(colors[:, None, :]).to(
                device, dtype=torch.float32
            )

            # Features Rest
            # features_rest should be (N, 15, 3) for SH=3
            num_sh_channels = (sh_degree + 1) ** 2 - 1
            features_rest_tensor = torch.zeros(
                (xyz_tensor.shape[0], num_sh_channels, 3),
                device=device,
                dtype=torch.float32,
            )

            # Opacity
            opacity_tensor = torch.ones(
                (xyz_tensor.shape[0], 1), device=device, dtype=torch.float32
            ).mul_(0.1)

            # Scaling
            scaling_tensor = torch.zeros(
                (xyz_tensor.shape[0], 3), device=device, dtype=torch.float32
            )

            # Rotation
            rotation_tensor = torch.tensor(
                [[1, 0, 0, 0]], device=device, dtype=torch.float32
            ).repeat(xyz_tensor.shape[0], 1)

            # Atomic Update
            # 原子更新所有参数
            self.gaussians._xyz = torch.nn.Parameter(xyz_tensor.requires_grad_(True))
            self.gaussians._features_dc = torch.nn.Parameter(
                features_dc_tensor.requires_grad_(True)
            )
            self.gaussians._features_rest = torch.nn.Parameter(
                features_rest_tensor.requires_grad_(True)
            )
            self.gaussians._opacity = torch.nn.Parameter(
                opacity_tensor.requires_grad_(True)
            )
            self.gaussians._scaling = torch.nn.Parameter(
                scaling_tensor.requires_grad_(True)
            )
            self.gaussians._rotation = torch.nn.Parameter(
                rotation_tensor.requires_grad_(True)
            )

            self.gaussians.active_sh_degree = sh_degree

            # Re-initialize optimizer and buffers for the new parameters
            # 为新参数重新初始化优化器和缓冲区
            if self._opt is not None:
                self.gaussians.training_setup(self._opt)
                # training_setup doesn't reset max_radii2D, so we do it manually
                # training_setup 不会重置 max_radii2D，所以我们手动重置
                self.gaussians.max_radii2D = torch.zeros(
                    (xyz_tensor.shape[0]), device=device
                )

            print(
                f"  [GS Import] Successfully imported {xyz_tensor.shape[0]} points from {mesh_path.name}"
            )
            self.bus.publish("gaussian.import_surface", mesh_path)

        except Exception as e:
            raise RuntimeError(f"Failed to create Gaussian parameters from mesh: {e}")

    # ------------------------------------------------------------------ #
    # Gaussian Property Accessors (SuGaR-inspired)
    # 高斯属性访问器（受 SuGaR 启发）
    # ------------------------------------------------------------------ #

    def get_gaussian_xyz(
        self, detach: bool = True, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Retrieve Gaussian center positions.

        获取高斯中心位置。

        Args:
            detach (bool): If True, detach from computation graph. Default True.
            mask (Optional[torch.Tensor]): Boolean mask to filter gaussians. Default None.

        Returns:
            torch.Tensor: Gaussian positions, shape (N, 3) or (M, 3) if masked.
        """
        if torch is None or self.gaussians is None:
            raise RuntimeError("GaussianSplattingAdapter not properly initialized.")

        xyz = self.gaussians.get_xyz
        if mask is not None:
            xyz = xyz[mask]
        if detach:
            xyz = xyz.detach()
        return xyz

    def get_gaussian_features(
        self, detach: bool = True, mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieve Gaussian SH features (DC and rest components).

        获取高斯球谐特征（DC 和 rest 分量）。

        Args:
            detach (bool): If True, detach from computation graph. Default True.
            mask (Optional[torch.Tensor]): Boolean mask to filter gaussians. Default None.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - features_dc: shape (N, 1, 3) or (M, 1, 3) if masked
                - features_rest: shape (N, K, 3) or (M, K, 3) if masked
        """
        if torch is None or self.gaussians is None:
            raise RuntimeError("GaussianSplattingAdapter not properly initialized.")

        features = self.gaussians.get_features
        features_dc = features[:, :1, :]
        features_rest = features[:, 1:, :]

        if mask is not None:
            features_dc = features_dc[mask]
            features_rest = features_rest[mask]

        if detach:
            features_dc = features_dc.detach()
            features_rest = features_rest.detach()

        return features_dc, features_rest

    def get_gaussian_scaling(
        self,
        detach: bool = True,
        mask: Optional[torch.Tensor] = None,
        activated: bool = True,
    ) -> torch.Tensor:
        """
        Retrieve Gaussian scaling parameters.

        获取高斯缩放参数。

        Args:
            detach (bool): If True, detach from computation graph. Default True.
            mask (Optional[torch.Tensor]): Boolean mask to filter gaussians. Default None.
            activated (bool): If True, return exp(scaling), else raw. Default True.

        Returns:
            torch.Tensor: Gaussian scales, shape (N, 3) or (M, 3) if masked.
        """
        if torch is None or self.gaussians is None:
            raise RuntimeError("GaussianSplattingAdapter not properly initialized.")

        if activated:
            scaling = self.gaussians.get_scaling
        else:
            scaling = self.gaussians._scaling

        if mask is not None:
            scaling = scaling[mask]
        if detach:
            scaling = scaling.detach()
        return scaling

    def get_gaussian_rotation(
        self,
        detach: bool = True,
        mask: Optional[torch.Tensor] = None,
        normalized: bool = True,
    ) -> torch.Tensor:
        """
        Retrieve Gaussian rotation quaternions.

        获取高斯旋转四元数。

        Args:
            detach (bool): If True, detach from computation graph. Default True.
            mask (Optional[torch.Tensor]): Boolean mask to filter gaussians. Default None.
            normalized (bool): If True, return normalized quaternions. Default True.

        Returns:
            torch.Tensor: Rotation quaternions, shape (N, 4) or (M, 4) if masked.
        """
        if torch is None or self.gaussians is None:
            raise RuntimeError("GaussianSplattingAdapter not properly initialized.")

        if normalized:
            rotation = self.gaussians.get_rotation
        else:
            rotation = self.gaussians._rotation

        if mask is not None:
            rotation = rotation[mask]
        if detach:
            rotation = rotation.detach()
        return rotation

    def get_gaussian_opacity(
        self,
        detach: bool = True,
        mask: Optional[torch.Tensor] = None,
        activated: bool = True,
    ) -> torch.Tensor:
        """
        Retrieve Gaussian opacity values.

        获取高斯不透明度值。

        Args:
            detach (bool): If True, detach from computation graph. Default True.
            mask (Optional[torch.Tensor]): Boolean mask to filter gaussians. Default None.
            activated (bool): If True, return sigmoid(opacity), else raw. Default True.

        Returns:
            torch.Tensor: Opacity values, shape (N, 1) or (M, 1) if masked.
        """
        if torch is None or self.gaussians is None:
            raise RuntimeError("GaussianSplattingAdapter not properly initialized.")

        if activated:
            opacity = self.gaussians.get_opacity
        else:
            opacity = self.gaussians._opacity

        if mask is not None:
            opacity = opacity[mask]
        if detach:
            opacity = opacity.detach()
        return opacity

    def get_gaussian_count(self) -> int:
        """
        Return the current number of Gaussians.

        返回当前高斯数量。

        Returns:
            int: Number of Gaussians.
        """
        if self.gaussians is None:
            return 0
        return self.gaussians.get_xyz.shape[0]

    def get_gaussian_covariance(
        self,
        return_full_matrix: bool = False,
        detach: bool = True,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute 3D covariance matrices for Gaussians (inspired by SuGaR.get_covariance).

        计算高斯的 3D 协方差矩阵（受 SuGaR.get_covariance 启发）。

        Args:
            return_full_matrix (bool): If True, return full 3x3 matrix.
                                       If False, return compact 6-element upper triangle.
            detach (bool): If True, detach from computation graph.
            mask (Optional[torch.Tensor]): Boolean mask to filter gaussians.

        Returns:
            torch.Tensor:
                - If return_full_matrix: shape (N, 3, 3) or (M, 3, 3)
                - Otherwise: shape (N, 6) or (M, 6) [xx, xy, xz, yy, yz, zz]
        """
        if torch is None or self.gaussians is None:
            raise RuntimeError("GaussianSplattingAdapter not properly initialized.")

        # Get rotation and scaling
        rotation = self.get_gaussian_rotation(detach=False, mask=mask, normalized=True)
        scaling = self.get_gaussian_scaling(detach=False, mask=mask, activated=True)

        # Convert quaternion to rotation matrix
        # rotation: (N, 4) [w, x, y, z]
        # We need to build R from quaternion
        w, x, y, z = rotation[:, 0], rotation[:, 1], rotation[:, 2], rotation[:, 3]

        # Rotation matrix from quaternion (following 3DGS convention)
        R = torch.zeros(
            (rotation.shape[0], 3, 3), device=rotation.device, dtype=rotation.dtype
        )
        R[:, 0, 0] = 1 - 2 * (y * y + z * z)
        R[:, 0, 1] = 2 * (x * y - w * z)
        R[:, 0, 2] = 2 * (x * z + w * y)
        R[:, 1, 0] = 2 * (x * y + w * z)
        R[:, 1, 1] = 1 - 2 * (x * x + z * z)
        R[:, 1, 2] = 2 * (y * z - w * x)
        R[:, 2, 0] = 2 * (x * z - w * y)
        R[:, 2, 1] = 2 * (y * z + w * x)
        R[:, 2, 2] = 1 - 2 * (x * x + y * y)

        # Build scale matrix S
        S = torch.zeros(
            (scaling.shape[0], 3, 3), device=scaling.device, dtype=scaling.dtype
        )
        S[:, 0, 0] = scaling[:, 0]
        S[:, 1, 1] = scaling[:, 1]
        S[:, 2, 2] = scaling[:, 2]

        # Covariance: Σ = R S S^T R^T
        RS = torch.bmm(R, S)  # (N, 3, 3)
        cov3d = torch.bmm(RS, RS.transpose(1, 2))  # (N, 3, 3)

        if not return_full_matrix:
            # Extract upper triangle: [xx, xy, xz, yy, yz, zz]
            cov3d_compact = torch.stack(
                [
                    cov3d[:, 0, 0],  # xx
                    cov3d[:, 0, 1],  # xy
                    cov3d[:, 0, 2],  # xz
                    cov3d[:, 1, 1],  # yy
                    cov3d[:, 1, 2],  # yz
                    cov3d[:, 2, 2],  # zz
                ],
                dim=1,
            )
            cov3d = cov3d_compact

        if detach:
            cov3d = cov3d.detach()

        return cov3d

    # ------------------------------------------------------------------ #
    # Gaussian State Inspection Utilities
    # 高斯状态检查工具
    # ------------------------------------------------------------------ #

    def get_gaussian_state_dict(
        self, detach: bool = True, include_gradients: bool = False
    ) -> Dict[str, Any]:
        """
        Comprehensive snapshot of all Gaussian parameters.

        所有高斯参数的综合快照。

        Args:
            detach (bool): If True, detach tensors from computation graph.
            include_gradients (bool): If True, include gradient information.

        Returns:
            Dict[str, Any]: Dictionary with keys: xyz, features_dc, features_rest,
                            scaling, rotation, opacity, covariance_compact, count
        """
        if torch is None or self.gaussians is None:
            raise RuntimeError("GaussianSplattingAdapter not properly initialized.")

        features_dc, features_rest = self.get_gaussian_features(detach=detach)

        state = {
            "xyz": self.get_gaussian_xyz(detach=detach),
            "features_dc": features_dc,
            "features_rest": features_rest,
            "scaling": self.get_gaussian_scaling(detach=detach, activated=True),
            "scaling_raw": self.get_gaussian_scaling(detach=detach, activated=False),
            "rotation": self.get_gaussian_rotation(detach=detach, normalized=True),
            "rotation_raw": self.get_gaussian_rotation(detach=detach, normalized=False),
            "opacity": self.get_gaussian_opacity(detach=detach, activated=True),
            "opacity_raw": self.get_gaussian_opacity(detach=detach, activated=False),
            "covariance_compact": self.get_gaussian_covariance(
                return_full_matrix=False, detach=detach
            ),
            "count": self.get_gaussian_count(),
        }

        if include_gradients and not detach:
            state["gradients"] = {
                "xyz": (
                    self.gaussians.get_xyz.grad
                    if self.gaussians.get_xyz.grad is not None
                    else None
                ),
                "scaling": (
                    self.gaussians._scaling.grad
                    if self.gaussians._scaling.grad is not None
                    else None
                ),
                "rotation": (
                    self.gaussians._rotation.grad
                    if self.gaussians._rotation.grad is not None
                    else None
                ),
                "opacity": (
                    self.gaussians._opacity.grad
                    if self.gaussians._opacity.grad is not None
                    else None
                ),
            }

        return state

    def filter_gaussians_by_mask(
        self, mask: torch.Tensor, detach: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Return filtered Gaussian properties based on boolean mask.

        基于布尔掩码返回过滤后的高斯属性。

        Args:
            mask (torch.Tensor): Boolean mask, shape (N,)
            detach (bool): If True, detach tensors from computation graph.

        Returns:
            Dict[str, torch.Tensor]: Filtered Gaussian properties
        """
        if torch is None or self.gaussians is None:
            raise RuntimeError("GaussianSplattingAdapter not properly initialized.")

        features_dc, features_rest = self.get_gaussian_features(
            detach=detach, mask=mask
        )

        return {
            "xyz": self.get_gaussian_xyz(detach=detach, mask=mask),
            "features_dc": features_dc,
            "features_rest": features_rest,
            "scaling": self.get_gaussian_scaling(detach=detach, mask=mask),
            "rotation": self.get_gaussian_rotation(detach=detach, mask=mask),
            "opacity": self.get_gaussian_opacity(detach=detach, mask=mask),
            "count": mask.sum().item(),
        }

    def compute_gaussian_importance(
        self, method: str = "gradient", top_k: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Rank Gaussians by importance metric.

        按重要性指标排序高斯。

        Args:
            method (str): Importance metric, one of:
                         - 'gradient': accumulated gradient norm (requires gradient tracking)
                         - 'opacity': by opacity value
                         - 'scale': by volume (product of scales)
            top_k (Optional[int]): If set, return only top K indices and scores

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - indices: sorted indices by importance (descending)
                - scores: importance scores
        """
        if torch is None or self.gaussians is None:
            raise RuntimeError("GaussianSplattingAdapter not properly initialized.")

        if method == "gradient":
            # Use accumulated densification stats if available
            if hasattr(self.gaussians, "xyz_gradient_accum"):
                scores = self.gaussians.xyz_gradient_accum.squeeze(-1).clone()
            else:
                raise ValueError(
                    "Gradient-based importance requires densification stats tracking"
                )

        elif method == "opacity":
            scores = self.get_gaussian_opacity(detach=True, activated=True).squeeze(-1)

        elif method == "scale":
            scales = self.get_gaussian_scaling(detach=True, activated=True)
            scores = scales.prod(dim=-1)  # volume ~ product of 3 scales

        else:
            raise ValueError(f"Unknown importance method: {method}")

        # Sort descending
        indices = torch.argsort(scores, descending=True)

        if top_k is not None:
            indices = indices[:top_k]
            scores = scores[indices]
        else:
            scores = scores[indices]

        return indices, scores

    # ------------------------------------------------------------------ #
    # Depth/Normal Rendering Helper (SuGaR-inspired)
    # 深度/法线渲染辅助方法（受 SuGaR 启发）
    # ------------------------------------------------------------------ #

    def render_depth_normal(
        self, camera, output_format: str = "dict"
    ) -> Dict[str, torch.Tensor]:
        """
        Render depth and normal maps in one pass (inspired by SuGaR.render_depth_and_normal).

        一次性渲染深度和法线图（受 SuGaR.render_depth_and_normal 启发）。

        Args:
            camera: Camera object (from Scene.getTrainCameras or similar)
            output_format (str): Output format, currently only 'dict' supported

        Returns:
            Dict[str, torch.Tensor]: Dictionary with keys:
                - 'depth': depth map, shape (H, W)
                - 'normal': normal map, shape (H, W, 3), in camera space
                - 'rgb': rendered RGB image, shape (H, W, 3)
        """
        if torch is None or self.gaussians is None:
            raise RuntimeError("GaussianSplattingAdapter not properly initialized.")

        # Render full package to get depth and other outputs
        bg = (
            torch.rand((3), device=self._background.device)
            if self._opt.random_background
            else self._background
        )

        render_pkg = self._render(
            camera,
            self.gaussians,
            self._pipe,
            bg,
            use_trained_exp=self._dataset.train_test_exp,
        )

        # Extract RGB
        rgb = render_pkg["render"]  # (3, H, W)
        rgb = rgb.permute(1, 2, 0)  # (H, W, 3)

        # Extract depth (if available in render package)
        depth = render_pkg.get("depth")  # (1, H, W) or (H, W)
        if depth is not None:
            if depth.ndim == 3:
                depth = depth.squeeze(0)  # (H, W)

        # Extract normal (if available)
        normal = render_pkg.get("normal") or render_pkg.get("normals")
        if normal is not None:
            if normal.ndim == 3 and normal.shape[0] == 3:
                normal = normal.permute(1, 2, 0)  # (H, W, 3)

        result = {
            "rgb": rgb.detach() if rgb is not None else None,
            "depth": depth.detach() if depth is not None else None,
            "normal": normal.detach() if normal is not None else None,
        }

        return result
