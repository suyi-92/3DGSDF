from __future__ import annotations

"""
Fusion wrapper for Gaussian Splatting & NeuS.

English: This module exposes a data service, adapter APIs, and an exchange bus to
deeply fuse different neural rendering systems without touching their internals.

该模块提供统一数据服务、适配器 API 以及交换总线，用于在完全解耦的前提下
深度融合 Gaussian Splatting 与 NeuS 等架构，所有交互都通过明确的接口完成。
"""

import argparse
import contextlib
import shutil
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import os

try:
    import trimesh
except ImportError:  # pragma: no cover - allows unit tests without trimesh
    trimesh = None

try:
    import torch
    import torch.nn.functional as F
except ImportError:  # pragma: no cover - allows unit tests without torch
    torch = None
    F = None

try:
    from PIL import Image
except ImportError:  # pragma: no cover
    Image = None

try:
    import imageio.v2 as imageio
except ImportError:  # pragma: no cover
    imageio = None


@dataclass
class SceneSpec:
    """
    High-level description of a scene and working directories used by the wrapper.

    对场景的高级描述以及包装器所使用的工作目录。
    """

    scene_name: str
    dataset_root: str
    gaussian_source_path: str
    gaussian_model_path: str
    neus_conf_path: str
    neus_case: str
    shared_workspace: str
    resolution_scales: Tuple[float, ...] = (1.0,)
    device: str = "cuda"
    white_background: bool = False


@dataclass
class GaussianIterationState:
    """
    Snapshot describing one Gaussian Splatting optimization step.

    描述一次高斯溅射优化步骤的快照。
    """

    iteration: int
    loss: float
    l1: float
    ssim: float
    lr_position: float
    num_gaussians: int


@dataclass
class NeuSIterationState:
    """
    Snapshot describing one NeuS optimization step.

    描述一次 NeuS 优化步骤的快照。
    """

    iteration: int
    loss: float
    color_loss: float
    eikonal_loss: float
    lr: float


@dataclass
class SparseBundlePaths:
    """
    Convenience record returned by the data service for COLMAP assets.

    数据服务为 COLMAP 资源返回的便捷记录
    """

    cameras: Path
    images: Path
    points3d: Path


@dataclass
class RayBatch:
    """
    Container describing a batch of rays sampled from the dataset.

    用于描述从数据集采样的一批光线的容器。
    """

    origins: Any
    directions: Any
    colors: Any
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RegisteredAPI:
    """
    Metadata that accompanies every registered API.

    随每个已注册 API 一同提供的元数据。
    """

    func: Callable[..., Any]
    description: str = ""

    def __call__(self, *args, **kwargs):
        """
        Invoke the registered API.

        调用已注册的 API。
        """
        return self.func(*args, **kwargs)


class DataService:
    """
    Provides a single authoritative view of the dataset (mip-NeRF360 style) and
    exposes accessor APIs so every consumer works off the same assets.

    统一管理 mip-NeRF360 风格数据集，所有组件通过它获取图像、相机与稀疏点，
    从而保证输入唯一且可追踪。
    """

    def __init__(
        self,
        dataset_root: Path,
        ray_sampler: Optional[Callable[["DataService", int], RayBatch]] = None,
    ):
        """
        Initialize the service with the mip-NeRF360 directory and optional sampler.

        使用 mip-NeRF360 数据目录和可选射线采样器初始化服务。
        """
        self.dataset_root = Path(dataset_root)
        self.scene_name = self.dataset_root.name
        self.images_dir = self.dataset_root / "images"
        self.sparse_dir = self.dataset_root / "sparse" / "0"
        self.poses_path = self.dataset_root / "poses_bounds.npy"
        self._ray_sampler = ray_sampler

        self._images = sorted(self.images_dir.glob("*"))
        self._poses_bounds: Optional[np.ndarray] = None

    # ------------------------------------------------------------------ #
    # Dataset inspection helpers
    # ------------------------------------------------------------------ #

    def list_images(self) -> List[Path]:
        """
        Return absolute paths to every image in the scene.

        列出场景内所有图像的绝对路径。
        """
        return list(self._images)

    def get_image_path(self, index_or_name: Any) -> Path:
        """
        Resolve a single image either by integer index or by filename.

        通过索引或文件名定位单张图像路径。
        """
        if isinstance(index_or_name, int):
            return self._images[index_or_name]
        if isinstance(index_or_name, str):
            target = self.images_dir / index_or_name
            if target.exists():
                return target
            raise FileNotFoundError(target)
        raise TypeError("index_or_name must be int or str")

    def get_sparse_bundle(self) -> SparseBundlePaths:
        """
        Return paths to COLMAP binary bundle files.

        返回 COLMAP bundle 文件路径。
        """
        return SparseBundlePaths(
            cameras=self.sparse_dir / "cameras.bin",
            images=self.sparse_dir / "images.bin",
            points3d=self.sparse_dir / "points3D.bin",
        )

    def load_poses_bounds(self) -> np.ndarray:
        """
        Load and cache poses_bounds.npy (mip-NeRF360 camera specification).

        加载并缓存 poses_bounds.npy，提供 mip-NeRF360 相机姿态。
        """
        if self._poses_bounds is None:
            if not self.poses_path.exists():
                raise FileNotFoundError(self.poses_path)
            self._poses_bounds = np.load(self.poses_path)
        return self._poses_bounds.copy()

    # ------------------------------------------------------------------ #
    # Sampling / materialization
    # ------------------------------------------------------------------ #

    def register_ray_sampler(
        self, sampler: Callable[["DataService", int], RayBatch]
    ) -> None:
        """
        Register a callable that knows how to sample rays. The callable receives this
        DataService and the batch size.

        注册射线采样器，供 NeuS/GS 通过同一函数抽样射线。
        """
        self._ray_sampler = sampler

    def sample_rays(self, batch_size: int, **kwargs) -> RayBatch:
        """
        Sample rays (+ colors) using the registered sampler. Optional kwargs are
        forwarded to the sampler to allow custom policies.

        调用注册的采样器生成射线与颜色，可通过 kwargs 指定策略。
        """
        if self._ray_sampler is None:
            raise RuntimeError(
                "Ray sampler not registered. Call register_ray_sampler first."
            )
        return self._ray_sampler(self, batch_size, **kwargs)

    def materialize_gaussian_scene(self, target_root: Path) -> Dict[str, Path]:
        """
        Prepare a Gaussian-Splatting friendly directory by mirroring images and sparse
        assets into `target_root`. Returns useful paths for downstream consumers.

        为 3DGS 生成可直接使用的目录结构（images/sparse/poses），减少数据复制。
        """
        target_root = Path(target_root)
        target_root.mkdir(parents=True, exist_ok=True)

        dest_images = target_root / "images"
        dest_sparse = target_root / "sparse" / "0"
        if not dest_images.exists():
            if self.images_dir.resolve() == dest_images.resolve():
                pass
            else:
                shutil.copytree(self.images_dir, dest_images)
        if not dest_sparse.exists():
            if self.sparse_dir.resolve() == dest_sparse.resolve():
                pass
            else:
                shutil.copytree(self.sparse_dir, dest_sparse)
        poses_target = target_root / "poses_bounds.npy"
        if (
            self.poses_path.resolve() != poses_target.resolve()
            or not poses_target.exists()
        ):
            shutil.copy2(self.poses_path, poses_target)

        return {
            "source": target_root,
            "images": dest_images,
            "sparse": dest_sparse,
            "poses": target_root / "poses_bounds.npy",
        }

    def _ensure_mask(self, src_image: Path, dst_mask: Path):
        """
        Create a white mask matching the source image resolution (if not present).

        如果目标 mask 不存在，则创建与源图像同尺寸的纯白 mask。
        """
        if dst_mask.exists():
            return
        dst_mask.parent.mkdir(parents=True, exist_ok=True)
        if Image is not None:
            with Image.open(src_image) as img:
                mask = Image.new("RGB", img.size, (255, 255, 255))
                mask.save(dst_mask)
                return
        if imageio is not None:
            arr = imageio.imread(str(src_image))
            mask = np.ones_like(arr, dtype=np.uint8) * 255
            imageio.imwrite(str(dst_mask), mask)
            return
        # Fallback: copy source file (may not be ideal but keeps pipeline running)
        shutil.copy2(src_image, dst_mask)

    def _copy_image_to_png(self, src_image: Path, dst_image: Path):
        """
        Copy/convert an image to PNG format expected by NeuS.

        将源图像转换为 NeuS 期望的 PNG 文件。
        """
        dst_image.parent.mkdir(parents=True, exist_ok=True)
        if Image is not None:
            with Image.open(src_image) as img:
                img.convert("RGB").save(dst_image, format="PNG")
                return
        if imageio is not None:
            arr = imageio.imread(str(src_image))
            imageio.imwrite(str(dst_image), arr)
            return
        shutil.copy2(src_image, dst_image)

    def _generate_neus_camera_dict(self) -> Dict[str, np.ndarray]:
        """
        Convert poses_bounds.npy into the camera dictionary expected by NeuS.

        将 poses_bounds.npy 转换为 NeuS 所需的相机字典。
        """
        poses_bounds = self.load_poses_bounds()
        poses = poses_bounds[:, :15].reshape(-1, 3, 5)
        poses_rot = poses[..., :4]
        hwf = poses[..., 4]
        H = hwf[:, 0:1]
        W = hwf[:, 1:2]
        F = hwf[:, 2:3]
        bottom = np.tile(
            np.array([[0, 0, 0, 1]], dtype=np.float32), (poses_rot.shape[0], 1, 1)
        )
        c2w = np.concatenate([poses_rot, bottom], axis=1)
        centers = c2w[:, :3, 3]
        center = centers.mean(axis=0)
        radius = np.max(np.linalg.norm(centers - center, axis=-1))
        scale = 1.0 / radius if radius > 0 else 1.0
        scale_mat = np.eye(4, dtype=np.float32)
        scale_mat[:3, :3] *= scale
        scale_mat[:3, 3] = -center * scale
        scale_mat_inv = np.linalg.inv(scale_mat)
        camera_dict: Dict[str, np.ndarray] = {}
        for idx, pose in enumerate(c2w):
            w2c = np.linalg.inv(pose)
            focal = F[idx, 0]
            width = W[idx, 0]
            height = H[idx, 0]
            K = np.array(
                [
                    [focal, 0, width / 2.0],
                    [0, focal, height / 2.0],
                    [0, 0, 1.0],
                ],
                dtype=np.float32,
            )
            P = K @ w2c[:3, :]
            world_mat = np.eye(4, dtype=np.float32)
            world_mat[:3, :] = P
            camera_dict[f"world_mat_{idx}"] = world_mat
            camera_dict[f"world_mat_inv_{idx}"] = np.linalg.inv(world_mat)
            camera_dict[f"scale_mat_{idx}"] = scale_mat
            camera_dict[f"scale_mat_inv_{idx}"] = scale_mat_inv
        return camera_dict

    def materialize_neus_scene(
        self, target_root: Path, scene_name: Optional[str] = None
    ) -> Dict[str, Path]:
        """
        Prepare a NeuS dataset directory (images/masks + cameras_sphere.npz).

        为 NeuS 构建数据目录（images/masks + cameras_sphere.npz）。
        """
        scene_label = scene_name or self.scene_name
        target_root = Path(target_root)
        scene_dir = target_root  # / scene_label
        image_dir = scene_dir / "image"
        mask_dir = scene_dir / "mask"
        scene_dir.mkdir(parents=True, exist_ok=True)
        image_dir.mkdir(parents=True, exist_ok=True)
        mask_dir.mkdir(parents=True, exist_ok=True)

        for img_path in self.list_images():
            dst_name = f"{img_path.stem}.png"
            dst = image_dir / dst_name
            if not dst.exists():
                self._copy_image_to_png(img_path, dst)
            mask_dst = mask_dir / dst_name
            self._ensure_mask(dst, mask_dst)

        camera_npz = scene_dir / "cameras_sphere.npz"
        if not camera_npz.exists():
            camera_dict = self._generate_neus_camera_dict()
            np.savez(camera_npz, **camera_dict)

        return {
            "scene_dir": scene_dir,
            "image_dir": image_dir,
            "mask_dir": mask_dir,
            "cameras": camera_npz,
        }


class MutableHandle(contextlib.AbstractContextManager):
    """
    Small helper so callers can temporarily mutate adapter state in a controlled way.

    一个小型辅助工具，使调用者能够以受控的方式暂时改变适配器状态。
    """

    def __init__(self, target: Any, on_commit: Optional[Callable[[Any], None]] = None):
        """
        Store target reference and optional callback invoked on successful exit.

        保存目标引用以及在上下文正常结束时调用的回调。
        """
        self._target = target
        self._on_commit = on_commit or (lambda _: None)

    def __enter__(self):
        """
        Return the mutable target object.

        返回可修改的目标对象。
        """
        return self._target

    def __exit__(self, exc_type, *_):
        """
        Commit modifications if no exception occurred.

        当没有异常时调用提交回调以应用更改。
        """
        if exc_type is None:
            self._on_commit(self._target)
        return False


class APIRegistry:
    """
    Global registry for pluggable APIs exposed by adapters and fusion logic.

    所有可扩展 API 的注册表，记录每个接口的函数与说明，方便其他工具发现并调用。
    """

    def __init__(self):
        """
        Initialize empty registry.

        初始化空的注册表。
        """
        self._apis: Dict[str, RegisteredAPI] = {}

    def register(self, name: str, func: Callable[..., Any], description: str = ""):
        """
        Register a new API endpoint and optional description.

        注册新的 API 端点及其描述。
        """
        if name in self._apis:
            raise ValueError(f"API '{name}' already registered.")
        self._apis[name] = RegisteredAPI(func=func, description=description)

    def call(self, name: str, *args, **kwargs):
        """
        Invoke a registered endpoint by name.

        通过名称调用已注册的端点。
        """
        if name not in self._apis:
            raise KeyError(f"API '{name}' not found.")
        return self._apis[name](*args, **kwargs)

    def describe(self) -> Dict[str, str]:
        """
        Return human-readable descriptions for all registered APIs.

        返回所有已注册 API 的说明文本。
        """
        return {name: api.description for name, api in self._apis.items()}


class ExchangeBus:
    """
    Lightweight publish/subscribe bus adapters can use to exchange intermediate data
    (meshes, render caches, statistics, ...).

    用于发布/订阅中间结果的轻量级总线（如 mesh、渲染缓存、统计信息等）。
    """

    def __init__(self):
        """
        Initialize subscriber map.

        初始化订阅者映射。
        """
        self._subscribers: Dict[str, List[Callable[[Any], None]]] = defaultdict(list)

    def publish(self, topic: str, payload: Any):
        """
        Notify all subscribers listening on a topic.

        向订阅某个主题的回调广播消息。
        """
        for callback in list(self._subscribers.get(topic, [])):
            callback(payload)

    def subscribe(self, topic: str, callback: Callable[[Any], None]):
        """
        Register a callback to receive updates for a topic.

        订阅主题以接收后续发布的消息。
        """
        self._subscribers[topic].append(callback)


# ------------------------------------------------------------------ #
# Utility Functions for Gaussian Field Extraction
# 高斯字段提取工具函数
# ------------------------------------------------------------------ #


def extract_gaussian_fields(
    gaussians: Any,
    fields: List[str],
    indices: Optional[torch.Tensor] = None,
    detach: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    Generic utility to extract multiple Gaussian fields at once.

    通用工具函数，一次性提取多个高斯字段。

    Args:
        gaussians: GaussianModel instance (from 3DGS)
        fields (List[str]): Field names to extract. Supported:
                           'xyz', 'features', 'features_dc', 'features_rest',
                           'scaling', 'rotation', 'opacity'
        indices (Optional[torch.Tensor]): Integer indices to extract subset.
                                          If None, extract all.
        detach (bool): If True, detach tensors from computation graph.

    Returns:
        Dict[str, torch.Tensor]: Mapping from field names to tensors.

    Example:
        >>> fields = extract_gaussian_fields(
        ...     gaussians,
        ...     ['xyz', 'opacity'],
        ...     indices=torch.tensor([0, 10, 20])
        ... )
        >>> print(fields['xyz'].shape)  # (3, 3)
    """
    if torch is None:
        raise RuntimeError("PyTorch is required for extract_gaussian_fields.")

    result = {}

    for field in fields:
        if field == "xyz":
            tensor = gaussians.get_xyz
        elif field == "features":
            tensor = gaussians.get_features
        elif field == "features_dc":
            tensor = gaussians.get_features[:, :1, :]
        elif field == "features_rest":
            tensor = gaussians.get_features[:, 1:, :]
        elif field == "scaling":
            tensor = gaussians.get_scaling
        elif field == "rotation":
            tensor = gaussians.get_rotation
        elif field == "opacity":
            tensor = gaussians.get_opacity
        else:
            raise ValueError(f"Unknown field: {field}")

        if indices is not None:
            tensor = tensor[indices]

        if detach:
            tensor = tensor.detach()

        result[field] = tensor

    return result


class AdapterBase:
    """
    Shared functionality for all adapters (API registration + bus access).

    适配器基类，封装 API 注册与交换总线，降低具体实现的样板代码。
    """

    def __init__(
        self,
        name: str,
        registry: APIRegistry,
        bus: ExchangeBus,
        data_service: DataService,
    ):
        """
        Keep references to registry, bus, and data service for derived adapters.

        保存注册表、消息总线与数据服务，供子类适配器使用。
        """
        self.name = name
        self.registry = registry
        self.bus = bus
        self.data_service = data_service

    def register_api(self, endpoint: str, func: Callable[..., Any], description: str):
        """
        Register adapter-specific API with automatic namespacing.

        使用命名空间注册适配器专属 API。
        """
        namespaced = f"{self.name}.{endpoint}"
        self.registry.register(namespaced, func, description)
        return namespaced


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
            setattr(args, k, v)

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
            "Run one gaussian-splatting optimization step and return metrics.",
        )
        self.register_api(
            "render",
            self.render,
            "Render a camera specification with the current Gaussian model.",
        )
        self.register_api(
            "export_surface",
            self.export_surface,
            "Export the current Gaussian point cloud as a .ply file.",
        )
        self.register_api(
            "import_surface",
            self.import_surface,
            "Import a mesh/point-cloud to initialize Gaussians.",
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
        cameras = self.scene.getTrainCameras().copy()
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

        # Extract camera identifiers - try multiple approaches for robustness
        camera_id = getattr(cam, "uid", None) or getattr(cam, "colmap_id", None)
        camera_name = getattr(cam, "image_name", None)

        # Build comprehensive payload with camera metadata
        payload: Dict[str, Any] = {
            "camera_id": camera_id if camera_id is not None else camera_name,
            "image_name": camera_name,
            "iteration": self._iteration,
            "width": getattr(cam, "image_width", None),
            "height": getattr(cam, "image_height", None),
        }

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
        mesh = trimesh.load_mesh(mesh_path)
        samples, face_idx = trimesh.sample.sample_surface(
            mesh, self.config.get("seed_points", 500000)
        )
        normals = mesh.face_normals[face_idx]
        colors = normals * 0.5 + 0.5
        xyz = torch.from_numpy(samples).to(self._background.device, dtype=torch.float32)
        self.gaussians._xyz = torch.nn.Parameter(xyz.requires_grad_(True))
        self.gaussians._features_dc = torch.nn.Parameter(
            torch.from_numpy(colors[:, None, :])
            .permute(0, 2, 1)
            .contiguous()
            .to(xyz.device)
            .requires_grad_(True)
        )
        self.gaussians._features_rest = torch.nn.Parameter(
            torch.zeros(
                (xyz.shape[0], 3, (sh_degree + 1) ** 2 - 1), device=xyz.device
            ).requires_grad_(True)
        )
        self.gaussians._opacity = torch.nn.Parameter(
            torch.ones((xyz.shape[0], 1), device=xyz.device)
            .mul_(0.1)
            .requires_grad_(True)
        )
        self.gaussians._scaling = torch.nn.Parameter(
            torch.zeros((xyz.shape[0], 3), device=xyz.device).requires_grad_(True)
        )
        self.gaussians._rotation = torch.nn.Parameter(
            torch.tensor(
                [[1, 0, 0, 0]] * xyz.shape[0], device=xyz.device
            ).requires_grad_(True)
        )
        self.gaussians.active_sh_degree = sh_degree
        self.bus.publish("gaussian.import_surface", mesh_path)

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
            "Run one NeuS optimization step.",
        )
        self.register_api(
            "export_surface",
            self.export_mesh,
            "Extract a NeuS mesh via marching cubes.",
        )
        self.register_api(
            "evaluate_sdf",
            self.evaluate_sdf,
            "Evaluate NeuS SDF at given points without gradient.",
        )
        self.register_api(
            "inject_supervision",
            self.inject_supervision,
            "Provide additional supervision (e.g. Gaussian renders).",
        )

    def mutable(self, attr: str) -> MutableHandle:
        """
        Return a handle to mutate an internal NeuS attribute.

        返回可用于修改 NeuS 内部属性的句柄。
        """
        if not hasattr(self.runner, attr):
            raise AttributeError(attr)
        return MutableHandle(getattr(self.runner, attr))

    def _camera_key_from_idx(self, idx: int) -> Any:
        ds = getattr(self.runner, "dataset", None)
        if ds and hasattr(ds, "images_lis"):
            try:
                return Path(ds.images_lis[idx]).stem
            except Exception:
                pass
        return idx

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
        try:
            device = ds.intrinsics_all_inv.device
            pixels_x = torch.randint(low=0, high=ds.W, size=[batch_size], device=device)
            pixels_y = torch.randint(low=0, high=ds.H, size=[batch_size], device=device)

            img = ds.images[idx]
            msk = ds.masks[idx]
            px_img = pixels_x.to(img.device)
            py_img = pixels_y.to(img.device)
            color = img[(py_img, px_img)].to(device)
            mask = msk[(py_img, px_img)].to(device)

            p = torch.stack(
                [pixels_x, pixels_y, torch.ones_like(pixels_y, device=device)], dim=-1
            ).float()
            p = torch.matmul(
                ds.intrinsics_all_inv[idx, None, :3, :3].to(device), p[:, :, None]
            ).squeeze(-1)
            rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)
            rays_v = torch.matmul(
                ds.pose_all[idx, None, :3, :3].to(device), rays_v[:, :, None]
            ).squeeze(-1)
            rays_o = ds.pose_all[idx, None, :3, 3].to(device).expand_as(rays_v)
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

        # Try to find cache entry by camera ID or name
        candidates = [idx, self._camera_key_from_idx(idx)]
        cache_entry = None
        for key in candidates:
            if key in self.depth_cache:
                cache_entry = self.depth_cache[key]
                break

        if cache_entry is None:
            return near, far

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
                    return near, far

        depth = cache_entry.get("depth")
        if depth is None or pixels_x is None or pixels_y is None:
            return near, far

        depth_tensor = (
            depth if isinstance(depth, torch.Tensor) else torch.as_tensor(depth)
        )

        try:
            # Sample depth at pixel locations
            sampled_depth = depth_tensor[(pixels_y.cpu(), pixels_x.cpu())]
        except Exception:
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
                return near, far

        except Exception:
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
        return near_new, far_new

    def _sample_gs_depth_normal(
        self,
        idx: int,
        pixels_x: Optional[torch.Tensor],
        pixels_y: Optional[torch.Tensor],
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if pixels_x is None or pixels_y is None:
            return None, None
        candidates = [idx, self._camera_key_from_idx(idx)]
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
        self, callback: Optional[Callable[[NeuSIterationState], None]] = None
    ):
        """
        Execute one NeuS optimization step and optionally report metrics.
        Follows NeuS/exp_runner.py train() loop pattern (lines 109-216).

        执行一次 NeuS 优化,并可通过回调汇报指标。
        """
        if torch is None:
            raise RuntimeError("PyTorch is required for NeuSAdapter.")
        r = self.runner

        # Update learning rate (exp_runner.py line 111)
        r.update_learning_rate()

        # Get image permutation and current image index
        image_perm = r.get_image_perm()
        idx = image_perm[r.iter_step % len(image_perm)]

        # Sample rays with pixel coordinates
        data, pixels_x, pixels_y = self._sample_rays_with_pixels(idx, r.batch_size)
        rays_o, rays_d, true_rgb, mask = (
            data[:, :3],
            data[:, 3:6],
            data[:, 6:9],
            data[:, 9:10],
        )

        # Compute near/far from sphere (exp_runner.py line 126)
        near, far = r.dataset.near_far_from_sphere(rays_o, rays_d)

        # Apply depth-guided sampling if available
        near, far = self._override_near_far_with_depth(
            idx, rays_o, rays_d, pixels_x, pixels_y, near, far
        )

        # Render (exp_runner.py lines 128-145)
        background = torch.ones([1, 3], device=r.device) if r.use_white_bkgd else None
        render_out = r.renderer.render(
            rays_o,
            rays_d,
            near,
            far,
            cos_anneal_ratio=r.get_cos_anneal_ratio(),
            background_rgb=background,
        )

        color_fine = render_out["color_fine"]
        gradients = render_out["gradients"]
        weight_sum = render_out["weight_sum"]

        # Compute depth as weighted mid_z (for geometric supervision)
        mu_z = (render_out["weights"] * render_out["mid_z_vals"]).sum(
            dim=-1, keepdim=True
        )

        # Compute normal as weighted gradients
        weighted_grad = (
            render_out["gradients"] * render_out["weights"][..., None]
        ).sum(dim=1)
        weighted_normal = torch.nn.functional.normalize(
            weighted_grad, dim=-1, eps=self.geom_loss_cfg.get("eps", 1e-6)
        )

        # Sample GS depth/normal for geometric supervision
        depth_gt, normal_gt = self._sample_gs_depth_normal(idx, pixels_x, pixels_y)

        # Color loss (exp_runner.py lines 154-159)
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

        # Eikonal loss (exp_runner.py line 167)
        eikonal_loss = ((torch.norm(gradients, dim=-1) - 1.0) ** 2).mean()

        # Mask loss (exp_runner.py line 169)
        mask_loss = torch.nn.functional.binary_cross_entropy(
            weight_sum.clip(1e-3, 1.0 - 1e-3), mask
        )

        # Geometric supervision losses (depth + normal consistency)
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

        # Total loss (exp_runner.py lines 171-175)
        loss = (
            color_loss
            + r.igr_weight * eikonal_loss
            + r.mask_weight * mask_loss
            + geom_loss
        )

        # Backward and optimizer step (exp_runner.py lines 177-179)
        r.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        r.optimizer.step()

        r.iter_step += 1

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

        无梯度评估 NeuS 的 SDF 网络，供外部模块查询。"""
        if torch is None:
            raise RuntimeError("PyTorch is required for NeuSAdapter.")
        if self.runner is None:
            raise RuntimeError("NeuS runner not bootstrapped.")
        with torch.no_grad():
            pts = points.to(self.runner.device)
            return self.runner.sdf_network.sdf(pts)


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
            "max_age": int(depth_cfg.get("max_age", 50)),
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

        缓存 3DGS 渲染的深度/法线，供 NeuS 采样引导使用。"""
        if not isinstance(payload, dict):
            return
        key = payload.get("camera_id") or payload.get("image_name")
        if key is None:
            return
        entry = {
            "depth": payload.get("depth"),
            "normal": payload.get("normal"),
            "iteration": payload.get("iteration", 0),
        }
        self.depth_cache[key] = entry

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

        利用 NeuS 的 SDF 结果对高斯进行生长/修剪引导。"""
        if torch is None:
            return
        if (
            getattr(self.neus, "runner", None) is None
            or getattr(self.gaussian, "gaussians", None) is None
            or getattr(self.gaussian, "scene", None) is None
        ):
            return

        gaussians = self.gaussian.gaussians
        opt = getattr(self.gaussian, "_opt", None)
        iteration = getattr(self.gaussian, "_iteration", 0)

        # Only densify during the active densification window
        if opt and iteration >= opt.densify_until_iter:
            return

        with torch.no_grad():
            xyz = gaussians.get_xyz
            if xyz is None or xyz.numel() == 0:
                return

            # Evaluate SDF at gaussian centers
            sdf = self.neus.evaluate_sdf(xyz)
            if not isinstance(sdf, torch.Tensor):
                return

            # Compute μ(s) = exp(-s²/(2σ²)) to weight points near the surface
            sigma = self.sdf_guidance_cfg.get("sigma", 0.5)
            mu = torch.exp(-((sdf**2) / (2 * max(sigma, 1e-6) ** 2)))

            # Get accumulated gradients (from train.py line 167: add_densification_stats)
            grad_accum = getattr(gaussians, "xyz_gradient_accum", None)
            denom = getattr(gaussians, "denom", None)

            if grad_accum is not None and denom is not None and denom.sum() > 0:
                # Normalize accumulated gradients
                grads = grad_accum / (denom + 1e-9)
            else:
                # Fallback: use proximity to surface as gradient proxy
                grads = mu.unsqueeze(-1).expand(-1, 3)

            # Ensure gradient shape matches xyz
            if grads.ndim == 2 and grads.shape[-1] != 3:
                grads = torch.norm(grads, dim=-1, keepdim=True).expand(-1, 3)

            # Compute enhanced gradient: ε_g = ∇g + ω_g * μ(s)
            omega_g = self.sdf_guidance_cfg.get("omega_g", 1.0)
            eps_g = grads + omega_g * mu.unsqueeze(-1).expand_as(grads)

            # Densify: follow train.py lines 169-171
            tau_g = self.sdf_guidance_cfg.get(
                "tau_g", 0.0002
            )  # Aligned with densify_grad_threshold
            size_threshold = (
                20 if opt and iteration > opt.opacity_reset_interval else None
            )

            # Use the official densify_and_prune method from GaussianModel
            # This handles both densification (clone/split) and pruning in one call
            if hasattr(gaussians, "densify_and_prune"):
                # Calculate gradient magnitudes for densification criterion
                grad_magnitude = torch.norm(eps_g, dim=-1, keepdim=True)

                # Min opacity threshold for pruning
                min_opacity = self.sdf_guidance_cfg.get("tau_p", 0.005)

                # Track counts before operation
                num_before = gaussians.get_xyz.shape[0]

                # Call official densify_and_prune
                gaussians.densify_and_prune(
                    max_grad=tau_g,
                    min_opacity=min_opacity,
                    extent=self.gaussian.scene.cameras_extent,
                    max_screen_size=size_threshold,
                )

                # Update statistics
                num_after = gaussians.get_xyz.shape[0]
                if num_after > num_before:
                    self._stats["densify_count"] += num_after - num_before
                elif num_after < num_before:
                    self._stats["prune_count"] += num_before - num_after

            else:
                # Fallback: separate densify and prune
                if torch.any(torch.norm(eps_g, dim=-1) > tau_g) and hasattr(
                    gaussians, "densify_and_clone"
                ):
                    num_before = gaussians.get_xyz.shape[0]
                    gaussians.densify_and_clone(
                        eps_g, tau_g, self.gaussian.scene.cameras_extent
                    )
                    self._stats["densify_count"] += (
                        gaussians.get_xyz.shape[0] - num_before
                    )

                # Prune condition: ε_p = σ_a - ω_p * (1 - μ(s))
                opacity = gaussians.get_opacity
                if opacity.ndim == 1:
                    opacity = opacity[:, None]
                omega_p = self.sdf_guidance_cfg.get("omega_p", 0.5)
                eps_p = opacity - omega_p * (1 - mu.unsqueeze(-1))
                prune_mask = eps_p.squeeze() < self.sdf_guidance_cfg.get("tau_p", 0.005)

                if torch.any(prune_mask) and hasattr(gaussians, "prune_points"):
                    self._stats["prune_count"] += prune_mask.sum().item()
                    gaussians.prune_points(prune_mask)

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

        # Step 1: Train NeuS with depth-guided sampling + geometric supervision
        neus_state = self.neus.train_step()

        # Step 2: Periodically export NeuS mesh and import to Gaussians
        if mesh_every > 0 and neus_state.iteration % mesh_every == 0:
            try:
                self.neus_to_gaussian()
            except Exception as e:
                print(f"Warning: mesh sync failed at iter {self._joint_iter}: {e}")

        # Step 3: Train Gaussian Splatting (publishes depth/normal via bus)
        gaussian_state = self.gaussian.train_step()

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
