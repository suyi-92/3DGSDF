from __future__ import annotations

"""
Data service for fusion wrapper.

数据服务模块，提供统一的数据集访问接口。
"""

import shutil
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

try:
    from PIL import Image
except ImportError:  # pragma: no cover
    Image = None

try:
    import imageio.v2 as imageio
except ImportError:  # pragma: no cover
    imageio = None

from .common import RayBatch, SparseBundlePaths


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
