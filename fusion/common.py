from __future__ import annotations

"""
Common components for fusion wrapper.

共享数据类、工具函数和基类，用于融合包装器的各个模块。
"""

import contextlib
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


# ------------------------------------------------------------------ #
# Data Classes
# 数据类
# ------------------------------------------------------------------ #


@dataclass
class SceneSpec:
    """
    High-level description of a scene and working directories used by the wrapper.

    对场景的高级描述以及包装器所使用的工作目录。
    """

    scene_name: str  # 场景名称
    dataset_root: str  # 原始数据集路径
    gaussian_source_path: str  # GS 输入工作空间
    gaussian_model_path: str  # GS 模型输出路径
    neus_conf_path: str  # NeuS 配置文件
    neus_case: str  # NeuS case 名称
    shared_workspace: str  # 融合共享工作空间
    resolution_scales: Tuple[float, ...] = (1.0,)
    device: str = "cuda"
    white_background: bool = False


@dataclass
class GaussianIterationState:
    """
    Snapshot describing one Gaussian Splatting optimization step.

    描述一次高斯溅射优化步骤的快照。
    """

    iteration: int  # 当前迭代次数
    loss: float  # 总损失
    l1: float  # 图像质量指标 (L1 Loss)
    ssim: float  # 图像质量指标 (SSIM)
    lr_position: float  # 位置学习率
    num_gaussians: int  # 当前高斯数量


@dataclass
class NeuSIterationState:
    """
    Snapshot describing one NeuS optimization step.

    描述一次 NeuS 优化步骤的快照。
    """

    iteration: int  # 当前迭代次数
    loss: float  # 总损失
    color_loss: float  # 颜色损失
    eikonal_loss: float  # Eikonal 正则化损失
    lr: float  # 学习率


@dataclass
class SparseBundlePaths:
    """
    Convenience record returned by the data service for COLMAP assets.

    数据服务为 COLMAP 资源返回的便捷记录
    """

    cameras: Path  # 相机参数文件路径
    images: Path  # 图像参数文件路径
    points3d: Path  # 3D 点云文件路径


@dataclass
class RayBatch:
    """
    Container describing a batch of rays sampled from the dataset.

    用于描述从数据集采样的一批光线的容器。
    """

    origins: Any  # 光线起点
    directions: Any  # 光线方向
    colors: Any  # 光线对应的颜色
    meta: Dict[str, Any] = field(default_factory=dict)  # 额外元数据


@dataclass
class RegisteredAPI:
    """
    Metadata that accompanies every registered API.

    随每个已注册 API 一同提供的元数据。
    """

    func: Callable[..., Any]  # 已注册 API 的可调用对象
    description: str = ""  # API 的简要说明

    def __call__(self, *args, **kwargs):
        """
        Invoke the registered API.

        调用已注册的 API。
        """
        return self.func(*args, **kwargs)


# ------------------------------------------------------------------ #
# Utility Classes
# 工具类
# ------------------------------------------------------------------ #


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
        data_service: Any,  # Avoid circular import, use Any for DataService type
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


# ------------------------------------------------------------------ #
# Utility Functions for Gaussian Field Extraction
# 高斯字段提取工具函数
# ------------------------------------------------------------------ #


def extract_gaussian_fields(
    gaussians: Any,
    fields: List[str],
    indices: Optional["torch.Tensor"] = None,
    detach: bool = True,
) -> Dict[str, "torch.Tensor"]:
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
