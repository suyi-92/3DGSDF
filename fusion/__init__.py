"""
Fusion wrapper for Gaussian Splatting & NeuS.

该模块提供统一数据服务、适配器 API 以及交换总线，用于在完全解耦的前提下
深度融合 Gaussian Splatting 与 NeuS 等架构，所有交互都通过明确的接口完成。

This module exposes a data service, adapter APIs, and an exchange bus to
deeply fuse different neural rendering systems without touching their internals.
"""

# 导出所有公共组件
from .common import (
    SceneSpec,
    GaussianIterationState,
    NeuSIterationState,
    SparseBundlePaths,
    RayBatch,
    RegisteredAPI,
    MutableHandle,
    APIRegistry,
    ExchangeBus,
    AdapterBase,
    extract_gaussian_fields,
)

from .data_service import DataService
from .gaussian_adapter import GaussianSplattingAdapter
from .neus_adapter import NeuSAdapter
from .fusion_wrapper import FusionWrapper

__all__ = [
    # 数据类 / Data classes
    "SceneSpec",
    "GaussianIterationState",
    "NeuSIterationState",
    "SparseBundlePaths",
    "RayBatch",
    "RegisteredAPI",
    # 核心组件 / Core components
    "DataService",
    "GaussianSplattingAdapter",
    "NeuSAdapter",
    "FusionWrapper",
    # 工具类 / Utility classes
    "MutableHandle",
    "APIRegistry",
    "ExchangeBus",
    "AdapterBase",
    # 工具函数 / Utility functions
    "extract_gaussian_fields",
]
