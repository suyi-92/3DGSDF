from __future__ import annotations

"""
Ray sampling utilities for GSDF fusion.

This module implements a sampler abstraction so that GS-guided depth sampling
logic can evolve independently from renderer code. The base ``RaySampler``
exposes a common ``get_z_vals`` interface, with concrete implementations for
uniform stratified sampling and the Gaussian Splatting guided strategy
proposed in GSDF Section 3.2.1.
"""

from typing import Any, Dict, Optional

try:  # pragma: no cover - optional dependency guard
    import torch
    import torch.nn as nn
except ImportError:  # pragma: no cover
    torch = None
    nn = object  # type: ignore


class RaySampler(nn.Module):
    """
    Abstract sampler interface returning per-ray depth samples.

    抽象采样器接口，负责为每条光线返回一组深度采样点。
    """

    def get_z_vals(self, rays_o, rays_d, near, far, **kwargs):
        """
        Compute depth samples along rays between near/far.

        计算介于近端和远端之间的光线深度采样点，需要子类实现具体逻辑。
        """
        raise NotImplementedError


class UniformSampler(RaySampler):
    """
    Standard stratified sampler between global near/far planes.

    在全局的 near/far 范围内执行分层采样的标准实现。
    """

    def __init__(self, training: bool = True):
        super().__init__()
        self.training = training

    def get_z_vals(self, rays_o, rays_d, near, far, n_samples: int, **_: Any):
        """
        Stratified sample depths for a batch of rays.

        对一批光线在 near/far 范围内进行分层深度采样，返回形状为
        [num_rays, n_samples] 的深度值张量。
        """
        if torch is None:  # pragma: no cover - runtime guard
            raise RuntimeError("PyTorch is required for UniformSampler.")

        device = rays_o.device
        dtype = rays_o.dtype
        num_rays = rays_o.shape[0]

        near_t = _reshape_bounds(near, num_rays, device, dtype)
        far_t = _reshape_bounds(far, num_rays, device, dtype)

        return _stratified_between(
            near_t, far_t, n_samples, dtype, device, self.training
        )


class GSGuidedSampler(RaySampler):
    """
    Implements the GSDF Section 3.2.1 depth-guided sampling strategy.

    Steps (per batch of rays):
    1) Render Gaussian Splatting depth/opacity for the rays.
    2) Query NeuS SDF once at the GS surface point.
    3) Sample within [D_gs - k * |sdf|, D_gs + k * |sdf|] with jitter.
    4) Fallback to uniform sampling when GS opacity is too low.

    实现 GSDF 第 3.2.1 节的深度引导采样策略：
    1）对光线执行 Gaussian Splatting 渲染以获取深度和不透明度；
    2）在 GS 表面点进行一次 NeuS SDF 查询；
    3）围绕 D_gs ± k * |sdf| 范围加入抖动采样；
    4）当 GS 不透明度过低时回退到均匀采样。
    """

    def __init__(
        self,
        gs_renderer: Any,
        sdf_network: Any,
        k_scale: float | list[float] = 4.0,
        min_spread: float = 0.02,
        opacity_threshold: float = 0.1,
        detach_gs_depth: bool = True,
        uniform_sampler: Optional[UniformSampler] = None,
    ):
        if torch is None:  # pragma: no cover - runtime guard
            raise RuntimeError("PyTorch is required for GSGuidedSampler.")
        super().__init__()
        self.gs_renderer = gs_renderer
        self.sdf_network = sdf_network
        # 支持粗/细两级采样窗口：k_scales 可为 float 或 [k_coarse, k_fine]
        if isinstance(k_scale, (list, tuple)):
            self.k_scales = [float(k) for k in k_scale]
        else:
            self.k_scales = [float(k_scale)]
        self.min_spread = float(min_spread)
        self.opacity_threshold = float(opacity_threshold)
        self.detach_gs_depth = detach_gs_depth
        self.uniform_sampler = uniform_sampler or UniformSampler()

    def get_z_vals(
        self,
        rays_o,
        rays_d,
        near,
        far,
        n_samples: int,
        gs_outputs: Optional[Dict[str, Any]] = None,
        **gs_kwargs: Any,
    ):
        """
        Depth-guided sampling using GS depth + one SDF query.

        依据 GS 渲染得到的深度和不透明度进行引导采样；若 GS 结果不可靠或缺失，
        会自动回退到均匀采样。返回形状为 [num_rays, n_samples] 的深度值。
        """
        if torch is None:  # pragma: no cover - runtime guard
            raise RuntimeError("PyTorch is required for GSGuidedSampler.")

        # 基本元数据：设备/数据类型/射线数量
        device = rays_o.device
        dtype = rays_o.dtype
        num_rays = rays_o.shape[0]

        # Step 0: 将 near/far 统一 reshape 成 [num_rays,1] 便于后续广播
        near_t = _reshape_bounds(near, num_rays, device, dtype)
        far_t = _reshape_bounds(far, num_rays, device, dtype)

        # Step 1: 获取 GS 渲染结果（深度 + 不透明度）；若调用方未传入则现场渲染
        if gs_outputs is None:
            if self.gs_renderer is None:
                raise ValueError(
                    "gs_renderer is required when gs_outputs is not provided."
                )
            gs_outputs = self.gs_renderer(rays_o=rays_o, rays_d=rays_d, **gs_kwargs)

        # Step 1.1: 解析深度与不透明度字段（兼容 alpha/alphas/opactiy 多种命名）
        depth = gs_outputs.get("depth") if isinstance(gs_outputs, dict) else None
        opacity = None
        if isinstance(gs_outputs, dict):
            opacity = (
                gs_outputs.get("opacity")
                or gs_outputs.get("alpha")
                or gs_outputs.get("alphas")
            )

        # Step 1.2: 若缺少深度，直接退回全局 uniform 采样
        if depth is None:
            # Missing GS depth entirely -> uniform fallback
            return self.uniform_sampler.get_z_vals(
                rays_o, rays_d, near_t, far_t, n_samples
            )

        # Step 1.3: 将深度/不透明度整理到统一形状，并可选择 detach 避免反传
        depth_t = _reshape_bounds(depth, num_rays, device, dtype)
        if self.detach_gs_depth:
            depth_t = depth_t.detach()

        opacity_t: Optional[torch.Tensor]
        if opacity is None:
            opacity_t = None
        else:
            opacity_t = _reshape_bounds(opacity, num_rays, device, dtype)
            if opacity_t.shape[-1] > 1:
                opacity_t = opacity_t[..., :1]

        # Step 2: 在 GS 表面点上做一次 SDF 查询（无梯度）
        surface_pts = rays_o + depth_t * rays_d
        with torch.no_grad():
            sdf_vals = self.sdf_network.sdf(surface_pts)
        sdf_abs = sdf_vals.abs()

        # Step 3: 按照 k_scales 自适应生成局部采样区间 [D_gs - k|sdf|, D_gs + k|sdf|]
        guided_chunks = []
        for k in self.k_scales:
            # 3.1 半径下限设为 min_spread，避免区间塌缩
            radius = torch.clamp(k * sdf_abs, min=self.min_spread)
            # 3.2 限制到全局 near/far 边界以内
            local_near = torch.maximum(depth_t - radius, near_t)
            local_far = torch.minimum(depth_t + radius, far_t)

            # 3.3 如果局部区间反转，则回退到全局 near/far
            invalid_interval = local_far <= local_near
            if invalid_interval.any():
                local_near = torch.where(invalid_interval, near_t, local_near)
                local_far = torch.where(invalid_interval, far_t, local_far)

            # 3.4 在局部区间内做分层采样，并根据 self.training 决定是否抖动
            guided_z_chunk = _stratified_between(
                local_near, local_far, n_samples, dtype, device, self.training
            )
            guided_chunks.append(guided_z_chunk)

        # Step 3.5: 拼接多尺度 z_vals（如 k=[3,1]）-> [num_rays, n_samples * len(k_scales)]
        guided_z = (
            torch.cat(guided_chunks, dim=-1)
            if len(guided_chunks) > 1
            else guided_chunks[0]
        )

        # Step 4: 如果 GS 不透明度过低，则对对应射线改用 uniform 采样（置信度回退）
        if opacity_t is not None:
            low_conf_mask = opacity_t.squeeze(-1) < self.opacity_threshold
        else:
            low_conf_mask = torch.zeros(num_rays, device=device, dtype=torch.bool)

        if low_conf_mask.any():
            uniform_z = self.uniform_sampler.get_z_vals(
                rays_o, rays_d, near_t, far_t, n_samples
            )
            low_conf_mask = low_conf_mask.unsqueeze(-1)
            guided_z = torch.where(low_conf_mask, uniform_z, guided_z)

        return guided_z


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _reshape_bounds(value: Any, num_rays: int, device, dtype):
    """
    Convert bounds/inputs to shape [num_rays, 1] on the target device.

    将传入的边界或输入张量转换为形状 [num_rays, 1]，并放到指定设备与数据类型上。
    """
    if isinstance(value, torch.Tensor):
        tensor = value.to(device=device, dtype=dtype)
    else:
        tensor = torch.as_tensor(value, device=device, dtype=dtype)

    if tensor.ndim == 0:
        tensor = tensor.expand(num_rays, 1)
    elif tensor.ndim == 1:
        tensor = tensor.reshape(num_rays, 1)
    else:
        tensor = tensor.reshape(num_rays, -1)

    if tensor.shape[1] > 1:
        tensor = tensor[:, :1]
    return tensor


def _stratified_between(
    near: torch.Tensor, far: torch.Tensor, n_samples: int, dtype, device, training: bool
):
    """
    Stratified sampling between per-ray near/far bounds.

    在每条光线对应的 near/far 范围内进行分层采样；训练模式下对区间加入随机抖动。
    """
    t_vals = torch.linspace(0.0, 1.0, steps=n_samples, device=device, dtype=dtype)
    t_vals = t_vals.view(1, -1)
    z_vals = near * (1.0 - t_vals) + far * t_vals

    if training:
        mids = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = torch.cat([mids, z_vals[..., -1:]], dim=-1)
        lower = torch.cat([z_vals[..., :1], mids], dim=-1)
        noise = torch.rand_like(lower)
        z_vals = lower + (upper - lower) * noise

    return z_vals
