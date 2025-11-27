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
    """Abstract sampler interface returning per-ray depth samples."""

    def get_z_vals(self, rays_o, rays_d, near, far, **kwargs):
        raise NotImplementedError


class UniformSampler(RaySampler):
    """Standard stratified sampler between global near/far planes."""

    def __init__(self, training: bool = True):
        super().__init__()
        self.training = training

    def get_z_vals(self, rays_o, rays_d, near, far, n_samples: int, **_: Any):
        if torch is None:  # pragma: no cover - runtime guard
            raise RuntimeError("PyTorch is required for UniformSampler.")

        device = rays_o.device
        dtype = rays_o.dtype
        num_rays = rays_o.shape[0]

        near_t = _reshape_bounds(near, num_rays, device, dtype)
        far_t = _reshape_bounds(far, num_rays, device, dtype)

        return _stratified_between(near_t, far_t, n_samples, dtype, device, self.training)


class GSGuidedSampler(RaySampler):
    """
    Implements the GSDF Section 3.2.1 depth-guided sampling strategy.

    Steps (per batch of rays):
    1) Render Gaussian Splatting depth/opacity for the rays.
    2) Query NeuS SDF once at the GS surface point.
    3) Sample within [D_gs - k * |sdf|, D_gs + k * |sdf|] with jitter.
    4) Fallback to uniform sampling when GS opacity is too low.
    """

    def __init__(
        self,
        gs_renderer: Any,
        sdf_network: Any,
        k_scale: float = 4.0,
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
        self.k_scale = float(k_scale)
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
        if torch is None:  # pragma: no cover - runtime guard
            raise RuntimeError("PyTorch is required for GSGuidedSampler.")

        device = rays_o.device
        dtype = rays_o.dtype
        num_rays = rays_o.shape[0]

        near_t = _reshape_bounds(near, num_rays, device, dtype)
        far_t = _reshape_bounds(far, num_rays, device, dtype)

        # 1) Obtain GS render outputs (depth + opacity/alpha)
        if gs_outputs is None:
            if self.gs_renderer is None:
                raise ValueError("gs_renderer is required when gs_outputs is not provided.")
            gs_outputs = self.gs_renderer(rays_o=rays_o, rays_d=rays_d, **gs_kwargs)

        depth = gs_outputs.get("depth") if isinstance(gs_outputs, dict) else None
        opacity = None
        if isinstance(gs_outputs, dict):
            opacity = (
                gs_outputs.get("opacity")
                or gs_outputs.get("alpha")
                or gs_outputs.get("alphas")
            )

        if depth is None:
            # Missing GS depth entirely -> uniform fallback
            return self.uniform_sampler.get_z_vals(rays_o, rays_d, near_t, far_t, n_samples)

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

        # 2) One-tap SDF query at GS surface points
        surface_pts = rays_o + depth_t * rays_d
        with torch.no_grad():
            sdf_vals = self.sdf_network.sdf(surface_pts)
        sdf_abs = sdf_vals.abs()

        # 3) Compute adaptive sampling interval
        radius = torch.clamp(self.k_scale * sdf_abs, min=self.min_spread)
        local_near = torch.maximum(depth_t - radius, near_t)
        local_far = torch.minimum(depth_t + radius, far_t)

        # Ensure valid ordering; otherwise fallback to global bounds
        invalid_interval = local_far <= local_near
        if invalid_interval.any():
            local_near = torch.where(invalid_interval, near_t, local_near)
            local_far = torch.where(invalid_interval, far_t, local_far)

        guided_z = _stratified_between(
            local_near, local_far, n_samples, dtype, device, self.training
        )

        # 5) Fallback to uniform sampling when GS opacity is unreliable
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
    """Convert bounds/inputs to shape [num_rays, 1] on the target device."""
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
    """Stratified sampling between per-ray near/far bounds."""
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
