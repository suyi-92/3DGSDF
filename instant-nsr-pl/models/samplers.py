from typing import Optional

import torch
import torch.nn as nn


class RaySampler(nn.Module):
    """Abstract interface for ray sampling."""

    def get_z_vals(self, rays_o, rays_d, near, far, **kwargs):
        raise NotImplementedError


class UniformSampler(RaySampler):
    """Standard stratified uniform sampler between the near and far planes."""

    def __init__(self, stratified: bool = True):
        super().__init__()
        self.stratified = stratified

    def get_z_vals(self, rays_o, rays_d, near, far, n_samples: int, stratified: Optional[bool] = None):
        device = rays_o.device
        dtype = rays_o.dtype
        stratified = self.stratified if stratified is None else stratified

        near = torch.as_tensor(near, device=device, dtype=dtype)
        far = torch.as_tensor(far, device=device, dtype=dtype)
        if near.ndim == 0:
            near = near.expand(rays_o.shape[0])
        if far.ndim == 0:
            far = far.expand_as(near)

        t_vals = torch.linspace(0.0, 1.0, n_samples, device=device, dtype=dtype)
        z_vals = near[..., None] * (1.0 - t_vals) + far[..., None] * t_vals

        if stratified:
            mids = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
            upper = torch.cat([mids, z_vals[..., -1:]], dim=-1)
            lower = torch.cat([z_vals[..., :1], mids], dim=-1)
            t_rand = torch.rand_like(lower)
            z_vals = lower + (upper - lower) * t_rand

        return z_vals


class GSGuidedSampler(RaySampler):
    """
    Depth-guided sampler following GSDF Section 3.2.1.

    The sampler uses Gaussian Splatting (GS) depth/opacity to focus samples
    around surfaces and probes the SDF once for adaptive intervals.
    """

    def __init__(self, gs_renderer, sdf_network, k_scale: float = 4.0, min_spread: float = 0.02,
                 opacity_threshold: float = 0.1, stratified: bool = True, detach_gs: bool = True):
        super().__init__()
        self.gs_renderer = gs_renderer
        self.sdf_network = sdf_network
        self.k_scale = k_scale
        self.min_spread = min_spread
        self.opacity_threshold = opacity_threshold
        self.stratified = stratified
        self.detach_gs = detach_gs
        self.uniform_sampler = UniformSampler(stratified=stratified)

    def _extract_depth_opacity(self, gs_outputs):
        depth = None
        opacity = None
        if isinstance(gs_outputs, dict):
            depth = gs_outputs.get('depth', None)
            opacity = gs_outputs.get('opacity', None) or gs_outputs.get('alpha', None)
        else:
            depth = getattr(gs_outputs, 'depth', None)
            opacity = getattr(gs_outputs, 'opacity', None)
            if opacity is None:
                opacity = getattr(gs_outputs, 'alpha', None)
        if depth is None:
            raise ValueError('GS renderer output must provide a depth map via "depth" key/attribute.')
        return depth, opacity

    def _query_sdf(self, points, sdf_kwargs):
        sdf_outputs = self.sdf_network(points, **sdf_kwargs)
        if isinstance(sdf_outputs, (tuple, list)):
            sdf_val = sdf_outputs[0]
        else:
            sdf_val = sdf_outputs
        return sdf_val.squeeze(-1)

    def get_z_vals(self, rays_o, rays_d, near, far, n_samples: int, gs_kwargs=None, sdf_kwargs=None,
                   stratified: Optional[bool] = None):
        gs_kwargs = gs_kwargs or {}
        sdf_kwargs = sdf_kwargs or {}
        stratified = self.stratified if stratified is None else stratified

        gs_outputs = self.gs_renderer(rays_o=rays_o, rays_d=rays_d, **gs_kwargs)
        depth_gs, opacity = self._extract_depth_opacity(gs_outputs)
        depth_gs = depth_gs.squeeze(-1)
        if self.detach_gs:
            depth_gs = depth_gs.detach()
        if opacity is None:
            opacity = torch.ones_like(depth_gs)
        opacity = opacity.squeeze(-1)

        surface_points = rays_o + rays_d * depth_gs[..., None]
        sdf_vals = self._query_sdf(surface_points, sdf_kwargs=sdf_kwargs)
        spread = self.k_scale * sdf_vals.abs()
        spread = torch.clamp_min(spread, self.min_spread)

        device = rays_o.device
        dtype = rays_o.dtype
        near = torch.as_tensor(near, device=device, dtype=dtype)
        far = torch.as_tensor(far, device=device, dtype=dtype)
        if near.ndim == 0:
            near = near.expand_as(depth_gs)
        if far.ndim == 0:
            far = far.expand_as(depth_gs)

        start = torch.maximum(depth_gs - spread, near)
        end = torch.minimum(depth_gs + spread, far)
        # Ensure the interval is not degenerate
        end = torch.maximum(end, start + 1e-6)

        t_vals = torch.linspace(0.0, 1.0, n_samples, device=device, dtype=dtype)
        guided = start[..., None] * (1.0 - t_vals) + end[..., None] * t_vals
        if stratified:
            mids = 0.5 * (guided[..., 1:] + guided[..., :-1])
            upper = torch.cat([mids, guided[..., -1:]], dim=-1)
            lower = torch.cat([guided[..., :1], mids], dim=-1)
            t_rand = torch.rand_like(lower)
            guided = lower + (upper - lower) * t_rand

        low_opacity_mask = opacity < self.opacity_threshold
        if low_opacity_mask.any():
            uniform = self.uniform_sampler.get_z_vals(rays_o, rays_d, near, far, n_samples, stratified=stratified)
            guided = torch.where(low_opacity_mask[..., None], uniform, guided)

        return guided
