"""
Unified training harness that runs GS->SDF guided sampling, SDF->GS densify/prune,
and joint depth/normal consistency losses via the fusion wrapper.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import torch

from fusion.wrapper import DataService, FusionWrapper, SceneSpec


def build_spec(args: argparse.Namespace) -> SceneSpec:
    scene_root = Path(args.dataset_root) / args.scene_name
    gaussian_source = Path(args.gaussian_source_root) / args.scene_name
    gaussian_model = Path(args.gaussian_model_root) / args.scene_name
    shared_workspace = Path(args.shared_workspace) / args.scene_name

    gaussian_source.mkdir(parents=True, exist_ok=True)
    gaussian_model.mkdir(parents=True, exist_ok=True)
    shared_workspace.mkdir(parents=True, exist_ok=True)

    return SceneSpec(
        scene_name=args.scene_name,
        dataset_root=str(scene_root),
        gaussian_source_path=str(gaussian_source),
        gaussian_model_path=str(gaussian_model),
        neus_conf_path=args.neus_conf,
        neus_case=args.neus_case or args.scene_name,
        shared_workspace=str(shared_workspace),
        white_background=args.white_background,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Joint GS<->SDF training harness (t3)")
    parser.add_argument("--scene_name", required=True, help="Scene identifier")
    parser.add_argument("--dataset_root", default="data", help="Dataset root containing the scene")
    parser.add_argument("--gaussian_repo", default="gaussian_splatting", help="Path to 3DGS repo")
    parser.add_argument("--neus_repo", default="NeuS", help="Path to NeuS repo")
    parser.add_argument("--gaussian_source_root", default="work/gaussian_sources", help="GS input workspace")
    parser.add_argument("--gaussian_model_root", default="work/gaussian_models", help="GS output checkpoints")
    parser.add_argument("--shared_workspace", default="work/fusion_exchange", help="Shared workspace for fusion")
    parser.add_argument("--neus_conf", default="NeuS/confs/wmask.conf", help="NeuS base config")
    parser.add_argument("--neus_case", default="", help="NeuS case name override")
    parser.add_argument("--white_background", action="store_true", help="Use white background for GS")
    parser.add_argument("--joint_iterations", type=int, default=200, help="Number of fusion steps to run")
    parser.add_argument("--mesh_every", type=int, default=100, help="Synchronize NeuS mesh to GS every N steps")
    # Depth-guided sampling params
    parser.add_argument("--dg_k", type=float, default=3.0, help="k factor for depth-guided near/far window")
    parser.add_argument("--dg_min_near", type=float, default=0.01, help="Minimum near plane clamp")
    parser.add_argument("--dg_max_far", type=float, default=100.0, help="Maximum far plane clamp")
    parser.add_argument("--dg_max_age", type=int, default=50, help="Max cache age before fallback")
    parser.add_argument("--geom_depth_w", type=float, default=1.0, help="Depth consistency weight")
    parser.add_argument("--geom_normal_w", type=float, default=0.1, help="Normal consistency weight")
    # SDF->GS parameters
    parser.add_argument("--sdf_sigma", type=float, default=0.5, help="Gaussian of SDF distance for mu(s)")
    parser.add_argument("--sdf_omega_g", type=float, default=1.0, help="Weight for SDF in densify")
    parser.add_argument("--sdf_omega_p", type=float, default=0.5, help="Weight for SDF in prune")
    parser.add_argument("--sdf_tau_g", type=float, default=1e-3, help="Threshold for densify trigger")
    parser.add_argument("--sdf_tau_p", type=float, default=0.01, help="Threshold for prune trigger")
    return parser.parse_args()


def main():
    args = parse_args()
    spec = build_spec(args)
    data_service = DataService(Path(spec.dataset_root))

    if torch.cuda.is_available():
        torch.set_default_tensor_type("torch.cuda.FloatTensor")

    fusion_cfg: Dict[str, Dict[str, float]] = {
        "depth_guidance": {
            "k": args.dg_k,
            "min_near": args.dg_min_near,
            "max_far": args.dg_max_far,
            "max_age": args.dg_max_age,
            "loss_weight": args.geom_depth_w,
        },
        "sdf_guidance": {
            "sigma": args.sdf_sigma,
            "omega_g": args.sdf_omega_g,
            "omega_p": args.sdf_omega_p,
            "tau_g": args.sdf_tau_g,
            "tau_p": args.sdf_tau_p,
        },
        "geom_loss": {"depth_w": args.geom_depth_w, "normal_w": args.geom_normal_w},
    }

    wrapper = FusionWrapper(
        spec=spec,
        gaussian_repo=Path(args.gaussian_repo),
        neus_repo=Path(args.neus_repo),
        data_service=data_service,
        gaussian_cfg={},
        neus_cfg={},
        fusion_cfg=fusion_cfg,
    )
    wrapper.bootstrap()

    for step in range(1, args.joint_iterations + 1):
        payload = wrapper.joint_step(mesh_every=args.mesh_every)
        if step % 10 == 0 or step == args.joint_iterations:
            print(
                f"[Fusion] step={step} "
                f"GS(loss={payload['gaussian'].loss:.4f}, gaussians={payload['gaussian'].num_gaussians}) "
                f"SDF(loss={payload['neus'].loss:.4f}, depth_hits={wrapper.neus._depth_hits})"
            )


if __name__ == "__main__":
    main()
