#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
t6.py - 带预热（Prewarm）的 GS-NeuS 联合训练脚本

功能概述：
1. 可选地先对 3DGS / NeuS 单独预热，并保存 checkpoint 元信息。
2. 支持从已有预热 checkpoint 加载后直接进入联合优化。
3. 默认行为与以往一致：不加任何预热参数时，直接从头联合训练。
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

from fusion import DataService, FusionWrapper, SceneSpec
from fusion.prewarm import (
    load_prewarm_gaussian,
    load_prewarm_neus,
    prewarm_gaussian,
    prewarm_neus,
)


def build_scene_spec(args: argparse.Namespace) -> SceneSpec:
    scene_root = Path(args.dataset_root) / args.scene_name
    gaussian_source = Path(args.gaussian_source_root) / args.scene_name
    gaussian_model = Path(args.gaussian_model_root) / args.scene_name
    shared_workspace = Path(args.shared_workspace) / args.scene_name

    for path in [gaussian_source, gaussian_model, shared_workspace]:
        path.mkdir(parents=True, exist_ok=True)

    return SceneSpec(
        scene_name=args.scene_name,
        dataset_root=str(scene_root),
        gaussian_source_path=str(gaussian_source),
        gaussian_model_path=str(gaussian_model),
        neus_conf_path=args.neus_conf,
        neus_case=args.neus_case or args.scene_name,
        shared_workspace=str(shared_workspace),
        white_background=args.white_background,
        device=args.device,
    )


def build_fusion_config(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "depth_guidance": {
            "k": args.dg_k,
            "min_near": args.dg_min_near,
            "max_far": args.dg_max_far,
            "max_age": args.dg_max_age,
        },
        "sdf_guidance": {
            "sigma": args.sdf_sigma,
            "omega_g": args.sdf_omega_g,
            "omega_p": args.sdf_omega_p,
            "tau_g": args.sdf_tau_g,
            "tau_p": args.sdf_tau_p,
        },
        "geom_loss": {
            "depth_w": args.geom_depth_w,
            "normal_w": args.geom_normal_w,
            "eps": 1e-6,
        },
        "ray_sampling": {
            "batch_size": args.batch_size,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GS-NeuS 联合训练 + 预热入口 (t6.py)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # 基础与路径
    parser.add_argument("--mode", type=str, default="fusion", choices=["fusion", "prewarm"])
    parser.add_argument("--scene_name", type=str, required=True)
    parser.add_argument("--dataset_root", type=str, default="data")
    parser.add_argument("--gaussian_repo", type=str, default="gaussian_splatting")
    parser.add_argument("--neus_repo", type=str, default="NeuS")
    parser.add_argument("--gaussian_source_root", type=str, default="work/gaussian_sources")
    parser.add_argument("--gaussian_model_root", type=str, default="work/gaussian_models")
    parser.add_argument("--shared_workspace", type=str, default="work/fusion_workspace")
    parser.add_argument("--neus_conf", type=str, default="NeuS/confs/womask.conf")
    parser.add_argument("--neus_case", type=str, default="")
    parser.add_argument("--white_background", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")

    # 联合训练
    parser.add_argument("--joint_iterations", type=int, default=30000)
    parser.add_argument("--mesh_every", type=int, default=500)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--save_every", type=int, default=5000)
    parser.add_argument("--validate_every", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=4096)

    # 预热控制
    parser.add_argument(
        "--fusion.prewarm_mode",
        dest="prewarm_mode",
        type=str,
        default="none",
        choices=["none", "load", "run_then_joint"],
        help="none: 不预热；load: 直接加载预热 checkpoint；run_then_joint: 先预热再联合",
    )
    parser.add_argument("--fusion.prewarm_3dgs", dest="prewarm_3dgs", action="store_true")
    parser.add_argument("--fusion.no_prewarm_3dgs", dest="prewarm_3dgs", action="store_false")
    parser.set_defaults(prewarm_3dgs=True)
    parser.add_argument("--fusion.prewarm_neus", dest="prewarm_neus", action="store_true")
    parser.add_argument("--fusion.no_prewarm_neus", dest="prewarm_neus", action="store_false")
    parser.set_defaults(prewarm_neus=True)
    parser.add_argument("--fusion.prewarm_3dgs_iters", type=int, default=10000)
    parser.add_argument("--fusion.prewarm_neus_iters", type=int, default=50000)
    parser.add_argument("--fusion.prewarm_output_root", type=str, default="output/fusion_prewarm")
    parser.add_argument("--fusion.prewarm_overwrite", action="store_true")
    parser.add_argument(
        "--fusion.allow_partial_prewarm",
        action="store_true",
        help="加载模式下允许缺失某一侧的 checkpoint，缺失时从头初始化",
    )

    # 深度、几何配置
    parser.add_argument("--dg_k", type=float, default=3.0)
    parser.add_argument("--dg_min_near", type=float, default=0.01)
    parser.add_argument("--dg_max_far", type=float, default=100.0)
    parser.add_argument("--dg_max_age", type=int, default=1000)
    parser.add_argument("--sdf_sigma", type=float, default=0.5)
    parser.add_argument("--sdf_omega_g", type=float, default=1.0)
    parser.add_argument("--sdf_omega_p", type=float, default=0.5)
    parser.add_argument("--sdf_tau_g", type=float, default=1e-3)
    parser.add_argument("--sdf_tau_p", type=float, default=0.01)
    parser.add_argument("--geom_depth_w", type=float, default=1.0)
    parser.add_argument("--geom_normal_w", type=float, default=0.1)

    return parser.parse_args()


def maybe_run_prewarm(wrapper: FusionWrapper, args: argparse.Namespace, prewarm_root: Path):
    results = {}
    if args.prewarm_3dgs:
        results["3dgs"] = prewarm_gaussian(
            wrapper,
            iterations=args.prewarm_3dgs_iters,
            log_every=args.log_every,
            output_root=prewarm_root,
            overwrite=args.prewarm_overwrite,
        )
    if args.prewarm_neus:
        results["neus"] = prewarm_neus(
            wrapper,
            iterations=args.prewarm_neus_iters,
            log_every=args.log_every,
            output_root=prewarm_root,
            overwrite=args.prewarm_overwrite,
        )
    return results


def maybe_load_prewarm(wrapper: FusionWrapper, args: argparse.Namespace, prewarm_root: Path):
    scene_root = prewarm_root / args.scene_name
    if args.prewarm_3dgs:
        meta_path = scene_root / "3dgs" / "meta.json"
        if meta_path.exists():
            load_prewarm_gaussian(wrapper.gaussian, meta_path)
            print(f"[Load] Loaded 3DGS prewarm from {meta_path}")
        elif not args.allow_partial_prewarm:
            raise FileNotFoundError(f"3DGS prewarm meta missing: {meta_path}")
    if args.prewarm_neus:
        meta_path = scene_root / "neus" / "meta.json"
        if meta_path.exists():
            load_prewarm_neus(wrapper.neus, meta_path)
            print(f"[Load] Loaded NeuS prewarm from {meta_path}")
        elif not args.allow_partial_prewarm:
            raise FileNotFoundError(f"NeuS prewarm meta missing: {meta_path}")


def run_joint_training(wrapper: FusionWrapper, args: argparse.Namespace):
    for step in range(1, args.joint_iterations + 1):
        payload = wrapper.joint_step(mesh_every=args.mesh_every, log_every=args.log_every)
        if args.save_every > 0 and step % args.save_every == 0:
            wrapper.gaussian.export_surface(iteration=payload["gaussian"].iteration)
        if args.validate_every > 0 and step % args.validate_every == 0:
            print(f"[Validate] fusion_step={payload['fusion_step']}")


def main():
    args = parse_args()
    spec = build_scene_spec(args)
    data_service = DataService(Path(spec.dataset_root))

    wrapper = FusionWrapper(
        spec=spec,
        gaussian_repo=Path(args.gaussian_repo),
        neus_repo=Path(args.neus_repo),
        data_service=data_service,
        gaussian_cfg={},
        neus_cfg={},
        fusion_cfg=build_fusion_config(args),
    )
    wrapper.bootstrap()

    prewarm_root = Path(args.prewarm_output_root)

    if args.mode == "prewarm":
        maybe_run_prewarm(wrapper, args, prewarm_root)
        return

    if args.prewarm_mode == "load":
        maybe_load_prewarm(wrapper, args, prewarm_root)
    elif args.prewarm_mode == "run_then_joint":
        maybe_run_prewarm(wrapper, args, prewarm_root)

    run_joint_training(wrapper, args)


if __name__ == "__main__":
    main()
