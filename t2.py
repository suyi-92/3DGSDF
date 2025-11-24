"""
Standalone training harness using the fusion wrapper to reproduce the original
Gaussian Splatting and NeuS pipelines.

English: Demonstrates how to feed mip-NeRF360 data into the adapters and run
per-architecture training loops without touching upstream code.

使用 FusionWrapper 还原 3DGS 和 NeuS 的原始训练流程（含数据导入与训练），
全部操作都通过包装器 API 完成。
"""

from __future__ import annotations

import torch
import argparse
from pathlib import Path
from typing import Dict

from fusion.wrapper import DataService, FusionWrapper, SceneSpec


def build_spec(args: argparse.Namespace) -> SceneSpec:
    """
    Convert CLI arguments into a SceneSpec instance.

    根据命令行参数创建 SceneSpec，统一管理路径配置。
    """
    raw_scene = Path(args.scene_name)
    if raw_scene.is_absolute() or raw_scene.parts[0] in ("..", "."):
        scene_root = raw_scene.resolve()
        scene_name = scene_root.name
    else:
        scene_root = Path(args.dataset_root) / raw_scene
        scene_name = raw_scene.name if raw_scene.name else str(raw_scene)

    gaussian_source = Path(args.gaussian_source_root) / scene_name
    gaussian_model = Path(args.gaussian_model_root) / scene_name
    shared_workspace = Path(args.shared_workspace) / scene_name

    gaussian_source.mkdir(parents=True, exist_ok=True)
    gaussian_model.mkdir(parents=True, exist_ok=True)
    shared_workspace.mkdir(parents=True, exist_ok=True)

    return SceneSpec(
        scene_name=scene_name,
        dataset_root=str(scene_root),
        gaussian_source_path=str(gaussian_source),
        gaussian_model_path=str(gaussian_model),
        neus_conf_path=args.neus_conf,
        neus_case=args.neus_case or args.scene_name,
        shared_workspace=str(shared_workspace),
        white_background=args.white_background,
    )


def run_gaussian_training(
    wrapper: FusionWrapper,
    iterations: int,
    log_every: int = 100,
):
    """
    Execute the Gaussian Splatting training loop for the requested number of
    iterations and log intermediate metrics.

    运行指定步数的 3DGS 训练循环，并定期打印损失指标。
    """
    for step in range(1, iterations + 1):
        state = wrapper.gaussian.train_step()
        if step % log_every == 0 or step == iterations:
            print(
                f"[3DGS] iter={state.iteration} loss={state.loss:.4f} "
                f"l1={state.l1:.4f} ssim={state.ssim:.4f} "
                f"gaussians={state.num_gaussians}"
            )


def run_neus_training(
    wrapper: FusionWrapper,
    iterations: int,
    log_every: int = 100,
):
    """
    Execute the NeuS training loop via the adapter for the requested steps.

    通过适配器运行 NeuS 训练循环，定期打印核心损失。
    """
    for step in range(1, iterations + 1):
        state = wrapper.neus.train_step()
        if step % log_every == 0 or step == iterations:
            print(
                f"[NeuS] iter={state.iteration} loss={state.loss:.4f} "
                f"color={state.color_loss:.4f} eikonal={state.eikonal_loss:.4f}"
            )


def parse_args() -> argparse.Namespace:
    """
    Build the command line interface for selecting scenes, paths, and modes.

    构建命令行参数，便于选择场景、路径和训练模式。
    """
    parser = argparse.ArgumentParser(description="Run 3DGS / NeuS via fusion wrapper")
    parser.add_argument(
        "--scene_name", required=True, help="Scene identifier / 场景名称"
    )
    parser.add_argument(
        "--dataset_root",
        default="data",
        help="Directory containing mip-NeRF360 scene folders / 数据根目录",
    )
    parser.add_argument(
        "--gaussian_repo",
        default="gaussian_splatting",
        help="Path to Gaussian-Splatting repo / 3DGS 仓库路径",
    )
    parser.add_argument(
        "--neus_repo",
        default="NeuS",
        help="Path to NeuS repo / NeuS 仓库路径",
    )
    parser.add_argument(
        "--gaussian_source_root",
        default="work/gaussian_sources",
        help="Workspace to materialize GS inputs / GS 输入工作区",
    )
    parser.add_argument(
        "--gaussian_model_root",
        default="work/gaussian_models",
        help="Directory storing GS checkpoints / GS 模型输出目录",
    )
    parser.add_argument(
        "--shared_workspace",
        default="work/fusion_exchange",
        help="Shared workspace for exchange artifacts / 共享工作区",
    )
    parser.add_argument(
        "--neus_conf",
        default="NeuS/confs/womask.conf",
        help="NeuS configuration file / NeuS 配置文件",
    )
    parser.add_argument(
        "--neus_case",
        default="",
        help="NeuS case override (default=scene_name) / NeuS case 名称",
    )
    parser.add_argument(
        "--mode",
        choices=["gaussian", "neus", "both"],
        default="both",
        help="Choose which pipeline(s) to run / 选择运行的训练流程",
    )
    parser.add_argument(
        "--gaussian_iterations",
        type=int,
        default=1000,
        help="Number of GS iterations / GS 迭代次数",
    )
    parser.add_argument(
        "--neus_iterations",
        type=int,
        default=1000,
        help="Number of NeuS iterations / NeuS 迭代次数",
    )
    parser.add_argument(
        "--white_background",
        action="store_true",
        help="Assume white background for GS / GS 白背景开关",
    )
    parser.add_argument(
        "--log_every",
        type=int,
        default=100,
        help="How often to print metrics / 指标打印间隔",
    )
    parser.add_argument(
        "--list_apis",
        action="store_true",
        help="List all registered fusion APIs / 输出所有注册的 API",
    )
    return parser.parse_args()


def main():
    """
    Entry point: bootstrap adapters and run the selected training routines.

    程序入口，初始化适配器并运行指定的训练过程。
    """
    args = parse_args()
    spec = build_spec(args)
    data_service = DataService(Path(spec.dataset_root))

    if torch.cuda.is_available():
        torch.set_default_tensor_type("torch.cuda.FloatTensor")

    wrapper = FusionWrapper(
        spec=spec,
        gaussian_repo=Path(args.gaussian_repo),
        neus_repo=Path(args.neus_repo),
        data_service=data_service,
        gaussian_cfg={},  # 可在此传入额外 3DGS 配置
        neus_cfg={},  # 可在此传入额外 NeuS 配置
    )
    wrapper.bootstrap()
    api_map = wrapper.describe_apis()
    print("Registered APIs / 已注册 API：")
    for name, desc in api_map.items():
        print(f"  - {name}: {desc}")

    # 示例：调用注册 API 渲染一帧并传给 NeuS 作为额外监督
    try:
        sample_camera = wrapper.gaussian.scene.getTrainCameras()[0]
        render = wrapper.registry.call("gaussian.render", sample_camera)
        wrapper.registry.call(
            "neus.inject_supervision",
            {"render": render, "camera": sample_camera.image_name},
        )
        print(
            f"[Example] Rendered view '{sample_camera.image_name}' "
            f"(mean={render.mean().item():.4f}) and sent to NeuS supervision."
        )
    except Exception as exc:
        print(f"[Example] Failed to run cross-model API demo: {exc}")

    if args.list_apis:
        return

    if args.mode in ("gaussian", "both"):
        run_gaussian_training(wrapper, args.gaussian_iterations, args.log_every)

    if args.mode in ("neus", "both"):
        run_neus_training(wrapper, args.neus_iterations, args.log_every)


if __name__ == "__main__":
    main()
