#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
t4.py - GS-NeuS 联合训练完整示例

这是一个完整的联合训练脚本，展示了如何使用 fusion.wrapper 进行 GaussianSplatting 和 NeuS 的深度融合训练。

主要功能：
1. GS → SDF：深度指导采样
2. SDF → GS：几何引导 densify/prune
3. 互相几何监督（深度和法线一致性损失）

使用示例：
    python t4.py --scene_name garden --joint_iterations 30000
    
    python t4.py --scene_name garden \
        --joint_iterations 30000 \
        --mesh_every 500 \
        --log_every 100 \
        --dg_k 3.0 \
        --geom_depth_w 1.0 \
        --geom_normal_w 0.1
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict

# 确保能导入项目模块
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    import torch
except ImportError:
    print("错误: 需要安装 PyTorch")
    print("请运行: pip install torch")
    sys.exit(1)

try:
    from fusion.wrapper import DataService, FusionWrapper, SceneSpec
except ImportError as e:
    print(f"错误: 无法导入 fusion.wrapper: {e}")
    print("请确保 fusion/ 目录在当前路径下")
    sys.exit(1)


def build_scene_spec(args: argparse.Namespace) -> SceneSpec:
    """
    根据命令行参数构建场景规格。

    Args:
        args: 命令行参数

    Returns:
        SceneSpec 对象
    """
    scene_root = Path(args.dataset_root) / args.scene_name
    if not scene_root.exists():
        raise FileNotFoundError(
            f"场景目录不存在: {scene_root}\n"
            f"请确保数据集位于 {args.dataset_root}/{args.scene_name}"
        )

    # 创建工作目录
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
    """
    根据命令行参数构建融合配置。

    Args:
        args: 命令行参数

    Returns:
        融合配置字典
    """
    return {
        # 深度指导采样配置
        "depth_guidance": {
            "k": args.dg_k,
            "min_near": args.dg_min_near,
            "max_far": args.dg_max_far,
            "max_age": args.dg_max_age,
        },
        # SDF 引导 densify/prune 配置
        "sdf_guidance": {
            "sigma": args.sdf_sigma,
            "omega_g": args.sdf_omega_g,
            "omega_p": args.sdf_omega_p,
            "tau_g": args.sdf_tau_g,
            "tau_p": args.sdf_tau_p,
        },
        # 几何损失配置
        "geom_loss": {
            "depth_w": args.geom_depth_w,
            "normal_w": args.geom_normal_w,
            "eps": 1e-6,
        },
    }


def print_config_summary(spec: SceneSpec, fusion_cfg: Dict[str, Any]):
    """打印配置摘要"""
    print("\n" + "=" * 60)
    print("GS-NeuS 联合训练配置")
    print("=" * 60)
    print(f"场景名称: {spec.scene_name}")
    print(f"数据集根目录: {spec.dataset_root}")
    print(f"设备: {spec.device}")
    print(f"白色背景: {spec.white_background}")
    print("\n--- 深度指导采样 ---")
    dg = fusion_cfg["depth_guidance"]
    print(f"  k (窗口乘数): {dg['k']}")
    print(f"  min_near: {dg['min_near']}")
    print(f"  max_far: {dg['max_far']}")
    print(f"  max_age (缓存过期): {dg['max_age']} 步")
    print("\n--- SDF 引导 Densify/Prune ---")
    sg = fusion_cfg["sdf_guidance"]
    print(f"  sigma (衰减率): {sg['sigma']}")
    print(f"  omega_g (densify 权重): {sg['omega_g']}")
    print(f"  omega_p (prune 权重): {sg['omega_p']}")
    print(f"  tau_g (densify 阈值): {sg['tau_g']}")
    print(f"  tau_p (prune 阈值): {sg['tau_p']}")
    print("\n--- 几何监督损失 ---")
    gl = fusion_cfg["geom_loss"]
    print(f"  depth_w (深度权重): {gl['depth_w']}")
    print(f"  normal_w (法线权重): {gl['normal_w']}")
    print("=" * 60 + "\n")


def training_callback(payload: Dict[str, Any]):
    """
    训练回调函数，可用于自定义日志记录或保存检查点。

    Args:
        payload: 包含 neus、gaussian、fusion_step、statistics 的字典
    """
    # 这里可以添加自定义逻辑，例如：
    # - 保存检查点
    # - 记录到 TensorBoard
    # - 发送到监控系统
    # - 等等
    pass


def parse_arguments() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="GS-NeuS 联合训练脚本 (t4.py)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # === 基础配置 ===
    parser.add_argument(
        "--scene_name",
        type=str,
        required=True,
        help="场景名称（必需）",
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        default="data",
        help="数据集根目录",
    )
    parser.add_argument(
        "--gaussian_repo",
        type=str,
        default="gaussian_splatting",
        help="3DGS 代码库路径",
    )
    parser.add_argument(
        "--neus_repo",
        type=str,
        default="NeuS",
        help="NeuS 代码库路径",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="运行设备 (cuda/cpu)",
    )

    # === 工作目录 ===
    parser.add_argument(
        "--gaussian_source_root",
        type=str,
        default="work/gaussian_sources",
        help="GS 输入工作空间",
    )
    parser.add_argument(
        "--gaussian_model_root",
        type=str,
        default="work/gaussian_models",
        help="GS 模型输出目录",
    )
    parser.add_argument(
        "--shared_workspace",
        type=str,
        default="work/fusion_workspace",
        help="融合共享工作空间",
    )

    # === NeuS 配置 ===
    parser.add_argument(
        "--neus_conf",
        type=str,
        default="NeuS/confs/wmask.conf",
        help="NeuS 基础配置文件",
    )
    parser.add_argument(
        "--neus_case",
        type=str,
        default="",
        help="NeuS case 名称（默认使用 scene_name）",
    )

    # === 训练参数 ===
    parser.add_argument(
        "--joint_iterations",
        type=int,
        default=30000,
        help="联合训练总迭代次数",
    )
    parser.add_argument(
        "--mesh_every",
        type=int,
        default=500,
        help="每 N 步同步 NeuS mesh 到 GS（0 表示禁用）",
    )
    parser.add_argument(
        "--log_every",
        type=int,
        default=100,
        help="每 N 步打印统计信息（0 表示禁用）",
    )
    parser.add_argument(
        "--white_background",
        action="store_true",
        help="使用白色背景（GS）",
    )

    # === 深度指导采样参数 ===
    depth_group = parser.add_argument_group("深度指导采样")
    depth_group.add_argument(
        "--dg_k",
        type=float,
        default=3.0,
        help="深度指导窗口乘数 k",
    )
    depth_group.add_argument(
        "--dg_min_near",
        type=float,
        default=0.01,
        help="最小 near 平面距离",
    )
    depth_group.add_argument(
        "--dg_max_far",
        type=float,
        default=100.0,
        help="最大 far 平面距离",
    )
    depth_group.add_argument(
        "--dg_max_age",
        type=int,
        default=50,
        help="深度缓存最大年龄（步数）",
    )

    # === SDF 引导 Densify/Prune 参数 ===
    sdf_group = parser.add_argument_group("SDF 引导 Densify/Prune")
    sdf_group.add_argument(
        "--sdf_sigma",
        type=float,
        default=0.5,
        help="mu(s) 的高斯衰减率 sigma",
    )
    sdf_group.add_argument(
        "--sdf_omega_g",
        type=float,
        default=1.0,
        help="SDF 在 densify 中的权重 omega_g",
    )
    sdf_group.add_argument(
        "--sdf_omega_p",
        type=float,
        default=0.5,
        help="SDF 在 prune 中的权重 omega_p",
    )
    sdf_group.add_argument(
        "--sdf_tau_g",
        type=float,
        default=0.0002,
        help="Densify 梯度阈值 tau_g",
    )
    sdf_group.add_argument(
        "--sdf_tau_p",
        type=float,
        default=0.005,
        help="Prune 不透明度阈值 tau_p",
    )

    # === 几何监督损失参数 ===
    geom_group = parser.add_argument_group("几何监督损失")
    geom_group.add_argument(
        "--geom_depth_w",
        type=float,
        default=1.0,
        help="深度一致性损失权重",
    )
    geom_group.add_argument(
        "--geom_normal_w",
        type=float,
        default=0.1,
        help="法线一致性损失权重",
    )

    return parser.parse_args()


def main():
    """主训练函数"""
    args = parse_arguments()

    # 设置设备
    if args.device == "cuda" and torch.cuda.is_available():
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
        print(f"使用 GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("使用 CPU")

    # 构建配置
    try:
        spec = build_scene_spec(args)
        fusion_cfg = build_fusion_config(args)
    except Exception as e:
        print(f"错误: {e}")
        sys.exit(1)

    # 打印配置摘要
    print_config_summary(spec, fusion_cfg)

    # 创建数据服务
    print("正在初始化数据服务...")
    data_service = DataService(Path(spec.dataset_root))

    # 创建融合 wrapper
    print("正在创建融合 wrapper...")
    wrapper = FusionWrapper(
        spec=spec,
        gaussian_repo=Path(args.gaussian_repo),
        neus_repo=Path(args.neus_repo),
        data_service=data_service,
        gaussian_cfg={},
        neus_cfg={},
        fusion_cfg=fusion_cfg,
    )

    # 引导两个系统
    print("正在引导 GaussianSplatting 和 NeuS...")
    try:
        wrapper.bootstrap()
    except Exception as e:
        print(f"引导失败: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    print("\n开始联合训练...\n")

    # 训练循环
    try:
        for step in range(1, args.joint_iterations + 1):
            # 执行一步联合训练
            payload = wrapper.joint_step(
                mesh_every=args.mesh_every,
                log_every=args.log_every,
                callback=training_callback,
            )

            # 简洁日志（每 10 步或最后一步）
            if step % 10 == 0 or step == args.joint_iterations:
                gs_state = payload["gaussian"]
                neus_state = payload["neus"]
                stats = payload["statistics"]

                print(
                    f"[步骤 {step:>6d}/{args.joint_iterations}] "
                    f"GS(loss={gs_state.loss:.4f}, n={gs_state.num_gaussians:>6d}) "
                    f"NeuS(loss={neus_state.loss:.4f}, color={neus_state.color_loss:.4f}) "
                    f"命中率={stats['depth_hit_rate']:.1%}"
                )

            # 中期检查点（可选）
            if step % 5000 == 0:
                print(f"\n到达检查点 {step}，统计信息：")
                stats = wrapper.get_statistics()
                for key, value in stats.items():
                    print(f"  {key}: {value}")
                print()

    except KeyboardInterrupt:
        print("\n\n训练被用户中断")
        print("保存当前状态...")
        # 这里可以添加保存逻辑

    except Exception as e:
        print(f"\n训练出错: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    # 训练完成
    print("\n" + "=" * 60)
    print("联合训练完成！")
    print("=" * 60)

    # 最终统计
    final_stats = wrapper.get_statistics()
    print("\n最终统计:")
    print(f"  总迭代数: {args.joint_iterations}")
    print(f"  最终高斯数量: {final_stats.get('num_gaussians', 'N/A')}")
    print(f"  深度缓存大小: {final_stats['depth_cache_size']}")
    print(f"  深度命中率: {final_stats['depth_hit_rate']:.2%}")
    print(f"  总 densify 数: {final_stats['densify_count']}")
    print(f"  总 prune 数: {final_stats['prune_count']}")
    print(f"  NeuS 迭代数: {final_stats.get('neus_iteration', 'N/A')}")

    print(f"\n模型保存在:")
    print(f"  GS: {spec.gaussian_model_path}")
    print(f"  NeuS: {spec.shared_workspace}/neus_exp")
    print()


if __name__ == "__main__":
    main()
