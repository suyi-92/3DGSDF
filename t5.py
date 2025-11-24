"""
t5.py - 完整的 GS-NeuS 联合训练脚本 (基于最新 fusion 包装器架构)

这是一个基于 fusion wrapper 架构的完整联合训练脚本，实现了：
1. GS → SDF：深度指导采样
2. SDF → GS：几何引导 densify/prune
3. 互相几何监督：深度和法线一致性

使用方法：
    python t5.py --scene_name garden --joint_iterations 30000

详细参数说明请参考 t4_usage_cn.md
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, Any

try:
    import torch
except ImportError:
    print("Error: PyTorch is required. Please install torch.")
    sys.exit(1)

from fusion import DataService, FusionWrapper, SceneSpec


def parse_args() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="GS-NeuS 完整联合训练脚本 (t5.py)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python t5.py --scene_name garden --joint_iterations 30000
  python t5.py --scene_name garden --joint_iterations 30000 --mesh_every 500 --log_every 100
  python t5.py --scene_name lego --dg_k 3.0 --geom_depth_w 1.0 --geom_normal_w 0.1
        """,
    )

    # ========== 基本参数 ==========
    parser.add_argument(
        "--scene_name",
        type=str,
        required=True,
        help="场景名称 (必需)",
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        default="data",
        help="数据集根目录 (默认: data)",
    )

    # ========== 路径配置 ==========
    parser.add_argument(
        "--gaussian_repo",
        type=str,
        default="gaussian_splatting",
        help="3DGS 代码仓库路径 (默认: gaussian_splatting)",
    )
    parser.add_argument(
        "--neus_repo",
        type=str,
        default="NeuS",
        help="NeuS 代码仓库路径 (默认: NeuS)",
    )
    parser.add_argument(
        "--gaussian_source_root",
        type=str,
        default="work/gaussian_sources",
        help="GS 输入工作空间根目录 (默认: work/gaussian_sources)",
    )
    parser.add_argument(
        "--gaussian_model_root",
        type=str,
        default="work/gaussian_models",
        help="GS 输出模型根目录 (默认: work/gaussian_models)",
    )
    parser.add_argument(
        "--shared_workspace",
        type=str,
        default="work/fusion_workspace",
        help="融合共享工作空间 (默认: work/fusion_workspace)",
    )
    parser.add_argument(
        "--neus_conf",
        type=str,
        default="NeuS/confs/womask.conf",
        help="NeuS 基础配置文件 (默认: NeuS/confs/womask.conf)",
    )
    parser.add_argument(
        "--neus_case",
        type=str,
        default="",
        help="NeuS case 名称 (默认: 使用 scene_name)",
    )

    # ========== 训练控制 ==========
    parser.add_argument(
        "--joint_iterations",
        type=int,
        default=30000,
        help="总联合训练迭代次数 (默认: 30000)",
    )
    parser.add_argument(
        "--mesh_every",
        type=int,
        default=500,
        help="每 N 步将 NeuS mesh 同步到 GS (默认: 500)",
    )
    parser.add_argument(
        "--log_every",
        type=int,
        default=100,
        help="每 N 步打印统计信息 (默认: 100)",
    )
    parser.add_argument(
        "--save_every",
        type=int,
        default=5000,
        help="每 N 步保存检查点 (默认: 5000)",
    )
    parser.add_argument(
        "--validate_every",
        type=int,
        default=1000,
        help="每 N 步进行验证 (默认: 1000, 0表示禁用)",
    )

    # ========== 深度指导采样参数 (GS → SDF) ==========
    depth_group = parser.add_argument_group("深度指导采样参数 (GS → SDF)")
    depth_group.add_argument(
        "--dg_k",
        type=float,
        default=3.0,
        help="采样窗口乘数 k，窗口大小 = k * |SDF| (默认: 3.0)",
    )
    depth_group.add_argument(
        "--dg_min_near",
        type=float,
        default=0.01,
        help="最小 near 平面距离 (默认: 0.01)",
    )
    depth_group.add_argument(
        "--dg_max_far",
        type=float,
        default=100.0,
        help="最大 far 平面距离 (默认: 100.0)",
    )
    depth_group.add_argument(
        "--dg_max_age",
        type=int,
        default=1000,
        help="深度缓存最大过期步数 (默认: 1000)",
    )

    # ========== SDF 引导 Densify/Prune 参数 (SDF → GS) ==========
    sdf_group = parser.add_argument_group("SDF 引导 Densify/Prune 参数 (SDF → GS)")
    sdf_group.add_argument(
        "--sdf_sigma",
        type=float,
        default=0.5,
        help="μ(s) 的高斯衰减率 σ (默认: 0.5)",
    )
    sdf_group.add_argument(
        "--sdf_omega_g",
        type=float,
        default=0.3,  # REDUCED: 1.0 → 0.3 to prevent over-densification
        help="Densify 时 SDF 的权重 ω_g (默认: 0.3, 建议范围 0.1-0.5)",
    )
    sdf_group.add_argument(
        "--sdf_omega_p",
        type=float,
        default=0.5,
        help="Prune 时 SDF 的权重 ω_p (默认: 0.5)",
    )
    sdf_group.add_argument(
        "--sdf_tau_g",
        type=float,
        default=0.0005,  # INCREASED: 0.0002 → 0.0005 for stricter densify threshold
        help="Densify 触发阈值 τ_g (默认: 0.0005)",
    )
    sdf_group.add_argument(
        "--sdf_tau_p",
        type=float,
        default=0.01,  # INCREASED: 0.005 → 0.01 for more aggressive pruning
        help="Prune 触发阈值 τ_p (默认: 0.01)",
    )

    # ========== 几何监督参数 (互相几何监督) ==========
    geom_group = parser.add_argument_group("几何监督参数 (互相几何监督)")
    geom_group.add_argument(
        "--geom_depth_w",
        type=float,
        default=1.0,
        help="深度一致性损失权重 (默认: 1.0)",
    )
    geom_group.add_argument(
        "--geom_normal_w",
        type=float,
        default=0.1,
        help="法线一致性损失权重 (默认: 0.1)",
    )
    geom_group.add_argument(
        "--geom_eps",
        type=float,
        default=1e-6,
        help="法线归一化时的 epsilon (默认: 1e-6)",
    )

    # ========== 其他参数 ==========
    parser.add_argument(
        "--white_background",
        action="store_true",
        help="使用白色背景 (默认: 黑色背景)",
    )
    parser.add_argument(
        "--resolution_scales",
        type=float,
        nargs="+",
        default=[1.0],
        help="分辨率缩放比例 (默认: [1.0])",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="训练设备 (默认: cuda)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子 (默认: 42)",
    )

    return parser.parse_args()


def build_scene_spec(args: argparse.Namespace) -> SceneSpec:
    """构建 SceneSpec 配置对象"""
    scene_root = Path(args.dataset_root) / args.scene_name
    gaussian_source = Path(args.gaussian_source_root) / args.scene_name
    gaussian_model = Path(args.gaussian_model_root) / args.scene_name
    shared_workspace = Path(args.shared_workspace) / args.scene_name

    # 创建必要的目录
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
        resolution_scales=tuple(args.resolution_scales),
        device=args.device,
        white_background=args.white_background,
    )


def build_fusion_config(args: argparse.Namespace) -> Dict[str, Dict[str, Any]]:
    """构建融合配置字典"""
    return {
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
        "geom_loss": {
            "depth_w": args.geom_depth_w,
            "normal_w": args.geom_normal_w,
            "eps": args.geom_eps,
        },
    }


def print_training_header(args: argparse.Namespace):
    """打印训练开始信息"""
    print("=" * 80)
    print(" " * 20 + "GS-NeuS 联合训练 (t5.py)")
    print("=" * 80)
    print(f"场景名称: {args.scene_name}")
    print(f"数据集路径: {args.dataset_root}/{args.scene_name}")
    print(f"总迭代次数: {args.joint_iterations}")
    print(f"设备: {args.device}")
    print("-" * 80)
    print("融合参数:")
    print(f"  深度指导采样: k={args.dg_k}, max_age={args.dg_max_age}")
    print(
        f"  SDF 引导 Densify/Prune: σ={args.sdf_sigma}, ω_g={args.sdf_omega_g}, ω_p={args.sdf_omega_p}"
    )
    print(f"  几何监督: depth_w={args.geom_depth_w}, normal_w={args.geom_normal_w}")
    print("=" * 80)
    print()


def print_iteration_stats(step: int, payload: Dict[str, Any], elapsed: float):
    """打印训练迭代统计信息"""
    gs_state = payload["gaussian"]
    neus_state = payload["neus"]
    stats = payload["statistics"]

    # 计算每步平均耗时
    avg_time = elapsed / step if step > 0 else 0

    # 简短的单行输出
    print(
        f"[步骤 {step:6d}] "
        f"GS(loss={gs_state.loss:.4f}, n={gs_state.num_gaussians:6d}) "
        f"NeuS(loss={neus_state.loss:.4f}, color={neus_state.color_loss:.4f}) "
        f"命中率={stats.get('depth_hit_rate', 0):.1%} "
        f"时间={avg_time:.3f}s/it"
    )


def save_checkpoint(wrapper: FusionWrapper, step: int, args: argparse.Namespace):
    """保存训练检查点"""
    try:
        # 保存 GaussianSplatting 模型
        gs_ply = wrapper.gaussian.export_surface(iteration=step)
        print(f"  ✓ 保存 GS 模型: {gs_ply}")

        # 保存 NeuS mesh (可选)
        if hasattr(wrapper.neus, "runner") and wrapper.neus.runner is not None:
            try:
                neus_mesh = wrapper.neus.export_mesh(resolution=256)
                print(f"  ✓ 保存 NeuS mesh: {neus_mesh}")
            except Exception as e:
                print(f"  ⚠ NeuS mesh 保存失败: {e}")

    except Exception as e:
        print(f"  ✗ 检查点保存失败: {e}")


def main():
    """主函数"""
    args = parse_args()

    # 设置随机种子
    if args.seed is not None:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    # 设置默认tensor类型
    if torch.cuda.is_available() and args.device == "cuda":
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
        print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU device")

    # 打印训练配置
    print_training_header(args)

    # 构建配置
    spec = build_scene_spec(args)
    fusion_cfg = build_fusion_config(args)

    # 初始化数据服务
    print("正在初始化数据服务...")
    data_service = DataService(Path(spec.dataset_root))

    # 初始化融合包装器
    print("正在初始化融合包装器...")
    wrapper = FusionWrapper(
        spec=spec,
        gaussian_repo=Path(args.gaussian_repo),
        neus_repo=Path(args.neus_repo),
        data_service=data_service,
        gaussian_cfg={},
        neus_cfg={},
        fusion_cfg=fusion_cfg,
    )

    # Bootstrap 初始化
    print("正在启动模型...")
    wrapper.bootstrap()
    print("✓ 初始化完成！")
    print()

    # 打印可用 API
    apis = wrapper.describe_apis()
    print(f"已注册 {len(apis)} 个 API:")
    for name, desc in list(apis.items())[:5]:
        print(f"  - {name}: {desc}")
    if len(apis) > 5:
        print(f"  ... 以及其他 {len(apis) - 5} 个 API")
    print()

    # 训练主循环
    print("=" * 80)
    print("开始联合训练...")
    print("=" * 80)
    start_time = time.time()
    last_log_time = start_time

    try:
        for step in range(1, args.joint_iterations + 1):
            # 执行一步联合训练
            payload = wrapper.joint_step(
                mesh_every=args.mesh_every,
                log_every=args.log_every,
            )

            # 定期打印简短统计
            current_time = time.time()
            if step % 10 == 0 or step == args.joint_iterations:
                elapsed = current_time - start_time
                print_iteration_stats(step, payload, elapsed)
                last_log_time = current_time

            # 保存检查点
            if args.save_every > 0 and step % args.save_every == 0:
                print(f"\n保存检查点 (步骤 {step})...")
                save_checkpoint(wrapper, step, args)
                print()

            # 验证 (可选)
            if args.validate_every > 0 and step % args.validate_every == 0:
                print(f"\n[步骤 {step}] 验证中...")
                # 这里可以添加验证逻辑，例如渲染测试图像
                # 当前仅打印统计信息
                stats = wrapper.get_statistics()
                print(f"  - 高斯数量: {stats.get('num_gaussians', 'N/A')}")
                print(f"  - 深度缓存大小: {stats.get('depth_cache_size', 0)}")
                print(f"  - 深度命中率: {stats.get('depth_hit_rate', 0):.2%}")
                print(
                    f"  - Densify/Prune: +{stats.get('densify_count', 0)} / -{stats.get('prune_count', 0)}"
                )
                print()

    except KeyboardInterrupt:
        print("\n\n训练被用户中断！")
        print("正在保存当前状态...")
        save_checkpoint(wrapper, wrapper._joint_iter, args)
        print("✓ 状态已保存")
        sys.exit(0)

    # 训练完成
    total_time = time.time() - start_time
    print("\n" + "=" * 80)
    print("训练完成！")
    print("=" * 80)
    print(f"总迭代次数: {args.joint_iterations}")
    print(f"总耗时: {total_time:.2f}s ({total_time/60:.2f}min)")
    print(f"平均速度: {total_time/args.joint_iterations:.3f}s/it")
    print()

    # 最终统计
    final_stats = wrapper.get_statistics()
    print("最终统计:")
    print(f"  - 高斯数量: {final_stats.get('num_gaussians', 'N/A')}")
    print(f"  - NeuS 迭代: {final_stats.get('neus_iteration', 'N/A')}")
    print(f"  - 深度命中率: {final_stats.get('depth_hit_rate', 0):.2%}")
    print(
        f"  - 总 Densify/Prune: +{final_stats.get('densify_count', 0)} / -{final_stats.get('prune_count', 0)}"
    )
    print()

    # 保存最终模型
    print("保存最终模型...")
    save_checkpoint(wrapper, args.joint_iterations, args)

    # 输出模型路径
    print("\n模型已保存至:")
    print(f"  - GS 模型: {spec.gaussian_model_path}")
    print(f"  - NeuS 模型: {spec.shared_workspace}/neus_exp")
    print()
    print("训练完成！感谢使用 GS-NeuS 联合训练系统 (t5.py)。")


if __name__ == "__main__":
    main()
