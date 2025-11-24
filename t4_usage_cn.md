# t4.py 使用说明

## 简介

`t4.py` 是 GS-NeuS 联合训练的完整示例脚本，展示了如何使用 `fusion/wrapper.py` 进行深度融合训练。

## 快速开始

### 基本用法

```bash
python t4.py --scene_name garden --joint_iterations 30000
```

### 完整示例（推荐配置）

```bash
python t4.py \
    --scene_name garden \
    --joint_iterations 30000 \
    --mesh_every 500 \
    --log_every 100 \
    --dg_k 3.0 \
    --geom_depth_w 1.0 \
    --geom_normal_w 0.1
```

## 数据集准备

确保你的数据集结构如下：

```
data/
  garden/
    images/
      IMG_0001.jpg
      IMG_0002.jpg
      ...
    sparse/
      0/
        cameras.bin
        images.bin
        points3D.bin
    poses_bounds.npy
```

## 主要参数

### 必需参数

- `--scene_name`: 场景名称（必需）

### 训练控制

- `--joint_iterations`: 总迭代次数（默认：30000）
- `--mesh_every`: 每 N 步同步 NeuS mesh 到 GS（默认：500）
- `--log_every`: 每 N 步打印统计（默认：100）

### 深度指导采样

- `--dg_k`: 窗口乘数（默认：3.0）
- `--dg_min_near`: 最小 near 平面（默认：0.01）
- `--dg_max_far`: 最大 far 平面（默认：100.0）
- `--dg_max_age`: 缓存过期步数（默认：50）

### SDF 引导 Densify/Prune

- `--sdf_sigma`: 衰减率（默认：0.5）
- `--sdf_omega_g`: Densify 权重（默认：1.0）
- `--sdf_omega_p`: Prune 权重（默认：0.5）
- `--sdf_tau_g`: Densify 阈值（默认：0.0002）
- `--sdf_tau_p`: Prune 阈值（默认：0.005）

### 几何监督

- `--geom_depth_w`: 深度一致性权重（默认：1.0）
- `--geom_normal_w`: 法线一致性权重（默认：0.1）

## 输出

训练完成后，模型保存在：

- **GaussianSplatting**: `work/gaussian_models/{scene_name}/`
- **NeuS**: `work/fusion_workspace/{scene_name}/neus_exp/`

## 监控训练

训练过程中会显示：

```
[步骤   100/30000] GS(loss=0.1234, n=156342) NeuS(loss=0.0567, color=0.0234) 命中率=87.3%
```

每 100 步会显示详细统计：

```
=== Fusion Statistics (iter 100) ===
Depth Cache: 25 entries
Depth Hit Rate: 87.32%
Num Gaussians: 156342
Densify/Prune: +5234 / -1823
NeuS Iter: 100
==================================================
```

## 故障排查

### 深度命中率过低

如果命中率 < 50%：
- 增加 `--dg_max_age`
- 检查 camera ID 是否匹配

### 训练不稳定

如果损失震荡：
- 减小 `--geom_depth_w` 和 `--geom_normal_w`
- 增大 `--dg_k` 扩大采样窗口

### 没有 Densify/Prune

如果高斯数量不变：
- 检查是否在 densification 窗口内
- 增大 `--sdf_omega_g`

## 更多信息

详细实现说明请参考：
- [`fusion_implementation_guide_cn.md`](fusion_implementation_guide_cn.md) - 中文实现指南
- [`fusion/README.md`](fusion/README.md) - 融合架构设计文档
