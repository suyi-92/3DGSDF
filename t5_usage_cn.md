# t5.py 使用说明

## 简介

`t5.py` 是基于最新 fusion wrapper 架构的**完整** GS-NeuS 联合训练脚本，相比 t3.py 和 t4.py 提供了更完善的功能和更详细的输出。

## 核心特性

### 三大融合功能

1. **GS → SDF：深度指导采样**
   - GaussianSplatting 渲染的深度图指导 NeuS 的采样区间
   - 自适应窗口：`near = D - k*|SDF|`, `far = D + k*|SDF|`
   - 深度缓存机制，支持过期策略

2. **SDF → GS：几何引导 densify/prune**
   - 利用 NeuS SDF 预测引导高斯点的生长和修剪
   - 计算 μ(s) = exp(-s²/(2σ²)) 权重靠近表面的点
   - 增强梯度：ε_g = ∇g + ω_g * μ(s)

3. **互相几何监督**
   - 深度一致性：|D_gs - D_sdf|
   - 法线一致性：1 - dot(n_gs, n_sdf)
   - 双向监督，提升几何质量

### 新增功能 (相比 t3.py/t4.py)

- ✅ 完整的训练统计和监控
- ✅ 定期检查点保存
- ✅ 可选的验证步骤
- ✅ 详细的日志输出
- ✅ 优雅的中断处理 (Ctrl+C)
- ✅ 训练进度追踪 (时间/速度统计)
- ✅ 可配置的随机种子
- ✅ 更清晰的参数分组和帮助信息

## 快速开始

### 基本用法

```bash
python t5.py --scene_name garden --joint_iterations 30000
```

### 推荐配置

```bash
python t5.py \
    --scene_name garden \
    --joint_iterations 30000 \
    --mesh_every 500 \
    --log_every 100 \
    --save_every 5000 \
    --dg_k 3.0 \
    --geom_depth_w 1.0 \
    --geom_normal_w 0.1
```

## 数据集准备

确保数据集结构如下：

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

## 参数详解

### 必需参数

- `--scene_name`: 场景名称 (必需)

### 路径配置

- `--dataset_root`: 数据集根目录 (默认: `data`)
- `--gaussian_repo`: 3DGS 代码仓库路径 (默认: `gaussian_splatting`)
- `--neus_repo`: NeuS 代码仓库路径 (默认: `NeuS`)
- `--gaussian_source_root`: GS 输入工作空间根目录 (默认: `work/gaussian_sources`)
- `--gaussian_model_root`: GS 输出模型根目录 (默认: `work/gaussian_models`)
- `--shared_workspace`: 融合共享工作空间 (默认: `work/fusion_workspace`)
- `--neus_conf`: NeuS 基础配置文件 (默认: `NeuS/confs/wmask.conf`)

### 训练控制

- `--joint_iterations`: 总联合训练迭代次数 (默认: 30000)
- `--mesh_every`: 每 N 步将 NeuS mesh 同步到 GS (默认: 500)
- `--log_every`: 每 N 步打印统计信息 (默认: 100)
- `--save_every`: 每 N 步保存检查点 (默认: 5000)
- `--validate_every`: 每 N 步进行验证 (默认: 1000，0表示禁用)

### 深度指导采样参数 (GS → SDF)

- `--dg_k`: 采样窗口乘数 k (默认: 3.0)
  - 窗口大小 = k × |SDF|
  - 较大的 k 提供更宽松的采样范围
- `--dg_min_near`: 最小 near 平面距离 (默认: 0.01)
- `--dg_max_far`: 最大 far 平面距离 (默认: 100.0)
- `--dg_max_age`: 深度缓存最大过期步数 (默认: 50)

### SDF 引导 Densify/Prune 参数 (SDF → GS)

- `--sdf_sigma`: μ(s) 的高斯衰减率 σ (默认: 0.5)
  - 控制 SDF 距离对权重的影响范围
- `--sdf_omega_g`: Densify 时 SDF 的权重 ω_g (默认: 1.0)
  - 越大则 SDF 对 densify 的影响越强
- `--sdf_omega_p`: Prune 时 SDF 的权重 ω_p (默认: 0.5)
  - 越大则 SDF 对 prune 的影响越强
- `--sdf_tau_g`: Densify 触发阈值 τ_g (默认: 0.0002)
- `--sdf_tau_p`: Prune 触发阈值 τ_p (默认: 0.005)

### 几何监督参数 (互相几何监督)

- `--geom_depth_w`: 深度一致性损失权重 (默认: 1.0)
  - 控制深度一致性对 NeuS 训练的影响
- `--geom_normal_w`: 法线一致性损失权重 (默认: 0.1)
  - 控制法线一致性对 NeuS 训练的影响
- `--geom_eps`: 法线归一化时的 epsilon (默认: 1e-6)

### 其他参数

- `--white_background`: 使用白色背景 (默认: 黑色背景)
- `--resolution_scales`: 分辨率缩放比例 (默认: [1.0])
- `--device`: 训练设备 (默认: cuda)
- `--seed`: 随机种子 (默认: 42)

## 输出说明

### 训练过程中

每 10 步会显示简短统计：
```
[步骤    100] GS(loss=0.1234, n=156342) NeuS(loss=0.0567, color=0.0234) 命中率=87.3% 时间=0.125s/it
```

每 100 步会显示详细统计（由 `--log_every` 控制）：
```
=== Fusion Statistics (iter 100) ===
Depth Cache: 25 entries
Depth Hit Rate: 87.32%
Num Gaussians: 156342
Densify/Prune: +5234 / -1823
NeuS Iter: 100
==================================================
```

### 模型保存位置

训练完成后，模型保存在：

- **GaussianSplatting**: `work/gaussian_models/{scene_name}/`
- **NeuS**: `work/fusion_workspace/{scene_name}/neus_exp/`

## 监控训练

### 理想状态指标

- **深度命中率**: > 80%
  - 如果过低，增加 `--dg_max_age`
- **Densify/Prune**: 持续有变化
  - 如果没有变化，检查是否在 densification 窗口内
- **损失值**: 持续下降
  - 如果震荡，减小几何监督权重

## 故障排查

### 深度命中率过低 (< 50%)

原因：深度缓存经常失效
解决方案：
- 增加 `--dg_max_age` (例如 100)
- 检查 camera ID 匹配是否正确

### 训练不稳定 (损失震荡)

原因：几何监督权重过大
解决方案：
- 减小 `--geom_depth_w` (例如 0.5)
- 减小 `--geom_normal_w` (例如 0.05)
- 增大 `--dg_k` 扩大采样窗口 (例如 5.0)

### 没有 Densify/Prune

原因：可能在 densification 窗口外，或权重过小
解决方案：
- 检查训练是否在 densification 窗口内 (通常前 15000 步)
- 增大 `--sdf_omega_g` (例如 2.0)
- 减小 `--sdf_tau_g` (例如 0.0001)

### 内存不足

原因：场景过大或高斯数量过多
解决方案：
- 减小 `--resolution_scales` (例如 [0.5])
- 增大 `--sdf_tau_p` 增强 prune (例如 0.01)
- 使用更小的场景测试

## 中断与恢复

### 优雅中断

训练过程中按 `Ctrl+C` 可以优雅中断：
- 自动保存当前状态
- 输出当前统计信息

### 恢复训练 (暂未实现)

当前版本不支持从检查点恢复训练，这是未来的改进方向。

## 高级用法

### 消融实验

测试不同融合组件的影响：

```bash
# 仅深度指导采样
python t5.py --scene_name garden --sdf_omega_g 0 --geom_normal_w 0

# 仅 SDF 引导 densify/prune
python t5.py --scene_name garden --dg_k 0 --geom_depth_w 0 --geom_normal_w 0

# 仅几何监督
python t5.py --scene_name garden --dg_k 0 --sdf_omega_g 0
```

### 性能优化

对于大场景，优化训练速度：

```bash
python t5.py \
    --scene_name large_scene \
    --mesh_every 1000 \
    --log_every 500 \
    --save_every 10000 \
    --validate_every 0
```

### 高质量训练

追求最佳质量：

```bash
python t5.py \
    --scene_name garden \
    --joint_iterations 50000 \
    --dg_k 2.5 \
    --sdf_sigma 0.3 \
    --geom_depth_w 2.0 \
    --geom_normal_w 0.5 \
    --mesh_every 250
```

## 与其他脚本对比

| 特性 | t3.py | t4.py | t5.py |
|------|-------|-------|-------|
| 基础联合训练 | ✅ | ✅ | ✅ |
| 深度指导采样 | ✅ | ✅ | ✅ |
| SDF 引导 densify/prune | ✅ | ✅ | ✅ |
| 几何监督 | ✅ | ✅ | ✅ |
| 详细统计输出 | ⚠️ | ✅ | ✅ |
| 检查点保存 | ❌ | ⚠️ | ✅ |
| 验证步骤 | ❌ | ❌ | ✅ |
| 优雅中断 | ❌ | ❌ | ✅ |
| 时间统计 | ⚠️ | ⚠️ | ✅ |
| 参数分组 | ❌ | ⚠️ | ✅ |
| 随机种子 | ❌ | ❌ | ✅ |

**推荐**: 使用 `t5.py` 进行完整训练

## 更多信息

- 架构设计: [`fusion/README.md`](fusion/README.md)
- 实现细节: [`fusion_implementation_guide_cn.md`](fusion_implementation_guide_cn.md) (如果存在)
- t3/t4 使用说明: [`t4_usage_cn.md`](t4_usage_cn.md)

## 示例工作流

### 1. 数据准备
```bash
# 确保数据集结构正确
ls data/garden/images/
ls data/garden/sparse/0/
```

### 2. 快速测试 (200 步)
```bash
python t5.py --scene_name garden --joint_iterations 200
```

### 3. 完整训练
```bash
python t5.py --scene_name garden --joint_iterations 30000
```

### 4. 检查结果
```bash
# GS 模型
ls work/gaussian_models/garden/

# NeuS 模型
ls work/fusion_workspace/garden/neus_exp/
```

## 常见问题 (FAQ)

**Q: 训练需要多长时间？**
A: 在单张 RTX 3090 上，30000 步约需 4-6 小时（取决于场景大小）。

**Q: 可以同时训练多个场景吗？**
A: 可以，只需在不同终端运行不同的 `--scene_name`。

**Q: 如何可视化结果？**
A: 使用 3DGS 官方查看器查看保存的 PLY 文件，或使用 MeshLab 查看 NeuS 导出的 mesh。

**Q: 训练失败了怎么办？**
A: 检查日志输出，常见问题包括数据路径错误、CUDA 内存不足等。

## 致谢

本脚本基于以下项目：
- [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting)
- [NeuS](https://github.com/Totoro97/NeuS)
- Fusion Wrapper 架构

---

**版本**: t5.py v1.0  
**最后更新**: 2025-11-22
