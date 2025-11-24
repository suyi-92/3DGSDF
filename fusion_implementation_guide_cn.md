# GS-NeuS 融合实现指南

本文档详细介绍了 `fusion/wrapper.py` 中实现的三大核心融合特性。

---

## 1. GS → SDF：深度指导采样

### 概述
利用 Gaussian Splatting 渲染的深度图引导 NeuS 的光线采样，在预测表面附近收窄 near/far 采样窗口。

### 实现位置
- **发布者**: [`GaussianSplattingAdapter._publish_render_outputs()`](file:///e:/11/NN_Project/3DGSDF/fusion/wrapper.py#L739-L763)
- **消费者**: [`NeuSAdapter._override_near_far_with_depth()`](file:///e:/11/NN_Project/3DGSDF/fusion/wrapper.py#L1068-L1155)

### 工作原理

1. **深度发布**（GS 端）
   ```python
   # 每次 GS 渲染后，将深度/法线发布到总线
   payload = {
       "camera_id": camera_id,
       "depth": depth_tensor,      # GS 渲染的深度图
       "normal": normal_tensor,    # 可选的法线图
       "iteration": iteration,
   }
   bus.publish("gaussian.render_outputs", payload)
   ```

2. **深度缓存**（Wrapper）
   - `FusionWrapper._on_gaussian_render()` 订阅总线消息
   - 按 `camera_id` 索引缓存深度/法线图
   - 实现基于年龄的缓存失效机制（`max_age` 参数）

3. **自适应采样**（NeuS 端）
   - 在 NeuS 训练中，对像素坐标 `(u, v)` 采样光线
   - 查询缓存的 GS 深度：`D = depth[u, v]`
   - 计算 3D 中心：`center = o + D * v`
   - 在中心点评估 SDF：`s = sdf_network(center)`
   - 自适应窗口：`near = D - k*|s|`，`far = D + k*|s|`

### 超参数配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `k` | 3.0 | SDF 不确定性的窗口乘数 |
| `min_near` | 0.01 | 最小 near 平面距离 |
| `max_far` | 100.0 | 最大 far 平面距离 |
| `max_age` | 50 | 缓存过期步数 |

配置示例：
```python
fusion_cfg = {
    "depth_guidance": {
        "k": 3.0,
        "min_near": 0.01,
        "max_far": 100.0,
        "max_age": 50,
    }
}
```

### 回退策略
- 缓存未命中 → 使用默认的球体 near/far
- SDF 评估失败 → 使用默认 near/far
- 缓存命中率会被追踪并在统计信息中报告

---

## 2. SDF → GS：几何引导的 Densify/Prune

### 概述
使用 NeuS 的 SDF 预测来引导 Gaussian 的增密和修剪，优先在隐式曲面附近添加高斯，移除远离曲面的高斯。

### 实现位置
- **主要逻辑**: [`FusionWrapper._sdf_guided_gaussian_update()`](file:///e:/11/NN_Project/3DGSDF/fusion/wrapper.py#L1457-L1562)
- **参考**: 遵循官方 `train.py` 第 164-174 行

### 工作原理

1. **在高斯中心评估 SDF**
   ```python
   xyz = gaussians.get_xyz  # N × 3
   sdf = neus.evaluate_sdf(xyz)  # N × 1
   ```

2. **计算表面接近度权重**
   ```python
   μ(s) = exp(-s² / (2σ²))
   ```
   - μ ≈ 1 在表面附近（|s| ≈ 0）
   - μ ≈ 0 远离表面

3. **增强的 Densify 梯度**
   ```python
   grad_accum = gaussians.xyz_gradient_accum  # 来自 train.py 167 行
   ε_g = grad_accum / denom + ω_g * μ(s)
   ```
   - 结合图像空间梯度与 SDF 接近度
   - 更高的 ε_g → 更可能增密

4. **基于不透明度的 Prune**
   ```python
   ε_p = opacity - ω_p * (1 - μ(s))
   prune_mask = ε_p < τ_p
   ```
   - 远离表面的点会被惩罚
   - 低 ε_p → 被修剪

5. **调用官方 Densify/Prune**
   ```python
   gaussians.densify_and_prune(
       max_grad=τ_g,
       min_opacity=τ_p,
       extent=scene.cameras_extent,
       max_screen_size=size_threshold
   )
   ```

### 超参数配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `σ` (sigma) | 0.5 | 控制 μ(s) 衰减率 |
| `ω_g` (omega_g) | 1.0 | SDF 在增密中的权重 |
| `ω_p` (omega_p) | 0.5 | SDF 在修剪中的权重 |
| `τ_g` (tau_g) | 0.0002 | 增密的梯度阈值 |
| `τ_p` (tau_p) | 0.005 | 修剪的不透明度阈值 |

配置示例：
```python
fusion_cfg = {
    "sdf_guidance": {
        "sigma": 0.5,
        "omega_g": 1.0,
        "omega_p": 0.5,
        "tau_g": 0.0002,  # 与 densify_grad_threshold 对齐
        "tau_p": 0.005,
    }
}
```

### 与官方方法的集成
- **梯度累积**: 使用官方 `add_densification_stats()` 的 `gaussians.xyz_gradient_accum` 和 `gaussians.denom`
- **Densify/Prune**: 直接调用 `gaussians.densify_and_prune()`，与 train.py 兼容
- **时机**: 仅在 `[densify_from_iter, densify_until_iter]` 窗口内运行

---

## 3. 互相几何监督

### 概述
向 NeuS 训练添加深度和法线一致性损失，使用 GS 渲染的深度/法线作为监督。

### 实现位置
- **深度/法线采样**: [`NeuSAdapter._sample_gs_depth_normal()`](file:///e:/11/NN_Project/3DGSDF/fusion/wrapper.py#L1124-L1160)
- **损失计算**: [`NeuSAdapter.train_step()`](file:///e:/11/NN_Project/3DGSDF/fusion/wrapper.py#L1162-L1273)

### 工作原理

1. **计算 NeuS 深度和法线**
   ```python
   # 深度：mid_z_vals 的加权和
   μ_z = (render_out["weights"] * render_out["mid_z_vals"]).sum(dim=-1)
   
   # 法线：加权梯度（归一化）
   weighted_grad = (gradients * weights[..., None]).sum(dim=1)
   weighted_normal = F.normalize(weighted_grad, dim=-1)
   ```

2. **采样 GS 深度/法线**
   ```python
   depth_gt, normal_gt = _sample_gs_depth_normal(idx, pixels_x, pixels_y)
   ```

3. **深度一致性损失**
   ```python
   L_depth = w_depth * |μ_z - depth_gt|
   ```

4. **法线一致性损失**
   ```python
   L_normal = w_normal * (1 - cos(n_gs, n_neus))
            = w_normal * (1 - dot(n_gs, n_neus))
   ```

5. **总损失**
   ```python
   loss = L_color + λ_eikonal * L_eikonal + λ_mask * L_mask + L_depth + L_normal
   ```

### 超参数配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `depth_w` | 1.0 | 深度一致性损失权重 |
| `normal_w` | 0.1 | 法线一致性损失权重 |
| `eps` | 1e-6 | 数值稳定性 epsilon |

配置示例：
```python
fusion_cfg = {
    "geom_loss": {
        "depth_w": 1.0,
        "normal_w": 0.1,
        "eps": 1e-6,
    }
}
```

### 损失调度
当前使用固定权重。如需自适应调度，可以：
- 在预热期间逐步增加 `depth_w` 和 `normal_w`
- 收敛后降低权重
- 实现类似 NeuS 的 `cos_anneal_ratio` 余弦退火

---

## 使用示例

```python
from pathlib import Path
from fusion.wrapper import DataService, FusionWrapper, SceneSpec

# 1. 定义场景规格
spec = SceneSpec(
    scene_name="garden",
    dataset_root="dataset/garden",
    gaussian_source_path="work/gs_garden",
    gaussian_model_path="work/gs_models/garden",
    neus_conf_path="NeuS/confs/wmask.conf",
    neus_case="garden",
    shared_workspace="work/fusion",
)

# 2. 配置融合参数
fusion_cfg = {
    "depth_guidance": {
        "k": 3.0,
        "min_near": 0.01,
        "max_far": 100.0,
        "max_age": 50,
    },
    "sdf_guidance": {
        "sigma": 0.5,
        "omega_g": 1.0,
        "omega_p": 0.5,
        "tau_g": 0.0002,
        "tau_p": 0.005,
    },
    "geom_loss": {
        "depth_w": 1.0,
        "normal_w": 0.1,
    },
}

# 3. 创建 wrapper
data_service = DataService(Path(spec.dataset_root))
wrapper = FusionWrapper(
    spec,
    gaussian_repo=Path("gaussian_splatting"),
    neus_repo=Path("NeuS"),
    data_service=data_service,
    gaussian_cfg={},
    neus_cfg={},
    fusion_cfg=fusion_cfg,
)

# 4. 引导两个系统
wrapper.bootstrap()

# 5. 联合训练循环
for i in range(30000):
    result = wrapper.joint_step(
        mesh_every=500,    # 每 500 迭代同步 NeuS 网格到 GS
        log_every=100,     # 每 100 迭代打印统计
    )
    
    # 可选：自定义监控
    if i % 1000 == 0:
        stats = wrapper.get_statistics()
        print(f"深度命中率: {stats['depth_hit_rate']:.2%}")
        print(f"高斯数量: {stats['num_gaussians']}")
```

---

## 监控和调试

### 统计追踪

调用 `wrapper.get_statistics()` 获取：

```python
{
    "depth_cache_size": 25,           # 缓存的深度图数量
    "depth_hit_rate": 0.87,           # 缓存命中率（87%）
    "num_gaussians": 156342,          # 当前高斯数量
    "densify_count": 5234,            # 总共添加的高斯数
    "prune_count": 1823,              # 总共移除的高斯数
    "neus_iteration": 10000,          # NeuS 迭代计数器
}
```

### 自动日志

在 `joint_step()` 中设置 `log_every=N` 以每 N 次迭代打印统计：

```
=== Fusion Statistics (iter 100) ===
Depth Cache: 25 entries
Depth Hit Rate: 87.32%
Num Gaussians: 156342
Densify/Prune: +5234 / -1823
NeuS Iter: 100
==================================================
```

### 故障排查

**深度命中率过低（< 50%）**
- 增加 `max_age` 以保持缓存条目更久
- 检查 GS 和 NeuS 数据集之间的 camera ID 是否匹配
- 验证深度图是否正在发布（检查总线订阅者）

**没有 Densify/Prune**
- 确保在增密窗口内（`densify_from_iter` 到 `densify_until_iter`）
- 验证 `xyz_gradient_accum` 是否正在填充（检查 GS 训练）
- 尝试增加 `omega_g` 以放大 SDF 影响

**NeuS 训练不稳定**
- 如果几何损失占主导，减少 `depth_w` 和 `normal_w`
- 增加 `k` 以扩大采样窗口
- 检查深度/法线图是否正确归一化

---

## API 参考

### 关键方法

#### `FusionWrapper.joint_step(mesh_every=100, log_every=100, callback=None)`
执行一次协调的训练迭代。

**返回**: 包含键 `neus`、`gaussian`、`fusion_step`、`statistics` 的字典

#### `FusionWrapper.get_statistics()`
返回当前融合统计信息。

**返回**: 包含缓存统计、命中率、计数等的字典

#### `GaussianSplattingAdapter._publish_render_outputs(cam, render_pkg)`
渲染后将深度/法线发布到交换总线。

#### `NeuSAdapter._override_near_far_with_depth(...)`
使用缓存的 GS 深度应用自适应采样。

**返回**: 元组 `(near_new, far_new)`

#### `NeuSAdapter.evaluate_sdf(points)`
在任意 3D 点评估 NeuS SDF（无梯度）。

**返回**: SDF 值的张量

---

## 参考资料

- **官方 3DGS**: [`gaussian_splatting/train.py`](file:///e:/11/NN_Project/3DGSDF/gaussian_splatting/train.py)
  - Densification: 第 164-174 行
  - 梯度累积: 第 167 行
  
- **官方 NeuS**: [`NeuS/exp_runner.py`](file:///e:/11/NN_Project/3DGSDF/NeuS/exp_runner.py)
  - 训练循环: 第 109-216 行
  - 损失计算: 第 154-175 行

- **融合 README**: [`fusion/README.md`](file:///e:/11/NN_Project/3DGSDF/fusion/README.md)
  - 详细路线图和设计理念
