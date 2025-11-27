面向 3D Gaussian Splatting 与 NeuS 的耦合训练，`fusion/` 以“适配器 + 总线”实现非侵入式集成，既复用各自代码仓，又能做双向引导（GS 深度 → NeuS 采样；NeuS SDF → GS 稠密/剪枝）。

## 目录与角色
```
fusion/
├─ __init__.py              # 导出常用接口
├─ common.py                # 配置/状态数据类；APIRegistry & ExchangeBus & MutableHandle
├─ data_service.py          # 统一数据集入口，生成 3DGS/NeuS 兼容目录
├─ gaussian_adapter.py      # 3DGS 训练封装，渲染并发布 depth/normal
├─ neus_adapter.py          # NeuS 训练封装，深度引导采样 & SDF 查询
├─ fusion_wrapper.py        # 顶层调度（bootstrap、joint_step、统计）
├─ samplers.py              # 深度采样策略（Uniform & GS-guided）
└─ adapter_bus_minimal_example.py  # 最小总线示例
```

## 核心模块梳理
- **common.py**
  - `SceneSpec`、`GaussianIterationState`、`NeuSIterationState`：一些配置和训练快照。
  - `APIRegistry`：可查询/调用的 API 注册表；`ExchangeBus`：发布订阅总线。
  - `MutableHandle`：上下文方式临时修改内部状态（如优化器）。
- **data_service.py**：对外统一数据接口，负责
  - 数据处理（列图像、poses_bounds、sparse bundle）。
  - `materialize_gaussian_scene` / `materialize_neus_scene`：生成对应目录、PNG/mask 转换、相机参数转换。
  - 光线采样中间键
- **gaussian_adapter.py**
  - 解析 3DGS 配置，初始化 `GaussianModel` 与优化器。
  - `train_step`：采样相机 → 渲染 → L1/SSIM 反传 （→ 通过总线发布 depth/normal）。
  - `import_surface`/`export_surface`：mesh/ply 互导；属性访问器暴露 xyz/feature/scale/rotation/opacity。
- **neus_adapter.py**
  - 解析 NeuS 配置并初始化 Runner。
  - 深度引导采样：从 `depth_cache` 取 GS 深度，生成 `[D - k|s|, D + k|s|]` 自适应采样窗，失效时回退 uniform。
  - `evaluate_sdf`、`export_surface`、`inject_supervision` 等 API。
- **fusion_wrapper.py**
  - 初始化总线/注册表/深度缓存，加载融合配置（depth_guidance、sdf_guidance、geom_loss）。
  - 订阅 `gaussian.render_outputs` 更新 `depth_cache`（含 alternative keys）。
  - `joint_step`：NeuS 训练 → 可选 mesh 同步到 GS → GS 训练并发布深度 → SDF 引导 densify/prune → 统计。
- **samplers.py**：`UniformSampler`（分层采样）与 `GSGuidedSampler`（GS 渲染 + 一次 SDF 查询的窄窗采样，低 opacity 时回退）。

## 运行
1) **Bootstrap**：传入 `SceneSpec`、外部仓根路径（`gaussian_splatting/`、`NeuS/`）、融合配置，`FusionWrapper.bootstrap()` 会：
   - 让 DataService 对同一原始数据生成 GS/NeuS 所需目录。
   - 创建总线/注册表、适配器，建立深度缓存订阅。
2) **联合训练循环 `joint_step`**：
   - NeuS：用深度缓存做采样窗覆盖，前向/反传，发布训练状态。
   - Mesh 同步：按周期 `neus_to_gaussian()` 将 NeuS 网格导入 GS（可配频率）。
   - GS：渲染当前相机，发布 depth/normal；常规 L1+SSIM 训练。
   - SDF 引导 densify/prune：调用 NeuS SDF，按 σ/ω/τ 策略增强/剪枝高斯。
   - 统计：深度命中率、当前高斯数、densify/prune 计数、NeuS iter 等。

## 通信与配置
- **总线主题**：`gaussian.render_outputs`（含 camera_id/alternative_keys、depth、normal、iteration）；`gaussian.train_step`；`neus.train_step`。
- **深度缓存键匹配**：索引、文件 stem、完整文件名、数字 stem 都可命中，`max_age` 控制过期。
- **可调参数**（融合侧）：
  - `depth_guidance`: `k`、`min_near`、`max_far`、`max_age`。
  - `sdf_guidance`: `sigma`、`omega_g`、`omega_p`、`tau_g`、`tau_p`。
  - `geom_loss`: `depth_w`、`normal_w`。
  - 采样策略：`GSGuidedSampler` 的 `k_scale`、`opacity_threshold`、`min_spread` 等。

## 要点
- 非侵入式集成：不改 3DGS/NeuS 源码，靠适配器和总线。
- 双向信息流：GS depth → NeuS 采样；NeuS SDF → GS densify/prune；定期 mesh 同步。
- 数据一致性：DataService 保证两端视图/相机定义一致。
- 可扩展性：APIRegistry/ExchangeBus 方便挂新模块或可视化；sampler 模块可替换实验策略。

