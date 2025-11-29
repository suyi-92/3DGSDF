# t6.py 使用说明（预热 + 联合优化）

`t6.py` 提供带预热能力的 GS-NeuS 联合训练入口。默认与现有脚本一致：不传预热参数即从头开始联合优化。常用示例如下：

1. **仅预热 3DGS 并保存元信息**

```bash
python t6.py --mode prewarm --scene_name garden \
  --fusion.prewarm_3dgs --fusion.no_prewarm_neus \
  --fusion.prewarm_3dgs_iters 30000 --log_every 200
```

2. **仅预热 NeuS 并保存元信息**

```bash
python t6.py --mode prewarm --scene_name garden \
  --fusion.no_prewarm_3dgs --fusion.prewarm_neus \
  --fusion.prewarm_neus_iters 200000 --log_every 500
```

3. **从已有预热 checkpoint 直接进入联合优化**

```bash
python t6.py --mode fusion --scene_name garden \
  --fusion.prewarm_mode load --joint_iterations 30000
```

4. **若未找到预热，则自动先预热再联合**

```bash
python t6.py --mode fusion --scene_name garden \
  --fusion.prewarm_mode run_then_joint \
  --fusion.prewarm_3dgs_iters 30000 --fusion.prewarm_neus_iters 200000 \
  --joint_iterations 30000
```

预热元信息默认保存在 `output/fusion_prewarm/<scene_name>/[3dgs|neus]/meta.json` 中，可通过 `--fusion.prewarm_output_root` 自定义。
