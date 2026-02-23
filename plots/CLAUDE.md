# plots/ — Training Loss Visualization

Qwen3-235B LoRA 微调实验的 loss 曲线对比图工具，用于学术论文。

## 文件说明

| 文件 | 用途 |
|------|------|
| `plot_loss.py` | 绘制 loss 曲线对比图（step-loss 和 time-loss 两张独立图） |
| `compare_stats.py` | 统计 ZeRO-LoRA vs KT-FT 在每 25 步的 time/loss 差异，以及同 loss 下最大时间差 |

## 数据源

三条训练曲线的 `trainer_log.jsonl`（LlamaFactory 格式），每行包含 `current_steps`, `loss`, `elapsed_time` 等字段：

- **ZeRO-LoRA**: `/mnt/data/lpl/kernel_new_test_adapter/zero_lora_qwen3_235b_livebench_math/trainer_log.jsonl`
- **KT-FT**: `/mnt/data/hxx/saves/qwen3-235-lora-math/trainer_log.jsonl`
- **KT-FT+KLoRA**: `/mnt/data/hxx/saves/qwen3-235-lora-math-le/trainer_log.jsonl`

如需更换数据源，直接修改脚本顶部的 `sources` 字典。

## plot_loss.py 使用方法

```bash
python plots/plot_loss.py
```

输出到 `plots/` 目录：
- `loss_vs_step.pdf` / `.png` — Training Loss vs. Step
- `loss_vs_time.pdf` / `.png` — Training Loss vs. Time（x 轴按 KT-FT 最大时间截断）

### 关键可调参数

| 参数 | 位置 | 说明 |
|------|------|------|
| `TARGET_LOSS` | 第 109 行 | 控制 time-loss 图中箭头标注的 y 坐标（loss 值），修改此值后箭头位置、speedup 数值、samples 数量全部自动重算 |
| `smooth(y, window=15)` | 第 58 行 | EMA 平滑窗口大小，越大越平滑 |
| `sources` | 第 27-31 行 | 数据源路径和图例名称 |
| `COLORS` | 第 34 行 | 三条曲线配色，当前为 Nature 风格（浅橙 / 蓝 / 绿） |

### 箭头标注逻辑

在 time-loss 图中，脚本会：
1. 在 ZeRO-LoRA 和 KT-FT 的 smoothed loss 曲线上找到 `TARGET_LOSS` 对应的两个点（相同 y，不同 x）
2. 从 ZeRO 点画箭头指向 KT-FT 点（深红色 #B71C1C）
3. 箭头上方标注 `"{samples} samples, same LoRA, {speedup}x faster"`
4. samples = 4 × avg_step，四舍五入到最近的 100

## compare_stats.py 使用方法

```bash
python plots/compare_stats.py
```

纯文本输出，两部分：
1. **每 25 步对比**：ZeRO vs KT-FT 的 time、loss、speedup
2. **同 loss 时间差**：在 smoothed 曲线的公共 loss 范围内，找到时间差最大的点

## 依赖

```
numpy, matplotlib, scipy
```

## 学术风格要点

- 字体：Times New Roman / DejaVu Serif + STIX math
- 配色：色盲友好（Nature 风格），箭头使用独立的学术深红 #B71C1C
- 曲线：淡色原始 loss + EMA smoothed 实线
- 图框：隐藏右侧和顶部边框，虚线网格
- 输出：300 DPI，tight bbox
