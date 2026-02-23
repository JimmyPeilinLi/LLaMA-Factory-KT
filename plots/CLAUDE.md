# plots/ — Training Loss Visualization

Qwen3-235B LoRA 微调实验的 loss 曲线对比图工具，用于学术论文。

## 文件说明

| 文件 | 用途 |
|------|------|
| `plot_loss.py` | 主绘图脚本：step-loss 和 time-loss 两张独立图 |
| `compare_stats.py` | 统计 ZeRO-LoRA vs KT-FT 在每 25 步的 time/loss 差异，以及同 loss 下最大时间差 |
| `plot_loss_tmp.py` | **【临时方案，除非用户非常确认使用，不然后续迭代不再用】** 混合数据源绘图 |

## 数据源

### plot_loss.py（主版本）

三条曲线全部来自 `trainer_log.jsonl`（LlamaFactory 格式），每行含 `current_steps`, `loss`, `elapsed_time`：

- **ZeRO-LoRA (offload paradigm)**: `/mnt/data/lpl/kernel_new_test_adapter/zero_lora_qwen3_235b_livebench_math/trainer_log.jsonl`
- **KT-FT**: `/mnt/data/hxx/saves/qwen3-235-lora-math/trainer_log.jsonl`
- **KT-FT+KLoRA**: `/mnt/data/hxx/saves/qwen3-235-lora-math-le/trainer_log.jsonl`

### plot_loss_tmp.py（临时版本）

> **【临时方案，除非用户非常确认使用，不然后续迭代不再用】**
>
> 背景：两版实验设置未对齐，ZeRO 数据来不及重跑。此脚本用混合数据源 + 比例映射生成临时图。

数据来源：
- **ZeRO-LoRA + KT-FT**: 使用 v1 的 trainer_log.jsonl（同 plot_loss.py）
- **KT-FT+KLoRA**: 使用 v2 的 `kt-ft-klora.json`（`[[timestamp, step, loss], ...]` 格式），通过与 `kt-ft.json` 的逐步比例映射到 v1 坐标系

映射逻辑：
1. Step 1-117（v2 数据覆盖范围）：`klora_value[step] = v1_ktft[step] × (v2_klora[step] / v2_ktft[step])`，time 和 loss 均逐步映射
2. Step 118-249（外推）：使用最后 20 步的均值比例（time_ratio≈1.127, smoothed_loss_ratio≈0.552）

## plot_loss.py 使用方法

```bash
python plots/plot_loss.py
```

输出到 `plots/` 目录：
- `loss_vs_step.pdf` / `.png` — Training Loss vs. Step
- `loss_vs_time.pdf` / `.png` — Training Loss vs. Time（x 轴按 KT-FT 最大时间截断）

### 关键可调参数

| 参数 | 说明 |
|------|------|
| `TARGET_LOSS` | 控制 time-loss 图中标注点的 y 坐标（loss 值），修改后点位置、speedup、samples 全部自动重算 |
| `smooth(y, window=15)` | EMA 平滑窗口大小，越大越平滑 |
| `sources` | 数据源路径和图例名称 |
| `COLORS` | 三条曲线配色，当前为 Nature 风格（浅橙 / 蓝 / 绿） |

### 标注点逻辑

在 time-loss 图中：
1. 在 ZeRO-LoRA 和 KT-FT 的 smoothed 曲线上找到 `TARGET_LOSS` 对应的两个点（同 y 不同 x）
2. 两个深红色圆点标记（#B71C1C）
3. 详细信息（speedup、samples、step）输出到终端而非图上

## plot_loss_tmp.py 使用方法

> **【临时方案，除非用户非常确认使用，不然后续迭代不再用】**

```bash
python plots/plot_loss_tmp.py          # 只有 smooth 线
python plots/plot_loss_tmp.py --raw    # 加上淡色原始 loss 背景
```

输出：`loss_vs_time_tmp.pdf` / `.png`

### 标注点

三个深红色圆点（#B71C1C）：
1. ZeRO-LoRA 在 `TARGET_LOSS` 对应的时间点
2. KT-FT 在 `TARGET_LOSS` 对应的时间点
3. KLoRA 在与 KT-FT 相同时间（x 坐标）处的 loss 点

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
- 配色：色盲友好（Nature 风格），标注点使用独立的学术深红 #B71C1C
- 曲线：EMA smoothed 实线（`--raw` 可叠加淡色原始 loss）
- 图框：隐藏右侧和顶部边框，虚线网格
- 输出：300 DPI，tight bbox
- ZeRO-LoRA 图注统一带 `(offload paradigm)` 后缀
