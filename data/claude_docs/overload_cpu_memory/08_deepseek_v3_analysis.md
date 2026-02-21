# 08 DeepSeek-V3 OOM 分析

## 概述

DeepSeek-V3 (671B) 在 2TB 内存、4 GPU 的环境下使用 DeepSpeed ZeRO-3 offload + LoRA 进行训练时，
在模型**创建阶段**（`deepspeed.zero.Init()`）就发生 OOM，根本没有执行到 shard-by-shard 权重加载阶段。

这是一个与 Qwen3 系列完全不同的问题：
- **Qwen3 的问题**：shard-by-shard 权重加载阶段的内存峰值（Phase 1），已通过 v4 方案解决
- **DeepSeek-V3 的问题**：模型参数 materialization 阶段（Phase 0）的内存不足，v4 方案无法覆盖

## 关键发现

### 1. OOM 发生在 Phase 0（模型创建），不是 Phase 1（权重加载）

从 `zero_dpsk3.log` 可以看到，只有 2 条来自 `loader.py` 的 DIAG 日志：
```
[DIAG] Before patch_zero3_model_loading: RSS=...
[DIAG] Before from_pretrained: RSS=...
```
**没有** `After from_pretrained` 日志，说明 `from_pretrained()` 从未完成。

进一步确认，DeepSpeed zero.Init 的参数 materialization 进度停止在 65%：
```
[DeepSpeed] partition 632 / 967 parameters
model.layers.40.mlp.experts.down_proj  (最后一个被 materialize 的参数)
```

### 2. 内存增长轨迹

从 `mem_log_new_dpsk3.csv` 系统级内存监控数据：

| 时间 | mem_used (GB) | mem_avail (GB) | 阶段 |
|------|-------------|---------------|------|
| 15:49:24 | 576 | 1439 | 基线（训练启动前） |
| 15:50:00 | 750 | 1265 | zero.Init 开始 |
| 15:53:00 | 1198 | 817 | ~40% 参数完成 |
| 15:55:00 | 1461 | 588 | ~55% 参数完成 |
| 15:57:00 | 1776 | 281 | ~63% 参数完成 |
| 15:57:47 | 1856 | 208 | 65% 参数完成，OOM killed |
| 15:58:07 | 1447 | 617 | 进程被杀，内存回收 |

**内存增长速率**：约 2.46 GB/s 平均，~80-100 GB/30秒

### 3. 为什么 Phase 0 占用这么多内存？

#### 3.1 DeepSeek-V3 模型参数规模

```
模型配置：
- num_hidden_layers: 61
- n_routed_experts: 256
- hidden_size: 7168
- moe_intermediate_size: 2048
- first_k_dense_replace: 3 (前3层是dense，后58层是MoE)
- kv_lora_rank: 512, q_lora_rank: 1536
```

每个 MoE 层在 BF16 下的参数大小：
```
experts.gate_up_proj: [256, 4096, 7168] × 2 bytes = 15.0 GB
experts.down_proj:    [256, 7168, 2048] × 2 bytes = 7.5 GB
shared_experts (3个):                              ≈ 0.75 GB
Attention (MLA):                                   ≈ 0.37 GB
其他:                                              ≈ 0.05 GB
每个 MoE 层合计:                                   ≈ 23.6 GB
```

全模型 BF16 参数量：
```
58 × 23.6 GB (MoE) + 3 × ~0.5 GB (dense) + ~5 GB (embeddings/lm_head)
≈ 1,376 GB (BF16 总量)
```

#### 3.2 ZeRO-3 分区后的理论内存

ZeRO-3 将参数分区到 4 个 rank：
```
理论分区后：1,376 GB / 4 = 344 GB/rank
4 个 rank 合计：1,376 GB
```

但实际观察到的是 **~1,812 GB**（系统级），约为理论值的 **1.3x**。

#### 3.3 额外内存开销来源

**关键问题：glibc malloc 内存保留**

在 `deepspeed.zero.Init()` 中，每个参数的 materialization 过程：
1. 创建全尺寸 BF16 tensor（例如 `experts.gate_up_proj`: 15.0 GB）
2. `torch.empty(...)` 分配内存
3. DeepSpeed 立刻将其分区（partition）到各 rank
4. 每个 rank 只保留 1/4 = 3.75 GB
5. 剩余的 3/4 = 11.25 GB 被 `free()`

**但是**：glibc 的 malloc 实现**不会立即将 `free()` 的内存归还给 OS**。
对于大块分配（通过 mmap），理论上应该立即归还，但实际行为受以下因素影响：
- PyTorch 的内存分配器层（caching allocator for CPU）
- glibc 的 `MALLOC_TRIM_THRESHOLD_` 默认值
- 内存碎片

这导致了 `top`/`htop` 看到的 RSS 远高于实际使用量。

#### 3.4 每个参数的内存峰值

在 materialize 单个参数时（以 `experts.gate_up_proj` 为例）：
```
步骤 1: 分配全尺寸 tensor = 15.0 GB
步骤 2: 分区，保留 1/4 = 3.75 GB
步骤 3: free 3/4 = 11.25 GB（但可能不归还 OS）
```

如果 malloc 不归还内存，累积到 layer 40 时：
```
累积"已分区保留" = 40层 × 23.6 GB / 4 rank ≈ 236 GB/rank
累积"已free但未归还" = 取决于 allocator 行为
实际 RSS ≈ 分区保留 + 未归还的 free 内存
```

### 4. Checkpoint 结构分析

DeepSeek-V3 checkpoint（FP8 量化）：
```
总 shard 数：163 个
总 key 数：91,991
  - expert keys：90,978 (99%)
  - non-expert keys：1,013 (1%)
  - weight_scale_inv keys：45,808 (FP8 量化 scale)
```

**重要**：checkpoint 是 FP8 格式，但 `deepspeed.zero.Init()` 在创建模型时使用 BF16，
这意味着：
- Phase 0（模型创建）：每个参数以 BF16 创建 → 1,376 GB
- Phase 1（权重加载）：从 FP8 checkpoint 加载 → 约 688 GB 磁盘大小
- 加载时 FP8→BF16 转换会有临时内存开销

### 5. 与 DeepSeek-V2-Lite (14B) 的对比

DeepSeek-V2-Lite 成功加载：
```
shard-by-shard 正常执行（3/3 boundary layers completed, 0 errors）
峰值 RSS ≈ 34 GB/process
htop 总内存 ≈ 80 GB（4 进程 × ~20 GB 稳态）
```

V2-Lite 之所以成功，是因为：
1. 模型小（14B），Phase 0 的参数 materialization 内存开销可控
2. MoE 规模小（64 experts vs V3 的 256 experts）
3. 分区后单 rank 只需约 7 GB

## 解决方案方向

### 方案 A：减少 malloc 内存保留（最低侵入性）

1. **设置 `MALLOC_TRIM_THRESHOLD_=0`**
   ```bash
   export MALLOC_TRIM_THRESHOLD_=0
   ```
   强制 glibc 在 free 时立即 trim heap。

2. **在 zero.Init 过程中定期调用 `malloc_trim`**
   ```python
   import ctypes
   libc = ctypes.CDLL("libc.so.6")
   libc.malloc_trim(0)  # 强制归还 free 的内存给 OS
   ```

3. **使用 jemalloc 替代 glibc malloc**
   ```bash
   LD_PRELOAD=/path/to/libjemalloc.so python ...
   ```
   jemalloc 的内存归还策略更激进。

### 方案 B：patch deepspeed.zero.Init 的参数创建过程

在每个参数创建和分区后，强制释放内存：
```python
# 在 deepspeed/runtime/zero/partition_parameters.py 中
# _post_init_method() 里，每次参数分区后：
import gc
gc.collect()
ctypes.CDLL("libc.so.6").malloc_trim(0)
```

### 方案 C：分层初始化模型（中等侵入性）

不使用 `deepspeed.zero.Init()` 全局 context，而是逐层创建模型：
1. 在 meta device 上创建整个模型（几乎不占内存）
2. 逐层将参数从 meta 转换为实际 tensor
3. 每层转换后立即进行 ZeRO-3 分区
4. 释放临时内存后再处理下一层

### 方案 D：直接在 meta device 创建 + 从 checkpoint 加载（最高效但最复杂）

1. 在 meta device 创建整个模型
2. 跳过 zero.Init() 的参数 materialization
3. 直接通过 shard-by-shard 加载从 checkpoint 填充参数
4. 需要处理 FP8→BF16 转换

## 推荐的下一步

1. **先尝试方案 A**：设置 `MALLOC_TRIM_THRESHOLD_=0` 或 `LD_PRELOAD=jemalloc`，
   这是最简单的，可能已经足够减少 ~30% 的内存开销

2. **如果方案 A 不够**：尝试方案 B，在 DeepSpeed 的参数创建循环中插入 `malloc_trim`

3. **需要更多调试信息**：在 `deepspeed.zero.Init` 的 `_post_init_method` 中添加内存日志，
   精确追踪每个参数创建和分区后的 RSS 变化

## 附录：日志关键片段

### OOM 前最后的进度
```
[2025-02-21 15:57:42] [DeepSpeed] partition 632 / 967 parameters
[2025-02-21 15:57:42] model.layers.40.mlp.experts.down_proj
```

### 进程被杀
```
exitcode: -15  (SIGTERM from OOM killer)
```

### 内存恢复
```
15:57:52  mem_used=1,856 GB  mem_avail=208 GB  ← OOM
15:58:07  mem_used=1,447 GB  mem_avail=617 GB  ← 进程被杀，回收中
15:58:22  mem_used=741 GB    mem_avail=1,323 GB ← 大部分已回收
```
