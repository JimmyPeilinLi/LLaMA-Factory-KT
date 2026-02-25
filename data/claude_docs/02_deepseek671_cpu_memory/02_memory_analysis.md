# ZeRO-3 CPU 内存超额占用深度分析

## 核心问题

> "按理来说模型参数应该只需要 671G 和 470G (BF16) 啊？"

实际远不止这些。本文档分析 DeepSeek-V3 (671B) 和 Qwen3-235B 在 ZeRO-3 offload 下，
CPU 内存为什么远超 "参数量 × 2 bytes" 的理论值。

---

## 1. 首先澄清：理论模型大小

### DeepSeek-V3

```
参数量: 671B (实际约 688B，含 embedding/lm_head)
FP8 checkpoint 大小: ~671 GB (磁盘上)
BF16 模型大小: 688B × 2 bytes = 1,376 GB    ← 这才是内存中的实际大小！
```

**重要**：DeepSeek-V3 的 checkpoint 是 FP8 格式（1 byte/param），但 ZeRO-3 初始化时
参数是以 **BF16** 创建的（2 bytes/param）。所以内存中的模型不是 671 GB，而是 **1,376 GB**。

### Qwen3-235B

```
参数量: 235.09B
BF16 checkpoint 大小: 438 GiB ≈ 470 GB
BF16 模型大小: 235B × 2 bytes = 470 GB     ← 理论值正确
```

### 对比总结

| 模型 | 参数量 | 磁盘大小 | BF16 内存大小 |
|---|---|---|---|
| DeepSeek-V3 | 671B | ~671 GB (FP8) | **1,376 GB** |
| Qwen3-235B | 235B | ~470 GB (BF16) | **470 GB** |

---

## 2. ZeRO-3 的内存分配机制

### Phase 0: 模型创建 (deepspeed.zero.Init)

`from_pretrained()` 在 `deepspeed.zero.Init()` 上下文中创建模型。每个参数经历：

```
┌──────────────────────────────────────────────────────────────────┐
│ 1. torch.empty(full_size, device='cuda:X')                       │
│    → GPU 上创建全尺寸参数                                         │
│    → 例: DeepSeek-V3 gate_up_proj = [256, 4096, 7168] = 15 GB    │
│                                                                    │
│ 2. dist.broadcast(param, rank_0, dp_group)                        │
│    → 广播参数数据（in-place, GPU→GPU via NCCL）                   │
│                                                                    │
│ 3. _partition_param():                                             │
│    a. partition_size = full_size / num_ranks                       │
│    b. partitioned_tensor = torch.empty(partition_size, cpu)  ← malloc  │
│    c. partitioned_tensor = pin_memory(partitioned_tensor)    ← cudaHostAlloc  │
│       → 步骤 b 的 malloc 内存被 free，但 glibc 可能不归还 OS      │
│    d. copy_(GPU_slice → CPU_pinned)                               │
│    e. free_param(GPU 全尺寸张量)                                   │
│                                                                    │
│ 4. GPU 内存循环使用，CPU pinned 内存累积                           │
└──────────────────────────────────────────────────────────────────┘
```

**理论上**：Phase 0 完成后，每 rank 持有 model_size / num_ranks 的 pinned 内存。
4 ranks 合计 = model_size。

**实际上**：由于步骤 3b→3c 的 malloc→free 不归还 OS，加上其他运行时开销，
实际内存远超理论值。

---

## 3. 内存超额的七大来源

### 来源 ①：glibc malloc 碎片化（最大因素，尤其对 DeepSeek-V3）

`_partition_param()` 中的 malloc→pin→free 循环导致 glibc 碎片：

```python
# 步骤 b: glibc malloc 分配
partitioned_tensor = torch.empty(partition_size, dtype=bf16, device='cpu')

# 步骤 c: CUDA pin_memory → cudaHostAlloc + memcpy + free(原 malloc)
partitioned_tensor = get_accelerator().pin_memory(partitioned_tensor)
```

当 `free()` 被调用时，glibc 可能将内存保留在 malloc arena 中（即 RSS 不降低），
原因包括：
- **arena 碎片**：大块被释放后，其两侧的小块阻止整个 arena 归还
- **mmap 阈值**：<128KB 的分配用 brk()，无法单独释放
- **arena 数量**：默认 8×CPU 核心数个 arena，每个 arena 独立管理

**patch 缓解措施：**
- `MALLOC_ARENA_MAX=2`：限制 arena 数量
- `MALLOC_TRIM_THRESHOLD_=0`：积极归还内存
- `patch_deepspeed_zero_init_memory`：跳过 malloc→pin 双重分配，直接 `torch.empty(pin_memory=True)`

**为什么 DeepSeek-V3 碎片更严重？**

| | DeepSeek-V3 | Qwen3-235B |
|---|---|---|
| experts.gate_up_proj 全尺寸 | [256, 4096, 7168] = **15 GB** | [128, 3072, 4096] = **3 GB** |
| 每 rank 分区大小 | 15/4 = **3.75 GB** | 3/4 = **0.75 GB** |
| 每次 malloc→free 碎片 | 3.75 GB 级别 | 0.75 GB 级别 |
| MoE 层数 × 大参数 | 58 层 × 2 = 116 次 | 94 层 × 2 = 188 次 |

DeepSeek-V3 的每次碎片比 Qwen3-235B 大 **5 倍**，虽然次数少，但累积碎片更严重。
大分配更容易留下无法合并的空洞。

### 来源 ②：CUDA pinned memory 页对齐开销

`cudaHostAlloc()` 分配页锁定内存，按 4KB（或更大 hugepage）对齐。

```
实际分配 = roundup(requested_size, page_size) + page_table_overhead
```

对于每个参数分区，对齐浪费通常 < 1%，但累积在大量参数上不可忽略。

**Qwen3-235B 实测：**
- 理论 pinned 总量: 470 GB
- 实测 Shmem: 581 GB
- 差值: 111 GB（其中包含 CUDA context、NCCL buffer、pinned 对齐等全部开销）

### 来源 ③：CUDA Context

每块 GPU 需要加载 CUDA 驱动、cuDNN 库、内核缓存等。

```
每 GPU CUDA context ≈ 1-2 GB
4 GPU 合计 ≈ 4-8 GB
```

这部分内存在 GPU 和 CPU 端都有占用（映射到 CPU 地址空间）。

### 来源 ④：NCCL 通信缓冲区

ZeRO-3 使用 NCCL 进行参数广播和梯度全规约。

```
每 rank NCCL buffer ≈ 1-2 GB
4 ranks 合计 ≈ 4-8 GB
```

NCCL 内部使用共享内存（Shmem），在 /proc/meminfo 中计入 Shmem。

### 来源 ⑤：Python / PyTorch 运行时开销

```
Python 解释器 + 导入模块: ~1.5 GB/进程
PyTorch 库 + CUDA 运行时:  ~2 GB/进程
4 进程合计:                ~14 GB
```

### 来源 ⑥：DeepSpeed 内部元数据

每个 ZeRO-3 参数需要维护：
- `ds_tensor`（CPU partition 引用）
- `ds_status`（AVAILABLE/NOT_AVAILABLE/INFLIGHT）
- `ds_shape`、`ds_numel`、`ds_id`
- `param_coordinator` 中的 fetch/release 队列

对于 DeepSeek-V3 的 967 个参数或 Qwen3 的 1037 个参数，元数据本身 < 1 GB。

### 来源 ⑦：Phase 0 中 GPU 上的瞬态全尺寸参数

在 Phase 0 中，每个参数会短暂地以**全尺寸**存在于 GPU 上：

```
DeepSeek-V3 experts.gate_up_proj: 15 GB on GPU (短暂)
→ 广播后分区到 CPU → GPU 内存释放
→ 下一个参数复用 GPU 内存
```

GPU 内存是循环使用的，但同一时刻需要容纳最大的单个参数（15 GB）。
这要求每块 GPU 至少有 15+ GB 空闲（H20 有 143 GB，没问题）。

---

## 4. 两个模型的内存预算对比

### DeepSeek-V3 (671B) — 估算（Phase 0 未完成）

```
                          每 rank (GB)    4 ranks 合计 (GB)
                          ────────────    ─────────────────
ZeRO-3 BF16 分区            344              1,376
glibc 碎片 (估计 ~30%)      ~103              ~413
CUDA context                  2                 8
NCCL buffer                   2                 8
Python/PyTorch 运行时          3.5              14
pinned 对齐 + 元数据          ~10              ~40
────────────────────────────────────────────────────
理论合计                     ~465             ~1,859

实测 (外推 100%):            ~620             ~2,481
实测倍率:                    ~1.80x
```

**注意**：上述"glibc 碎片 ~30%"是保守估计。实际外推值 2,481 GB 远超理论合计 1,859 GB，
说明碎片率可能更高（~50%+），或存在其他未量化的开销。
这也解释了为什么仅补丁（修复双重分配）只能从 697→729 参数（+4.6%），
而不能根本性解决问题。

### Qwen3-235B — 实测

```
                          每 rank (GB)    4 ranks 合计 (GB)
                          ────────────    ─────────────────
ZeRO-3 BF16 分区            117.5             470
glibc 碎片 (实测 ~8%)        ~9.4              ~37.6
CUDA context                  2                 8
NCCL buffer                   2                 8
Python/PyTorch 运行时          3.5              14
pinned 对齐 + 元数据          ~5               ~20
Dataloader workers (16个)     —                ~24
────────────────────────────────────────────────────
理论合计                     ~140             ~582

实测稳态:                    147.9 (RSS)       649 (系统)
Shmem (pinned memory):        —                581
AnonPages (私有):             —                20.2
实测倍率:                    1.26x (per rank)  1.21x (系统级)
```

### 对比总结

| 指标 | DeepSeek-V3 (671B) | Qwen3-235B |
|---|---|---|
| 参数量 | 671B | 235B |
| BF16 模型大小 | **1,376 GB** | **470 GB** |
| ZeRO-3 每 rank | 344 GB | 117.5 GB |
| 最大单参数分区 | 3.75 GB | 0.75 GB |
| **实测/理论倍率** | **~1.80x** | **~1.26x** |
| **Phase 0 结果** | **OOM (75.4%)** | **成功** |
| glibc 碎片率 | ~50%+ | ~8% |
| 原因 | 大分区 → 严重碎片 | 小分区 → 轻微碎片 |

---

## 5. 为什么 DeepSeek-V3 碎片 1.80x 而 Qwen3 只有 1.26x？

核心原因是 **MoE 专家的单个参数尺寸差异巨大**：

```
DeepSeek-V3:
  256 experts × gate_up_proj = 单个 tensor [256, 4096, 7168] = 15 GB
  256 experts × down_proj    = 单个 tensor [256, 7168, 2048] = 7.5 GB
  → 分区后每 rank: 3.75 GB 和 1.875 GB

Qwen3-235B:
  128 experts 分别存储为独立 tensor
  单个 expert 的 gate_proj: [1536, 4096] = 12 MB
  → 融合后 gate_up_proj: [128, 3072, 4096] ≈ 3 GB
  → 分区后每 rank: ~0.75 GB
```

glibc malloc 的碎片率与单次分配大小正相关：
- **3.75 GB 的分配→释放→重新分配**：留下巨大的空洞，glibc 倾向于保留整个 arena
- **0.75 GB 的分配→释放→重新分配**：空洞较小，glibc 更容易合并和归还

此外，DeepSeek-V3 的 `_partition_param` 循环中，每层有 **2 次**大分配：
- gate_up_proj: 3.75 GB → free → pin → 3.75 GB
- down_proj: 1.875 GB → free → pin → 1.875 GB

每层累计碎片 ≈ 5.625 GB，58 层 MoE 累计 ≈ **326 GB 碎片**。
这与实测数据（理论 1,376 GB → 实际外推 ~2,481 GB，差 ~1,105 GB）在数量级上吻合
（326 GB 碎片 + 其他开销 = 总超额）。

---

## 6. 关键结论

### 6.1 DeepSeek-V3 (671B) 在 2TB 机器上无法通过 Phase 0

即使用了所有可用补丁（直接 pinned 分配 + gc + malloc_trim + arena 限制），
外推全模型仍需 ~2,481 GB >> 1,923 GB 总内存。

**可能的解决路径（按可行性排序）：**

1. **增加 swap**（快速，临时方案）：在 NVMe 上创建 500-800 GB swap，
   Phase 0 期间允许 swap，Phase 0 完成后 swap 不再需要。

2. **使用 jemalloc 替换 glibc malloc**：
   `LD_PRELOAD=libjemalloc.so.2 torchrun ...`
   jemalloc 的碎片率通常远低于 glibc，可能将倍率从 1.8x 降至 1.3-1.4x。

3. **禁用 pin_memory**：消除 malloc→pin 双重分配（但降低训练速度）。

4. **meta device 初始化**（根本方案）：跳过 Phase 0 的参数物化，
   直接在 meta device 上创建模型骨架，然后从 checkpoint 逐参数加载。

### 6.2 Qwen3-235B 在当前机器上运行良好

- Phase 0 + Phase 1 成功完成
- 稳态内存仅 649 GB (可用 1273 GB)
- 1.26x 倍率可接受，碎片率低

### 6.3 "671G" 的误解

DeepSeek-V3 的 671 GB 是 **FP8 checkpoint 的磁盘大小**（1 byte/param），
而不是内存中的模型大小。ZeRO-3 初始化以 BF16 创建参数，
实际内存中的模型大小是 **1,376 GB**（2 bytes/param）。

所以不是 "671 GB 模型为什么用了 1871 GB"，
而是 "**1,376 GB BF16 模型 + ~1,100 GB 运行时开销** = ~2,481 GB（外推值）"。
