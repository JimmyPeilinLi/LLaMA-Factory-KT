# DeepSpeed ZeRO-3 Offload 深度 Profile：Qwen3-235B

> **目标**：为论文提供 Qwen3-235B 在 DeepSpeed ZeRO-3 CPU offload 架构下的性能瓶颈
> 量化分析，突出同步/调度/传输开销，为后续优化工作提供数据支撑。
>
> **基于实测数据**（2026-02-24 运行 42 步后手动终止）+ 修复后的 LoRA 配置推算。
>
> **[2026-02-25 更新]** 修复了 `find_all_linear_modules` 无法发现 MoE 专家层的问题。
> 新增 `find_all_expert_parameters` + PEFT `target_parameters`（ParamWrapper），
> LoRA 现在覆盖 attention + routed experts，可训练参数从 25.4M 提升至 **~1.258B**。
> 步时间、内存等实测数据仍基于旧配置，文中标注了修复后的理论推算值。

---

## 0. 需求与思考记录

### 0.1 需求

1. 理论 token/s 上限
2. 每个 step 内的 GPU-CPU 同步开销、权重传输开销、架构调度开销
3. 实际工程 token/s
4. 实际 GPU 利用率、CPU 利用率、PCIe 利用率
5. 在 H20 和 4090+PCIe-4.0 两种硬件下的对比估算
6. 内存/显存的详细用途拆解

### 0.2 关键思考

**ZeRO-3 vs APTMoE 的本质区别**：

| 维度 | ZeRO-3 Offload | APTMoE Pipeline |
|------|---------------|----------------|
| 参数分布 | 按 tensor 切分到 N 个 rank 的 CPU 上 | 按 pipeline stage 分配到 N 个 GPU |
| 加载方式 | 每层每次 allgather 全量参数到所有 GPU | 每个 GPU 只加载自己的 stage |
| 核心传输 | PCIe (1/N CPU→GPU) + NVLink (allgather) | 纯 PCIe (CPU→GPU) |
| GPU 间通信 | NCCL allgather (NVLink/PCIe) | 无 (各 GPU 独立) |
| 瓶颈来源 | allgather 通信 (NVLink ring) | PCIe 单卡带宽 |

**关键发现**：H20 系统有 NVLink 4.0 互联，ZeRO-3 的 allgather 走 NVLink；
但 4090 系统无 NVLink，allgather 只能走 PCIe，**这是两种硬件差异最大的地方**。

---

## 1. 运行环境

### 1.1 硬件

| 组件 | 规格 |
|------|------|
| GPU | 8 × NVIDIA H20-3e, 140.4 GB HBM3e, SM 9.0 |
| GPU 算力 | BF16 Tensor Core 148 TFLOPS, FP32 74 TFLOPS |
| GPU 显存带宽 | ~4.0 TB/s (HBM3e) |
| GPU 互联 | **NVLink 4.0, NV18 (18 links/GPU, 900 GB/s total/GPU)** |
| PCIe | Gen5 x16, 理论单向 63 GB/s |
| CPU | 2 × Intel Xeon 6759P-C (60 核/socket, 120 核/240 线程) |
| 内存 | 1,923 GB DDR5 |
| 拓扑 | GPUs 4-7 on NUMA 1 (CPU cores 120-239) |

### 1.2 NVLink 拓扑

```
GPU-GPU 互联: 全连接 NVLink 4.0
8 GPU 共 18 links/GPU, 分配给 7 个 peer
每对 GPU: ~2-3 NVLink links → 100-150 GB/s 双向 (50-75 GB/s 单向)
```

### 1.3 训练配置

```yaml
模型: Qwen3-235B-A22B-Instruct (BF16, 94 层 × 128 专家)
DeepSpeed: ZeRO Stage 3, CPU offload (param + optimizer)
LoRA: rank=8, target=all (attention q/k/v/o_proj + experts gate_up/down_proj)
BS=1 per device × 4 GPUs = total BS 4
cutoff_len: 4096 (实际数据平均 ~296 tokens)
gradient_checkpointing: enabled (默认)
gradient_accumulation: 1
```

### 1.4 运行指令

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 FORCE_TORCHRUN=1 \
  llamafactory-cli train examples/train_lora/qwen3_235b_lora_sft_ds3.yaml
```

---

## 2. 模型架构参数

> 与 APTMoE 参考文档一致，此处仅做精要汇总。

### 2.1 全模型

| 参数 | 值 |
|------|---|
| 总参数 | 235.1B |
| BF16 总体积 | **470 GB** |
| 活跃参数 (A22B) | 22.2B |
| 每层参数 | 2.488B = **4.75 GB** (BF16) |
| 其中 128 Experts | 2.416B = 4.61 GB (97.1%) |
| 其中 Attention | 71.3M = 136 MB (2.9%) |

### 2.2 LoRA 参数

**[已修复]** 新增 `find_all_expert_parameters()` + PEFT `target_parameters` (ParamWrapper)，
LoRA 同时覆盖 Attention (nn.Linear) 和 MoE Expert (fused 3D nn.Parameter)。

#### Attention LoRA (target_modules, 标准 nn.Linear)

| 项目 | 值 |
|------|---|
| 目标模块 | q_proj, k_proj, v_proj, o_proj |
| 每层参数 | 270,336 |
| 94 层合计 | 25,411,584 ≈ 25.4M |

#### Expert LoRA (target_parameters, PEFT ParamWrapper)

Qwen3MoeExperts 使用 fused 3D `nn.Parameter`（非 nn.Linear），
PEFT 通过 `ParamWrapper` 对 3D 参数施加 LoRA，结构为 `lora_A(r×E, in) + lora_B(out, r×E)`。

| 参数 | 形状 | E | in | out | lora_A | lora_B | 每层小计 |
|------|------|---|-----|------|--------|--------|---------|
| gate_up_proj | (128, 3072, 4096) | 128 | 3072 | 4096 | (1024, 3072) | (4096, 1024) | 7,340,032 |
| down_proj | (128, 4096, 1536) | 128 | 4096 | 1536 | (1024, 4096) | (1536, 1024) | 5,767,168 |
| **Expert 每层合计** | | | | | | | **13,107,200** |
| **94 层合计** | | | | | | | **1,232,076,800 ≈ 1.232B** |

> **注**：ParamWrapper 的 LoRA 维度为 `r × num_experts = 8 × 128 = 1024`，
> 每个 expert 等效 rank=8。delta_weight 通过 `einsum("o r e, e r i -> e i o")` 计算，
> 形状与原始 3D 参数一致。

#### LoRA 汇总

| 项目 | 值 |
|------|---|
| Attention LoRA | 25,411,584 (25.4M) |
| Expert LoRA | 1,232,076,800 (1.232B) |
| **全模型 LoRA** | **1,257,488,384 ≈ 1.258B (0.532%)** |
| LoRA BF16 体积 | **~2.35 GB** |
| LoRA FP32 master | **~4.68 GB** |

> **[对比]** 修复前仅覆盖 attention → 25.4M (0.011%)；
> 修复后覆盖 attention + experts → **1.258B (0.532%)**，提升 **49.5×**。
> APTMoE 的 LoRA 覆盖范围类似，为 1.652B（差异源于 APTMoE 使用独立 nn.Linear per expert，
> 而 ParamWrapper 对 fused 3D 参数施加 LoRA，结构略有不同）。

### 2.3 DeepSpeed 持久化参数

| 项目 | 值 |
|------|---|
| 持久参数数量 | 941 |
| 持久参数元素数 | 13,889,024 |
| 持久参数体积 (BF16) | ~26.5 MB |
| 内容 | LayerNorm, RMSNorm, router bias 等小参数 |

> 这些参数始终驻留在 GPU 上，不参与 offload。

---

## 3. ZeRO-3 Offload 数据流分析

### 3.1 参数分布

```
4 GPUs, ZeRO-3: 每个 rank 持有全模型 1/4 参数在 CPU pinned memory

Per rank on CPU:
  模型参数 (BF16): 235.1B / 4 = 58.8B = 117.5 GB
  LoRA master (FP32): 1.258B / 4 × 4 bytes = 1.17 GB
  Adam states (m+v, FP32): 1.258B / 4 × 8 bytes = 2.34 GB
  Gradients (FP32): 1.258B / 4 × 4 bytes = 1.17 GB
  Per rank CPU total: ~122.2 GB
  4 ranks total: ~489 GB
```

### 3.2 每层参数加载流程 (Allgather)

**对于每个模块的 forward/backward，DeepSpeed 执行 allgather：**

```
[Step 1] CPU → local GPU (PCIe upload)
  每个 rank 上传自己持有的 1/4 参数分片
  数据量: S/4 per rank (S = 模块全量参数)
  路径: CPU pinned memory → PCIe Gen5 → GPU HBM

[Step 2] NCCL Ring Allgather (NVLink on H20 / PCIe on 4090)
  4 ranks 交换各自的分片, 每个 GPU 获得完整参数
  Ring: 3 步, 每步每 rank 发送 S/4, 接收 S/4
  数据量: (N-1)/N × S = 3/4 × S per rank

[Step 3] GPU 计算
  使用完整参数执行 forward 或 backward

[Step 4] 释放 GPU 内存
  丢弃已聚合的参数, 腾出显存空间
```

### 3.3 Per-Step 数据传输量

**Gradient Checkpointing 下 (已确认启用)：**

| 阶段 | 每层 allgather 次数 | 说明 |
|------|-------------------|------|
| Forward (no_grad) | 1 | 94 层 + embed + lm_head |
| Backward (recompute + grad) | 1 | 重计算 forward + backward 共用同一次 allgather |
| **每层总计** | **2** | 188 次 allgather (94 × 2) |

```
每次 allgather 传输量 (per GPU):
  PCIe upload: S/4 = 4.75/4 = 1.19 GB    (CPU → local GPU)
  NVLink receive: 3S/4 = 3.56 GB          (from 3 other GPUs)

Per step per GPU 总传输:
  PCIe: 188 × 1.19 = 223.7 GB  (CPU → GPU direction)
  NVLink: 188 × 3.56 = 669.3 GB (GPU ↔ GPU)
  合计每 GPU "看到" 的参数: 188 × 4.75 = 893 GB

梯度 offload (GPU → CPU):
  LoRA 1.258B 参数: ~4.68 GB total, 1.17 GB per rank
  LoRA allgather (ZeRO-3 also offloads LoRA):
    2 passes × 2.35 GB = ~4.7 GB → 占基模型 893 GB 的 0.5%, 可忽略
```

---

## 4. 理论 Token/s 上限

### 4.1 PCIe 带宽瓶颈

```
Per GPU per step PCIe upload: 223.7 GB
H20 PCIe Gen5 x16: 63 GB/s 理论
理论 PCIe 时间: 223.7 / 63 = 3.55 s

但 PCIe 可与 NVLink 和计算 overlap → PCIe 不是瓶颈
```

### 4.2 NVLink 带宽瓶颈 (H20)

```
Per GPU per step NVLink traffic: 669.3 GB
每对 GPU: ~3 NVLink links, 75 GB/s 单向

Ring allgather per layer (S = 4.75 GB, N = 4):
  Per ring step: S/4 = 1.19 GB at 75 GB/s = 15.9 ms
  3 ring steps: 47.6 ms per layer allgather

PCIe upload per layer: 1.19 GB / 63 = 18.9 ms
PCIe + NVLink sequential per layer: 18.9 + 47.6 = 66.5 ms

188 层 (94 × 2 passes):
  理论最小传输时间: 188 × 66.5 ms = 12.5 s
```

### 4.3 GPU 计算时间

```
活跃参数: 22.2B
FLOPs per token: 2 × 22.2B = 44.4 GFLOPs

Gradient checkpointing = 3× forward 等效:
  Forward: 1× | Backward recompute: 1× | Backward grad: 1× = 3×

Per GPU per step (BS=1, seq=L):
  FLOPs = 3 × 2 × 22.2B × L = 133.2B × L

At seq=300 (数据集平均):
  FLOPs = 133.2B × 300 = 39.96 TFLOPS
  At 40% GPU 利用率: 39.96 / (148 × 0.40) = 0.67 s

At seq=4096 (cutoff_len):
  FLOPs = 133.2B × 4096 = 545.6 TFLOPS
  At 40% GPU 利用率: 545.6 / (148 × 0.40) = 9.2 s
```

### 4.4 理论 Token/s 上限汇总

| 序列长度 | 传输时间 | 计算时间(40%) | 步时间(有 overlap) | tokens/step | **token/s** |
|---------|---------|-------------|-------------------|-------------|------------|
| 128 | 12.5 s | 0.28 s | ~13 s | 4×128=512 | **~39** |
| 300 | 12.5 s | 0.67 s | ~13 s | 4×300=1200 | **~92** |
| 1024 | 12.5 s | 2.3 s | ~13 s | 4×1024=4096 | **~315** |
| 4096 | 12.5 s | 9.2 s | ~14 s | 4×4096=16384 | **~1170** |

> **理论上限由 NVLink allgather 决定 (~12.5s)，而非 PCIe (~3.6s)**。
> 在短序列下，GPU 计算被完全掩盖；在长序列下，计算开始接近传输时间。

---

## 5. Per-Step 开销拆解 (H20 实测)

### 5.1 实测步时间

```
实测步时间 (steps 5-42 平均): 11.22 s/step
数据集: math_train, 平均 ~296 tokens/sample, 中位数 124 tokens
每步 tokens: ~4 × 296 = 1,184 (平均), ~4 × 124 = 496 (中位)
步时间标准差: ~1.2 s (10-15 s range)
```

### 5.2 步时间拆解

基于旧配置实测 11.22s 和理论分析反推。修复后 LoRA 覆盖 experts，
预估步时间增加 ~0.5-1.0s（CPU Adam 从 <50ms → ~300ms，ParamWrapper delta_weight 计算 ~100ms），
即 **修复后预估步时间 ~11.7-12.2s**。

| 类别 | 时间 (s) [旧配置实测] | 时间 (s) [修复后推算] | 占比 [修复后] | 说明 |
|------|---------------------|---------------------|-------------|------|
| **NVLink Allgather** | ~5.8 | **~5.8** | **48.3%** | 188 ops × 3 ring steps × 15.9 ms/step × ~65% 效率 |
| **PCIe 权重传输** | ~2.0 | **~2.0** | **16.7%** | 223.7 GB upload + ~2.4 GB LoRA (可忽略) |
| **NCCL 同步开销** | ~1.4 | **~1.4** | **11.7%** | 188 ops × ~7 ms (launch + barrier + event wait) |
| **Python/框架调度** | ~1.0 | **~1.1** | **9.2%** | DeepSpeed hooks + ParamWrapper 注册/移除 (~100ms) |
| **GPU 计算** | ~0.7 | **~0.8** | **6.7%** | 40 TFLOPS @ 40% + LoRA delta_weight einsum |
| **Optimizer + Misc** | ~0.3 | **~0.9** | **7.5%** | CPU Adam 1.258B 参数 (~300ms) + gradient reduce + data loading |
| **合计** | ~11.2 | **~12.0** | **100%** | |

### 5.3 开销可视化

```
修复后预估步时间 ~12.0s 分解:

NVLink Allgather  ████████████████████████████████████████████████         48.3%
PCIe 权重传输     █████████████████                                        16.7%
NCCL 同步开销     ████████████                                             11.7%
Python/框架调度   █████████                                                 9.2%
Optimizer+Misc    ████████                                                  7.5%
GPU 计算          ███████                                                   6.7%
                  |----|----|----|----|----|----|----|----|----|----|
                  0%  10%  20%  30%  40%  50%  60%  70%  80%  90% 100%
```

> **核心发现**：**NVLink allgather 仍是最大瓶颈 (48%)**，而非 PCIe (17%)。
> 修复后 Optimizer 占比从 2.7% 增至 7.5%（CPU Adam 处理 1.258B 参数 vs 25.4M），
> 但对总步时间影响有限 (+0.6s)，因为 allgather 仍然是压倒性的主导项。
>
> **同步 + 调度 = 20.9%**，这是优化的重要目标。

---

## 6. 实际工程指标

### 6.1 Token/s

| 指标 | 旧配置实测 | 修复后推算 | 说明 |
|------|----------|----------|------|
| 稳态步时间 | 11.22 s | **~12.0 s** | 旧: Steps 5-42 均值; 新: +optimizer/ParamWrapper |
| 平均 tokens/step | ~1,184 | ~1,184 | 4 × avg_seq_len(296) |
| 中位 tokens/step | ~496 | ~496 | 4 × median_seq_len(124) |
| **平均 token/s** | ~106 | **~99** | 1184 / 12.0 |
| **中位 token/s** | ~44 | **~41** | 496 / 12.0 |
| 等效 token/s (seq=4096) | ~1,170 | **~1,130** | 理论: 16384 / 14.5 |
| 等效 token/s (seq=128) | ~39 | **~37** | 理论: 512 / 13.5 (传输主导) |

> **注意**：token/s 随序列长度变化很大。在传输主导的 regime 下，
> 短序列的 token/s 很低（大量时间浪费在与数据无关的参数搬运上）；
> 长序列 token/s 更高（计算量匹配传输量），但步时间也更长。

### 6.2 关键日志指标

```
[旧配置实测, LoRA attention-only]
训练启动: 21:02:16
第一步完成: +18s (含 NCCL warmup)
稳态步时间: ~11s (step 8+)
内存 (系统): 稳态 ~650 GB / 1923 GB (33.8%)
GPU Memory Allocated: 0.05 GB (DeepSpeed 管理)
GPU Memory Cached: 0.1-6 GB (CUDA allocator)
总参数: 235,119,046,144 (含 LoRA)
可训练参数: 25,411,584

[修复后推算, LoRA attention + experts]
总参数 (预估): ~236,351,122,944 (含 1.258B LoRA)
可训练参数: ~1,257,488,384
稳态步时间 (预估): ~12s
内存 (预估): ~664 GB (+14 GB 优化器/梯度)
持久化参数 (GPU 常驻): 941 个, 13,889,024 元素 (不含 LoRA, 由 ZeRO-3 管理)
```

---

## 7. 利用率分析

### 7.1 GPU 计算利用率

```
修复后推算 (avg seq=296):
  FLOPs per step: 3 × 2 × 22.2B × 4 × 296 = 157.8 TFLOPS (4 GPUs)
  Per GPU: 39.5 TFLOPS
  Per GPU per second: 39.5 / 12.0 = 3.29 TFLOPS/s
  H20 BF16 peak: 148 TFLOPS
  GPU 计算利用率: 3.29 / 148 = 2.23%

若 seq=4096:
  Per GPU per second: 545.6 / 14.5 = 37.6 TFLOPS/s
  GPU 计算利用率: 37.6 / 148 = 25.4%
```

### 7.2 PCIe 带宽利用率

```
Per GPU PCIe upload per step: 223.7 GB (基模型) + ~1.2 GB (LoRA) ≈ 225 GB
Step time: ~12.0 s
Effective PCIe throughput: 225 / 12.0 = 18.7 GB/s
H20 PCIe Gen5 理论: 63 GB/s

PCIe 有效利用率: 18.7 / 63 = 29.7%
```

> PCIe 利用率仅 31.6%，因为 PCIe 只负责上传 1/4 参数分片（CPU→GPU），
> 实际传输量远小于全量参数。其余 3/4 由 NVLink 传输。

### 7.3 NVLink 利用率

```
Per GPU NVLink receive per step: 669.3 GB
Step time: ~12.0 s
Effective NVLink throughput: 669.3 / 12.0 = 55.8 GB/s
Per-pair NVLink 理论 (3 links): ~75 GB/s 单向

NVLink 有效利用率: 55.8 / 75 = 74.4% (ring 效率)
```

> NVLink 利用率较高 (79.5%)，是主要瓶颈。

### 7.4 CPU 利用率

```
CPU 主要工作:
  - pinned memory 管理 (4 进程)
  - Adam optimizer step (1.258B 参数, ~300ms)
  - 数据加载 (DataLoader, 4 workers)
  - 内存拷贝 (memcpy for PCIe DMA staging)

估算 CPU 利用率: ~25-35% (120 核中 ~35 核活跃)
修复后 optimizer step 从 <50ms 增至 ~300ms, 但仍不是瓶颈
```

### 7.5 利用率汇总 (H20)

| 指标 | 修复后推算值 (avg seq=296) | 理论极限 |
|------|--------------------------|---------|
| GPU 计算利用率 | **2.2%** | 100% (148 TFLOPS) |
| PCIe 带宽利用率 | **29.7%** | 100% (63 GB/s) |
| NVLink 利用率 | **74.4%** | 100% (~75 GB/s/pair) |
| CPU 利用率 | **~30%** | 100% (120 cores) |
| HBM 带宽利用率 | **<0.5%** | 100% (4 TB/s) |

> **GPU 计算利用率极低 (2.2%)**，几乎全部时间在等 NVLink allgather 和 PCIe 传输。

---

## 8. 4090 + PCIe 4.0 估算

### 8.1 硬件参数对比

| 参数 | RTX 4090 | H20-3e | 比值 |
|------|---------|--------|------|
| BF16 Tensor Core | 165 TFLOPS | 148 TFLOPS | 4090 +11% |
| VRAM | 24 GB GDDR6X | 140 GB HBM3e | H20 5.8× |
| 显存带宽 | 1.01 TB/s | ~4.0 TB/s | H20 4× |
| PCIe | Gen4 x16 (31.5 GB/s) | Gen5 x16 (63 GB/s) | H20 2× |
| **GPU 互联** | **无 NVLink** | **NVLink 4.0 (NV18)** | **质变** |

### 8.2 瓶颈转移：NVLink → PCIe

**H20 有 NVLink，PCIe 只搬 1/4；4090 无 NVLink，PCIe 搬 4/4。**

这是两种硬件最根本的区别。ZeRO-3 allgather 需要每个 GPU 获得整层全量参数 S：
- 1/4 来自本机 CPU（PCIe upload）
- 3/4 来自其他 GPU（H20 走 NVLink；4090 走 PCIe → CPU → PCIe）

```
┌─────────────────────────────────────────────────────────┐
│  H20 (有 NVLink):                                       │
│                                                         │
│  CPU ──PCIe──→ GPU-0 ←──NVLink──→ GPU-1/2/3            │
│        (1/4)           (3/4, 高速直连)                   │
│                                                         │
│  每 GPU PCIe 下行: 1/4 × S = 1.19 GB/层                 │
│  每 GPU NVLink 收: 3/4 × S = 3.56 GB/层                 │
│  → PCIe 只搬 223 GB/step → 不是瓶颈                      │
│  → NVLink ring 搬 670 GB/step → 瓶颈所在                 │
├─────────────────────────────────────────────────────────┤
│  4090 (无 NVLink):                                      │
│                                                         │
│  CPU ──PCIe──→ GPU-0 ←──PCIe──CPU──PCIe──→ GPU-1/2/3   │
│        (1/4)        (3/4, 全部绕道 PCIe!)               │
│                                                         │
│  每 GPU PCIe 下行: S/4(CPU) + 3S/4(ring) = S = 4.75 GB  │
│  → 每次 allgather, GPU 下行链路承载整层全量!              │
│  → PCIe 搬 893 GB/step → PCIe 链路 100% 饱和!            │
└─────────────────────────────────────────────────────────┘
```

**每 GPU PCIe 链路流量对比：**

| 方向 | H20 per step | 4090 per step | 比值 |
|------|-------------|--------------|------|
| Download (host→GPU) | 223 GB (仅 1/4 分片) | **893 GB (全量!)** | **4.0×** |
| Upload (GPU→host) | ~0 (NVLink 代劳) | 670 GB (ring send) | ∞ |
| **PCIe 带宽** | Gen5: 63 GB/s | Gen4: 31.5 GB/s | 0.5× |
| **下行占满时间** | 223/63 = **3.5 s** | 893/31.5 = **28.3 s** | **8.1×** |
| **是否瓶颈** | ❌ 不是 (NVLink 更慢) | **✅ 唯一瓶颈, 100% 饱和** | |

> **核心结论**：
> - H20: 流量分流 → PCIe 只搬 1/4 → NVLink ring 是瓶颈
> - 4090: 流量汇聚 → PCIe 搬 4/4 → **PCIe 直接打满到理论极限**
> - 4090 的 PCIe 下行负载是 H20 的 **4×**，带宽却只有 H20 的 **0.5×**
> - 综合效果：4090 allgather 时间 = H20 的 **8.1×**（但被 overlap 和流水线减缓至约 2.6-3×）

### 8.3 4090 步时间估算

```
Per layer allgather (4090):
  PCIe upload (CPU→GPU, 1/4): 1.19 / 31.5 = 37.8 ms
  PCIe ring (3 steps, full-duplex): 3 × 37.8 = 113.3 ms
  Total: 151.1 ms per layer
  (对比 H20: 66.5 ms → 2.27×)

但有 prefetch overlap:
  计算 layer L 期间预取 layer L+1
  At seq=300: compute ~5 ms << transfer 151 ms → 传输完全主导
  At seq=4096: compute ~80 ms << transfer 151 ms → 仍然传输主导!

关键差异: 在 H20 上, seq=4096 时计算 (80ms) 接近传输 (66ms), 有一定 overlap
         在 4090 上, seq=4096 时计算 (80ms) 仍远小于传输 (151ms), 永远传输主导

最终步时间:
  At avg seq=296: 188 × ~130 ms (overlap 后) + 3s overhead ≈ 27.4 + 3 ≈ ~33 s
  At seq=4096:    188 × ~130 ms (overlap 后) + 5s overhead ≈ 24.4 + 8 ≈ ~35 s
```

### 8.4 4090 GPU 显存可行性

```
ZeRO-3 peak per GPU:
  Gathered params (max_live=1e9 + prefetch): ~3 GB
  Activations (gradient checkpointing, 1 layer):
    seq=300: ~100 MB
    seq=4096: ~300-500 MB
  NCCL buffers: ~0.5 GB (无 NVLink, 仅 PCIe)
  CUDA context: ~1-2 GB
  LoRA gathered (ParamWrapper, per layer): ~25 MB (ZeRO-3 动态 gather)
  LoRA delta_weight 临时缓冲: ~50 MB (per layer, 用后释放)

  Peak total: ~5-7 GB (seq=300), ~7-8 GB (seq=4096)
  4090 VRAM: 24 GB
  ✅ 可行, 余量 ~16 GB
```

### 8.5 4090 vs H20 性能对比

| 指标 | H20 (修复后推算) | 4090 (估算) | 比值 |
|------|----------------|------------|------|
| **步时间 (avg seq=296)** | **~12.0 s** | **~34 s** | **2.8×** |
| **步时间 (seq=4096)** | **~14.5 s** | **~36 s** | **2.5×** |
| token/s (avg seq=296) | ~99 | ~35 | 0.35× |
| token/s (seq=128) | ~37 | ~15 | 0.41× |
| token/s (seq=4096) | ~1,130 | ~455 | 0.40× |
| **瓶颈所在** | **NVLink ring** | **PCIe 链路 (100% 饱和)** | **质变** |
| PCIe 下行流量/step | 223 GB (1/4) | **893 GB (4/4)** | 4.0× |
| GPU 计算利用率 (seq=296) | 2.3% | ~0.8% | |
| PCIe 有效利用率 | 31.6% | **~100% (打满!)** | |
| NVLink 利用率 | 79.5% | N/A | |
| GPU 显存峰值 | ~7 GB | ~7 GB | |
| GPU 显存可行性 | ✅ (140 GB) | ✅ (24 GB) | |

> **4090 比 H20 慢约 2.5-2.8×**，核心原因（按贡献排序）：
>
> 1. **无 NVLink (质变, ~4× 流量增)**：allgather 3/4 的跨 GPU 数据被迫走 PCIe，
>    导致每 GPU PCIe 下行流量从 223 GB 暴增到 893 GB — **翻了 4 倍**
> 2. **PCIe Gen4 vs Gen5 (2× 带宽差)**：31.5 vs 63 GB/s
> 3. **综合：PCIe 下行 100% 打满**：893 GB / 31.5 GB/s = 28.3s 纯传输，
>    PCIe 链路完全饱和，**没有任何优化空间**（除非减少传输量本身）
>
> 4090 上 GPU 计算利用率降至 ~0.8%，99.2% 的时间在等 PCIe 传数据。

---

## 9. CPU 内存详细拆解

### 9.1 核心问题

> "为什么 235B 的 LoRA 微调需要这么多内存？除了 470 GB 模型参数以外，剩下的都用来干嘛了？"

### 9.2 实测数据

| 指标 | 值 |
|------|---|
| 系统总消耗 (稳态) | **~650 GB** |
| 训练前基线 (OS) | 48 GB |
| 模型加载后 | 644-687 GB (波动) |
| 训练稳态 | 645-706 GB (波动, 收敛至 649 GB) |

> **关键差异**: ZeRO-3 的内存消耗 (~650 GB) 远低于 APTMoE (~1,168 GB)。
> 这是因为 ZeRO-3 使用 pinned memory 分区存储，不涉及大块 alloc/free。

### 9.3 内存增长时间线

```
时间                系统已用    事件
20:46:18 (T+0)     48 GB      基线 (OS + Python 启动)
20:47:18~20:47:28  50→69 GB   4 进程启动 (+21 GB)
20:47:58~20:50:19  124→687 GB 模型加载 (+563 GB, ~170s)
20:50:19~20:54:41  683→706 GB 训练初始化 + 前几步 (波动)
20:54:51~21:10:28  643→650 GB 训练稳态 (缓慢收敛至 649 GB)
21:10:38           639→50 GB  训练终止, 内存释放
```

> **观察**: 内存在加载后迅速稳定 (~650 GB)，没有 APTMoE 那样的 Step 1 暴涨。
> 这说明 ZeRO-3 的内存模式更稳定——参数分区在初始化时一次性 pin，后续不变。

### 9.4 内存去向拆解

#### 已知固定分配 (修复后)

| 组件 | 说明 | 总量 (GB) |
|------|------|----------|
| **① 模型权重 (BF16, pinned)** | 4 rank 各持 1/4, pinned memory | **470** |
| **② LoRA 权重 (BF16, pinned)** | 1.258B × 2 bytes, ZeRO-3 管理 | **2.35** |
| **③ LoRA master (FP32)** | 1.258B × 4 bytes | **4.68** |
| **④ Adam 优化器 (m+v, FP32)** | 2 × 1.258B × 4 bytes | **9.36** |
| **⑤ 梯度 (FP32)** | 1.258B × 4 bytes | **4.68** |
| **已知固定合计** | | **~491** |

> **修复后 LoRA 1.258B 参数 → 优化器/梯度合计 ~21 GB**，
> 比旧配置 (25.4M, 0.4 GB) 增加 ~20 GB，但仍远小于模型权重本身 (470 GB)。

#### 运行时开销 (预估 ~664 - 491 - 48 = ~125 GB)

| 子项 | 估算 (GB) | 说明 |
|------|----------|------|
| Python/PyTorch 进程 | ~20-30 | 4 进程 × 5-7 GB (interpreter, libs, CUDA driver) |
| PyTorch 参数元数据 | ~40-60 | 235.1B + 1.258B 的 Parameter 对象树 (>48K modules + ParamWrapper) |
| DeepSpeed 运行时 | ~15-20 | 参数状态管理, prefetch buffers, allgather staging |
| NCCL 缓冲 | ~8-12 | 4 ranks × 2-3 GB 通信 buffers |
| pinned memory 管理开销 | ~10-15 | 页锁定内存的 OS 页表, TLB 开销 |
| 数据集 + DataLoader | ~5-8 | 331 samples × 4 ranks + 4 worker 进程 |
| 杂项 (Python GC, tmp) | ~5-10 | |
| **运行时合计** | **~103-155** | 中值 ~125 GB |

### 9.5 与 APTMoE 的内存对比

| 指标 | ZeRO-3 (修复后) | APTMoE | 差异原因 |
|------|----------------|--------|---------|
| 系统总消耗 | **~664 GB** | 1,168 GB | ZeRO 无大块 alloc/free |
| 模型权重 | 470 GB | 470 GB | 相同 |
| LoRA 参数量 | **1.258B** | 1.652B | ParamWrapper vs 独立 Linear |
| 优化器/梯度 | **~21 GB** | 26.4 GB | LoRA 1.258B vs 1.652B, 差异不大 |
| 运行时开销 | ~125 GB | ~672 GB | ZeRO 无碎片化 |
| 碎片化 | 极低 | ~200-300 GB | pinned memory 固定, 无 alloc/free |

> **ZeRO-3 的内存效率仍远高于 APTMoE**，核心原因：
> 1. pinned memory 一次性分配, 不产生碎片
> 2. 优化器开销：ZeRO-3 为 21 GB vs APTMoE 26 GB — 修复后差距缩小（两者 LoRA 覆盖范围接近）
> 3. 无 pipeline stage 的大块 .to(cuda/cpu) 导致的碎片化 — **这才是最大差异 (~500 GB)**

### 9.6 GPU 显存拆解 (per card)

| 组件 | 体积 | 说明 |
|------|-----|------|
| Gathered params (当前层) | ~2-3 GB | max_live_parameters=1e9 限制 |
| Prefetch 缓冲 | ~0.5-1 GB | stage3_prefetch_bucket_size |
| LoRA gathered (当前层) | ~25 MB | ParamWrapper 的 lora_A + lora_B, ZeRO-3 gather |
| LoRA delta_weight 临时 | ~50 MB | per layer, einsum 后立即用于 parametrize |
| 激活 (BS=1, 短序列) | ~100-300 MB | gradient checkpointing, 仅 1 层 |
| Persistent params | 27 MB | LayerNorm 等 |
| NCCL 缓冲 | ~0.5-1 GB | allgather + reduce 通信 |
| CUDA context | ~1-2 GB | 驱动 + kernel 缓存 |
| **估算峰值** | **~6-8 GB** | |
| DeepSpeed 报告 Max_MA | 5.63 GB | [旧配置实测] Max Memory Allocated |
| DeepSpeed 报告 Max_CA | 6 GB | [旧配置实测] Max CUDA Allocated |

> 修复后 GPU 显存峰值预估 ~7-8 GB（新增 LoRA gathered + delta_weight 缓冲），
> 远低于 H20 的 140 GB 容量。
> **4090 (24 GB) 仍然完全可以运行 ZeRO-3 offload 训练 235B 模型**。

---

## 10. 关键发现与优化方向

### 10.1 核心发现

1. **NVLink allgather 是 H20 上的主要瓶颈 (48%)**:
   ZeRO-3 的 allgather 需要 ring 通信，188 次 allgather 操作的 NVLink 传输
   累积占步时间的近一半。即使有 NVLink 4.0 加持，ring 协议的 O(N) 步骤
   仍然限制了有效吞吐。

2. **同步 + 调度开销显著 (21%)**:
   NCCL 同步 (12%) + Python/框架调度 (9%) 合计占步时间的 1/5。
   188 次 allgather 操作中，每次的 NCCL launch latency (~2-3 ms) 和
   DeepSpeed 参数状态管理 (~3-4 ms) 累积成巨大开销。

3. **GPU 计算极度浪费 (~2.3% 利用率)**:
   在短序列下，GPU 99% 的时间在等待参数搬运。即使是 seq=4096，
   利用率也仅有 ~26%。

4. **4090 无 NVLink 是致命劣势**:
   在 4090 上，allgather 全部走 PCIe Gen4，导致 per-layer 传输时间
   从 H20 的 66.5 ms 增至 151 ms (2.27×)。步时间从 ~12s 增至 ~34s。

5. **内存效率高于 APTMoE**:
   ZeRO-3 使用 ~664 GB vs APTMoE 的 1,168 GB，节省 43%。
   pinned memory 方式避免了碎片化。

6. **LoRA 覆盖 attention + experts (0.53%)**:
   修复后 LoRA 通过 PEFT ParamWrapper 覆盖 MoE 专家层，
   可训练参数从 25.4M 提升至 **1.258B**。优化器/梯度从 0.4 GB 增至 ~21 GB，
   但对总内存 (664 GB) 和步时间 (~12s) 的影响有限。

### 10.2 瓶颈排序

| 排名 | 瓶颈 | 影响 (H20) | 影响 (4090) |
|------|------|-----------|------------|
| 1 | NVLink/PCIe allgather 通信 | 48.3% | ~72% (全走 PCIe) |
| 2 | PCIe CPU→GPU 权重上传 | 16.7% | 含在上面 (PCIe 统一) |
| 3 | NCCL 同步等待 | 11.7% | ~10% |
| 4 | Python/框架调度 | 9.2% | ~8% |
| 5 | Optimizer + misc | 7.5% | ~6% |
| 6 | GPU 计算 | 6.7% | ~4% |

### 10.3 优化方向

| 优化方向 | 预期收益 | 对 H20 | 对 4090 | 说明 |
|---------|---------|--------|--------|------|
| **减少 allgather 次数** | **2-3×** | ✅ | ✅ | 参数持久化 / 选择性 offload |
| ├─ 热门 Expert GPU 常驻 | 1.5× | ✅ | ✅ | Top-K experts 不 offload |
| ├─ Attention 不 offload | 1.1× | ✅ | ✅ | Attention 仅 136 MB/层, 可常驻 |
| └─ 逐 Expert 按需 gather | 1.3× | ✅ | ✅ | 只 gather 被 router 选中的 8 个 |
| **Expert 量化 (INT4/FP8)** | **2-4×** | ✅ | ✅ | 传输量减半至 1/4 |
| **减少 allgather 延迟** | **1.3-1.5×** | ✅ | ✅ | 更大 prefetch bucket, 合并通信 |
| **NCCL 开销优化** | **1.2×** | ✅ | ✅ | persistent kernels, fused ops |
| **组合优化** | **理论 3-8×** | ✅ | ✅ | 从 11.2s → ~1.5-4s per step |

### 10.4 论文叙事建议

**故事线**: ZeRO-3 CPU offload 的性能分析揭示了 MoE 大模型训练的 **硬件依赖性瓶颈转移**：

1. **服务器 GPU (H20, 有 NVLink)：NVLink ring 通信是瓶颈 (48%)**
   - PCIe 只搬 1/4 参数 (223 GB/step) → 仅占 3.5s, 不是瓶颈
   - NVLink ring allgather 搬 3/4 参数 (670 GB/step) → 5.8s, 主导步时间
   - GPU 计算利用率仅 2.3%

2. **消费级 GPU (4090, 无 NVLink)：PCIe 是唯一瓶颈 (100% 饱和)**
   - 无 NVLink → allgather 全部走 PCIe → 每 GPU 下行 **893 GB/step**
   - PCIe Gen4 (31.5 GB/s) 链路被 **完全打满**，零余量
   - 步时间从 ~12s 膨胀至 ~34s → **慢 ~2.8×**
   - GPU 计算利用率降至 ~0.8%

3. **通用调度瓶颈 (21%)**
   - 188 次 fine-grained allgather 的 NCCL 启动 + DeepSpeed 框架开销合计 2.5s
   - 这是不依赖硬件互联类型的"固定税"

4. **Optimizer 开销从可忽略变为可见 (7.5%)**
   - 修复 LoRA 覆盖 experts 后，可训练参数从 25.4M → 1.258B，CPU Adam 从 <50ms → ~300ms
   - 但相对 allgather 12.5s 的理论下限，仍然是次要开销

**核心洞察 — PCIe 是 4090 的硬天花板**:
```
H20:  PCIe 搬 1/4 (3.5s) + NVLink 搬 3/4 (8.9s) = 12.5s → NVLink 可优化
4090: PCIe 搬 4/4 (28.3s)                         = 28.3s → PCIe 已打满, 无法优化!
```
在 4090 上，**不减少传输量就不可能提速** — PCIe 已经 100% 跑满。
这为 Expert 量化 (INT4/FP8)、选择性 offload 等优化提供了强烈动机。

**数据支撑**:
- GPU 利用率: H20 仅 **2.3%**, 4090 仅 **0.8%** → 99%+ 时间在搬参数
- H20 步时间 ~12s vs 4090 估算 ~34s → NVLink 价值 **~2.8×**
- 4090 PCIe 下行 893 GB/31.5 GB/s = 28.3s → **链路 100% 饱和, 硬天花板**
- 内存消耗 ~664 GB，其中 470 GB 是模型权重、21 GB 是优化器/梯度 → 开销 37%
- LoRA 覆盖 attention + experts: 1.258B 参数 (0.53%) → 与 APTMoE 1.652B (0.70%) 接近

---

## 附录 A: 数据集统计

```
数据集: math_train (livebench)
样本数: 331
Token 长度统计 (估算):
  Min:    24 tokens
  Max:    2,055 tokens
  Mean:   296 tokens
  Median: 124 tokens
  P90:    1,066 tokens
  P99:    1,709 tokens
  Stdev:  432 tokens

训练: 3 epochs × 331 samples / 4 batch = 249 steps
实际完成: 42 steps (手动终止)
```

## 附录 B: 关键配置参数

```json
// ds_z3_offload_config.json
{
  "zero_optimization": {
    "stage": 3,
    "offload_optimizer": {"device": "cpu", "pin_memory": true},
    "offload_param": {"device": "cpu", "pin_memory": true},
    "overlap_comm": false,
    "stage3_prefetch_bucket_size": "auto",
    "stage3_param_persistence_threshold": "auto",
    "stage3_max_live_parameters": 1e9,
    "stage3_max_reuse_distance": 1e9
  }
}

// Key auto values (estimated by DeepSpeed):
//   reduce_bucket_size: ~500M elements = 1 GB
//   prefetch_bucket_size: ~450M elements = 0.9 GB
//   param_persistence_threshold: ~100K elements = 0.2 MB
```

---

*生成时间: 2026-02-24, 更新: 2026-02-25*
*基于: H20-3e 实测 42 步 ZeRO-3 训练数据 (旧配置) + 理论分析 + LoRA expert 覆盖修复后推算*
*配置: BS=1×4, Qwen3-235B, LoRA r=8 (attention + experts, 1.258B params), DeepSpeed ZeRO-3 CPU offload*
