# DeepSpeed ZeRO-3 Offload 深度 Profile：Qwen3-235B（4090 实测）

> **目标**：基于 4090 机器上的实际运行测试，量化 Qwen3-235B 在 DeepSpeed ZeRO-3 CPU offload 架构下的性能瓶颈、步时间构成、利用率与内存开销。
>
> **方法**：遵循 `zero3_profile_methodology.md`，并参考 `zero3_qwen3_235b_profile.md` 的章节结构与计算口径。
>
> **实测对象**：`/mnt/data2/models/Qwen3-235B-A22B-Instruct-2507`，4× RTX 4090（无 NVLink）。

---

## 0. 需求与思考记录

### 0.1 本文回答的问题

1. 4090（无 NVLink）上 ZeRO-3 offload 的实际步时间是多少？
2. 瓶颈主要在 PCIe、NCCL 同步、框架调度还是 GPU 计算？
3. 实际 token/s、GPU/CPU/PCIe 利用率如何？
4. CPU 内存与 GPU 显存分别用在什么地方？
5. 与此前 H20（有 NVLink）测试相比，差异量级有多大？

### 0.2 关键前提（与 H20 的根本差异）

- **4090 无 NVLink**，`nvidia-smi topo -m` 未出现 `NV*`，GPU 间 allgather 只能走 PCIe。
- 在 ZeRO-3 allgather 中，每个 GPU 每次都需要得到整层全量参数；无 NVLink 时，这部分流量会压到 PCIe。
- 因此 4090 的瓶颈从 H20 的 **NVLink ring** 转为 **PCIe（唯一通道）**。

---

## 1. 运行环境（4090 机器实测）

### 1.1 硬件

| 组件 | 规格（实测） |
|------|-------------|
| GPU | 8 × NVIDIA GeForce RTX 4090 |
| 本次使用 | GPUs `4,5,6,7`（同 NUMA 组） |
| 单卡显存 | 49,140 MiB（约 48 GiB，`nvidia-smi` 显示） |
| 架构 | Ada Lovelace |
| PCIe（GPU Device Max） | Gen4 x16 |
| PCIe（Host Max） | Gen5 |
| NVLink | 无 |
| CPU | 2 × Intel Xeon Platinum 8488C |
| CPU 线程 | 192 线程 |
| 内存 | 2.0 TiB |

> 注：`nvidia-smi -q` 在空闲时显示 `PCIe Generation Current = 1`（链路降速省电），训练负载时会提升到更高代际；硬件能力上限以 `Device Max = Gen4` 为准。

### 1.2 GPU 拓扑（实测）

`nvidia-smi topo -m` 显示：

- `GPU4-7` 之间为 `NODE`
- 不存在 `NV#` 标记
- 说明 **无 NVLink**，GPU 间通信走 PCIe（并跨主桥/NUMA 互联）

### 1.3 训练配置（本次测试）

```yaml
模型: /mnt/data2/models/Qwen3-235B-A22B-Instruct-2507
DeepSpeed: ZeRO Stage 3, CPU offload (param + optimizer)
LoRA: rank=8, lora_target=all（**实测日志显示仅 attention 生效，trainable=25.4M**）
BS=1 per device × 4 GPUs = total BS 4
cutoff_len: 4096
gradient_checkpointing: enabled (默认)
max_steps: 50
```

### 1.4 运行命令（实测）

```bash
cd /home/lpl/zero-baseline/LLaMA-Factory-KT
export PATH=/mnt/data/lpl/anaconda3/envs/llama/bin:$PATH
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=/usr/local/cuda-12.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH
CUDA_VISIBLE_DEVICES=4,5,6,7 FORCE_TORCHRUN=1 \
  /mnt/data/lpl/anaconda3/envs/llama/bin/llamafactory-cli train \
  examples/train_lora/qwen3_235b_lora_sft_ds3_4090_profile.yaml
```

监控同时开启：

- `mem_monitor.log`（每 10s）
- `nvidia-smi dmon -s tump -d 10`（PCIe Rx/Tx、SM%、显存%、功耗）
- `mpstat 10`（系统 CPU 利用率）

---

## 2. 模型架构参数（Qwen3-235B-A22B）

### 2.1 全模型（配置 + safetensors index）

| 参数 | 值 |
|------|---|
| 总参数（官方命名） | 235B 级 |
| `model.safetensors.index.json` 总体积 | `470,187,269,120` bytes（BF16 权重总大小） |
| 层数 | 94 |
| 专家数 / 层 | 128 |
| 每 token 激活专家数 | 8（A22B） |
| hidden size | 4096 |
| `moe_intermediate_size` | 1536 |
| 注意力头数 / KV 头数 | 64 / 4 |
| `head_dim` | 128 |

### 2.2 每层参数量（用于传输量估算）

按 `config.json` 计算（attention + experts + router + norm）：

```text
Attention total   = 71,303,168 params
Experts total     = 2,415,919,104 params   (128 experts)
Router + norms    = 532,480 params
Per-layer total   = 2,487,754,752 params
BF16 bytes/layer  = 4,975,509,504 bytes
```

换算：

- **约 4.98 GB（十进制）/层**
- **约 4.63 GiB（二进制）/层**

> 参考 H20 文档中使用了 `~4.75 GB/层` 的近似值。本文在 4090 传输估算里优先使用上面的**实算值**，并在对比时注明口径差异。

### 2.3 LoRA 参数量（本次实测 vs 理论全覆盖）

本次 `run2` 训练日志显示：

```text
Number of trainable parameters = 25,411,584
```

这对应 **attention-only LoRA**（q/k/v/o_proj），未包含 MoE experts ParamWrapper。

| 项目 | 值 |
|------|---|
| 本次实测 trainable（attention-only） | **25,411,584 (~25.4M)** |
| 若 attention + experts 全覆盖（参考文档公式） | 1,257,488,384 (~1.258B) |
| 差异 | 49.5× |

> 本文性能实测结果以 **25.4M trainable** 的实际运行日志为准；与 H20 参考文档中的“修复后 1.258B LoRA”对比时会单独说明口径差异。

---

## 3. ZeRO-3 Offload 数据流分析（4090 版本）

### 3.1 参数分布（4 GPU）

ZeRO-3 下每个 rank 在 CPU pinned memory 持有全模型约 `1/4` 分片，训练时按模块/层进行 allgather。

### 3.2 4090（无 NVLink）每层 allgather 关键差异

设：

- `N = 4`（GPU 数）
- `S = 4.98 GB`（每层 BF16 参数，十进制近似）
- `G = 2`（gradient checkpointing 下每层 forward/backward 共 2 次 allgather）
- `L = 94`

则每次 allgather（per GPU）：

- 本地 CPU → GPU 上传：`S / N`
- 其余 `3/4` 分片通过 ring 交换获得
- **无 NVLink 时 ring 也走 PCIe**，因此总 PCIe 下行近似等于整层 `S`

### 3.3 每步参数传输量（4090，无 NVLink）

allgather 次数：

```text
L × G = 94 × 2 = 188
```

每 GPU 每步：

```text
PCIe 下行（近似）= 188 × S ≈ 188 × 4.98 = 935.4 GB
```

若按 `PCIe Gen4 x16` 理论单向 `31.5 GB/s`：

```text
纯 PCIe 下行时间下界 ≈ 935.4 / 31.5 = 29.7 s / step
```

这也是 4090 上步时间的大头来源。

---

## 4. 理论 Token/s 上限（4090，PCIe 主导）

### 4.1 理论下界（传输）

在无 NVLink 的 4090 上，短序列与中长序列都高度受 PCIe 限制：

```text
step_time ≈ PCIe_transfer + NCCL_sync + framework_overhead + compute + optimizer

其中 PCIe_transfer 是主导项
```

### 4.2 计算时间量级（用于说明“为何被掩盖”）

活跃参数近似按 A22B：

```text
FLOPs/token ≈ 2 × 22.2B
gradient checkpointing 等效 ≈ 3× forward
Per GPU per step FLOPs ≈ 3 × 2 × 22.2B × seq_len
```

相较于 `~30s` 级别的 PCIe 传输下界，短序列和中等序列的 GPU 计算时间通常被掩盖。

### 4.3 本文实测关注点

本文不只给理论估算，还会基于实际跑步数据给出：

- 稳态步时间（去 warmup）
- 平均/中位 tokens/step 与 token/s
- PCIe Rx/Tx、SM%、CPU busy 实测统计
- 内存时间线与稳态占用

---

## 5. Per-Step 开销拆解（4090 实测）

### 5.1 实测步时间

基于 `trainer_log.jsonl`（42 步，`logging_steps=1`）：

- **Step 1（warmup）**：`85 s`（`elapsed_time: 0:01:25`）
- **Steps 2-42**（41 个间隔）：平均 **53.41 s/step**
- **稳态统计（去除前 5 个间隔，对应 steps 7-42）**：
  - **平均：53.31 s/step**
  - P50：`53 s`
  - P95：`55 s`
  - 范围：`52-56 s`

关键时间点（绝对时间）：

| 事件 | 时间 |
|------|------|
| `Shard-by-shard loading START` | 2026-02-25 06:08:27 |
| `Shard-by-shard loading DONE` | 2026-02-25 06:24:13 |
| `***** Running training *****` | 2026-02-25 06:24:25 |
| Step 42 完成（按 `trainer_log.jsonl`） | `06:24:25 + 0:37:55` ≈ **07:02:20** |
| 训练收尾完成（含保存） | **2026-02-25 07:04:59** |

> `trainer_log.jsonl` 末尾有两条 `current_steps=42`：第一条是最后一步完成时间（`0:37:55`），第二条是训练收尾总耗时（`0:40:34`，含 checkpoint 保存与统计落盘）。

### 5.2 步时间拆解（理论值 + 残差法）

先给出两个口径（都很有用）：

1. **理论下界（理想 PCIe payload）**  
   使用实算每层参数体积 `S=4.9755 GB`（十进制）：
   ```text
   per-step PCIe 下行 payload ≈ 188 × S = 935.4 GB
   理想 Gen4 x16 下行时间下界 ≈ 935.4 / 31.5 = 29.7 s
   ```

2. **实测 payload 吞吐回推（基于 dmon steady）**  
   训练稳态窗口中，4 卡平均 `rxpci ≈ 18.15 GB/s`（每卡）：
   ```text
   等效 payload 时间 ≈ 935.4 / 18.15 = 51.5 s
   ```

与实测稳态步时间对比（`53.31 s`）：

| 项目 | 时间 | 占稳态步时间 |
|------|------|-------------|
| 理想 PCIe payload 下界 | **29.7 s** | **55.7%** |
| dmon 回推 payload 时间（更贴近实测） | **~51.5 s** | **96.6%** |
| 其余开销（NCCL/DS/Python 调度 + 间隙 + 统计误差） | **~1.8 s** | **~3.4%** |

补充说明（数量级）：

- GPU 计算时间（按 avg seq≈296、A22B、4090 BF16 165 TFLOPS、40% kernel 效率）约 **0.6 s/step** 量级，且大量与通信重叠。
- 本次实测为 **attention-only LoRA（25.4M）**，CPU optimizer 开销远小于参考文档中的 1.258B LoRA 场景。

> 这组结果说明：4090 上确实是 **PCIe/通信主导**，但不是“理论带宽 100% payload 打满”，而是“**有效 payload 吞吐只有约 58% 理论值** + 大量细粒度通信/调度损耗”。

### 5.3 开销可视化

```text
稳态步时间 ~53.3s 的直观拆解（4090, attention-only LoRA）:

PCIe payload（dmon 回推） ████████████████████████████████████████████████████████  ~96.6%
其它开销（同步/调度等）  ██                                                  ~3.4%

理论 PCIe payload 下界    ████████████████████████████████                      ~55.7%
（与实测差值即为有效吞吐损失 + 协议/调度开销）
```

---

## 6. 实际工程指标（4090 实测）

### 6.1 Token/s

`trainer_state.json` 的 `num_input_tokens_seen` 为 `0`（该配置下未启用 token 计数），因此本文沿用参考文档的 `math_train` 数据集长度统计（Mean=296, Median=124）进行 token/s 换算。

基于稳态步时间 `53.31 s/step`：

| 指标 | 值 | 说明 |
|------|---|------|
| 稳态步时间（steps 7-42） | **53.31 s** | `trainer_log.jsonl` 间隔统计 |
| 稳态 steps/s | **0.01876** | `1 / 53.31` |
| 平均 tokens/step（参考均值 296） | **1,184** | `4 × 296` |
| 中位 tokens/step（参考中位 124） | **496** | `4 × 124` |
| **平均 token/s（估算）** | **22.2** | `1184 / 53.31` |
| **中位 token/s（估算）** | **9.3** | `496 / 53.31` |

> 这部分 token/s 基于同数据集统计口径换算，适合与 H20 参考文档做横向对比；不是基于 `num_input_tokens_seen` 的精确累计值（该字段为 0）。

### 6.2 关键日志指标

```text
训练配置（实测）:
  GPUs: 4,5,6,7 (4 × RTX 4090, no NVLink)
  max_steps: 42
  trainable params: 25,411,584 (attention-only LoRA)

启动与加载:
  Dataset load log:       2026-02-25 06:06:19
  Shard loading start:    2026-02-25 06:08:27
  Shard loading done:     2026-02-25 06:24:13   (15m46s)
  Running training start: 2026-02-25 06:24:25
  Step 1 complete:        ~06:25:50 (85s warmup)

训练阶段:
  Step 42 complete:       ~07:02:20 (elapsed 0:37:55)
  Training completed log: 2026-02-25 07:04:59 (elapsed 0:40:34, 含保存/收尾)
  收尾额外耗时:           ~159s (40:34 - 37:55)

DeepSpeed / CPUAdam:
  CPUAdam JIT load（4 rank）: ~2.72s / 2.84s / 2.93s / 3.32s
  本次需显式设置 CUDA_HOME=/usr/local/cuda-12.8（否则会因 CUDA 13.0 vs torch 12.8 mismatch 失败）
```

DeepSpeed auto bucket（train.log 实测打印）：

- `reduce_bucket_size = 16,777,216`
- `stage3_prefetch_bucket_size = 15,099,494`
- `stage3_param_persistence_threshold = 40,960`

---

## 7. 利用率分析（4090 实测 + 计算）

### 7.1 GPU 计算利用率

按 A22B 活跃参数近似（`22.2B`）和参考数据集均长 `296 tokens` 估算：

```text
Per GPU FLOPs/step ≈ 3 × 2 × 22.2B × 296 = 39.43 TFLOPs
稳态步时间 ≈ 53.31 s
Per GPU 实际算力吞吐 ≈ 39.43 / 53.31 = 0.74 TFLOPS/s

RTX 4090 BF16 peak (spec): 165 TFLOPS
GPU 计算利用率 ≈ 0.74 / 165 = 0.45%
```

> 注意：这和 `nvidia-smi dmon` 的 `sm% ≈ 99%` 不矛盾。`sm%` 表示 GPU 在忙（包括通信 kernel、等待/同步相关活动），不等价于 Tensor Core 算力利用率。

### 7.2 PCIe 带宽利用率

使用稳态步时间 `53.31s` 与实算 payload（`935.4 GB/step/GPU`）：

```text
理论下行 payload 速率 ≈ 935.4 / 53.31 = 17.56 GB/s
相对 Gen4 x16 单向理论 31.5 GB/s => 55.7%
```

结合 `dmon`（稳态短窗口，37 个采样点，5s 间隔）实测：

- 平均 `rxpci`（4 卡均值）≈ **18.15 GB/s / GPU**
- 平均 `txpci`（4 卡均值）≈ **16.21 GB/s / GPU**
- 单向下行利用率（按 `rxpci`）≈ **57.6%**（`18.15 / 31.5`）
- 双向合计利用率（按 `rx+tx` vs `63 GB/s` duplex）≈ **54.5%**

> 结论：4090 的瓶颈仍然是 PCIe/通信路径，但实测显示 **有效 payload 吞吐只有理论值的约 55-58%**，大量损耗来自细粒度 allgather 的协议开销、同步等待与调度气泡。

### 7.3 CPU 利用率（mpstat 实测）

`mpstat` 实测：

| 口径 | busy%（192 线程总口径） |
|------|------------------------|
| 全流程平均（加载+训练+保存） | **3.77%** |
| 稳态短窗口平均（5s×40 个样本） | **4.23%** |
| 稳态短窗口 P50 | **4.21%** |
| 稳态短窗口 P95 | **4.41%** |

换算成逻辑核等效占用：

- `4.23% × 192 ≈ 8.1` 逻辑核

> 在本次 **attention-only LoRA（25.4M）** 场景下，CPU optimizer 并不是主瓶颈；CPU 使用率远低于 PCIe/通信瓶颈对应的等待时间占比。

### 7.4 GPU/PCIe dmon 观测

稳态短窗口（从 `2026-02-25 06:25:52` 开始，约 37 个 dmon 采样点，5s 间隔）：

4 卡平均（按每卡均值再取平均）：

| 指标 | 值 |
|------|---|
| `rxpci` | **18,147.6 MB/s** (~18.15 GB/s) |
| `txpci` | **16,207.1 MB/s** (~16.21 GB/s) |
| `sm%` | **99.24%** |
| `mem%` | **7.81%** |
| `fb` 占用 | **16,155.9 MB** (~15.78 GiB) |
| 功耗 | **128.4 W** |
| 温度 | **42.9 °C** |

单卡范围（稳态窗口内）：

- PCIe `rxpci` 均值约 `17.4-19.1 GB/s`
- PCIe `txpci` 均值约 `15.7-17.1 GB/s`
- `fb` 占用稳定在约 `16.08-16.35 GB`
- 功耗大致 `124-145W` 峰值区间（单卡 max 约 `154W`）

> 观测特征很一致：**PCIe 持续高吞吐 + SM 几乎常亮 + 显存控制器占用不高**，符合 ZeRO-3 allgather/通信主导而非密集算力主导的模式。

---

## 8. 与 H20（参考文档）对比

> H20 侧数值来自 `zero3_qwen3_235b_profile.md`（含“实测+推算”口径），本节用于展示 4090 的瓶颈转移与量级差异。

本次 4090 实测和 H20 参考文档（修复后推算口径）对比：

| 指标 | H20（参考文档） | 4090（本次实测） | 对比 |
|------|----------------|------------------|------|
| LoRA trainable params | ~1.258B（attention+experts，推算口径） | **25.4M（attention-only，实测）** | 口径不同（本次更轻） |
| 稳态步时间 | ~12.0 s | **53.3 s** | **4090 慢 ~4.4×** |
| 平均 token/s（按 mean=296） | ~99 | **~22.2** | **0.22×** |
| GPU 计算利用率（估算） | ~2.2% | **~0.45%** | 更低 |
| PCIe 利用率 | ~29.7%（H20, PCIe只搬1/4） | **~57.6%（4090, dmon下行实测）** | 4090 更依赖 PCIe |
| NVLink 利用率 | ~74.4% | N/A | 4090 无 NVLink |
| GPU 显存占用（实测/观测） | H20 文档估算 ~6-8 GB | **~16.1 GB（dmon steady）** | 4090 实测显著更高但仍可运行 |

额外对比（与 H20 文档中 4090 估算）：

- H20 文档对 4090 的步时间估算约 **34s**
- 本次实测稳态约 **53.3s**
- **实测约为估算的 1.57×**

这说明仅用“理论 PCIe 带宽 + 数据量”估算会显著低估 4090 的实际耗时，原因主要是：

1. 细粒度 allgather 的协议/同步开销
2. 有效 PCIe payload 吞吐低于理论值（约 55-58%）
3. 无 NVLink 下更多时间花在通信效率损失而不是纯 payload 传输

---

## 9. CPU 内存与 GPU 显存拆解（4090 实测）

### 9.1 系统内存时间线（mem_monitor）

本次按方法论文档的 `free -g | awk '/Mem:/{print $3}'` 口径记录（每 10s）：

| 阶段 | `free -g used`（GB） | 说明 |
|------|---------------------|------|
| 启动基线 | ~31 | 训练前 |
| 加载阶段 | ~44-72（均值 ~61.7） | `zero.Init + shard loading` |
| 训练步阶段 | ~50-57（均值 ~52.4） | `06:25:52` 到 `07:02:20` |
| 训练后保存 | ~57-58 | checkpoint 保存 |

> **重要口径说明**：这个 `free -g used` 指标会把大量文件缓存归到 `buff/cache`，因此会显著低估实际内存压力。  
> 对 ZeRO-3 这类大模型加载，建议同时参考 `train.log` 中的 `[DIAG] Avail/Total`（见下一节）。

### 9.2 CPU 内存用途拆解

使用 `train.log` 中 rank0 的 `[DIAG]` 日志（`used_effective = Total - MemAvailable`）：

| 指标 | 值 |
|------|---|
| 加载阶段 `used_effective` 均值 | **~646.3 GB** |
| 加载阶段 `used_effective` P95 | **~663.5 GB** |
| 加载阶段峰值 | **~667.6 GB**（06:19:16） |
| `Shard loading START` 附近 | **~640.1 GB** |
| `Shard loading DONE` 附近 | **~655.0 GB** |

这比 `free -g used` 更符合实际 ZeRO-3 235B 的内存规模。

本次 **attention-only LoRA（25.4M）** 的已知固定项（十进制近似）：

| 组件 | 体积（GB） |
|------|-----------|
| 基模型 BF16 权重（index 实测） | **470.19** |
| LoRA 权重 BF16（25.4M） | ~0.05 |
| LoRA master FP32 | ~0.10 |
| Adam m+v FP32 | ~0.20 |
| Grad FP32 | ~0.10 |
| **固定项合计** | **~470.65** |

对比 `used_effective ~640-668 GB`，可见额外 `~170-197 GB` 主要来自：

- ZeRO-3/pinned memory 管理开销
- PyTorch/DeepSpeed 元数据与运行时缓冲
- 文件缓存（模型分片读取）
- NCCL/通信缓冲与 DataLoader/Python 进程开销

> 即使在 attention-only LoRA（优化器很小）场景下，**运行时开销仍然远大于 LoRA 优化器本身**。

### 9.3 GPU 显存观测（dmon + DeepSpeed 日志）

`dmon` 稳态短窗口显示（4 卡）：

- `fb` 占用均值约 **16.16 GB / 卡**
- 4 卡中最高稳定值约 **16.35 GB / 卡**
- 远低于 4090 的 24 GB 容量，仍有约 **7.5 GB** 余量

与参考文档的“~7-8GB 估算”相比，本次实测更高，说明真实运行中的缓冲/allocator/NCCL/ZeRO bucket 开销不应低估。

同时，从 `dmon` 可见：

- `sm%` ≈ 99%
- `mem%`（显存控制器占用）仅 ~8%
- 功耗仅 ~128W 平均（远低于 450W 上限）

这进一步支持“**GPU 在忙通信与等待，而不是做高强度矩阵计算**”的判断。

---

## 10. 关键发现与优化方向（4090 结论）

### 10.1 关键发现（本次 4090 实测）

1. **4090 稳态步时间约 53.3s，显著慢于先前 34s 估算**
   - 比 H20 文档中的 4090 估算慢约 **1.57×**
   - 比 H20 参考文档（~12s）慢约 **4.4×**

2. **PCIe/通信路径是绝对主导，但有效 payload 吞吐仅约理论值 55-58%**
   - 理想下界（按 31.5 GB/s）约 **29.7s**
   - 实际稳态 **53.3s**
   - `dmon` 回推 payload 时间约 **51.5s**

3. **`sm%≈99%` 不代表算力利用率高**
   - 估算 GPU 计算利用率仅 **~0.45%**（avg seq≈296）
   - 说明 dmon 的 `sm%` 在 ZeRO-3 offload 场景里会把通信/等待阶段也计作“忙”

4. **内存侧实际压力依然很高（~640-668GB），但 `free -g used` 会低估**
   - `mem_monitor`（`free -g used`）只看到 `~50-70GB`
   - `train.log` `[DIAG] MemTotal-MemAvailable` 显示加载阶段峰值 **~667.6GB**

5. **本次实际运行是 attention-only LoRA（25.4M），不是 experts+attention 的 1.258B LoRA**
   - 日志没有 `Found expert parameters`
   - `Number of trainable parameters = 25,411,584`
   - 因此本报告的步时间/CPU优化器结论对应的是 **更轻的 LoRA 口径**

### 10.2 优化方向（面向 4090）

1. **优先减少 allgather/PCIe 传输量**
   - 热门 expert 常驻 GPU
   - attention 常驻或选择性 offload
   - expert 量化（INT4/FP8）降低传输字节数

2. **提高通信有效吞吐（而不是只看理论带宽）**
   - 减少细粒度 allgather 次数
   - 合并通信、增大 bucket、降低 launch 次数
   - 评估 `overlap_comm` 与预取策略

3. **修复/验证 expert LoRA 发现链路**
   - 本次日志未发现 `expert_parameters`
   - 在继续做论文级对比前，需先确认 `find_all_expert_parameters()` 对当前 Qwen3-MoE 路径是否生效

4. **用 Nsight / nsys 做下一步精细 profile**
   - 目前已经可确认“通信主导”
   - 下一步要分清 `PCIe payload`、`NCCL wait`、`DeepSpeed hook` 的具体占比

---

## 附录 A: 数据集统计（沿用同配置口径）

> `math_train (livebench)`，统计口径与参考文档一致。本次 `trainer_state.json` 中 `num_input_tokens_seen = 0`，因此 token/s 换算使用下列参考统计值。

参考值（来自 H20 报告，同数据集同模板配置）：

```text
样本数: 331
Mean:   296 tokens
Median: 124 tokens
P90:    1066 tokens
P99:    1709 tokens
```

---

## 附录 B: 关键配置文件

- 训练 YAML：`examples/train_lora/qwen3_235b_lora_sft_ds3_4090_profile.yaml`
- DeepSpeed：`examples/deepspeed/ds_z3_offload_config.json`
- 方法论：`data/claude_docs/03_paper_estimate/zero3_profile_methodology.md`
- H20 参考：`data/claude_docs/03_paper_estimate/zero3_qwen3_235b_profile.md`

---

*生成时间: 2026-02-25（4090 实测）*
