# ZeRO-3 Profile 文档生成方法论

> **目的**：说明 `zero3_qwen3_235b_profile.md` 的数据来源、计算方法和推导逻辑，
> 便于在新硬件（如 4090 机器）上复现同类分析。

---

## 1. 数据采集

### 1.1 需要收集的原始数据

| 编号 | 数据 | 采集方式 | 用于 |
|------|------|---------|------|
| D1 | GPU 硬件参数 | `nvidia-smi -q` | 算力、显存、PCIe 版本 |
| D2 | GPU 拓扑 | `nvidia-smi topo -m` | NVLink 有无、PCIe 拓扑 |
| D3 | CPU / 内存 | `lscpu`, `free -h` | CPU 核数、内存容量 |
| D4 | 训练日志 | `train.log` (stdout) | 步时间、LoRA 参数、DeepSpeed 信息 |
| D5 | 训练损失 | `trainer_log.jsonl` | 每步 loss 和 step time |
| D6 | 系统内存监控 | 自定义脚本 (见 1.2) | 内存时间线 |
| D7 | 模型配置 | `config.json` | 层数、专家数、hidden_size 等 |
| D8 | DeepSpeed 配置 | `ds_z3_offload_config.json` | ZeRO stage、offload 策略 |
| D9 | 训练 YAML | `qwen3_235b_lora_sft_ds3.yaml` | LoRA rank、BS、cutoff_len 等 |
| D10 | safetensors index | `model.safetensors.index.json` | 参数命名、分片方式 |

### 1.2 系统内存监控脚本

训练前启动，每 10 秒记录一次系统内存：

```bash
# mem_monitor.sh — 在训练前启动
while true; do
  echo "$(date '+%H:%M:%S') $(free -g | awk '/Mem:/{print $3}')" >> mem_monitor.log
  sleep 10
done
```

启动方式：
```bash
bash mem_monitor.sh &
MEM_PID=$!
# ... 运行训练 ...
kill $MEM_PID
```

### 1.3 训练运行命令

```bash
# H20 机器 (4 GPUs)
CUDA_VISIBLE_DEVICES=4,5,6,7 FORCE_TORCHRUN=1 \
  llamafactory-cli train examples/train_lora/qwen3_235b_lora_sft_ds3.yaml \
  2>&1 | tee /path/to/output/train.log

# 4090 机器 (调整 CUDA_VISIBLE_DEVICES 为实际 GPU 编号)
CUDA_VISIBLE_DEVICES=0,1,2,3 FORCE_TORCHRUN=1 \
  llamafactory-cli train examples/train_lora/qwen3_235b_lora_sft_ds3.yaml \
  2>&1 | tee /path/to/output/train.log
```

建议至少跑 **40-50 步**（前 5 步 warmup，后 35+ 步取稳态均值）。
可以在 YAML 中设 `max_steps: 50` 限制步数。

---

## 2. 硬件参数表（需在目标机器上采集）

### 2.1 采集命令

```bash
# GPU 基本信息
nvidia-smi -q | grep -E "Product Name|Total FB|PCIe Generation|Link Width"

# GPU 拓扑（关键：判断有无 NVLink）
nvidia-smi topo -m

# CPU
lscpu | grep -E "Model name|CPU\(s\)|Thread|Socket|NUMA"

# 内存
free -h

# PCIe 带宽实测（可选，需 cuda-samples）
# /usr/local/cuda/samples/1_Utilities/bandwidthTest/bandwidthTest
```

### 2.2 关键硬件参数速查

| 参数 | H20 值 | 4090 值 (需填) | 来源 |
|------|--------|---------------|------|
| BF16 TFLOPS | 148 | 165 (spec) | nvidia-smi -q / 官方 spec |
| VRAM | 140 GB | 24 GB | nvidia-smi |
| PCIe 版本 | Gen5 x16 | Gen4 x16 (需确认) | nvidia-smi -q |
| PCIe 理论单向 | 63 GB/s | 31.5 GB/s | Gen5=63, Gen4=31.5, Gen3=15.75 |
| NVLink | 有 (NV18) | 无 (需确认) | nvidia-smi topo -m |
| NVLink 带宽/对 | ~75 GB/s | N/A | 18 links / 7 peers × 25 GB/s/link |

> **4090 关键确认**：`nvidia-smi topo -m` 中 GPU 间是否显示 `NV*`。
> 若全是 `SYS` 或 `PHB` 或 `PIX`，则无 NVLink，allgather 全走 PCIe。

---

## 3. 模型参数计算

### 3.1 基础参数（来自 config.json）

```python
import json
with open("config.json") as f:
    c = json.load(f)

num_layers       = c["num_hidden_layers"]      # 94
num_experts      = c["num_experts"]             # 128
experts_per_tok  = c["num_experts_per_tok"]     # 8
hidden_size      = c["hidden_size"]             # 4096
moe_inter_size   = c["moe_intermediate_size"]   # 1536
num_heads        = c["num_attention_heads"]      # 64
num_kv_heads     = c["num_key_value_heads"]      # 4
head_dim         = hidden_size // num_heads      # 64? 需看 config
vocab_size       = c["vocab_size"]               # 151936
```

### 3.2 每层参数量

```
Attention:
  q_proj: hidden × (num_heads × head_dim)         = 4096 × 8192 = 33,554,432
  k_proj: hidden × (num_kv_heads × head_dim)      = 4096 × 512  =  2,097,152
  v_proj: same as k_proj                                         =  2,097,152
  o_proj: (num_heads × head_dim) × hidden          = 8192 × 4096 = 33,554,432
  q_norm, k_norm: head_dim × num_heads/kv_heads
  Attention total ≈ 71.3M

Experts (per expert):
  gate_proj: hidden × moe_inter = 4096 × 1536 =  6,291,456
  up_proj:   same                              =  6,291,456
  down_proj: moe_inter × hidden = 1536 × 4096 =  6,291,456
  Per expert: 18,874,368
  128 experts: 2,415,919,104 ≈ 2.416B

Router: num_experts × hidden = 128 × 4096 = 524,288
LayerNorm: 2 × hidden = 8,192

每层 total ≈ 2.488B = 4.75 GB (BF16)
```

### 3.3 LoRA 参数量

```
Attention LoRA (per layer, rank=r):
  每个 Linear(in, out) 的 LoRA: r × in + r × out
  q_proj: r × 4096 + r × 8192 = r × 12288
  k_proj: r × 4096 + r × 512  = r × 4608
  v_proj: same                 = r × 4608
  o_proj: r × 8192 + r × 4096 = r × 12288
  Per layer: r × 33792       (r=8 → 270,336)

Expert LoRA (ParamWrapper, per layer):
  gate_up_proj (E, 2*inter, hidden):
    lora_A: (r×E) × dim_0 = (r×E) × (2*inter)
    lora_B: dim_1 × (r×E) = hidden × (r×E)
    小计: r×E × (2*inter + hidden)
  down_proj (E, hidden, inter):
    lora_A: (r×E) × hidden
    lora_B: inter × (r×E)
    小计: r×E × (hidden + inter)
  Per layer: r × E × (2*inter + hidden + hidden + inter)
           = r × E × (3*inter + 2*hidden)
           = 8 × 128 × (3×1536 + 2×4096)
           = 1024 × 12800
           = 13,107,200

Total per layer: 270,336 + 13,107,200 = 13,377,536
Total model: 13,377,536 × 94 = 1,257,488,384 ≈ 1.258B
```

---

## 4. 传输量计算

### 4.1 核心公式

```
N = GPU 数量
S = 每层参数体积 (BF16 bytes)
G = gradient checkpointing 下 allgather 次数 = 2 (forward + backward)
L = 层数

每次 allgather per GPU:
  PCIe upload (CPU → local GPU): S / N
  Ring receive (from other GPUs):
    有 NVLink:  (N-1)/N × S  via NVLink
    无 NVLink:  (N-1)/N × S  via PCIe (与 upload 共享 PCIe 带宽!)

Per step per GPU:
  allgather 次数 = L × G (+2 for embed/lm_head, 通常可忽略)
  PCIe upload total = allgather次数 × S/N

  有 NVLink:
    NVLink total = allgather次数 × (N-1)/N × S
    PCIe 和 NVLink 是不同链路，可以并行

  无 NVLink (关键差异!):
    所有 allgather 流量都走 PCIe
    每次 allgather 每 GPU 需要下行整层: S (不是 S/N)
    Ring 中每步 GPU 既发又收 S/N，都走 PCIe
    PCIe total download ≈ allgather次数 × S
```

### 4.2 有 NVLink (H20) vs 无 NVLink (4090)

```
S = 4.75 GB, N = 4, allgather次数 = 188

有 NVLink:
  PCIe upload/step/GPU = 188 × 4.75/4 = 223 GB
  NVLink/step/GPU = 188 × 3/4 × 4.75 = 670 GB
  瓶颈: max(PCIe_time, NVLink_ring_time)

无 NVLink:
  每次 allgather, GPU 需要通过 PCIe 获得全部 S:
    1/4 from local CPU (upload)
    3/4 from ring (每个 ring step 通过 PCIe 中转)
  PCIe download/step/GPU = 188 × S = 893 GB
  PCIe 是唯一通道 → 瓶颈
```

### 4.3 时间计算

```
有 NVLink, per layer:
  PCIe time = S/N / PCIe_BW
  Ring time = (N-1) steps × S/N / NVLink_pair_BW
  Layer time = PCIe_time + Ring_time  (串行: 先 upload, 再 ring)

  理想有 prefetch overlap: layer_time × L × G

无 NVLink, per layer:
  Ring 也走 PCIe:
    PCIe upload: S/N / PCIe_BW
    Ring 3 steps: 3 × S/N / PCIe_BW  (每步发 S/N + 收 S/N, 全双工)
  Per layer = (1 + N-1) × S/N / PCIe_BW = S / PCIe_BW

  全 step: L × G × S / PCIe_BW
```

---

## 5. 步时间拆解方法

### 5.1 实测步时间

从 `trainer_log.jsonl` 提取：

```python
import json

times = []
with open("trainer_log.jsonl") as f:
    for line in f:
        entry = json.loads(line)
        if "train_steps_per_second" in entry:
            step_time = 1.0 / entry["train_steps_per_second"]
            times.append(step_time)

# 去掉前 5 步 warmup
steady = times[5:]
avg_step = sum(steady) / len(steady)
print(f"稳态步时间: {avg_step:.2f} s  ({len(steady)} steps)")
```

### 5.2 拆解公式

步时间拆解为 6 个部分，方法论如下：

| 组件 | 计算方式 | 公式 / 来源 |
|------|---------|------------|
| **Allgather 通信** | 理论 + 效率折扣 | `allgather次数 × per_layer_transfer_time × efficiency` |
| **PCIe 权重传输** | 与 allgather 重叠程度估计 | 有 NVLink 时: 未被 overlap 的部分; 无 NVLink 时: 包含在 allgather 中 |
| **NCCL 同步** | per-op 启动开销 × op 数 | `allgather次数 × ~5-8 ms` |
| **Python/框架调度** | per-op 框架开销 × op 数 | `allgather次数 × ~3-5 ms` |
| **GPU 计算** | FLOPs / GPU throughput | `3 × 2 × active_params × seq_len / (TFLOPS × utilization)` |
| **Optimizer** | CPU Adam 基准 | `~0.24 ms/M params` (经验值, CPU Adam) |

**拆解步骤：**

1. 先算 **GPU 计算时间** (最确定):
   ```
   compute = 3 × 2 × active_params × avg_seq_len / (TFLOPS × 1e12 × efficiency)
   efficiency ≈ 0.3-0.5 (kernel 效率, memory-bound 折扣)
   ```

2. 再算 **Optimizer 时间**:
   ```
   optimizer = trainable_params / 1e6 × 0.24 ms  (CPU Adam 经验值)
   ```

3. 算 **理论传输时间** (公式见 Section 4.3):
   ```
   有 NVLink: transfer = allgather次数 × (PCIe_per_layer + Ring_per_layer)
   无 NVLink: transfer = allgather次数 × S / PCIe_BW
   ```

4. **NCCL 同步 + 框架调度** = 实测步时间 - 计算 - optimizer - 传输
   - 如果差值为正: 说明有额外开销 (NCCL launch, DeepSpeed hooks)
   - 通常占 15-25% (经验值)
   - 按 NCCL:框架 ≈ 60:40 拆分

5. **PCIe 权重传输** (有 NVLink 时):
   - 在 NVLink ring 和 PCIe upload 之间拆分
   - PCIe upload 与 NVLink ring 可以 overlap
   - 未 overlap 部分 ≈ 30% 的 PCIe time (经验值)

### 5.3 无 NVLink 时的简化

在 4090（无 NVLink）上，拆解更简单：

```
步时间 ≈ PCIe_transfer + NCCL_sync + framework_overhead + compute + optimizer

其中:
  PCIe_transfer = allgather次数 × S / PCIe_BW        (绝对主导)
  NCCL_sync     = allgather次数 × ~5 ms               (~15%)
  framework     = allgather次数 × ~3 ms               (~8%)
  compute       = (很小, 被 overlap 掉大部分)
  optimizer     = trainable_params / 1e6 × 0.24 ms
```

> **4090 上的关键简化**：因为 PCIe 是唯一通道且 100% 饱和，
> 传输时间几乎等于 `allgather次数 × S / PCIe_BW`。
> NCCL 同步和框架开销无法被 overlap，是纯加项。

---

## 6. 利用率计算

### 6.1 GPU 计算利用率

```
FLOPs_per_step = 3 × 2 × active_params × batch_size × avg_seq_len
per_GPU = FLOPs_per_step / N_GPUs
per_second = per_GPU / step_time
utilization = per_second / peak_TFLOPS
```

### 6.2 PCIe 利用率

```
有 NVLink:
  PCIe_traffic = allgather次数 × S / N              (仅 1/N)
  utilization = (PCIe_traffic / step_time) / PCIe_BW

无 NVLink:
  PCIe_traffic = allgather次数 × S                   (全量!)
  utilization = (PCIe_traffic / step_time) / PCIe_BW
  (通常接近 100%, 因为 PCIe 已饱和)
```

### 6.3 NVLink 利用率

```
NVLink_traffic = allgather次数 × (N-1)/N × S
utilization = (NVLink_traffic / step_time) / NVLink_pair_BW
```

---

## 7. 内存拆解方法

### 7.1 已知固定分配

```
模型权重 (BF16): total_params × 2 bytes
LoRA 权重 (BF16): lora_params × 2 bytes     (ZeRO-3 也 offload)
LoRA master (FP32): lora_params × 4 bytes
Adam m+v (FP32): lora_params × 8 bytes
梯度 (FP32): lora_params × 4 bytes

已知固定 = 上述之和
```

### 7.2 运行时开销（实测 - 已知固定 - OS 基线）

```
运行时开销 = 实测稳态内存 - 已知固定 - OS基线

构成:
  Python/PyTorch 进程:  N_ranks × 5-7 GB
  参数元数据:           total_params / 1e6 × ~0.2 MB  (经验值)
  DeepSpeed 运行时:     ~15-20 GB
  NCCL 缓冲:           N_ranks × 2-3 GB
  pinned memory 开销:   ~10-15 GB
  数据集:              ~5-8 GB
```

### 7.3 GPU 显存拆解

```
Gathered params:       min(max_live_parameters × 2, S)
Prefetch buffer:       prefetch_bucket_size × 2
LoRA gathered:         lora_per_layer × 2              (ParamWrapper 临时)
Activations:           BS × seq_len × hidden × ~4      (gradient checkpoint, 1 层)
Persistent params:     persistent_count × 2
NCCL buffers:          ~0.5-1 GB
CUDA context:          ~1-2 GB
```

---

## 8. 4090 迁移 Checklist

### 8.1 迁移前确认

- [ ] 确认 GPU 数量和可用 GPU 编号
- [ ] `nvidia-smi topo -m` 确认无 NVLink
- [ ] `nvidia-smi -q` 确认 PCIe 版本 (Gen4 x16?)
- [ ] 确认系统内存 ≥ 700 GB (470 GB 模型 + 运行时开销)
- [ ] 确认 conda 环境 `zero` 可用，transformers/peft/deepspeed 版本一致
- [ ] 确认 LLaMA-Factory-KT 代码含 `find_all_expert_parameters` 修复

### 8.2 预期差异

| 指标 | H20 | 4090 预期 | 原因 |
|------|-----|----------|------|
| 步时间 | ~12s | **~34s** | PCIe 搬 4/4 (无 NVLink) |
| PCIe 利用率 | 30% | **~100%** | PCIe 是唯一通道 |
| GPU 计算利用率 | 2.2% | **~0.8%** | 步时间 3× 但 FLOPS 类似 |
| 系统内存 | ~664 GB | **~664 GB** | 不依赖 GPU 型号 |
| GPU 显存峰值 | ~7 GB | **~7 GB** | ZeRO-3 offload, 与 VRAM 无关 |

### 8.3 实测后需更新的文档位置

`zero3_qwen3_235b_profile.md` 中以下部分需用 4090 实测数据替换/补充：

1. **Section 1.1**: 硬件表 → 补充 4090 实际参数
2. **Section 5**: 步时间拆解 → 用 4090 实测步时间重新拆解
3. **Section 6**: token/s → 用 4090 实测值
4. **Section 7**: 利用率 → 用 4090 实测值计算
5. **Section 8**: 4090 估算 → 替换为实测值，对比估算准确度
6. **Section 9**: 内存 → 用 4090 实测 mem_monitor.log

### 8.4 数据采集脚本（一键运行）

```bash
#!/bin/bash
# collect_profile_data.sh - 在 4090 机器上运行
# 用法: bash collect_profile_data.sh <output_dir> <gpu_ids>
# 示例: bash collect_profile_data.sh /path/to/output 0,1,2,3

OUTPUT_DIR=${1:-./profile_output}
GPU_IDS=${2:-0,1,2,3}
mkdir -p "$OUTPUT_DIR"

echo "=== 采集硬件信息 ==="
nvidia-smi -q > "$OUTPUT_DIR/nvidia_smi_q.txt"
nvidia-smi topo -m > "$OUTPUT_DIR/nvidia_topo.txt"
lscpu > "$OUTPUT_DIR/lscpu.txt"
free -h > "$OUTPUT_DIR/free.txt"

echo "=== 启动内存监控 ==="
(while true; do
  echo "$(date '+%Y-%m-%d %H:%M:%S') $(free -g | awk '/Mem:/{print $3}')"
  sleep 10
done) > "$OUTPUT_DIR/mem_monitor.log" &
MEM_PID=$!

echo "=== 启动训练 (PID: $$, MEM_MONITOR: $MEM_PID) ==="
CUDA_VISIBLE_DEVICES=$GPU_IDS FORCE_TORCHRUN=1 \
  llamafactory-cli train examples/train_lora/qwen3_235b_lora_sft_ds3.yaml \
  2>&1 | tee "$OUTPUT_DIR/train.log"

echo "=== 停止内存监控 ==="
kill $MEM_PID 2>/dev/null

echo "=== 拷贝训练输出 ==="
TRAIN_OUTPUT=$(grep "output_dir" examples/train_lora/qwen3_235b_lora_sft_ds3.yaml | awk '{print $2}')
if [ -f "$TRAIN_OUTPUT/trainer_log.jsonl" ]; then
  cp "$TRAIN_OUTPUT/trainer_log.jsonl" "$OUTPUT_DIR/"
fi

echo "=== 完成! 数据在 $OUTPUT_DIR ==="
ls -la "$OUTPUT_DIR"
```

---

## 9. 关键假设与局限

### 9.1 假设

1. **Ring allgather**: ZeRO-3 使用 NCCL ring allgather，N-1 步，每步传 S/N
2. **PCIe 全双工**: 上下行可同时进行（对 ring 有利）
3. **无 NVLink 时 ring 走 PCIe**: 经 CPU 中转，GPU→PCIe→CPU→PCIe→GPU
4. **Gradient checkpointing**: forward 不存激活，backward 重算 → 每层 2 次 allgather
5. **CPU Adam**: DeepSpeed CPU offload 使用 fused CPU Adam optimizer
6. **lora_dropout = 0**: ParamWrapper 要求 dropout=0

### 9.2 局限

1. **步时间拆解是估算**: 无法直接 profile 各组件时间（需 nsys/nvprof），
   用理论值 + 残差法逆推
2. **overlap 程度不确定**: PCIe 与 NVLink、计算与传输的 overlap 取决于
   DeepSpeed prefetch 调度，无法精确量化
3. **NCCL 同步开销**: per-op 5-8 ms 是经验值，实际取决于 NCCL 版本和拓扑
4. **CPU Adam 时间**: 0.24 ms/M 是经验值，受 CPU 型号和内存带宽影响
5. **内存运行时开销**: 通过差值法估算，无法精确归因每个子项

### 9.3 如何提高精度

- **nsys profile**: `nsys profile --trace=cuda,nvtx,osrt` 可获得精确的 kernel 和通信时间线
- **DeepSpeed flops profiler**: `ds_config` 中开启 `flops_profiler`
- **PCIe 实测带宽**: `cuda-samples/bandwidthTest` 获得实际 PCIe 吞吐
- **torch.cuda.Event**: 在代码中插入 timing event 精确测量各阶段

---

*生成时间: 2026-02-25*
*配套文档: zero3_qwen3_235b_profile.md*
