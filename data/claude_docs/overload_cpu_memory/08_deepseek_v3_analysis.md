# 08 DeepSeek-V3 OOM 分析

## 当前状态

**已实施修复，尚未测试。** 修复基于"双重分配"假设（见下文不确定性说明）。
如果修复无效，本文档包含诊断方法和备选方案的具体实施指引。

## 问题描述

DeepSeek-V3 (671B) 在 2TB 内存、4 GPU 的环境下使用 DeepSpeed ZeRO-3 offload + LoRA 训练时，
在模型**创建阶段**（`deepspeed.zero.Init()`）就 OOM，**从未执行到 shard-by-shard 权重加载**。

与 Qwen3 系列问题完全不同：

| | Qwen3 系列 | DeepSeek-V3 |
|---|---|---|
| OOM 阶段 | Phase 1: 权重加载 | **Phase 0: 模型创建 (`deepspeed.zero.Init()`)** |
| v4 shard-by-shard 方案 | 已解决 | **不覆盖**（Phase 0 在 Phase 1 之前发生） |
| 日志标志 | 有 shard-by-shard DIAG 日志 | 只有 `Before from_pretrained`，没有 `After` |

## 实验数据

### 实验 1：无任何优化
- 停在 **632/967 参数**（layer 40, `experts.down_proj`）
- 系统内存 576GB → **1,856GB** → OOM killed (exitcode -15)
- 日志：`/mnt/data/lpl/ls/zero_dpsk3.log`（旧版）

### 实验 2：`MALLOC_TRIM_THRESHOLD_=0`
- 停在 **697/967 参数**（layer 44, `experts.gate_up_proj`）
- 改善约 10%（多活了 4 个 MoE 层），但仍 OOM
- 日志：`/mnt/data/lpl/ls/zero_dpsk3.log`（当前版本）

### 关键推导

```
全模型 BF16 = 1,376 GB
ZeRO-3 分区后（4 rank 合计）= 1,376 GB（每 rank 344 GB）
理论峰值 = 1,376 GB + 系统开销 ~100GB = ~1,476 GB → 应该放得下 2TB

但实际 OOM 在 ~1,856 GB（才 72% 参数）
外推全模型 = 1,856 / 0.72 ≈ 2,578 GB → 比理论值多 ~1,100 GB（1.87x 倍率）
```

这 ~1.87x 倍率说明有大量内存没有被正确归还给 OS。

## 根因假设（尚未完全验证）

### 假设：`_partition_param` 双重分配

在 `deepspeed/runtime/zero/partition_parameters.py` 的 `_partition_param()` (line 1613-1620)：

```python
# 步骤 1: glibc malloc
partitioned_tensor = torch.empty(partition_size, dtype=param.dtype, device='cpu')   # 3.75GB

# 步骤 2: cudaHostAlloc + copy + free(步骤1)
if device == OffloadDeviceEnum.cpu and self.pin_memory:
    partitioned_tensor = get_accelerator().pin_memory(partitioned_tensor)            # 另一个 3.75GB
```

如果步骤 2 的 `free()` 不归还内存，每个参数分区就会保留 2 倍的内存。

**支持这个假设的证据：**
- MALLOC_TRIM_THRESHOLD_=0 有 ~10% 改善（说明确实有 glibc 内存保留问题）
- pin_memory=true 是默认配置，确实会触发双重分配

**不完全匹配的地方：**
- 如果是纯粹的双重分配问题，MALLOC_TRIM_THRESHOLD_=0 应该改善更多（理论上接近 50%），但实际只改善了 10%
- 可能还有其他内存开销来源（NCCL 缓冲区、PyTorch 内部状态、CUDA context 等）

**结论：双重分配是问题的一部分，但可能不是唯一原因。** 实施的修复仍然值得测试，因为消除双重分配本身是正确的优化。

### 每个参数的完整 flow

```
1. torch.empty(full_size, device='cuda:X')    → GPU 创建全尺寸参数 (15GB for gate_up_proj)
2. dist.broadcast(param.data, 0, dp_group)    → GPU 间广播（NCCL, in-place）
3. _partition_param:
   a. torch.empty(partition_size, device='cpu') → CPU malloc 3.75GB        ← 可能泄漏
   b. pin_memory()                              → cudaHostAlloc 3.75GB + copy + free(a)
   c. copy_(GPU slice → CPU pinned)             → 复制分区数据
   d. free_param()                              → 释放 GPU 全尺寸张量
4. GPU 内存循环使用，CPU pinned 内存累积
```

代码位置：
- DeepSpeed Init: `/mnt/data/lpl/anaconda3/envs/llama/lib/python3.11/site-packages/deepspeed/runtime/zero/partition_parameters.py`
- `_partition_param`: line 1555
- `free_param`: line 282
- `_post_init_method`: line 1088
- `_zero_init_param`: line 1056

## 已实施的修复

### 代码变更

**文件 1：`src/llamafactory/model/patcher.py`**

新增函数 `patch_deepspeed_zero_init_memory()` (line 215-416)，monkey-patch 两个方法：

1. **`Init._partition_param`** — 消除双重分配 + 强制 gc/malloc_trim
   ```python
   # 原始（双重分配）：
   partitioned_tensor = torch.empty(size, device='cpu')     # malloc
   partitioned_tensor = pin_memory(partitioned_tensor)       # cudaHostAlloc + copy + free

   # 修复（直接 pinned）：
   partitioned_tensor = torch.empty(size, device='cpu', pin_memory=True)  # 跳过 malloc
   ```
   大参数分区后额外执行 `del one_dim_param; gc.collect(); libc.malloc_trim(0)`

2. **`Init._post_init_method`** — 模块级清理 + 诊断日志
   - 每个 >100MB 参数的模块处理后执行 `gc.collect()` + `malloc_trim(0)`
   - 每 10 秒输出一次 `[DIAG] zero.Init progress: ... | RSS=...GB, Avail=...GB`

**文件 2：`src/llamafactory/model/loader.py`**

在 `from_pretrained()` 之前调用 `patch_deepspeed_zero_init_memory()`（line 174）。

### 什么没有改

- DeepSpeed 源码没有被修改（纯 monkey-patch）
- v4 shard-by-shard 加载逻辑不受影响
- deepspeed config (`ds_z3_offload_config.json`) 不需要改

## 测试方法

### 第一步：运行测试

```bash
cd /home/lpl/zero-baseline/LlamaFactory

# 建议配合设置（可选，可以先不设看看纯 patch 效果）
export MALLOC_ARENA_MAX=2
export MALLOC_TRIM_THRESHOLD_=0

# 运行 DeepSeek-V3 训练
deepspeed --num_gpus 4 src/llamafactory/launcher.py \
    examples/train_lora/dpsk3_lora_sft_ds3.yaml
```

同时在另一个终端监控内存：
```bash
while true; do echo "$(date +%H:%M:%S) $(grep MemAvailable /proc/meminfo)"; sleep 15; done
```

### 第二步：判断结果

**成功标志：**
1. 日志中出现 `[DIAG] After from_pretrained:` → Phase 0 完成，进入 Phase 1
2. 日志中出现 `[DIAG] ===== Shard-by-shard loading START =====` → Phase 1 开始
3. 进程没有被 OOM killed（没有 exitcode -15）
4. `[DIAG] zero.Init progress:` 日志显示 Avail 始终 > 100GB

**失败标志：**
1. 仍然 OOM killed（exitcode -15）
2. 只有 `Before from_pretrained`，没有 `After from_pretrained`
3. `[DIAG] zero.Init progress:` 日志中 Avail 持续下降到 < 200GB

**部分改善标志：**
1. 进度从 697/967 提升（如到 800+），但仍 OOM
2. 说明修复有效但不充分，需要叠加其他措施

### 第三步：如果部分改善或失败

按优先级尝试：

**3a. 叠加 jemalloc（最快尝试）**
```bash
# 找到 jemalloc
find / -name "libjemalloc*" 2>/dev/null
# 或安装
apt-get install libjemalloc-dev  # 或 conda install jemalloc

# 使用
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so.2 deepspeed --num_gpus 4 ...
```

**3b. 禁用 pin_memory（可快速验证是否是 pin 相关问题）**

修改 `examples/deepspeed/ds_z3_offload_config.json`：
```json
"offload_param": {
    "device": "cpu",
    "pin_memory": false    // ← 改为 false
}
```
**注意**：这会降低训练速度（CPU↔GPU 传输变慢），但可以验证 pin_memory 是否是主要内存来源。

**3c. 收集更详细的诊断数据**

如果上述方法都不够，需要在 `_partition_param` 内部添加逐参数 RSS 日志：

```python
# 在 patcher.py 的 _patched_partition_param 中，free_param 前后添加：
import os
def _get_rss_gb():
    with open(f"/proc/{os.getpid()}/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024 / 1024
    return -1

rss_before = _get_rss_gb()
free_param(param)
gc.collect()
if _has_malloc_trim: _libc.malloc_trim(0)
rss_after = _get_rss_gb()
logger.info_rank0(
    f"[DIAG] param {param.ds_id} shape={param.ds_shape} "
    f"RSS: {rss_before:.2f}→{rss_after:.2f}GB (delta={rss_after-rss_before:+.2f}GB)"
)
```

这会产出每个参数的 RSS 变化，可以精确定位哪些参数导致内存增长。

## 备选方案（如果所有上述方法都不够）

### 方案 D：meta device 初始化（彻底解决，但实现复杂）

核心思路：跳过 `deepspeed.zero.Init()` 的 materialization，直接从 checkpoint 加载。

**实现步骤：**

1. 允许 ZeRO-3 使用 `low_cpu_mem_usage=True`：
   ```python
   # patcher.py patch_config() 中，移除 ZeRO-3 的限制：
   # 原始：init_kwargs["low_cpu_mem_usage"] = ... and (not is_deepspeed_zero3_enabled())
   # 修改：init_kwargs["low_cpu_mem_usage"] = model_args.low_cpu_mem_usage
   ```

2. 模型在 meta device 上创建（0 内存）

3. 在 shard-by-shard 加载中，对每个参数：
   - 从 checkpoint 读取权重
   - 在 CPU 上创建实际 tensor
   - 手动调用 DeepSpeed 的 `_convert_to_deepspeed_param()` + `partition()`
   - 释放全尺寸 tensor

**难点：**
- 需要处理 WeightConverter（expert fusion）
- 需要手动管理 DeepSpeed 的参数元数据（ds_id, ds_tensor, ds_status 等）
- 需要处理 `dist.broadcast()` 同步
- FP8→BF16 转换

**工作量估计：** 200-400 行代码，需要深度理解 DeepSpeed partition_parameters.py

### 方案 E：增加 swap

如果以上都不行，可以临时使用 swap 扩展内存（只在 Phase 0 期间需要）：
```bash
# 创建 500GB swap 文件（在快速 NVMe 上）
sudo fallocate -l 500G /mnt/data/swapfile
sudo chmod 600 /mnt/data/swapfile
sudo mkswap /mnt/data/swapfile
sudo swapon /mnt/data/swapfile
```
Phase 0 完成后 swap 不再使用（Phase 1 的 shard-by-shard 加载内存可控）。

## 模型参数详情（参考）

```
模型配置：
- model_type: deepseek_v3
- num_hidden_layers: 61 (前3层 dense, 后58层 MoE)
- n_routed_experts: 256
- hidden_size: 7168
- moe_intermediate_size: 2048
- intermediate_size: 18432 (shared experts 和 dense layers 使用)
- kv_lora_rank: 512, q_lora_rank: 1536
- quantization_config: fp8, weight_block_size [128, 128]
- 总参数：967 个 nn.Parameter

每 MoE 层 BF16 大小：
- experts.gate_up_proj: [256, 4096, 7168] × 2B = 15.0 GB
- experts.down_proj:    [256, 7168, 2048] × 2B = 7.5 GB
- shared_experts:                               ≈ 0.75 GB
- Attention (MLA):                              ≈ 0.37 GB
- 合计:                                        ≈ 23.6 GB

全模型 BF16: 58 × 23.6 + 3 × ~0.5 + ~5 ≈ 1,376 GB

Checkpoint: 163 shards, 91,991 keys (99% expert keys), FP8 格式
训练配置: examples/train_lora/dpsk3_lora_sft_ds3.yaml
DeepSpeed配置: examples/deepspeed/ds_z3_offload_config.json (stage3, offload CPU, pin_memory=true)
```
