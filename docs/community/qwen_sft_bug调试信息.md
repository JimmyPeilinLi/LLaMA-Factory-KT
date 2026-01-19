# Qwen MoE SFT 训练 Forward Cache Stack Overflow 问题

**状态**: 已解决
**日期**: 2026-01-16
**影响配置**: `examples/ktransformers/train_lora/qwen3moe_lora_sft_kt.yaml`

## 问题描述

运行 Qwen3-30B-A3B 模型的 LoRA SFT 训练时，程序崩溃并报错：

```
Forward cache stack overflow
cache_stack_top_ = 1
max_cache_depth_ = 1
Hint: If you are doing inference (forward only without backward),
      set save_for_backward=False in forward_sft() call.
      Or increase max_cache_depth in MOESFTConfig.
terminate called after throwing an instance of 'std::runtime_error'
  what():  Forward cache stack overflow
```

## GDB 调试信息

```
Thread 302 "numa_0_m_0" received signal SIGABRT, Aborted.
#9  0x00007ffd258cd870 in AMX_SFT_MOE_TP<amx::GemmKernel224BF, AMX_MOE_TP>::push_cache (this=0x7fcb9084ef60)
    at /home/lpl/ktransformers-llama/kt-kernel/operators/amx/sft_moe.hpp:1868
#10 0x00007ffd258669d8 in AMX_SFT_MOE_TP<amx::GemmKernel224BF, AMX_MOE_TP>::forward_sft (...)
    at /home/lpl/ktransformers-llama/kt-kernel/operators/amx/sft_moe.hpp:534
```

崩溃发生在 ktransformers C++ 内核的 `push_cache` 函数中，因为缓存栈已满但仍尝试压入新的激活值。

## 根因分析

### 关键发现：DeepSeek vs Qwen 配置差异

| 配置项 | DeepSeek 配置 | Qwen 配置 |
|--------|--------------|-----------|
| `disable_gradient_checkpointing` | `true` | 未设置 (默认 `false`) |
| `gradient_accumulation_steps` | 8 | 8 |

**DeepSeek 正常运行，Qwen 崩溃**，原因在于 gradient checkpointing 的行为。

### Bug 位置

**文件**: `src/llamafactory/model/model_utils/kt_moe.py`
**行号**: 591

```python
# 修复前的代码
self.training,  # save_for_backward: only save cache when training
```

### 问题机制

`self.training` 是 PyTorch 模块的训练模式标志，即使在 `torch.no_grad()` 上下文中也为 `True`。

Gradient checkpointing 的工作流程：
1. **原始 forward** 在 `torch.no_grad()` 下运行，但 `self.training=True` → **压入缓存**
2. **backward 期间**，checkpointing 在 `torch.enable_grad()` 下重新运行 forward → **再次压入缓存**
3. **backward** 只弹出一次 → **缓存中残留 +1 项**

每个 micro-batch 后缓存多出 1 项，最终导致溢出！

### DeepSeek 为何正常

设置 `disable_gradient_checkpointing: true` 后：
- 每个 micro-batch 只有一次 forward → 压入
- 每个 micro-batch 只有一次 backward → 弹出
- 缓存保持平衡，无溢出

## 解决方案

### 代码修复 (已应用)

**文件**: `src/llamafactory/model/model_utils/kt_moe.py`
**行号**: 591

```python
# 修复前
self.training,  # save_for_backward: only save cache when training

# 修复后
self.training and torch.is_grad_enabled(),  # save_for_backward: only when training AND gradients enabled (for gradient checkpointing compatibility)
```

### 修复原理

- 原始 forward 在 `torch.no_grad()` 下：`True and False = False` → **不压入缓存**
- 重新运行 forward 在 `torch.enable_grad()` 下：`True and True = True` → **压入缓存**
- backward：**弹出** → 缓存平衡！

### 备选方案（临时解决）

如果无法修改代码，可在 YAML 配置中添加：

```yaml
disable_gradient_checkpointing: true
```

但这会使用更多 GPU 显存。

## 测试验证

修复后，Qwen3 MoE 模型应能正常运行 SFT 训练：

```bash
llamafactory-cli train examples/ktransformers/train_lora/qwen3moe_lora_sft_kt.yaml
```

## 相关文件

- `src/llamafactory/model/model_utils/kt_moe.py` - KT MoE 包装器实现
- `src/llamafactory/model/model_utils/checkpointing.py` - Gradient checkpointing 实现
- `src/llamafactory/hparams/model_args.py` - KTransformers 参数定义
- `examples/ktransformers/train_lora/qwen3moe_lora_sft_kt.yaml` - Qwen3 训练配置
- `examples/ktransformers/train_lora/deepseek2_lora_sft_kt.yaml` - DeepSeek 训练配置（参考）

## 总结

此 bug 是由于 KTMoEWrapper 的 `save_for_backward` 标志没有正确处理 gradient checkpointing 场景导致的。通过检查 `torch.is_grad_enabled()` 状态，确保只在真正需要计算梯度时才保存激活值到缓存，从而与 PyTorch 的 gradient checkpointing 机制正确配合。

---

# Qwen MoE SFT 训练 NaN 问题 #1: Router 权重归一化

**状态**: 已解决
**日期**: 2026-01-16
**影响配置**: `examples/ktransformers/train_lora/qwen3moe_lora_sft_kt.yaml`

## 问题描述

Router 权重归一化时缺少除零保护，导致 NaN。

## 解决方案

**文件**: `src/llamafactory/model/model_utils/kt_moe.py`

```python
# 修复后（与 LLaMA-Factory moe.py 一致）
topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True).clamp(min=1e-9)
```

---

# Qwen MoE SFT 训练 NaN 问题 #2: kt-kernel backward 异常

**状态**: 已解决
**日期**: 2026-01-19
**影响配置**: `examples/ktransformers/train_lora/qwen3moe_lora_sft_kt.yaml`

## 问题描述

修复 Router 归一化问题后，训练仍然出现 NaN：
- **约 100 步后 backward 输出极端值** (~9e31) 并产生 NaN/Inf
- DeepSeek 正常，Qwen 失败

## 模型对比分析

| 参数 | Qwen3-30B-A3B | DeepSeekV2-Lite | 备注 |
|------|---------------|-----------------|------|
| moe_intermediate_size | **768** | 1408 | Qwen 更小 |
| num_experts | **128** | 64 | Qwen 更多 |
| experts_per_tok | 8 | 6 | |
| num_layers | 48 | 27 | |

## 调试日志对比

### BWD STEP max_abs 值对比

| Step | DeepSeek | Qwen | 差异 |
|------|----------|------|------|
| 1 | **7.75e+00** | **6.24e+07** | **800 万倍** |
| 2 | 2.33e+00 | 2.75e+03 | 1000 倍 |
| 3 | 1.04e+00 | 1.58e+03 | 1500 倍 |
| ... | 1e-1 ~ 1e+02 | 1e+02 ~ 1e+04 | 100-1000 倍 |

### 关键发现

1. **Qwen Step 1 就有 6.24e+07** — 第一步就出问题，不是累积导致
2. **DeepSeek 完全正常** — 没有任何 BWD ALERT
3. **问题在 kt-kernel backward() 内部**:

```
[BWD STEP 101] max_abs=1.39e+02, nan_total=0, inf_total=0  ← 正常
[BWD ALERT L47] Anomaly detected!
  grad_output: range=[-5.00e-04, 4.23e-04], nan=0, inf=0   ← 输入正常！
  grad_input:  range=[-2.96e+31, 3.55e+31], nan=5608, inf=48  ← 输出爆炸！
```

**结论**: `grad_output` 完全正常，`grad_input` 爆炸 → 问题 100% 在 kt-kernel backward() 内部

## 可能原因（待排查）

1. **内存 Buffer 分配问题** — 128 专家可能导致某些 buffer 分配不足或重叠
2. **Intermediate size 768 对齐问题** — 768 比 1408 小，可能有 AMX 对齐问题
3. **初始化问题** — 第一步就有 6.24e+07，说明初始化就有问题
4. **并行竞争** — 多线程数据冲突
5. **Cache 不匹配** — forward cache 与实际数据不一致

## backward 流程分析

```
backward() 三步：
1. backward_down_amx()     → 输出 grad_intermediate_
2. backward_activation()   → 输出 grad_gate_output_, grad_up_output_
3. backward_gate_up_amx()  → 输出 grad_input
```

### 调试结果（2026-01-19）

```
[BWD ALERT L47] Anomaly detected! (qlen=464, total_tokens=3712)
  grad_output:       range=[-5.07e-04, 4.23e-04], nan=0, inf=0       ✓ 正常
  grad_intermediate: range=[-5.29e-05, 6.08e-05], nan=0, inf=0       ✓ 正常
  grad_gate:         range=[-3.13e-04, 2.46e-04], nan=0, inf=0       ✓ 正常
  grad_up:           range=[-4.16e-04, 4.69e-04], nan=0, inf=0       ✓ 正常
  grad_input:        range=[-3.74e+31, 3.99e+31], nan=7160, inf=40   ✗ 爆炸！
```

**结论**: 问题 100% 出在 `backward_gate_up_amx()` 函数！

所有中间缓冲区（grad_intermediate, grad_gate, grad_up）都正常，只有 `backward_gate_up_amx` 的输出 `grad_input` 爆炸。

## 调试代码位置

**文件**: `/home/lpl/ktransformers-llama/kt-kernel/operators/amx/sft_moe.hpp`

已添加分步调试输出，异常时打印：
```
[BWD ALERT L47] Anomaly detected! (qlen=464, total_tokens=3712)
  grad_output:       range=[...], nan=0, inf=0
  grad_intermediate: range=[...], nan=?, inf=?  <- backward_down
  grad_gate:         range=[...], nan=?, inf=?  <- backward_activation
  grad_up:           range=[...], nan=?, inf=?  <- backward_activation
  grad_input:        range=[...], nan=?, inf=?  <- backward_gate_up
```

## 根因分析：Buffer Pool 溢出

### 问题定位

问题 100% 出在 `backward_gate_up_amx()` 函数，根因是 **共享 buffer pool 分配不足导致内存溢出**。

### Buffer Pool 分配机制

**文件**: `/home/lpl/ktransformers-llama/kt-kernel/operators/amx/sft_moe.hpp`

#### 1. Pool 初始化时的大小计算（一次性分配）

在 `allocate_buffers()` 中 (line 1116)：

```cpp
// 假设最坏情况：max_len 个 token，每个选 num_experts_per_tok 个专家
size_t max_total_tokens = ((config_.max_len * config_.num_experts_per_tok + M_STEP - 1) / M_STEP) * M_STEP;

// Qwen3: max_len=2048, num_experts_per_tok=8, M_STEP=32
// max_total_tokens = round_up(2048 * 8) = 16384 "虚拟 token 槽位"

// 然后为共享 pool 分配固定大小
grad_output_bf16_pool_bytes_ = max_total_tokens * hidden_size * sizeof(bf16);
backward_bc_pool_bytes_ = BufferC::size(max_total_tokens, intermediate) + BufferC::size(max_total_tokens, hidden);
```

**这是固定大小的共享 pool，不是每个 expert 独立分配！**

#### 2. 运行时动态分配（每次 backward）

在 `backward_down_amx()` 中 (line 2303-2322)，从共享 pool 中切分：

```cpp
char* grad_output_bf16_ptr = (char*)grad_output_bf16_pool_;  // 从 pool 开头开始

for (int task_id = 0; task_id < activated_expert; task_id++) {
  int expert_idx = m_expert_id_map_[task_id];
  int m = m_local_num_[expert_idx];  // 这个专家实际收到的 token 数（动态的！）

  // ❗关键问题：round up to M_STEP
  size_t local_max_m = ((m + M_STEP - 1) / M_STEP) * M_STEP;

  grad_output_bf16_ptr_[expert_idx] = (ggml_bf16_t*)grad_output_bf16_ptr;
  grad_output_bf16_ptr += align64(local_max_m * hidden_size * sizeof(bf16));  // 指针前移
}
```

### Bug 机制

1. **Pool 大小**: 按 `max_total_tokens` (一次 round up) 计算
2. **实际分配**: 每个专家的 `local_max_m` 独立 round up to M_STEP

**数学证明 (Qwen3 配置)**:

- max_len=2048, num_experts_per_tok=8, M_STEP=32
- max_total_tokens = round_up(2048 × 8) = 16384
- 如果 128 个专家每个平均有 ~128 tokens:
  - Pool: 为 16384 tokens 分配
  - 实际: 128 × round_up(128) = 128 × 128 = 16384 ✓ (刚好)

- **但 token 分布不均匀时**:
  - Expert A: 1 token → round up to 32 (浪费 31)
  - Expert B: 33 tokens → round up to 64 (浪费 31)
  - 128 专家平均每个浪费 ~16 → 总浪费 2048 → **16384 + 2048 = 18432 > 16384 溢出!**

### Token 分配的动态性

每个 batch 的 token → expert 分配是完全动态的，由 router 决定：

```
Batch: 464 tokens, 每个选 8 个专家
→ 总共 464 × 8 = 3712 个 (token, expert) 对

这些对如何分配到 128 个专家？完全动态！

可能的分布 A (均匀):
  Expert 0: 29 tokens, Expert 1: 29 tokens, ..., Expert 127: 29 tokens
  每个 round up: 29 → 32
  总和: 128 × 32 = 4096 > 3712 ❌ 溢出!

可能的分布 B (极端不均匀):
  Expert 0: 1 token, Expert 1: 200 tokens, Expert 2: 0, ...
  Expert 0 round up: 1 → 32 (浪费 31)
  这种情况浪费更严重
```

### 为什么 Qwen 128 专家更容易出问题？

设 M_STEP = 32，总 token-expert 对 = T

| 模型 | 专家数 N | 平均每专家 T/N | 最大 Rounding 浪费 |
|------|----------|----------------|-------------------|
| DeepSeek | 64 | T/64 | 64 × 31 = **1984** |
| **Qwen** | **128** | **T/128** | 128 × 31 = **3968** |

**关键洞察**：
- 专家越多 → 每个专家 token 数越少 → rounding 浪费比例越大
- Qwen 128 专家，每个平均只收 ~29 tokens → round up 到 32，浪费 ~10%
- DeepSeek 64 专家，每个平均 ~58 tokens → round up 到 64，浪费 ~10%
- **但 Qwen 有 128 个专家都在浪费，总浪费量是 DeepSeek 的两倍！**

### 溢出如何导致 NaN？

```
1. buffer pool 只有 16384 slots
2. 实际分配需要 16384 + rounding_waste (最多 3968)
3. 当 rounding_waste 足够大时，写入越界
4. 越界写入覆盖了其他数据或读取未初始化内存
5. backward_gate_up_amx 读取这些"垃圾数据"
6. 垃圾数据 × weights → 极端值 (~1e31) → NaN
```

**Step 102 突然爆炸**是因为那个 batch 的 token 分布恰好触发了足够大的 rounding 浪费。

### 与之前分析的关系

| 之前的假设 | 实际情况 |
|-----------|----------|
| Step 1 就有 6.24e+07 是初始化问题 | 不是初始化问题，是第一次触发 buffer 分布不均时就溢出了 |
| 问题在 backward_gate_up_amx | ✓ 正确，溢出发生在这里的 buffer 分配 |
| 128 专家配置导致问题 | ✓ 正确，更多专家 = 更多 rounding 浪费 |
| intermediate_size=768 有影响 | 次要因素，主因是专家数 |

## 修复方案

### 方案 A：修改 pool 大小计算（推荐）

在 `allocate_buffers()` 中为 per-expert rounding 留出 buffer：

**文件**: `/home/lpl/ktransformers-llama/kt-kernel/operators/amx/sft_moe.hpp`
**位置**: line 1116 附近

```cpp
// 原来的计算
size_t max_total_tokens = ((config_.max_len * config_.num_experts_per_tok + M_STEP - 1) / M_STEP) * M_STEP;

// 修复：为每个可能激活的专家留出 M_STEP-1 的 rounding 余量
// 最坏情况：所有 expert_num 个专家都被激活，每个浪费 M_STEP-1
size_t rounding_overhead = config_.expert_num * (M_STEP - 1);
size_t max_total_tokens_with_overhead = max_total_tokens + rounding_overhead;

// 使用 max_total_tokens_with_overhead 计算所有依赖 per-expert allocation 的 pool
backward_bc_pool_bytes_ = T::BufferC::required_size(max_total_tokens_with_overhead, config_.intermediate_size) +
                          T::BufferC::required_size(max_total_tokens_with_overhead, config_.hidden_size);
```

**需要修改的 pool** (line 1121-1154):

| 变量名 | 行号 | 说明 |
|--------|------|------|
| `lora_ba_pool_bytes_` | 1121 | LoRA BufferA |
| `lora_bc_inter_pool_bytes_` | 1126 | LoRA BufferC intermediate |
| `lora_bc_out_pool_bytes_` | 1131 | LoRA BufferC output |
| `lora_intermediate_bf16_pool_bytes_` | 1136 | LoRA BF16 intermediate |
| `backward_ba_pool_bytes_` | 1143 | Backward BufferA |
| `backward_bc_pool_bytes_` | 1150-1151 | Backward BufferC |
| `grad_output_bf16_pool_bytes_` | 1154 | Gradient output BF16 |

### 修复步骤

1. **修改 pool 大小计算** - 在 `allocate_buffers()` 中添加 `rounding_overhead` (line 924-930)
2. **重新编译 kt-kernel**
3. **测试验证** - 运行 Qwen3 MoE SFT，确认 NaN 消失

## 参考日志

- Qwen 日志: `/tmp/llama_full.log`
- DeepSeek 日志: `/tmp/llama_full2.log`

## 相关文件

- `/home/lpl/ktransformers-llama/kt-kernel/operators/amx/sft_moe.hpp` - KT MoE 内核实现
- `src/llamafactory/model/model_utils/kt_moe.py` - KT MoE Python 包装器

## 验证结果

**2026-01-19**: 修复已验证成功，Qwen3 MoE SFT 训练正常运行，无 NaN 问题。

修复内容：在 `allocate_buffers()` 中添加 `rounding_overhead` 计算，为 per-expert M_STEP rounding 预留空间。
