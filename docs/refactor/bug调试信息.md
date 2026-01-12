# KTransformers SFT Bug 调试记录

## BUG-007: backward_down SIGSEGV（✅ 已解决）

### 问题描述
在 `backward_down` 函数中访问 `grad_down_b` 缓冲区时发生 SIGSEGV。

### 调试历程

#### 第一阶段：发现 lora_rank Object Slicing

**症状**：Python 端设置 `lora_rank=8`，但 C++ 端显示 `lora_rank=16`

**根因**：
- `TP_MOE_SFT` 构造函数将 `MOESFTConfig` 转为 `GeneralMOEConfig` 传递给基类
- `GeneralMOEConfig` 不包含 `lora_rank` 字段（Object Slicing）
- 创建 `AMX_SFT_MOE_TP` 时使用默认值 `lora_rank=16`

**修复**：
```cpp
// sft_moe.hpp
void set_lora_params(int rank, float alpha) {
    lora_rank_ = rank;
    lora_scaling_ = alpha / rank;
}

// moe-sft-tp.hpp 构造函数
for (int i = 0; i < tp_count; i++) {
    tps[i]->set_lora_params(config.lora_rank, config.lora_alpha);
}
```

**状态**：✓ 已修复，lora_rank 现在正确显示为 8

#### 第二阶段：添加详细调试信息

添加了以下调试输出：
1. `[DEBUG backward_down]` - 打印 task_id, expert_idx, config 信息
2. `[DEBUG grad_B]` - 打印循环前的 offset 和指针信息
3. 边界检查 - 如果索引越界打印 `[CRASH]`

#### 第三阶段：调试输出分析（2026-01-06）

**调试输出**：
```
[DEBUG backward_down] task_id=0, expert_idx=0, num_tokens=48, qlen=48, k=6
[DEBUG backward_down] config: hidden_size=2048, intermediate_size=1408, expert_num=64, lora_rank=8
[DEBUG backward_down] lora_b_offset=0, max_valid_offset=1048576
[DEBUG backward_down] grad_down_b=0x7fa926980000, down_lora_b_=0x881e4200
...
[DEBUG grad_B] expert_idx=2, lora_b_offset=32768, hidden_size=2048, lora_rank=8
[DEBUG grad_B] grad_down_b=0x7fa926980000, max_valid_idx=1048576
```

**关键发现**：
| 检查项 | 结果 |
|--------|------|
| lora_rank | 8 ✓ 正确 |
| max_valid_idx | 1048576 ✓ (64×2048×8) |
| grad_down_b 指针 | 0x7fa926980000 (非空) |
| `[CRASH]` 输出 | 无 - 索引在边界内 |
| SIGSEGV | 仍然发生 |

**结论**：索引计算正确且在边界内，但仍然崩溃

### ✅ 根因确认（2026-01-06）

**第四阶段调试输出**：
```
grad_down_lora_b: shape=torch.Size([64, 2048, 8]), numel=1048576,
                  ptr=0x7ff9d6980000, device=cuda:0  ← GPU!
```

**GDB memory mappings 确认**：
```
0x7ff9d6000000 - 0x7ffa20000000  ---p  (无权限区域)
```

地址 `0x7ff9d6980000` 不在任何有效的 CPU 内存映射中！

**根因**：`torch.zeros_like()` 继承原 tensor 的 device。LoRA 参数在 GPU 上，梯度 tensor 也在 GPU 上，但 C++ AMX 代码需要 CPU 内存访问！

### ✅ 修复方案

```python
# kt_moe.py backward() - 添加 device="cpu"
grad_gate_lora_a = torch.zeros_like(ctx.lora_params["gate_lora_a"].data, device="cpu")
grad_gate_lora_b = torch.zeros_like(ctx.lora_params["gate_lora_b"].data, device="cpu")
grad_up_lora_a = torch.zeros_like(ctx.lora_params["up_lora_a"].data, device="cpu")
grad_up_lora_b = torch.zeros_like(ctx.lora_params["up_lora_b"].data, device="cpu")
grad_down_lora_a = torch.zeros_like(ctx.lora_params["down_lora_a"].data, device="cpu")
grad_down_lora_b = torch.zeros_like(ctx.lora_params["down_lora_b"].data, device="cpu")
```

**状态**：已修复 (kt_moe.py:481-489)

### 第五阶段：梯度设备不匹配（2026-01-06）

**错误**：
```
RuntimeError: attempting to assign a gradient with device type 'cpu'
to a tensor with device type 'cuda'
```

**原因**：梯度在 CPU（AMX 需要），但 LoRA 参数在 GPU（`model.to("cuda")` 会移动）

**修复（方案 A）**：
```python
# kt_moe.py:516-521
def accumulate_grad(param: nn.Parameter, grad: torch.Tensor):
    grad_on_device = grad.to(param.device)  # CPU → GPU
    if param.grad is None:
        param.grad = grad_on_device.clone()
    else:
        param.grad.add_(grad_on_device)
```

**新增配置项**：
- `kt_moe_lora_device: gpu` (model_args.py:514-521, YAML line 48)
- 支持 `gpu`（方案 A，已实现）和 `cpu`（方案 B，抛出 NotImplementedError）

**状态**：已修复

---

## BUG-006: Forward cache stack overflow（已解决）

**症状**：forward 过程中 cache stack overflow

**根因**：`gradient_checkpointing` 会多次调用 forward，每次都 push cache 但不 pop

**修复**：在 YAML 中添加 `disable_gradient_checkpointing: true`

**状态**：✓ 已解决

---

## 修改文件汇总

| 文件 | 修改内容 | Bug |
|------|----------|-----|
| `kt_moe.py` | 梯度 tensor 添加 `device="cpu"` (line 481-489) | BUG-007 |
| `kt_moe.py` | `accumulate_grad` 添加 CPU→GPU 传输 (line 508-513) | BUG-007 |
| `model_args.py` | 添加 `kt_moe_lora_device` 配置 | BUG-007 |
| `sft_moe.hpp` | 添加 `set_lora_params()` 修复 Object Slicing | BUG-007 |
| `moe-sft-tp.hpp` | 调用 `set_lora_params()` | BUG-007 |
| `deepseek2_lora_sft_kt.yaml` | 添加 `disable_gradient_checkpointing: true` | BUG-006 |
| `deepseek2_lora_sft_kt.yaml` | 添加 `kt_moe_lora_device: gpu` | BUG-007 |

---

## 清理记录（2026-01-06）

已删除所有调试代码：
- `kt_moe.py`: 删除 `[DEBUG BUG-007]` logger.info 语句
- `sft_moe.hpp`: 删除 `set_lora_params()`、`backward_down()` 中的 printf 调试输出

---

## BUG-008: KTrainer._maybe_log_save_evaluate() 参数不兼容（✅ 已解决）

### 问题描述

```
TypeError: KTrainer._maybe_log_save_evaluate() got an unexpected keyword argument 'learning_rate'
```

训练成功运行第一个 step 后，在 `_maybe_log_save_evaluate()` 调用时报错。

### 根因

`KTrainer` 重写了父类 `_maybe_log_save_evaluate()` 方法，但方法签名缺少新版 transformers Trainer 传递的 `learning_rate` 参数。

### 修复

在 `kt_trainer.py:255` 方法签名中添加 `learning_rate=None` 参数：

```python
def _maybe_log_save_evaluate(self, tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval, learning_rate=None):
    self._update_lora_pointers()
    return super()._maybe_log_save_evaluate(..., learning_rate=learning_rate)
```

**状态**：✓ 已修复

---

## BUG-009: 训练产生 NaN - PEFT 包装空权重层（✅ 已解决）

### 问题描述

训练产生的 adapter 文件包含大量 NaN 值：

| 组件 | NaN 张量数 | 说明 |
|------|------------|------|
| shared_experts | 312 | 所有层的 shared experts LoRA |
| self_attn | 216 | 所有层的 Attention LoRA |
| dense_mlp | 6 | Layer 0 的 Dense MLP LoRA |
| routed_experts | **0** | MoE 路由专家 LoRA **正常** |

### 关键观察

NaN **只出现在 GPU 上训练的部分**（PEFT 管理），KT AMX (CPU) 处理的 routed_experts **没有 NaN**。

### 根因分析

**代码执行顺序** (`loader.py`):
1. `load_kt_model()` - 创建 `MOELayerWrapper`，调用 `_clear_original_expert_weights()` 清空专家权重为 `torch.empty(0)`
2. `init_adapter()` - PEFT 遍历模型，包装所有 Linear 层

**问题**:
- `MOELayerWrapper.__init__()` 存储了 `self.original_moe = original_moe`
- PEFT 的 `get_peft_model()` 通过 `named_modules()` 遍历模型
- 发现 `wrapper.original_moe.experts.N.{gate,up,down}_proj` 层（Linear 类型）
- **但这些层的 weight 已被清空为 `torch.empty(0)`！**
- PEFT 仍然包装这些层，创建 LoRA

**导致的问题**:
1. LoRA A/B 矩阵基于空权重创建，维度可能不正确
2. 前向传播对空权重操作，产生异常值
3. 梯度计算数值不稳定，产生 NaN

### 修复方案

**删除 `self.original_moe = original_moe` 赋值**

修改 `kt_moe.py` 的 `MOELayerWrapper.__init__()`:

```python
# 修改前 (line 567)
self.original_moe = original_moe

# 修改后
# NOTE: Do NOT store original_moe as self.original_moe!
# PEFT's get_peft_model() uses named_modules() to find Linear layers.
# If we store original_moe, PEFT will find original_moe.experts.N.{gate,up,down}_proj
# which have empty weights (cleared by _clear_original_expert_weights).
# This causes NaN during training.
# We only need router and shared_experts, which are stored separately below.
```

`router` 和 `shared_experts` 已经被单独存储，不需要保留 `original_moe` 引用。

### 影响

- **修复前**: PEFT 发现并包装 ~64×26 = 1664 个空权重的专家层
- **修复后**: PEFT 只包装有效权重的层 (self_attn, shared_experts, dense_mlp)

**状态**：✅ 已修复 (kt_moe.py:567)

---

## 修改文件汇总（更新）

| 文件 | 修改内容 | Bug |
|------|----------|-----|
| `kt_moe.py` | 删除 `self.original_moe = original_moe` | BUG-009 |
| `kt_moe.py` | 添加 `self._is_kt_moe_wrapper = True` 标记 | BUG-010 |
| `kt_moe.py` | 梯度 tensor 添加 `device="cpu"` | BUG-007 |
| `kt_moe.py` | `accumulate_grad` 添加 CPU→GPU 传输 | BUG-007 |
| `adapter.py` | 跳过 KT MoE LoRA 参数的 float32 upcast | BUG-010 |
| `model_args.py` | 添加 `kt_moe_lora_device` 配置 | BUG-007 |
| `sft_moe.hpp` | 添加 `set_lora_params()` 修复 Object Slicing | BUG-007 |
| `moe-sft-tp.hpp` | 调用 `set_lora_params()` | BUG-007 |
| `deepseek2_lora_sft_kt.yaml` | 添加 `disable_gradient_checkpointing: true` | BUG-006 |
| `deepseek2_lora_sft_kt.yaml` | 添加 `kt_moe_lora_device: gpu` | BUG-007 |

---

## BUG-010: AMX Forward 产生 NaN（🔄 调查中）

### 问题描述

训练第一个 forward pass 就产生 NaN，从 Layer 1 的 AMX forward 开始。

### 诊断日志分析（2026-01-09）

**日志文件**: `/home/lpl/LLaMA-Factory-KT/kt_nan_diag.log`

#### 关键发现 1: NaN 首次出现在 Layer 1 的 AMX forward

```
[ERROR] [Layer 1] NaN in moe_output (from AMX)!
[ERROR] [Layer 2] NaN in moe_output (from AMX)!
[ERROR] [Layer 2] NaN in shared_experts output!
...
```

**注意**: Layer 1 只有 `moe_output` NaN，**shared_experts 没有 NaN**！
- 这证明 NaN 来源是 **AMX forward 计算**，不是 shared_experts
- Layer 2+ 的 shared_experts 有 NaN 是因为输入 `hidden_states` 已被 Layer 1 污染

#### 关键发现 2: Upcasting 导致 dtype 不匹配

```
[INFO] Upcasting trainable params to float32.  ← 问题根源！
...
[ERROR] [Layer 1] NaN in moe_output (from AMX)!  ← 第一层就 NaN
```

### ✅ 根因确认

**问题链路**:

1. `create_lora_params()` 在 CPU 上创建 **bfloat16** 的 LoRA 参数
2. `MOESFTConfig` 存储这些参数的 CPU 地址（指向 bfloat16 数据）
3. `init_adapter()` 执行 `param.data = param.data.to(torch.float32)` **将所有 trainable 参数转换成 float32**
4. AMX forward 使用原来的指针读取数据，但内存中已经是 **float32**（4 字节）
5. **AMX 以 bfloat16（2 字节）解释 float32 数据 → 产生垃圾值/NaN**

**问题代码位置** (`adapter.py:343-345`):
```python
if is_trainable and cast_trainable_params_to_fp32:
    for param in filter(lambda p: p.requires_grad, model.parameters()):
        param.data = param.data.to(torch.float32)  # ← 把 bf16 LoRA 转成 fp32!
```

### ✅ 修复方案

**修改 1**: 在 `MOELayerWrapper` 添加标记 (`kt_moe.py:584-586`)

```python
# Marker for adapter.py to identify KT MoE wrappers
# Used to skip float32 upcast for LoRA parameters (BUG-010 fix)
self._is_kt_moe_wrapper = True
```

**修改 2**: 跳过 MoE LoRA 参数的 upcast (`adapter.py:343-363`)

```python
if is_trainable and cast_trainable_params_to_fp32:
    # BUG-010 fix: Collect KT MoE LoRA parameters that must stay in bfloat16
    kt_moe_lora_param_ids = set()
    for name, module in model.named_modules():
        if getattr(module, '_is_kt_moe_wrapper', False):
            for param in module.parameters():
                kt_moe_lora_param_ids.add(id(param))

    # Upcast trainable params except KT MoE LoRA parameters
    upcast_count = 0
    for param in filter(lambda p: p.requires_grad, model.parameters()):
        if id(param) not in kt_moe_lora_param_ids:
            param.data = param.data.to(torch.float32)
            upcast_count += 1

    if kt_moe_lora_param_ids:
        logger.info_rank0(
            f"Kept {len(kt_moe_lora_param_ids)} KT MoE LoRA parameters in bfloat16, "
            f"upcast {upcast_count} other parameters to float32"
        )
```

### 修复后预期日志

```
[INFO] Kept 156 KT MoE LoRA parameters in bfloat16, upcast 378 other parameters to float32
```

### 状态（尝试 1 - upcast 修复）

⚠️ **upcast 修复已实施但 NaN 仍存在**

日志确认 upcast 跳过生效：
```
[INFO] Kept 416 KT MoE LoRA parameters in bfloat16, upcast 222 other parameters to float32
```

但 NaN 仍然出现，说明 **upcast 不是真正的根因**。

### 继续调查（2026-01-09）

#### 新的诊断代码

在 `MOEAMXFunction.forward()` 中添加了更详细的诊断：

1. **输入数据检查** - 检查 `hidden_states` 是否有 NaN/Inf
2. **Routing weights 检查** - 检查 `topk_weights` 是否有 NaN/Inf
3. **LoRA 参数检查** - 检查 LoRA 权重是否有 NaN，是否在正确的设备上
4. **输出检查** - 检查 AMX forward 输出，并打印输入/权重范围

#### 关键差异：LlamaFactory vs ktransformers 测试代码

| 项目 | ktransformers 测试 | LlamaFactory |
|------|-------------------|--------------|
| LoRA A 初始化 | `randn / 100` (小值) | `kaiming_uniform` (较大值) |
| LoRA B 初始化 | `zeros` | `zeros` |
| 输入数据 | `randn / 100` (缩小100倍) | 真实模型 hidden_states |
| Base weights | `randn` (随机) | 预训练权重 |

**可能的问题**：数值范围超出 bfloat16 精度范围，导致计算溢出/NaN。

ktransformers 测试文件：`/home/lpl/ktransformers/kt-kernel/examples/test_moe_sft_amx_no_tp.py`
