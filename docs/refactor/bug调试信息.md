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
