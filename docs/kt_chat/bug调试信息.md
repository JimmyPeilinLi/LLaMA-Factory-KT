# KTransformers Chat 推理功能 Bug 调试信息

## Bug 记录

---

### BUG-010: KT SFT 训练 NaN 问题 - LoRA 指针失效 (最新)

**问题描述**:
KT SFT 训练时，AMX forward 输出全是 NaN，即使输入和 LoRA 参数值都正常。

**诊断日志**: `/home/lpl/LLaMA-Factory-KT/kt_nan_diag.log`

#### 尝试 1：跳过 MoE LoRA float32 upcast (失败)

**假设**: PEFT 的 `_cast_lora_weights_to_float32()` 把 MoE LoRA 转成 float32，与 BF16 权重不兼容。

**修复**:
```python
# adapter.py
def _cast_lora_weights_to_float32(model):
    for name, module in model.named_modules():
        if hasattr(module, '_is_kt_moe_wrapper') and module._is_kt_moe_wrapper:
            continue  # 跳过 KT MoE wrapper
        # ... 原有逻辑
```

**结果**: NaN 仍存在 ❌

#### 尝试 2：添加 `_apply()` 方法保护 LoRA 设备 (部分成功)

**诊断日志** (2026-01-09 14:35:50):
```
[ERROR] LoRA param down_lora_a on cuda:0, expected CPU!
[ERROR] NaN/Inf in AMX output!
```

**根因**: HuggingFace Trainer 调用 `model.to(device)`，PyTorch 的 `nn.Module._apply()` 把所有参数移到 CUDA，包括 LoRA 参数。AMX 是 CPU 指令集，无法读取 CUDA 张量。

**修复**:
```python
# kt_moe.py - MOELayerWrapper
def _apply(self, fn, recurse=True):
    result = super()._apply(fn, recurse)
    # 强制 LoRA 参数回 CPU
    for k, v in self.lora_params.items():
        if v.data.device.type != 'cpu':
            v.data = v.data.to('cpu')
    return result
```

**验证结果** (2026-01-09 14:54:26):
```
[ERROR] NaN/Inf in AMX output!
[ERROR]   Input range: [-3.2812, 9.3750]  ← 输入正常
[ERROR]   down_lora_a range: [-0.0094, 0.0093]  ← LoRA 值正常
[ERROR]   (没有设备错误了！)
```

**结果**: 设备问题已修复，但 NaN 仍存在 ⚠️

#### 尝试 3：在 _apply() 中更新 AMX LoRA 指针 (失败)

**修复**:
```python
# kt_moe.py - MOELayerWrapper._apply()
def _apply(self, fn, recurse=True):
    result = super()._apply(fn, recurse)
    for k, v in self.lora_params.items():
        if v.data.device.type != 'cpu':
            v.data = v.data.to('cpu')
    self.update_lora_pointers()  # 在 _apply 中更新指针
    return result
```

**验证结果** (2026-01-09 15:12:01):
```
[ERROR] [MOEAMXFunction.forward] NaN/Inf in AMX output!
  Input range: [-3.2812, 9.3750]  ← 输入正常
  down_lora_a range: [-0.0094, 0.0093]  ← LoRA 值正常
  (没有设备错误了！)  ← _apply() 设备修复有效
```

**结果**: NaN 仍存在 ❌

**分析**: `_apply()` 的调用时机不可控，可能：
1. 被多次调用（模型加载、PEFT 包装、Trainer 初始化等）
2. 在 `update_lora_pointers()` 之后还有其他操作导致指针失效

#### 尝试 4：在 forward() 中更新指针 (待验证)

**根因**: `_apply()` 调用时机不可控，只有在 `forward()` 中更新才能确保指针有效。

**修复**:
```python
# kt_moe.py - MOELayerWrapper.forward()
def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    # CRITICAL: 每次 forward 前更新指针
    self.update_lora_pointers()

    # ... rest of forward logic ...
```

**性能考虑**:
- `update_lora_pointers()` 开销很小（6 次 `data_ptr()` + 1 次 submit/sync）
- 对于 SFT 训练可以忽略不计

**状态**: 待验证

#### 为什么 ktransformers 单元测试没有这个问题

ktransformers 测试代码 (`test_moe_sft_amx_no_tp.py`):
- 不使用 HuggingFace Trainer
- 直接在 CPU 上创建 LoRA 参数，手动调用 AMX forward
- 没有任何 `.to(device)` 调用
- 所以 LoRA 参数始终在 CPU 上，指针不会失效

而 LlamaFactory 集成:
- 使用 HuggingFace Trainer
- Trainer 内部会调用 `model.to(device)` 或类似操作
- 导致所有参数（包括 LoRA）被移到 CUDA，然后强制回 CPU
- 内存地址改变，指针失效

---

### BUG-009: MOELayerWrapper 存储 original_moe 导致显存浪费 (已修复)

**问题描述**:
`MOELayerWrapper` 存储了 `original_moe` 引用，导致原始 MoE 层无法被垃圾回收，显存浪费。

**修复**:
移除 `self.original_moe = original_moe` 存储，不再保留原始层的引用。

**状态**: ✅ 已修复

---

### Bug 5: torch.multinomial 采样错误 - 训练产生 NaN (最新)

**问题描述**:
KT Chat 推理时，生成阶段报概率张量包含无效值错误。

**错误信息**:
```
probability tensor contains either `inf`, `nan` or element < 0
torch.multinomial(probs, num_samples=1)
```

**诊断过程**:

使用 `scripts/inspect_safetensors.py` 检查 adapter 文件：
```bash
python scripts/inspect_safetensors.py /mnt/data/lpl/ls/saves/Kllama_deepseekV2/adapter_model.safetensors
```

然后检查 NaN 值：
```python
from safetensors import safe_open
import torch

with safe_open(adapter_file, framework='pt') as f:
    for key in f.keys():
        tensor = f.get_tensor(key)
        if torch.isnan(tensor).any():
            print(f'NaN: {key}')
```

**诊断结果**:

adapter 文件中有 **534 个张量包含 NaN 值**！

| 组件 | NaN 张量数 | 说明 |
|------|-----------|------|
| `shared_experts` | 312 | 所有层的 shared experts LoRA |
| `self_attn` | 216 | 所有层的 Attention LoRA |
| `dense_mlp` | 6 | Layer 0 的 Dense MLP LoRA |
| `routed_experts` | **0** | MoE 路由专家 LoRA **正常** |

**根因**:

这是**训练问题**，不是推理代码问题。训练过程中产生了 NaN 值，可能原因：
1. 学习率过高导致梯度爆炸
2. Gradient clipping 不足
3. BF16 精度问题
4. 特定层的梯度累积问题

**结论**:

KT Chat 推理代码正常，需要修复训练流程以避免 NaN 值。

---

### Bug 4: Layer 0 Dense MLP 设备不匹配

**问题描述**:
KT Chat 推理时，Layer 0 的 Dense MLP 报设备不匹配错误。

**错误信息**:
```
RuntimeError: Expected all tensors to be on the same device, but got mat2 is on cpu
File "modeling_deepseek.py", line 389, in forward
    down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
```

**关键观察**:
- 错误发生在 `DeepseekMLP.forward()`，这是 Layer 0 的 Dense MLP，不是 MoE
- DeepSeek-V2-Lite: Layer 0 是 Dense MLP，Layers 1-26 是 MoE

**原因分析**:

`kt_loader.py` 的 `move_non_experts_to_gpu()` 函数中：

```python
moe_module = getattr(layer, moe_config.moe_layer_attr, None)  # moe_layer_attr = "mlp"
if moe_module is None:  # 问题：Layer 0 的 mlp 不是 None！
    layer.mlp.to(device)  # 这行被跳过了
    continue
```

对于 DeepSeek-V2：
- `moe_config.moe_layer_attr = "mlp"`
- Layer 0: `layer.mlp = DeepseekMLP` (不是 None，是 Dense MLP 对象)
- 所以条件 `moe_module is None` 为 False
- **Layer 0 的 MLP 没有被移到 GPU！**

**解决方案**:

修改检查逻辑，使用 `experts` 属性来区分 MoE 和 Dense MLP：

```python
# 修改前
if moe_module is None:
    layer.mlp.to(device)
    continue

# 修改后
if moe_module is None or not hasattr(moe_module, moe_config.experts_attr):
    layer.mlp.to(device)
    continue
```

**修复文件**: `src/llamafactory/model/model_utils/kt_loader.py` 第 305-313 行

---

### Bug 1: merge_and_unload() 维度不匹配

**问题描述**:
使用 KT 训练的 LoRA adapter 进行推理时，`merge_and_unload()` 报错。

**错误信息**:
```
RuntimeError: The size of tensor a (0) must match the size of tensor b (2048) at non-singleton dimension 1
```

**原因分析**:
1. KT 模式下，MoE 层的基础权重被移到 CPU，GPU 上的权重张量被清空（shape 变为 `[0, ...]`）
2. PEFT 的 `merge_and_unload()` 尝试将 LoRA 权重合并到基础权重
3. 空张量与 LoRA 权重维度不匹配，导致报错

**解决方案**:
在 KT 模式下跳过 `merge_and_unload()`，改为直接使用 PEFT 包装的模型。

**代码修改** (`adapter.py`):
```python
# 旧代码
model = PeftModel.from_pretrained(model, adapter, **init_kwargs)
model = model.merge_and_unload()

# 新代码
if is_kt_mode:
    model = PeftModel.from_pretrained(model, adapter, **init_kwargs)
    # 不调用 merge_and_unload()
else:
    model = PeftModel.from_pretrained(model, adapter, **init_kwargs)
    model = model.merge_and_unload()
```

---

### Bug 2: Expected all tensors to be on the same device

**问题描述**:
跳过 `merge_and_unload()` 后，推理时报设备不匹配错误。

**错误信息**:
```
RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu! (when checking argument for argument mat2 in method wrapper_CUDA_mm)
```

**错误堆栈**:
```
File "transformers/models/deepseek_v2/modeling_deepseek_v2.py", line 325, in forward
    gate_up = self.gate_proj(x)
File "peft/tuners/lora/layer.py", line 598, in forward
    result = self.base_layer(x, *args, **kwargs)
```

**原因分析**:
1. PEFT 仍然包装了 MoE 层（gate_proj, up_proj, down_proj）
2. PEFT 的 LoRA 层尝试对 `base_layer` 进行前向计算
3. 但 `base_layer` 的权重在 CPU（被 KT 移走），输入 `x` 在 GPU
4. CPU 张量与 GPU 张量相乘导致设备不匹配

**调试输出**:
```
PEFT config: {'default': LoraConfig(
    target_modules={'up_proj', 'kv_a_proj_with_mqa', 'o_proj', 'gate_proj', 'q_proj', 'kv_b_proj', 'down_proj'},
    ...
)}
```
可以看到 PEFT 配置中仍包含 MoE 模块。

**解决方案**:
需要阻止 PEFT 包装 MoE 层，只包装 Attention 层。

---

### Bug 3: PeftModel.from_pretrained() 不接受 target_modules 参数

**问题描述**:
尝试通过传入 `target_modules` 参数来过滤模块，但无效。

**尝试的代码**:
```python
# 尝试只加载 Attention LoRA
model = PeftModel.from_pretrained(
    model, adapter,
    target_modules=attention_targets,  # 这个参数被忽略了
    **init_kwargs
)
```

**结果**:
PEFT 仍然使用 `adapter_config.json` 中保存的 `target_modules`，忽略传入的参数。

**原因分析**:
查看 PEFT 源码发现，`PeftModel.from_pretrained()` 会从 adapter 目录读取 `adapter_config.json`，直接使用其中的配置，不接受外部覆盖。

**解决方案**:
创建临时目录，修改 `adapter_config.json` 后再加载。

**最终代码** (`adapter.py`):
```python
if attention_targets != original_targets:
    # 创建临时目录
    temp_dir = tempfile.mkdtemp(prefix="kt_adapter_")
    try:
        # 复制文件，修改配置
        for file in os.listdir(adapter):
            src = os.path.join(adapter, file)
            dst = os.path.join(temp_dir, file)
            if file == "adapter_config.json":
                # 写入修改后的配置
                modified_config = adapter_config.copy()
                modified_config["target_modules"] = attention_targets
                with open(dst, "w") as f:
                    json.dump(modified_config, f)
            else:
                # 其他文件使用符号链接
                os.symlink(src, dst)

        # 从临时目录加载
        model = PeftModel.from_pretrained(model, temp_dir, **init_kwargs)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
```

---

## 调试工具

### 调试代码 (loader.py)

在 `load_model()` 函数末尾添加了调试输出：

```python
# Debug: Print KT and LoRA loading status
if int(os.getenv("LOCAL_RANK", "0")) == 0:
    print("\n" + "=" * 60)
    print("DEBUG: Model structure after loading")
    print("=" * 60)

    # Check KT wrappers
    kt_wrappers = getattr(model, "_kt_wrappers", [])
    print(f"KT wrappers count: {len(kt_wrappers)}")

    # Check MoE LoRA params
    kt_moe_lora = getattr(model, "_kt_moe_lora_params", {})
    print(f"KT MoE LoRA layers: {list(kt_moe_lora.keys())}")

    # Print LoRA-related parameters
    lora_params = [(name, param.shape) for name, param in model.named_parameters()
                   if "lora" in name.lower()]
    print(f"\nLoRA parameters count: {len(lora_params)}")
    if lora_params:
        print("LoRA parameters:")
        for name, shape in lora_params[:20]:
            print(f"  {name}: {shape}")
        if len(lora_params) > 20:
            print(f"  ... and {len(lora_params) - 20} more")

    # Check PEFT config
    if hasattr(model, "peft_config"):
        print(f"\nPEFT config: {model.peft_config}")

    print("=" * 60 + "\n")
```

### 调试输出解读

**正常输出示例**:
```
============================================================
DEBUG: Model structure after loading
============================================================
KT wrappers count: 26
KT MoE LoRA layers: []

LoRA parameters count: 168
LoRA parameters:
  base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight: torch.Size([8, 2048])
  base_model.model.model.layers.0.self_attn.q_proj.lora_B.default.weight: torch.Size([1536, 8])
  ...

PEFT config: {'default': LoraConfig(
    target_modules={'kv_a_proj_with_mqa', 'o_proj', 'q_proj', 'kv_b_proj'},
    ...
)}
============================================================
```

**检查要点**:
1. `KT wrappers count: 26` - 应有 26 个 MoE 层（DeepSeek-V2-Lite 有 27 层，第 0 层是 dense MLP）
2. `target_modules` 不应包含 `gate_proj`, `up_proj`, `down_proj`
3. LoRA parameters 应该只有 Attention 相关的参数

**异常输出示例**:
```
PEFT config: {'default': LoraConfig(
    target_modules={'up_proj', 'kv_a_proj_with_mqa', 'o_proj', 'gate_proj', 'q_proj', 'kv_b_proj', 'down_proj'},
    ...
)}
```
如果 `target_modules` 包含 MoE 模块，说明过滤失败。

---

## 调试步骤

### 步骤 1: 验证 KT wrapper 创建

```python
# 检查 model 是否有 _kt_wrappers 属性
wrappers = getattr(model, "_kt_wrappers", None)
print(f"KT wrappers: {wrappers is not None}")
print(f"Count: {len(wrappers) if wrappers else 0}")
```

### 步骤 2: 验证 adapter 配置

```python
import json
import os

adapter_path = "/mnt/data/lpl/ls/saves/Kllama_deepseekV2"
config_path = os.path.join(adapter_path, "adapter_config.json")

with open(config_path, "r") as f:
    config = json.load(f)

print(f"target_modules: {config.get('target_modules', [])}")
```

### 步骤 3: 验证临时目录创建

在 `adapter.py` 中添加日志：
```python
logger.info_rank0(f"Created temp dir: {temp_dir}")
logger.info_rank0(f"Files: {os.listdir(temp_dir)}")
```

### 步骤 4: 验证 MoE LoRA 加载

```python
# 检查 adapter 文件中的 MoE LoRA 权重
from safetensors import safe_open

adapter_file = "/mnt/data/lpl/ls/saves/Kllama_deepseekV2/adapter_model.safetensors"
moe_keys = []

with safe_open(adapter_file, framework="pt") as f:
    for key in f.keys():
        if "gate_proj" in key or "up_proj" in key or "down_proj" in key:
            moe_keys.append(key)

print(f"MoE LoRA keys count: {len(moe_keys)}")
print(f"Sample keys: {moe_keys[:5]}")
```

---

## 关键代码位置

| 功能 | 文件 | 行号/函数 |
|------|------|----------|
| KT 模式检测 | `adapter.py` | `is_kt_mode = getattr(model, "_kt_wrappers", None) is not None` |
| 临时目录创建 | `adapter.py` | `temp_dir = tempfile.mkdtemp(prefix="kt_adapter_")` |
| MoE 模块过滤 | `adapter.py` | `moe_modules = {"gate_proj", "up_proj", "down_proj"}` |
| MoE LoRA 加载 | `kt_moe.py` | `load_moe_lora_from_adapter()` |
| 调试输出 | `loader.py` | `print("DEBUG: Model structure after loading")` |
