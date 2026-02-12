# Bug: gate_lora_a shape mismatch (lora_rank=1 vs 12)

## 错误信息

```
ValueError: gate_lora_a shape mismatch: expected (128, 1, 2048), got (128, 12, 2048)
```

中间维度 `1` vs `12` 是 `lora_rank`。C++ 内核以 `lora_rank=1` 初始化，但 PEFT LoRA 层使用的是 YAML 配置中的 `lora_rank=12`。

## 复现命令

```bash
CUDA_VISIBLE_DEVICES=1 LD_PRELOAD=~/mimalloc/build/libmimalloc.so accelerate launch \
  --num_cpu_threads_per_process 64 \
  --config_file examples/accelerate/kt_config.yaml \
  src/train.py /home/lpl/kt-refactor/LLaMA-Factory-KT/examples/ktransformers/train_lora/qwen3moe_lora_sft_kt.yaml
```

## 涉及的配置文件

### LLaMA-Factory 训练 YAML (`examples/ktransformers/train_lora/qwen3moe_lora_sft_kt.yaml`)

```yaml
lora_rank: 12          # → 解析到 finetuning_args.lora_rank
use_kt: true           # → 解析到 model_args.use_kt
kt_backend: AMXBF16    # → 解析到 model_args.kt_backend
kt_num_threads: 60     # → 解析到 model_args.kt_num_threads
```

注意：这些字段被解析为 `ModelArguments` 和 `FinetuningArguments`，**不是** `TrainingArguments`。
YAML 中没有 `kt_config` 或 `accelerator_config` 字段。

### Accelerate 启动器配置 (`examples/accelerate/kt_config.yaml`)

```yaml
kt_config:
  enabled: true
  kt_backend: AMXBF16
  kt_num_threads: 64
  lora_rank: 12       # ← 有正确的值
  lora_alpha: 24
  model_max_length: 512
```

---

## 完整调用链分析

### 阶段 1: Accelerate 启动器设置环境变量

**文件**: `accelerate/src/accelerate/utils/launch.py:100-145`
**函数**: `_apply_kt_config_to_env()`

启动器读取 `kt_config.yaml` 中的 `kt_config` 节，通过以下映射设置环境变量：

```python
mapping = {
    "lora_rank": "ACCELERATE_KT_LORA_RANK",      # → "12"
    "lora_alpha": "ACCELERATE_KT_LORA_ALPHA",      # → "24"
    "kt_backend": "ACCELERATE_KT_BACKEND",          # → "AMXBF16"
    "kt_num_threads": "ACCELERATE_KT_NUM_THREADS",  # → "64"
    # ... 其他字段
}
```

结果：`ACCELERATE_USE_KT=true`、`ACCELERATE_KT_LORA_RANK=12` 等环境变量被正确设置。

### 阶段 2: TrainingArguments 初始化 HfTrainerKTConfig

**文件**: `transformers/src/transformers/training_args.py:1819-1856`
**位置**: `TrainingArguments.__post_init__()`

```python
# 优先级: self.kt_config > accelerator_config.kt_config > ACCELERATE_USE_KT env var
kt_config_dict = None

# 1. 检查 self.kt_config — 未设置（LLaMA-Factory YAML 没有此字段）
if self.kt_config is not None:
    ...  # 跳过

# 2. 检查 accelerator_config.kt_config — 未设置
if kt_config_dict is None and self.accelerator_config is not None:
    kt_config_dict = getattr(self.accelerator_config, "kt_config", None)  # → None

# 3. kt_config_dict 仍然是 None，但 ACCELERATE_USE_KT=true
if kt_config_dict is not None or strtobool(os.environ.get("ACCELERATE_USE_KT", "false")):
    # ↑ True（因为环境变量）
    self.hf_kt_config = HfTrainerKTConfig(kt_config_dict)  # kt_config_dict = None!
```

**关键问题**: `HfTrainerKTConfig(None)` 被创建，内部 `_kt_config = {}`（空字典）。

### 阶段 3: HfTrainerKTConfig 通过弱引用暴露给全局

**文件**: `transformers/src/transformers/integrations/kt.py:34-37`

```python
class HfTrainerKTConfig:
    def __init__(self, kt_config_dict):
        self._kt_config = kt_config_dict if kt_config_dict is not None else {}
        set_kt_config(self)  # ← 设置全局弱引用
```

**文件**: `transformers/src/transformers/integrations/kt.py:79-92`

```python
_kt_config_weak_ref = weakref.ref(kt_config)  # 全局弱引用

def _get_kt_config():
    return _kt_config_weak_ref()  # 返回 HfTrainerKTConfig 实例
```

### 阶段 4: from_pretrained 触发 MoE 层包装

**文件**: `transformers/src/transformers/modeling_utils.py:5145-5160`

在 `AutoModelForCausalLM.from_pretrained()` 内部：

```python
if is_kt_expert_loading_enabled():
    kt_config = _get_kt_config()   # → HfTrainerKTConfig（_kt_config={}）
    if kt_config is not None:
        wrappers = wrap_moe_layers_with_kt_wrapper(model, kt_config)
        model._kt_wrappers = wrappers
```

### 阶段 5: C++ 内核以 lora_rank=1 初始化（BUG 所在）

**文件**: `accelerate/src/accelerate/utils/kt_moe.py:1641-1642`
**函数**: `wrap_moe_layers_with_kt_wrapper()`

```python
# kt_plugin 是 HfTrainerKTConfig 实例（_kt_config={}）
lora_rank = getattr(kt_plugin, "lora_rank", 1) or 1   # ← 返回 1！
lora_alpha = getattr(kt_plugin, "lora_alpha", 1.0) or 1.0
```

`getattr(kt_plugin, "lora_rank", 1)` 的执行过程：
1. 调用 `HfTrainerKTConfig.__getattr__("lora_rank")`
2. 在 `_kt_config = {}` 中查找 → 不存在
3. `raise AttributeError`
4. `getattr` 返回默认值 `1`

**文件**: `accelerate/src/accelerate/utils/kt_moe.py:1755-1771`

C++ 内核包装器以 `lora_rank=1` 创建：

```python
wrapper = KTMoEWrapper(
    ...
    lora_rank=lora_rank,    # ← 1（错误！）
    lora_alpha=lora_alpha,  # ← 1.0（错误！）
    ...
)
```

### 阶段 6: PEFT LoRA 注入（lora_rank=12）

PEFT 使用 `finetuning_args.lora_rank=12` 创建 LoRA 适配器，这是正确的。

### 阶段 7: kt_adapt_peft_lora 尝试同步 → 形状不匹配

**文件**: `transformers/src/transformers/trainer.py:2571-2573`

```python
if self.is_kt_enabled and kt_adapt_peft_lora is not None:
    kt_model = self.accelerator.unwrap_model(self.model)
    kt_adapt_peft_lora(kt_model)
```

**文件**: `accelerate/src/accelerate/utils/kt_moe.py:2161-2208`
**函数**: `_create_lora_view_buffers()`

```python
lora_rank = gate_lora[0].weight.shape[0]   # ← 12（从 PEFT 实际权重读取）
buffers = {
    "gate_lora_a": torch.zeros(num_experts, lora_rank, hidden_size, ...),
    # → shape = (128, 12, 2048)
}
```

**文件**: `accelerate/src/accelerate/utils/kt_moe.py:2101`

```python
wrapper.wrapper.init_lora_weights(**lora_buffers)
# C++ 内核期望 (128, 1, 2048) 但收到 (128, 12, 2048) → ValueError!
```

**文件**: `ktransformers/kt-kernel/python/utils/amx_sft.py:456-477`

```python
def init_lora_weights(self, **kwargs):
    expected_shapes = {
        "gate_lora_a": (self.num_experts, self.lora_rank, self.hidden_size),
        # → (128, 1, 2048)  因为 self.lora_rank=1
    }
    for name, tensor in provided_tensors.items():
        expected = expected_shapes[name]
        if tensor.shape != expected:
            raise ValueError(f"{name} shape mismatch: expected {expected}, got {tuple(tensor.shape)}")
            # → "gate_lora_a shape mismatch: expected (128, 1, 2048), got (128, 12, 2048)"
```

---

## 两个配置对象的对比

| 特性 | `HfTrainerKTConfig` | `KTransformersPlugin` |
|------|---------------------|-----------------------|
| 定义位置 | `transformers/integrations/kt.py` | `accelerate/utils/dataclasses.py` |
| 创建时机 | `TrainingArguments.__post_init__` | `Trainer.create_accelerator_and_postprocess` |
| 使用场景 | `from_pretrained` 期间（模型加载） | Accelerator 内部（训练期间） |
| 环境变量回退 | **无**（BUG 根因） | **有**（`__post_init__` 中读取） |
| `lora_rank` 值 | `AttributeError` → 默认 1 | 12（从 `ACCELERATE_KT_LORA_RANK` 读取） |

**核心断裂点**: `HfTrainerKTConfig` 在 `from_pretrained` 期间被使用来初始化 C++ 内核，
但它不像 `KTransformersPlugin` 那样读取环境变量。环境变量 `ACCELERATE_KT_LORA_RANK=12`
已经被正确设置，但 `HfTrainerKTConfig` 完全忽略了它。

---

## 修复方案

### 修改文件

`transformers/src/transformers/integrations/kt.py`

### 修改内容

为 `HfTrainerKTConfig` 添加环境变量回退机制，使其与 `KTransformersPlugin` 行为一致。

#### 1. 新增环境变量映射表（模块级）

与 `accelerate/src/accelerate/utils/launch.py:_apply_kt_config_to_env` 中的映射一致：

```python
_KT_ENV_MAPPING: dict[str, tuple[str, type]] = {
    "kt_backend": ("ACCELERATE_KT_BACKEND", str),
    "kt_num_threads": ("ACCELERATE_KT_NUM_THREADS", int),
    "kt_tp_enabled": ("ACCELERATE_KT_TP_ENABLED", bool),
    "kt_threadpool_count": ("ACCELERATE_KT_THREADPOOL_COUNT", int),
    "kt_max_cache_depth": ("ACCELERATE_KT_MAX_CACHE_DEPTH", int),
    "kt_num_gpu_experts": ("ACCELERATE_KT_NUM_GPU_EXPERTS", int),
    "kt_weight_path": ("ACCELERATE_KT_WEIGHT_PATH", str),
    "kt_use_lora_experts": ("ACCELERATE_KT_USE_LORA_EXPERTS", bool),
    "kt_lora_expert_num": ("ACCELERATE_KT_LORA_EXPERT_NUM", int),
    "kt_lora_expert_intermediate_size": ("ACCELERATE_KT_LORA_EXPERT_INTERMEDIATE_SIZE", int),
    "lora_rank": ("ACCELERATE_KT_LORA_RANK", int),
    "lora_alpha": ("ACCELERATE_KT_LORA_ALPHA", float),
    "model_max_length": ("ACCELERATE_KT_MODEL_MAX_LENGTH", int),
    "kt_skip_expert_loading": ("ACCELERATE_KT_SKIP_EXPERT_LOADING", bool),
    "bypass_device_map_check": ("ACCELERATE_KT_BYPASS_DEVICE_MAP", bool),
    "skip_device_placement": ("ACCELERATE_KT_SKIP_DEVICE_PLACEMENT", bool),
}
```

#### 2. 新增 `_parse_env_value` 辅助函数

```python
def _parse_env_value(raw: str, target_type: type) -> Any:
    if target_type is bool:
        return raw.lower() in ("1", "true", "yes")
    if target_type is int:
        return int(raw)
    if target_type is float:
        return float(raw)
    return raw
```

#### 3. 新增 `_get_from_env` 静态方法

```python
@staticmethod
def _get_from_env(name: str) -> Any:
    entry = _KT_ENV_MAPPING.get(name)
    if entry is None:
        return None
    env_key, target_type = entry
    raw = os.environ.get(env_key)
    if raw is None:
        return None
    try:
        return _parse_env_value(raw, target_type)
    except (ValueError, TypeError):
        return raw
```

#### 4. 修改 `__getattr__` — 添加环境变量回退

```python
def __getattr__(self, name: str) -> Any:
    if name.startswith("_"):
        raise AttributeError(name)
    cfg = self.__dict__.get("_kt_config", {})
    if isinstance(cfg, dict) and name in cfg:
        return cfg[name]
    if hasattr(cfg, name):
        return getattr(cfg, name)
    # 新增：回退到环境变量
    env_val = self._get_from_env(name)
    if env_val is not None:
        return env_val
    raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")
```

#### 5. 修改 `_get` — 添加环境变量回退

```python
def _get(self, key: str, default: Any = None) -> Any:
    cfg = self._kt_config
    if isinstance(cfg, dict):
        val = cfg.get(key, None)
    else:
        val = getattr(cfg, key, None)
    if val is not None:
        return val
    # 新增：回退到环境变量
    env_val = self._get_from_env(key)
    if env_val is not None:
        return env_val
    return default
```

### 修复后的数据流

```
accelerate 启动器
  │
  ├─ 读取 kt_config.yaml → kt_config.lora_rank=12
  ├─ 设置 ACCELERATE_KT_LORA_RANK=12
  │
  ▼
TrainingArguments.__post_init__()
  │
  ├─ self.kt_config = None
  ├─ accelerator_config.kt_config = None
  ├─ ACCELERATE_USE_KT = "true" → 进入 KT 分支
  ├─ HfTrainerKTConfig(None) → _kt_config = {}
  │
  ▼
from_pretrained() → wrap_moe_layers_with_kt_wrapper()
  │
  ├─ getattr(kt_plugin, "lora_rank", 1)
  ├─ HfTrainerKTConfig.__getattr__("lora_rank")
  ├─ _kt_config = {} → 未找到
  ├─ 【修复】_get_from_env("lora_rank") → ACCELERATE_KT_LORA_RANK → 12  ✓
  │
  ▼
KTMoEWrapper(..., lora_rank=12)  ✓
  │
  ▼
kt_adapt_peft_lora() → init_lora_weights(gate_lora_a=(128, 12, 2048))
  │
  ▼
C++ 内核期望 (128, 12, 2048) → 匹配！ ✓
```

### 为什么选择在 HfTrainerKTConfig 修复

1. **根因修复**: 问题的本质是 `HfTrainerKTConfig` 缺少环境变量回退，而 `KTransformersPlugin` 有。
   两者应该有对称的行为。

2. **单点修复**: 只修改一个文件 (`transformers/integrations/kt.py`)，不需要改动 accelerate 或
   LLaMA-Factory 的代码。

3. **通用性**: 不仅修复 `lora_rank`，也修复所有通过环境变量传递的 KT 配置属性
   （如 `lora_alpha`、`kt_backend` 等），防止类似问题在其他属性上出现。

4. **向后兼容**: 环境变量只在 dict 中找不到时才作为回退，不会覆盖显式配置。
