# KT 模式下模型加载问题

本文档记录了 KT 模式加载过程中遇到的两个连锁问题及其解决方案。

---

## 问题 1: GPU 显存被占满

### 现象

使用 KTransformers (KT) 后端训练时，启动后 GPU 显存几乎被占满。
KT 的设计是将 MoE 专家层放在 CPU 上用 AMX 加速，GPU 只需放置 Attention、Embedding、
Router、Shared Experts 等非专家参数。GPU 显存占用应该很小。

### 根因

`parser.py:509` 无条件设置 `model_args.device_map = {"": get_current_device()}`，
即 `{"": "cuda:0"}`。这告诉 `from_pretrained` → `dispatch_model` 把所有参数都放到 GPU，
包括 KT 已跳过加载的专家空 placeholder tensor。

虽然 `kt_skip_expert_loading` 跳过了专家权重的实际加载（不从 checkpoint 读取），
但 `torch.empty(shape, device="cpu")` 创建的 placeholder tensor 仍会被
`dispatch_model` 移到 GPU，分配大量显存。

---

## 问题 2: Meta tensor 导致 LoRA 创建失败

### 现象

尝试用混合 device_map（experts→cpu, others→cuda）修复问题 1 后，出现新错误：

```
File "accelerate/src/accelerate/utils/kt_moe.py", line 2205, in _create_lora_view_buffers
    buffers[key_a][expert_idx].copy_(lora_A.weight.data.to(dtype=dtype))
NotImplementedError: Cannot copy out of meta tensor; no data!
```

以及 accelerate 的警告：

```
Some parameters are on the meta device because they were offloaded to the cpu.
```

### 根因

当 `dispatch_model` 收到一个混合 device_map（部分模块在 GPU、部分在 CPU）时，
accelerate 采用 **CPU offload 策略**：

1. 把 CPU 映射的模块参数替换为 **meta tensor**（不占实际内存）
2. 添加 `AlignDevicesHook`
3. forward 前将参数从 CPU 缓存恢复到模块
4. forward 后重新 offload 为 meta

这导致专家模块的参数变成 meta tensor。之后 PEFT 在 meta tensor 上创建 LoRA 权重，
LoRA 权重也变成 meta。当 `kt_adapt_peft_lora` 尝试 copy LoRA 权重时，失败。

### 为什么混合 device_map 不可行

```
from_pretrained(device_map={experts→"cpu", others→"cuda:0"})
  ├─ _load_pretrained_model(): 专家 key 过滤，非专家加载
  ├─ KT wrapping: MoE → KTMoELayerWrapper
  └─ dispatch_model(): ← 问题发生处
       ├─ 非专家 → cuda:0 ✓
       └─ 专家 → "cpu" → meta tensor offload ✗
            └─ PEFT creates LoRA on meta → copy fails ✗
```

---

## 解决方案: CPU-first 加载 + 手动 GPU 迁移

### 核心思路

1. 用 `device_map="cpu"` 将整个模型加载到 CPU（`dispatch_model` 不做 offload）
2. KT wrapping 在 `from_pretrained` 中完成（C++ 内核加载专家权重）
3. PEFT LoRA 在 CPU 上创建（所有 tensor 都是真实的，不是 meta）
4. 最后手动将非专家部分移到 GPU

这个方案利用了 `kt_loader.py` 中已有的 `get_kt_loading_kwargs()` 和
`move_non_experts_to_gpu()` 函数。

### 改动 1: `patcher.py` — 设置 device_map="cpu"

**文件**: `LLaMA-Factory-KT/src/llamafactory/model/patcher.py`

```python
    if not (is_deepspeed_zero3_enabled() or is_fsdp_enabled()) and init_kwargs["low_cpu_mem_usage"]:
        if "device_map" not in init_kwargs and model_args.device_map:
            init_kwargs["device_map"] = model_args.device_map

        if init_kwargs.get("device_map", None) == "auto":
            init_kwargs["offload_folder"] = model_args.offload_folder

        # KT: load entire model to CPU first, then move non-experts to GPU
        # after PEFT init (in loader.py). We cannot use a mixed device_map
        # (experts=cpu, others=cuda) because accelerate's dispatch_model
        # converts CPU-mapped params to meta tensors (offload strategy),
        # which breaks PEFT LoRA creation on expert modules.
        if getattr(model_args, "use_kt", False):
            init_kwargs["device_map"] = "cpu"
            logger.info_rank0("KT mode: loading model to CPU, ...")
```

### 改动 2: `loader.py` — PEFT 后迁移非专家到 GPU

**文件**: `LLaMA-Factory-KT/src/llamafactory/model/loader.py`

在 `init_adapter()` 之后，添加 GPU 迁移：

```python
    model = init_adapter(config, model, model_args, finetuning_args, is_trainable)

    # KT: move non-expert parts to GPU after PEFT LoRA init.
    if getattr(model_args, "use_kt", False) and is_trainable:
        from accelerate.utils.kt_moe import get_moe_arch_config
        from .model_utils.kt_loader import move_non_experts_to_gpu

        moe_config = get_moe_arch_config(config)
        # init_adapter returns PeftModel; move_non_experts_to_gpu expects
        # the original PreTrainedModel (accesses model.model.layers)
        base_model = model.get_base_model() if hasattr(model, "get_base_model") else model
        move_non_experts_to_gpu(base_model, moe_config)
        # Set hf_device_map so Trainer skips _move_model_to_device
        model.hf_device_map = {"non_experts": 0, "experts": "cpu"}
```

### 改动 3: `kt_loader.py` — 修复 import 路径

**文件**: `LLaMA-Factory-KT/src/llamafactory/model/model_utils/kt_loader.py`

`from .kt_moe import ...` → `from accelerate.utils.kt_moe import ...`
（`kt_moe.py` 实际位于 accelerate 包中，不在 LLaMA Factory 中）

---

## 修复后的完整加载流程

```
get_train_args()
  └─ _inject_kt_config()                    ← 06_config_bug 的修复
       └─ 把 LLaMA Factory YAML 参数注入 TrainingArguments

load_model()
  ├─ patch_config()
  │    └─ use_kt=True → init_kwargs["device_map"] = "cpu"
  │
  ├─ from_pretrained(device_map="cpu")
  │    ├─ _load_pretrained_model()
  │    │    ├─ kt_skip_expert_loading → 过滤专家 key
  │    │    ├─ 非专家权重 → CPU tensor ✓
  │    │    └─ 专家 meta tensor → empty CPU tensor ✓ (真实 tensor，非 meta)
  │    │
  │    ├─ KT wrapping (line 5145)
  │    │    ├─ MoE 模块 → KTMoELayerWrapper
  │    │    └─ C++ AMX 内核加载专家权重
  │    │
  │    └─ dispatch_model(device_map={"": "cpu"})
  │         └─ 所有模块都在 CPU → 单设备 → 无 offload hooks ✓
  │
  ├─ init_adapter()  (PEFT LoRA)
  │    ├─ 在 CPU tensor 上创建 LoRA 权重 ✓ (非 meta)
  │    └─ 返回 PeftModel
  │
  └─ move_non_experts_to_gpu()               ← 本次修复的关键
       ├─ model.model.embed_tokens → cuda:0
       ├─ model.model.norm → cuda:0
       ├─ model.lm_head → cuda:0
       ├─ 每层:
       │    ├─ self_attn → cuda:0 (含 Attention LoRA)
       │    ├─ input_layernorm → cuda:0
       │    ├─ post_attention_layernorm → cuda:0
       │    ├─ Dense 层: mlp → cuda:0
       │    └─ MoE 层:
       │         ├─ router (gate) → cuda:0
       │         ├─ shared_experts → cuda:0
       │         └─ routed experts → 留在 CPU ✓
       │              └─ 专家 LoRA 也留在 CPU ✓
       │                   (kt_adapt_peft_lora 稍后处理)
       └─ 设置 model.hf_device_map → Trainer 不再移动模型

trainer.train()
  └─ kt_adapt_peft_lora()
       └─ 从 CPU 上的专家 LoRA 权重创建 KT buffer ✓
```

---

## `move_non_experts_to_gpu()` 工作原理

**文件**: `kt_loader.py`

```python
def move_non_experts_to_gpu(model, moe_config, device="cuda:0"):
    # 1. 全局层 → GPU
    model.model.embed_tokens.to(device)
    model.model.norm.to(device)
    model.lm_head.to(device)

    for layer_idx, layer in enumerate(model.model.layers):
        # 2. Attention + LayerNorm → GPU
        layer.self_attn.to(device)
        layer.input_layernorm.to(device)
        layer.post_attention_layernorm.to(device)

        # 3. 判断是 Dense 还是 MoE
        moe_module = getattr(layer, moe_config.moe_layer_attr, None)
        if moe_module is None or not hasattr(moe_module, moe_config.experts_attr):
            # Dense 层（如 DeepSeek-V2 的 layer 0）→ 整个 MLP 到 GPU
            layer.mlp.to(device)
            continue

        # 4. MoE 层：只移动 router 和 shared_experts
        router = getattr(moe_module, moe_config.router_attr, None)
        if router is not None:
            router.to(device)

        if hasattr(moe_module, "shared_experts") and moe_module.shared_experts is not None:
            moe_module.shared_experts.to(device)

        # 5. Routed experts 留在 CPU（KT C++ 内核管理）
```

**PeftModel 适配**：`init_adapter` 返回 PeftModel，但 `move_non_experts_to_gpu`
需要原始模型（访问 `model.model.layers`）。通过 `model.get_base_model()` 获取原始
`DeepseekV2ForCausalLM`。

**Trainer 适配**：设置 `model.hf_device_map` 使 `Trainer._move_model_to_device()`
检测到多设备模型并跳过自动迁移（trainer.py:922-926）。

---

## 最终设备分布

| 组件 | 设备 | 说明 |
|---|---|---|
| Embedding / LM Head | GPU | 需要 GPU 计算 |
| Attention (+ LoRA) | GPU | 需要 GPU 计算 |
| LayerNorm | GPU | 需要 GPU 计算 |
| Router (gate) | GPU | 路由计算在 GPU |
| Shared Experts | GPU | 共享专家走正常 forward |
| **Routed Experts (placeholder)** | **CPU** | KT C++ 内核管理实际权重 |
| **Expert LoRA** | **CPU** | kt_adapt_peft_lora 处理 |

---

## 为什么不使用 build_kt_device_map（混合 device_map）

虽然 `build_kt_device_map()` 构建了一个合理的混合 device_map
（experts→cpu, others→cuda），但它不可用于实际加载，原因：

1. **accelerate dispatch_model 的 offload 行为**：
   混合 device_map 触发 `dispatch_model` 的 CPU offload 策略，
   将 CPU 映射的参数替换为 meta tensor + AlignDevicesHook

2. **PEFT LoRA 在 meta tensor 上失败**：
   PEFT 在 meta tensor 上创建 LoRA 权重 → 权重也是 meta →
   `kt_adapt_peft_lora` 的 `copy_()` 操作失败

3. **Dense 层的 warning**：
   对于 DeepSeek-V2 等有 `first_k_dense_replace=1` 的模型，
   layer 0 没有 experts 子模块，build_kt_device_map 生成的 expert key
   不匹配，产生大量 warning

`build_kt_device_map()` 和 `build_kt_device_map_simplified()` 仍保留在
`kt_loader.py` 中，仅作为参考实现。实际加载流程使用 `device_map="cpu"` +
`move_non_experts_to_gpu()`。

---

## 与其他修复的关系

| | 06_config_bug: `_inject_kt_config` | 07_load_bug: 本修复 |
|---|---|---|
| 问题 | KT 参数为空/默认值 | GPU 显存占满 + meta tensor |
| 原因 | HfTrainerKTConfig 拿到空 dict | device_map 和 dispatch_model |
| 修复位置 | `parser.py` | `patcher.py` + `loader.py` |
| 修复时机 | 参数解析阶段 | 模型加载阶段 |

---

## 验证方法

1. 运行训练命令
2. 观察日志：
   - 应出现 `KT mode: loading model to CPU, will move non-experts to GPU after LoRA init.`
   - 应出现 `Moved non-expert parameters to cuda:0`
3. 观察 GPU 显存：
   - `nvidia-smi` 应显示 GPU 显存占用远小于模型总参数量
4. 训练应正常启动并进行前向/后向传播
5. `kt_adapt_peft_lora` 不应再报 meta tensor 错误

---

## 关于 pyinstrument 报错

日志末尾的 `TypeError: 'NoneType' object is not callable` 来自 pyinstrument 的
清理代码（`stack_sampler.py`），发生在 Python 进程退出时。这不是根因，只是进程因
主错误崩溃退出时的附带效应。修复主错误后此问题会自行消失。
