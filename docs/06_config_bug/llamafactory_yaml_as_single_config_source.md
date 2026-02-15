# LLaMA Factory YAML 作为 KTransformers 唯一配置源

## 背景与问题

当前 KT 配置分散在两个 YAML 文件中，存在参数重复、值冲突、以及配置断裂的问题：

### 问题 1: LLaMA Factory YAML 中的 KT 参数是死代码

LLaMA Factory 训练 YAML（如 `qwen3moe_lora_sft_kt.yaml`）中的 KT 参数：

```yaml
use_kt: true
kt_backend: AMXBF16
kt_num_threads: 60
lora_rank: 12
```

这些字段被正确解析到 `ModelArguments` 和 `FinetuningArguments`，但运行时不被 KT 初始化流程消费。
KT 初始化依赖的是 `HfTrainerKTConfig`，而该对象从 `TrainingArguments.kt_config` 或
`accelerator_config.kt_config` 构建——这两个字段 LLaMA Factory YAML 都没有设置。

### 问题 2: accelerate YAML 的 kt_config 传递断裂

accelerate 配置（`examples/accelerate/kt_config.yaml`）中的 `kt_config:` 节：

```yaml
kt_config:
  enabled: true
  kt_backend: AMXBF16
  kt_num_threads: 64
  lora_rank: 12
  lora_alpha: 24
  model_max_length: 512
```

由 `accelerate launch` 启动器解析，通过 `_apply_kt_config_to_env()` 转为环境变量
（如 `ACCELERATE_KT_LORA_RANK=12`）。但 `TrainingArguments.__post_init__` 中：

1. `self.kt_config` 为 None（LLaMA Factory 没设置这个字段）
2. `accelerator_config.kt_config` 也为 None（accelerate 启动器不设置这个属性）
3. 仅因 `ACCELERATE_USE_KT=true` 进入 KT 分支
4. 结果：`HfTrainerKTConfig(None)` 被创建，`_kt_config = {}`（空字典）

这导致 `from_pretrained` 阶段 `wrap_moe_layers_with_kt_wrapper()` 读取到的 `lora_rank` 等参数
全部回退到硬编码默认值（`lora_rank=1`、`lora_alpha=1.0`），引发 shape mismatch 错误。

详见 [lora_rank_shape_mismatch.md](./lora_rank_shape_mismatch.md)。

### 问题 3: 用户需要在两个文件中维护重复参数

`lora_rank`、`lora_alpha`、`kt_backend`、`kt_num_threads` 等参数需要同时出现在：
- LLaMA Factory 训练 YAML（供 PEFT / HfArgumentParser 使用）
- accelerate kt_config.yaml（供 KT MoE 内核使用）

两处不一致时行为不可预测，且用户难以排查。

---

## 解决方案

**核心思路**：让 LLaMA Factory 训练 YAML 成为唯一配置源。在 `get_train_args()` 的后处理阶段，
从已解析的 dataclass 中收集所有 KT 相关参数，主动注入到 `TrainingArguments`，
使得后续 `from_pretrained` 和 `Trainer` 都能拿到完整的 KT 配置。

---

## 改动详情

### 改动 1: 在 parser.py 中新增 `_inject_kt_config()` 函数

**文件**: `LLaMA-Factory-KT/src/llamafactory/hparams/parser.py`

#### 新增函数

```python
def _inject_kt_config(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "TrainingArguments",
    finetuning_args: "FinetuningArguments",
) -> None:
    if not model_args.use_kt:
        return

    kt_config_dict = {
        "enabled": True,
        "kt_skip_expert_loading": True,
        "kt_backend": model_args.kt_backend,
        "kt_num_threads": model_args.kt_num_threads,
        "kt_tp_enabled": model_args.kt_tp_enabled,
        "kt_threadpool_count": model_args.kt_threadpool_count,
        "kt_max_cache_depth": model_args.kt_max_cache_depth,
        "kt_num_gpu_experts": model_args.kt_num_gpu_experts,
        "kt_weight_path": model_args.kt_weight_path,
        "kt_use_lora_experts": model_args.kt_use_lora_experts,
        "kt_lora_expert_num": model_args.kt_lora_expert_num,
        "kt_lora_expert_intermediate_size": model_args.kt_lora_expert_intermediate_size,
        "lora_rank": finetuning_args.lora_rank,
        "lora_alpha": finetuning_args.lora_alpha,
        "model_max_length": data_args.cutoff_len,
    }

    from transformers.integrations.kt import HfTrainerKTConfig

    training_args.hf_kt_config = HfTrainerKTConfig(kt_config_dict)
    os.environ["ACCELERATE_USE_KT"] = "true"

    if training_args.accelerator_config is not None:
        training_args.accelerator_config.kt_config = kt_config_dict
```

该函数执行三件事：

1. **构建完整的 `kt_config_dict`**：从 `model_args`（KT 后端参数）、`finetuning_args`（LoRA 参数）、
   `data_args`（序列长度）中收集所有 KT 内核需要的配置，键名与 `KTransformersPlugin` 字段名一致。

2. **创建 `HfTrainerKTConfig` 并挂载到 `training_args.hf_kt_config`**：
   - 构造函数内部调用 `set_kt_config(self)` 设置全局 weakref
   - 后续 `from_pretrained` 通过 `_get_kt_config()` 拿到这个完整的 config
   - 存在 `training_args` 上保持强引用，防止 weakref 被 GC

3. **设置 `ACCELERATE_USE_KT` 环境变量和 `accelerator_config.kt_config`**：
   - 确保 `Trainer.create_accelerator_and_postprocess()` 中 `KTransformersPlugin` 以显式 kwargs 创建
   - `KTransformersPlugin` 的 `__post_init__` 中，所有字段都已有显式值，不再回退到环境变量

#### 调用位置

在 `get_train_args()` 中，所有后处理完成之后、日志输出之前调用：

```python
    # 后处理完成，lora_alpha 已计算（rank*2），cutoff_len 已就绪
    model_args.model_max_length = data_args.cutoff_len
    model_args.block_diag_attn = data_args.neat_packing
    data_args.packing = data_args.packing if data_args.packing is not None else finetuning_args.stage == "pt"

    # ---- 新增 ----
    _inject_kt_config(model_args, data_args, training_args, finetuning_args)

    # Log on each process the small summary
    logger.info(...)
```

**时序关键**：必须在 `finetuning_args.__post_init__` 之后调用，因为 `lora_alpha` 在那里被计算为
`lora_rank * 2`。放在此位置保证所有 derived 值都已就绪。

#### kt_config_dict 字段来源

| 键 | 来源 | 说明 |
|---|---|---|
| `enabled` | 固定 `True` | `use_kt=True` 时必然启用 |
| `kt_skip_expert_loading` | 固定 `True` | KT 包装器自行加载专家权重 |
| `kt_backend` | `model_args.kt_backend` | 默认 `"AMXBF16"` |
| `kt_num_threads` | `model_args.kt_num_threads` | 默认 `60` |
| `kt_tp_enabled` | `model_args.kt_tp_enabled` | 默认 `True` |
| `kt_threadpool_count` | `model_args.kt_threadpool_count` | 默认 `4` |
| `kt_max_cache_depth` | `model_args.kt_max_cache_depth` | 默认 `2` |
| `kt_num_gpu_experts` | `model_args.kt_num_gpu_experts` | 默认 `0` |
| `kt_weight_path` | `model_args.kt_weight_path` | 默认 `None` |
| `kt_use_lora_experts` | `model_args.kt_use_lora_experts` | 默认 `False` |
| `kt_lora_expert_num` | `model_args.kt_lora_expert_num` | 默认 `2` |
| `kt_lora_expert_intermediate_size` | `model_args.kt_lora_expert_intermediate_size` | 默认 `1024` |
| `lora_rank` | `finetuning_args.lora_rank` | 默认 `8` |
| `lora_alpha` | `finetuning_args.lora_alpha` | `__post_init__` 中已计算为 `rank*2` |
| `model_max_length` | `data_args.cutoff_len` | 默认 `1024` |

**不包含的字段**（有合理默认值或由运行时动态设置）：
- `kt_checkpoint_files`、`kt_sharded_metadata`：由 `from_pretrained` 动态填充
- `bypass_device_map_check`：默认 `True`
- `skip_device_placement`：默认 `True`

---

### 改动 2: 精简 accelerate 配置 YAML

**文件**: `LLaMA-Factory-KT/examples/accelerate/kt_config.yaml`

删除整个 `kt_config:` 节，只保留 accelerate 启动器必需的字段：

```yaml
compute_environment: LOCAL_MACHINE
distributed_type: NO
num_processes: 1
```

**变更前**:
```yaml
compute_environment: LOCAL_MACHINE
distributed_type: NO
num_processes: 1

kt_config:
  enabled: true
  kt_backend: AMXBF16
  kt_num_threads: 64
  kt_tp_enabled: true
  kt_threadpool_count: 2
  kt_num_gpu_experts: 0
  kt_use_lora_experts: false
  kt_lora_expert_num: 0
  kt_lora_expert_intermediate_size: 1024
  lora_rank: 12
  lora_alpha: 24
  model_max_length: 512
```

---

### 改动 3: 删除 loader.py 中的注释死代码

**文件**: `LLaMA-Factory-KT/src/llamafactory/model/loader.py`

删除内容：

1. 注释掉的 `load_kt_model()` 调用块（原 lines 147-154）：
```python
    # KTransformers MoE backend (handles MoE layers with CPU AMX acceleration)
    # if model_args.use_kt:
    #     if not KT_KERNEL_AVAILABLE:
    #         raise ImportError(
    #             "kt_kernel not found. Please install kt_kernel to use 'use_kt'."
    #         )
    #     logger.info_rank0("Loading model with KTransformers MoE backend")
    #     model = load_kt_model(config, model_args, finetuning_args)
```

2. 因此变为未使用的导入：
```python
from accelerate.utils import KT_KERNEL_AVAILABLE, load_kt_model
```

---

## 改动后的执行流程

```
get_train_args()
  ├─ _parse_train_args()
  │    └─ TrainingArguments.__post_init__()
  │         └─ kt_config=None, accelerator_config.kt_config=None
  │            → 如果 ACCELERATE_USE_KT 在 env 中: 创建空的 HfTrainerKTConfig
  │            → 如果不在 env 中: 跳过 KT 初始化
  │
  ├─ 参数校验 & 后处理
  │    ├─ finetuning_args.__post_init__(): lora_alpha = lora_rank * 2
  │    ├─ model_args.model_max_length = data_args.cutoff_len
  │    └─ ... 其他后处理 ...
  │
  └─ _inject_kt_config()                          ← 新增
       ├─ 构建 kt_config_dict（完整，包含所有参数）
       ├─ HfTrainerKTConfig(完整 dict)
       │    └─ set_kt_config() → 更新全局 weakref（覆盖之前可能创建的空 config）
       ├─ training_args.hf_kt_config = 新 config   （强引用防止 GC）
       ├─ os.environ["ACCELERATE_USE_KT"] = "true"
       └─ accelerator_config.kt_config = dict

run_sft()
  ├─ load_model()
  │    └─ from_pretrained()
  │         └─ _get_kt_config()
  │              → 通过 weakref 拿到完整的 HfTrainerKTConfig
  │         └─ wrap_moe_layers_with_kt_wrapper(model, kt_config)
  │              └─ getattr(kt_config, "lora_rank", 1) → 12  ✓
  │              └─ getattr(kt_config, "kt_num_threads", 1) → 60  ✓
  │              └─ KTMoEWrapper(..., lora_rank=12, ...)  ✓
  │
  └─ Trainer.__init__()
       └─ create_accelerator_and_postprocess()
            └─ KTransformersPlugin(**kt_config_dict)
                 → 所有字段已有显式值，__post_init__ 中不回退到环境变量
```

---

## 边界情况分析

### 旧 accelerate YAML 未清理

如果用户忘记删除 accelerate YAML 中的 `kt_config:` 节：

1. `accelerate launch` 启动器照常设置环境变量
2. `TrainingArguments.__post_init__` 可能从 env var 创建一个空的 `HfTrainerKTConfig`
3. 我们的 `_inject_kt_config()` 随后用完整 dict 覆盖 `hf_kt_config` 和 `accelerator_config.kt_config`
4. **LLaMA Factory 的值总是胜出**

### 不用 accelerate launch 直接运行

```bash
python src/train.py config.yaml
```

1. 没有环境变量，`TrainingArguments.__post_init__` 完全跳过 KT
2. 我们的 `_inject_kt_config()` 正常设置一切
3. **正常工作**

### 用户同时在 TrainingArguments 设了 kt_config

1. `__post_init__` 用它创建一个 `HfTrainerKTConfig`
2. 我们的 `_inject_kt_config()` 用 LLaMA Factory YAML 的值覆盖它
3. **LLaMA Factory YAML 优先**（单一配置源原则）

### use_kt=False

1. `_inject_kt_config()` 检查 `model_args.use_kt`，为 False 直接返回
2. **无任何副作用**

---

## 与 env var fallback 修改的关系

[lora_rank_shape_mismatch.md](./lora_rank_shape_mismatch.md) 中提出了另一个修改方案：
在 `HfTrainerKTConfig` 中添加环境变量回退（`_get_from_env`），使其在 dict 中找不到值时
从 `ACCELERATE_KT_*` 环境变量读取。

**两个方案的关系**：

| | `_inject_kt_config`（本方案） | env var fallback |
|---|---|---|
| 修改位置 | LLaMA Factory `parser.py` | transformers `integrations/kt.py` |
| 解决范围 | LLaMA Factory 使用场景 | 所有使用 `HfTrainerKTConfig` 的场景 |
| 原理 | 在源头构建完整 dict，不依赖 env var | 让 `HfTrainerKTConfig` 自行读取 env var |

**对于 LLaMA Factory 用户**，本方案已完全解决 shape mismatch 问题，env var fallback 不是必需的。

**对于直接用 accelerate + transformers 的用户**（不经过 LLaMA Factory），仍需要 env var fallback。

两个修改互不冲突，可以并存作为 defense-in-depth。

---

## 验证方法

1. 确认 accelerate YAML 中不再有 `kt_config:` 节
2. 确认 LLaMA Factory 训练 YAML 中有正确的 KT 参数（`use_kt`、`lora_rank` 等）
3. 运行训练命令：
   ```bash
   CUDA_VISIBLE_DEVICES=1 LD_PRELOAD=~/mimalloc/build/libmimalloc.so accelerate launch \
     --num_cpu_threads_per_process 64 \
     --config_file examples/accelerate/kt_config.yaml \
     src/train.py examples/ktransformers/train_lora/qwen3moe_lora_sft_kt.yaml
   ```
4. 检查日志：
   - 应出现 `Injected KT config from LLaMA Factory args: backend=AMXBF16, threads=60, lora_rank=12, lora_alpha=24, model_max_length=...`
   - `[kt_moe INIT]` 应显示正确的 `lora_rank`（12，非 1）
   - `WorkerPool` 应显示正确的线程数（如 `[numa:threads][0:60]`）
   - 不再出现 `gate_lora_a shape mismatch` 错误
5. 训练应正常开始前向/后向传播
