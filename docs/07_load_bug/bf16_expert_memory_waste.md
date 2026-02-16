# BF16 Expert 权重冗余加载问题（内存浪费 Bug）

## 问题描述

在 KTransformers (KT) + FSDP2 训练流程中，DeepSeek-V3 671B 模型训练时被 Linux OOM Killer
杀掉 (exitcode: -9)。系统总 RAM 2015 GB，可用约 1354 GB，但在加载过程中 CPU 内存耗尽。

根本原因：**BF16 专家权重被冗余加载了一次**，在整个加载过程中存在两份完整的 BF16 专家数据。

---

## 内存浪费计算

### DeepSeek-V3 671B
- 58 个 MoE 层, 256 experts/层
- 每个 expert: gate_proj + up_proj + down_proj
  - hidden_size=7168, intermediate_size=2048
  - 每个 expert ≈ 3 × (7168 × 2048) × 2 bytes(BF16) ≈ 84 MB
- 每层: 256 × 84 MB ≈ 21 GB
- **全部 BF16 experts: 58 × 21 GB ≈ 1218 GB**
- INT8 量化后 (KT kernel): ≈ 609 GB

峰值内存 = 1218 GB (冗余 BF16) + 单层 BF16 临时 (~21 GB) + 累积 INT8 (~609 GB) + 非专家参数 + overhead
→ 远超可用 RAM → OOM Killed

### Qwen3-235B-A22B
- 94 层(多数为 MoE), 128 experts/层
- 每个 expert ≈ 3 × (4096 × 1536) × 2 bytes ≈ 36 MB (近似值)
- 每层: 128 × 36 MB ≈ 4.5 GB
- **全部 BF16 experts: ~94 × 4.5 GB ≈ 423 GB**

即使 235B 不会被 OOM Killed，423 GB 的冗余内存浪费也是不可接受的。

---

## Bug 根因追踪

### 涉及文件

| 文件 | 位置 | 角色 |
|------|------|------|
| `transformers/modeling_utils.py` | L874-912 | **Bug 所在**: 冗余 BF16 加载 |
| `transformers/modeling_utils.py` | L483-500 | 第一次过滤: `load_state_dict` 跳过 expert keys |
| `transformers/modeling_utils.py` | L5435-5444 | expert key mapping 构建 |
| `transformers/modeling_utils.py` | L5625-5638 | 后处理: meta → empty CPU placeholder |
| `accelerate/utils/kt_moe.py` | L1696-1840 | KT wrapping 循环 |
| `accelerate/utils/kt_moe.py` | L709-770 | `load_experts_from_checkpoint_files()` |
| `accelerate/utils/kt_moe.py` | L431-484 | `extract_moe_weights()` (fallback) |
| `accelerate/utils/kt_moe.py` | L487-567 | `_clear_original_expert_weights()` |

### 完整加载时序

```
from_pretrained()
  │
  ├─ _load_pretrained_model()  (modeling_utils.py:5435)
  │    │
  │    ├─ 构建 kt_expert_key_mapping
  │    │   将 expert keys 从 key_renaming_mapping 分离出去
  │    │   非 expert keys → 正常加载路径
  │    │
  │    ├─ 循环处理每个 shard file:
  │    │   │
  │    │   ├─ load_state_dict()  (L483-500)
  │    │   │   跳过 expert keys (skip_kt_experts=True)
  │    │   │   ✓ 正确：expert keys 不进入 state_dict
  │    │   │
  │    │   ├─ _load_state_dict_into_meta_model()  (L860-869)
  │    │   │   非 expert 权重 → 加载到 model params
  │    │   │   ✓ 正确
  │    │   │
  │    │   └─ ★ BUG BLOCK ★  (L874-912)
  │    │       if kt_expert_key_mapping and not kt_weight_path:
  │    │           重新打开 shard file
  │    │           读取 BF16 expert 权重
  │    │           _load_state_dict_into_meta_model(device_map={"":"cpu"})
  │    │           → expert 权重作为 BF16 CPU tensor 进入 model params
  │    │
  │    │       ❌ 这些 BF16 权重在后续 wrapping 中 **从未被使用**
  │    │       ❌ 它们占据大量内存直到 _clear_original_expert_weights() 释放
  │    │
  │    ├─ 所有 shard 处理完毕后:
  │    │   placeholder 后处理 (L5625-5638)
  │    │   → 但 bug block 已经将 expert params 变成了 BF16 CPU tensor (非 meta)
  │    │   → 所以 placeholder 代码跳过它们 (只处理 meta device 上的参数)
  │    │
  │    └─ stash checkpoint files  (L921-929)
  │        将 shard file 路径存到 kt_config._kt_config["kt_checkpoint_files"]
  │        ✓ 正确：为 wrapping 循环准备
  │
  ├─ KT wrapping  (kt_moe.py:1696-1840)
  │    │
  │    ├─ use_checkpoint_files = True  (因为 checkpoint_files 已 stash)
  │    │
  │    └─ 循环每个 MoE layer:
  │        │
  │        ├─ load_experts_from_checkpoint_files()  (L1734-1742)
  │        │   从磁盘 safetensors 文件直接读取 BF16 expert 权重
  │        │   ❌ 与 bug block 的加载完全重复！
  │        │
  │        ├─ KTMoEWrapper.load_weights_from_tensors()  (L1796-1801)
  │        │   将 BF16 转为 INT8 存入 C++ kernel
  │        │
  │        ├─ del gate_proj, up_proj, down_proj  (L1828)
  │        │   释放本层临时 BF16 数据
  │        │
  │        └─ _clear_original_expert_weights()  (L1836)
  │            将 model 中的 expert params 替换为 1-byte fake tensor
  │            ❌ 这才释放了 bug block 加载的 BF16 数据
  │            ❌ 但只释放当前层，其他层的 BF16 仍在内存中
  │
  └─ 返回 model (expert params 已是 1-byte fake tensor)
```

### 内存时间线（修复前）

```
时间 →
内存 ↑
     ┌───────────────────────────────────────────────────────────┐
     │                                                           │
  1218GB │████████████████████ BF16 experts in model (bug block)█████│
     │███████████████████████████████████████████████████████████│
     │                                                    ↓      │
     │                                              clear_original│
     │                                                           │
   609GB │                    ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ INT8 in C++ kernel ▓│
     │                    ▓ (growing per layer) ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓│
     │                    ↑                                      │
     │                wrapping starts                            │
     │                                                           │
    21GB │                    ░ per-layer BF16 temp (checkpoint read)│
     │                    ░ (load + del, per layer)              │
     │                                                           │
     └───────────────────────────────────────────────────────────┘
      load_shard   wrap layer 0  ...  wrap layer N    done
```

**峰值 ≈ 1218 GB (BF16 in model) + ~609 GB (INT8 final) + ~21 GB (per-layer temp) ≈ 1848 GB**

→ 超出可用 RAM (1354 GB) → OOM Killed

### 内存时间线（修复后）

```
时间 →
内存 ↑
     ┌───────────────────────────────────────────────────────────┐
     │                                                           │
   609GB │                    ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ INT8 in C++ kernel ▓│
     │                    ▓ (growing per layer) ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓│
     │                                                           │
    21GB │                    ░ per-layer BF16 temp (checkpoint read)│
     │                    ░ (load + del, per layer)              │
     │                                                           │
     └───────────────────────────────────────────────────────────┘
      load_shard   wrap layer 0  ...  wrap layer N    done
```

**峰值 ≈ 609 GB (INT8 final) + ~21 GB (per-layer temp) + 非专家参数 ≈ 650 GB**

→ 远小于可用 RAM → 正常运行

---

## Bug Block 详解

**文件**: `transformers/src/transformers/modeling_utils.py` L874-912

```python
# L874: 在每个 shard file 处理完非 expert 权重后
if kt_expert_key_mapping:
    kt_config = _get_kt_config()
    kt_weight_path = getattr(kt_config, "kt_weight_path", None) if kt_config is not None else None

    if not kt_weight_path:
        # ❌ BUG: 下面这段代码将 BF16 expert 权重加载到 model params 中
        # 但这些权重在 KT wrapping 时完全不被使用！
        # wrapping 循环总是使用 load_experts_from_checkpoint_files() 从磁盘直接读取
        expert_state: dict[str, torch.Tensor] = {}
        expert_keys_in_shard = []
        if shard_file.endswith(".safetensors"):
            with safe_open(shard_file, framework="pt") as f:
                shard_keys = set(f.keys())
                expert_keys_in_shard = [k for k in kt_expert_key_mapping if k in shard_keys]
                for k in expert_keys_in_shard:
                    expert_state[kt_expert_key_mapping[k]] = f.get_tensor(k)  # ← 读取 BF16 tensor
        else:
            full_state_dict = torch.load(shard_file, ...)
            expert_keys_in_shard = [k for k in kt_expert_key_mapping if k in full_state_dict]
            for k in expert_keys_in_shard:
                expert_state[kt_expert_key_mapping[k]] = full_state_dict[k]

        if expert_state:
            _load_state_dict_into_meta_model(
                model, expert_state, shard_file, ...,
                device_map={"": "cpu"},  # ← 加载到 CPU
                ...
            )
    else:
        # ✓ 当 kt_weight_path 已设置时正确跳过
        pass
```

### 为什么这些 BF16 权重从未被使用

在 KT wrapping 循环 (`accelerate/utils/kt_moe.py:1710-1747`) 中：

```python
if is_rank_0:
    if use_kt_weight_path:
        # 路径 A: 从 kt_weight_path 加载 INT8, 同时从 checkpoint 读 BF16 for backward
        gate_proj, up_proj, down_proj = load_experts_from_checkpoint_files(...)
    elif use_checkpoint_files:        # ← 当 kt_weight_path=None 时走这条路
        # 路径 B: 从 checkpoint files 直接读 BF16 expert 权重
        gate_proj, up_proj, down_proj = load_experts_from_checkpoint_files(...)  # ← 从磁盘读！
    else:
        # 路径 C: fallback，从 model params 中提取
        gate_proj, up_proj, down_proj = extract_moe_weights(moe_module, moe_config)
```

关键判断:
- `use_checkpoint_files = bool(checkpoint_files) and not use_kt_weight_path`
- `checkpoint_files` 来自 `kt_plugin.kt_checkpoint_files`
- 这个值在 `load_shard_file` 的 L921-929 中被 stash

所以当 `kt_weight_path=None` 时:
- `use_kt_weight_path = False`
- `checkpoint_files` 已在 stash 阶段设置
- `use_checkpoint_files = True`
- **走路径 B**: `load_experts_from_checkpoint_files()` 从磁盘 safetensors 直接读取

→ **Bug block 加载到 model params 中的 BF16 权重完全没有被读取**

→ 路径 C (`extract_moe_weights()`) 只在 checkpoint_files 不可用时才会被使用，
  但对于所有 safetensors 模型，checkpoint_files 总是可用的。

---

## 修复方案

### 方案: 跳过冗余 BF16 加载

**文件**: `transformers/src/transformers/modeling_utils.py` L874-912

将 bug block 改为无条件跳过 BF16 专家加载。当 `kt_weight_path` 未设置时，
wrapping 循环会通过 `load_experts_from_checkpoint_files()` 从磁盘直接读取。

Expert params 留在 meta device 上，由后处理代码 (L5625-5638) 替换为
轻量级 CPU placeholder (`torch.empty`)，保证 PEFT 可以发现参数形状。

修改后的代码:

```python
if kt_expert_key_mapping:
    kt_config = _get_kt_config()
    kt_weight_path = getattr(kt_config, "kt_weight_path", None) if kt_config is not None else None

    # 不再将 BF16 expert 权重加载到 model params 中。
    # 当 kt_weight_path 未设置时，wrapping 循环会使用
    # load_experts_from_checkpoint_files() 从磁盘 safetensors 直接读取。
    # Expert params 留在 meta device，后处理代码会替换为 CPU placeholder。
    if os.environ.get("ACCELERATE_KT_DEBUG", "0") == "1":
        print(
            f"[KT load_shard_file] SKIPPING {len(kt_expert_key_mapping)} expert keys "
            f"from shard {shard_file} (kt_weight_path={kt_weight_path!r}, "
            f"wrapping loop loads from checkpoint files directly)"
        )

    # stash shard info on KT config for later runtime use
    ...
```

### 对 fallback 路径的影响

- 路径 C (`extract_moe_weights()`) 依赖 model params 中有真实权重
- 修复后 expert params 为 placeholder → 路径 C 将返回空/假数据
- 但路径 C 只在 checkpoint_files 不可用时触发
- 对于所有 safetensors 模型（DeepSeek-V3、Qwen3 等），checkpoint_files 总是可用的
- 若需支持 .bin 格式模型，可在 wrapping 循环中添加额外 fallback

### 对 FSDP2 流程的影响

FSDP2 流程中:
1. `from_pretrained` 中 expert params 为 placeholder (非 BF16)
2. KT wrapping 在 `from_pretrained` 内完成，expert 权重进入 C++ kernel
3. `_clear_original_expert_weights` 将 expert params 替换为 1-byte fake tensor
4. `original_sd = model.state_dict()` 只包含 fake expert tensor (几乎零内存)
5. FSDP2 sharding 和 state_dict 分发正常工作

→ FSDP2 流程不受影响

---

## 修复后用户不需要 BF16 模型的原因

用户提出: "我全程不需要使用 BF16 模型，除了在量化之前"。

这是正确的。修复后的完整权重流转:

```
磁盘 safetensors (BF16)
  │
  ├─ 非 expert 权重 → load_state_dict → model params (BF16 on CPU)
  │                                       → move_non_experts_to_gpu → GPU
  │
  └─ expert 权重 → ❌ 不再加载到 model params
                  → wrapping 循环 per-layer 按需读取:
                    safetensors → BF16 tensor (临时, ~21GB/层)
                      → load_weights_from_tensors() → INT8 in C++ kernel
                      → del BF16 tensor (释放)
                    循环下一层...
```

**BF16 expert 权重只在量化瞬间存在于内存中** (per-layer ~21GB)，
量化完成后立即释放。不存在全量 BF16 expert 权重同时驻留内存的情况。

---

## 验证方法

1. 应用修复到 `transformers/src/transformers/modeling_utils.py`
2. 设置 `ACCELERATE_KT_DEBUG=1` 观察日志确认:
   - 应出现 `SKIPPING ... expert keys from shard` (不再加载 BF16)
   - wrapping 循环应出现 `Loading expert weights from checkpoint files`
3. 监控内存:
   - 671B 模型: 峰值 RAM 应 < 700 GB (而非之前的 > 1800 GB)
   - 235B 模型: 峰值 RAM 应显著降低
4. 训练应正常启动，loss 正常下降
