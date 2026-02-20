# 6. Debug Log

## Issue 1: Weight Conversion RuntimeError (v1 → Fixed in v2)

### Error

After the first implementation (naive shard-by-shard, no boundary buffering):

```
RuntimeError: Failed to apply weight conversion for
'model.layers.5.mlp.experts.gate_up_proj'.
Sizes of tensors must match except in dimension 1.
Expected size 118 but got size 117
```

### Root Cause

Qwen3-30B-A3B has 128 experts per MoE layer. Weight conversions fuse per-expert
`gate_proj` and `up_proj` into `gate_up_proj` via `torch.cat`. For layer 5,
experts are split across shard 2 and shard 3:

- Shard 2: experts 0-59 `gate_proj`
- Shard 3: experts 60-117 `gate_proj` + all `up_proj`

When loading shard 2 alone, `_apply_weight_conversions_to_state_dict` attempts to
convert with only 60 experts' gate_proj instead of 118, producing a size mismatch.

### Fix

Implemented boundary-layer buffering (v2):
1. Read safetensors index to identify layers spanning multiple shards
2. Buffer boundary-layer keys across all shards
3. Load complete-layer keys per-shard immediately
4. After all shards: load buffered boundary keys with all source tensors present

---

## Issue 2: ALL Non-Expert Weights MISSING (v2 → Fixed in v3)

### Error

After the v2 implementation (boundary-layer buffering):

```
Qwen3MoeForCausalLM LOAD REPORT from: /mnt/data/models/Qwen3-30B-A3B
model.layers.{0...47}.self_attn.k_proj.weight         | MISSING |
model.layers.{0...47}.self_attn.q_proj.weight         | MISSING |
model.layers.{0...47}.self_attn.v_proj.weight         | MISSING |
model.layers.{0...47}.self_attn.o_proj.weight         | MISSING |
model.layers.{0...47}.self_attn.q_norm.weight         | MISSING |
model.layers.{0...47}.self_attn.k_norm.weight         | MISSING |
model.layers.{0...47}.input_layernorm.weight          | MISSING |
model.layers.{0...47}.post_attention_layernorm.weight | MISSING |
model.layers.{0...47}.mlp.gate.weight                 | MISSING |
model.embed_tokens.weight                             | MISSING |
model.norm.weight                                     | MISSING |
lm_head.weight                                        | MISSING |
```

All non-expert weights (attention, layernorm, gate, embeddings, lm_head) are MISSING.
Only MoE expert weights (`mlp.experts.*`) were loaded successfully.

### Root Cause

Bug in `_apply_weight_conversions_to_state_dict` (`deepspeed.py:293-415`) when
both `WeightConverter` and `WeightRenaming` entries coexist in `conversion_mapping`.

**Class hierarchy** (these are siblings, NOT parent-child):
```
WeightTransform (base)
├── WeightRenaming  — simple key rename, has convert() method
└── WeightConverter  — tensor fusion, has convert() method
```

**The bug** (deepspeed.py line 379):
```python
for renamed_key, mapping in conversion_mapping.items():
    if not isinstance(mapping, WeightConverter):
        continue  # ← SKIPS WeightRenaming!
```

**What happens**:
1. Line 353: ALL keys are `state_dict.pop()`-ed from the dict
2. Expert keys → added to `conversion_mapping` as `WeightConverter` entries
3. Non-expert keys → added to `conversion_mapping` as auto-created `WeightRenaming`
4. Line 379: converter loop processes only `WeightConverter`, **skips `WeightRenaming`**
5. Line 401-409: tries to process remaining keys in `state_dict` — but it's **empty**
6. Result: `new_state_dict` only has converted expert keys, non-expert keys are LOST

**Why the original (unpatched) loading works**:
The original loading without our patch likely ran on a transformers version where
`_apply_weight_conversions_to_state_dict` was not yet called inside
`_load_state_dict_into_zero3_model`, or `weight_mapping` was None. The weight
conversion framework is a relatively new addition to transformers.

### Fix (v3)

Completely restructured the approach to avoid the buggy function:

1. **Non-expert keys**: Load directly via `_load_state_dict_into_zero3_model` with
   `weight_mapping=None` (checkpoint key names match model key names for attention,
   layernorm, etc., so no conversion is needed)

2. **Expert keys**: Call `_apply_weight_conversions_to_state_dict` ourselves with
   ONLY expert keys. Since all keys match `WeightConverter` patterns, no
   `WeightRenaming` entries are auto-created, and the bug is not triggered.
   Then load the converted result with `weight_mapping=None`.

3. **Boundary buffering**: Now only buffers expert keys (which need cross-shard
   conversion). Non-expert keys from boundary layers are loaded directly per-shard,
   further reducing the boundary buffer size.

**Memory improvement**: The expert-only boundary buffer is significantly smaller
than the v2 whole-layer boundary buffer. For Qwen3-30B-A3B with 14 boundary layers,
the buffer now holds only expert weights (~1.1GB/layer × 14 ≈ 15GB) instead of
all layer weights (~1.25GB/layer × 14 ≈ 17.5GB). The non-expert weights (~9 keys
per layer, ~few MB) are loaded and freed immediately per shard.

---

## Issue 3: FrozenInstanceError (v3 → Fixed in v3)

### Error

```
dataclasses.FrozenInstanceError: cannot assign to field 'weight_mapping'
```

When trying to create a copy of `LoadStateDictConfig` with `weight_mapping=None`:
```python
no_wm_config = copy.copy(load_config)
no_wm_config.weight_mapping = None  # ← FrozenInstanceError!
```

### Root Cause

`LoadStateDictConfig` is a frozen dataclass (`@dataclass(frozen=True)`). Normal
attribute assignment is blocked.

### Fix

Use `object.__setattr__` to bypass the frozen check:
```python
object.__setattr__(no_wm_config, "weight_mapping", None)
```

---

## Issue 4: v3 Ineffective for Large Models (v3 → Fixed in v4)

### Problem

For Qwen3-235B-A22B-Instruct-2507 (94 layers, 118 shards, ALL 94 layers are
boundary layers), the v3 `expert_boundary_buffer` accumulates the ENTIRE expert
portion of the model (~423 GiB), providing only ~3.4% savings.

| Metric | Qwen3-30B-A3B | Qwen3-235B-A22B |
|--------|---------------|-----------------|
| Boundary layers | 14/48 (29%) | **94/94 (100%)** |
| v3 buffer size | ~15 GiB | **~423 GiB** |
| v3 savings/rank | ~39 GiB (65%) | **~15 GiB (3.4%)** |

### Root Cause

The v3 boundary detection is binary: either a layer is fully in one shard (load
immediately) or it spans shards (buffer ALL until end). For Qwen3-235B, each layer
has ~384 expert tensors (~4.5 GiB) while each shard is only ~3.7 GiB. EVERY layer
overflows its shard boundary, so the entire expert portion is buffered.

### Fix (v4): Per-Layer Completion Tracking

Replaced `_find_multi_shard_layers()` (returns `set[int]`) with
`_build_boundary_layer_info()` (returns `dict[int, int]`: layer_id → expected
expert key count).

Changed flat `expert_boundary_buffer: dict` to per-layer
`per_layer_buffer: dict[int, dict]`. After processing each shard, check each
buffered layer: if collected count >= expected count, immediately convert and load,
then free from buffer.

**Result**: Buffer limited to 2-3 layers at any time (~9-14 GiB), regardless of
how many boundary layers exist.

| Metric | v3 (235B) | v4 (235B) |
|--------|-----------|-----------|
| Buffer peak | ~423 GiB | **~9-14 GiB** |
| Peak per rank | ~544 GiB | **~135 GiB** |
| Savings/rank | ~16 GiB (3%) | **~425 GiB (76%)** |

---

## Other Issues in the Test Run (Not Related to Our Patch)

### CUDA Version Mismatch

```
CUDAMismatchException: Installed CUDA version 13.0 does not match the version
torch was compiled with 12.8
```

Environment issue. DeepSpeed's CPUAdamBuilder JIT compile detects CUDA mismatch.
Fix: `DS_BUILD_CPU_ADAM=1 pip install deepspeed` or `DS_SKIP_CUDA_CHECK=1`.

### DeepSpeedCPUAdam Destructor Error

```
AttributeError: 'DeepSpeedCPUAdam' object has no attribute 'ds_opt_adam'
```

Cascade from CUDA mismatch — constructor failed, destructor can't clean up.

### PYTORCH_CUDA_ALLOC_CONF Deprecation Warning

Rename environment variable from `PYTORCH_CUDA_ALLOC_CONF` to `PYTORCH_ALLOC_CONF`.

---

## Testing Plan

1. **Non-MoE model** (e.g., Qwen2.5-7B): Test shard-by-shard loading without
   weight conversions. Should work with the simple path.

2. **MoE model** (Qwen3-30B-A3B): Test shard-by-shard loading with expert key
   separation and per-layer tracking. Verify weight conversions succeed and
   NO MISSING keys in the LOAD REPORT.

3. **Large MoE model** (Qwen3-235B-A22B): Test per-layer completion tracking with
   all-boundary-layer scenario. Verify buffer stays small (2-3 layers).

4. **Memory measurement**: Compare peak CPU memory (RSS) between original and
   patched loading. Use `psutil` or `/proc/self/status` VmRSS.

5. **Training correctness**: Run a few training steps and verify loss values match
   the original loading path.

6. **Single-shard model**: Test with a model that has only one checkpoint file.
   Should take the shard-by-shard path but with only one iteration.

7. **Quantized model**: Verify that quantized models correctly fall back to the
   original loading path.
