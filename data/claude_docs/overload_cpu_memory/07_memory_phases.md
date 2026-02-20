# 7. CPU Memory Phases Analysis

Based on the v3 patch running Qwen3-235B-A22B-Instruct-2507 with DeepSpeed ZeRO-3
offload, LoRA fine-tuning, 4 GPUs.

## Model Statistics

| Item | Value |
|------|-------|
| Total parameters | 235.09B |
| BF16 model size | 438 GiB (470 GB) |
| Layers | 94 |
| Experts per layer | 128 |
| Checkpoint shards | 118 (~3.7 GiB each) |
| Expert keys | 36,096 (94 layers × 128 experts × 3 projections) |
| Non-expert keys | 849 (94 × 9 per-layer + 3 global) |
| Expert weight size | 423 GiB (96.6%) |
| Non-expert weight size | 14.9 GiB (3.4%) |
| Boundary layers | **94 (ALL layers)** — 73 span 2 shards, 21 span 3 shards |

---

## Phase 0: Model Creation under `deepspeed.zero.Init()`

**Log**: `07:14:35 → 07:16:04` (~89 seconds)

```
Detected DeepSpeed ZeRO-3: activating zero.init() for this model
...
finished initializing model - num_params = 1037, num_elems = 235.09B
```

**What happens**:
- `from_pretrained` creates the model inside a `deepspeed.zero.Init()` context
  manager (`modeling_utils.py:3618`)
- Every `nn.Parameter` creation is intercepted by DeepSpeed
- Each parameter is immediately **partitioned** across 4 ranks: each rank stores
  only 1/4 of the flattened parameter tensor on CPU (with `pin_memory=True`,
  per `offload_param` config)
- Parameter values are uninitialized placeholders — actual weights loaded in Phase 1-2

**Memory per rank**:
- ZeRO-3 partitioned params: 235B × 2 bytes / 4 ranks = **~117.5 GiB**
- These are empty (uninitialized) but memory is allocated

---

## Phase 1: Shard-by-Shard Loading (Our Patch)

**Log**: `07:16:04 → 07:16:55` (~51 seconds, 118 shards)

```
Loading model weights for DeepSpeed ZeRO-3 (118 shards, low CPU memory mode,
  94 boundary layers buffered).
Loading shard 1/118
Loading shard 2/118
...
Loading shard 118/118
```

**What happens for each shard** (code: `patcher.py` shard loop):

```
For shard i:
  ┌─ load_state_dict(ckpt_file)          # ~3.7 GiB into CPU
  │
  ├─ Split by converter_re regex:
  │   ├─ non_expert_keys  (~0.1 GiB)    # attention, layernorm, gate, etc.
  │   └─ expert_keys      (~3.6 GiB)    # mlp.experts.N.{gate,up,down}_proj
  │
  ├─ _load_state_dict_into_zero3_model(model, non_expert_keys, no_wm_config)
  │   │  For each key matching a model parameter:
  │   │    1. GatheredParameters: rank 0 gathers full param from all ranks
  │   │    2. rank 0: _load_from_state_dict() assigns checkpoint tensor
  │   │    3. Parameter re-partitioned: each rank keeps 1/4, rest freed
  │   └─ non_expert_keys freed immediately after loading
  │
  ├─ expert_keys → expert_boundary_buffer  (buffer grows by ~3.6 GiB)
  │   Since ALL 94 layers are boundary layers, ALL expert keys are buffered
  │
  └─ gc.collect()
```

**Memory per rank during Phase 1**:

| Shard # | ZeRO-3 Partitions | Boundary Buffer | Shard Transient | Total |
|---------|-------------------|-----------------|-----------------|-------|
| 1       | ~117.5 GiB        | ~3.6 GiB        | ~3.7 GiB        | ~125 GiB |
| 30      | ~117.5 GiB        | ~108 GiB        | ~3.7 GiB        | ~229 GiB |
| 60      | ~117.5 GiB        | ~216 GiB        | ~3.7 GiB        | ~337 GiB |
| 90      | ~117.5 GiB        | ~324 GiB        | ~3.7 GiB        | ~445 GiB |
| 118     | ~117.5 GiB        | **~423 GiB**    | ~3.7 GiB        | **~544 GiB** |

The boundary buffer linearly accumulates all expert keys across all 118 shards.

---

## Phase 2: Boundary Buffer Conversion and Loading

**Log**: `07:16:56 →` (duration depends on model size)

```
Loading 36096 buffered boundary-layer expert keys
```

**What happens** (code: `_convert_and_load(expert_boundary_buffer)`):

```
1. _apply_weight_conversions_to_state_dict(model, buffer, weight_mapping)
   │
   │  For each of 94 layers × 128 experts:
   │    ├─ gate_proj [1536,4096] + up_proj [1536,4096]
   │    │   → MergeModulelist(dim=0) → Concatenate(dim=1)
   │    │   → gate_up_proj [128, 1536*2, 4096]   (fused for all experts)
   │    │
   │    └─ down_proj [4096,1536]
   │        → MergeModulelist(dim=0)
   │        → down_proj [128, 4096, 1536]   (stacked for all experts)
   │
   │  Source tensors freed as conversion proceeds
   │  Returns new_state_dict with 94×2 = 188 converted tensors
   │
2. _load_state_dict_into_zero3_model(model, converted, no_wm_config)
   │  Same GatheredParameters pattern as Phase 1
   │
3. Free buffer and converted tensors
4. gc.collect()
```

**Memory per rank during Phase 2**:
- Peak: ZeRO-3 partitions (~117.5 GiB) + boundary buffer (~423 GiB)
  + conversion temporaries ≈ **~550-600 GiB**
- After completion: buffer freed → back to ~117.5 GiB

---

## Phase 3: Post-Loading (Adapter, Training Setup)

**What happens**:
- `init_adapter()`: applies LoRA to attention projections (q/k/v/o_proj)
- Trainable params: ~26M (LoRA rank-8 on 94 layers × 4 projections)
- `model.train()` enables gradient computation for LoRA params only

---

## Phase 4: DeepSpeed Optimizer Initialization

**What happens** (`trainer.train()` → `deepspeed.initialize()`):
- Creates optimizer for trainable (LoRA) params only
- FP32 master weights: 26M × 4 bytes = ~0.1 GiB
- Adam momentum (m): ~0.1 GiB
- Adam variance (v): ~0.1 GiB

---

## Steady State (Training)

| Component | Per Rank |
|-----------|----------|
| ZeRO-3 partitioned BF16 params (CPU) | ~117.5 GiB |
| LoRA FP32 master + m + v (CPU) | ~0.3 GiB |
| Gradient buffers, activations (GPU) | varies |
| **Total CPU** | **~118 GiB** |

---

## v3 Issue (Fixed in v4): Patch Ineffective for Qwen3-235B

For Qwen3-235B where **ALL 94 layers are boundary layers**, the v3
`expert_boundary_buffer` accumulated the ENTIRE expert portion of the model:

| Metric | Original (merged state_dict) | v3 Patch |
|--------|------------------------------|----------|
| Peak per-rank during loading | ~438 GiB | **~423 GiB** |
| Savings | baseline | **~15 GiB (3.4%)** |

### Root Cause

The v3 boundary-layer detection was binary: either a layer is fully in one shard
(immediate), or it spans shards (buffer ALL until end). For Qwen3-235B, each layer
has ~384 expert tensors (~4.5 GiB), while each shard is only ~3.7 GiB. EVERY layer
overflows its shard boundary, so ALL expert keys were buffered.

### v4 Fix: Per-Layer Completion Tracking (Implemented)

Instead of buffering ALL boundary-layer expert keys until the end, v4 tracks expert
keys per-layer via `_build_boundary_layer_info()` which pre-computes the expected
expert key count per boundary layer from the safetensors index.

```
For each shard:
  1. Load shard, split expert / non-expert
  2. Non-expert: load immediately (same as v3)
  3. Expert: add to per-layer buffer (dict[layer_id] → dict[key] → tensor)
  4. Check: for each layer in per-layer buffer, if collected == expected
     → convert and load immediately, free from buffer

This limits the buffer to at most 2-3 layers worth of expert keys at any time
(only layers whose keys are currently split across the current shard boundary).
```

**v4 memory profile for Qwen3-235B**:

| Metric | Original | v3 Patch | v4 Patch |
|--------|----------|----------|----------|
| Buffer peak | ~438 GiB | ~423 GiB | **~9-14 GiB** |
| Peak per-rank | ~560 GiB | ~544 GiB | **~135 GiB** |
| Savings/rank | baseline | ~16 GiB (3%) | **~425 GiB (76%)** |

### Comparison across models

| Metric | Qwen3-30B-A3B | Qwen3-235B-A22B |
|--------|---------------|-----------------|
| Layers | 48 | 94 |
| Shards | 16 | 118 |
| Boundary layers | 14 (29%) | **94 (100%)** |
| Expert keys/layer | 384 | 384 |
| v4 buffer (max) | ~2-3 layers (~4-9 GiB) | ~2-3 layers (~9-14 GiB) |
| v4 savings/rank | ~39 GiB (65%) | **~425 GiB (76%)** |
