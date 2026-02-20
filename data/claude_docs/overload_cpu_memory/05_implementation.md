# 5. Implementation Details (v4)

## Design Principle

Reduce peak CPU memory during ZeRO-3 model loading by loading checkpoint shards
one at a time instead of merging all shards into a single state_dict.

For MoE models with weight conversions, we separate keys into two categories:
- **Non-expert keys** (attention, layernorm, embeddings, etc.): loaded directly
  per shard — no conversion needed since checkpoint and model key names match
- **Expert keys** (per-expert gate_proj, up_proj, down_proj): need conversion
  (fusion/stacking), handled explicitly to avoid a bug in transformers

## v3 → v4 Improvement: Per-Layer Completion Tracking

**Problem with v3**: For large models like Qwen3-235B-A22B where ALL 94 layers are
boundary layers (keys span multiple shards), the v3 `expert_boundary_buffer`
accumulated the ENTIRE expert portion of the model (~423 GiB), providing only ~3.4%
savings vs the original monolithic loading.

**v4 solution**: Instead of buffering all boundary-layer expert keys until the end,
track expert keys per-layer and convert+load each layer as soon as all its expected
expert keys arrive. This limits the buffer to at most 2-3 layers at any time.

## Files Modified

### `src/llamafactory/model/patcher.py`

Three functions:

#### `_build_boundary_layer_info(checkpoint_files, converter_re) → dict[int, int]`

Reads `model.safetensors.index.json` to find layers whose keys span multiple shards
AND count the expected number of expert keys (matching `converter_re`) per boundary
layer. Returns `{layer_id: expected_expert_key_count}`.

This enables per-layer completion tracking: when the number of collected expert keys
for a layer equals the expected count, that layer is immediately converted and loaded.

#### `_build_converter_pattern(weight_mapping) → re.Pattern | None`

Builds a compiled regex from `WeightConverter.source_patterns` to classify
checkpoint keys as "expert" (needing conversion) vs "non-expert" (direct load).

Example for qwen3_moe:
```
mlp\.experts\.[^.]+\.gate_proj\.weight|mlp\.experts\.[^.]+\.up_proj\.weight|mlp\.experts\.[^.]+\.down_proj\.weight
```

#### `patch_zero3_model_loading()`

Monkey-patches `PreTrainedModel._load_pretrained_model`. Key logic:

```python
def _load_pretrained_model_shard_by_shard(cls, model, state_dict, checkpoint_files, load_config):
    # 1. Guard: only activate for ZeRO-3 + non-quantized + no pre-loaded state_dict
    # 2. Detect weight converters, build converter regex
    # 3. Build boundary layer info: {layer_id: expected_expert_key_count}
    # 4. Create no_wm_config = copy of load_config with weight_mapping=None

    per_layer_buffer = {}  # layer_id → {key: tensor}

    for ckpt_file in checkpoint_files:
        shard = load_state_dict(ckpt_file)

        if not has_weight_converters:
            # Simple path: load entire shard directly
            _load_state_dict_into_zero3_model(model, shard, no_wm_config)
        else:
            # Split shard into expert vs non-expert keys
            expert_keys = {k: v matching converter_re}
            non_expert_keys = {everything else}

            # Non-expert: load immediately (no conversion needed)
            _load_state_dict_into_zero3_model(model, non_expert_keys, no_wm_config)

            # Expert: further split into boundary vs complete layers
            for key in expert_keys:
                if layer is boundary → per_layer_buffer[layer_id][key] = ...
                else → complete_expert_keys[key] = ...

            # Convert complete-layer experts and load
            converted = _apply_weight_conversions_to_state_dict(model, complete_expert_keys, weight_mapping)
            _load_state_dict_into_zero3_model(model, converted, no_wm_config)

            # *** v4 KEY CHANGE: check for completed boundary layers ***
            for layer_id in per_layer_buffer:
                if len(per_layer_buffer[layer_id]) >= boundary_layer_info[layer_id]:
                    # All expert keys for this layer have arrived — convert and load NOW
                    converted = _apply_weight_conversions_to_state_dict(...)
                    _load_state_dict_into_zero3_model(model, converted, no_wm_config)
                    del per_layer_buffer[layer_id]  # free memory immediately

        gc.collect()

    # Fallback: handle any remaining incomplete layers (shouldn't happen)
    if per_layer_buffer:
        logger.warning(...)
        # convert and load remaining
```

### `src/llamafactory/model/loader.py`

- Import `patch_zero3_model_loading` from `patcher.py`
- Call it before `from_pretrained` at line ~171

## Why `weight_mapping=None`?

`_load_state_dict_into_zero3_model` internally calls `_apply_weight_conversions_to_state_dict`
when `load_config.weight_mapping` is non-empty. This function has a bug where
`WeightRenaming` entries are dropped (see 06_debug_log.md Issue 2). By setting
`weight_mapping=None`, we prevent that buggy call and handle conversions ourselves.

## Why Calling `_apply_weight_conversions_to_state_dict` With Only Expert Keys Works

When the state_dict contains ONLY keys matching `WeightConverter` patterns (expert
keys), the function creates ONLY `WeightConverter` entries in `conversion_mapping`.
No `WeightRenaming` entries are auto-created, so the `isinstance` check bug is
never triggered. All entries are processed correctly.

## Why Per-Layer Conversion Works

`_apply_weight_conversions_to_state_dict` handles partial state dicts correctly.
It checks each key against `model_state_dict` (built from the full model), and only
processes keys that exist in the input. So passing a single layer's 384 expert keys
(128 experts × 3 projections) produces the correct 2 converted keys (`gate_up_proj`
and `down_proj`) for that layer only.

## Memory Profile

### Qwen3-30B-A3B (48 layers, 14 boundary, 16 shards, 4 GPUs)

| Phase | Original | v3/v4 Patched |
|-------|----------|---------------|
| Per-shard load | ~4GB | ~4GB |
| Merged state_dict | ~60GB | 0 (not created) |
| Expert boundary buffer | N/A | ~4-9GB (2-3 layers) |
| **Peak per process** | **~60GB** | **~13GB** |
| **Peak total (4 proc)** | **~240GB** | **~52GB** |
| Steady-state (training) | ~30GB | ~30GB |

### Qwen3-235B-A22B (94 layers, ALL boundary, 118 shards, 4 GPUs)

| Phase | Original | v3 Patched | v4 Patched |
|-------|----------|------------|------------|
| ZeRO-3 partitions | ~117.5 GiB | ~117.5 GiB | ~117.5 GiB |
| Merged/buffer peak | ~438 GiB | ~423 GiB | ~9-14 GiB |
| **Peak per rank** | **~560 GiB** | **~544 GiB** | **~135 GiB** |
| **Savings per rank** | baseline | ~16 GiB (3%) | **~425 GiB (76%)** |
| Steady-state | ~118 GiB | ~118 GiB | ~118 GiB |
