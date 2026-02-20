# 5. Implementation Details (v3)

## Design Principle

Reduce peak CPU memory during ZeRO-3 model loading by loading checkpoint shards
one at a time instead of merging all shards into a single state_dict.

For MoE models with weight conversions, we separate keys into two categories:
- **Non-expert keys** (attention, layernorm, embeddings, etc.): loaded directly
  per shard — no conversion needed since checkpoint and model key names match
- **Expert keys** (per-expert gate_proj, up_proj, down_proj): need conversion
  (fusion/stacking), handled explicitly to avoid a bug in transformers

## Files Modified

### `src/llamafactory/model/patcher.py`

Three functions:

#### `_find_multi_shard_layers(checkpoint_files) → set[int]`

Reads `model.safetensors.index.json` to find layers whose keys span multiple shards.
Used to determine which expert keys need cross-shard buffering.

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
    # 3. Find boundary layers from safetensors index
    # 4. Create no_wm_config = copy of load_config with weight_mapping=None

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
                if layer is boundary → expert_boundary_buffer[key] = ...
                else → complete_expert_keys[key] = ...

            # Convert complete-layer experts and load
            converted = _apply_weight_conversions_to_state_dict(model, complete_expert_keys, weight_mapping)
            _load_state_dict_into_zero3_model(model, converted, no_wm_config)

        gc.collect()

    # Convert and load buffered boundary-layer experts
    if expert_boundary_buffer:
        converted = _apply_weight_conversions_to_state_dict(model, expert_boundary_buffer, weight_mapping)
        _load_state_dict_into_zero3_model(model, converted, no_wm_config)
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

## Memory Profile (Qwen3-30B-A3B, 4 GPUs)

| Phase | Original | v3 Patched |
|-------|----------|------------|
| Per-shard load | ~4GB | ~4GB |
| Merged state_dict | ~60GB | 0 (not created) |
| Expert boundary buffer | N/A | ~15GB |
| **Peak per process** | **~60GB** | **~19GB** |
| **Peak total (4 proc)** | **~240GB** | **~76GB** |
| Steady-state (training) | ~30GB | ~30GB |
