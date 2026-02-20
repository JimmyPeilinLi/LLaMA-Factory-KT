# 5. Implementation Details

## Files Modified

### `src/llamafactory/model/patcher.py`

Two new functions added:

#### `_find_multi_shard_layers(checkpoint_files)`

Reads `model.safetensors.index.json` to identify which model layers have keys
spread across multiple checkpoint shards.

```python
def _find_multi_shard_layers(checkpoint_files: list[str]) -> set[int]:
    """
    Returns set of layer indices whose keys span multiple shards.

    Algorithm:
    1. Find model.safetensors.index.json in same directory as checkpoint files
    2. Parse weight_map: {param_key: shard_filename}
    3. For each key matching '.layers.N.', track which shard(s) layer N appears in
    4. Return layers that appear in more than one shard
    """
```

**Note**: Uses regex `r"\.layers\.(\d+)\."` to extract layer indices. This works
for all transformer models following the HuggingFace naming convention
(`model.layers.N.xxx`). Models with different naming would not be detected, but
would fall back to the original loading path.

#### `patch_zero3_model_loading()`

Monkey-patches `PreTrainedModel._load_pretrained_model` with a shard-by-shard
implementation.

**Key logic**:

```python
def _load_pretrained_model_shard_by_shard(cls, model, state_dict, checkpoint_files, load_config):
    # 1. Guard: only activate for ZeRO-3 + non-quantized + no pre-loaded state_dict
    if not (is_deepspeed_zero3_enabled() and not is_quantized and state_dict is None and checkpoint_files):
        return _original_load(...)

    # 2. Detect weight converters (MoE models)
    weight_mapping = getattr(load_config, "weight_mapping", None)
    has_weight_converters = any(isinstance(v, WeightConverter) for v in weight_mapping)

    # 3. Find boundary layers (if weight converters exist)
    multi_shard_layers = _find_multi_shard_layers(checkpoint_files) if has_weight_converters else set()

    # 4. For each shard:
    for ckpt_file in checkpoint_files:
        shard_state_dict = load_state_dict(ckpt_file)

        if multi_shard_layers:
            # Separate keys: boundary-layer keys → buffer, others → load immediately
            for key in shard_state_dict:
                if key matches boundary layer:
                    boundary_buffer[key] = shard_state_dict[key]
                else:
                    complete_keys[key] = shard_state_dict[key]

            _load_state_dict_into_zero3_model(model, complete_keys, load_config)
        else:
            # No conversions: load entire shard directly
            _load_state_dict_into_zero3_model(model, shard_state_dict, load_config)

        gc.collect()

    # 5. Load buffered boundary-layer keys (now complete)
    if boundary_buffer:
        _load_state_dict_into_zero3_model(model, boundary_buffer, load_config)

    return LoadStateDictInfo(...)
```

### `src/llamafactory/model/loader.py`

- Added import: `patch_zero3_model_loading` from `patcher.py`
- Added call before `from_pretrained`:
  ```python
  # Patch model loading to be shard-by-shard for ZeRO-3 (reduces peak CPU memory)
  patch_zero3_model_loading()
  ```
- The patch is guarded by `_zero3_loading_patched` flag, so it only runs once even
  if `load_model` is called multiple times.

## How `_load_state_dict_into_zero3_model` Handles Partial State Dicts

The function at `transformers/integrations/deepspeed.py:418-496` already supports
partial state dicts by design:

```python
# Line 470-471 comment:
# "In sharded models, each shard has only part of the full state_dict"
```

It iterates over the model's named parameters, and for each parameter:
1. Checks if the corresponding key exists in the state dict
2. If yes: gathers the parameter, assigns the weight, re-partitions
3. If no: adds to `missing_keys` set

This means we can call it multiple times with different subsets of keys, and the
union of all calls will cover all parameters.

## Missing Keys Handling

Since we call `_load_state_dict_into_zero3_model` multiple times, each call reports
"missing keys" for parameters not in that particular partial dict. We track the
intersection of missing keys across all calls — a key is truly missing only if it
was missing in ALL calls.

```python
if all_missing_keys is None:
    all_missing_keys = shard_missing
else:
    all_missing_keys = all_missing_keys.intersection(shard_missing)
```

## Current Status

**Not yet fully tested.** The first implementation (naive shard-by-shard without
boundary buffering) failed with weight conversion errors on Qwen3-30B-A3B. The
second implementation (with boundary-layer buffering) was written but has known
potential issues — see [06_debug_log.md](06_debug_log.md).
