# 4. Solution Design

## Approach: Shard-by-Shard Loading with Boundary-Layer Buffering

### Core Idea

Instead of merging all checkpoint shards into one monolithic `merged_state_dict`,
load each shard independently and immediately feed it to
`_load_state_dict_into_zero3_model`. This reduces peak CPU memory from ~60GB per
process (full model) to ~4GB per process (one shard).

### Challenge: Cross-Shard Weight Conversions

For MoE models like Qwen3-30B-A3B, some model layers have their checkpoint keys
spread across multiple shards. Weight conversions (e.g., `gate_proj` + `up_proj` →
`gate_up_proj`) require ALL source tensors to be present simultaneously.

### v3 Strategy: Expert/Non-Expert Separation + Boundary Buffering

**Non-expert keys** (attention, layernorm, embeddings, etc.) are loaded directly per
shard with `weight_mapping=None` — their checkpoint key names match model key names,
so no conversion is needed.

**Expert keys** (per-expert gate_proj, up_proj, down_proj) are separated:
- **Complete-layer expert keys**: layer fully in one shard → convert and load immediately
- **Boundary-layer expert keys**: layer spans multiple shards → buffer until all shards processed

**Limitation**: For large models (Qwen3-235B) where ALL layers are boundary layers,
the buffer accumulates the entire expert portion (~423 GiB), defeating the purpose.

### v4 Strategy: Per-Layer Completion Tracking

Instead of buffering ALL boundary-layer expert keys until the end, track expert keys
per-layer and convert+load each layer as soon as all its expected keys arrive:

```
_build_boundary_layer_info():
  Read safetensors index → for each boundary layer, count expected expert keys
  Returns: {layer_id: expected_expert_key_count}

For each shard:
  1. Load shard, split expert / non-expert using converter_re
  2. Non-expert: load immediately (same as v3)
  3. Expert keys from non-boundary layers: convert and load immediately
  4. Expert keys from boundary layers: add to per_layer_buffer[layer_id]
  5. After adding: check each buffered layer —
     if collected_count >= expected_count:
       → convert and load immediately, free from buffer

Buffer holds at most 2-3 layers at any time (layers currently straddling a shard
boundary), regardless of total boundary layer count.
```

### How to Identify Boundary Layers

Read `model.safetensors.index.json` which maps every parameter key to its shard file:

```json
{
  "weight_map": {
    "model.layers.5.mlp.experts.0.gate_proj.weight": "model-00002-of-00016.safetensors",
    "model.layers.5.mlp.experts.60.gate_proj.weight": "model-00003-of-00016.safetensors",
    ...
  }
}
```

A layer is a "boundary layer" if its keys appear in more than one shard file.
In v4, we also count how many expert keys (matching the converter regex) each
boundary layer has, enabling completion detection.

### Memory Trade-off

| Scenario | Peak Memory Per Process |
|----------|----------------------|
| Original (all shards merged) | ~60GB (30B) / ~438 GiB (235B) |
| v3 shard-by-shard + boundary buffer | ~19GB (30B) / ~423 GiB (235B) |
| v4 shard-by-shard + per-layer tracking | ~13GB (30B) / ~14 GiB (235B) |

The v4 buffer size is bounded by: `max_concurrent_boundary_layers × expert_size_per_layer`.
Typically 2-3 layers at any time, regardless of model size.

### Implementation Location

**Monkey-patch** `PreTrainedModel._load_pretrained_model` in
`src/llamafactory/model/patcher.py`. This avoids modifying the transformers library
directly and keeps the fix contained within LlamaFactory.

### Fallback Behavior

The patch only activates when ALL conditions are met:
1. DeepSpeed ZeRO-3 is enabled
2. Model is not quantized
3. No pre-loaded `state_dict` is provided
4. Checkpoint files are available

Otherwise, it falls back to the original `_load_pretrained_model`.

For weight conversions, an additional fallback: if the safetensors index is not
found, fall back to the original monolithic loading.

### Alternative Approaches Considered

1. **Upstream transformers fix**: Would require modifying `modeling_utils.py` in
   the transformers library. More robust but:
   - Requires PR acceptance and release cycle
   - Need to handle many more edge cases
   - Can be done later as a proper fix

2. **Pre-partition checkpoint files by layer**: Reorganize safetensors files so no
   layer spans multiple shards. Complex and model-specific.

3. **Lazy loading with safetensors**: Use safetensors' memory-mapping to avoid
   loading entire shards. Possible but requires deeper changes to the loading path.

4. ~~**Stream conversion keys on-demand**: Only buffer the specific keys needed for
   weight conversion, not entire boundary layers.~~ → This is essentially what v4
   does with per-layer tracking.
