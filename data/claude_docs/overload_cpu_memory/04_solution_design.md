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

### Two-Phase Loading Strategy

**Phase 1**: For each shard, classify keys:
- **Complete-layer keys**: Keys belonging to layers that are fully contained in a
  single shard → load immediately
- **Boundary-layer keys**: Keys belonging to layers that span multiple shards →
  buffer in memory

**Phase 2**: After all shards are processed, the boundary buffer contains all keys
for cross-shard layers → load them in one batch.

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

### Memory Trade-off

The boundary buffer still holds data in memory, but only for boundary layers:

| Scenario | Peak Memory Per Process |
|----------|----------------------|
| Original (all shards merged) | ~60GB |
| Shard-by-shard (no conversions) | ~4GB (one shard) |
| Shard-by-shard + boundary buffer | ~4GB + boundary data |

For Qwen3-30B-A3B with ~15 boundary layers:
- Each MoE layer ≈ 1.25GB → boundary buffer ≈ 19GB
- Still saves ~41GB per process vs original (~164GB total across 4 processes)

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

4. **Stream conversion keys on-demand**: Only buffer the specific keys needed for
   weight conversion, not entire boundary layers. More complex but more memory
   efficient. Could be a future optimization.
