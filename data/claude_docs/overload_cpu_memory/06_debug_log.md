# 6. Debug Log

## Issue 1: Weight Conversion RuntimeError (Fixed in v2)

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

## Known Potential Issues (Not Yet Tested)

### Issue 2: Weight Conversions for Complete-Layer Keys Within a Shard

**Concern**: When we call `_load_state_dict_into_zero3_model` with `complete_keys`
(non-boundary-layer keys from a single shard), the function internally calls
`_apply_weight_conversions_to_state_dict`. This conversion function processes ALL
keys in the dict. If some non-boundary layers also have weight conversions (e.g.,
MoE layers fully contained in a single shard), the conversion should work. But if
the conversion function's pattern matching picks up keys that don't have all their
sources present (e.g., non-layer keys or keys with unexpected naming), it could
fail.

**Mitigation**: Verify that `_apply_weight_conversions_to_state_dict` gracefully
handles partial state dicts where only some conversions are applicable.

### Issue 3: `_find_multi_shard_layers` Regex Limitation

**Concern**: The regex `r"\.layers\.(\d+)\."` only matches keys following the
standard transformer naming convention. Models with different naming patterns
(e.g., `.blocks.N.` or `.decoder.layers.N.`) would not be detected.

**Impact**: If boundary layers are not detected, the code falls back to original
monolithic loading (safe but no memory improvement).

**Mitigation**: Could be extended to support multiple patterns, but the current
fallback behavior is safe.

### Issue 4: Missing Keys Intersection Logic

**Concern**: The intersection approach for tracking missing keys assumes that a
parameter appears in exactly one shard. If the same key appears in multiple shards
(unlikely but possible with custom checkpoints), the intersection would incorrectly
report it as not missing.

**Mitigation**: Standard HuggingFace safetensors checkpoints never duplicate keys
across shards, so this should not be an issue in practice.

### Issue 5: Boundary Buffer Memory Size

**Concern**: For models with many boundary layers, the boundary buffer could still
be substantial. For Qwen3-30B-A3B with ~15 boundary layers at ~1.25GB each, the
buffer would be ~19GB per process.

**Impact**: Still saves ~41GB per process vs the original ~60GB, but not as
dramatic as the savings for non-MoE models (which would go from ~60GB to ~4GB).

### Issue 6: Non-Layer Keys (Embeddings, LM Head)

**Concern**: Some model keys don't match the `.layers.N.` pattern (e.g.,
`model.embed_tokens.weight`, `lm_head.weight`). In the current implementation,
these are classified as "complete keys" (not boundary-layer keys) and loaded with
the shard they appear in. This should be correct, but needs verification.

### Issue 7: `load_config` Attribute Access

**Concern**: The code accesses `getattr(load_config, "weight_mapping", None)` to
detect weight conversions. This attribute might not exist in all versions of
transformers, or its structure might differ.

**Mitigation**: The `getattr` with default `None` handles the missing attribute
case. If `weight_mapping` is None, no weight converters are detected, and the code
takes the simpler path (no boundary buffering).

---

## Testing Plan

1. **Non-MoE model** (e.g., Qwen2.5-7B): Test shard-by-shard loading without
   weight conversions. Should work with the simple path.

2. **MoE model** (Qwen3-30B-A3B): Test shard-by-shard loading with boundary-layer
   buffering. Verify weight conversions succeed.

3. **Memory measurement**: Compare peak CPU memory (RSS) between original and
   patched loading. Use `psutil` or `/proc/self/status` VmRSS.

4. **Training correctness**: Run a few training steps and verify loss values match
   the original loading path.

5. **Single-shard model**: Test with a model that has only one checkpoint file.
   Should take the shard-by-shard path but with only one iteration.

6. **Quantized model**: Verify that quantized models correctly fall back to the
   original loading path.
