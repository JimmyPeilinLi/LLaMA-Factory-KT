# 3. Root Cause Analysis

## User's Initial Hypothesis (Disproved)

> DeepSpeed treats the 235B frozen base model parameters as trainable,
> creating Adam FP32 master weights + momentum/variance (m/v) states on CPU
> for ALL parameters, not just LoRA parameters.

### Why This Hypothesis Is Wrong

There are **three independent levels** of `requires_grad` filtering that ensure
only LoRA adapter parameters receive optimizer states:

#### Level 1: HuggingFace Trainer (`transformers/integrations/deepspeed.py:611`)
```python
model_parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
```
Only trainable parameters are passed to `deepspeed.initialize()`.

#### Level 2: DeepSpeed Engine (`deepspeed/runtime/engine.py:335`)
```python
# Inside DeepSpeedEngine.__init__
# The optimizer receives only the filtered parameter groups from Level 1
```

#### Level 3: ZeRO-3 Optimizer (`deepspeed/runtime/zero/stage3.py:491-507`)
```python
def _get_trainable_parameter_groups(self):
    for param_group in self.optimizer.param_groups:
        trainable_params = [p for p in param_group["params"] if p.requires_grad]
```
Even within the optimizer, ZeRO-3 re-filters for `requires_grad=True`.

#### PEFT `requires_grad` Setup (`peft/tuners/tuners_utils.py:423`)
```python
def _mark_only_adapters_as_trainable(model):
    # Base model params → requires_grad = False
    # LoRA A/B params  → requires_grad = True
```

**Conclusion**: The optimizer states (FP32 master weights, momentum, variance) are
only created for the small set of LoRA parameters (~26M params for rank-8 LoRA on
Qwen3-30B-A3B), NOT for the 30B base model parameters.

---

## Actual Root Cause: Monolithic State Dict Loading

The real cause of the ~300GB CPU memory spike is in
`transformers/modeling_utils.py:4200-4210`:

```python
# Inside PreTrainedModel._load_pretrained_model()
if is_deepspeed_zero3_enabled() and not is_quantized:
    if state_dict is None:
        merged_state_dict = {}
        for ckpt_file in checkpoint_files:
            merged_state_dict.update(
                load_state_dict(ckpt_file, map_location="cpu",
                                weights_only=load_config.weights_only)
            )
        state_dict = merged_state_dict
    error_msgs, missing_keys = _load_state_dict_into_zero3_model(
        model, state_dict, load_config
    )
```

### What Happens Step by Step

1. **Each process** (4 GPUs = 4 processes) enters `_load_pretrained_model`
2. Each process iterates over ALL 16 checkpoint shards
3. Each shard is loaded into CPU memory and merged into `merged_state_dict`
4. After loading all shards, `merged_state_dict` holds the ENTIRE model (~60GB in BF16)
5. `_load_state_dict_into_zero3_model` then distributes the weights across processes
6. `merged_state_dict` is freed, memory drops

### Memory Math

| Component | Per Process | Total (4 processes) |
|-----------|-------------|---------------------|
| Model BF16 weights | ~60GB | ~240GB |
| Intermediate loading overhead | ~15GB | ~60GB |
| **Total peak** | **~75GB** | **~300GB** |

After `merged_state_dict` is freed and weights are partitioned across processes:

| Component | Per Process | Total (4 processes) |
|-----------|-------------|---------------------|
| Partitioned BF16 params | ~15GB | ~60GB |
| LoRA optimizer states | ~0.3GB | ~1.2GB |
| Other overhead | ~15GB | ~60GB |
| **Total steady-state** | **~30GB** | **~120GB** |

### Why This Code Exists

The ZeRO-3 loading path requires a complete state dict because:

1. **`deepspeed.zero.Init` context**: During `from_pretrained`, the model is created
   inside a `deepspeed.zero.Init()` context manager. This means all parameters are
   immediately partitioned across processes as "empty" placeholders.

2. **`_load_state_dict_into_zero3_model`**: This function uses
   `deepspeed.zero.GatheredParameters` to temporarily gather each parameter, assign
   the checkpoint weight, then re-partition it.

3. **Weight Conversions**: For MoE models like Qwen3-30B-A3B, checkpoint keys need
   to be fused (e.g., per-expert `gate_proj` + `up_proj` → `gate_up_proj`). The
   conversion function `_apply_weight_conversions_to_state_dict` requires all source
   tensors to be present simultaneously.

### Complicating Factor: MoE Weight Conversions

The simplest fix would be to load one shard at a time. However, Qwen3-30B-A3B has
weight conversions that span shard boundaries:

```
Shard 2: experts.0.gate_proj (layer 5), experts.1-59.gate_proj (layer 5)
Shard 3: experts.60-117.gate_proj (layer 5), experts.0-117.up_proj (layer 5)
```

The conversion needs ALL 118 experts' `gate_proj` and `up_proj` for layer 5 to
produce `gate_up_proj`. If we load shard 2 alone, we only have experts 0-59's
`gate_proj`, and the conversion fails with:

```
RuntimeError: Failed to apply weight conversion for
'model.layers.5.mlp.experts.gate_up_proj'.
Sizes of tensors must match except in dimension 1.
Expected size 118 but got size 117
```

This is why a naive shard-by-shard approach doesn't work for MoE models.
