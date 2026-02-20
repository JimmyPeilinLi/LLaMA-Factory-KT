# 2. Architecture Analysis

## Key Code Paths

### 2.1 Model Loading Flow

```
loader.py: load_model()
  ├── patch_config()         # patcher.py:126 — sets low_cpu_mem_usage=False for ZeRO-3
  ├── from_pretrained()      # HuggingFace transformers
  │   ├── get_init_context() # modeling_utils.py:3610 — wraps in deepspeed.zero.Init()
  │   ├── cls(config)        # Creates model with partitioned params (ZeRO-3 Init context)
  │   └── _load_pretrained_model()  # modeling_utils.py:4160
  │       └── ZeRO-3 branch (line 4200-4210):
  │           ├── merged_state_dict = {}
  │           ├── for ckpt_file in checkpoint_files:
  │           │     merged_state_dict.update(load_state_dict(ckpt_file))  ← FULL MERGE
  │           ├── _load_state_dict_into_zero3_model(model, merged_state_dict)
  │           └── (merged_state_dict freed after loading)
  ├── init_adapter()         # adapter.py:321 — applies LoRA, sets requires_grad
  └── model.train()
```

### 2.2 DeepSpeed Initialization Flow

```
Trainer.train()
  └── deepspeed_init()  # transformers/integrations/deepspeed.py:555
      ├── model_parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
      │   ↑ CRITICAL: only trainable (LoRA) params passed to optimizer
      └── deepspeed.initialize(model, optimizer, model_parameters)
          └── DeepSpeedZeroOptimizer_Stage3.__init__()
              ├── _get_trainable_parameter_groups()  # stage3.py:491
              │   └── trainable_params = [p for p in params if p.requires_grad]
              ├── _create_fp16_partitions_with_defragmentation()  # Only trainable
              ├── _create_fp32_partitions()  # stage3.py:832 — FP32 master only for trainable
              └── initialize_optimizer_states()  # Only for trainable params
```

### 2.3 LoRA requires_grad Setup

```
adapter.py: _setup_lora_tuning()
  └── get_peft_model(model, peft_config)  # PEFT library
      └── _mark_only_adapters_as_trainable()  # peft/tuners/tuners_utils.py:423
          ├── Base model params → requires_grad = False
          └── LoRA A/B params → requires_grad = True
```

### 2.4 Weight Conversion (MoE Models)

For Qwen3-30B-A3B, the checkpoint stores per-expert weights separately.
The model expects them fused:

```
Checkpoint keys:                    Model key:
  experts.0.gate_proj.weight  ──┐
  experts.0.up_proj.weight    ──┤── torch.cat ──→  experts.gate_up_proj
  experts.1.gate_proj.weight  ──┤
  experts.1.up_proj.weight    ──┘
```

This conversion is handled by `_apply_weight_conversions_to_state_dict()` in
`transformers/integrations/deepspeed.py:300-415`, which requires ALL source
tensors to be present simultaneously.

## Key Files

| File | Role |
|------|------|
| `src/llamafactory/model/loader.py` | Model loading entry point |
| `src/llamafactory/model/patcher.py` | Model/config patching |
| `src/llamafactory/model/adapter.py` | LoRA/adapter setup, requires_grad |
| `examples/deepspeed/ds_z3_offload_config.json` | DeepSpeed ZeRO-3 config |
| `transformers/modeling_utils.py:4200-4210` | ZeRO-3 state_dict loading |
| `transformers/integrations/deepspeed.py:418-496` | `_load_state_dict_into_zero3_model` |
| `transformers/integrations/deepspeed.py:300-415` | `_apply_weight_conversions_to_state_dict` |
| `deepspeed/runtime/zero/stage3.py:491-507` | `_get_trainable_parameter_groups` |
| `deepspeed/runtime/zero/stage3.py:832-923` | `_create_fp32_partitions` |

## DeepSpeed ZeRO-3 Offload Config

```json
{
  "zero_optimization": {
    "stage": 3,
    "offload_optimizer": { "device": "cpu", "pin_memory": true },
    "offload_param": { "device": "cpu", "pin_memory": true },
    "overlap_comm": false,
    "contiguous_gradients": true,
    "sub_group_size": 1e9,
    "stage3_gather_16bit_weights_on_model_save": true
  }
}
```
