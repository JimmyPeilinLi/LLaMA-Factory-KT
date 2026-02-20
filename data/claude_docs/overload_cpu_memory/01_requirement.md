# 1. Problem Statement / Requirement

## User Observation

Running command:
```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 FORCE_TORCHRUN=1 llamafactory-cli train examples/train_lora/qwen3_lora_sft_ds3.yaml
```

Config: `examples/train_lora/qwen3_lora_sft_ds3.yaml`
```yaml
model_name_or_path: /mnt/data/models/Qwen3-30B-A3B
finetuning_type: lora
lora_rank: 8
lora_target: all
deepspeed: examples/deepspeed/ds_z3_offload_config.json
per_device_train_batch_size: 1
bf16: true
```

## Observed Behavior

CPU memory collection happens in **two distinct phases**:

1. **Phase 1 (Spike)**: CPU memory rises to ~300GB
   - This value resembles **full fine-tuning** memory, not LoRA fine-tuning
   - Much of this appears to be "cached" memory (OS page cache)
2. **Phase 2 (Drop)**: Memory drops to ~120GB and stabilizes for normal training
   - This is the expected memory for LoRA fine-tuning with ZeRO-3

## User's Initial Hypothesis

> DeepSpeed might be treating the 235B base model parameters as trainable, creating
> Adam FP32 master weights + momentum/variance (m/v) states on CPU for ALL parameters,
> not just the LoRA parameters.

If true, for a 30B model:
- FP32 master weights: 30B x 4 bytes = 120GB
- Momentum (m): 30B x 4 bytes = 120GB
- Variance (v): 30B x 4 bytes = 120GB
- Total: 360GB (close to observed ~300GB considering overhead)

## Goal

Understand the true root cause and reduce the ~300GB CPU memory spike during initialization.

## Model Details

- **Qwen3-30B-A3B**: MoE (Mixture of Experts) model
- Total parameters: ~30B
- Active parameters per token: ~3B
- BF16 model size: ~60GB
- 16 safetensors checkpoint shards
- Has **weight conversions**: per-expert `gate_proj` + `up_proj` are fused into `gate_up_proj`
