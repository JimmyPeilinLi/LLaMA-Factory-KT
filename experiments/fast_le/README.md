# FAST LoRA Expert refactor

This directory contains the reproducible validation assets for the `fast_le` branch. The implementation is owned entirely by LLaMA-Factory; KTransformers remains the base MoE execution backend.

## Scope

- Preserve standard PEFT LoRA rank, alpha, dropout, targets, YAML entry point, and Trainer behavior.
- Add one independent GPU LoRA Expert residual branch beside every supported MoE block.
- Use the same module and formula for native Hugging Face MoE and KT-wrapped MoE.
- Save LE weights through PEFT `modules_to_save` and save reconstruction metadata separately.
- Explicitly reject the deprecated KT-owned LE switches.
- Do not include MixLoRA, MoLoRA, TTQ timing shortcuts, or formal quality claims in this refactor.

## Files

- `IMPLEMENTATION_REPORT.md`: math, ownership boundary, integration flow, and checkpoint contract.
- `VALIDATION_REPORT.md`: unit tests, 35B/397B smoke results, diagnostics, and checkpoint audits.
- `AGENT_CONTEXT.md`: compact continuation context that supersedes old experimental-agent notes.
- `qwen3_5_35b_le_smoke_qjh007.yaml`: formal five-step 35B smoke.
- `qwen3_5_397b_le_smoke_qjh007.yaml`: formal five-step 397B smoke.
- `qwen3_5_35b_le_resume_smoke_qjh007.yaml`: real checkpoint-5 to checkpoint-6 resume smoke.
- `fsdp2_kt_bf16_1gpu_qjh007.yaml`: single-process FSDP2/KT launcher used by the smokes.
- `audit_checkpoint.py`: offline safetensors, metadata, metrics, and update audit.

The user-facing example is `examples/ktransformers/train_lora/qwen3_5moe_lora_expert_sft_kt.yaml`. It uses ordinary LLaMA-Factory arguments and contains no site-specific smoke paths.

## Standard launch

From the repository root on qjh007:

```bash
python -m accelerate.commands.launch \
  --config_file experiments/fast_le/fsdp2_kt_bf16_1gpu_qjh007.yaml \
  llamafactory-cli train \
  experiments/fast_le/qwen3_5_35b_le_smoke_qjh007.yaml
```

The 397B and resume commands differ only in the final YAML path. There is no custom training runner.

## Result locations

The checked checkpoints and logs are intentionally outside Git:

- `/home/lpl/fast_le/runs/qwen3_5_35b_fast_le_smoke_seed42_qjh007`
- `/home/lpl/fast_le/runs/qwen3_5_397b_fast_le_smoke_seed42_qjh007`
- `/home/lpl/fast_le/runs/qwen3_5_35b_fast_le_resume_smoke_seed42_qjh007`

Each run has a `fast_le_checkpoint_audit.json` with `status: pass` and `errors: []`.
