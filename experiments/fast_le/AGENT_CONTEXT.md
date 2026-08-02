# fast_le continuation context

This file is the authoritative compact context for future work. It supersedes old local-agent notes about MixLoRA, MoLoRA, TTQ shortcuts, or the earlier standard-LoRA campaign.

## Current scope

The completed scope is only the clean LLaMA-Factory-owned LoRA Expert refactor plus engineering validation.

Do not restart MixLoRA or MoLoRA integration from old context. Do not use the old TTQ runner, curve alignment, fixed-time shortcuts, or legacy KT-owned LE. Formal quality comparisons are a separate future task.

## Repository and machines

- Shared NFS repository: `/home/lpl/fast_le/LLaMA-Factory-KT`
- Branch: `fast_le`
- Base commit: `62ae362455801d4900a5132c7a30b23dc5fc3802`
- Origin: `git@github.com:JimmyPeilinLi/LLaMA-Factory-KT.git`
- Upstream: `git@github.com:hiyouga/LLaMA-Factory.git`
- Development and all new code: remote qjh shared NFS only
- Runtime validation: qjh007
- The old qjh007-local-only checkout was removed before this work.

Use a proxy only for Git clone/fetch if the cluster network requires it. Never use that proxy for model downloads.

## Core decisions

- Standard PEFT LoRA is unchanged.
- LE is a residual beside the base MoE, never inside the KT kernel.
- Native and KT paths share exactly one implementation and formula.
- Registered module name: `fast_le`.
- Metadata file: `lora_expert_config.json`.
- Legacy KT LE fields are rejected and forced off.
- Checkpoint has two weight domains: PEFT adapter/LE and existing KT fused expert LoRA.
- No KTransformers or kt_kernel source edit is allowed for this refactor.

Formula:

```text
mean_e(down_e(silu(gate_e(x)) * up_e(x)))
```

Initialization: Kaiming-uniform gate/up, exact-zero down.

## Validated artifacts

35B formal run:

`/home/lpl/fast_le/runs/qwen3_5_35b_fast_le_smoke_seed42_qjh007`

397B formal run:

`/home/lpl/fast_le/runs/qwen3_5_397b_fast_le_smoke_seed42_qjh007`

35B real resume run:

`/home/lpl/fast_le/runs/qwen3_5_35b_fast_le_resume_smoke_seed42_qjh007`

Each has a passing `fast_le_checkpoint_audit.json`. The formal runs have complete checkpoint-5 directories; the resume run has checkpoint-6.

## Important runtime detail

Qwen3.5 is multimodal even for text-only SFT. Under FSDP, LLaMA-Factory may append dummy image tokens after cutoff. Set KT's global flattened-token buffer above the derived cutoff when needed; the 397B smoke uses 1024 for an observed qlen of 584.

With this exact kt_kernel/PyTorch/FSDP2 stack, 397B non-reentrant activation checkpointing caused a C++ forward-cache underflow. The successful 397B smoke explicitly uses `disable_gradient_checkpointing: true`. The 35B formal and resume smokes validate the gradient-checkpointed path.

## Test status

- Focused tests: 11 passed.
- Full offline tree: 311 collected; 108 passed, 124 skipped, 5 xfailed, 1 xpassed, 65 failed, and 9 setup errors. Failures/errors were missing uncached Hugging Face fixtures, with no fast_le test in that set; see `/home/lpl/fast_le/test_logs/full_pytest_20260802.log`.
- 35B checkpoint-5 audit: pass.
- 397B checkpoint-5 audit: pass.
- 35B checkpoint-6 resume audit: pass.
- Targeted Ruff/format, Python compile, license, and diff checks: pass.
- No unrelated formatter changes remain.

## Future experimental boundary

A future efficacy campaign must start from this branch and separately define:

- Beyond Dialogue and Coser full data splits
- standard LoRA versus LE controls
- equal total trainable-parameter controls
- equal active/per-token parameter controls
- seeds and statistical reporting
- standard TTQ semantics without temporary time/step coercion

Do not interpret smoke losses as quality comparisons, and do not align random curves or hardcode results.
