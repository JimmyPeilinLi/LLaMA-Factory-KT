# LoRA Expert implementation report

## Repository baseline

The refactor was developed in the shared NFS checkout:

- Repository: `/home/lpl/fast_le/LLaMA-Factory-KT`
- Branch: `fast_le`
- Synchronized base: `62ae362455801d4900a5132c7a30b23dc5fc3802`
- At refactor start, `main`, `origin/main`, `upstream/main`, `fast_le`, and `origin/fast_le` all pointed to that commit.
- Mathematical reference: `9dd3907a5c863275107f908d30a8a626a7f37528`

No source file under KTransformers or `kt_kernel` is modified by this branch.

## Mathematical contract

For a base MoE output `M(x)`, the branch returns

```text
M(x) + (1 / E) * sum_e down_e(silu(gate_e(x)) * up_e(x))
```

Every LE processes every token; there is no LE router. For hidden size `h`, LE width `m`, and `E` experts, the exact additional parameter count per MoE block is

```text
3 * E * h * m
```

Initialization follows the reference implementation:

- `gate` and `up`: PyTorch Kaiming uniform.
- `down`: exact zero.
- Initial residual: bitwise zero in the input dtype.
- First backward: `down` receives gradients while zero-init makes `gate/up` gradients zero.
- After the first update: `gate/up` receive gradients.

The module computes in its parameter dtype and casts the residual back to the input dtype.

## Ownership and injection order

All new LE state belongs to LLaMA-Factory:

1. LLaMA-Factory loads the model through its normal loader.
2. KT may replace the base MoE blocks and register its wrappers.
3. LLaMA-Factory discovers ordinary PEFT LoRA targets before LE insertion, so standard LoRA target discovery is unchanged.
4. LLaMA-Factory attaches the same `LoRAExperts` module to native MoE blocks or KT wrappers.
5. A forward hook adds the LE residual to tensor, tuple, or list outputs.
6. PEFT receives `fast_le` in `modules_to_save`; rank, alpha, dropout, and target modules are otherwise unchanged.

The registered child is named `fast_le`. PEFT 0.18 treats names containing `lora_` specially; avoiding such a child name prevents duplicate/filtered checkpoint keys.

## User-facing arguments

The LE arguments are ordinary finetuning arguments:

- `use_lora_expert`
- `lora_expert_num`
- `lora_expert_intermediate_size`

They are valid only for `finetuning_type: lora`, and their positive dimensions are validated.

KT arguments do not own the LE. The deprecated fields `kt_use_lora_experts`, `kt_lora_expert_num`, and `kt_lora_expert_intermediate_size` are rejected if enabled or sized through YAML, active KT config, or environment variables. The KT environment is then forced to keep legacy LE disabled.

## Native and KT targeting

Native targeting recognizes known sparse-MoE classes and the `SparseMoeBlock` suffix. KT targeting uses the wrappers already registered by the active model. Metadata stores the exact ordered module paths and rejects path, hidden-size, or wrapper-set mismatches on reconstruction.

Both paths instantiate the same class, initialization, formula, hook, metadata schema, and PEFT save path.

## Checkpoint contract

Every training checkpoint and final output contains `lora_expert_config.json`. Schema version 1 records:

- implementation: `llamafactory`
- module name: `fast_le`
- exact formula and initialization
- `num_experts`, `intermediate_size`, and `hidden_size`
- exact ordered target module paths

The PEFT adapter contains standard attention LoRA plus LE modules. Representative key families are:

```text
...q_proj.lora_A.weight
...q_proj.lora_B.weight
...mlp.fast_le.experts.0.le_gate.weight
...mlp.fast_le.experts.0.le_up.weight
...mlp.fast_le.experts.0.le_down.weight
```

KT's existing fused expert LoRA saver remains separate and produces exactly six keys per MoE layer:

```text
layers.N.experts.{gate,up,down}_lora_{a,b}
```

Loading an adapter first reads `lora_expert_config.json`, reconstructs LE before PEFT weight loading, validates the model paths, and then follows normal PEFT resume or merge behavior. A real Trainer resume also restores KT fused tensors, optimizer, scheduler, RNG, and Trainer state.

## KT compatibility details

`KTransformersArguments.apply_kt_config` now normalizes the Transformers, Accelerate, and environment integration points without changing KT source.

KT flattens each per-device batch and its distributed path gathers local qlens on rank 0. Therefore the derived buffer capacity is:

```text
cutoff_len * max(train_microbatch, eval_microbatch) * WORLD_SIZE
```

The optional `kt_model_max_length` is a minimum global flattened-token capacity. It is useful when a multimodal collator appends dummy image tokens after preprocessing. It changes buffer allocation only, not data truncation.

FSDP2 uses non-reentrant gradient checkpointing. The branch also selects non-reentrant checkpointing for KT and allows frozen Conv3D vision towers on torch 2.9 while still rejecting trainable Conv3D.

## Main code locations

- `src/llamafactory/model/model_utils/lora_expert.py`: module, targeting, hook, metadata save/load.
- `src/llamafactory/model/adapter.py`: PEFT creation, metadata reconstruction, and `modules_to_save`.
- `src/llamafactory/hparams/finetuning_args.py`: LE user arguments.
- `src/llamafactory/hparams/model_args.py`: KT normalization, legacy rejection, and token capacity.
- `src/llamafactory/train/callbacks.py` and `train/tuner.py`: metadata at checkpoint/final save.
- `src/llamafactory/model/model_utils/checkpointing.py`: KT/FSDP2 checkpoint mode.
- `src/llamafactory/model/loader.py`: torch 2.9 Conv3D compatibility guard.

## Non-goals

This refactor does not implement MixLoRA, MoLoRA, LE routing, TTQ curve alignment, or a new KTransformers kernel. It does not claim training-quality superiority from five-step smoke tests.
