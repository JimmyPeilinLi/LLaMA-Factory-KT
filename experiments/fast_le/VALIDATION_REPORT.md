# LoRA Expert validation report

## Environment

Validation ran on qjh007 from the shared NFS repository, using the standard LLaMA-Factory YAML/CLI path.

- LLaMA-Factory: 0.9.6.dev0
- Python: 3.11.15
- torch: 2.9.1+cu128
- Transformers-KT: 5.6.0
- Accelerate-KT: 1.14.0
- PEFT: 0.18.1
- kt_kernel: 0.6.2.post4
- GPU: qjh007 GPU0, H20 with 97,871 MiB
- Dataset: the existing Beyond Dialogue train/validation files in the prior experiment data directory
- Seed and data seed: 42

No model was downloaded through a proxy. The 35B model was read from qjh007 storage and the 397B model from the cluster-mounted qjh002 model path.

## Automated regression tests

The focused suite completed 11/11 tests with four benign third-party warnings. Coverage includes:

- elementwise formula reference
- exact zero-init no-op
- first-step and second-step gradient contract
- `E` experts versus one concatenated wide SwiGLU MLP
- exact parameter count and exact trainable-name set
- native and KT target equivalence
- tensor and tuple output hooks
- real tiny Qwen3.5-MoE no-op
- CUDA placement and single-rank FSDP2
- PEFT safetensors save/reload output equality
- LLaMA-Factory automatic create, resume, inference merge, and metadata callback
- standard YAML parsing and legacy KT rejection
- world-size-aware KT capacity and explicit capacity override
- torch 2.9 frozen/trainable Conv3D guard
- KT non-reentrant checkpoint selection

The complete `tests + tests_v1` tree also collected all 311 tests. An offline full run produced 108 passed, 124 skipped, 5 expected failures, 1 unexpected pass, 65 failures, and 9 setup errors. The failed/error set did not contain a fast_le test. Environment failures were dominated by missing uncached Hugging Face fixtures such as `llamafactory/tiny-random-qwen3`; optional-backend tests also reported that Megatron Bridge was not installed. The full log is `/home/lpl/fast_le/test_logs/full_pytest_20260802.log`. Models were not downloaded merely to turn these environment failures green.

Targeted Ruff, format, Python compilation, `git diff --check`, and the repository license check passed. Repository-wide `make style` was not used for the final tree because qjh007 has Ruff 0.16.1 while the Makefile pins 0.15.5; four unrelated formatting side effects were removed. All changed Python files pass the available Ruff checks.

## 35B five-step smoke

Configuration: `qwen3_5_35b_le_smoke_qjh007.yaml`

- Model: Qwen3.5-35B-A3B
- Data: 48 Beyond Dialogue train and 48 validation samples
- LoRA: rank 8, alpha 16, dropout 0.1, target `all`
- LE: E=1, width=256, 40 blocks
- LE parameters: 62,914,560
- LLaMA-Factory visible trainable parameters: 71,040,000
- KT fused expert LoRA tensors injected into optimizer: 240
- Standard launcher: one-process FSDP2 + KT AMXBF16
- Gradient checkpointing: enabled, non-reentrant
- Optimizer steps: 5

Finite train losses:

```text
5.53125, 5.375, 3.9609375, 3.7421875, 3.453125
```

Final eval loss: `3.692708`.

Checkpoint-5 audit: `pass`, no errors.

- Adapter: 500 tensors, 284,237,272 bytes
- Attention: 190 LoRA A and 190 paired LoRA B tensors
- LE: 40 gate, 40 up, and 40 down tensors
- Fused expert LoRA: 240 tensors, 1,258,316,600 bytes
- All tensors finite
- All 190 attention B, all 40 zero-initialized LE down, and all 120 fused B tensors became nonzero
- No legacy `lora_expert` key and no duplicate `modules_to_save` key

## 397B five-step smoke

Configuration: `qwen3_5_397b_le_smoke_qjh007.yaml`

- Model: Qwen3.5-397B-A17B, 94 local BF16 shards
- Data: 8 Beyond Dialogue train and 8 validation samples
- LoRA: rank 8, alpha 16, dropout 0.1, target `all`
- LE: E=1, width=256, 60 blocks
- LE parameters: 188,743,680
- LLaMA-Factory visible trainable parameters: 211,522,560
- KT fused expert LoRA tensors injected into optimizer: 360
- Optimizer steps: 5
- Gradient checkpointing: disabled for this smoke; see diagnostics below
- Observed GPU memory during training: about 62.3 GiB, below the 97.9 GiB device capacity

Finite train losses:

```text
14.0625, 14.0625, 11.4375, 11.9375, 8.625
```

Finite grad norms:

```text
1612, 1824, 289, 205.1, 291.1
```

Final eval loss: `10.65625`.

Checkpoint-5 audit: `pass`, no errors.

- Adapter: 750 tensors, 846,206,696 bytes
- Attention: 285 LoRA A and 285 paired LoRA B tensors
- LE: 60 gate, 60 up, and 60 down tensors
- Fused expert LoRA: 360 tensors, 7,549,786,040 bytes
- Optimizer state: 16,792,798,423 bytes
- All tensors finite
- All 285 attention B, all 60 zero-initialized LE down, and all 180 fused B tensors became nonzero
- Metadata contains the exact 60 ordered `model.language_model.layers.N.mlp` paths
- No legacy or duplicate key family

## Real checkpoint resume smoke

Configuration: `qwen3_5_35b_le_resume_smoke_qjh007.yaml`

The resume YAML intentionally omits `use_lora_expert`. It supplies checkpoint-5 as both the adapter path and Trainer resume path.

Observed behavior:

- `lora_expert_config.json` automatically reconstructed 40 LE modules.
- PEFT loaded the saved adapter.
- Trainer injected 240 fused expert LoRA parameters into the optimizer and loaded the checkpoint model state.
- Trainer reported resume from global step 5 and fast-forwarded ten micro-batches.
- Step 6 completed with loss `3.352` and grad norm `7.681`; its learning rate was already zero because checkpoint-5 had exhausted the original scheduler, so this step was not used as evidence of a new weight update.
- Final eval loss was `3.6901`.
- Checkpoint-6 audit passed with `global_step: 6` and no errors.
- Adapter safetensors were byte-identical between checkpoint-5 and checkpoint-6: SHA-256 `746b07435d01d065470bf3d1991d87caf90891d7a7f5d0a02177961c667a7d3a`.
- Fused expert LoRA safetensors were byte-identical between checkpoint-5 and checkpoint-6: SHA-256 `61f0f68f971f20c06e36b1254618f112ce52af3efdd7f671bdb9054a89614580`.

This is a real checkpoint load, execution, and byte-identical resave test. Parameter-update evidence comes from the five-step formal runs above.

## Diagnostic history and resolved runtime constraints

Three failed 397B attempts were retained as external logs, not hidden or overwritten.

1. The initial run failed before any optimizer step because the preprocessed cutoff was 512 but the FSDP multimodal collator appended dummy image tokens, producing qlen 584. A global KT buffer minimum of 1024 fixed this without changing data.
2. With capacity fixed, KT's non-reentrant checkpoint path hit a fused forward-cache underflow at the first backward.
3. A 35B matrix showed that some larger micro-batch or accumulation contexts passed, but the workaround did not generalize to the 60-layer 397B model.
4. Using LLaMA-Factory's actual `disable_gradient_checkpointing: true` model argument eliminated the cache underflow. The successful 397B run retained batch size 1 and gradient accumulation 1.

The successful 35B formal smoke still validates LE with FSDP2 and non-reentrant gradient checkpointing. The 397B no-GC choice is an explicit environment/runtime compatibility setting, not a mathematical change.

External diagnostic logs:

- `stdout_attempt1_qlen584.log`
- `stdout_attempt2_cache_underflow.log`
- `stdout_attempt3_cache_underflow_gradacc2.log`

## Reproducible checkpoint audit

Example:

```bash
python experiments/fast_le/audit_checkpoint.py \
  /path/to/checkpoint-5 \
  --expected-layers 60 \
  --expected-steps 5 \
  --expected-rank 8 \
  --expected-alpha 16 \
  --expected-dropout 0.1 \
  --output-json /path/to/fast_le_checkpoint_audit.json
```

The audit requires standard PEFT, LE metadata, fused expert LoRA, optimizer, FSDP state, RNG, scheduler, Trainer state, and training arguments. It validates key families, exact layer coverage, shapes, finite values, paired A/B keys, nonzero updates, metrics, and absence of legacy, duplicate, or unknown keys.

## Interpretation limits

These runs prove integration, save/load, finite optimization, and parameter updates. Five steps on small sample subsets do not establish downstream quality or that LE outperforms standard LoRA. Formal Beyond Dialogue and Coser comparisons, parameter-matched controls, repeated seeds, and any MixLoRA/MoLoRA work are intentionally deferred.
