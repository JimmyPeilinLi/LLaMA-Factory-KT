# KT-compatible attention LoRA variants

This branch implements the three algorithms defined by
`05-cross-variant-le-ttq-selection-and-integration.md` entirely in LLaMA-Factory. It does not require a
KTransformers source change. All three methods use exact canonical GPU attention module names and reject MLP,
router, routed/shared expert, fused-wrapper, and LoRA Expert paths.

The audited Qwen3.5-397B-A17B inventory is 190 logical projections: 30 linear-attention layers times five
families plus 10 full-attention layers times four families. LLaMA-Factory resolves and hashes that inventory before
PEFT injection, then verifies every injected A/B pair by owning PEFT module identity.

## Implementations

- `AltLoRA-Attn`: ordinary AltLoRA (not AltLoRA+) in `altlora_attn.py`. It starts with B because PEFT initializes B
  to zero, alternates B/A at complete optimizer-step boundaries, uses FP32 rank-space `torch.linalg.solve` with
  Tikhonov regularization, and applies the paper's first-moment basis alignment. Only selected attention A/B pairs
  leave AdamW; KT fused expert-LoRA and optional LoRA Expert parameters remain in the baseline AdamW groups.
  Both factor gradients may be materialized for FSDP compatibility, but the trainer discards the inactive factor
  before unscale/clipping/step, so the update is exact while the paper's half-backward memory saving is not claimed.
- `PLoP-Attn`: attention-only PLoP in `plop_attn.py`. `scripts/plan_plop_attn_kt.py` performs a separate
  inference-only NFN probe on frozen training examples, masks padding, selects the lowest-scoring K projection
  families, and writes four hashed artifacts. Final SFT starts in a new process and injects only the exact names in
  the frozen target artifact. It never uses suffix expansion or variable rank.
- `Bi-LoRA-Attn`: rank-sliced Bi-LoRA in `bilora_attn.py`. Attention PEFT uses total rank 16 and alpha 32, preserving
  scale 2. The first rank 8 receives AdamW descent; the second rank 8 receives SGD ascent and is projected after each
  optimizer step to the global `sqrt(sum_i ||B2_i A2_i||_F^2) <= rho` ball. Evaluation masks the auxiliary A channels
  with local PEFT-module hooks. Every checkpoint also receives a deployable `adapter_primary/` rank-8 export. The
  public `lora_rank: 8, lora_alpha: 16` contract is intentionally retained so KT fused expert-LoRA remains unchanged.
  Its separate `fused_expert_lora.safetensors` is copied byte-for-byte into the primary export, never rank-sliced.

The algorithms were cross-checked against the paper equations and the following official-code revisions:

- AltLoRA: `94b219882abcaa6004d2029fd8a09cd190a8a9aa`
- PLoP: `561524d0e10bfd4380c0211366cc775a11819204`
- Bi-LoRA: `e2db644c1c68f069af3bc8737222e9951ed61b3e`

## PLoP planning

Run the planner as a separate Accelerate job with the same KT/FSDP configuration intended for training:

```bash
accelerate launch --config_file examples/ktransformers/accelerate/fsdp2_kt_int8.yaml \
  scripts/plan_plop_attn_kt.py \
  examples/ktransformers/train_lora/qwen3_5moe_plop_attn_sft_kt.yaml \
  --output-dir /path/to/frozen-plop-plan \
  --probe-examples 100 --probe-seed 20260803 --select-k 3
```

After the planner exits, set `plop_attn_target_manifest_path` to `plop_attn_exact_targets.json` and
`plop_attn_score_manifest_path` to `plop_attn_nfn_scores.json` in both the standalone and `+LE` SFT YAMLs. A final
run fails closed if either artifact does not match the current eligible inventory or if the target families are not
the lowest K families in the score artifact.

## Example matrix

The six YAMLs in this directory cover each base method alone and with the LLaMA-Factory-owned LoRA Expert:

- `qwen3_5moe_altlora_attn_sft_kt.yaml`
- `qwen3_5moe_altlora_attn_lora_expert_sft_kt.yaml`
- `qwen3_5moe_plop_attn_sft_kt.yaml`
- `qwen3_5moe_plop_attn_lora_expert_sft_kt.yaml`
- `qwen3_5moe_bilora_attn_sft_kt.yaml`
- `qwen3_5moe_bilora_attn_lora_expert_sft_kt.yaml`

Use a fresh process for every final run. Do not reuse the PLoP probe process. The `+LE` YAML in each pair changes
only LoRA Expert settings and its output path; variant method settings stay identical.
