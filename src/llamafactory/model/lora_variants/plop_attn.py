# Copyright 2026 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import hashlib
import json
import math
import os
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch import nn

from .common import (
    ATTENTION_FAMILIES,
    TargetManifest,
    canonical_json_sha256,
    file_sha256,
    get_logical_weight,
    load_target_manifest,
    select_target_manifest,
    tensor_to_full,
)


PLOP_NFN_SCORES_NAME = "plop_attn_nfn_scores.json"
PLOP_SELECTED_FAMILIES_NAME = "plop_attn_selected_families.json"
PLOP_EXACT_TARGETS_NAME = "plop_attn_exact_targets.json"
PLOP_PROBE_MANIFEST_NAME = "plop_attn_probe_manifest.json"
PLOP_ARTIFACT_SCHEMA_VERSION = 1


def _atomic_json_dump(payload: dict[str, Any], path: Path) -> None:
    temporary_path = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    with open(temporary_path, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    os.replace(temporary_path, path)


def compute_nfn_per_token(
    weight: torch.Tensor,
    hidden_states: torch.Tensor,
    random_hidden_states: torch.Tensor,
    *,
    epsilon: float = 1.0e-8,
) -> torch.Tensor:
    # PyTorch batches row vectors, so W z from the paper is written as z @ W.T here.  Keep this
    # function free of aggregation policy: it should map one mathematical score to one valid token.
    if weight.ndim != 2 or hidden_states.ndim != 2 or random_hidden_states.shape != hidden_states.shape:
        raise ValueError("PLoP-Attn NFN expects a 2D weight and equally-shaped 2D real/random inputs.")
    if hidden_states.shape[1] != weight.shape[1]:
        raise ValueError("PLoP-Attn NFN input width does not match the weight input width.")
    actual_norm = (hidden_states @ weight.mT).norm(dim=-1)
    random_output_norm = (random_hidden_states @ weight.mT).norm(dim=-1).clamp_min(epsilon)
    return actual_norm / random_output_norm


class PLoPAttnScorer:
    r"""Forward-only deterministic NFN scorer restricted to an already-audited attention inventory.

    Guidance for adding another placement/scoring algorithm:

    - Candidate discovery is intentionally outside this class.  Consume a ``TargetManifest`` instead
      of scanning ``named_modules`` again, so a new score cannot silently include MLP or expert paths.
    - Put the per-token mathematical statistic in a small pure helper such as
      ``compute_nfn_per_token``.  The hook should only obtain ``z`` and the logical frozen ``W``.
    - Preserve the reduction hierarchy: valid tokens -> examples -> probe batches/modules -> families.
      Do not replace it with one global token-weighted mean unless the new method defines that formula.
    - Placement is a two-phase contract.  Scoring emits hashed exact-name artifacts; final training
      reloads those artifacts in a fresh process.  A future method should not mutate PEFT targets online.
    """

    def __init__(
        self,
        model: nn.Module,
        eligible: TargetManifest,
        *,
        seed: int = 20260803,
        epsilon: float = 1.0e-8,
    ) -> None:
        if eligible.kind != "eligible":
            raise ValueError("PLoP-Attn scorer requires an eligible target manifest.")
        self.model = model
        self.eligible = eligible
        self.seed = int(seed)
        self.epsilon = float(epsilon)
        self.started_at = time.monotonic()
        self._active_batch: int | None = None
        self._attention_mask: torch.Tensor | None = None
        self._seen_in_batch: set[str] = set()
        self._score_sum = {record.name: 0.0 for record in eligible.records}
        self._score_count = {record.name: 0 for record in eligible.records}
        self._probe_call_count = {record.name: 0 for record in eligible.records}
        self._valid_token_count = {record.name: 0 for record in eligible.records}
        self._handles = []
        for record in eligible.records:
            module = model.get_submodule(record.name)
            if get_logical_weight(module) is None:
                raise ValueError(f"PLoP-Attn target no longer exposes one logical base weight: {record.name}.")
            self._handles.append(module.register_forward_pre_hook(self._make_hook(record.name), with_kwargs=True))

    def _make_hook(self, name: str):
        def hook(module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
            if self._active_batch is None:
                return
            if name in self._seen_in_batch:
                raise RuntimeError(f"PLoP-Attn logical projection was invoked more than once in one probe: {name}.")
            self._seen_in_batch.add(name)
            if args:
                hidden_states = args[0]
            else:
                hidden_states = kwargs.get("hidden_states")
            if not isinstance(hidden_states, torch.Tensor) or hidden_states.ndim < 2:
                raise TypeError(f"PLoP-Attn hook received invalid hidden states at {name}.")

            weight = get_logical_weight(module)
            if weight is None:
                raise RuntimeError(f"PLoP-Attn lost the logical weight for {name} during scoring.")
            weight = tensor_to_full(weight).detach().to(dtype=torch.float32)
            hidden_states = hidden_states.detach().to(device=weight.device, dtype=torch.float32)
            if hidden_states.shape[-1] != weight.shape[1]:
                raise ValueError(
                    f"PLoP-Attn input/weight mismatch at {name}: {hidden_states.shape[-1]} != {weight.shape[1]}."
                )

            if self._attention_mask is not None:
                attention_mask = self._attention_mask.to(device=hidden_states.device, dtype=torch.bool)
                if attention_mask.ndim != 2:
                    raise ValueError("PLoP-Attn requires a 2D tokenizer attention mask during probing.")
                if hidden_states.ndim == 2 and attention_mask.numel() == hidden_states.shape[0]:
                    hidden_states = hidden_states.view(*attention_mask.shape, hidden_states.shape[-1])
                if hidden_states.ndim != 3 or tuple(hidden_states.shape[:2]) != tuple(attention_mask.shape):
                    raise ValueError(
                        f"PLoP-Attn attention mask shape mismatch at {name}: "
                        f"{tuple(attention_mask.shape)} != {tuple(hidden_states.shape[:-1])}."
                    )
            else:
                if hidden_states.ndim == 2:
                    hidden_states = hidden_states.unsqueeze(0)
                if hidden_states.ndim != 3:
                    raise ValueError(f"PLoP-Attn requires [batch, sequence, hidden] inputs at {name}.")
                attention_mask = torch.ones(hidden_states.shape[:2], device=hidden_states.device, dtype=torch.bool)
            valid_per_example = attention_mask.sum(dim=-1)
            if torch.any(valid_per_example == 0):
                raise ValueError(f"PLoP-Attn probe contains no valid tokens at {name}.")

            # The draw key includes the logical module and probe batch.  This makes the random reference
            # independent of hook registration order while retaining the paper's equal-norm Gaussian draw.
            generator = torch.Generator(device=hidden_states.device)
            seed_payload = f"{self.seed}:{self._active_batch}:{name}".encode()
            draw_seed = int.from_bytes(hashlib.sha256(seed_payload).digest()[:8], "little") % (2**63 - 1)
            generator.manual_seed(draw_seed)
            gaussian = torch.randn(
                hidden_states.shape,
                generator=generator,
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )
            input_norm = hidden_states.norm(dim=-1, keepdim=True)
            random_norm = gaussian.norm(dim=-1, keepdim=True).clamp_min(self.epsilon)
            random_hidden_states = gaussian * (input_norm / random_norm)
            token_scores = compute_nfn_per_token(
                weight,
                hidden_states.reshape(-1, hidden_states.shape[-1]),
                random_hidden_states.reshape(-1, random_hidden_states.shape[-1]),
                epsilon=self.epsilon,
            ).view(hidden_states.shape[:2])
            # First average tokens within each example.  ``finalize`` then averages examples for a module
            # and modules for a family, matching the frozen PLoP-Attn selection contract.
            example_scores = (token_scores * attention_mask).sum(dim=-1) / valid_per_example
            if not torch.isfinite(example_scores).all():
                raise FloatingPointError(f"PLoP-Attn produced a non-finite NFN score at {name}.")
            self._score_sum[name] += float(example_scores.sum().item())
            self._score_count[name] += hidden_states.shape[0]
            self._probe_call_count[name] += 1
            self._valid_token_count[name] += int(valid_per_example.sum().item())

        return hook

    @contextmanager
    def batch(self, batch_index: int, attention_mask: torch.Tensor | None) -> Iterator[None]:
        if self._active_batch is not None:
            raise RuntimeError("PLoP-Attn probe batches cannot be nested.")
        self._active_batch = int(batch_index)
        self._attention_mask = attention_mask
        self._seen_in_batch.clear()
        body_failed = False
        try:
            # FSDP2 unsharding temporarily reads parameter version counters.  ``inference_mode`` creates
            # versionless tensors and therefore fails before the first projection hook on torch 2.9.  PLoP
            # only needs to suppress autograd, so ``no_grad`` is the exact (and FSDP-compatible) contract.
            with torch.no_grad():
                yield
        except BaseException:
            body_failed = True
            raise
        finally:
            missing = sorted(set(self._score_sum) - self._seen_in_batch)
            self._active_batch = None
            self._attention_mask = None
            # Preserve the original forward exception.  The completeness error is meaningful only after a
            # successful forward; raising it while unwinding would hide the actual runtime incompatibility.
            if missing and not body_failed:
                raise RuntimeError(f"PLoP-Attn probe did not invoke every eligible target: {missing[:5]}.")

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def finalize(self) -> dict[str, Any]:
        # Accumulate only scalar sufficient statistics across data-parallel ranks.  Module/family ordering
        # comes from the immutable eligible manifest, not from runtime hook completion order.
        names = [record.name for record in self.eligible.records]
        first_module = self.model.get_submodule(self.eligible.records[0].name)
        first_weight = get_logical_weight(first_module)
        if first_weight is None:
            raise RuntimeError("PLoP-Attn lost its first logical weight before score aggregation.")
        values = torch.tensor(
            [
                [
                    self._score_sum[name],
                    self._score_count[name],
                    self._valid_token_count[name],
                    self._probe_call_count[name],
                ]
                for name in names
            ],
            dtype=torch.float64,
            device=first_weight.device,
        )
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(values, op=dist.ReduceOp.SUM)
        values = values.cpu()

        module_scores = []
        family_values: dict[str, list[float]] = {family: [] for family in ATTENTION_FAMILIES}
        records_by_name = {record.name: record for record in self.eligible.records}
        for index, name in enumerate(names):
            score_sum, score_count, valid_tokens, probe_calls = values[index].tolist()
            if score_count <= 0:
                raise RuntimeError(f"PLoP-Attn has no NFN observations for {name}.")
            score = score_sum / score_count
            record = records_by_name[name]
            family_values[record.family].append(score)
            module_scores.append(
                {
                    "name": name,
                    "family": record.family,
                    "nfn": score,
                    "probe_calls": int(probe_calls),
                    "probe_examples": int(score_count),
                    "valid_tokens": int(valid_tokens),
                    "weight_shape": list(record.weight_shape),
                    "weight_dtype": record.weight_dtype,
                    "weight_device": record.weight_device,
                }
            )

        family_scores = {}
        for family, scores in family_values.items():
            if not scores:
                raise RuntimeError(f"PLoP-Attn has no eligible modules for family {family}.")
            family_scores[family] = {"nfn": sum(scores) / len(scores), "module_count": len(scores)}
        core = {
            "schema_version": PLOP_ARTIFACT_SCHEMA_VERSION,
            "method": "plop_attn",
            "eligible_sha256": self.eligible.sha256,
            "seed": self.seed,
            "num_random_draws": 1,
            "epsilon": self.epsilon,
            "module_scores": module_scores,
            "family_scores": family_scores,
            "probe_wall_seconds": time.monotonic() - self.started_at,
        }
        return {**core, "sha256": canonical_json_sha256(core)}


def _validate_score_artifact(payload: dict[str, Any], eligible: TargetManifest) -> None:
    if payload.get("schema_version") != PLOP_ARTIFACT_SCHEMA_VERSION or payload.get("method") != "plop_attn":
        raise ValueError("PLoP-Attn score artifact has an unsupported schema or method.")
    core = {key: value for key, value in payload.items() if key != "sha256"}
    if payload.get("sha256") != canonical_json_sha256(core):
        raise ValueError("PLoP-Attn score artifact content hash is invalid.")
    if payload.get("eligible_sha256") != eligible.sha256:
        raise ValueError("PLoP-Attn score artifact was produced for a different eligible inventory.")
    if payload.get("num_random_draws") != 1:
        raise ValueError("PLoP-Attn first-version score artifacts require exactly one random draw.")
    module_scores = payload.get("module_scores")
    family_scores = payload.get("family_scores")
    if not isinstance(module_scores, list) or not isinstance(family_scores, dict):
        raise ValueError("PLoP-Attn score artifact is missing module/family score mappings.")
    eligible_by_name = {record.name: record for record in eligible.records}
    if [record.get("name") for record in module_scores] != list(eligible.exact_target_names):
        raise ValueError("PLoP-Attn module scores do not exactly match the eligible target order.")
    recomputed: dict[str, list[float]] = {family: [] for family in ATTENTION_FAMILIES}
    for record in module_scores:
        name = record["name"]
        eligible_record = eligible_by_name[name]
        score = float(record["nfn"])
        if record.get("family") != eligible_record.family or not math.isfinite(score) or score < 0:
            raise ValueError(f"PLoP-Attn score artifact has an invalid record for {name}.")
        if int(record.get("probe_examples", 0)) <= 0 or int(record.get("valid_tokens", 0)) <= 0:
            raise ValueError(f"PLoP-Attn score artifact has no valid observations for {name}.")
        recomputed[eligible_record.family].append(score)
    if set(family_scores) != set(ATTENTION_FAMILIES):
        raise ValueError("PLoP-Attn family scores do not cover the exact attention family allowlist.")
    for family, scores in recomputed.items():
        if not scores:
            raise ValueError(f"PLoP-Attn score artifact has no modules for {family}.")
        expected_score = sum(scores) / len(scores)
        if int(family_scores[family].get("module_count", 0)) != len(scores) or not math.isclose(
            float(family_scores[family]["nfn"]), expected_score, rel_tol=1.0e-12, abs_tol=1.0e-12
        ):
            raise ValueError(f"PLoP-Attn family score is inconsistent for {family}.")


def load_plop_training_selection(
    eligible: TargetManifest,
    target_manifest_path: str | Path,
    *,
    select_k: int,
    score_manifest_path: str | Path | None = None,
) -> tuple[TargetManifest, tuple[str, ...]]:
    # This is the phase boundary: convert a frozen placement decision back into the current model's
    # canonical records, then prove it selects complete families and (when supplied) the lowest scores.
    frozen = load_target_manifest(target_manifest_path, expected_kind="selected")
    selected = select_target_manifest(eligible, frozen.exact_target_names)
    if selected.sha256 != frozen.sha256 or selected.records != frozen.records:
        raise ValueError("PLoP-Attn target artifact does not exactly match the current model inventory.")
    selected_families = selected.families
    if len(selected_families) != select_k:
        raise ValueError(f"PLoP-Attn expected {select_k} selected families, found {len(selected_families)}.")
    expected_names = tuple(record.name for record in eligible.records if record.family in selected_families)
    if expected_names != selected.exact_target_names:
        raise ValueError("PLoP-Attn target artifact must contain every module from each selected family.")

    if score_manifest_path is not None:
        with open(score_manifest_path, encoding="utf-8") as stream:
            score_payload = json.load(stream)
        _validate_score_artifact(score_payload, eligible)
        ranked_families = sorted(
            score_payload["family_scores"], key=lambda family: (score_payload["family_scores"][family]["nfn"], family)
        )
        if tuple(sorted(ranked_families[:select_k])) != selected_families:
            raise ValueError("PLoP-Attn target families are not the lowest-NFN families in the score artifact.")
    return selected, selected_families


def write_plop_artifacts(
    output_dir: str | Path,
    eligible: TargetManifest,
    scores: dict[str, Any],
    *,
    select_k: int,
    probe_manifest: dict[str, Any],
) -> TargetManifest:
    _validate_score_artifact(scores, eligible)
    ranked_families = sorted(scores["family_scores"], key=lambda key: (scores["family_scores"][key]["nfn"], key))
    selected_families = tuple(sorted(ranked_families[:select_k]))
    if len(selected_families) != select_k:
        raise ValueError(f"Cannot select {select_k} PLoP-Attn families from the score artifact.")
    selected = select_target_manifest(
        eligible, [record.name for record in eligible.records if record.family in selected_families]
    )
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    score_path = output_path / PLOP_NFN_SCORES_NAME
    family_path = output_path / PLOP_SELECTED_FAMILIES_NAME
    target_path = output_path / PLOP_EXACT_TARGETS_NAME
    _atomic_json_dump(scores, score_path)
    _atomic_json_dump(
        {
            "schema_version": PLOP_ARTIFACT_SCHEMA_VERSION,
            "method": "plop_attn",
            "eligible_sha256": eligible.sha256,
            "select_k": select_k,
            "selected_families": list(selected_families),
        },
        family_path,
    )
    _atomic_json_dump(selected.to_dict(), target_path)
    manifest = {
        **probe_manifest,
        "schema_version": PLOP_ARTIFACT_SCHEMA_VERSION,
        "method": "plop_attn",
        "candidate_scope": "attention_only",
        "attention_mask_rule": "tokenizer_non_padding_prompt_and_response",
        "eligible_sha256": eligible.sha256,
        "selected_sha256": selected.sha256,
        "select_k": select_k,
        "selected_families": list(selected_families),
        "exact_target_names": list(selected.exact_target_names),
        "selected_lora_parameter_count": sum(
            int(probe_manifest["lora_rank"]) * (record.in_features + record.out_features)
            for record in selected.records
        ),
        "artifact_sha256": {
            score_path.name: file_sha256(score_path),
            family_path.name: file_sha256(family_path),
            target_path.name: file_sha256(target_path),
        },
    }
    manifest["sha256"] = canonical_json_sha256(manifest)
    _atomic_json_dump(manifest, output_path / PLOP_PROBE_MANIFEST_NAME)
    return selected


__all__ = [
    "PLOP_EXACT_TARGETS_NAME",
    "PLOP_NFN_SCORES_NAME",
    "PLOP_PROBE_MANIFEST_NAME",
    "PLOP_SELECTED_FAMILIES_NAME",
    "PLoPAttnScorer",
    "compute_nfn_per_token",
    "load_plop_training_selection",
    "write_plop_artifacts",
]
