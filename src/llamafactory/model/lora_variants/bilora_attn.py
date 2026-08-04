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

import json
import os
import shutil
import weakref
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .common import (
    KT_LORA_ELIGIBLE_MANIFEST_NAME,
    KT_LORA_SELECTED_MANIFEST_NAME,
    KT_LORA_VARIANT_CONFIG_NAME,
    KTLoraVariantConfig,
    LoraPair,
    adamw_constructor_kwargs,
    copy_full_to_tensor,
    rebuild_adamw_param_groups,
    tensor_to_full,
)


class BiLoraAttnOptimizer(torch.optim.AdamW):
    r"""Bi-LoRA attention optimizer: AdamW descent on rank r1 and constrained SGD ascent on rank r2.

    Contributor map from the Bi-LoRA equations to code:

    - PEFT stores one ``A: [r1+r2, in]`` and one ``B: [out, r1+r2]``.  Slices ``A[:r1]``/``B[:, :r1]``
      are the primary branch and ``A[r1:]``/``B[:, r1:]`` are the auxiliary branch, giving
      ``Delta W = s (B1 A1 + B2 A2)`` without introducing new model parameters.
    - ``_update_attention_pair`` implements the descent/ascent rule.  ``_project_auxiliary_perturbation``
      implements the constraint and is called once per complete optimizer step, never per microbatch.
    - Optimizer moments are allocated only for primary slices.  Method-global values such as ``rho`` and
      the completed-step count must also be round-tripped by ``state_dict``/``load_state_dict``.
    - ``super().step()`` remains responsible for all non-attention parameters.  New adversarial variants
      must not route KT fused expert-LoRA or optional LoRA Expert parameters into the ascent branch.
    """

    _STATE_KEY = "kt_bilora_attn"

    def __init__(
        self,
        baseline_optimizer: torch.optim.Optimizer,
        pairs: tuple[LoraPair, ...],
        *,
        primary_rank: int,
        auxiliary_rank: int,
        rho: float,
        auxiliary_lr_ratio: float,
    ) -> None:
        if baseline_optimizer.state:
            raise ValueError("Bi-LoRA-Attn must replace AdamW before the baseline optimizer has taken a step.")
        for pair in pairs:
            if pair.lora_a.shape[0] != primary_rank + auxiliary_rank:
                raise ValueError(
                    f"Bi-LoRA-Attn rank mismatch for {pair.name}: expected {primary_rank + auxiliary_rank}, "
                    f"found {pair.lora_a.shape[0]}."
                )

        param_groups = rebuild_adamw_param_groups(baseline_optimizer, pairs)
        super().__init__(param_groups, **adamw_constructor_kwargs(baseline_optimizer))
        self.primary_rank = int(primary_rank)
        self.auxiliary_rank = int(auxiliary_rank)
        self.rho = float(rho)
        self.auxiliary_lr_ratio = float(auxiliary_lr_ratio)
        self.variant_step = 0

    @staticmethod
    def _full_fp32(tensor: torch.Tensor) -> torch.Tensor:
        return tensor_to_full(tensor).detach().to(dtype=torch.float32)

    @torch.no_grad()
    def _adamw_primary_update(
        self,
        parameter: torch.Tensor,
        gradient: torch.Tensor,
        state: dict[str, Any],
        group: dict[str, Any],
    ) -> None:
        # ``parameter`` is an FP32 view of one primary slice.  State is keyed by the owning full PEFT
        # parameter but intentionally has only the slice shape, avoiding moments for the auxiliary rank.
        beta1, beta2 = group["betas"]
        if group.get("amsgrad", False) or group.get("maximize", False):
            raise ValueError("Bi-LoRA-Attn only supports ordinary non-AMSGrad AdamW descent.")

        exp_avg = state.get("bilora_exp_avg")
        exp_avg_sq = state.get("bilora_exp_avg_sq")
        if exp_avg is None:
            exp_avg = torch.zeros_like(parameter)
            exp_avg_sq = torch.zeros_like(parameter)
        else:
            exp_avg = exp_avg.to(device=parameter.device, dtype=parameter.dtype)
            exp_avg_sq = exp_avg_sq.to(device=parameter.device, dtype=parameter.dtype)

        step = int(state.get("bilora_step", 0)) + 1
        exp_avg.mul_(beta1).add_(gradient, alpha=1.0 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(gradient, gradient, value=1.0 - beta2)
        learning_rate = float(group["lr"])
        parameter.mul_(1.0 - learning_rate * float(group["weight_decay"]))
        step_size = learning_rate / (1.0 - beta1**step)
        denominator = exp_avg_sq.sqrt().div_((1.0 - beta2**step) ** 0.5).add_(float(group["eps"]))
        parameter.addcdiv_(exp_avg, denominator, value=-step_size)

        state["bilora_step"] = step
        # Preserve the base AdamW serialization invariant for torch 2.9 while keeping the
        # Bi-LoRA-specific integer counter authoritative for the sliced primary update.
        state["step"] = torch.tensor(float(step), dtype=torch.float32)
        state["bilora_exp_avg"] = exp_avg
        state["bilora_exp_avg_sq"] = exp_avg_sq

    @torch.no_grad()
    def _update_attention_pair(self, group: dict[str, Any]) -> None:
        lora_a, lora_b = group["params"]
        if lora_a.grad is None or lora_b.grad is None:
            raise RuntimeError(f"Bi-LoRA-Attn received a missing A/B gradient for {group['pair_name']}.")

        a = self._full_fp32(lora_a)
        b = self._full_fp32(lora_b)
        grad_a = self._full_fp32(lora_a.grad)
        grad_b = self._full_fp32(lora_b.grad)
        split = self.primary_rank

        # Primary minimizes the task loss; auxiliary maximizes the same accumulated loss.  Applying the
        # positive gradient directly is equivalent to the author's negate-then-SGD implementation.
        self._adamw_primary_update(a[:split], grad_a[:split], self.state[lora_a], group)
        self._adamw_primary_update(b[:, :split], grad_b[:, :split], self.state[lora_b], group)
        ascent_lr = float(group["lr"]) * self.auxiliary_lr_ratio
        a[split:].add_(grad_a[split:], alpha=ascent_lr)
        b[:, split:].add_(grad_b[:, split:], alpha=ascent_lr)
        copy_full_to_tensor(lora_a, a)
        copy_full_to_tensor(lora_b, b)

    @torch.no_grad()
    def _project_auxiliary_perturbation(self) -> None:
        auxiliary_factors: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []
        squared_norm = None
        for group in self.param_groups:
            if group.get("variant_role") != "attention_pair":
                continue
            lora_a, lora_b = group["params"]
            a = self._full_fp32(lora_a)
            b = self._full_fp32(lora_b)
            a2 = a[self.primary_rank :]
            b2 = b[:, self.primary_rank :]
            # ||B2 A2||_F^2 = sum((B2^T B2) * (A2 A2^T)); this avoids materializing an out x in matrix.
            module_squared_norm = ((b2.mT @ b2) * (a2 @ a2.mT)).sum().to(dtype=torch.float64)
            if squared_norm is None:
                squared_norm = module_squared_norm
            else:
                squared_norm = squared_norm + module_squared_norm.to(squared_norm.device)
            auxiliary_factors.append((lora_a, lora_b, a, b))

        if squared_norm is None:
            raise RuntimeError("Bi-LoRA-Attn found no attention pairs for global auxiliary projection.")
        perturbation_norm = squared_norm.clamp_min(0.0).sqrt()
        if not torch.isfinite(perturbation_norm):
            raise FloatingPointError("Bi-LoRA-Attn auxiliary perturbation norm is non-finite.")
        if perturbation_norm.item() <= self.rho:
            return

        # Scaling both factors by sqrt(rho / c) scales every B2 A2 product by rho / c.
        factor_scale = (self.rho / perturbation_norm).sqrt()
        for lora_a, lora_b, a, b in auxiliary_factors:
            a[self.primary_rank :].mul_(factor_scale.to(a.device))
            b[:, self.primary_rank :].mul_(factor_scale.to(b.device))
            copy_full_to_tensor(lora_a, a)
            copy_full_to_tensor(lora_b, b)

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            raise ValueError("Bi-LoRA-Attn does not support optimizer closures.")

        # As in AltLoRA, temporarily hide only attention A/B from inherited AdamW.  This preserves the
        # exact baseline optimizer path (including late KT runtime parameter injection) for everything else.
        attention_grads: list[tuple[torch.Tensor, torch.Tensor | None]] = []
        for group in self.param_groups:
            if group.get("variant_role") != "attention_pair":
                continue
            for parameter in group["params"]:
                attention_grads.append((parameter, parameter.grad))
                parameter.grad = None

        try:
            loss = super().step()
        finally:
            for parameter, gradient in attention_grads:
                parameter.grad = gradient
        found_inf = getattr(self, "found_inf", None)
        if isinstance(found_inf, torch.Tensor) and found_inf.item() != 0:
            return loss
        for group in self.param_groups:
            if group.get("variant_role") == "attention_pair":
                self._update_attention_pair(group)
        self._project_auxiliary_perturbation()
        self.variant_step += 1
        return loss

    def state_dict(self) -> dict[str, Any]:
        state_dict = super().state_dict()
        state_dict[self._STATE_KEY] = {
            "schema_version": 1,
            "variant_step": self.variant_step,
            "primary_rank": self.primary_rank,
            "auxiliary_rank": self.auxiliary_rank,
            "rho": self.rho,
            "auxiliary_lr_ratio": self.auxiliary_lr_ratio,
        }
        return state_dict

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        state_dict = dict(state_dict)
        metadata = state_dict.pop(self._STATE_KEY, None)
        if not isinstance(metadata, dict) or metadata.get("schema_version") != 1:
            raise ValueError("Optimizer checkpoint has no compatible Bi-LoRA-Attn metadata.")
        expected = (self.primary_rank, self.auxiliary_rank, self.rho, self.auxiliary_lr_ratio)
        actual = (
            int(metadata["primary_rank"]),
            int(metadata["auxiliary_rank"]),
            float(metadata["rho"]),
            float(metadata["auxiliary_lr_ratio"]),
        )
        if actual != expected:
            raise ValueError(f"Bi-LoRA-Attn optimizer checkpoint hyperparameters differ: {actual} != {expected}.")
        super().load_state_dict(state_dict)
        self.variant_step = int(metadata["variant_step"])


def register_bilora_eval_hooks(pairs: tuple[LoraPair, ...], primary_rank: int) -> None:
    r"""Zero auxiliary A channels only while each audited PEFT attention module is in eval mode."""
    # Masking A's last channels is algebraically equivalent to dropping B2 A2, but avoids replacing or
    # monkeypatching PEFT's Linear class.  Keep future inference-only branch rules local in the same way.
    for pair in pairs:
        existing_rank = getattr(pair.module, "_kt_bilora_primary_rank", None)
        if existing_rank is not None:
            if existing_rank != primary_rank:
                raise ValueError(f"Conflicting Bi-LoRA primary rank hook at {pair.name}.")
            continue

        adapter = getattr(pair.module, "active_adapter", "default")
        if isinstance(adapter, (list, tuple)):
            if len(adapter) != 1:
                raise ValueError(f"Bi-LoRA-Attn requires one active adapter at {pair.name}.")
            adapter = adapter[0]
        lora_a_module = pair.module.lora_A[adapter]
        total_rank = pair.lora_a.shape[0]
        parent_reference = weakref.ref(pair.module)

        def primary_only_hook(
            _module: nn.Module,
            _args: tuple[Any, ...],
            output: torch.Tensor,
            *,
            parent_reference=parent_reference,
            total_rank=total_rank,
        ) -> torch.Tensor:
            parent = parent_reference()
            if parent is None or parent.training:
                return output
            if output.shape[-1] != total_rank:
                raise RuntimeError("Bi-LoRA-Attn eval hook observed an unexpected LoRA A output rank.")
            return torch.cat((output[..., :primary_rank], torch.zeros_like(output[..., primary_rank:])), dim=-1)

        hook = lora_a_module.register_forward_hook(primary_only_hook)
        setattr(pair.module, "_kt_bilora_primary_rank", primary_rank)
        setattr(pair.module, "_kt_bilora_eval_hook", hook)


def _slice_primary_weights(
    state_dict: dict[str, torch.Tensor], primary_rank: int, expected_pairs: int
) -> dict[str, torch.Tensor]:
    sliced: dict[str, torch.Tensor] = {}
    found_a = found_b = 0
    for name, tensor in state_dict.items():
        if ".lora_A." in name or name.endswith(".lora_A.weight"):
            if tensor.ndim != 2 or tensor.shape[0] < primary_rank:
                raise ValueError(f"Cannot slice Bi-LoRA primary A tensor {name} with shape {tuple(tensor.shape)}.")
            sliced[name] = tensor[:primary_rank].contiguous()
            found_a += 1
        elif ".lora_B." in name or name.endswith(".lora_B.weight"):
            if tensor.ndim != 2 or tensor.shape[1] < primary_rank:
                raise ValueError(f"Cannot slice Bi-LoRA primary B tensor {name} with shape {tuple(tensor.shape)}.")
            sliced[name] = tensor[:, :primary_rank].contiguous()
            found_b += 1
        else:
            sliced[name] = tensor
    if found_a != expected_pairs or found_b != expected_pairs:
        raise ValueError(f"Bi-LoRA primary export found an invalid number of A/B tensors: A={found_a}, B={found_b}.")
    return sliced


def export_bilora_primary_adapter(adapter_dir: str | Path, variant_config: KTLoraVariantConfig) -> Path:
    r"""Write a deployable rank-r1 PEFT adapter while preserving non-LoRA modules such as LoRA Expert."""
    # Deployment is a separate artifact: slice only attention A/B tensors, rewrite PEFT rank/alpha so the
    # scale stays constant, and copy independent KT fused/LoRA Expert payloads without transformation.
    adapter_path = Path(adapter_dir)
    config_path = adapter_path / "adapter_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Bi-LoRA primary export cannot find {config_path}.")
    primary_rank = int(variant_config.method["primary_rank"])
    total_rank = int(variant_config.lora_rank)
    primary_alpha = round(variant_config.lora_alpha * primary_rank / total_rank)
    output_path = adapter_path / "adapter_primary"
    output_path.mkdir(parents=True, exist_ok=True)

    safe_path = adapter_path / "adapter_model.safetensors"
    binary_path = adapter_path / "adapter_model.bin"
    if safe_path.is_file():
        from safetensors.torch import load_file, save_file

        state_dict = load_file(safe_path, device="cpu")
        save_file(
            _slice_primary_weights(state_dict, primary_rank, len(variant_config.exact_target_names)),
            output_path / safe_path.name,
            metadata={"format": "pt"},
        )
    elif binary_path.is_file():
        state_dict = torch.load(binary_path, map_location="cpu", weights_only=True)
        torch.save(
            _slice_primary_weights(state_dict, primary_rank, len(variant_config.exact_target_names)),
            output_path / binary_path.name,
        )
    else:
        raise FileNotFoundError(f"Bi-LoRA primary export found no PEFT adapter weights in {adapter_path}.")

    with open(config_path, encoding="utf-8") as stream:
        adapter_config = json.load(stream)
    adapter_config["r"] = primary_rank
    adapter_config["lora_alpha"] = primary_alpha
    adapter_config["rank_pattern"] = {}
    adapter_config["alpha_pattern"] = {}
    temporary_config = output_path / f"adapter_config.json.tmp.{os.getpid()}"
    with open(temporary_config, "w", encoding="utf-8") as stream:
        json.dump(adapter_config, stream, indent=2, sort_keys=True)
        stream.write("\n")
    os.replace(temporary_config, output_path / "adapter_config.json")

    for filename in (
        "README.md",
        "fused_expert_lora.safetensors",
        "lora_expert_config.json",
        KT_LORA_ELIGIBLE_MANIFEST_NAME,
        KT_LORA_SELECTED_MANIFEST_NAME,
    ):
        source = adapter_path / filename
        if source.is_file():
            shutil.copy2(source, output_path / filename)
    primary_variant_config = variant_config.to_dict()
    primary_variant_config["lora_rank"] = primary_rank
    primary_variant_config["lora_alpha"] = primary_alpha
    primary_variant_config["method"] = {
        **primary_variant_config["method"],
        "deployment": "primary_only",
    }
    temporary_variant_config = output_path / f"{KT_LORA_VARIANT_CONFIG_NAME}.tmp.{os.getpid()}"
    with open(temporary_variant_config, "w", encoding="utf-8") as stream:
        json.dump(primary_variant_config, stream, indent=2, sort_keys=True)
        stream.write("\n")
    os.replace(temporary_variant_config, output_path / KT_LORA_VARIANT_CONFIG_NAME)
    return output_path


__all__ = ["BiLoraAttnOptimizer", "export_bilora_primary_adapter", "register_bilora_eval_hooks"]
