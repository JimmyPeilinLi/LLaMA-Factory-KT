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

from pathlib import Path

import torch
from typing_extensions import override

from ...model.lora_variants.altlora_attn import AltLoraAttnOptimizer
from ...model.lora_variants.bilora_attn import BiLoraAttnOptimizer, export_bilora_primary_adapter
from ...model.lora_variants.common import (
    get_variant_config,
    save_variant_artifacts,
    validate_runtime_attention_placement,
)
from .trainer import CustomSeq2SeqTrainer


class KTLoraVariantTrainer(CustomSeq2SeqTrainer):
    r"""Thin SFT trainer integration that preserves the current HF/transformers-kt training loop."""

    def _variant_optimizer(self) -> torch.optim.Optimizer | None:
        optimizer = self.optimizer
        visited = set()
        while optimizer is not None and id(optimizer) not in visited:
            visited.add(id(optimizer))
            if isinstance(optimizer, (AltLoraAttnOptimizer, BiLoraAttnOptimizer)):
                return optimizer
            optimizer = getattr(optimizer, "optimizer", None)
        return None

    def _discard_altlora_inactive_gradients(self) -> None:
        optimizer = self._variant_optimizer()
        if isinstance(optimizer, AltLoraAttnOptimizer):
            optimizer.discard_inactive_gradients()

    def _verify_runtime_attention_contract(self, model: torch.nn.Module) -> None:
        if getattr(model, "_kt_lora_runtime_contract_verified", False):
            return
        eligible = getattr(model, "_kt_lora_eligible_manifest", None)
        selected = getattr(model, "_kt_lora_selected_manifest", None)
        if eligible is None or selected is None:
            raise RuntimeError("KT LoRA variant manifests are unavailable after accelerator preparation.")

        runtime_pairs = validate_runtime_attention_placement(model, eligible, selected)
        variant_optimizer = self._variant_optimizer()
        if variant_optimizer is not None:
            pair_groups = {
                group.get("pair_name"): group
                for group in variant_optimizer.param_groups
                if group.get("variant_role") == "attention_pair"
            }
            if set(pair_groups) != {pair.name for pair in runtime_pairs}:
                raise RuntimeError("KT LoRA variant optimizer attention groups differ from the runtime target pairs.")
            for pair in runtime_pairs:
                parameters = pair_groups[pair.name]["params"]
                if len(parameters) != 2 or parameters[0] is not pair.lora_a or parameters[1] is not pair.lora_b:
                    raise RuntimeError(
                        f"KT LoRA variant optimizer owns stale FSDP parameters for {pair.name}."
                    )
        else:
            optimizer = self.optimizer
            visited = set()
            while getattr(optimizer, "optimizer", None) is not None and id(optimizer) not in visited:
                visited.add(id(optimizer))
                optimizer = optimizer.optimizer
            optimizer_parameter_ids = {
                id(parameter) for group in optimizer.param_groups for parameter in group["params"]
            }
            missing = [
                pair.name
                for pair in runtime_pairs
                if id(pair.lora_a) not in optimizer_parameter_ids or id(pair.lora_b) not in optimizer_parameter_ids
            ]
            if missing:
                raise RuntimeError(f"PLoP-Attn optimizer does not own runtime PEFT parameters: {missing[:5]}.")

        # Replace the pre-FSDP references so later eval hooks and diagnostics observe the live DTensors.
        setattr(model, "_kt_lora_variant_pairs", runtime_pairs)
        setattr(model, "_kt_lora_runtime_contract_verified", True)

    @override
    def training_step(self, model, inputs, *args, **kwargs):
        self._verify_runtime_attention_contract(model)
        return super().training_step(model, inputs, *args, **kwargs)

    @override
    def _clip_grad_norm(self, model):
        self._discard_altlora_inactive_gradients()
        return super()._clip_grad_norm(model)

    @override
    def _get_grad_norm(self, model, grad_norm=None):
        self._discard_altlora_inactive_gradients()
        return super()._get_grad_norm(model, grad_norm=grad_norm)

    @override
    def create_optimizer(self, *args, **kwargs) -> torch.optim.Optimizer:
        optimizer = super().create_optimizer(*args, **kwargs)
        variant = self.finetuning_args.kt_lora_variant
        if variant in ["vanilla", "plop_attn"]:
            return optimizer
        if self._variant_optimizer() is not None:
            return optimizer
        optimizer_name = getattr(self.args.optim, "value", str(self.args.optim))
        if optimizer_name != "adamw_torch":
            raise ValueError(f"KT LoRA variants require `optim: adamw_torch`, got {self.args.optim!r}.")

        pairs = getattr(self.model, "_kt_lora_variant_pairs", None)
        if not isinstance(pairs, tuple) or not pairs:
            raise RuntimeError("KT LoRA variant PEFT pairs are unavailable when the optimizer is created.")
        if variant == "altlora_attn":
            optimizer = AltLoraAttnOptimizer(
                optimizer,
                pairs,
                regularizer=self.finetuning_args.altlora_reg,
                beta1=self.finetuning_args.altlora_beta1,
                switch_every=self.finetuning_args.altlora_switch_every_optimizer_steps,
            )
        elif variant == "bilora_attn":
            optimizer = BiLoraAttnOptimizer(
                optimizer,
                pairs,
                primary_rank=self.finetuning_args.bilora_primary_rank,
                auxiliary_rank=self.finetuning_args.bilora_aux_rank,
                rho=self.finetuning_args.bilora_rho,
                auxiliary_lr_ratio=self.finetuning_args.bilora_aux_lr_ratio,
            )
        else:
            raise ValueError(f"Unsupported KT LoRA optimizer variant: {variant!r}.")
        self.optimizer = optimizer
        return optimizer

    @override
    def save_model(self, output_dir: str | None = None, *args, **kwargs) -> None:
        super().save_model(output_dir, *args, **kwargs)
        if not self.args.should_save:
            return
        output_path = Path(output_dir or self.args.output_dir)
        model = self.accelerator.unwrap_model(self.model)
        config = get_variant_config(model)
        if config is None:
            return
        save_variant_artifacts(model, output_path)
        if config.variant == "bilora_attn":
            export_bilora_primary_adapter(output_path, config)


__all__ = ["KTLoraVariantTrainer"]
