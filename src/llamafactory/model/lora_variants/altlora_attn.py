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

from typing import Any

import torch

from .common import LoraPair, copy_full_to_tensor, rebuild_adamw_param_groups, tensor_to_full


class AltLoraAttnOptimizer(torch.optim.AdamW):
    r"""Ordinary AltLoRA on audited GPU attention pairs, with a B-first alternating schedule."""

    _STATE_KEY = "kt_altlora_attn"

    def __init__(
        self,
        baseline_optimizer: torch.optim.Optimizer,
        pairs: tuple[LoraPair, ...],
        *,
        regularizer: float,
        beta1: float,
        switch_every: int,
    ) -> None:
        if baseline_optimizer.state:
            raise ValueError("AltLoRA-Attn must replace AdamW before the baseline optimizer has taken a step.")
        param_groups = rebuild_adamw_param_groups(baseline_optimizer, pairs)
        super().__init__(param_groups, **baseline_optimizer.defaults)
        self.regularizer = float(regularizer)
        self.beta1 = float(beta1)
        self.switch_every = int(switch_every)
        self.variant_step = 0

    def _active_factor(self) -> str:
        phase = (self.variant_step // self.switch_every) % 2
        return "B" if phase == 0 else "A"

    def discard_inactive_gradients(self) -> None:
        active_index = 1 if self._active_factor() == "B" else 0
        for group in self.param_groups:
            if group.get("variant_role") == "attention_pair":
                group["params"][1 - active_index].grad = None

    @staticmethod
    def _full_fp32(tensor: torch.Tensor) -> torch.Tensor:
        return tensor_to_full(tensor).detach().to(dtype=torch.float32)

    def _solve(self, gram: torch.Tensor, right_hand_side: torch.Tensor, pair_name: str, factor: str) -> torch.Tensor:
        try:
            solution = torch.linalg.solve(gram, right_hand_side)
        except RuntimeError as error:
            condition = torch.linalg.cond(gram).item()
            raise RuntimeError(
                f"AltLoRA-Attn rank-space solve failed for {pair_name}.{factor}; condition={condition:.6e}."
            ) from error
        if not torch.isfinite(solution).all():
            condition = torch.linalg.cond(gram).item()
            raise FloatingPointError(
                f"AltLoRA-Attn rank-space solve was non-finite for {pair_name}.{factor}; condition={condition:.6e}."
            )
        return solution

    def _right_solve(
        self, right_hand_side: torch.Tensor, gram: torch.Tensor, pair_name: str, factor: str
    ) -> torch.Tensor:
        return self._solve(gram.mT, right_hand_side.mT, pair_name, factor).mT

    @torch.no_grad()
    def _update_attention_pair(self, group: dict[str, Any], active_factor: str) -> None:
        lora_a, lora_b = group["params"]
        a = self._full_fp32(lora_a)
        b = self._full_fp32(lora_b)
        scaling = float(group["lora_scaling"])
        rank = a.shape[0]
        identity = torch.eye(rank, dtype=torch.float32, device=a.device)

        if active_factor == "A":
            if lora_a.grad is None:
                raise RuntimeError(f"AltLoRA-Attn received a missing A gradient for {group['pair_name']}.")
            grad_a = self._full_fp32(lora_a.grad)
            parameter = lora_a
            parameter_full = a
            opposite = b
            gram = b.mT @ b + self.regularizer * identity
            raw_direction = self._solve(gram, grad_a, group["pair_name"], "A") / (scaling**2)
            state = self.state[parameter]
            momentum = state.get("alt_momentum")
            previous_opposite = state.get("alt_opposite_basis")
            if momentum is None or previous_opposite is None:
                aligned_momentum = torch.zeros_like(raw_direction)
            elif torch.equal(previous_opposite.to(device=b.device, dtype=b.dtype), b):
                aligned_momentum = momentum.to(device=b.device, dtype=b.dtype)
            else:
                aligned_momentum = self._solve(
                    gram,
                    b.mT @ previous_opposite.to(device=b.device, dtype=b.dtype) @ momentum.to(b.device),
                    group["pair_name"],
                    "A-momentum",
                )
        else:
            if lora_b.grad is None:
                raise RuntimeError(f"AltLoRA-Attn received a missing B gradient for {group['pair_name']}.")
            grad_b = self._full_fp32(lora_b.grad)
            parameter = lora_b
            parameter_full = b
            opposite = a
            gram = a @ a.mT + self.regularizer * identity
            raw_direction = self._right_solve(grad_b, gram, group["pair_name"], "B") / (scaling**2)
            state = self.state[parameter]
            momentum = state.get("alt_momentum")
            previous_opposite = state.get("alt_opposite_basis")
            if momentum is None or previous_opposite is None:
                aligned_momentum = torch.zeros_like(raw_direction)
            elif torch.equal(previous_opposite.to(device=a.device, dtype=a.dtype), a):
                aligned_momentum = momentum.to(device=a.device, dtype=a.dtype)
            else:
                aligned_momentum = self._right_solve(
                    momentum.to(a.device) @ previous_opposite.to(device=a.device, dtype=a.dtype) @ a.mT,
                    gram,
                    group["pair_name"],
                    "B-momentum",
                )

        updated_momentum = self.beta1 * aligned_momentum + (1.0 - self.beta1) * raw_direction
        learning_rate = float(group["lr"])
        weight_decay = float(group["weight_decay"])
        parameter_full.add_(updated_momentum + weight_decay * parameter_full, alpha=-learning_rate)
        copy_full_to_tensor(parameter, parameter_full)
        state["alt_momentum"] = updated_momentum
        state["alt_opposite_basis"] = opposite.clone()

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            raise ValueError("AltLoRA-Attn does not support optimizer closures.")

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

        active_factor = self._active_factor()
        for group in self.param_groups:
            if group.get("variant_role") == "attention_pair":
                self._update_attention_pair(group, active_factor)
        self.variant_step += 1
        return loss

    def state_dict(self) -> dict[str, Any]:
        state_dict = super().state_dict()
        state_dict[self._STATE_KEY] = {
            "schema_version": 1,
            "variant_step": self.variant_step,
            "regularizer": self.regularizer,
            "beta1": self.beta1,
            "switch_every": self.switch_every,
            "first_factor": "B",
        }
        return state_dict

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        state_dict = dict(state_dict)
        metadata = state_dict.pop(self._STATE_KEY, None)
        if not isinstance(metadata, dict) or metadata.get("schema_version") != 1:
            raise ValueError("Optimizer checkpoint has no compatible AltLoRA-Attn metadata.")
        expected = (self.regularizer, self.beta1, self.switch_every, "B")
        actual = (
            float(metadata["regularizer"]),
            float(metadata["beta1"]),
            int(metadata["switch_every"]),
            metadata["first_factor"],
        )
        if actual != expected:
            raise ValueError(f"AltLoRA-Attn optimizer checkpoint hyperparameters differ: {actual} != {expected}.")
        super().load_state_dict(state_dict)
        self.variant_step = int(metadata["variant_step"])


__all__ = ["AltLoraAttnOptimizer"]
