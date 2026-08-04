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

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

import torch
from torch import nn
from transformers.utils.hub import cached_file

from ...extras import logging


if TYPE_CHECKING:
    from os import PathLike


logger = logging.get_logger(__name__)

LORA_EXPERT_CONFIG_NAME = "lora_expert_config.json"
LORA_EXPERT_MODULE_NAME = "fast_le"
LORA_EXPERT_FORMULA = "mean_e(down_e(silu(gate_e(x)) * up_e(x)))"
LORA_EXPERT_SCHEMA_VERSION = 1
_LORA_EXPERT_CONFIG_ATTR = "_llamafactory_lora_expert_config"
_LORA_EXPERT_HOOK_ATTR = "_llamafactory_lora_expert_hook"

_NATIVE_MOE_CLASS_NAMES = {
    "DeepseekV2MoE",
    "DeepseekV3MoE",
    "Glm4MoeMoE",
    "MixtralSparseMoeBlock",
    "PhimoeSparseMoeBlock",
    "Qwen2MoeSparseMoeBlock",
    "Qwen3MoeSparseMoeBlock",
    "Qwen3_5MoeSparseMoeBlock",
}


@dataclass(frozen=True)
class LoRAExpertConfig:
    r"""Architecture metadata required to reconstruct LLaMA-Factory-owned LoRA Experts."""

    num_experts: int
    intermediate_size: int
    hidden_size: int
    target_modules: tuple[str, ...]
    schema_version: int = LORA_EXPERT_SCHEMA_VERSION
    module_name: str = LORA_EXPERT_MODULE_NAME
    formula: str = LORA_EXPERT_FORMULA
    initialization: str = "kaiming_uniform_gate_up_zero_down"
    implementation: str = "llamafactory"

    def __post_init__(self) -> None:
        if self.schema_version != LORA_EXPERT_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported LoRA Expert schema version {self.schema_version}; expected {LORA_EXPERT_SCHEMA_VERSION}."
            )
        if self.num_experts <= 0:
            raise ValueError("LoRA Expert num_experts must be greater than 0.")
        if self.intermediate_size <= 0:
            raise ValueError("LoRA Expert intermediate_size must be greater than 0.")
        if self.hidden_size <= 0:
            raise ValueError("LoRA Expert hidden_size must be greater than 0.")
        if not self.target_modules:
            raise ValueError("LoRA Expert target_modules cannot be empty.")
        if self.module_name != LORA_EXPERT_MODULE_NAME:
            raise ValueError(f"Unsupported LoRA Expert module name: {self.module_name}.")
        if self.formula != LORA_EXPERT_FORMULA:
            raise ValueError(f"Unsupported LoRA Expert formula: {self.formula}.")
        if self.implementation != "llamafactory":
            raise ValueError("LoRA Expert metadata is not owned by LLaMA-Factory.")

    def to_dict(self) -> dict[str, Any]:
        values = asdict(self)
        values["target_modules"] = list(self.target_modules)
        return values

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> LoRAExpertConfig:
        values = dict(values)
        values["target_modules"] = tuple(values["target_modules"])
        return cls(**values)


class LoRAExpertMLP(nn.Module):
    r"""One SwiGLU expert: down(silu(gate(x)) * up(x))."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.le_gate = nn.Linear(hidden_size, intermediate_size, bias=False, **factory_kwargs)
        self.le_up = nn.Linear(hidden_size, intermediate_size, bias=False, **factory_kwargs)
        self.le_down = nn.Linear(intermediate_size, hidden_size, bias=False, **factory_kwargs)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.le_gate.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.le_up.weight, a=math.sqrt(5))
        nn.init.zeros_(self.le_down.weight)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(self.le_gate.weight.dtype)
        output = self.le_down(torch.nn.functional.silu(self.le_gate(hidden_states)) * self.le_up(hidden_states))
        return output.to(input_dtype)


class LoRAExperts(nn.Module):
    r"""An un-routed mean of independent SwiGLU experts evaluated for every token."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if num_experts <= 0:
            raise ValueError("LoRA Expert num_experts must be greater than 0.")
        if intermediate_size <= 0:
            raise ValueError("LoRA Expert intermediate_size must be greater than 0.")

        self.experts = nn.ModuleList(
            [LoRAExpertMLP(hidden_size, intermediate_size, device=device, dtype=dtype) for _ in range(num_experts)]
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        output = sum(expert(hidden_states) for expert in self.experts)
        return output / len(self.experts)


def _get_text_hidden_size(model: nn.Module) -> int:
    config = getattr(model, "config", None)
    configs = []
    if config is not None and hasattr(config, "get_text_config"):
        try:
            configs.append(config.get_text_config())
        except TypeError:
            pass
    if config is not None and hasattr(config, "text_config"):
        configs.append(config.text_config)
    configs.append(config)

    for candidate in configs:
        hidden_size = getattr(candidate, "hidden_size", None)
        if isinstance(hidden_size, int) and hidden_size > 0:
            return hidden_size

    raise ValueError("Cannot determine hidden_size for LoRA Expert injection from the model config.")


def _is_native_moe_block(module: nn.Module) -> bool:
    class_name = module.__class__.__name__
    return class_name in _NATIVE_MOE_CLASS_NAMES or class_name.endswith("SparseMoeBlock")


def _get_kt_wrappers(model: nn.Module) -> list[nn.Module]:
    for candidate in model.modules():
        wrappers = candidate.__dict__.get("_kt_wrappers")
        if wrappers:
            return list(wrappers)
    return []


def _resolve_targets(
    model: nn.Module,
    use_kt: bool,
    target_names: tuple[str, ...] | None,
) -> list[tuple[str, nn.Module]]:
    name_by_id = {id(module): name for name, module in model.named_modules()}
    if target_names is not None:
        targets = [(name, model.get_submodule(name)) for name in target_names]
    elif use_kt:
        wrappers = _get_kt_wrappers(model)
        if not wrappers:
            raise ValueError("KTransformers is enabled but no KT MoE wrappers were found for LoRA Expert injection.")
        missing = [wrapper for wrapper in wrappers if id(wrapper) not in name_by_id]
        if missing:
            raise ValueError("A KT MoE wrapper is not registered in the model module tree.")
        targets = [(name_by_id[id(wrapper)], wrapper) for wrapper in wrappers]
    else:
        targets = [(name, module) for name, module in model.named_modules() if _is_native_moe_block(module)]

    if not targets:
        raise ValueError("No supported MoE blocks were found for LoRA Expert injection.")

    if use_kt:
        wrapper_ids = {id(wrapper) for wrapper in _get_kt_wrappers(model)}
        if wrapper_ids and {id(module) for _, module in targets} != wrapper_ids:
            raise ValueError("LoRA Expert metadata target modules do not match the active KT MoE wrappers.")

    return targets


def _infer_device_dtype(target: nn.Module, model: nn.Module) -> tuple[torch.device, torch.dtype]:
    fallback = None
    for owner in (target, model):
        for parameter in owner.parameters():
            if not parameter.is_floating_point() or parameter.device.type == "meta":
                continue
            if parameter.device.type != "cpu":
                return parameter.device, parameter.dtype
            if fallback is None:
                fallback = (parameter.device, parameter.dtype)

    return fallback or (torch.device("cpu"), torch.get_default_dtype())


def _lora_expert_forward_hook(
    module: nn.Module,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    output: Any,
) -> Any:
    if args:
        hidden_states = args[0]
    elif "hidden_states" in kwargs:
        hidden_states = kwargs["hidden_states"]
    else:
        raise ValueError("LoRA Expert MoE hook did not receive hidden_states.")

    residual = getattr(module, LORA_EXPERT_MODULE_NAME)(hidden_states)
    if isinstance(output, tuple):
        return (output[0] + residual, *output[1:])
    if isinstance(output, list):
        return [output[0] + residual, *output[1:]]
    if not isinstance(output, torch.Tensor):
        raise TypeError(f"Unsupported MoE output type for LoRA Expert: {type(output).__name__}.")
    return output + residual


def get_lora_expert_config(model: nn.Module | None) -> LoRAExpertConfig | None:
    if model is None:
        return None

    queue = [model]
    visited = set()
    while queue:
        candidate = queue.pop()
        if id(candidate) in visited:
            continue
        visited.add(id(candidate))
        config = candidate.__dict__.get(_LORA_EXPERT_CONFIG_ATTR)
        if isinstance(config, LoRAExpertConfig):
            return config
        for attribute in ("module", "base_model", "model", "pretrained_model"):
            child = getattr(candidate, attribute, None)
            if isinstance(child, nn.Module):
                queue.append(child)

    return None


def set_lora_expert_config(model: nn.Module, config: LoRAExpertConfig) -> None:
    setattr(model, _LORA_EXPERT_CONFIG_ATTR, config)


def attach_lora_experts(
    model: nn.Module,
    *,
    num_experts: int | None = None,
    intermediate_size: int | None = None,
    use_kt: bool,
    config: LoRAExpertConfig | None = None,
) -> LoRAExpertConfig:
    existing_config = get_lora_expert_config(model)
    if existing_config is not None:
        if config is not None and config != existing_config:
            raise ValueError("A different LoRA Expert configuration is already attached to the model.")
        return existing_config

    if config is not None:
        if num_experts is not None and num_experts != config.num_experts:
            raise ValueError("Requested LoRA Expert num_experts does not match checkpoint metadata.")
        if intermediate_size is not None and intermediate_size != config.intermediate_size:
            raise ValueError("Requested LoRA Expert intermediate_size does not match checkpoint metadata.")
        num_experts = config.num_experts
        intermediate_size = config.intermediate_size
        hidden_size = config.hidden_size
        model_hidden_size = _get_text_hidden_size(model)
        if hidden_size != model_hidden_size:
            raise ValueError(
                f"LoRA Expert checkpoint hidden_size {hidden_size} does not match model hidden_size {model_hidden_size}."
            )
        target_names = config.target_modules
    else:
        if num_experts is None or intermediate_size is None:
            raise ValueError("num_experts and intermediate_size are required for a new LoRA Expert.")
        hidden_size = _get_text_hidden_size(model)
        target_names = None

    targets = _resolve_targets(model, use_kt, target_names)
    resolved_names = tuple(name for name, _ in targets)
    if config is not None and resolved_names != config.target_modules:
        raise ValueError("LoRA Expert target module order does not match checkpoint metadata.")

    resolved_config = config or LoRAExpertConfig(
        num_experts=num_experts,
        intermediate_size=intermediate_size,
        hidden_size=hidden_size,
        target_modules=resolved_names,
    )
    for name, module in targets:
        if getattr(module, "lora_experts", None) is not None:
            raise ValueError(
                f"Legacy KT-owned LoRA Experts are active at {name}. Disable kt_use_lora_experts and use "
                "use_lora_expert instead."
            )
        if hasattr(module, LORA_EXPERT_MODULE_NAME):
            raise ValueError(f"LoRA Expert module is already attached at {name}.")

        device, dtype = _infer_device_dtype(module, model)
        expert_module = LoRAExperts(
            hidden_size,
            intermediate_size,
            num_experts,
            device=device,
            dtype=dtype,
        )
        setattr(module, LORA_EXPERT_MODULE_NAME, expert_module)
        hook = module.register_forward_hook(_lora_expert_forward_hook, with_kwargs=True)
        setattr(module, _LORA_EXPERT_HOOK_ATTR, hook)

    set_lora_expert_config(model, resolved_config)
    parameter_count = 3 * len(targets) * num_experts * hidden_size * intermediate_size
    logger.info_rank0(
        f"Attached LLaMA-Factory LoRA Experts to {len(targets)} MoE blocks "
        f"(E={num_experts}, width={intermediate_size}, trainable parameters={parameter_count:,})."
    )
    return resolved_config


def save_lora_expert_config(config: LoRAExpertConfig, output_dir: str | PathLike[str]) -> None:
    output_dir = os.fspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    config_path = os.path.join(output_dir, LORA_EXPERT_CONFIG_NAME)
    temporary_path = f"{config_path}.tmp.{os.getpid()}"
    with open(temporary_path, "w", encoding="utf-8") as config_file:
        json.dump(config.to_dict(), config_file, indent=2, sort_keys=True)
        config_file.write("\n")
    os.replace(temporary_path, config_path)


def load_lora_expert_config(
    adapter_path: str | PathLike[str],
    *,
    subfolder: str | None = None,
    cache_dir: str | None = None,
    revision: str | None = None,
    token: str | bool | None = None,
) -> LoRAExpertConfig | None:
    adapter_path = os.fspath(adapter_path)
    if os.path.isfile(adapter_path):
        config_path = adapter_path if os.path.basename(adapter_path) == LORA_EXPERT_CONFIG_NAME else None
    elif os.path.isdir(adapter_path):
        config_path = os.path.join(adapter_path, subfolder or "", LORA_EXPERT_CONFIG_NAME)
        if not os.path.isfile(config_path):
            config_path = None
    else:
        config_path = cached_file(
            adapter_path,
            LORA_EXPERT_CONFIG_NAME,
            subfolder=subfolder or "",
            cache_dir=cache_dir,
            revision=revision,
            token=token,
            _raise_exceptions_for_gated_repo=False,
            _raise_exceptions_for_missing_entries=False,
            _raise_exceptions_for_connection_errors=False,
        )

    if config_path is None:
        return None

    with open(config_path, encoding="utf-8") as config_file:
        values = json.load(config_file)
    return LoRAExpertConfig.from_dict(values)
