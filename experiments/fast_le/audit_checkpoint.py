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

"""Audit a fast_le LoRA checkpoint without loading the base model."""

import argparse
import json
import math
import re
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open


REQUIRED_CHECKPOINT_FILES = (
    "adapter_config.json",
    "adapter_model.safetensors",
    "fused_expert_lora.safetensors",
    "lora_expert_config.json",
    "optimizer.pt",
    "pytorch_model_fsdp.bin",
    "rng_state.pth",
    "scheduler.pt",
    "trainer_state.json",
    "training_args.bin",
)
FUSED_SUFFIXES = {
    "down_lora_a",
    "down_lora_b",
    "gate_lora_a",
    "gate_lora_b",
    "up_lora_a",
    "up_lora_b",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint_dir", type=Path)
    parser.add_argument("--expected-layers", type=int, required=True)
    parser.add_argument("--expected-steps", type=int, required=True)
    parser.add_argument("--expected-rank", type=int)
    parser.add_argument("--expected-alpha", type=float)
    parser.add_argument("--expected-dropout", type=float)
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def _is_positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _matches_number(value: Any, expected: float) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isclose(float(value), expected)


def _empty_group() -> dict[str, Any]:
    return {
        "all_finite": True,
        "dtypes": defaultdict(int),
        "max_abs": 0.0,
        "nonzero_elements": 0,
        "nonzero_tensor_count": 0,
        "numel": 0,
        "shapes": defaultdict(int),
        "tensor_count": 0,
    }


def _adapter_group(key: str) -> str:
    if ".fast_le." in key:
        if ".le_down." in key:
            return "le_down"
        if ".le_gate." in key:
            return "le_gate"
        if ".le_up." in key:
            return "le_up"
        return "le_other"
    if ".lora_A." in key:
        return "attention_lora_a"
    if ".lora_B." in key:
        return "attention_lora_b"
    return "other"


def _fused_group(key: str) -> str:
    if key.endswith("_lora_a"):
        return "fused_lora_a"
    if key.endswith("_lora_b"):
        return "fused_lora_b"
    return "other"


def _inspect_tensors(path: Path, classifier: Callable[[str], str]) -> tuple[dict[str, Any], set[str]]:
    groups: defaultdict[str, dict[str, Any]] = defaultdict(_empty_group)
    keys: set[str] = set()
    with safe_open(path, framework="pt", device="cpu") as tensors:
        for key in tensors.keys():
            tensor = tensors.get_tensor(key)
            keys.add(key)
            group = groups[classifier(key)]
            finite = bool(torch.isfinite(tensor).all().item())
            nonzero_elements = int(torch.count_nonzero(tensor).item())
            shape = "x".join(str(size) for size in tensor.shape) or "scalar"
            dtype = str(tensor.dtype).removeprefix("torch.")

            group["all_finite"] = group["all_finite"] and finite
            group["dtypes"][dtype] += 1
            group["nonzero_elements"] += nonzero_elements
            group["nonzero_tensor_count"] += int(nonzero_elements > 0)
            group["numel"] += tensor.numel()
            group["shapes"][shape] += 1
            group["tensor_count"] += 1
            if tensor.numel():
                group["max_abs"] = max(group["max_abs"], float(tensor.abs().max().item()))

    normalized_groups = {}
    for name, group in sorted(groups.items()):
        group["dtypes"] = dict(sorted(group["dtypes"].items()))
        group["shapes"] = dict(sorted(group["shapes"].items()))
        normalized_groups[name] = group

    return {
        "bytes": path.stat().st_size,
        "groups": normalized_groups,
        "tensor_count": len(keys),
    }, keys


def _main() -> int:
    args = _parse_args()
    checkpoint_dir = args.checkpoint_dir.resolve()
    errors: list[str] = []

    def check(condition: bool, message: str) -> None:
        if not condition:
            errors.append(message)

    file_sizes = {}
    for name in REQUIRED_CHECKPOINT_FILES:
        path = checkpoint_dir / name
        check(path.is_file(), f"missing required checkpoint file: {name}")
        if path.is_file():
            file_sizes[name] = path.stat().st_size

    if errors:
        result = {
            "checkpoint_dir": str(checkpoint_dir),
            "errors": errors,
            "files": file_sizes,
            "status": "fail",
        }
    else:
        adapter_config = _load_json(checkpoint_dir / "adapter_config.json")
        le_config = _load_json(checkpoint_dir / "lora_expert_config.json")
        trainer_state = _load_json(checkpoint_dir / "trainer_state.json")
        adapter_stats, adapter_keys = _inspect_tensors(checkpoint_dir / "adapter_model.safetensors", _adapter_group)
        fused_stats, fused_keys = _inspect_tensors(checkpoint_dir / "fused_expert_lora.safetensors", _fused_group)

        check(adapter_config.get("peft_type") == "LORA", "PEFT adapter type is not LORA")
        check("fast_le" in (adapter_config.get("modules_to_save") or []), "fast_le is absent from modules_to_save")
        check(adapter_config.get("target_modules"), "standard PEFT target_modules is empty")
        if args.expected_rank is not None:
            check(adapter_config.get("r") == args.expected_rank, "PEFT rank changed")
        if args.expected_alpha is not None:
            check(_matches_number(adapter_config.get("lora_alpha"), args.expected_alpha), "PEFT alpha changed")
        if args.expected_dropout is not None:
            check(_matches_number(adapter_config.get("lora_dropout"), args.expected_dropout), "PEFT dropout changed")

        targets = le_config.get("target_modules") or []
        num_experts = le_config.get("num_experts")
        hidden_size = le_config.get("hidden_size")
        intermediate_size = le_config.get("intermediate_size")
        expected_targets = [f"model.language_model.layers.{index}.mlp" for index in range(args.expected_layers)]
        check(le_config.get("implementation") == "llamafactory", "LE implementation is not LLaMA-Factory")
        check(le_config.get("module_name") == "fast_le", "LE module name is not fast_le")
        check(
            le_config.get("formula") == "mean_e(down_e(silu(gate_e(x)) * up_e(x)))",
            "LE formula metadata changed",
        )
        check(targets == expected_targets, "LE target path list is not the exact expected layer list")
        valid_num_experts = _is_positive_int(num_experts)
        valid_hidden_size = _is_positive_int(hidden_size)
        valid_intermediate_size = _is_positive_int(intermediate_size)
        check(valid_num_experts, "invalid LE expert count")
        check(valid_hidden_size, "invalid LE hidden size")
        check(valid_intermediate_size, "invalid LE intermediate size")

        adapter_groups = adapter_stats["groups"]
        check("le_other" not in adapter_groups, "unknown fast_le tensor keys detected")
        check("other" not in adapter_groups, "unexpected non-LoRA adapter tensor keys detected")
        if valid_num_experts:
            expected_le_tensors = args.expected_layers * num_experts
            for group_name in ("le_down", "le_gate", "le_up"):
                group = adapter_groups.get(group_name, {})
                check(group.get("tensor_count") == expected_le_tensors, f"wrong {group_name} tensor count")
                check(group.get("all_finite") is True, f"{group_name} contains non-finite values")

        if valid_num_experts and valid_hidden_size and valid_intermediate_size:
            expected_shapes = {
                "le_down": f"{hidden_size}x{intermediate_size}",
                "le_gate": f"{intermediate_size}x{hidden_size}",
                "le_up": f"{intermediate_size}x{hidden_size}",
            }
            for group_name, shape in expected_shapes.items():
                check(
                    adapter_groups.get(group_name, {}).get("shapes") == {shape: expected_le_tensors},
                    f"wrong {group_name} shapes",
                )

        attention_a = {key for key in adapter_keys if ".lora_A." in key}
        attention_b = {key for key in adapter_keys if ".lora_B." in key}
        expected_attention_b = {key.replace(".lora_A.", ".lora_B.") for key in attention_a}
        check(attention_a and attention_b == expected_attention_b, "attention LoRA A/B keys are not paired")
        check(adapter_groups.get("attention_lora_a", {}).get("all_finite") is True, "LoRA A is non-finite")
        check(adapter_groups.get("attention_lora_b", {}).get("all_finite") is True, "LoRA B is non-finite")

        le_down = {key for key in adapter_keys if ".le_down." in key}
        expected_le_gate = {key.replace(".le_down.", ".le_gate.") for key in le_down}
        expected_le_up = {key.replace(".le_down.", ".le_up.") for key in le_down}
        check(expected_le_gate <= adapter_keys, "LE gate keys are not paired with every down key")
        check(expected_le_up <= adapter_keys, "LE up keys are not paired with every down key")
        check(not any("lora_expert" in key for key in adapter_keys), "legacy LE keys leaked into adapter")
        check(not any(".modules_to_save." in key for key in adapter_keys), "duplicate PEFT wrapper keys detected")

        fused_layer_suffixes: defaultdict[int, set[str]] = defaultdict(set)
        for key in fused_keys:
            match = re.fullmatch(r"layers\.(\d+)\.experts\.(.+)", key)
            check(match is not None, f"unexpected fused expert key: {key}")
            if match is not None:
                fused_layer_suffixes[int(match.group(1))].add(match.group(2))

        check(set(fused_layer_suffixes) == set(range(args.expected_layers)), "fused LoRA layer set is incomplete")
        check(
            all(suffixes == FUSED_SUFFIXES for suffixes in fused_layer_suffixes.values()),
            "fused LoRA A/B key set is incomplete",
        )
        check(fused_stats["tensor_count"] == args.expected_layers * 6, "wrong fused LoRA tensor count")
        check("other" not in fused_stats["groups"], "unexpected fused expert tensor keys detected")
        check(fused_stats["groups"].get("fused_lora_a", {}).get("all_finite") is True, "fused LoRA A is non-finite")
        check(fused_stats["groups"].get("fused_lora_b", {}).get("all_finite") is True, "fused LoRA B is non-finite")

        check(
            adapter_groups.get("attention_lora_b", {}).get("nonzero_tensor_count", 0) > 0,
            "attention LoRA B remained entirely zero after training",
        )
        check(
            adapter_groups.get("le_down", {}).get("nonzero_tensor_count", 0) > 0,
            "zero-initialized LE down projections did not update",
        )
        check(
            fused_stats["groups"].get("fused_lora_b", {}).get("nonzero_tensor_count", 0) > 0,
            "zero-initialized fused MoE LoRA B did not update",
        )

        check(trainer_state.get("global_step") == args.expected_steps, "trainer global_step is incorrect")
        metric_names = {"eval_loss", "grad_norm", "loss"}
        metric_values = [
            value
            for row in trainer_state.get("log_history", [])
            for name, value in row.items()
            if name in metric_names
        ]
        check(
            metric_values
            and all(
                isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
                for value in metric_values
            ),
            "training metrics are non-finite",
        )
        train_losses = [
            row["loss"] for row in trainer_state.get("log_history", []) if "loss" in row and "eval_loss" not in row
        ]
        eval_losses = [row["eval_loss"] for row in trainer_state.get("log_history", []) if "eval_loss" in row]
        check(len(train_losses) == args.expected_steps, "not every optimizer step has a logged loss")
        check(bool(eval_losses), "checkpoint trainer state has no eval loss")

        result = {
            "adapter": adapter_stats,
            "adapter_config": {
                "lora_alpha": adapter_config.get("lora_alpha"),
                "lora_dropout": adapter_config.get("lora_dropout"),
                "modules_to_save": adapter_config.get("modules_to_save"),
                "r": adapter_config.get("r"),
                "target_modules": sorted(adapter_config.get("target_modules") or []),
            },
            "checkpoint_dir": str(checkpoint_dir),
            "errors": errors,
            "files": file_sizes,
            "fused_expert_lora": fused_stats,
            "lora_expert_config": le_config,
            "status": "pass" if not errors else "fail",
            "trainer": {
                "eval_losses": eval_losses,
                "global_step": trainer_state.get("global_step"),
                "train_losses": train_losses,
            },
        }

    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(rendered + "\n", encoding="utf-8")
    return int(bool(errors))


if __name__ == "__main__":
    raise SystemExit(_main())
