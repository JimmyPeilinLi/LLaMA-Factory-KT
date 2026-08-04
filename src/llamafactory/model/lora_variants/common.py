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
import importlib.util
import inspect
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import torch
from torch import nn
from transformers.utils.hub import cached_file


KT_LORA_VARIANT_CONFIG_NAME = "kt_lora_variant_config.json"
KT_LORA_ELIGIBLE_MANIFEST_NAME = "kt_lora_eligible_targets.json"
KT_LORA_SELECTED_MANIFEST_NAME = "kt_lora_selected_targets.json"
KT_LORA_VARIANT_SCHEMA_VERSION = 1

ATTENTION_FAMILIES = (
    "in_proj_a",
    "in_proj_b",
    "in_proj_qkv",
    "in_proj_z",
    "out_proj",
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
)
LINEAR_ATTENTION_FAMILIES = frozenset(ATTENTION_FAMILIES[:5])
FULL_ATTENTION_FAMILIES = frozenset(ATTENTION_FAMILIES[5:])
FORBIDDEN_PATH_PARTS = (
    ".lora_experts.",
    ".mlp.",
    ".experts.",
    ".shared_experts.",
    ".shared_expert.",
    ".router.",
    ".generate_linear.",
    ".prefill_linear.",
)
ACCELERATOR_DEVICE_TYPES = frozenset(("cuda", "xpu", "npu"))


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def python_package_source_sha256(package_name: str) -> str:
    specification = importlib.util.find_spec(package_name)
    if specification is None:
        raise ImportError(f"Cannot locate the {package_name} package for source hashing.")
    if specification.submodule_search_locations:
        package_roots = [Path(location) for location in specification.submodule_search_locations]
    elif specification.origin is not None:
        package_roots = [Path(specification.origin).parent]
    else:
        raise ImportError(f"Cannot locate the {package_name} package source tree.")

    paths = sorted(
        (path, root)
        for root in package_roots
        for path in root.rglob("*.py")
        if path.is_file() and "__pycache__" not in path.parts
    )
    if not paths:
        raise ValueError(f"No Python source files found for {package_name}.")
    digest = hashlib.sha256()
    for path, root in paths:
        digest.update(root.name.encode("utf-8"))
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class TargetRecord:
    name: str
    family: str
    module_type: str
    in_features: int
    out_features: int
    weight_shape: tuple[int, int]
    weight_dtype: str
    weight_device: str


@dataclass(frozen=True)
class TargetManifest:
    kind: Literal["eligible", "selected"]
    expected_count: int
    records: tuple[TargetRecord, ...]
    sha256: str

    @property
    def exact_target_names(self) -> tuple[str, ...]:
        return tuple(record.name for record in self.records)

    @property
    def families(self) -> tuple[str, ...]:
        return tuple(sorted({record.family for record in self.records}))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": KT_LORA_VARIANT_SCHEMA_VERSION,
            "kind": self.kind,
            "expected_count": self.expected_count,
            "actual_count": len(self.records),
            "families": list(self.families),
            "exact_target_names": list(self.exact_target_names),
            "records": [asdict(record) for record in self.records],
            "sha256": self.sha256,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any], *, expected_kind: Literal["eligible", "selected"]) -> "TargetManifest":
        if payload.get("schema_version") != KT_LORA_VARIANT_SCHEMA_VERSION:
            raise ValueError(f"Unsupported KT LoRA target schema: {payload.get('schema_version')!r}.")
        if payload.get("kind") != expected_kind:
            raise ValueError(f"Expected a {expected_kind} target manifest, got {payload.get('kind')!r}.")

        records = tuple(
            TargetRecord(
                name=record["name"],
                family=record["family"],
                module_type=record["module_type"],
                in_features=int(record["in_features"]),
                out_features=int(record["out_features"]),
                weight_shape=tuple(record["weight_shape"]),
                weight_dtype=record["weight_dtype"],
                weight_device=record["weight_device"],
            )
            for record in payload["records"]
        )
        if any(record.family not in ATTENTION_FAMILIES for record in records):
            raise ValueError("Target manifest contains a projection outside the audited attention allowlist.")
        if tuple(sorted(record.name for record in records)) != tuple(record.name for record in records):
            raise ValueError("Target manifest records must be sorted by exact module name.")
        if len(records) != len({record.name for record in records}):
            raise ValueError("Target manifest contains duplicate exact module names.")
        if payload.get("actual_count") != len(records):
            raise ValueError("Target manifest actual_count does not match its records.")
        if payload.get("families") != sorted({record.family for record in records}):
            raise ValueError("Target manifest families do not match its records.")
        if payload.get("exact_target_names") != [record.name for record in records]:
            raise ValueError("Target manifest exact_target_names do not match its records.")
        for record in records:
            if record.in_features <= 0 or record.out_features <= 0:
                raise ValueError(f"Target manifest contains non-positive dimensions for {record.name}.")
            if record.weight_shape != (record.out_features, record.in_features):
                raise ValueError(f"Target manifest weight shape is inconsistent for {record.name}.")

        digest = canonical_json_sha256([asdict(record) for record in records])
        if payload.get("sha256") != digest:
            raise ValueError("Target manifest content hash does not match its records.")
        manifest = cls(
            kind=expected_kind,
            expected_count=int(payload["expected_count"]),
            records=records,
            sha256=digest,
        )
        if len(records) != manifest.expected_count:
            raise ValueError(f"{expected_kind.title()} target manifest does not contain its expected module count.")
        return manifest


@dataclass(frozen=True)
class LoraPair:
    name: str
    module: nn.Module
    lora_a: nn.Parameter
    lora_b: nn.Parameter
    scaling: float


@dataclass(frozen=True)
class KTLoraVariantConfig:
    variant: Literal["altlora_attn", "plop_attn", "bilora_attn"]
    eligible_sha256: str
    selected_sha256: str
    exact_target_names: tuple[str, ...]
    lora_rank: int
    lora_alpha: int
    lora_dropout: float
    method: dict[str, Any]
    schema_version: int = KT_LORA_VARIANT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.variant not in ["altlora_attn", "plop_attn", "bilora_attn"]:
            raise ValueError(f"Unsupported KT LoRA variant: {self.variant!r}.")
        if self.schema_version != KT_LORA_VARIANT_SCHEMA_VERSION:
            raise ValueError(f"Unsupported KT LoRA variant schema: {self.schema_version}.")
        if self.lora_rank <= 0 or self.lora_alpha <= 0 or not 0 <= self.lora_dropout < 1:
            raise ValueError("KT LoRA variant rank/alpha/dropout metadata is invalid.")
        if not self.exact_target_names or tuple(sorted(set(self.exact_target_names))) != self.exact_target_names:
            raise ValueError("KT LoRA variant exact target names must be non-empty, unique, and sorted.")
        for digest in (self.eligible_sha256, self.selected_sha256):
            if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
                raise ValueError("KT LoRA variant metadata contains an invalid SHA-256 digest.")
        if not isinstance(self.method, dict) or not self.method:
            raise ValueError("KT LoRA variant method metadata must be a non-empty mapping.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "KTLoraVariantConfig":
        payload = dict(payload)
        if payload.get("schema_version") != KT_LORA_VARIANT_SCHEMA_VERSION:
            raise ValueError(f"Unsupported KT LoRA variant metadata schema: {payload.get('schema_version')!r}.")
        payload["exact_target_names"] = tuple(payload["exact_target_names"])
        return cls(**payload)


def get_logical_weight(module: nn.Module) -> torch.Tensor | None:
    candidates = [module]
    for attribute in ("orig_module", "base_layer"):
        candidate = getattr(module, attribute, None)
        if isinstance(candidate, nn.Module):
            candidates.append(candidate)

    for candidate in candidates:
        weight = getattr(candidate, "weight", None)
        if isinstance(weight, torch.Tensor) and weight.ndim == 2:
            return weight
    return None


def verify_kt_fused_expert_lora_contract(model: nn.Module, *, expected_rank: int) -> int:
    wrappers = None
    for candidate in model.modules():
        candidate_wrappers = candidate.__dict__.get("_kt_wrappers")
        if candidate_wrappers:
            wrappers = list(candidate_wrappers)
            break
    if not wrappers:
        raise ValueError("KT LoRA variants require registered KTransformers MoE wrappers.")
    for wrapper in wrappers:
        layer_index = getattr(wrapper, "layer_idx", "unknown")
        if not getattr(wrapper, "_fused_experts", False):
            raise ValueError(
                f"KT LoRA variants require independently-created fused expert LoRA; layer {layer_index} is not fused."
            )
        if getattr(wrapper, "_lora_rank", None) != expected_rank:
            raise ValueError(
                f"KT fused expert LoRA rank mismatch at layer {layer_index}: "
                f"{getattr(wrapper, '_lora_rank', None)} != {expected_rank}."
            )
        if getattr(wrapper, "lora_experts", None) is not None:
            raise ValueError("Legacy KT-owned LoRA Experts must remain disabled for KT LoRA variants.")
    return len(wrappers)


def _expected_attention_target_count(model: nn.Module) -> int | None:
    config = getattr(model, "config", None)
    config = getattr(config, "text_config", config)
    layer_types = getattr(config, "layer_types", None)
    if not isinstance(layer_types, (list, tuple)) or not layer_types:
        return None

    count = 0
    for layer_type in layer_types:
        if layer_type == "linear_attention":
            count += len(LINEAR_ATTENTION_FAMILIES)
        elif layer_type == "full_attention":
            count += len(FULL_ATTENTION_FAMILIES)
        else:
            return None
    return count


def _make_manifest(
    records: list[TargetRecord], kind: Literal["eligible", "selected"], expected_count: int
) -> TargetManifest:
    ordered = tuple(sorted(records, key=lambda record: record.name))
    payload = [asdict(record) for record in ordered]
    return TargetManifest(
        kind=kind,
        expected_count=expected_count,
        records=ordered,
        sha256=canonical_json_sha256(payload),
    )


def resolve_eligible_gpu_lora_targets(
    model: nn.Module,
    *,
    require_gpu: bool = True,
    expected_count: int | None = None,
    fsdp_staging_device: str | None = None,
) -> TargetManifest:
    """Resolve the exact logical GPU attention projections allowed by the KT variant contract.

    FSDP2 discovers PEFT targets before ``Accelerator.prepare`` moves and shards the model.  During
    that narrow phase, a base weight may legitimately be staged on CPU.  ``fsdp_staging_device``
    records its required post-prepare accelerator type in the manifest; callers must still invoke
    :func:`validate_runtime_attention_placement` before the first forward.  Meta tensors and CPU
    staging outside this explicit FSDP path remain fail-closed.
    """
    if fsdp_staging_device is not None and fsdp_staging_device not in ACCELERATOR_DEVICE_TYPES:
        raise ValueError(
            "FSDP attention staging must name an accelerator device type, got "
            f"{fsdp_staging_device!r}."
        )

    records: list[TargetRecord] = []
    seen_weight_ids: set[int] = set()
    for name, module in model.named_modules():
        family = name.rsplit(".", 1)[-1]
        if family not in ATTENTION_FAMILIES:
            continue

        normalized_name = f".{name.lower()}."
        if any(part in normalized_name for part in FORBIDDEN_PATH_PARTS):
            raise ValueError(f"Forbidden module path matched the attention allowlist: {name}.")
        if family in LINEAR_ATTENTION_FAMILIES and ".linear_attn." not in normalized_name:
            continue
        if family in FULL_ATTENTION_FAMILIES and ".self_attn." not in normalized_name:
            continue

        weight = get_logical_weight(module)
        if weight is None:
            raise ValueError(f"Eligible logical attention module has no unique 2D base weight: {name}.")
        if id(weight) in seen_weight_ids:
            raise ValueError(f"Eligible logical attention weight was discovered more than once: {name}.")
        seen_weight_ids.add(id(weight))
        weight_device = weight.device.type
        if require_gpu and weight_device not in ACCELERATOR_DEVICE_TYPES:
            if weight_device == "cpu" and fsdp_staging_device is not None:
                # The persisted manifest describes the device contract that will be checked after
                # FSDP2 replaces the staged Parameter with its accelerator-backed DTensor.
                weight_device = fsdp_staging_device
            else:
                raise ValueError(
                    f"Eligible attention base weight is not on an accelerator: {name} ({weight.device})."
                )

        out_features, in_features = (int(dim) for dim in weight.shape)
        records.append(
            TargetRecord(
                name=name,
                family=family,
                module_type=type(module).__name__,
                in_features=in_features,
                out_features=out_features,
                weight_shape=(out_features, in_features),
                weight_dtype=str(weight.dtype).removeprefix("torch."),
                weight_device=weight_device,
            )
        )

    derived_count = _expected_attention_target_count(model)
    required_count = expected_count if expected_count is not None else derived_count
    if required_count is None:
        raise ValueError("Cannot derive the expected eligible attention target count from model.config.layer_types.")
    if len(records) != required_count:
        raise ValueError(
            f"Eligible attention target inventory mismatch: expected {required_count}, found {len(records)}."
        )
    return _make_manifest(records, "eligible", required_count)


def validate_runtime_attention_placement(
    model: nn.Module,
    eligible: TargetManifest,
    selected: TargetManifest,
) -> "tuple[LoraPair, ...]":
    """Validate the manifest and injected PEFT parameters after FSDP/device preparation.

    The manifest uses canonical pre-injection names, while PEFT and FSDP may add wrapper prefixes.
    Matching therefore accepts only an exact name or a single canonical suffix and still requires
    one unique logical module for every eligible target.
    """
    eligible_names = set(eligible.exact_target_names)
    if not set(selected.exact_target_names).issubset(eligible_names):
        raise ValueError("Runtime selected attention targets are outside the eligible manifest.")

    named_modules = tuple(model.named_modules())
    for record in eligible.records:
        matches = [
            (module_name, module)
            for module_name, module in named_modules
            if module_name == record.name or module_name.endswith(f".{record.name}")
        ]
        if len(matches) != 1:
            raise ValueError(
                f"Runtime attention target {record.name} resolved {len(matches)} modules; expected exactly one."
            )
        module_name, module = matches[0]
        weight = get_logical_weight(module)
        if weight is None:
            raise ValueError(f"Runtime attention target has no unique 2D base weight: {module_name}.")
        if weight.device.type not in ACCELERATOR_DEVICE_TYPES:
            raise ValueError(
                f"Runtime attention base weight is not on an accelerator: {module_name} ({weight.device})."
            )
        if tuple(int(dim) for dim in weight.shape) != record.weight_shape:
            raise ValueError(
                f"Runtime attention base shape differs at {module_name}: "
                f"{tuple(weight.shape)} != {record.weight_shape}."
            )
        runtime_dtype = str(weight.dtype).removeprefix("torch.")
        if runtime_dtype != record.weight_dtype or weight.device.type != record.weight_device:
            raise ValueError(
                f"Runtime attention dtype/device differs at {module_name}: "
                f"{runtime_dtype}/{weight.device.type} != {record.weight_dtype}/{record.weight_device}."
            )

    pairs = resolve_injected_lora_pairs(model, selected)
    for pair in pairs:
        for factor_name, parameter in (("A", pair.lora_a), ("B", pair.lora_b)):
            if parameter.device.type not in ACCELERATOR_DEVICE_TYPES:
                raise ValueError(
                    f"Runtime attention LoRA {factor_name} is not on an accelerator: "
                    f"{pair.name} ({parameter.device})."
                )
            if not parameter.requires_grad:
                raise ValueError(f"Runtime attention LoRA {pair.name}.{factor_name} is unexpectedly frozen.")
    return pairs


def select_target_manifest(
    eligible: TargetManifest, exact_target_names: list[str] | tuple[str, ...]
) -> TargetManifest:
    selected_names = tuple(sorted(exact_target_names))
    if not selected_names or len(selected_names) != len(set(selected_names)):
        raise ValueError("Selected target names must be non-empty and unique.")
    eligible_by_name = {record.name: record for record in eligible.records}
    unknown = sorted(set(selected_names) - set(eligible_by_name))
    if unknown:
        raise ValueError(f"Selected targets are outside the eligible attention inventory: {unknown[:5]}.")
    return _make_manifest([eligible_by_name[name] for name in selected_names], "selected", len(selected_names))


def _adapter_name(module: nn.Module) -> str:
    active = getattr(module, "active_adapter", None)
    if isinstance(active, str):
        return active
    if isinstance(active, (list, tuple)) and len(active) == 1:
        return active[0]
    active_adapters = getattr(module, "active_adapters", None)
    if isinstance(active_adapters, (list, tuple)) and len(active_adapters) == 1:
        return active_adapters[0]
    return "default"


def _module_dict_item(container: Any, key: str) -> nn.Module | None:
    if isinstance(container, (nn.ModuleDict, dict)) and key in container:
        value = container[key]
        return value if isinstance(value, nn.Module) else None
    return None


def resolve_injected_lora_pairs(model: nn.Module, selected: TargetManifest) -> tuple[LoraPair, ...]:
    """Resolve A/B by PEFT Linear module identity, never by parameter adjacency or name substring."""
    selected_names = set(selected.exact_target_names)
    matches: dict[str, list[LoraPair]] = {name: [] for name in selected_names}
    unexpected_modules: list[str] = []
    for module_name, module in model.named_modules():
        lora_a_dict = getattr(module, "lora_A", None)
        lora_b_dict = getattr(module, "lora_B", None)
        if lora_a_dict is None or lora_b_dict is None:
            continue
        canonical_matches = [
            name for name in selected_names if module_name == name or module_name.endswith(f".{name}")
        ]
        if not canonical_matches:
            if any(len(container) > 0 for container in (lora_a_dict, lora_b_dict)):
                unexpected_modules.append(module_name)
            continue
        if len(canonical_matches) != 1:
            raise ValueError(f"Ambiguous canonical target match for injected PEFT module {module_name}.")

        adapter = _adapter_name(module)
        lora_a_module = _module_dict_item(lora_a_dict, adapter)
        lora_b_module = _module_dict_item(lora_b_dict, adapter)
        if lora_a_module is None or lora_b_module is None:
            raise ValueError(f"Missing complete PEFT A/B pair for {module_name} adapter {adapter}.")
        lora_a = getattr(lora_a_module, "weight", None)
        lora_b = getattr(lora_b_module, "weight", None)
        if not isinstance(lora_a, nn.Parameter) or not isinstance(lora_b, nn.Parameter):
            raise ValueError(f"PEFT A/B weights are not Parameters for {module_name}.")
        if lora_a.ndim != 2 or lora_b.ndim != 2 or lora_a.shape[0] != lora_b.shape[1]:
            raise ValueError(
                f"Invalid PEFT A/B shapes for {module_name}: {tuple(lora_a.shape)}, {tuple(lora_b.shape)}."
            )
        scaling_dict = getattr(module, "scaling", None)
        if not isinstance(scaling_dict, dict) or adapter not in scaling_dict:
            raise ValueError(f"Missing PEFT scaling for {module_name} adapter {adapter}.")
        canonical_name = canonical_matches[0]
        matches[canonical_name].append(LoraPair(canonical_name, module, lora_a, lora_b, float(scaling_dict[adapter])))

    missing = [name for name, pairs in matches.items() if len(pairs) == 0]
    duplicated = [name for name, pairs in matches.items() if len(pairs) > 1]
    if missing or duplicated or unexpected_modules:
        raise ValueError(
            "Injected PEFT target mismatch: "
            f"missing={missing[:5]}, duplicated={duplicated[:5]}, unexpected={unexpected_modules[:5]}."
        )
    return tuple(matches[name][0] for name in sorted(matches))


def set_variant_config(model: nn.Module, config: KTLoraVariantConfig) -> None:
    setattr(model, "_kt_lora_variant_config", config)


def set_variant_manifests(model: nn.Module, eligible: TargetManifest, selected: TargetManifest) -> None:
    setattr(model, "_kt_lora_eligible_manifest", eligible)
    setattr(model, "_kt_lora_selected_manifest", selected)


def get_variant_config(model: nn.Module) -> KTLoraVariantConfig | None:
    queue: list[nn.Module] = [model]
    visited: set[int] = set()
    while queue:
        current = queue.pop()
        if id(current) in visited:
            continue
        visited.add(id(current))
        config = current.__dict__.get("_kt_lora_variant_config")
        if isinstance(config, KTLoraVariantConfig):
            return config
        for attribute in ("module", "base_model", "model", "pretrained_model"):
            child = getattr(current, attribute, None)
            if isinstance(child, nn.Module):
                queue.append(child)
    return None


def save_variant_artifacts(model: nn.Module, output_dir: str | Path) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    config = get_variant_config(model)
    if config is None:
        return
    _atomic_json_dump(config.to_dict(), output_path / KT_LORA_VARIANT_CONFIG_NAME)

    for attribute, filename in (
        ("_kt_lora_eligible_manifest", KT_LORA_ELIGIBLE_MANIFEST_NAME),
        ("_kt_lora_selected_manifest", KT_LORA_SELECTED_MANIFEST_NAME),
    ):
        manifest = getattr(model, attribute, None)
        if isinstance(manifest, TargetManifest):
            _atomic_json_dump(manifest.to_dict(), output_path / filename)


def _atomic_json_dump(payload: dict[str, Any], path: Path) -> None:
    temporary_path = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    with open(temporary_path, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    os.replace(temporary_path, path)


def load_target_manifest(path: str | Path, *, expected_kind: Literal["eligible", "selected"]) -> TargetManifest:
    with open(path, encoding="utf-8") as stream:
        payload = json.load(stream)
    return TargetManifest.from_dict(payload, expected_kind=expected_kind)


def load_variant_config(
    adapter_path: str | os.PathLike[str],
    *,
    subfolder: str | None = None,
    cache_dir: str | None = None,
    revision: str | None = None,
    token: str | bool | None = None,
) -> KTLoraVariantConfig | None:
    adapter_path = os.fspath(adapter_path)
    if os.path.isfile(adapter_path):
        config_path = adapter_path if os.path.basename(adapter_path) == KT_LORA_VARIANT_CONFIG_NAME else None
    elif os.path.isdir(adapter_path):
        config_path = os.path.join(adapter_path, subfolder or "", KT_LORA_VARIANT_CONFIG_NAME)
        if not os.path.isfile(config_path):
            config_path = None
    else:
        config_path = cached_file(
            adapter_path,
            KT_LORA_VARIANT_CONFIG_NAME,
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
    with open(config_path, encoding="utf-8") as stream:
        return KTLoraVariantConfig.from_dict(json.load(stream))


def tensor_to_full(tensor: torch.Tensor) -> torch.Tensor:
    full_tensor = getattr(tensor, "full_tensor", None)
    return full_tensor() if callable(full_tensor) else tensor


@torch.no_grad()
def copy_full_to_tensor(target: torch.Tensor, value: torch.Tensor) -> None:
    try:
        from torch.distributed.tensor import DTensor, distribute_tensor
    except ImportError:
        target.copy_(value.to(device=target.device, dtype=target.dtype))
        return

    if isinstance(target, DTensor):
        distributed = distribute_tensor(
            value.to(device=target.device, dtype=target.dtype),
            device_mesh=target.device_mesh,
            placements=target.placements,
        )
        target.to_local().copy_(distributed.to_local())
    else:
        target.copy_(value.to(device=target.device, dtype=target.dtype))


def rebuild_adamw_param_groups(optimizer: torch.optim.Optimizer, pairs: tuple[LoraPair, ...]) -> list[dict[str, Any]]:
    if type(optimizer) is not torch.optim.AdamW:
        raise ValueError(
            f"KT LoRA variants require the exact torch.optim.AdamW baseline, got {type(optimizer).__name__}."
        )

    owner: dict[int, dict[str, Any]] = {}
    pair_ids = {id(param) for pair in pairs for param in (pair.lora_a, pair.lora_b)}
    rebuilt: list[dict[str, Any]] = []
    for group in optimizer.param_groups:
        options = {key: value for key, value in group.items() if key != "params"}
        remaining = []
        for param in group["params"]:
            if id(param) in owner:
                raise ValueError("Baseline optimizer owns the same parameter more than once.")
            owner[id(param)] = options
            if id(param) not in pair_ids:
                remaining.append(param)
        if remaining:
            rebuilt.append({**options, "params": remaining, "variant_role": "baseline"})

    for pair in pairs:
        a_options = owner.get(id(pair.lora_a))
        b_options = owner.get(id(pair.lora_b))
        if a_options is None or b_options is None:
            raise ValueError(f"Baseline optimizer does not own both attention factors for {pair.name}.")
        comparable = ("lr", "betas", "eps", "weight_decay", "amsgrad", "maximize")
        if any(a_options.get(key) != b_options.get(key) for key in comparable):
            raise ValueError(f"A/B baseline optimizer options differ for {pair.name}.")
        rebuilt.append(
            {
                **a_options,
                "params": [pair.lora_a, pair.lora_b],
                "variant_role": "attention_pair",
                "pair_name": pair.name,
                "lora_scaling": pair.scaling,
            }
        )

    runtime_options = {key: value for key, value in optimizer.param_groups[-1].items() if key != "params"}
    rebuilt.append({**runtime_options, "params": [], "variant_role": "baseline_runtime_injection"})
    expected_ids = {id(param) for group in optimizer.param_groups for param in group["params"]}
    actual_ids = {id(param) for group in rebuilt for param in group["params"]}
    if expected_ids != actual_ids:
        raise ValueError("Variant optimizer parameter ownership differs from the baseline optimizer.")
    return rebuilt


def adamw_constructor_kwargs(optimizer: torch.optim.Optimizer) -> dict[str, Any]:
    """Return only public AdamW constructor defaults supported by the installed torch version.

    Newer torch releases keep internal optimizer flags (for example
    ``decoupled_weight_decay`` in torch 2.9) inside ``Optimizer.defaults`` even though
    ``AdamW.__init__`` does not accept them as keyword arguments.  The complete flags remain
    preserved in the rebuilt parameter groups; this filter is only for constructing AdamW.
    """
    parameters = inspect.signature(torch.optim.AdamW.__init__).parameters
    return {
        key: value
        for key, value in optimizer.defaults.items()
        if key in parameters and key not in ("self", "params")
    }
