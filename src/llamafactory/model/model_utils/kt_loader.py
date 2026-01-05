# Copyright 2025 the LlamaFactory team.
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

"""
KTransformers Model Loader

This module provides model loading utilities for KT fine-tuning integration.
The key feature is controlling weight loading via device_map to avoid loading
MoE experts to GPU first and then copying to CPU.

Design:
- Attention, Embedding, LM Head -> GPU
- Router (gate) -> GPU
- MoE Experts -> CPU (or skip via "meta" device if loading from kt_weight_path)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from ...extras import logging


if TYPE_CHECKING:
    from transformers import PretrainedConfig, PreTrainedModel

    from ...hparams import ModelArguments
    from .kt_moe import MOEArchConfig


logger = logging.get_logger(__name__)


# Check if safetensors is available
try:
    from safetensors import safe_open
    SAFETENSORS_AVAILABLE = True
except ImportError:
    SAFETENSORS_AVAILABLE = False
    safe_open = None


def _get_layers_prefix(config: "PretrainedConfig") -> str:
    """Get the prefix for transformer layers based on model architecture."""
    arch = config.architectures[0] if config.architectures else ""

    # Most models use "model.layers"
    if any(x in arch for x in ["Deepseek", "Qwen", "Mixtral", "Llama"]):
        return "model.layers"

    # Fallback
    return "model.layers"


def _get_moe_expert_pattern(
    layer_idx: int,
    expert_idx: int,
    moe_config: MOEArchConfig,
    layers_prefix: str,
) -> str:
    """Get the full parameter name pattern for a specific expert."""
    # e.g., "model.layers.0.mlp.experts.0"
    return f"{layers_prefix}.{layer_idx}.{moe_config.moe_layer_attr}.{moe_config.experts_attr}.{expert_idx}"


def _get_shared_expert_pattern(
    layer_idx: int,
    moe_config: MOEArchConfig,
    layers_prefix: str,
) -> str:
    """Get the full parameter name pattern for shared experts."""
    # e.g., "model.layers.0.mlp.shared_experts"
    return f"{layers_prefix}.{layer_idx}.{moe_config.moe_layer_attr}.shared_experts"


def build_kt_device_map(
    config: "PretrainedConfig",
    model_args: "ModelArguments",
) -> dict[str, str | int]:
    """
    Build device_map for KT model loading with hybrid GPU/CPU expert support.

    This function creates a device_map using the "layer-first-GPU, then override experts"
    strategy. This approach works with models that have mixed dense/MoE layers.

    Strategy:
    1. Map entire layer to GPU (covers attention, layernorms, dense MLP, shared experts, router)
    2. Override individual routed experts based on kt_num_gpu_experts setting

    For KT fine-tuning:
    - GPU experts (0 to kt_num_gpu_experts-1) -> GPU (cuda:0)
    - CPU experts (kt_num_gpu_experts to total-1) -> CPU or "meta"

    If kt_weight_path is provided, CPU experts are set to "meta" (skip loading)
    because they will be loaded separately from the preprocessed weight file.

    Args:
        config: HuggingFace model configuration
        model_args: Model arguments containing kt_num_gpu_experts and kt_weight_path

    Returns:
        device_map dictionary mapping module names to devices
    """
    from .kt_moe import get_moe_arch_config

    moe_config = get_moe_arch_config(config)
    layers_prefix = _get_layers_prefix(config)
    num_layers = config.num_hidden_layers
    num_experts = moe_config.expert_num
    num_gpu_experts = model_args.kt_num_gpu_experts
    has_kt_weight_path = model_args.kt_weight_path is not None

    device_map: dict[str, str | int] = {}

    # 1. Global layers -> GPU
    device_map["model.embed_tokens"] = "cuda:0"
    device_map["model.norm"] = "cuda:0"
    device_map["lm_head"] = "cuda:0"

    # 2. Per-layer mappings
    for layer_idx in range(num_layers):
        layer_prefix = f"{layers_prefix}.{layer_idx}"

        # Step 1: Map entire layer to GPU
        # This covers attention, layernorms, dense MLP, shared experts, router, etc.
        device_map[layer_prefix] = "cuda:0"

        # Step 2: Override individual routed experts
        # If this layer doesn't have experts (dense layer), these keys won't match
        # anything and will be safely ignored
        moe_prefix = f"{layer_prefix}.{moe_config.moe_layer_attr}"

        for expert_idx in range(num_experts):
            expert_key = f"{moe_prefix}.{moe_config.experts_attr}.{expert_idx}"

            if expert_idx < num_gpu_experts:
                # GPU experts -> GPU (already covered by layer mapping, but explicit)
                device_map[expert_key] = "cuda:0"
            else:
                # CPU experts -> CPU or meta
                if has_kt_weight_path:
                    # Skip loading, will load from kt_weight_path
                    device_map[expert_key] = "meta"
                else:
                    # Load from HuggingFace to CPU directly
                    device_map[expert_key] = "cpu"

    logger.info(
        f"Built KT device_map: {num_gpu_experts} GPU experts, "
        f"{num_experts - num_gpu_experts} CPU experts "
        f"({'skip loading' if has_kt_weight_path else 'load to CPU'})"
    )

    return device_map


def build_kt_device_map_simplified(
    config: "PretrainedConfig",
    model_args: "ModelArguments",
) -> dict[str, str | int]:
    """
    Simplified device_map builder using "layer-first-GPU, then override experts" strategy.

    This approach is more general and works with models that have:
    - Dense MLP layers (no experts)
    - MoE layers (with routed experts)
    - Mixed architectures (some dense, some MoE layers)

    Strategy:
    1. Map entire layer to GPU
    2. Override only routed experts to CPU/meta

    This ensures:
    - Dense layers work automatically (no experts to override)
    - MoE layers have experts on CPU, everything else on GPU
    - Shared experts stay on GPU (part of the layer)

    Args:
        config: HuggingFace model configuration
        model_args: Model arguments

    Returns:
        device_map dictionary
    """
    from .kt_moe import get_moe_arch_config

    moe_config = get_moe_arch_config(config)
    layers_prefix = _get_layers_prefix(config)
    num_layers = config.num_hidden_layers
    num_gpu_experts = model_args.kt_num_gpu_experts
    has_kt_weight_path = model_args.kt_weight_path is not None

    device_map: dict[str, str | int] = {}

    # Global layers -> GPU
    device_map["model.embed_tokens"] = "cuda:0"
    device_map["model.norm"] = "cuda:0"
    device_map["lm_head"] = "cuda:0"

    for layer_idx in range(num_layers):
        layer_prefix = f"{layers_prefix}.{layer_idx}"

        # Step 1: Map entire layer to GPU
        # This covers attention, layernorms, dense MLP, shared experts, router, etc.
        device_map[layer_prefix] = "cuda:0"

        # Step 2: Override only routed experts to CPU/meta
        # If this layer doesn't have experts (dense layer), this key won't match anything
        # and will be safely ignored
        experts_prefix = f"{layer_prefix}.{moe_config.moe_layer_attr}.{moe_config.experts_attr}"

        if num_gpu_experts == 0:
            # All experts to CPU/meta
            if has_kt_weight_path:
                device_map[experts_prefix] = "meta"
            else:
                device_map[experts_prefix] = "cpu"
        else:
            # Hybrid mode: need per-expert mapping
            # Fall back to detailed mapping
            return build_kt_device_map(config, model_args)

    logger.info(
        f"Built simplified KT device_map: all layers on GPU, "
        f"routed experts on {'meta' if has_kt_weight_path else 'CPU'}"
    )

    return device_map


def get_kt_loading_kwargs(
    config: "PretrainedConfig",
    model_args: "ModelArguments",
) -> dict[str, Any]:
    """
    Get kwargs for AutoModelForCausalLM.from_pretrained() for KT loading.

    Args:
        config: HuggingFace model configuration
        model_args: Model arguments

    Returns:
        Dictionary of kwargs for from_pretrained()
    """
    # Use simplified mapping for pure CPU mode (kt_num_gpu_experts=0)
    if model_args.kt_num_gpu_experts == 0:
        device_map = build_kt_device_map_simplified(config, model_args)
    else:
        device_map = build_kt_device_map(config, model_args)

    return {
        "config": config,
        "torch_dtype": torch.bfloat16,
        "device_map": device_map,
        "trust_remote_code": model_args.trust_remote_code,
        "token": model_args.hf_hub_token,
        "low_cpu_mem_usage": True,
    }


# =============================================================================
# kt_weight_path Loading Functions
# =============================================================================


def _find_safetensor_files(kt_weight_path: str) -> list[str]:
    """Find all safetensors files in the kt_weight_path directory."""
    if not os.path.isdir(kt_weight_path):
        raise FileNotFoundError(f"kt_weight_path directory not found: {kt_weight_path}")

    safetensor_files = []
    for file in sorted(os.listdir(kt_weight_path)):
        if file.endswith(".safetensors"):
            safetensor_files.append(os.path.join(kt_weight_path, file))

    if not safetensor_files:
        raise FileNotFoundError(f"No safetensors files found in {kt_weight_path}")

    return safetensor_files


def _load_kt_weight_index(kt_weight_path: str) -> dict[str, str]:
    """
    Build an index mapping tensor keys to their safetensors file paths.

    Args:
        kt_weight_path: Path to directory containing preprocessed weights

    Returns:
        Dictionary mapping tensor keys to file paths
    """
    if not SAFETENSORS_AVAILABLE:
        raise ImportError("safetensors is required for loading kt_weight_path")

    index = {}
    safetensor_files = _find_safetensor_files(kt_weight_path)

    for file_path in safetensor_files:
        with safe_open(file_path, framework="pt") as f:
            for key in f.keys():
                index[key] = file_path

    logger.info(f"Indexed {len(index)} tensors from {len(safetensor_files)} safetensors files")
    return index


@dataclass
class INT8ExpertWeights:
    """Container for INT8 expert weights with scales."""
    gate_proj: torch.Tensor  # [expert_num, intermediate_size, hidden_size], INT8
    gate_scale: torch.Tensor  # [expert_num, intermediate_size], F32
    up_proj: torch.Tensor    # [expert_num, intermediate_size, hidden_size], INT8
    up_scale: torch.Tensor   # [expert_num, intermediate_size], F32
    down_proj: torch.Tensor  # [expert_num, hidden_size, intermediate_size], INT8
    down_scale: torch.Tensor # [expert_num, hidden_size], F32


def load_experts_from_kt_weight_path(
    kt_weight_path: str,
    layer_idx: int,
    num_experts: int,
    hidden_size: int,
    intermediate_size: int,
) -> INT8ExpertWeights:
    """
    Load INT8 preprocessed expert weights from kt_weight_path for a specific layer.

    The preprocessed weights format (from convert_cpu_weights.py):
    - blk.{layer}.ffn_gate_exps.{expert}.numa.{numa_idx}.weight (INT8, flattened)
    - blk.{layer}.ffn_gate_exps.{expert}.numa.{numa_idx}.scale (F32)
    - Same for ffn_up_exps and ffn_down_exps

    The weights are stored per-NUMA partition. This function merges the NUMA partitions
    and stacks all experts into a single tensor.

    Args:
        kt_weight_path: Path to preprocessed weights directory
        layer_idx: Layer index (MoE layer index, e.g., 1 for DeepSeek-V2 since layer 0 is dense)
        num_experts: Total number of experts
        hidden_size: Model hidden size
        intermediate_size: MoE intermediate size

    Returns:
        INT8ExpertWeights containing gate/up/down projections with scales

    Raises:
        ImportError: If safetensors is not available
        FileNotFoundError: If weight files are not found
    """
    if not SAFETENSORS_AVAILABLE:
        raise ImportError("safetensors is required for loading kt_weight_path")

    # Build index if not already built
    index = _load_kt_weight_index(kt_weight_path)

    # Detect number of NUMA partitions by checking keys
    numa_count = 0
    test_key_prefix = f"blk.{layer_idx}.ffn_gate_exps.0.numa."
    for key in index.keys():
        if key.startswith(test_key_prefix) and key.endswith(".weight"):
            numa_idx = int(key.split("numa.")[1].split(".")[0])
            numa_count = max(numa_count, numa_idx + 1)

    if numa_count == 0:
        raise FileNotFoundError(
            f"No weights found for layer {layer_idx} in {kt_weight_path}. "
            f"Expected keys like 'blk.{layer_idx}.ffn_gate_exps.0.numa.0.weight'"
        )

    logger.info(f"Loading INT8 weights for layer {layer_idx}: {num_experts} experts, {numa_count} NUMA partitions")

    # Storage for all experts
    gate_weights_list = []
    gate_scales_list = []
    up_weights_list = []
    up_scales_list = []
    down_weights_list = []
    down_scales_list = []

    # Load weights for each expert
    for expert_idx in range(num_experts):
        # Load and merge NUMA partitions for gate projection
        gate_w_parts = []
        gate_s_parts = []
        for numa_idx in range(numa_count):
            w_key = f"blk.{layer_idx}.ffn_gate_exps.{expert_idx}.numa.{numa_idx}.weight"
            s_key = f"blk.{layer_idx}.ffn_gate_exps.{expert_idx}.numa.{numa_idx}.scale"

            if w_key not in index:
                raise FileNotFoundError(f"Weight key not found: {w_key}")

            with safe_open(index[w_key], framework="pt") as f:
                gate_w_parts.append(f.get_tensor(w_key))
            with safe_open(index[s_key], framework="pt") as f:
                gate_s_parts.append(f.get_tensor(s_key))

        # Merge NUMA partitions: each partition has half the output dim
        # Weight: [numa0_out, hidden] + [numa1_out, hidden] -> [intermediate, hidden]
        # But weights are flattened, so we just concatenate
        gate_w = torch.cat(gate_w_parts, dim=0)  # Concatenate flattened weights
        gate_s = torch.cat(gate_s_parts, dim=0)  # Concatenate scales

        # Load and merge NUMA partitions for up projection
        up_w_parts = []
        up_s_parts = []
        for numa_idx in range(numa_count):
            w_key = f"blk.{layer_idx}.ffn_up_exps.{expert_idx}.numa.{numa_idx}.weight"
            s_key = f"blk.{layer_idx}.ffn_up_exps.{expert_idx}.numa.{numa_idx}.scale"

            with safe_open(index[w_key], framework="pt") as f:
                up_w_parts.append(f.get_tensor(w_key))
            with safe_open(index[s_key], framework="pt") as f:
                up_s_parts.append(f.get_tensor(s_key))

        up_w = torch.cat(up_w_parts, dim=0)
        up_s = torch.cat(up_s_parts, dim=0)

        # Load and merge NUMA partitions for down projection
        down_w_parts = []
        down_s_parts = []
        for numa_idx in range(numa_count):
            w_key = f"blk.{layer_idx}.ffn_down_exps.{expert_idx}.numa.{numa_idx}.weight"
            s_key = f"blk.{layer_idx}.ffn_down_exps.{expert_idx}.numa.{numa_idx}.scale"

            with safe_open(index[w_key], framework="pt") as f:
                down_w_parts.append(f.get_tensor(w_key))
            with safe_open(index[s_key], framework="pt") as f:
                down_s_parts.append(f.get_tensor(s_key))

        down_w = torch.cat(down_w_parts, dim=0)
        down_s = torch.cat(down_s_parts, dim=0)

        # Reshape flattened weights to proper shape
        # gate/up: [intermediate_size, hidden_size] -> flatten -> [intermediate_size * hidden_size]
        # down: [hidden_size, intermediate_size] -> flatten -> [hidden_size * intermediate_size]
        gate_w = gate_w.view(intermediate_size, hidden_size)
        up_w = up_w.view(intermediate_size, hidden_size)
        down_w = down_w.view(hidden_size, intermediate_size)

        gate_weights_list.append(gate_w)
        gate_scales_list.append(gate_s)
        up_weights_list.append(up_w)
        up_scales_list.append(up_s)
        down_weights_list.append(down_w)
        down_scales_list.append(down_s)

    # Stack all experts: [expert_num, ...]
    gate_proj = torch.stack(gate_weights_list, dim=0).contiguous()
    gate_scale = torch.stack(gate_scales_list, dim=0).contiguous()
    up_proj = torch.stack(up_weights_list, dim=0).contiguous()
    up_scale = torch.stack(up_scales_list, dim=0).contiguous()
    down_proj = torch.stack(down_weights_list, dim=0).contiguous()
    down_scale = torch.stack(down_scales_list, dim=0).contiguous()

    logger.info(
        f"Loaded INT8 weights for layer {layer_idx}: "
        f"gate={gate_proj.shape}, up={up_proj.shape}, down={down_proj.shape}"
    )

    return INT8ExpertWeights(
        gate_proj=gate_proj,
        gate_scale=gate_scale,
        up_proj=up_proj,
        up_scale=up_scale,
        down_proj=down_proj,
        down_scale=down_scale,
    )


def load_kt_expert_weights(
    kt_weight_path: str,
    layer_idx: int,
    num_experts: int,
    moe_config: MOEArchConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    """
    Load preprocessed expert weights from kt_weight_path for a specific layer.

    Note: This is the legacy function. Use load_experts_from_kt_weight_path() instead
    for INT8 weights with scales.

    Args:
        kt_weight_path: Path to preprocessed weights directory
        layer_idx: Layer index
        num_experts: Total number of experts
        moe_config: MoE architecture configuration

    Returns:
        Tuple of (gate_proj, up_proj, down_proj) tensors, or None if not found
    """
    # Redirect to the new function
    logger.warning(
        "load_kt_expert_weights() is deprecated. "
        "Use load_experts_from_kt_weight_path() for INT8 weights with scales."
    )
    return None


def check_experts_on_meta_device(
    model: "PreTrainedModel",
    moe_config: MOEArchConfig,
) -> bool:
    """
    Check if MoE experts are on meta device (i.e., weights not loaded).

    Args:
        model: The loaded model
        moe_config: MoE architecture configuration

    Returns:
        True if experts are on meta device, False otherwise
    """
    # Check the first MoE layer's first expert
    for layer in model.model.layers:
        moe_module = getattr(layer, moe_config.moe_layer_attr, None)
        if moe_module is None:
            continue

        experts = getattr(moe_module, moe_config.experts_attr, None)
        if experts is None:
            continue

        # Check the first expert's first weight
        if len(experts) > 0:
            first_expert = experts[0]
            gate_name = moe_config.weight_names[0]
            gate_proj = getattr(first_expert, gate_name, None)
            if gate_proj is not None:
                return gate_proj.weight.device.type == "meta"

    return False


def get_expert_device(
    model: "PreTrainedModel",
    moe_config: MOEArchConfig,
) -> str:
    """
    Get the device type of MoE experts.

    Args:
        model: The loaded model
        moe_config: MoE architecture configuration

    Returns:
        Device type string ("cpu", "cuda", "meta", etc.)
    """
    for layer in model.model.layers:
        moe_module = getattr(layer, moe_config.moe_layer_attr, None)
        if moe_module is None:
            continue

        experts = getattr(moe_module, moe_config.experts_attr, None)
        if experts is None:
            continue

        if len(experts) > 0:
            first_expert = experts[0]
            gate_name = moe_config.weight_names[0]
            gate_proj = getattr(first_expert, gate_name, None)
            if gate_proj is not None:
                return str(gate_proj.weight.device.type)

    return "unknown"
