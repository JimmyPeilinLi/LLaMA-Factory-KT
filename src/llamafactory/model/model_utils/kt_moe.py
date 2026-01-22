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
KTransformers MoE Backend Integration via KTMoEWrapper

This module provides KT acceleration for MoE layers in large language models
using the unified KTMoEWrapper factory interface from kt-kernel.

Key features:
- Unified interface via KTMoEWrapper factory
- Automatic CPUInfer singleton management
- TP/no-TP mode via threadpool_count parameter
- Zero-copy LoRA weight design
- Support for multiple quantization methods (AMXBF16_SFT, AMXINT8_SFT, etc.)

Migration from direct C++ bindings:
- KTMoEWrapper replaces direct kt_kernel_ext.moe.* calls
- BaseSFTMoEWrapper methods replace manual task submission
- CPUInfer is managed automatically by kt-kernel
"""

from __future__ import annotations

import importlib.util as _u
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...extras import logging
from .kt_loader import load_experts_from_kt_weight_path


if TYPE_CHECKING:
    from transformers import PretrainedConfig, PreTrainedModel

    from ...hparams import FinetuningArguments, ModelArguments


logger = logging.get_logger(__name__)

# Check if kt_kernel is available
KT_KERNEL_AVAILABLE = _u.find_spec("kt_kernel") is not None

if KT_KERNEL_AVAILABLE:
    try:
        from kt_kernel.experts import KTMoEWrapper
    except ImportError:
        KT_KERNEL_AVAILABLE = False
        KTMoEWrapper = None


# =============================================================================
# Exception Classes
# =============================================================================


class KTAMXError(Exception):
    """Base exception for KT AMX errors."""

    pass


class KTAMXNotAvailableError(KTAMXError):
    """kt_kernel not installed or AMX not supported."""

    pass


class KTAMXModelNotSupportedError(KTAMXError):
    """Model architecture not supported."""

    pass


class KTAMXConfigError(KTAMXError):
    """Configuration error."""

    pass


# =============================================================================
# MoE Configuration
# =============================================================================


@dataclass
class MOEArchConfig:
    """MoE architecture configuration for different model types."""

    moe_layer_attr: str  # Attribute name for MoE layer in transformer block
    router_attr: str  # Attribute name for router in MoE layer
    experts_attr: str  # Attribute name for experts list in MoE layer
    weight_names: tuple[str, str, str]  # (gate_proj, up_proj, down_proj) names
    expert_num: int  # Total number of experts
    intermediate_size: int  # MLP intermediate dimension
    num_experts_per_tok: int  # Number of experts per token (top-k)
    has_shared_experts: bool = False  # Whether model has shared experts
    router_type: str = "linear"  # Router type: "linear" (Qwen/Mixtral) or "deepseek_gate" (DeepSeek)


def get_moe_arch_config(config: "PretrainedConfig") -> MOEArchConfig:
    """
    Get MoE architecture configuration based on model type.

    Args:
        config: HuggingFace model configuration

    Returns:
        MOEArchConfig for the model

    Raises:
        KTAMXModelNotSupportedError: If model architecture is not supported
    """
    arch = config.architectures[0] if config.architectures else ""

    if "DeepseekV2" in arch:
        return MOEArchConfig(
            moe_layer_attr="mlp",
            router_attr="gate",
            experts_attr="experts",
            weight_names=("gate_proj", "up_proj", "down_proj"),
            expert_num=config.n_routed_experts,
            intermediate_size=config.moe_intermediate_size,
            num_experts_per_tok=config.num_experts_per_tok,
            has_shared_experts=getattr(config, "n_shared_experts", 0) > 0,
            router_type="deepseek_gate",  # DeepSeek router returns (topk_idx, topk_weight, aux_loss)
        )
    elif "DeepseekV3" in arch:
        return MOEArchConfig(
            moe_layer_attr="mlp",
            router_attr="gate",
            experts_attr="experts",
            weight_names=("gate_proj", "up_proj", "down_proj"),
            expert_num=config.n_routed_experts,
            intermediate_size=config.moe_intermediate_size,
            num_experts_per_tok=config.num_experts_per_tok,
            has_shared_experts=getattr(config, "n_shared_experts", 0) > 0,
            router_type="deepseek_gate",  # DeepSeek router returns (topk_idx, topk_weight, aux_loss)
        )
    elif "Qwen2Moe" in arch or "Qwen3Moe" in arch:
        return MOEArchConfig(
            moe_layer_attr="mlp",
            router_attr="gate",
            experts_attr="experts",
            weight_names=("gate_proj", "up_proj", "down_proj"),
            expert_num=config.num_experts,
            intermediate_size=config.moe_intermediate_size,
            num_experts_per_tok=config.num_experts_per_tok,
            has_shared_experts=getattr(config, "shared_expert_intermediate_size", 0) > 0,
        )
    elif "Mixtral" in arch:
        return MOEArchConfig(
            moe_layer_attr="block_sparse_moe",
            router_attr="gate",
            experts_attr="experts",
            weight_names=("w1", "w3", "w2"),  # gate=w1, up=w3, down=w2
            expert_num=config.num_local_experts,
            intermediate_size=config.intermediate_size,
            num_experts_per_tok=config.num_experts_per_tok,
            has_shared_experts=False,
        )
    else:
        raise KTAMXModelNotSupportedError(
            f"Model architecture {arch} not supported for KT AMX. "
            f"Supported architectures: DeepseekV2, DeepseekV3, Qwen2Moe, Qwen3Moe, Mixtral"
        )


def is_moe_layer(layer: nn.Module, moe_config: MOEArchConfig) -> bool:
    """Check if a transformer layer contains MoE."""
    moe_module = getattr(layer, moe_config.moe_layer_attr, None)
    if moe_module is None:
        return False
    return hasattr(moe_module, moe_config.experts_attr)


def get_moe_module(layer: nn.Module, moe_config: MOEArchConfig) -> nn.Module | None:
    """Get MoE module from transformer layer."""
    moe_module = getattr(layer, moe_config.moe_layer_attr, None)
    if moe_module is None:
        return None
    if not hasattr(moe_module, moe_config.experts_attr):
        return None
    return moe_module


# =============================================================================
# Weight Extraction
# =============================================================================


def extract_moe_weights(
    moe_module: nn.Module, moe_config: MOEArchConfig
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Extract MoE expert weights from the module.

    Args:
        moe_module: MoE layer module
        moe_config: MoE architecture configuration

    Returns:
        Tuple of (gate_proj, up_proj, down_proj) with shape
        [expert_num, intermediate_size/hidden_size, hidden_size/intermediate_size]
    """
    experts = getattr(moe_module, moe_config.experts_attr)
    gate_name, up_name, down_name = moe_config.weight_names

    gate_weights = []
    up_weights = []
    down_weights = []

    for expert in experts:
        gate_weights.append(getattr(expert, gate_name).weight.data)
        up_weights.append(getattr(expert, up_name).weight.data)
        down_weights.append(getattr(expert, down_name).weight.data)

    # Stack to [expert_num, out_features, in_features]
    gate_proj = torch.stack(gate_weights, dim=0)
    up_proj = torch.stack(up_weights, dim=0)
    down_proj = torch.stack(down_weights, dim=0)

    return gate_proj, up_proj, down_proj


def _clear_original_expert_weights(moe_module: nn.Module, moe_config: MOEArchConfig) -> None:
    """
    Clear original expert weights to free memory.

    After the AMX kernel has loaded the weights, we can release the HuggingFace
    weight references so Python GC can reclaim the memory. This is especially
    important when using kt_weight_path, where the original BF16 weights were
    loaded just as placeholders.

    Args:
        moe_module: Original MoE module with expert weights
        moe_config: MoE architecture configuration
    """
    experts = getattr(moe_module, moe_config.experts_attr, None)
    if experts is None:
        return

    for expert_idx, expert in enumerate(experts):
        for weight_name in moe_config.weight_names:
            proj = getattr(expert, weight_name, None)
            if proj is not None and hasattr(proj, "weight"):
                # Replace weight with an empty tensor to release memory
                # Note: We keep the module structure but release the large tensor
                original_device = proj.weight.device
                original_dtype = proj.weight.dtype
                proj.weight = nn.Parameter(
                    torch.empty(0, device=original_device, dtype=original_dtype),
                    requires_grad=False,
                )

    logger.debug(f"Cleared original expert weights for MoE module")


# =============================================================================
# LoRA Initialization
# =============================================================================


def create_lora_params(
    expert_num: int,
    hidden_size: int,
    intermediate_size: int,
    lora_rank: int,
    lora_alpha: float,
    device: str = "cpu",
    dtype: torch.dtype = torch.bfloat16,
) -> dict[str, nn.Parameter]:
    """
    Create LoRA parameters for MoE layer.

    Args:
        expert_num: Number of experts
        hidden_size: Hidden dimension
        intermediate_size: MLP intermediate dimension
        lora_rank: LoRA rank
        lora_alpha: LoRA scaling factor
        device: Device to create tensors on
        dtype: Data type for tensors

    Returns:
        Dictionary of LoRA parameters
    """
    # LoRA A matrices: initialized with kaiming_uniform
    # LoRA B matrices: initialized with zeros

    # Gate projection LoRA
    gate_lora_a = torch.zeros(expert_num, lora_rank, hidden_size, dtype=dtype, device=device)
    gate_lora_b = torch.zeros(expert_num, intermediate_size, lora_rank, dtype=dtype, device=device)

    # Up projection LoRA
    up_lora_a = torch.zeros(expert_num, lora_rank, hidden_size, dtype=dtype, device=device)
    up_lora_b = torch.zeros(expert_num, intermediate_size, lora_rank, dtype=dtype, device=device)

    # Down projection LoRA
    down_lora_a = torch.zeros(expert_num, lora_rank, intermediate_size, dtype=dtype, device=device)
    down_lora_b = torch.zeros(expert_num, hidden_size, lora_rank, dtype=dtype, device=device)

    # Initialize A matrices with kaiming_uniform
    for tensor in [gate_lora_a, up_lora_a, down_lora_a]:
        nn.init.kaiming_uniform_(tensor, a=math.sqrt(5))

    # B matrices remain zeros (standard LoRA initialization)

    return {
        "gate_lora_a": nn.Parameter(gate_lora_a),
        "gate_lora_b": nn.Parameter(gate_lora_b),
        "up_lora_a": nn.Parameter(up_lora_a),
        "up_lora_b": nn.Parameter(up_lora_b),
        "down_lora_a": nn.Parameter(down_lora_a),
        "down_lora_b": nn.Parameter(down_lora_b),
    }


# =============================================================================
# LoRA Experts Modules
# =============================================================================


class LoRAExpertMLP(nn.Module):
    """
    Single LoRA Expert with SwiGLU activation structure.

    This module mimics the structure of shared experts in MoE models,
    using SwiGLU (gate * silu(up)) -> down pattern.

    Initialization:
    - gate_proj and up_proj: Kaiming uniform initialization
    - down_proj: Zero initialization (ensures output = 0 at training start)
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ):
        """
        Initialize LoRA Expert MLP.

        Args:
            hidden_size: Model hidden dimension
            intermediate_size: MLP intermediate dimension (e.g., 1024)
            device: Device to place parameters on
            dtype: Data type for parameters
        """
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False, device=device, dtype=dtype)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False, device=device, dtype=dtype)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False, device=device, dtype=dtype)
        self.act_fn = nn.SiLU()

        # Zero initialize down_proj to ensure output = 0 at training start
        nn.init.zeros_(self.down_proj.weight)
        # Kaiming initialization for gate and up projections
        nn.init.kaiming_uniform_(self.gate_proj.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.up_proj.weight, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with SwiGLU activation.

        Args:
            x: Input tensor [batch, seq_len, hidden_size]

        Returns:
            Output tensor [batch, seq_len, hidden_size]
        """
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class LoRAExperts(nn.Module):
    """
    LoRA Experts module containing multiple LoRA Expert MLPs.

    Unlike routed experts, LoRA Experts process ALL tokens (no routing).
    The outputs from all experts are summed and averaged.

    This design:
    - Provides trainable capacity similar to shared experts
    - Works with frozen routed experts (CPU AMX)
    - Runs entirely on GPU for efficient training
    """

    def __init__(
        self,
        num_experts: int,
        hidden_size: int,
        intermediate_size: int,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ):
        """
        Initialize LoRA Experts module.

        Args:
            num_experts: Number of LoRA Experts
            hidden_size: Model hidden dimension
            intermediate_size: MLP intermediate dimension
            device: Device to place parameters on
            dtype: Data type for parameters
        """
        super().__init__()
        self.experts = nn.ModuleList([
            LoRAExpertMLP(hidden_size, intermediate_size, device, dtype)
            for _ in range(num_experts)
        ])
        self.num_experts = num_experts

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through all LoRA Experts.

        All experts process all tokens, and outputs are averaged.

        Args:
            hidden_states: Input tensor [batch, seq_len, hidden_size]

        Returns:
            Output tensor [batch, seq_len, hidden_size]
        """
        output = torch.zeros_like(hidden_states)
        for expert in self.experts:
            output = output + expert(hidden_states)
        return output / self.num_experts


# =============================================================================
# KTMoE Autograd Function
# =============================================================================


class KTMoEFrozenFunction(torch.autograd.Function):
    """
    Custom autograd function for KTMoEWrapper forward with frozen experts (LoRA Experts mode).

    This function:
    - Computes forward pass through routed experts (with dummy LoRA)
    - Computes backward to propagate gradients to hidden_states
    - Does NOT update any LoRA gradients (they are frozen)
    """

    @staticmethod
    def forward(
        ctx,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        wrapper: Any,
        hidden_size: int,
        num_experts_per_tok: int,
        layer_idx: int = -1,
    ) -> torch.Tensor:
        """Forward pass with frozen experts."""
        original_device = hidden_states.device
        original_dtype = hidden_states.dtype
        batch_size, seq_len, _ = hidden_states.shape

        qlen = batch_size * seq_len
        input_flat = hidden_states.view(qlen, hidden_size).to(torch.bfloat16).cpu().contiguous()
        expert_ids = topk_ids.view(qlen, num_experts_per_tok).to(torch.int64).cpu().contiguous()
        weights = topk_weights.view(qlen, num_experts_per_tok).to(torch.float32).cpu().contiguous()

        # Forward with save_for_backward=True to enable gradient propagation
        output = wrapper.forward_sft(
            hidden_states=input_flat,
            expert_ids=expert_ids,
            weights=weights,
            save_for_backward=True,  # Need this for backward gradient computation
        )

        ctx.wrapper = wrapper
        ctx.hidden_size = hidden_size
        ctx.qlen = qlen
        ctx.batch_size = batch_size
        ctx.seq_len = seq_len
        ctx.original_device = original_device
        ctx.original_dtype = original_dtype

        output = output.view(batch_size, seq_len, hidden_size)
        return output.to(device=original_device, dtype=original_dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """Backward pass - compute grad_hidden_states but ignore LoRA gradients."""
        qlen = ctx.qlen
        hidden_size = ctx.hidden_size

        grad_output_flat = grad_output.view(qlen, hidden_size).to(torch.float32).cpu().contiguous()

        # Call wrapper's backward to get grad_input
        # The grad_loras will be computed but we ignore them (frozen)
        grad_input, _ = ctx.wrapper.backward(grad_output_flat)

        grad_input = grad_input.view(ctx.batch_size, ctx.seq_len, hidden_size)
        grad_input = grad_input.to(device=ctx.original_device, dtype=ctx.original_dtype)

        # Return None for non-Tensor inputs
        # forward args: hidden_states, topk_ids, topk_weights, wrapper, hidden_size, num_experts_per_tok, layer_idx
        return grad_input, None, None, None, None, None, None


class KTMoEFunction(torch.autograd.Function):
    """
    Custom autograd function for KTMoEWrapper forward/backward.

    This bridges PyTorch autograd with kt-kernel's KTMoEWrapper implementation.
    Uses the unified SFT interface (forward_sft, backward) instead of direct C++ calls.
    """

    @staticmethod
    def forward(
        ctx,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        wrapper: Any,  # BaseSFTMoEWrapper instance
        lora_params: dict[str, nn.Parameter],
        hidden_size: int,
        num_experts_per_tok: int,
        layer_idx: int = -1,
        training: bool = True,
    ) -> torch.Tensor:
        """
        Forward pass using KTMoEWrapper.

        Args:
            ctx: Autograd context
            hidden_states: Input tensor [batch, seq_len, hidden_size]
            topk_ids: Expert indices from router [num_tokens, num_experts_per_tok]
            topk_weights: Routing weights from router [num_tokens, num_experts_per_tok]
            wrapper: BaseSFTMoEWrapper instance from KTMoEWrapper
            lora_params: LoRA parameter dictionary
            hidden_size: Hidden dimension
            num_experts_per_tok: Number of experts per token
            layer_idx: Layer index for debugging
            training: Whether in training mode (save_for_backward=True) or inference (False)

        Returns:
            Output tensor [batch, seq_len, hidden_size]
        """
        # Save original device and dtype
        original_device = hidden_states.device
        original_dtype = hidden_states.dtype
        batch_size, seq_len, _ = hidden_states.shape

        # Flatten inputs for wrapper
        qlen = batch_size * seq_len
        input_flat = hidden_states.view(qlen, hidden_size).to(torch.bfloat16).cpu().contiguous()
        expert_ids = topk_ids.view(qlen, num_experts_per_tok).to(torch.int64).cpu().contiguous()
        weights = topk_weights.view(qlen, num_experts_per_tok).to(torch.float32).cpu().contiguous()

        # Call wrapper's forward_sft
        output = wrapper.forward_sft(
            hidden_states=input_flat,
            expert_ids=expert_ids,
            weights=weights,
            save_for_backward=training,
        )

        # Save for backward
        ctx.wrapper = wrapper
        ctx.lora_params = lora_params
        ctx.hidden_size = hidden_size
        ctx.qlen = qlen
        ctx.batch_size = batch_size
        ctx.seq_len = seq_len
        ctx.original_device = original_device
        ctx.original_dtype = original_dtype

        # Reshape and return
        output = output.view(batch_size, seq_len, hidden_size)
        return output.to(device=original_device, dtype=original_dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """
        Backward pass using KTMoEWrapper.

        Args:
            ctx: Autograd context
            grad_output: Gradient from upstream [batch, seq_len, hidden_size]

        Returns:
            Tuple of gradients (grad_hidden_states, None, None, ...)
        """
        # Prepare grad_output
        qlen = ctx.qlen
        hidden_size = ctx.hidden_size

        grad_output_flat = grad_output.view(qlen, hidden_size).to(torch.float32).cpu().contiguous()

        # Call wrapper's backward
        grad_input, grad_loras = ctx.wrapper.backward(grad_output_flat)

        # Accumulate LoRA gradients to Parameters
        def accumulate_grad(param: nn.Parameter, grad: torch.Tensor):
            grad_on_device = grad.to(param.device)
            grad_fp32 = grad_on_device.float()
            if param.grad is None:
                # Keep an fp32 copy for higher-precision accumulation while satisfying autograd dtype checks.
                param._kt_grad_fp32 = grad_fp32.clone()
                param.grad = grad_fp32.to(dtype=param.dtype)
                return
            if not hasattr(param, "_kt_grad_fp32") or param._kt_grad_fp32 is None:
                param._kt_grad_fp32 = grad_fp32.clone()
            else:
                if param._kt_grad_fp32.device != param.device:
                    param._kt_grad_fp32 = param._kt_grad_fp32.to(param.device)
                param._kt_grad_fp32.add_(grad_fp32)
            if param.grad.dtype != param.dtype:
                param.grad = param.grad.to(dtype=param.dtype)
            param.grad.add_(grad_fp32.to(dtype=param.dtype))

        accumulate_grad(ctx.lora_params["gate_lora_a"], grad_loras["grad_gate_lora_a"])
        accumulate_grad(ctx.lora_params["gate_lora_b"], grad_loras["grad_gate_lora_b"])
        accumulate_grad(ctx.lora_params["up_lora_a"], grad_loras["grad_up_lora_a"])
        accumulate_grad(ctx.lora_params["up_lora_b"], grad_loras["grad_up_lora_b"])
        accumulate_grad(ctx.lora_params["down_lora_a"], grad_loras["grad_down_lora_a"])
        accumulate_grad(ctx.lora_params["down_lora_b"], grad_loras["grad_down_lora_b"])

        # Reshape grad_input and return
        grad_input = grad_input.view(ctx.batch_size, ctx.seq_len, hidden_size)
        grad_input = grad_input.to(device=ctx.original_device, dtype=ctx.original_dtype)

        # Return None for non-Tensor inputs
        # forward args: hidden_states, topk_ids, topk_weights, wrapper, lora_params, hidden_size, num_experts_per_tok, layer_idx, training
        return grad_input, None, None, None, None, None, None, None, None


# =============================================================================
# KTMoE Layer Wrapper
# =============================================================================


class KTMoELayerWrapper(nn.Module):
    """
    Wrapper for MoE layer using KTMoEWrapper.

    This replaces the original MoE layer's forward method with KTMoEWrapper implementation.
    Uses the unified KTMoEWrapper factory interface for SFT operations.

    Supports two modes:
    1. Per-expert LoRA mode: Each routed expert has its own LoRA parameters
    2. LoRA Experts mode: Separate trainable MLP modules that process all tokens
    """

    def __init__(
        self,
        original_moe: nn.Module,
        wrapper: Any,  # BaseSFTMoEWrapper instance
        lora_params: dict[str, nn.Parameter] | None,
        moe_config: MOEArchConfig,
        hidden_size: int,
        layer_idx: int,
        lora_experts: "LoRAExperts | None" = None,
    ):
        """
        Initialize KTMoE layer wrapper.

        Args:
            original_moe: Original MoE module (kept for router access)
            wrapper: BaseSFTMoEWrapper instance from KTMoEWrapper
            lora_params: LoRA parameter dictionary (None if using LoRA Experts mode)
            moe_config: MoE architecture configuration
            hidden_size: Hidden dimension
            layer_idx: Layer index
            lora_experts: LoRA Experts module (None if using per-expert LoRA mode)
        """
        super().__init__()
        # NOTE: Do NOT store original_moe as self.original_moe!
        # PEFT's get_peft_model() uses named_modules() to find Linear layers.
        # If we store original_moe, PEFT will find original_moe.experts.N.{gate,up,down}_proj
        # which have empty weights (cleared by _clear_original_expert_weights).
        # This causes NaN during training.
        # We only need router and shared_experts, which are stored separately below.

        # Marker for adapter.py to identify KT MoE wrappers
        self._is_kt_moe_wrapper = True

        self.wrapper = wrapper
        self.moe_config = moe_config
        self.hidden_size = hidden_size
        self.layer_idx = layer_idx
        self.router_type = moe_config.router_type  # "linear" or "deepseek_gate"

        # LoRA Experts mode vs per-expert LoRA mode
        self.lora_experts = lora_experts

        # Store LoRA params
        if lora_experts is not None:
            # LoRA Experts mode: store dummy lora_params (frozen, for KT wrapper compatibility)
            # These are NOT added to ParameterDict to avoid being optimized
            self._dummy_lora_params = lora_params  # Keep reference for KT wrapper
            self.lora_params = None  # No trainable per-expert LoRA
        else:
            # Per-expert LoRA mode: store as module parameters for optimizer
            self._dummy_lora_params = None
            self.lora_params = nn.ParameterDict(lora_params) if lora_params else None

        # Get router from original MoE
        self.router = getattr(original_moe, moe_config.router_attr)

        # Store shared experts if present
        if moe_config.has_shared_experts and hasattr(original_moe, "shared_experts"):
            self.shared_experts = original_moe.shared_experts
        else:
            self.shared_experts = None

        # Dirty flag for LoRA pointer updates (only update after optimizer.step)
        # Initialize to True to ensure first forward updates pointers
        self._lora_pointers_dirty = True

    def _apply(self, fn, recurse=True):
        """
        Override _apply to prevent LoRA parameters from moving to CUDA
        and update wrapper with new LoRA weight pointers.

        AMX kernel requires LoRA weights on CPU. Standard _apply would move
        all parameters to CUDA when model.to(device) is called by Trainer.

        Args:
            fn: Function to apply to tensors (e.g., lambda t: t.cuda())
            recurse: Whether to recurse into child modules

        Returns:
            self
        """
        # Apply to all other components normally (router, shared_experts, lora_experts, etc.)
        result = super()._apply(fn, recurse)

        # Per-expert LoRA mode: force LoRA params back to CPU
        if self.lora_params is not None:
            for k, v in self.lora_params.items():
                if v.data.device.type != "cpu":
                    v.data = v.data.to("cpu")

            # CRITICAL: Update wrapper with new LoRA weight pointers
            # The memory address may have changed after _apply
            self.update_lora_pointers()
            self._lora_pointers_dirty = False

        # LoRA Experts mode: dummy lora params must stay on CPU for KT wrapper
        # (they are not in ParameterDict so won't be moved, but keep this for safety)
        if self._dummy_lora_params is not None:
            for k, v in self._dummy_lora_params.items():
                if v.data.device.type != "cpu":
                    v.data = v.data.to("cpu")

        return result

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass using KTMoEWrapper.

        Supports two modes:
        1. Per-expert LoRA mode: Routed experts with per-expert LoRA on CPU AMX
        2. LoRA Experts mode: Frozen routed experts + trainable LoRA Experts on GPU

        Args:
            hidden_states: Input tensor [batch, seq_len, hidden_size]

        Returns:
            Output tensor [batch, seq_len, hidden_size]
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Get topk_ids and topk_weights based on router type
        if self.router_type == "deepseek_gate":
            # DeepSeek router expects 3D input
            router_output = self.router(hidden_states)
            if len(router_output) == 2:
                topk_ids, topk_weights = router_output
            else:
                topk_ids, topk_weights, _ = router_output  # Ignore aux_loss during inference
        else:
            # Qwen/Mixtral router is nn.Linear, expects 2D input, returns raw logits
            router_logits = self.router(hidden_states.view(-1, self.hidden_size))
            # Manually apply softmax and topk
            routing_weights = F.softmax(router_logits, dim=-1, dtype=torch.float32)
            topk_weights, topk_ids = torch.topk(
                routing_weights, self.moe_config.num_experts_per_tok, dim=-1
            )
            topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

        # Apply MoE forward based on mode
        if self.lora_experts is not None:
            # LoRA Experts mode: frozen routed experts (with dummy LoRA) + LoRA Experts
            moe_output = self._forward_frozen_experts(hidden_states, topk_ids, topk_weights)
        else:
            # Per-expert LoRA mode
            # Only update LoRA pointers when dirty (after optimizer.step or initialization)
            if self._lora_pointers_dirty:
                self.update_lora_pointers()
                self._lora_pointers_dirty = False
            moe_output = KTMoEFunction.apply(
                hidden_states,
                topk_ids,
                topk_weights,
                self.wrapper,
                dict(self.lora_params),
                self.hidden_size,
                self.moe_config.num_experts_per_tok,
                self.layer_idx,
                self.training and torch.is_grad_enabled(),  # save_for_backward: only save cache when training
            )

        # Handle shared experts if present
        if self.shared_experts is not None:
            shared_output = self.shared_experts(hidden_states)
            moe_output = moe_output + shared_output

        # Handle LoRA Experts if present
        if self.lora_experts is not None:
            lora_output = self.lora_experts(hidden_states)
            moe_output = moe_output + lora_output

        return moe_output

    def _forward_frozen_experts(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass through routed experts with frozen dummy LoRA (for LoRA Experts mode).

        Uses KTMoEFrozenFunction to enable gradient propagation to hidden_states
        while keeping the dummy LoRA parameters frozen.

        Args:
            hidden_states: Input tensor [batch, seq_len, hidden_size]
            topk_ids: Expert indices from router [num_tokens, num_experts_per_tok]
            topk_weights: Routing weights from router [num_tokens, num_experts_per_tok]

        Returns:
            Output tensor [batch, seq_len, hidden_size]
        """
        return KTMoEFrozenFunction.apply(
            hidden_states,
            topk_ids,
            topk_weights,
            self.wrapper,
            self.hidden_size,
            self.moe_config.num_experts_per_tok,
            self.layer_idx,
        )

    def update_lora_pointers(self):
        """
        Update wrapper with current LoRA weight pointers.

        This must be called after optimizer.step().
        Only applies to per-expert LoRA mode (not LoRA Experts mode).
        """
        if self.lora_params is not None and self.lora_experts is None:
            self.wrapper.update_lora_weights()


# =============================================================================
# Main Functions
# =============================================================================


def wrap_moe_layers_with_kt_wrapper(
    model: "PreTrainedModel",
    model_args: "ModelArguments",
    finetuning_args: "FinetuningArguments",
) -> list[KTMoELayerWrapper]:
    """
    Replace model's MoE layers with KTMoEWrapper-based wrappers.

    Supports two modes:
    1. Per-expert LoRA mode (default): Each routed expert has its own LoRA parameters
    2. LoRA Experts mode: Frozen routed experts + trainable LoRA Experts on GPU

    Args:
        model: HuggingFace model
        model_args: Model arguments
        finetuning_args: Finetuning arguments

    Returns:
        List of KTMoELayerWrapper instances
    """
    moe_config = get_moe_arch_config(model.config)
    hidden_size = model.config.hidden_size
    lora_rank = finetuning_args.lora_rank
    lora_alpha = finetuning_args.lora_alpha

    # Check if using LoRA Experts mode
    use_lora_experts = getattr(model_args, "kt_use_lora_experts", False)

    wrappers = []
    moe_layer_count = 0

    # Determine KT backend method
    kt_backend_map = {
        "AMXBF16": "AMXBF16_SFT",
        "AMXINT8": "AMXINT8_SFT",
        "AMXINT4": "AMXINT4_SFT",
    }
    kt_method = kt_backend_map.get(model_args.kt_backend, "AMXBF16_SFT")

    # Determine threadpool_count for TP configuration
    threadpool_count = model_args.kt_threadpool_count if model_args.kt_tp_enabled else 1

    # Check if using kt_weight_path (INT8 weights from preprocessed files)
    use_kt_weight_path = model_args.kt_weight_path is not None
    if use_kt_weight_path:
        logger.info(f"Loading INT8 weights from kt_weight_path: {model_args.kt_weight_path}")

    if use_lora_experts:
        logger.info(
            f"Using LoRA Experts mode: {model_args.kt_lora_expert_num} experts, "
            f"intermediate_size={model_args.kt_lora_expert_intermediate_size}"
        )

    # Iterate through transformer layers
    for layer_idx, layer in enumerate(model.model.layers):
        moe_module = get_moe_module(layer, moe_config)
        if moe_module is None:
            continue

        # Log layer info
        mode_str = "LoRA Experts" if use_lora_experts else "per-expert LoRA"
        logger.info(f"Wrapping MoE layer {layer_idx} with KTMoEWrapper (method={kt_method}, tp={threadpool_count}, mode={mode_str})")

        # 1. Load/Extract MoE weights
        if use_kt_weight_path:
            # Load INT8 weights from kt_weight_path
            int8_weights = load_experts_from_kt_weight_path(
                kt_weight_path=model_args.kt_weight_path,
                layer_idx=layer_idx,
                num_experts=moe_config.expert_num,
                hidden_size=hidden_size,
                intermediate_size=moe_config.intermediate_size,
            )
            gate_proj = int8_weights.gate_proj
            up_proj = int8_weights.up_proj
            down_proj = int8_weights.down_proj
        else:
            # Extract BF16 weights from model
            gate_proj, up_proj, down_proj = extract_moe_weights(moe_module, moe_config)
            # Ensure weights are on CPU and contiguous
            gate_proj = gate_proj.cpu().to(torch.bfloat16).contiguous()
            up_proj = up_proj.cpu().to(torch.bfloat16).contiguous()
            down_proj = down_proj.cpu().to(torch.bfloat16).contiguous()

        # 2. Create LoRA parameters or LoRA Experts based on mode
        if use_lora_experts:
            # LoRA Experts mode: create dummy LoRA params (not trained) for KT wrapper compatibility
            # We use lora_rank=1 (minimum) with zero-initialized B matrices so output = base + 0
            dummy_lora_rank = 1
            lora_params = create_lora_params(
                expert_num=moe_config.expert_num,
                hidden_size=hidden_size,
                intermediate_size=moe_config.intermediate_size,
                lora_rank=dummy_lora_rank,
                lora_alpha=1.0,
            )
            # Freeze the dummy LoRA params (they won't be trained)
            for param in lora_params.values():
                param.requires_grad = False

            # Create LoRA Experts module (this is what we actually train)
            lora_experts = LoRAExperts(
                num_experts=model_args.kt_lora_expert_num,
                hidden_size=hidden_size,
                intermediate_size=model_args.kt_lora_expert_intermediate_size,
                device="cuda",
                dtype=torch.bfloat16,
            )
            wrapper_lora_rank = dummy_lora_rank
            wrapper_lora_alpha = 1.0
        else:
            # Per-expert LoRA mode
            lora_params = create_lora_params(
                expert_num=moe_config.expert_num,
                hidden_size=hidden_size,
                intermediate_size=moe_config.intermediate_size,
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
            )
            lora_experts = None
            wrapper_lora_rank = lora_rank
            wrapper_lora_alpha = lora_alpha

        # 3. Create KTMoEWrapper instance (always SFT mode for training)
        wrapper = KTMoEWrapper(
            layer_idx=layer_idx,
            num_experts=moe_config.expert_num,
            num_experts_per_tok=moe_config.num_experts_per_tok,
            hidden_size=hidden_size,
            moe_intermediate_size=moe_config.intermediate_size,
            num_gpu_experts=0,  # All routed experts on CPU
            cpuinfer_threads=model_args.kt_num_threads,
            threadpool_count=threadpool_count,
            weight_path="",  # Not used when loading from tensors
            chunked_prefill_size=model_args.model_max_length or 4096,
            method=kt_method,
            mode="sft",
            lora_rank=wrapper_lora_rank,
            lora_alpha=wrapper_lora_alpha,
            max_cache_depth=getattr(model_args, "kt_max_cache_depth", 1),
        )

        # 4. Create physical-to-logical expert mapping (identity for now)
        physical_to_logical_map = torch.arange(moe_config.expert_num, dtype=torch.int64, device="cpu")

        # 5. Load base weights from tensors
        wrapper.load_weights_from_tensors(
            gate_proj=gate_proj,
            up_proj=up_proj,
            down_proj=down_proj,
            physical_to_logical_map_cpu=physical_to_logical_map,
        )

        # 6. Initialize LoRA weights in wrapper
        # For per-expert LoRA mode: these are trainable
        # For LoRA Experts mode: these are dummy (frozen) for KT wrapper compatibility
        if lora_params is not None:
            wrapper.init_lora_weights(
                gate_lora_a=lora_params["gate_lora_a"].data,
                gate_lora_b=lora_params["gate_lora_b"].data,
                up_lora_a=lora_params["up_lora_a"].data,
                up_lora_b=lora_params["up_lora_b"].data,
                down_lora_a=lora_params["down_lora_a"].data,
                down_lora_b=lora_params["down_lora_b"].data,
            )

        # 7. Create layer wrapper
        layer_wrapper = KTMoELayerWrapper(
            original_moe=moe_module,
            wrapper=wrapper,
            lora_params=lora_params,
            moe_config=moe_config,
            hidden_size=hidden_size,
            layer_idx=layer_idx,
            lora_experts=lora_experts,
        )

        # 8. Replace MoE module in layer
        setattr(layer, moe_config.moe_layer_attr, layer_wrapper)

        # Store base weights reference to prevent garbage collection
        layer_wrapper._base_weights = (gate_proj, up_proj, down_proj)

        wrappers.append(layer_wrapper)
        moe_layer_count += 1

        # Clear original HuggingFace expert weights to free memory
        _clear_original_expert_weights(moe_module, moe_config)

    mode_str = "LoRA Experts" if use_lora_experts else "per-expert LoRA"
    logger.info(f"Wrapped {moe_layer_count} MoE layers with KTMoEWrapper ({mode_str} mode)")
    return wrappers


def load_kt_model(
    config: "PretrainedConfig",
    model_args: "ModelArguments",
    finetuning_args: "FinetuningArguments",
) -> "PreTrainedModel":
    """
    Load model with KTMoEWrapper backend.

    This is the main entry point for KT model loading.

    The loading strategy avoids GPU memory overflow by controlling device_map:
    - Attention, Embedding, LM Head -> GPU
    - Router -> GPU
    - GPU experts (0 to kt_num_gpu_experts-1) -> GPU
    - CPU experts (kt_num_gpu_experts to total-1) -> CPU

    Args:
        config: HuggingFace model configuration
        model_args: Model arguments
        finetuning_args: Finetuning arguments

    Returns:
        Model with MoE layers wrapped by KTMoEWrapper

    Raises:
        KTAMXNotAvailableError: If kt_kernel is not available
        KTAMXModelNotSupportedError: If model architecture is not supported
    """
    from transformers import AutoModelForCausalLM

    from .kt_loader import get_kt_loading_kwargs

    # Validate setup
    if not KT_KERNEL_AVAILABLE:
        raise KTAMXNotAvailableError(
            "kt_kernel not found. Please install kt_kernel to enable KT MoE support."
        )

    # Check model architecture support
    _ = get_moe_arch_config(config)

    logger.info("Loading model with KTMoEWrapper backend")
    logger.info(
        f"KT config: kt_num_gpu_experts={model_args.kt_num_gpu_experts}, "
        f"kt_weight_path={model_args.kt_weight_path}, "
        f"kt_backend={model_args.kt_backend}, "
        f"kt_tp_enabled={model_args.kt_tp_enabled}"
    )

    # 1. Build loading kwargs with custom device_map
    loading_kwargs = get_kt_loading_kwargs(config, model_args)

    # 2. Load HuggingFace model with controlled device placement
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        **loading_kwargs,
    )

    # 3. Move non-expert parts to GPU (experts stay on CPU for KT AMX)
    from .kt_loader import move_non_experts_to_gpu, get_expert_device

    moe_config = get_moe_arch_config(config)
    move_non_experts_to_gpu(model, moe_config, device="cuda:0")

    # Verify expert device placement
    expert_device = get_expert_device(model, moe_config)
    logger.info(f"MoE experts on device: {expert_device}")

    # 4. Wrap MoE layers with KTMoEWrapper
    wrappers = wrap_moe_layers_with_kt_wrapper(model, model_args, finetuning_args)

    # 5. Store references on model
    model._kt_wrappers = wrappers
    model._kt_tp_enabled = model_args.kt_tp_enabled

    # 6. Collect all MoE LoRA parameters (only for per-expert LoRA mode)
    moe_lora_params = {}
    for wrapper in wrappers:
        if wrapper.lora_params is not None:
            moe_lora_params[wrapper.layer_idx] = dict(wrapper.lora_params)
    model._kt_moe_lora_params = moe_lora_params

    # Store LoRA Experts mode flag
    model._kt_use_lora_experts = getattr(model_args, "kt_use_lora_experts", False)

    logger.info("Model loaded with KTMoEWrapper backend successfully")
    return model


def get_kt_lora_params(model: "PreTrainedModel") -> list[nn.Parameter]:
    """
    Get all MoE LoRA parameters from KT model.

    This includes:
    - Per-expert LoRA parameters (in per-expert LoRA mode)
    - LoRA Experts parameters (in LoRA Experts mode)

    Args:
        model: Model with KT wrappers (can be wrapped by PeftModel)

    Returns:
        List of LoRA parameters
    """
    params = []
    # Handle PeftModel wrapping - try to find _kt_wrappers on base model
    wrappers = getattr(model, "_kt_wrappers", None)
    if wrappers is None:
        base_model = model
        for attr in ["base_model", "model"]:
            if hasattr(base_model, attr):
                base_model = getattr(base_model, attr)
                wrappers = getattr(base_model, "_kt_wrappers", None)
                if wrappers:
                    break

    if wrappers:
        for wrapper in wrappers:
            # Per-expert LoRA mode
            if wrapper.lora_params is not None:
                params.extend(wrapper.lora_params.values())
            # LoRA Experts mode
            if wrapper.lora_experts is not None:
                params.extend(wrapper.lora_experts.parameters())
    return params


def update_kt_lora_pointers(model: "PreTrainedModel"):
    """
    Mark LoRA weight pointers as dirty for all KT wrappers.

    After optimizer.step(), tensor storage addresses may change.
    This marks wrappers as needing pointer updates on next forward pass.
    The actual update is deferred to forward() to avoid redundant sync calls.

    Args:
        model: Model with KT wrappers (can be wrapped by PeftModel)
    """
    # Handle PeftModel wrapping - try to find _kt_wrappers on base model
    wrappers = getattr(model, "_kt_wrappers", None)
    if wrappers is None:
        base_model = model
        for attr in ["base_model", "model"]:
            if hasattr(base_model, attr):
                base_model = getattr(base_model, attr)
                wrappers = getattr(base_model, "_kt_wrappers", None)
                if wrappers:
                    break

    if wrappers:
        for wrapper in wrappers:
            wrapper._lora_pointers_dirty = True


def load_moe_lora_from_adapter(model: "PreTrainedModel", adapter_path: str):
    """
    Load MoE LoRA weights from PEFT adapter into KT wrappers.

    PEFT saves MoE LoRA with keys like:
    - base_model.model.model.layers.{layer}.mlp.original_moe.experts.{expert}.gate_proj.lora_A.weight

    KT expects:
    - gate_lora_a: [num_experts, lora_rank, hidden_size]
    - gate_lora_b: [num_experts, intermediate_size, lora_rank]

    Args:
        model: Model with KT wrappers
        adapter_path: Path to PEFT adapter directory
    """
    import os
    import re
    from safetensors import safe_open

    # After PeftModel.from_pretrained(), _kt_wrappers is on the inner model
    # We need to unwrap to find it
    wrappers = getattr(model, "_kt_wrappers", [])
    if not wrappers:
        # Try to find wrappers on the base model (for PeftModel case)
        base_model = model
        for attr in ["base_model", "model"]:
            if hasattr(base_model, attr):
                base_model = getattr(base_model, attr)
                wrappers = getattr(base_model, "_kt_wrappers", [])
                if wrappers:
                    logger.info(f"Found _kt_wrappers on unwrapped model ({attr})")
                    break
    if not wrappers:
        logger.warning("No KT wrappers found, skipping MoE LoRA loading")
        return

    # Build layer_idx -> wrapper mapping (only for wrappers with per-expert LoRA)
    wrapper_map = {w.layer_idx: w for w in wrappers if w.lora_params is not None}
    if not wrapper_map:
        logger.warning("No KT wrappers with per-expert LoRA found, skipping MoE LoRA loading")
        return

    # Load adapter weights
    adapter_file = os.path.join(adapter_path, "adapter_model.safetensors")
    if not os.path.exists(adapter_file):
        adapter_file = os.path.join(adapter_path, "adapter_model.bin")
        if not os.path.exists(adapter_file):
            logger.warning(f"No adapter file found at {adapter_path}")
            return

    logger.info(f"Loading MoE LoRA from {adapter_file}")

    # Parse MoE LoRA weights from adapter
    # Two key formats are supported:
    #
    # 1. KT-trained adapter format (original_moe.experts):
    #    base_model.model.model.layers.{layer}.mlp.original_moe.experts.{expert}.{proj}.lora_{A/B}.weight
    #
    # 2. Non-KT (standard PEFT) adapter format (experts with .default):
    #    base_model.model.model.layers.{layer}.mlp.experts.{expert}.{proj}.lora_{A/B}.default.weight
    #
    moe_pattern_kt = re.compile(
        r"base_model\.model\.model\.layers\.(\d+)\.mlp\.original_moe\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.lora_(A|B)\.weight"
    )
    moe_pattern_peft = re.compile(
        r"base_model\.model\.model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.lora_(A|B)\.weight"
    )

    # Group weights by layer
    layer_weights = {}  # layer_idx -> {expert_idx -> {proj -> {A/B: tensor}}}
    matched_kt_format = 0
    matched_peft_format = 0

    with safe_open(adapter_file, framework="pt") as f:
        for key in f.keys():
            # Try KT format first
            match = moe_pattern_kt.match(key)
            if match:
                matched_kt_format += 1
            else:
                # Try non-KT PEFT format
                match = moe_pattern_peft.match(key)
                if match:
                    matched_peft_format += 1
            if match:
                layer_idx = int(match.group(1))
                expert_idx = int(match.group(2))
                proj_name = match.group(3)  # gate_proj, up_proj, down_proj
                ab = match.group(4)  # A or B

                if layer_idx not in layer_weights:
                    layer_weights[layer_idx] = {}
                if expert_idx not in layer_weights[layer_idx]:
                    layer_weights[layer_idx][expert_idx] = {}
                if proj_name not in layer_weights[layer_idx][expert_idx]:
                    layer_weights[layer_idx][expert_idx][proj_name] = {}

                tensor = f.get_tensor(key)
                layer_weights[layer_idx][expert_idx][proj_name][ab] = tensor

    # Convert and load into KT wrappers
    loaded_count = 0
    for layer_idx, experts_dict in layer_weights.items():
        if layer_idx not in wrapper_map:
            logger.warning(f"No KT wrapper for layer {layer_idx}, skipping")
            continue

        wrapper = wrapper_map[layer_idx]
        num_experts = wrapper.moe_config.expert_num
        lora_rank = wrapper.lora_params["gate_lora_a"].shape[1]
        hidden_size = wrapper.hidden_size
        intermediate_size = wrapper.moe_config.intermediate_size

        # Initialize tensors for all experts
        gate_lora_a = torch.zeros(num_experts, lora_rank, hidden_size, dtype=torch.bfloat16)
        gate_lora_b = torch.zeros(num_experts, intermediate_size, lora_rank, dtype=torch.bfloat16)
        up_lora_a = torch.zeros(num_experts, lora_rank, hidden_size, dtype=torch.bfloat16)
        up_lora_b = torch.zeros(num_experts, intermediate_size, lora_rank, dtype=torch.bfloat16)
        down_lora_a = torch.zeros(num_experts, lora_rank, intermediate_size, dtype=torch.bfloat16)
        down_lora_b = torch.zeros(num_experts, hidden_size, lora_rank, dtype=torch.bfloat16)

        # Fill in from adapter weights
        for expert_idx, proj_dict in experts_dict.items():
            if expert_idx >= num_experts:
                continue

            for proj_name, ab_dict in proj_dict.items():
                # PEFT format: lora_A.weight [lora_rank, in_features]
                #              lora_B.weight [out_features, lora_rank]
                # KT format: lora_a [num_experts, lora_rank, in_features]
                #            lora_b [num_experts, out_features, lora_rank]

                if "A" in ab_dict:
                    a_tensor = ab_dict["A"].to(torch.bfloat16)  # [lora_rank, in_features]
                    if proj_name == "gate_proj":
                        gate_lora_a[expert_idx] = a_tensor
                    elif proj_name == "up_proj":
                        up_lora_a[expert_idx] = a_tensor
                    elif proj_name == "down_proj":
                        down_lora_a[expert_idx] = a_tensor

                if "B" in ab_dict:
                    b_tensor = ab_dict["B"].to(torch.bfloat16)  # [out_features, lora_rank]
                    if proj_name == "gate_proj":
                        gate_lora_b[expert_idx] = b_tensor
                    elif proj_name == "up_proj":
                        up_lora_b[expert_idx] = b_tensor
                    elif proj_name == "down_proj":
                        down_lora_b[expert_idx] = b_tensor

        # Copy to wrapper's lora_params (in-place to maintain tensor pointers)
        device = wrapper.lora_params["gate_lora_a"].device
        wrapper.lora_params["gate_lora_a"].data.copy_(gate_lora_a.to(device))
        wrapper.lora_params["gate_lora_b"].data.copy_(gate_lora_b.to(device))
        wrapper.lora_params["up_lora_a"].data.copy_(up_lora_a.to(device))
        wrapper.lora_params["up_lora_b"].data.copy_(up_lora_b.to(device))
        wrapper.lora_params["down_lora_a"].data.copy_(down_lora_a.to(device))
        wrapper.lora_params["down_lora_b"].data.copy_(down_lora_b.to(device))

        loaded_count += 1
        logger.debug(f"Loaded MoE LoRA for layer {layer_idx} ({len(experts_dict)} experts)")

    # Update wrapper pointers
    update_kt_lora_pointers(model)

    logger.info(
        f"Loaded MoE LoRA into {loaded_count} KT wrappers from {adapter_path} "
        f"(matched {matched_kt_format} KT-format keys, {matched_peft_format} PEFT-format keys)"
    )


def save_moe_lora_to_adapter(model: "PreTrainedModel", output_dir: str) -> None:
    """
    Save MoE LoRA weights to adapter file by merging with existing Attention LoRA.

    This function:
    1. Reads existing adapter_model.safetensors (contains Attention LoRA from PEFT)
    2. Converts KT MoE LoRA format to PEFT format
    3. Merges and writes back to adapter_model.safetensors

    Args:
        model: Model with KT wrappers containing MoE LoRA parameters
        output_dir: Directory containing the adapter file

    Key format conversion:
        KT format:   [num_experts, lora_rank, features] (batched)
        PEFT format: [lora_rank, features] per expert (unbatched)

    PEFT key pattern:
        base_model.model.model.layers.{layer}.mlp.original_moe.experts.{expert}.{proj}.lora_{A/B}.weight
    """
    import os
    from safetensors import safe_open
    from safetensors.torch import save_file

    # Get KT wrappers
    wrappers = getattr(model, "_kt_wrappers", [])
    if not wrappers:
        logger.warning("No KT wrappers found, skipping MoE LoRA saving")
        return

    # Read existing adapter file (Attention LoRA)
    adapter_file = os.path.join(output_dir, "adapter_model.safetensors")
    if not os.path.exists(adapter_file):
        adapter_file_bin = os.path.join(output_dir, "adapter_model.bin")
        if os.path.exists(adapter_file_bin):
            # Load from .bin file
            state_dict = torch.load(adapter_file_bin, map_location="cpu", weights_only=True)
        else:
            # No existing adapter, create empty state_dict
            logger.warning(f"No existing adapter file found at {output_dir}, creating new one")
            state_dict = {}
    else:
        # Load from safetensors
        state_dict = {}
        with safe_open(adapter_file, framework="pt") as f:
            for key in f.keys():
                state_dict[key] = f.get_tensor(key)

    # Convert and add MoE LoRA weights
    moe_lora_count = 0
    for wrapper in wrappers:
        # Skip wrappers without per-expert LoRA (LoRA Experts mode)
        if wrapper.lora_params is None:
            continue

        layer_idx = wrapper.layer_idx
        num_experts = wrapper.moe_config.expert_num

        # Get KT format tensors (already on CPU as bfloat16)
        gate_lora_a = wrapper.lora_params["gate_lora_a"].data.cpu()  # [E, r, H]
        gate_lora_b = wrapper.lora_params["gate_lora_b"].data.cpu()  # [E, I, r]
        up_lora_a = wrapper.lora_params["up_lora_a"].data.cpu()      # [E, r, H]
        up_lora_b = wrapper.lora_params["up_lora_b"].data.cpu()      # [E, I, r]
        down_lora_a = wrapper.lora_params["down_lora_a"].data.cpu()  # [E, r, I]
        down_lora_b = wrapper.lora_params["down_lora_b"].data.cpu()  # [E, H, r]

        # Convert to PEFT format (per-expert)
        for expert_idx in range(num_experts):
            base_key = f"base_model.model.model.layers.{layer_idx}.mlp.original_moe.experts.{expert_idx}"

            # Each expert's LoRA weights: [lora_rank, features] or [features, lora_rank]
            state_dict[f"{base_key}.gate_proj.lora_A.weight"] = gate_lora_a[expert_idx].clone()
            state_dict[f"{base_key}.gate_proj.lora_B.weight"] = gate_lora_b[expert_idx].clone()
            state_dict[f"{base_key}.up_proj.lora_A.weight"] = up_lora_a[expert_idx].clone()
            state_dict[f"{base_key}.up_proj.lora_B.weight"] = up_lora_b[expert_idx].clone()
            state_dict[f"{base_key}.down_proj.lora_A.weight"] = down_lora_a[expert_idx].clone()
            state_dict[f"{base_key}.down_proj.lora_B.weight"] = down_lora_b[expert_idx].clone()

            moe_lora_count += 6  # 6 tensors per expert

        logger.debug(f"Added MoE LoRA for layer {layer_idx} ({num_experts} experts)")

    # Save merged state_dict
    output_file = os.path.join(output_dir, "adapter_model.safetensors")
    save_file(state_dict, output_file, metadata={"format": "pt"})

    logger.info(
        f"Saved MoE LoRA to {output_file}: "
        f"{len(wrappers)} layers, {moe_lora_count} MoE LoRA tensors added, "
        f"{len(state_dict)} total tensors"
    )


def save_kt_moe_to_adapter(model: "PreTrainedModel", output_dir: str) -> None:
    """
    Unified function to save KT MoE weights to adapter file.

    Automatically detects the mode (per-expert LoRA or LoRA Experts) and saves accordingly.
    This function merges KT weights with existing Attention LoRA from PEFT.

    Args:
        model: Model with KT wrappers
        output_dir: Directory containing the adapter file
    """
    wrappers = getattr(model, "_kt_wrappers", [])
    if not wrappers:
        logger.warning("No KT wrappers found, skipping KT MoE saving")
        return

    # Detect mode by checking the first wrapper
    has_lora_experts = any(w.lora_experts is not None for w in wrappers)
    has_lora_params = any(w.lora_params is not None for w in wrappers)

    if has_lora_experts:
        save_lora_experts_to_adapter(model, output_dir)
    elif has_lora_params:
        save_moe_lora_to_adapter(model, output_dir)
    else:
        logger.warning("No trainable KT MoE parameters found, skipping saving")


def load_kt_moe_from_adapter(model: "PreTrainedModel", adapter_path: str) -> None:
    """
    Unified function to load KT MoE weights from adapter file.

    Automatically detects the mode (per-expert LoRA or LoRA Experts) and loads accordingly.

    Args:
        model: Model with KT wrappers
        adapter_path: Path to PEFT adapter directory
    """
    wrappers = getattr(model, "_kt_wrappers", [])
    if not wrappers:
        # Try to find wrappers on the base model (for PeftModel case)
        base_model = model
        for attr in ["base_model", "model"]:
            if hasattr(base_model, attr):
                base_model = getattr(base_model, attr)
                wrappers = getattr(base_model, "_kt_wrappers", [])
                if wrappers:
                    break
    if not wrappers:
        logger.warning("No KT wrappers found, skipping KT MoE loading")
        return

    # Detect mode by checking the wrappers
    has_lora_experts = any(w.lora_experts is not None for w in wrappers)
    has_lora_params = any(w.lora_params is not None for w in wrappers)

    if has_lora_experts:
        load_lora_experts_from_adapter(model, adapter_path)
    elif has_lora_params:
        load_moe_lora_from_adapter(model, adapter_path)
    else:
        logger.warning("No trainable KT MoE parameters found, skipping loading")


def save_lora_experts_to_adapter(model: "PreTrainedModel", output_dir: str) -> None:
    """
    Save LoRA Experts weights to adapter file by merging with existing Attention LoRA.

    This function:
    1. Reads existing adapter_model.safetensors (contains Attention LoRA from PEFT)
    2. Adds LoRA Experts weights
    3. Merges and writes back to adapter_model.safetensors

    Args:
        model: Model with KT wrappers containing LoRA Experts
        output_dir: Directory containing the adapter file

    Key pattern for LoRA Experts:
        base_model.model.model.layers.{layer}.mlp.lora_experts.{expert_idx}.{gate_proj|up_proj|down_proj}.weight
    """
    import os
    from safetensors import safe_open
    from safetensors.torch import save_file

    # Get KT wrappers
    wrappers = getattr(model, "_kt_wrappers", [])
    if not wrappers:
        logger.warning("No KT wrappers found, skipping LoRA Experts saving")
        return

    # Read existing adapter file (Attention LoRA)
    adapter_file = os.path.join(output_dir, "adapter_model.safetensors")
    if not os.path.exists(adapter_file):
        adapter_file_bin = os.path.join(output_dir, "adapter_model.bin")
        if os.path.exists(adapter_file_bin):
            state_dict = torch.load(adapter_file_bin, map_location="cpu", weights_only=True)
        else:
            logger.warning(f"No existing adapter file found at {output_dir}, creating new one")
            state_dict = {}
    else:
        state_dict = {}
        with safe_open(adapter_file, framework="pt") as f:
            for key in f.keys():
                state_dict[key] = f.get_tensor(key)

    # Add LoRA Experts weights
    lora_expert_count = 0
    for wrapper in wrappers:
        if wrapper.lora_experts is None:
            continue

        layer_idx = wrapper.layer_idx
        for expert_idx, expert in enumerate(wrapper.lora_experts.experts):
            base_key = f"base_model.model.model.layers.{layer_idx}.mlp.lora_experts.{expert_idx}"

            state_dict[f"{base_key}.gate_proj.weight"] = expert.gate_proj.weight.data.cpu().clone()
            state_dict[f"{base_key}.up_proj.weight"] = expert.up_proj.weight.data.cpu().clone()
            state_dict[f"{base_key}.down_proj.weight"] = expert.down_proj.weight.data.cpu().clone()

            lora_expert_count += 3  # 3 tensors per expert

        logger.debug(f"Added LoRA Experts for layer {layer_idx} ({len(wrapper.lora_experts.experts)} experts)")

    # Save merged state_dict
    output_file = os.path.join(output_dir, "adapter_model.safetensors")
    save_file(state_dict, output_file, metadata={"format": "pt"})

    logger.info(
        f"Saved LoRA Experts to {output_file}: "
        f"{len(wrappers)} layers, {lora_expert_count} LoRA Expert tensors added, "
        f"{len(state_dict)} total tensors"
    )


def load_lora_experts_from_adapter(model: "PreTrainedModel", adapter_path: str) -> None:
    """
    Load LoRA Experts weights from adapter file into KT wrappers.

    Args:
        model: Model with KT wrappers containing LoRA Experts
        adapter_path: Path to PEFT adapter directory

    Key pattern for LoRA Experts:
        base_model.model.model.layers.{layer}.mlp.lora_experts.{expert_idx}.{gate_proj|up_proj|down_proj}.weight
    """
    import os
    import re
    from safetensors import safe_open

    # Get KT wrappers
    wrappers = getattr(model, "_kt_wrappers", [])
    if not wrappers:
        base_model = model
        for attr in ["base_model", "model"]:
            if hasattr(base_model, attr):
                base_model = getattr(base_model, attr)
                wrappers = getattr(base_model, "_kt_wrappers", [])
                if wrappers:
                    break
    if not wrappers:
        logger.warning("No KT wrappers found, skipping LoRA Experts loading")
        return

    # Build layer_idx -> wrapper mapping
    wrapper_map = {w.layer_idx: w for w in wrappers if w.lora_experts is not None}
    if not wrapper_map:
        logger.warning("No LoRA Experts found in KT wrappers, skipping")
        return

    # Load adapter weights
    adapter_file = os.path.join(adapter_path, "adapter_model.safetensors")
    if not os.path.exists(adapter_file):
        adapter_file = os.path.join(adapter_path, "adapter_model.bin")
        if not os.path.exists(adapter_file):
            logger.warning(f"No adapter file found at {adapter_path}")
            return

    logger.info(f"Loading LoRA Experts from {adapter_file}")

    # Pattern for LoRA Experts keys
    lora_expert_pattern = re.compile(
        r"base_model\.model\.model\.layers\.(\d+)\.mlp\.lora_experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight"
    )

    # Group weights by layer
    layer_weights = {}  # layer_idx -> {expert_idx -> {proj -> tensor}}
    matched_count = 0

    with safe_open(adapter_file, framework="pt") as f:
        for key in f.keys():
            match = lora_expert_pattern.match(key)
            if match:
                layer_idx = int(match.group(1))
                expert_idx = int(match.group(2))
                proj_name = match.group(3)

                if layer_idx not in layer_weights:
                    layer_weights[layer_idx] = {}
                if expert_idx not in layer_weights[layer_idx]:
                    layer_weights[layer_idx][expert_idx] = {}

                layer_weights[layer_idx][expert_idx][proj_name] = f.get_tensor(key)
                matched_count += 1

    # Load into KT wrappers
    loaded_count = 0
    for layer_idx, experts_dict in layer_weights.items():
        if layer_idx not in wrapper_map:
            logger.warning(f"No KT wrapper with LoRA Experts for layer {layer_idx}, skipping")
            continue

        wrapper = wrapper_map[layer_idx]
        for expert_idx, proj_dict in experts_dict.items():
            if expert_idx >= len(wrapper.lora_experts.experts):
                logger.warning(f"Expert index {expert_idx} out of range for layer {layer_idx}, skipping")
                continue

            expert = wrapper.lora_experts.experts[expert_idx]

            if "gate_proj" in proj_dict:
                expert.gate_proj.weight.data.copy_(proj_dict["gate_proj"].to(expert.gate_proj.weight.device))
            if "up_proj" in proj_dict:
                expert.up_proj.weight.data.copy_(proj_dict["up_proj"].to(expert.up_proj.weight.device))
            if "down_proj" in proj_dict:
                expert.down_proj.weight.data.copy_(proj_dict["down_proj"].to(expert.down_proj.weight.device))

        loaded_count += 1
        logger.debug(f"Loaded LoRA Experts for layer {layer_idx} ({len(experts_dict)} experts)")

    logger.info(
        f"Loaded LoRA Experts into {loaded_count} KT wrappers from {adapter_path} "
        f"(matched {matched_count} keys)"
    )
