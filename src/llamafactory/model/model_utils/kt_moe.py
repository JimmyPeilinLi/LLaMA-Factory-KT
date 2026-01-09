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
KTransformers MoE Backend Integration

This module provides KT acceleration for MoE layers in large language models.
It replaces MoE layer forward/backward with CPU-accelerated implementations,
while keeping Attention layers on GPU with native HuggingFace implementation.

Key features:
- Only handles MoE layers (no kt_optimize_rule YAML required)
- Zero-copy design for LoRA weights
- Support for forward/backward with gradient computation
- LoRA handled in C++ kernel for MoE layers
"""

from __future__ import annotations

import importlib.util as _u
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...extras import logging
from .kt_loader import load_experts_from_kt_weight_path, INT8ExpertWeights


if TYPE_CHECKING:
    from transformers import PretrainedConfig, PreTrainedModel

    from ...hparams import FinetuningArguments, ModelArguments


logger = logging.get_logger(__name__)

# Check if kt_kernel is available
KT_KERNEL_AVAILABLE = _u.find_spec("kt_kernel") is not None

if KT_KERNEL_AVAILABLE:
    try:
        from kt_kernel import kt_kernel_ext
    except ImportError:
        KT_KERNEL_AVAILABLE = False
        kt_kernel_ext = None


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
# CPUInfer Management
# =============================================================================


def init_cpu_infer(model_args: "ModelArguments") -> Any:
    """
    Initialize KT CPUInfer instance.

    Args:
        model_args: Model arguments containing KT AMX configuration

    Returns:
        CPUInfer instance

    Raises:
        KTAMXNotAvailableError: If kt_kernel is not available
    """
    if not KT_KERNEL_AVAILABLE:
        raise KTAMXNotAvailableError(
            "kt_kernel not found. Please install kt_kernel to enable KT MoE support."
        )

    num_threads = model_args.kt_num_threads

    if model_args.kt_tp_enabled:
        # TP mode: automatic NUMA partitioning
        logger.info(f"Creating CPUInfer with TP enabled, {num_threads} threads")
        cpu_infer = kt_kernel_ext.CPUInfer(num_threads)
    else:
        # Single NUMA mode
        logger.info(f"Creating CPUInfer without TP, {num_threads} threads")
        pool_config = kt_kernel_ext.WorkerPoolConfig()
        pool_config.subpool_count = 1
        pool_config.subpool_numa_map = [0]
        pool_config.subpool_thread_count = [num_threads]
        cpu_infer = kt_kernel_ext.CPUInfer(pool_config)

    return cpu_infer


# =============================================================================
# MOE AMX Function (Custom Autograd)
# =============================================================================


class MOEAMXFunction(torch.autograd.Function):
    """
    Custom autograd function for AMX MOE forward/backward.

    This bridges PyTorch autograd with KT AMX C++ implementation.
    """

    @staticmethod
    def forward(
        ctx,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        moe_amx: Any,
        cpu_infer: Any,
        lora_params: dict[str, nn.Parameter],
        moe_config: MOEArchConfig,
        hidden_size: int,
        num_experts_per_tok: int,
    ) -> torch.Tensor:
        """
        Forward pass using AMX operator.

        Args:
            ctx: Autograd context
            hidden_states: Input tensor [batch, seq_len, hidden_size]
            topk_ids: Expert indices from router [num_tokens, num_experts_per_tok]
            topk_weights: Routing weights from router [num_tokens, num_experts_per_tok]
            moe_amx: AMXBF16_SFT_MOE or AMXInt8_SFT_MOE instance
            cpu_infer: CPUInfer instance
            lora_params: LoRA parameter dictionary
            moe_config: MoE architecture config
            hidden_size: Hidden dimension
            num_experts_per_tok: Number of experts per token

        Returns:
            Output tensor [batch, seq_len, hidden_size]
        """
        # Save original device and dtype
        original_device = hidden_states.device
        original_dtype = hidden_states.dtype
        batch_size, seq_len, _ = hidden_states.shape

        # 1. Flatten topk results for AMX (routing already done by caller)
        qlen = batch_size * seq_len
        expert_ids = topk_ids.view(qlen, num_experts_per_tok).to(torch.int64).cpu().contiguous()
        weights = topk_weights.view(qlen, num_experts_per_tok).to(torch.float32).cpu().contiguous()

        # 2. Prepare input
        input_data = hidden_states.view(qlen, hidden_size).to(torch.bfloat16).cpu().contiguous()
        output = torch.zeros((qlen, hidden_size), dtype=torch.bfloat16, device="cpu").contiguous()

        # 3. Batch size tensor
        bsz_tensor = torch.tensor([qlen], device="cpu")

        # 4. Call AMX forward
        cpu_infer.submit(
            moe_amx.forward_sft_task(
                bsz_tensor.data_ptr(),
                num_experts_per_tok,
                expert_ids.data_ptr(),
                weights.data_ptr(),
                input_data.data_ptr(),
                output.data_ptr(),
                True,  # save_for_backward
            )
        )
        cpu_infer.sync()

        # 5. Save for backward
        ctx.moe_amx = moe_amx
        ctx.cpu_infer = cpu_infer
        ctx.lora_params = lora_params
        ctx.hidden_size = hidden_size
        ctx.qlen = qlen
        ctx.batch_size = batch_size
        ctx.seq_len = seq_len
        ctx.original_device = original_device
        ctx.original_dtype = original_dtype

        # 6. Reshape and return
        output = output.view(batch_size, seq_len, hidden_size)
        return output.to(device=original_device, dtype=original_dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """
        Backward pass using AMX operator.

        Args:
            ctx: Autograd context
            grad_output: Gradient from upstream [batch, seq_len, hidden_size]

        Returns:
            Tuple of gradients (grad_hidden_states, None, None, ...)
        """
        # 1. Prepare grad_output
        qlen = ctx.qlen
        hidden_size = ctx.hidden_size

        grad_output_flat = grad_output.view(qlen, hidden_size).to(torch.bfloat16).cpu().contiguous()

        # 2. Allocate gradient buffers
        grad_input = torch.zeros((qlen, hidden_size), dtype=torch.bfloat16, device="cpu").contiguous()

        # BUG-007 fix: 梯度 tensor 必须在 CPU 上，因为 AMX C++ 代码需要 CPU 内存访问
        # torch.zeros_like() 会继承原 tensor 的 device，如果 LoRA 参数在 GPU 上，
        # 梯度也会在 GPU 上，导致 C++ 代码访问 GPU 内存时 SIGSEGV
        grad_gate_lora_a = torch.zeros_like(ctx.lora_params["gate_lora_a"].data, device="cpu")
        grad_gate_lora_b = torch.zeros_like(ctx.lora_params["gate_lora_b"].data, device="cpu")
        grad_up_lora_a = torch.zeros_like(ctx.lora_params["up_lora_a"].data, device="cpu")
        grad_up_lora_b = torch.zeros_like(ctx.lora_params["up_lora_b"].data, device="cpu")
        grad_down_lora_a = torch.zeros_like(ctx.lora_params["down_lora_a"].data, device="cpu")
        grad_down_lora_b = torch.zeros_like(ctx.lora_params["down_lora_b"].data, device="cpu")

        # 3. Call AMX backward
        ctx.cpu_infer.submit(
            ctx.moe_amx.backward_task(
                grad_output_flat.data_ptr(),
                grad_input.data_ptr(),
                grad_gate_lora_a.data_ptr(),
                grad_gate_lora_b.data_ptr(),
                grad_up_lora_a.data_ptr(),
                grad_up_lora_b.data_ptr(),
                grad_down_lora_a.data_ptr(),
                grad_down_lora_b.data_ptr(),
            )
        )
        ctx.cpu_infer.sync()

        # 4. Accumulate LoRA gradients to Parameters
        # BUG-007 fix: 梯度在 CPU 上计算（AMX 需要），但 param 在 GPU 上
        # 需要将梯度移动到 param 所在的设备
        def accumulate_grad(param: nn.Parameter, grad: torch.Tensor):
            grad_on_device = grad.to(param.device)  # CPU → GPU (方案 A)
            if param.grad is None:
                param.grad = grad_on_device.clone()
            else:
                param.grad.add_(grad_on_device)

        accumulate_grad(ctx.lora_params["gate_lora_a"], grad_gate_lora_a)
        accumulate_grad(ctx.lora_params["gate_lora_b"], grad_gate_lora_b)
        accumulate_grad(ctx.lora_params["up_lora_a"], grad_up_lora_a)
        accumulate_grad(ctx.lora_params["up_lora_b"], grad_up_lora_b)
        accumulate_grad(ctx.lora_params["down_lora_a"], grad_down_lora_a)
        accumulate_grad(ctx.lora_params["down_lora_b"], grad_down_lora_b)

        # 5. Reshape grad_input and return
        grad_input = grad_input.view(ctx.batch_size, ctx.seq_len, hidden_size)
        grad_input = grad_input.to(device=ctx.original_device, dtype=ctx.original_dtype)

        # Return None for non-Tensor inputs
        # forward args: hidden_states, topk_ids, topk_weights, moe_amx, cpu_infer, lora_params, moe_config, hidden_size, num_experts_per_tok
        return grad_input, None, None, None, None, None, None, None, None


# =============================================================================
# MOE Layer Wrapper
# =============================================================================


class MOELayerWrapper(nn.Module):
    """
    Wrapper for MoE layer with AMX acceleration.

    This replaces the original MoE layer's forward method with AMX implementation.
    """

    def __init__(
        self,
        original_moe: nn.Module,
        moe_amx: Any,
        cpu_infer: Any,
        lora_params: dict[str, nn.Parameter],
        moe_config: MOEArchConfig,
        hidden_size: int,
        layer_idx: int,
    ):
        """
        Initialize MOE layer wrapper.

        Args:
            original_moe: Original MoE module (kept for router access)
            moe_amx: AMXBF16_SFT_MOE or AMXInt8_SFT_MOE instance
            cpu_infer: CPUInfer instance
            lora_params: LoRA parameter dictionary
            moe_config: MoE architecture configuration
            hidden_size: Hidden dimension
            layer_idx: Layer index
        """
        super().__init__()
        self.original_moe = original_moe
        self.moe_amx = moe_amx
        self.cpu_infer = cpu_infer
        self.moe_config = moe_config
        self.hidden_size = hidden_size
        self.layer_idx = layer_idx
        self.router_type = moe_config.router_type  # "linear" or "deepseek_gate"

        # Store LoRA params as module parameters for optimizer
        self.lora_params = nn.ParameterDict(lora_params)

        # Get router from original MoE
        self.router = getattr(original_moe, moe_config.router_attr)

        # Store shared experts if present
        if moe_config.has_shared_experts and hasattr(original_moe, "shared_experts"):
            self.shared_experts = original_moe.shared_experts
        else:
            self.shared_experts = None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass using AMX acceleration.

        Args:
            hidden_states: Input tensor [batch, seq_len, hidden_size]

        Returns:
            Output tensor [batch, seq_len, hidden_size]
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Get topk_ids and topk_weights based on router type
        if self.router_type == "deepseek_gate":
            # DeepSeek router expects 3D input and returns (topk_idx, topk_weight, aux_loss)
            topk_ids, topk_weights, aux_loss = self.router(hidden_states)
        else:
            # Qwen/Mixtral router is nn.Linear, expects 2D input, returns raw logits
            router_logits = self.router(hidden_states.view(-1, self.hidden_size))
            # Manually apply softmax and topk
            routing_weights = F.softmax(router_logits, dim=-1, dtype=torch.float32)
            topk_weights, topk_ids = torch.topk(
                routing_weights, self.moe_config.num_experts_per_tok, dim=-1
            )
            topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

        # Apply AMX forward with unified interface
        moe_output = MOEAMXFunction.apply(
            hidden_states,
            topk_ids,
            topk_weights,
            self.moe_amx,
            self.cpu_infer,
            dict(self.lora_params),
            self.moe_config,
            self.hidden_size,
            self.moe_config.num_experts_per_tok,
        )

        # Handle shared experts if present
        if self.shared_experts is not None:
            shared_output = self.shared_experts(hidden_states)
            moe_output = moe_output + shared_output

        return moe_output

    def update_lora_pointers(self):
        """
        Update AMX operator with current LoRA weight pointers.

        This must be called after optimizer.step() in TP mode.
        """
        self.cpu_infer.submit(
            self.moe_amx.update_lora_weights_task(
                self.lora_params["gate_lora_a"].data.data_ptr(),
                self.lora_params["gate_lora_b"].data.data_ptr(),
                self.lora_params["up_lora_a"].data.data_ptr(),
                self.lora_params["up_lora_b"].data.data_ptr(),
                self.lora_params["down_lora_a"].data.data_ptr(),
                self.lora_params["down_lora_b"].data.data_ptr(),
            )
        )
        self.cpu_infer.sync()


# =============================================================================
# Main Functions
# =============================================================================


def wrap_moe_layers_with_amx(
    model: "PreTrainedModel",
    model_args: "ModelArguments",
    finetuning_args: "FinetuningArguments",
    cpu_infer: Any,
) -> list[MOELayerWrapper]:
    """
    Replace model's MoE layers with AMX-accelerated wrappers.

    Args:
        model: HuggingFace model
        model_args: Model arguments
        finetuning_args: Finetuning arguments
        cpu_infer: CPUInfer instance

    Returns:
        List of MOELayerWrapper instances
    """
    moe_config = get_moe_arch_config(model.config)
    hidden_size = model.config.hidden_size
    lora_rank = finetuning_args.lora_rank
    lora_alpha = finetuning_args.lora_alpha

    wrappers = []
    moe_layer_count = 0

    # Determine AMX backend class
    if model_args.kt_backend == "AMXBF16":
        AMX_MOE_CLASS = kt_kernel_ext.moe.AMXBF16_SFT_MOE
    else:
        AMX_MOE_CLASS = kt_kernel_ext.moe.AMXInt8_SFT_MOE

    # Check if using kt_weight_path (INT8 weights from preprocessed files)
    use_kt_weight_path = model_args.kt_weight_path is not None
    if use_kt_weight_path:
        logger.info(f"Loading INT8 weights from kt_weight_path: {model_args.kt_weight_path}")

    # Iterate through transformer layers
    for layer_idx, layer in enumerate(model.model.layers):
        moe_module = get_moe_module(layer, moe_config)
        if moe_module is None:
            continue

        # Log layer info
        if use_kt_weight_path:
            logger.info(
                f"Wrapping MoE layer {layer_idx} with KT AMX (loading INT8 from kt_weight_path)"
            )
        else:
            # Log the device of the first expert's weights
            first_expert = getattr(moe_module, moe_config.experts_attr)[0]
            first_weight = getattr(first_expert, moe_config.weight_names[0]).weight
            logger.info(
                f"Wrapping MoE layer {layer_idx} with KT AMX "
                f"(expert weights on {first_weight.device})"
            )

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
            gate_scale = int8_weights.gate_scale
            up_scale = int8_weights.up_scale
            down_scale = int8_weights.down_scale
        else:
            # Extract BF16 weights from model
            gate_proj, up_proj, down_proj = extract_moe_weights(moe_module, moe_config)
            # Ensure weights are on CPU and contiguous
            # With the new device_map, experts should already be on CPU,
            # so .cpu() is typically a no-op here
            gate_proj = gate_proj.cpu().to(torch.bfloat16).contiguous()
            up_proj = up_proj.cpu().to(torch.bfloat16).contiguous()
            down_proj = down_proj.cpu().to(torch.bfloat16).contiguous()
            gate_scale = None
            up_scale = None
            down_scale = None

        # 2. Create LoRA parameters (always BF16)
        lora_params = create_lora_params(
            expert_num=moe_config.expert_num,
            hidden_size=hidden_size,
            intermediate_size=moe_config.intermediate_size,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
        )

        # 3. Create MOESFTConfig
        config = kt_kernel_ext.moe.MOESFTConfig()
        config.expert_num = moe_config.expert_num
        config.num_experts_per_tok = moe_config.num_experts_per_tok
        config.hidden_size = hidden_size
        config.intermediate_size = moe_config.intermediate_size
        config.lora_rank = lora_rank
        config.lora_alpha = lora_alpha
        config.max_cache_depth = getattr(model_args, "kt_max_cache_depth", 1)
        config.max_len = model_args.model_max_length or 4096
        config.layer_idx = layer_idx

        # Set base weight pointers
        config.gate_proj = gate_proj.data_ptr()
        config.up_proj = up_proj.data_ptr()
        config.down_proj = down_proj.data_ptr()

        # Set scale pointers for INT8 mode
        if use_kt_weight_path and gate_scale is not None:
            config.gate_scale = gate_scale.data_ptr()
            config.up_scale = up_scale.data_ptr()
            config.down_scale = down_scale.data_ptr()

        # Set LoRA weight pointers (zero-copy)
        config.gate_lora_a = lora_params["gate_lora_a"].data.data_ptr()
        config.gate_lora_b = lora_params["gate_lora_b"].data.data_ptr()
        config.up_lora_a = lora_params["up_lora_a"].data.data_ptr()
        config.up_lora_b = lora_params["up_lora_b"].data.data_ptr()
        config.down_lora_a = lora_params["down_lora_a"].data.data_ptr()
        config.down_lora_b = lora_params["down_lora_b"].data.data_ptr()

        # Set thread pool
        config.pool = cpu_infer.backend_

        # 4. Create AMX MOE instance
        moe_amx = AMX_MOE_CLASS(config)

        # 5. Load base weights
        cpu_infer.submit(moe_amx.load_weights_task())
        cpu_infer.sync()

        # 6. Warm up
        cpu_infer.submit(moe_amx.warm_up_task())
        cpu_infer.sync()

        # 7. Create wrapper
        wrapper = MOELayerWrapper(
            original_moe=moe_module,
            moe_amx=moe_amx,
            cpu_infer=cpu_infer,
            lora_params=lora_params,
            moe_config=moe_config,
            hidden_size=hidden_size,
            layer_idx=layer_idx,
        )

        # 8. Replace MoE module in layer
        setattr(layer, moe_config.moe_layer_attr, wrapper)

        # Store base weights reference to prevent garbage collection
        if use_kt_weight_path:
            wrapper._base_weights = (gate_proj, up_proj, down_proj)
            wrapper._base_scales = (gate_scale, up_scale, down_scale)
        else:
            wrapper._base_weights = (gate_proj, up_proj, down_proj)

        wrappers.append(wrapper)
        moe_layer_count += 1

        # Clear original HuggingFace expert weights to free memory
        # This is safe because:
        # - BF16 mode: weights were already extracted and stored in wrapper._base_weights
        # - INT8 mode: weights were loaded from kt_weight_path, HuggingFace weights are not needed
        _clear_original_expert_weights(moe_module, moe_config)

    logger.info(f"Wrapped {moe_layer_count} MoE layers with KT AMX")
    return wrappers


def load_kt_model(
    config: "PretrainedConfig",
    model_args: "ModelArguments",
    finetuning_args: "FinetuningArguments",
) -> "PreTrainedModel":
    """
    Load model with KT AMX MOE backend.

    This is the main entry point for KT AMX model loading.

    The loading strategy avoids GPU memory overflow by controlling device_map:
    - Attention, Embedding, LM Head -> GPU
    - Router -> GPU
    - GPU experts (0 to kt_num_gpu_experts-1) -> GPU
    - CPU experts (kt_num_gpu_experts to total-1) -> CPU

    Note: We always load expert weights to CPU (never use "meta" device) because
    HuggingFace Trainer's dispatch_model() cannot handle meta tensors. After
    wrapping with AMX, the original HuggingFace weights are cleared to save memory.

    Args:
        config: HuggingFace model configuration
        model_args: Model arguments
        finetuning_args: Finetuning arguments

    Returns:
        Model with MoE layers wrapped by KT AMX

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

    logger.info("Loading model with KT AMX MOE backend")
    logger.info(
        f"KT config: kt_num_gpu_experts={model_args.kt_num_gpu_experts}, "
        f"kt_weight_path={model_args.kt_weight_path}, "
        f"kt_backend={model_args.kt_backend}"
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

    # 4. Initialize CPUInfer
    cpu_infer = init_cpu_infer(model_args)

    # 5. Wrap MoE layers with AMX
    wrappers = wrap_moe_layers_with_amx(model, model_args, finetuning_args, cpu_infer)

    # 6. Store references on model
    model._kt_wrappers = wrappers
    model._kt_cpu_infer = cpu_infer
    model._kt_tp_enabled = model_args.kt_tp_enabled

    # 7. Collect all MoE LoRA parameters
    moe_lora_params = {}
    for wrapper in wrappers:
        moe_lora_params[wrapper.layer_idx] = dict(wrapper.lora_params)
    model._kt_moe_lora_params = moe_lora_params

    logger.info("Model loaded with KT AMX MOE backend successfully")
    return model


def get_kt_lora_params(model: "PreTrainedModel") -> list[nn.Parameter]:
    """
    Get all MoE LoRA parameters from KT AMX model.

    Args:
        model: Model with KT AMX wrappers

    Returns:
        List of LoRA parameters
    """
    params = []
    if hasattr(model, "_kt_wrappers"):
        for wrapper in model._kt_wrappers:
            params.extend(wrapper.lora_params.values())
    return params


def update_kt_lora_pointers(model: "PreTrainedModel"):
    """
    Update LoRA weight pointers for all AMX wrappers.

    Must be called after optimizer.step() in TP mode.

    Args:
        model: Model with KT AMX wrappers
    """
    if hasattr(model, "_kt_wrappers"):
        for wrapper in model._kt_wrappers:
            wrapper.update_lora_pointers()
