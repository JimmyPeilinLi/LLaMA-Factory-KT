#!/usr/bin/env python
# coding=utf-8
"""
Debug script for MoE backward pass comparison.

This script:
1. Creates a single MoE layer with known weights
2. Runs forward and backward through KT backend
3. Runs forward and backward through pure PyTorch
4. Compares grad_input at each stage to identify where error is introduced

Usage:
    # Enable C++ dump first by editing sft_moe.hpp line 153: remove "return false;"
    python scripts/debug_moe_backward.py --dump-dir ./backward_debug

Environment variables:
    SFT_MOE_DUMP=1          Enable C++ intermediate value dump
    SFT_MOE_DUMP_DIR=dir    Directory for C++ dumps
"""

import os
import sys
import struct
import argparse
import shutil
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# Try to import kt_kernel
try:
    from kt_kernel.experts import KTMoEWrapper
    HAS_KT_KERNEL = True
except ImportError:
    HAS_KT_KERNEL = False
    print("WARNING: kt_kernel not available")


# ============================================================================
# Configuration
# ============================================================================
DEFAULT_CONFIG = {
    "expert_num": 8,
    "hidden_size": 2048,
    "intermediate_size": 1024,
    "lora_rank": 8,
    "lora_alpha": 16,
    "qlen": 32,
    "k": 2,  # top-k experts
    "num_threads": 8,
    "tp_count": 2,  # No TP for simpler debugging
}

# Weight scaling for numerical stability
WEIGHT_SCALE = 0.01
INPUT_SCALE = 0.1
GRAD_SCALE = 0.01


# ============================================================================
# Dump Utilities
# ============================================================================

def read_matrix_file(filepath: str) -> tuple:
    """Read binary matrix file in the format: rows(int32), cols(int32), data(float32)"""
    if not os.path.exists(filepath):
        return None, None, None

    with open(filepath, "rb") as f:
        rows, cols = struct.unpack("ii", f.read(8))
        data = np.frombuffer(f.read(rows * cols * 4), dtype=np.float32)
        data = data.reshape(rows, cols)
    return rows, cols, data


def save_matrix_file(filepath: str, data: np.ndarray):
    """Save matrix to binary file"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    if len(data.shape) == 1:
        rows, cols = 1, data.shape[0]
        data = data.reshape(1, -1)
    else:
        rows, cols = data.shape

    with open(filepath, "wb") as f:
        f.write(np.array([rows, cols], dtype=np.int32).tobytes())
        f.write(data.astype(np.float32).tobytes())


def compare_tensors(t1: torch.Tensor, t2: torch.Tensor, name: str, threshold: float = 0.01):
    """Compare two tensors and print detailed statistics"""
    t1 = t1.float().detach()
    t2 = t2.float().detach()

    if t1.shape != t2.shape:
        print(f"[{name}] SHAPE MISMATCH: {t1.shape} vs {t2.shape}")
        return False

    diff = (t1 - t2).abs()

    # Statistics
    abs_max = diff.max().item()
    abs_mean = diff.mean().item()

    # Cosine similarity
    cos_sim = F.cosine_similarity(t1.flatten().unsqueeze(0), t2.flatten().unsqueeze(0)).item()

    # Relative error
    t1_abs_mean = t1.abs().mean().item() + 1e-12
    rel_error = abs_mean / t1_abs_mean

    passed = rel_error < threshold and cos_sim > 0.99
    status = "\033[92mPASS\033[0m" if passed else "\033[91mFAIL\033[0m"

    print(f"[{name}] {status}")
    print(f"    shape: {list(t1.shape)}")
    print(f"    abs_max: {abs_max:.6e}, abs_mean: {abs_mean:.6e}")
    print(f"    cos_sim: {cos_sim:.6f}, rel_error: {rel_error:.6e}")
    print(f"    t1 stats: mean={t1.mean().item():.6e}, std={t1.std().item():.6e}, "
          f"min={t1.min().item():.6e}, max={t1.max().item():.6e}")
    print(f"    t2 stats: mean={t2.mean().item():.6e}, std={t2.std().item():.6e}, "
          f"min={t2.min().item():.6e}, max={t2.max().item():.6e}")

    # Show first few values
    flat1 = t1.flatten()[:10]
    flat2 = t2.flatten()[:10]
    print(f"    first 10 t1: [{', '.join([f'{v:.6f}' for v in flat1.tolist()])}]")
    print(f"    first 10 t2: [{', '.join([f'{v:.6f}' for v in flat2.tolist()])}]")

    return passed


# ============================================================================
# PyTorch Reference Implementation
# ============================================================================

def silu(x):
    """SiLU activation function"""
    return x * torch.sigmoid(x)


def silu_backward(gate_out, up_out, grad_intermediate):
    """
    Backward pass for SiLU activation: act_out = silu(gate_out) * up_out

    Returns:
        grad_gate_out: gradient w.r.t. gate_out
        grad_up_out: gradient w.r.t. up_out
    """
    sigmoid_gate = torch.sigmoid(gate_out)
    silu_gate = gate_out * sigmoid_gate

    # grad_up_out = grad_intermediate * silu(gate_out)
    grad_up_out = grad_intermediate * silu_gate

    # grad_gate_out = grad_intermediate * up_out * silu'(gate_out)
    # silu'(x) = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
    silu_grad = sigmoid_gate * (1 + gate_out - gate_out * sigmoid_gate)
    grad_gate_out = grad_intermediate * up_out * silu_grad

    return grad_gate_out, grad_up_out


class PyTorchMoEReference:
    """Pure PyTorch MoE implementation for reference"""

    def __init__(self, gate_proj, up_proj, down_proj,
                 gate_lora_a, gate_lora_b, up_lora_a, up_lora_b,
                 down_lora_a, down_lora_b, lora_scaling, config):
        self.gate_proj = gate_proj.float()
        self.up_proj = up_proj.float()
        self.down_proj = down_proj.float()
        self.gate_lora_a = gate_lora_a.float()
        self.gate_lora_b = gate_lora_b.float()
        self.up_lora_a = up_lora_a.float()
        self.up_lora_b = up_lora_b.float()
        self.down_lora_a = down_lora_a.float()
        self.down_lora_b = down_lora_b.float()
        self.lora_scaling = lora_scaling
        self.config = config

        # Forward cache for backward
        self.cache = {}

    def forward(self, input_tensor, expert_ids, routing_weights):
        """
        Forward pass with caching for backward

        Args:
            input_tensor: [qlen, hidden_size]
            expert_ids: [qlen, k]
            routing_weights: [qlen, k]

        Returns:
            output: [qlen, hidden_size]
        """
        qlen = input_tensor.shape[0]
        k = expert_ids.shape[1]
        hidden_size = self.config["hidden_size"]
        expert_num = self.config["expert_num"]

        x = input_tensor.float()

        # Compute m_local_num and positions
        m_local_num = [0] * expert_num
        m_local_pos = [[0] * k for _ in range(qlen)]

        for i in range(qlen):
            for j in range(k):
                eid = expert_ids[i, j].item()
                m_local_pos[i][j] = m_local_num[eid]
                m_local_num[eid] += 1

        activated_experts = [i for i in range(expert_num) if m_local_num[i] > 0]

        # Pack input per expert
        packed_inputs = {}
        for expert_idx in activated_experts:
            tokens = []
            for i in range(qlen):
                for j in range(k):
                    if expert_ids[i, j].item() == expert_idx:
                        tokens.append(x[i])
            packed_inputs[expert_idx] = torch.stack(tokens)

        # Process each expert
        expert_outputs = {}
        expert_caches = {}

        for expert_idx in activated_experts:
            packed_x = packed_inputs[expert_idx]  # [m, hidden_size]

            # Gate: x @ gate_proj.T + x @ gate_lora_a.T @ gate_lora_b.T * scaling
            gate_proj = self.gate_proj[expert_idx]  # [intermediate, hidden]
            gate_base = torch.mm(packed_x, gate_proj.t())
            gate_lora_inter = torch.mm(packed_x, self.gate_lora_a[expert_idx].t())
            gate_lora = torch.mm(gate_lora_inter, self.gate_lora_b[expert_idx].t()) * self.lora_scaling
            gate_out = gate_base + gate_lora

            # Up: x @ up_proj.T + x @ up_lora_a.T @ up_lora_b.T * scaling
            up_proj = self.up_proj[expert_idx]  # [intermediate, hidden]
            up_base = torch.mm(packed_x, up_proj.t())
            up_lora_inter = torch.mm(packed_x, self.up_lora_a[expert_idx].t())
            up_lora = torch.mm(up_lora_inter, self.up_lora_b[expert_idx].t()) * self.lora_scaling
            up_out = up_base + up_lora

            # Activation: silu(gate) * up
            act_out = silu(gate_out) * up_out

            # Down: act @ down_proj.T + act @ down_lora_a.T @ down_lora_b.T * scaling
            down_proj = self.down_proj[expert_idx]  # [hidden, intermediate]
            down_base = torch.mm(act_out, down_proj.t())
            down_lora_inter = torch.mm(act_out, self.down_lora_a[expert_idx].t())
            down_lora = torch.mm(down_lora_inter, self.down_lora_b[expert_idx].t()) * self.lora_scaling
            down_out = down_base + down_lora

            expert_outputs[expert_idx] = down_out
            expert_caches[expert_idx] = {
                "packed_x": packed_x,
                "gate_out": gate_out,
                "up_out": up_out,
                "act_out": act_out,
            }

        # Weighted merge
        output = torch.zeros(qlen, hidden_size, dtype=torch.float32)
        for i in range(qlen):
            for j in range(k):
                expert_idx = expert_ids[i, j].item()
                if expert_idx in expert_outputs:
                    pos = m_local_pos[i][j]
                    weight = routing_weights[i, j].item()
                    output[i] += expert_outputs[expert_idx][pos] * weight

        # Save cache for backward
        self.cache = {
            "input_tensor": x,
            "expert_ids": expert_ids,
            "routing_weights": routing_weights,
            "m_local_num": m_local_num,
            "m_local_pos": m_local_pos,
            "activated_experts": activated_experts,
            "packed_inputs": packed_inputs,
            "expert_caches": expert_caches,
        }

        return output

    def backward(self, grad_output, dump_dir=None):
        """
        Backward pass computing grad_input

        Args:
            grad_output: [qlen, hidden_size]
            dump_dir: optional directory to dump intermediate values

        Returns:
            grad_input: [qlen, hidden_size]
        """
        cache = self.cache
        qlen = grad_output.shape[0]
        k = cache["expert_ids"].shape[1]
        hidden_size = self.config["hidden_size"]

        grad_output = grad_output.float()

        # Scatter grad_output to experts with routing weights
        packed_grad_outputs = {}
        for expert_idx in cache["activated_experts"]:
            grad_tokens = []
            for i in range(qlen):
                for j in range(k):
                    if cache["expert_ids"][i, j].item() == expert_idx:
                        weight = cache["routing_weights"][i, j].item()
                        grad_tokens.append(grad_output[i] * weight)
            packed_grad_outputs[expert_idx] = torch.stack(grad_tokens)

        # Process each expert's backward
        expert_grad_inputs = {}

        for expert_idx in cache["activated_experts"]:
            grad_out = packed_grad_outputs[expert_idx]  # [m, hidden_size]
            expert_cache = cache["expert_caches"][expert_idx]
            packed_x = expert_cache["packed_x"]
            gate_out = expert_cache["gate_out"]
            up_out = expert_cache["up_out"]
            act_out = expert_cache["act_out"]

            # Dump grad_output for this expert
            if dump_dir:
                save_matrix_file(f"{dump_dir}/py_backward_grad_output_e{expert_idx}.bin",
                               grad_out.numpy())

            # =====================================================
            # Stage 1: backward_down - compute grad_intermediate
            # =====================================================
            # down_base backward: grad_out @ down_proj
            down_proj = self.down_proj[expert_idx]  # [hidden, intermediate]
            grad_intermediate_base = torch.mm(grad_out, down_proj)

            if dump_dir:
                save_matrix_file(f"{dump_dir}/py_backward_down_base_e{expert_idx}.bin",
                               grad_intermediate_base.numpy())

            # down_lora backward: grad_out @ down_lora_b @ down_lora_a
            # This adds to grad_intermediate
            down_lora_b = self.down_lora_b[expert_idx]  # [hidden, lora_rank]
            down_lora_a = self.down_lora_a[expert_idx]  # [lora_rank, intermediate]
            grad_down_lora_inter = torch.mm(grad_out, down_lora_b)  # [m, lora_rank]
            grad_intermediate_lora = torch.mm(grad_down_lora_inter, down_lora_a) * self.lora_scaling

            # Total grad_intermediate
            grad_intermediate = grad_intermediate_base + grad_intermediate_lora

            if dump_dir:
                save_matrix_file(f"{dump_dir}/py_backward_grad_intermediate_e{expert_idx}.bin",
                               grad_intermediate.numpy())

            # =====================================================
            # Stage 2: backward_activation
            # =====================================================
            # Dump cached gate_out and up_out used in activation backward
            if dump_dir:
                save_matrix_file(f"{dump_dir}/py_backward_act_gate_cache_e{expert_idx}.bin",
                               gate_out.numpy())
                save_matrix_file(f"{dump_dir}/py_backward_act_up_cache_e{expert_idx}.bin",
                               up_out.numpy())

            grad_gate_out, grad_up_out = silu_backward(gate_out, up_out, grad_intermediate)

            if dump_dir:
                save_matrix_file(f"{dump_dir}/py_backward_grad_gate_out_e{expert_idx}.bin",
                               grad_gate_out.numpy())
                save_matrix_file(f"{dump_dir}/py_backward_grad_up_out_e{expert_idx}.bin",
                               grad_up_out.numpy())

            # =====================================================
            # Stage 3: backward_gate_up - compute grad_input
            # =====================================================
            # gate_base backward: grad_gate_out @ gate_proj
            gate_proj = self.gate_proj[expert_idx]  # [intermediate, hidden]
            grad_input_gate_base = torch.mm(grad_gate_out, gate_proj)

            if dump_dir:
                save_matrix_file(f"{dump_dir}/py_backward_gate_base_e{expert_idx}.bin",
                               grad_input_gate_base.numpy())

            # gate_lora backward: grad_gate_out @ gate_lora_b @ gate_lora_a * scaling
            gate_lora_b = self.gate_lora_b[expert_idx]  # [intermediate, lora_rank]
            gate_lora_a = self.gate_lora_a[expert_idx]  # [lora_rank, hidden]
            gate_lora_inter = torch.mm(grad_gate_out, gate_lora_b)
            grad_input_gate_lora = torch.mm(gate_lora_inter, gate_lora_a) * self.lora_scaling

            if dump_dir:
                save_matrix_file(f"{dump_dir}/py_backward_gate_lora_inter_e{expert_idx}.bin",
                               gate_lora_inter.numpy())
                save_matrix_file(f"{dump_dir}/py_backward_gate_lora_e{expert_idx}.bin",
                               grad_input_gate_lora.numpy())

            # up_base backward: grad_up_out @ up_proj
            up_proj = self.up_proj[expert_idx]  # [intermediate, hidden]
            grad_input_up_base = torch.mm(grad_up_out, up_proj)

            if dump_dir:
                save_matrix_file(f"{dump_dir}/py_backward_up_base_e{expert_idx}.bin",
                               grad_input_up_base.numpy())

            # up_lora backward: grad_up_out @ up_lora_b @ up_lora_a * scaling
            up_lora_b = self.up_lora_b[expert_idx]  # [intermediate, lora_rank]
            up_lora_a = self.up_lora_a[expert_idx]  # [lora_rank, hidden]
            up_lora_inter = torch.mm(grad_up_out, up_lora_b)
            grad_input_up_lora = torch.mm(up_lora_inter, up_lora_a) * self.lora_scaling

            if dump_dir:
                save_matrix_file(f"{dump_dir}/py_backward_up_lora_inter_e{expert_idx}.bin",
                               up_lora_inter.numpy())
                save_matrix_file(f"{dump_dir}/py_backward_up_lora_e{expert_idx}.bin",
                               grad_input_up_lora.numpy())

            # Sum all grad_input components
            grad_input_expert = (grad_input_gate_base + grad_input_gate_lora +
                               grad_input_up_base + grad_input_up_lora)

            if dump_dir:
                save_matrix_file(f"{dump_dir}/py_backward_grad_input_expert_e{expert_idx}.bin",
                               grad_input_expert.numpy())

            expert_grad_inputs[expert_idx] = grad_input_expert

        # Scatter expert grad_inputs back to original positions (NO routing weight here!)
        grad_input = torch.zeros(qlen, hidden_size, dtype=torch.float32)

        for i in range(qlen):
            for j in range(k):
                expert_idx = cache["expert_ids"][i, j].item()
                if expert_idx in expert_grad_inputs:
                    pos = cache["m_local_pos"][i][j]
                    grad_input[i] += expert_grad_inputs[expert_idx][pos]

        if dump_dir:
            save_matrix_file(f"{dump_dir}/py_backward_grad_input_final.bin", grad_input.numpy())

        return grad_input


# ============================================================================
# KT Backend Wrapper
# ============================================================================

def create_kt_wrapper(config, gate_proj, up_proj, down_proj,
                      gate_lora_a, gate_lora_b, up_lora_a, up_lora_b,
                      down_lora_a, down_lora_b):
    """Create KTMoEWrapper instance"""
    if not HAS_KT_KERNEL:
        print("ERROR: kt_kernel not available")
        return None

    lora_scaling = config["lora_alpha"] / config["lora_rank"]

    wrapper = KTMoEWrapper(
        layer_idx=0,
        num_experts=config["expert_num"],
        num_experts_per_tok=config["k"],
        hidden_size=config["hidden_size"],
        moe_intermediate_size=config["intermediate_size"],
        num_gpu_experts=0,
        cpuinfer_threads=config["num_threads"],
        threadpool_count=config["tp_count"],
        weight_path="",
        chunked_prefill_size=1024,
        method="AMXINT8_SFT",  # BF16 backend
        mode="sft",
        lora_rank=config["lora_rank"],
        lora_alpha=config["lora_alpha"],
        max_cache_depth=2,
    )

    # Load weights
    physical_to_logical_map = torch.arange(config["expert_num"], dtype=torch.int64)
    wrapper.load_weights_from_tensors(
        gate_proj=gate_proj,
        up_proj=up_proj,
        down_proj=down_proj,
        physical_to_logical_map_cpu=physical_to_logical_map,
    )

    # Initialize LoRA weights
    wrapper.init_lora_weights(
        gate_lora_a=gate_lora_a,
        gate_lora_b=gate_lora_b,
        up_lora_a=up_lora_a,
        up_lora_b=up_lora_b,
        down_lora_a=down_lora_a,
        down_lora_b=down_lora_b,
    )

    return wrapper


def run_kt_forward_backward(wrapper, input_tensor, expert_ids, routing_weights, grad_output, dump_dir=None):
    """Run KT forward and backward"""
    if wrapper is None:
        return None, None

    # Set dump environment if requested
    if dump_dir:
        os.environ["SFT_MOE_DUMP"] = "1"
        os.environ["SFT_MOE_DUMP_DIR"] = dump_dir

    # Forward
    output = wrapper.forward_sft(
        hidden_states=input_tensor,
        expert_ids=expert_ids,
        weights=routing_weights,
        save_for_backward=True,
    )

    # Backward
    grad_input, grad_loras, grad_weights = wrapper.backward(grad_output)

    # Clean up environment
    if dump_dir:
        del os.environ["SFT_MOE_DUMP"]
        del os.environ["SFT_MOE_DUMP_DIR"]

    return output, grad_input


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Debug MoE backward pass")
    parser.add_argument("--dump-dir", default="./backward_debug", help="Directory for dumps")
    parser.add_argument("--threshold", type=float, default=0.05, help="Error threshold")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--skip-kt", action="store_true", help="Skip KT backend (Python only)")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    config = DEFAULT_CONFIG.copy()
    lora_scaling = config["lora_alpha"] / config["lora_rank"]

    print("=" * 80)
    print("MoE Backward Debug")
    print("=" * 80)
    print(f"Config: {config}")
    print(f"LoRA scaling: {lora_scaling}")
    print(f"Dump dir: {args.dump_dir}")
    print("=" * 80)

    # Clean up and create dump directories
    cpp_dir = f"{args.dump_dir}/cpp"
    py_dir = f"{args.dump_dir}/py"
    for d in [cpp_dir, py_dir]:
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d, exist_ok=True)

    # Initialize weights
    print("\n[Initializing weights]")
    gate_proj = (torch.randn(config["expert_num"], config["intermediate_size"], config["hidden_size"],
                            dtype=torch.bfloat16) * WEIGHT_SCALE).contiguous()
    up_proj = (torch.randn(config["expert_num"], config["intermediate_size"], config["hidden_size"],
                          dtype=torch.bfloat16) * WEIGHT_SCALE).contiguous()
    down_proj = (torch.randn(config["expert_num"], config["hidden_size"], config["intermediate_size"],
                            dtype=torch.bfloat16) * WEIGHT_SCALE).contiguous()

    gate_lora_a = (torch.randn(config["expert_num"], config["lora_rank"], config["hidden_size"],
                              dtype=torch.bfloat16) * WEIGHT_SCALE).contiguous()
    gate_lora_b = (torch.randn(config["expert_num"], config["intermediate_size"], config["lora_rank"],
                              dtype=torch.bfloat16) * WEIGHT_SCALE).contiguous()
    up_lora_a = (torch.randn(config["expert_num"], config["lora_rank"], config["hidden_size"],
                            dtype=torch.bfloat16) * WEIGHT_SCALE).contiguous()
    up_lora_b = (torch.randn(config["expert_num"], config["intermediate_size"], config["lora_rank"],
                            dtype=torch.bfloat16) * WEIGHT_SCALE).contiguous()
    down_lora_a = (torch.randn(config["expert_num"], config["lora_rank"], config["intermediate_size"],
                              dtype=torch.bfloat16) * WEIGHT_SCALE).contiguous()
    down_lora_b = (torch.randn(config["expert_num"], config["hidden_size"], config["lora_rank"],
                              dtype=torch.bfloat16) * WEIGHT_SCALE).contiguous()

    # Generate test data
    print("\n[Generating test data]")
    input_tensor = (torch.randn(config["qlen"], config["hidden_size"], dtype=torch.bfloat16)
                   * INPUT_SCALE).contiguous()

    # Generate random expert assignments (top-k from all experts)
    expert_ids = torch.stack([
        torch.randperm(config["expert_num"])[:config["k"]]
        for _ in range(config["qlen"])
    ]).to(torch.int64).contiguous()

    # Generate random routing weights (normalized)
    routing_weights = torch.rand(config["qlen"], config["k"], dtype=torch.float32).contiguous()
    routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)

    # Generate random grad_output
    grad_output = (torch.randn(config["qlen"], config["hidden_size"], dtype=torch.bfloat16)
                  * GRAD_SCALE).contiguous()

    print(f"  input_tensor: {input_tensor.shape}, dtype={input_tensor.dtype}")
    print(f"  expert_ids: {expert_ids.shape}, dtype={expert_ids.dtype}")
    print(f"  routing_weights: {routing_weights.shape}, dtype={routing_weights.dtype}")
    print(f"  grad_output: {grad_output.shape}, dtype={grad_output.dtype}")

    # Run PyTorch reference
    print("\n[Running PyTorch reference forward + backward]")
    py_ref = PyTorchMoEReference(
        gate_proj, up_proj, down_proj,
        gate_lora_a, gate_lora_b, up_lora_a, up_lora_b,
        down_lora_a, down_lora_b, lora_scaling, config
    )
    py_output = py_ref.forward(input_tensor, expert_ids, routing_weights)
    py_grad_input = py_ref.backward(grad_output.float(), dump_dir=py_dir)
    print(f"  py_output: {py_output.shape}")
    print(f"  py_grad_input: {py_grad_input.shape}")

    # Run KT backend
    kt_output = None
    kt_grad_input = None

    if not args.skip_kt and HAS_KT_KERNEL:
        print("\n[Running KT backend forward + backward]")
        wrapper = create_kt_wrapper(
            config, gate_proj, up_proj, down_proj,
            gate_lora_a, gate_lora_b, up_lora_a, up_lora_b,
            down_lora_a, down_lora_b
        )

        if wrapper:
            kt_output, kt_grad_input = run_kt_forward_backward(
                wrapper, input_tensor, expert_ids, routing_weights,
                grad_output, dump_dir=cpp_dir
            )
            print(f"  kt_output: {kt_output.shape}")
            print(f"  kt_grad_input: {kt_grad_input.shape}")
    else:
        print("\n[Skipping KT backend]")

    # Compare results
    print("\n" + "=" * 80)
    print("Comparison Results")
    print("=" * 80)

    if kt_output is not None:
        print("\n[Forward Output Comparison]")
        compare_tensors(py_output, kt_output.float(), "forward_output", args.threshold)

    if kt_grad_input is not None:
        print("\n[Backward grad_input Comparison]")
        compare_tensors(py_grad_input, kt_grad_input.float(), "grad_input", args.threshold)

        # Compare intermediate dumps if available
        print("\n[Comparing Intermediate Values]")

        # Find activated experts
        activated_experts = set()
        for i in range(config["qlen"]):
            for j in range(config["k"]):
                activated_experts.add(expert_ids[i, j].item())

        stages = [
            ("backward_grad_output", "backward_grad_output_tp0"),
            ("backward_down_base", "backward_down_base_tp0"),
            ("backward_grad_intermediate", "backward_grad_intermediate_tp0"),
            # Cached values used in activation backward
            ("backward_act_gate_cache", "backward_act_gate_cache_tp0"),
            ("backward_act_up_cache", "backward_act_up_cache_tp0"),
            # Activation backward outputs
            ("backward_grad_gate_out", "backward_grad_gate_out_tp0"),
            ("backward_grad_up_out", "backward_grad_up_out_tp0"),
            # Gate/Up backward
            ("backward_gate_base", "backward_gate_base_tp0"),
            ("backward_up_base", "backward_up_base_tp0"),
            ("backward_gate_lora_inter", "backward_gate_lora_inter_tp0"),
            ("backward_gate_lora", "backward_gate_lora_tp0"),
            ("backward_up_lora_inter", "backward_up_lora_inter_tp0"),
            ("backward_up_lora", "backward_up_lora_tp0"),
            ("backward_grad_input_expert", "backward_grad_input_expert_tp0"),
        ]

        for expert_idx in sorted(activated_experts)[:3]:  # Check first 3 experts
            print(f"\n  Expert {expert_idx}:")
            for py_stage, cpp_stage in stages:
                py_file = f"{py_dir}/py_{py_stage}_e{expert_idx}.bin"
                cpp_file = f"{cpp_dir}/{cpp_stage}_e{expert_idx}.bin"

                _, _, py_data = read_matrix_file(py_file)
                _, _, cpp_data = read_matrix_file(cpp_file)

                if py_data is not None and cpp_data is not None:
                    # Handle shape mismatch (C++ may have padding)
                    if py_data.shape != cpp_data.shape:
                        if py_data.shape[0] == cpp_data.shape[0] and cpp_data.shape[1] > py_data.shape[1]:
                            cpp_data = cpp_data[:, :py_data.shape[1]]

                    if py_data.shape == cpp_data.shape:
                        py_t = torch.from_numpy(py_data)
                        cpp_t = torch.from_numpy(cpp_data)
                        diff = (py_t - cpp_t).abs()
                        cos_sim = F.cosine_similarity(py_t.flatten().unsqueeze(0),
                                                      cpp_t.flatten().unsqueeze(0)).item()

                        status = "PASS" if cos_sim > 0.99 else "FAIL"
                        color = "\033[92m" if status == "PASS" else "\033[91m"
                        print(f"    [{color}{status}\033[0m] {py_stage}: "
                              f"cos_sim={cos_sim:.6f}, abs_max={diff.max().item():.6e}")
                    else:
                        print(f"    [SHAPE] {py_stage}: py={py_data.shape} vs cpp={cpp_data.shape}")
                else:
                    missing = []
                    if py_data is None:
                        missing.append("py")
                    if cpp_data is None:
                        missing.append("cpp")
                    print(f"    [MISSING] {py_stage}: missing {', '.join(missing)}")

    print("\n" + "=" * 80)
    print("Done!")
    print("=" * 80)

    # Print instructions if C++ dump is missing
    if kt_grad_input is not None:
        cpp_files = list(Path(cpp_dir).glob("*.bin"))
        if not cpp_files:
            print("\nNOTE: C++ dumps not found. To enable C++ dumps:")
            print("  1. Edit /home/star/hxx/ktransformers/kt-kernel/operators/amx/sft_moe.hpp")
            print("  2. In is_dump_enabled() function (line ~153), remove 'return false;'")
            print("  3. Rebuild kt-kernel: cd /home/star/hxx/ktransformers/kt-kernel && ./build.sh")
            print("  4. Re-run this script")


if __name__ == "__main__":
    main()
