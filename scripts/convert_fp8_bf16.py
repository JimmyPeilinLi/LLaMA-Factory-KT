#!/usr/bin/env python3
"""
Convert DeepSeek-V3 FP8 checkpoint:
  - Non-expert weights (attn, norms, embeddings, shared_experts, dense MLP, gate)
    → dequantize FP8 to BF16 (already-BF16 weights kept as-is)
  - MoE expert weights (mlp.experts.*)
    → skipped entirely (not saved; will stay on meta device when loaded)
  - weight_scale_inv keys
    → consumed during dequant, then discarded
  - quantization_config
    → removed from config.json

Usage:
    python scripts/convert_fp8_to_bf16.py \
        --src /mnt/raid/models/DeepSeek-V3 \
        --dst /mnt/raid/models/DeepSeek-V3-BF16
"""

import argparse
import json
import os
import shutil
from collections import defaultdict

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from tqdm import tqdm

BLOCK_SIZE = 128
MAX_SHARD_BYTES = 5 * 1024**3  # 5 GB per output shard


# ── helpers ──────────────────────────────────────────────────────────────────

def is_expert_key(name: str) -> bool:
    return ".mlp.experts." in name


def is_scale_inv_key(name: str) -> bool:
    return name.endswith(".weight_scale_inv")


def dequantize_fp8(weight: torch.Tensor, scale_inv: torch.Tensor) -> torch.Tensor:
    """Block-wise dequantize an FP8-E4M3 weight to BF16 using its scale_inv.

    Handles partial blocks (when dimensions aren't divisible by BLOCK_SIZE)
    by zero-padding, dequantizing, then slicing back.
    """
    m, n = weight.shape
    bs = BLOCK_SIZE
    sm, sn = scale_inv.shape  # number of row / col blocks

    # padded dimensions (may equal m, n if already aligned)
    pm, pn = sm * bs, sn * bs

    if pm != m or pn != n:
        weight_padded = torch.zeros(pm, pn, dtype=weight.dtype)
        weight_padded[:m, :n] = weight
    else:
        weight_padded = weight

    # reshape into blocks: [sm, bs, sn, bs]
    w = weight_padded.reshape(sm, bs, sn, bs).to(torch.bfloat16)
    s = scale_inv.to(torch.bfloat16).unsqueeze(1).unsqueeze(3)  # [sm, 1, sn, 1]
    result = (w * s).reshape(pm, pn)

    return result[:m, :n].contiguous()


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Convert DeepSeek-V3 FP8 → BF16 (non-expert) + meta (expert)")
    parser.add_argument("--src", type=str, default="/mnt/raid/models/DeepSeek-V3",
                        help="Source FP8 checkpoint directory")
    parser.add_argument("--dst", type=str, default="/mnt/raid/models/DeepSeek-V3-BF16",
                        help="Destination BF16 checkpoint directory")
    args = parser.parse_args()

    src_dir = args.src
    dst_dir = args.dst
    os.makedirs(dst_dir, exist_ok=True)

    # ── load index ───────────────────────────────────────────────────────────
    index_path = os.path.join(src_dir, "model.safetensors.index.json")
    with open(index_path) as f:
        index = json.load(f)

    weight_map = index["weight_map"]

    # group keys by source shard
    shard_to_keys: dict[str, list[str]] = defaultdict(list)
    for key, shard in weight_map.items():
        shard_to_keys[shard].append(key)

    # classify keys
    expert_keys = [k for k in weight_map if is_expert_key(k)]
    scale_keys  = [k for k in weight_map if is_scale_inv_key(k) and not is_expert_key(k)]
    convert_keys = [
        k for k in weight_map
        if not is_expert_key(k) and not is_scale_inv_key(k)
    ]
    print(f"Expert keys (skip):      {len(expert_keys)}")
    print(f"Scale-inv keys (consume): {len(scale_keys)}")
    print(f"Keys to convert/copy:    {len(convert_keys)}")

    # ── pass 1: collect all non-expert scale_inv tensors ─────────────────────
    print("\n[Pass 1] Collecting non-expert scale_inv values …")
    scale_inv_map: dict[str, torch.Tensor] = {}
    for shard_name in tqdm(sorted(shard_to_keys), desc="shards"):
        keys_in_shard = shard_to_keys[shard_name]
        needed = [k for k in keys_in_shard if is_scale_inv_key(k) and not is_expert_key(k)]
        if not needed:
            continue
        with safe_open(os.path.join(src_dir, shard_name), framework="pt", device="cpu") as f:
            for k in needed:
                scale_inv_map[k] = f.get_tensor(k)
    print(f"  Collected {len(scale_inv_map)} scale_inv tensors\n")

    # ── pass 2: dequantize non-expert weights and save ───────────────────────
    print("[Pass 2] Converting and saving …")
    new_weight_map: dict[str, str] = {}
    out_shard_idx = 0
    cur_tensors: dict[str, torch.Tensor] = {}
    cur_size = 0
    total_bytes_saved = 0

    def flush():
        nonlocal out_shard_idx, cur_tensors, cur_size, total_bytes_saved
        if not cur_tensors:
            return
        out_shard_idx += 1
        # use placeholder total; rename later
        fname = f"model-{out_shard_idx:05d}-of-XXXXX.safetensors"
        save_file(cur_tensors, os.path.join(dst_dir, fname))
        for key in cur_tensors:
            new_weight_map[key] = fname
        total_bytes_saved += cur_size
        print(f"  Saved {fname}  ({len(cur_tensors)} tensors, {cur_size / 1e9:.2f} GB)")
        cur_tensors = {}
        cur_size = 0

    for shard_name in tqdm(sorted(shard_to_keys), desc="shards"):
        keys_in_shard = shard_to_keys[shard_name]
        # only process non-expert, non-scale-inv keys
        process = [
            k for k in keys_in_shard
            if not is_expert_key(k) and not is_scale_inv_key(k)
        ]
        if not process:
            continue

        with safe_open(os.path.join(src_dir, shard_name), framework="pt", device="cpu") as f:
            for key in process:
                tensor = f.get_tensor(key)

                if tensor.dtype == torch.float8_e4m3fn:
                    # dequantize FP8 → BF16
                    sinv_key = key + "_scale_inv"
                    if sinv_key not in scale_inv_map:
                        raise RuntimeError(f"Missing scale_inv for {key}")
                    tensor = dequantize_fp8(tensor, scale_inv_map[sinv_key])

                nbytes = tensor.numel() * tensor.element_size()
                if cur_size + nbytes > MAX_SHARD_BYTES and cur_tensors:
                    flush()
                cur_tensors[key] = tensor
                cur_size += nbytes

    flush()

    # ── rename shard files with correct total ────────────────────────────────
    total_shards = out_shard_idx
    for i in range(1, total_shards + 1):
        old = f"model-{i:05d}-of-XXXXX.safetensors"
        new = f"model-{i:05d}-of-{total_shards:05d}.safetensors"
        os.rename(os.path.join(dst_dir, old), os.path.join(dst_dir, new))

    final_weight_map = {}
    for key, fname in new_weight_map.items():
        final_weight_map[key] = fname.replace("XXXXX", f"{total_shards:05d}")

    # ── write new index ──────────────────────────────────────────────────────
    new_index = {
        "metadata": {"total_size": total_bytes_saved},
        "weight_map": dict(sorted(final_weight_map.items())),
    }
    with open(os.path.join(dst_dir, "model.safetensors.index.json"), "w") as f:
        json.dump(new_index, f, indent=2)
    print(f"\nIndex written: {len(final_weight_map)} keys, {total_shards} shards, "
          f"{total_bytes_saved / 1e9:.2f} GB total")

    # ── config.json: remove quantization_config ──────────────────────────────
    with open(os.path.join(src_dir, "config.json")) as f:
        config = json.load(f)
    config.pop("quantization_config", None)
    with open(os.path.join(dst_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    print("Wrote config.json (quantization_config removed)")

    # ── copy tokenizer & other small files ───────────────────────────────────
    copy_names = {
        "tokenizer.json", "tokenizer_config.json",
        "special_tokens_map.json", "generation_config.json",
        "configuration.json",
    }
    for fname in os.listdir(src_dir):
        if fname in copy_names:
            shutil.copy2(os.path.join(src_dir, fname), os.path.join(dst_dir, fname))
            print(f"Copied {fname}")

    print(f"\nDone!  Converted model → {dst_dir}")


if __name__ == "__main__":
    main()