#!/usr/bin/env python
"""Inspect safetensors files in a folder and print tensor information with sampling."""

import argparse
import random
import sys
from pathlib import Path

from safetensors import safe_open


def format_size(num_bytes: int) -> str:
    """Format bytes to human readable string."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if abs(num_bytes) < 1024.0:
            return f"{num_bytes:.2f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.2f} PB"


def get_dtype_size(dtype: str) -> int:
    """Get size in bytes for a dtype string."""
    dtype_sizes = {
        "F64": 8,
        "F32": 4,
        "F16": 2,
        "BF16": 2,
        "I64": 8,
        "I32": 4,
        "I16": 2,
        "I8": 1,
        "U8": 1,
        "BOOL": 1,
    }
    return dtype_sizes.get(dtype, 4)


def print_tensor_sample(tensor, tensor_name: str, max_elements: int = 20):
    """Print a sample of tensor values."""
    print(f"\n  Sample values for '{tensor_name}':")
    flat = tensor.flatten()
    total = flat.numel()

    if total <= max_elements:
        indices = list(range(total))
    else:
        indices = sorted(random.sample(range(total), max_elements))

    values = [f"{flat[i].item():.6g}" for i in indices]
    print(f"    Shape: {tuple(tensor.shape)}, Dtype: {tensor.dtype}")
    print(f"    Min: {tensor.min().item():.6g}, Max: {tensor.max().item():.6g}, Mean: {tensor.float().mean().item():.6g}")
    print(f"    Sample indices: {indices[:10]}{'...' if len(indices) > 10 else ''}")
    print(f"    Sample values: [{', '.join(values[:10])}{'...' if len(values) > 10 else ''}]")


def inspect_safetensors_folder(folder_path: str, sample_ratio: float = 0.05, seed: int = 42) -> None:
    """Inspect all safetensors files in a folder and print tensor information."""
    random.seed(seed)

    folder = Path(folder_path)
    if not folder.exists():
        print(f"Error: Folder not found: {folder_path}")
        sys.exit(1)

    if not folder.is_dir():
        print(f"Error: {folder_path} is not a directory")
        sys.exit(1)

    safetensor_files = sorted(folder.glob("*.safetensors"))
    if not safetensor_files:
        print(f"Error: No .safetensors files found in {folder_path}")
        sys.exit(1)

    print(f"Folder: {folder_path}")
    print(f"Found {len(safetensor_files)} safetensors file(s)")
    print("=" * 100)

    grand_total_params = 0
    grand_total_bytes = 0
    grand_total_tensors = 0
    all_tensor_info = []  # (file_path, key) tuples for sampling

    # First pass: collect all tensor info and print sizes
    for sf_file in safetensor_files:
        print(f"\nFile: {sf_file.name}")
        print(f"File size: {format_size(sf_file.stat().st_size)}")
        print("-" * 80)

        file_params = 0
        file_bytes = 0

        with safe_open(str(sf_file), framework="pt") as f:
            keys = list(f.keys())
            print(f"Tensors in this file: {len(keys)}\n")
            print(f"{'Tensor Name':<60} {'Shape':<25} {'Dtype':<10} {'Size':<15}")
            print("-" * 110)

            for key in sorted(keys):
                tensor = f.get_tensor(key)
                shape = tuple(tensor.shape)
                dtype = str(tensor.dtype).replace("torch.", "")
                num_elements = tensor.numel()
                byte_size = tensor.element_size() * num_elements

                file_params += num_elements
                file_bytes += byte_size

                shape_str = str(shape)
                print(f"{key:<60} {shape_str:<25} {dtype:<10} {format_size(byte_size):<15}")

                all_tensor_info.append((str(sf_file), key))

        grand_total_params += file_params
        grand_total_tensors += len(keys)
        grand_total_bytes += file_bytes

        print(f"\nFile summary: {len(keys)} tensors, {file_params:,} params, {format_size(file_bytes)}")

    # Print grand summary
    print("\n" + "=" * 100)
    print("GRAND SUMMARY")
    print("=" * 100)
    print(f"  Total files: {len(safetensor_files)}")
    print(f"  Total tensors: {grand_total_tensors}")
    print(f"  Total parameters: {grand_total_params:,}")
    print(f"  Total size (in tensors): {format_size(grand_total_bytes)}")

    # Second pass: sample tensors and print values
    num_samples = max(1, int(len(all_tensor_info) * sample_ratio))
    sampled_tensors = random.sample(all_tensor_info, min(num_samples, len(all_tensor_info)))

    print("\n" + "=" * 100)
    print(f"SAMPLED TENSOR VALUES ({len(sampled_tensors)} tensors, {sample_ratio*100:.1f}% sample rate)")
    print("=" * 100)

    for file_path, key in sampled_tensors:
        with safe_open(file_path, framework="pt") as f:
            tensor = f.get_tensor(key)
            print(f"\nFile: {Path(file_path).name}")
            print_tensor_sample(tensor, key)


def main():
    parser = argparse.ArgumentParser(
        description="Inspect safetensors files in a folder",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python inspect_safetensors.py /path/to/model/folder
  python inspect_safetensors.py /path/to/model/folder --sample-ratio 0.1
  python inspect_safetensors.py /path/to/model/folder --seed 123
        """
    )
    parser.add_argument("folder", help="Path to folder containing safetensors files")
    parser.add_argument(
        "--sample-ratio",
        type=float,
        default=0.05,
        help="Ratio of tensors to sample for value printing (default: 0.05 = 5%%)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling (default: 42)"
    )
    args = parser.parse_args()

    inspect_safetensors_folder(args.folder, args.sample_ratio, args.seed)


if __name__ == "__main__":
    main()
