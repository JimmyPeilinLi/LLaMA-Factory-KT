#!/usr/bin/env python
"""Inspect safetensors file and print tensor information."""

import argparse
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


def inspect_safetensors(file_path: str) -> None:
    """Inspect a safetensors file and print tensor information."""
    path = Path(file_path)
    if not path.exists():
        print(f"Error: File not found: {file_path}")
        sys.exit(1)

    print(f"File: {file_path}")
    print(f"File size: {format_size(path.stat().st_size)}")
    print("-" * 80)

    total_params = 0
    total_bytes = 0

    with safe_open(file_path, framework="pt") as f:
        keys = f.keys()
        print(f"Total tensors: {len(keys)}\n")
        print(f"Tensor Name\tShape\tDtype\tSize")
        print("-" * 80)

        for key in sorted(keys):
            tensor = f.get_tensor(key)
            shape = tuple(tensor.shape)
            dtype = str(tensor.dtype).replace("torch.", "")
            num_elements = tensor.numel()
            byte_size = tensor.element_size() * num_elements

            total_params += num_elements
            total_bytes += byte_size

            print(f"{key}\t{shape}\t{dtype}\t{format_size(byte_size)}")

    print("=" * 105)
    print(f"\nSummary:")
    print(f"  Total tensors: {len(keys)}")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Total size (in tensors): {format_size(total_bytes)}")


def main():
    parser = argparse.ArgumentParser(description="Inspect safetensors file")
    parser.add_argument("file", help="Path to safetensors file")
    args = parser.parse_args()

    inspect_safetensors(args.file)


if __name__ == "__main__":
    main()
