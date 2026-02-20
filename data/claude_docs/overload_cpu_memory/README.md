# DeepSpeed ZeRO-3 + LoRA CPU Memory Spike Investigation

## Overview

This directory documents the investigation and fix for a CPU memory spike issue when using
DeepSpeed ZeRO-3 offload with LoRA fine-tuning in LlamaFactory.

## Documents

| File | Description |
|------|-------------|
| [01_requirement.md](01_requirement.md) | Problem statement and user observation |
| [02_architecture_analysis.md](02_architecture_analysis.md) | Code architecture and data flow analysis |
| [03_root_cause.md](03_root_cause.md) | Root cause analysis (with code evidence) |
| [04_solution_design.md](04_solution_design.md) | Solution design and approach |
| [05_implementation.md](05_implementation.md) | Final implementation details |
| [06_debug_log.md](06_debug_log.md) | Debug process and issues encountered |
