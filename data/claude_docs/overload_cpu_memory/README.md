# DeepSpeed ZeRO-3 + LoRA CPU Memory Spike Investigation

## Overview

This directory documents the investigation and fix for CPU memory issues when using
DeepSpeed ZeRO-3 offload with LoRA fine-tuning in LlamaFactory.

Two distinct problems have been identified:
1. **Phase 1 问题 (Qwen3 系列)**: shard-by-shard 权重加载阶段内存峰值 → 已通过 v4 方案解决
2. **Phase 0 问题 (DeepSeek-V3)**: `deepspeed.zero.Init()` 模型创建阶段 OOM → 待解决

**Current version: v4** — Per-layer completion tracking for boundary-layer expert keys.

## Documents

| File | Description |
|------|-------------|
| [01_requirement.md](01_requirement.md) | Problem statement and user observation |
| [02_architecture_analysis.md](02_architecture_analysis.md) | Code architecture and data flow analysis |
| [03_root_cause.md](03_root_cause.md) | Root cause analysis (with code evidence) |
| [04_solution_design.md](04_solution_design.md) | Solution design: v1→v2→v3→v4 evolution |
| [05_implementation.md](05_implementation.md) | v4 implementation details and memory profile |
| [06_debug_log.md](06_debug_log.md) | Debug process and issues encountered (v1→v4) |
| [07_memory_phases.md](07_memory_phases.md) | CPU memory phases analysis for Qwen3-235B |
| [08_deepseek_v3_analysis.md](08_deepseek_v3_analysis.md) | **DeepSeek-V3 OOM 分析: Phase 0 问题** |

## Version History

| Version | Key Change | Limitation |
|---------|-----------|------------|
| v1 | Naive shard-by-shard loading | RuntimeError on MoE weight conversion |
| v2 | Boundary-layer buffering (whole layer) | All non-expert weights MISSING |
| v3 | Expert/non-expert separation, avoid buggy API | Ineffective for 235B (all layers are boundary) |
| v4 | Per-layer completion tracking | 解决 Phase 1，但不覆盖 Phase 0 (DeepSeek-V3 OOM) |
