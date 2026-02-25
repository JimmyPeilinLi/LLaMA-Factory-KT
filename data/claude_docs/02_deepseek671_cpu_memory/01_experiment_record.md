# DeepSeek-V3 (671B) & Qwen3-235B ZeRO-3 CPU 内存实测记录

## 实验环境

| 项目 | 值 |
|---|---|
| 机器 | 1923 GB RAM (1.88 TB), 240 cores, 无 swap |
| GPU | 8× NVIDIA H20-3e (143 GB), 本次使用 4 块 |
| Python | /mnt/raid/lpl/miniconda3/envs/zero/bin/python 3.11 |
| PyTorch | 2.9.1+cu128 |
| DeepSpeed | 0.18.4 |
| Transformers | 5.0.0 |
| LlamaFactory | 0.9.5.dev0 (branch: zero, commit: b07b6c98) |
| 启动方式 | torchrun --nproc_per_node=4 |
| 环境变量 | MALLOC_ARENA_MAX=2, MALLOC_TRIM_THRESHOLD_=0 |

---

## 实验一：DeepSeek-V3 (671B)

### 配置

| 项目 | 值 |
|---|---|
| 模型 | DeepSeek-V3, FP8 checkpoint (163 shards) |
| 模型路径 | /mnt/raid/models/DeepSeek-V3 |
| 训练方式 | SFT + LoRA (rank=8, target=all), bf16 |
| DeepSpeed | ZeRO-3 + CPU offload (pin_memory=true) |
| 数据集 | math_train (max_samples=1000) |
| 已应用补丁 | patch_deepspeed_zero_init_memory + patch_zero3_model_loading |

### 结果

| 指标 | 值 |
|---|---|
| **Phase 0 进度** | **729/967 参数 (75.4%)** |
| **停止位置** | model.layers.46.mlp.experts.gate_up_proj |
| **运行时间** | ~8.5 分钟 |
| **初始可用内存** | 1828 GB |
| **终止时可用内存** | 51 GB (系统已用 1871 GB) |
| **终止原因** | 内存监控脚本触发自动 kill (阈值 150 GB) |
| **结局** | Phase 0 未完成，OOM，从未进入 Phase 1 |

### 与历史实验对比

| 实验 | 优化措施 | 进度 | 停止层 | 系统峰值使用 |
|---|---|---|---|---|
| 实验 1 (旧机器) | 无 | 632/967 (65.4%) | layer 40, experts.down_proj | ~1856 GB |
| 实验 2 (旧机器) | MALLOC_TRIM_THRESHOLD_=0 | 697/967 (72.1%) | layer 44, experts.gate_up_proj | OOM killed |
| **本次** | **补丁 + MALLOC_TRIM + ARENA_MAX** | **729/967 (75.4%)** | **layer 46, experts.gate_up_proj** | **1871 GB** |

### 外推估算

```
已完成 75.4% 时系统使用 1871 GB
外推 100%: 1871 / 0.754 ≈ 2,481 GB
BF16 理论值: 1,376 GB
倍率: 2,481 / 1,376 ≈ 1.80x
```

---

## 实验二：Qwen3-235B-A22B

### 配置

| 项目 | 值 |
|---|---|
| 模型 | Qwen3-235B-A22B-Instruct-2507, BF16 checkpoint (118 shards) |
| 模型路径 | /mnt/raid/models/Qwen3-235B-A22B-Instruct-2507 |
| 训练方式 | SFT + LoRA (rank=8, target=all), bf16 |
| DeepSpeed | ZeRO-3 + CPU offload (pin_memory=true) |
| 数据集 | math_train (max_samples=1000) |
| 已应用补丁 | patch_deepspeed_zero_init_memory + patch_zero3_model_loading (v4) |

### 各阶段实测内存

| 阶段 | 时间 | 每 rank RSS | 系统可用 | 系统已用 |
|---|---|---|---|---|
| 初始状态 | 20:47:46 | 1.66 GB | 1845 GB | 78 GB |
| Phase 0 开始（zero.Init） | 20:47:48 | 2.10 GB | 1845 GB | 78 GB |
| Phase 0 中期（537/1037） | 20:49:01 | 77.25 GB | 1525 GB | 398 GB |
| Phase 0 结束（1021/1037） | 20:50:06 | 144.72 GB | 1251 GB | 672 GB |
| Phase 1 开始（shard loading） | 20:50:08 | ~145 GB | ~1242 GB | ~681 GB |
| Phase 1 中期（shard 38/118） | 20:53:38 | 148.71 GB | 1242 GB | 681 GB |
| Phase 1 峰值 | — | **150.59 GB** | ~1222 GB | **706 GB** |
| Phase 1 结束（94/94 层） | 21:01:32 | 146.86 GB | 1278 GB | 645 GB |
| After from_pretrained | 21:01:32 | 146.86 GB | 1278 GB | 645 GB |
| After init_adapter | 21:01:33 | 146.89 GB | 1278 GB | 645 GB |
| **稳态训练** | 21:03+ | **~147.9 GB** | **1273 GB** | **649 GB** |

### 稳态系统内存详细分布 (/proc/meminfo)

| 指标 | 值 | 说明 |
|---|---|---|
| MemTotal | 1923.3 GB | |
| MemAvailable | 1272.8 GB | |
| AnonPages | 20.2 GB | 进程私有内存（Python, PyTorch 元数据等） |
| **Shmem** | **581.0 GB** | **CUDA pinned memory（模型参数主体）** |
| Mapped | 582.6 GB | 内存映射文件 + pinned memory |
| Cached | 1429.7 GB | 页缓存（含 safetensors 文件缓存） |
| PageTables | 1.3 GB | 页表开销 |

### 每进程内存

| 进程 | RSS | VSZ | 说明 |
|---|---|---|---|
| 主训练进程 ×4 | 147.9 GB | 757 GB | ZeRO-3 分区 + 运行时开销 |
| Dataloader worker ×16 | 1.5 GB | 743 GB | fork 后共享大部分内存 |

### 关键计算

```
BF16 模型大小: 235B × 2 bytes = 470 GB
ZeRO-3 每 rank 理论分区: 470 / 4 = 117.5 GB
实测每 rank RSS: 147.9 GB
每 rank 开销: 147.9 - 117.5 = 30.4 GB (1.26x)

系统级:
  基线使用: ~78 GB
  稳态使用: 649 GB
  模型相关: 649 - 78 = 571 GB (vs 理论 470 GB)
  系统级倍率: 571 / 470 = 1.21x
```

### 结果：训练成功

训练正常进行，loss 从 5.0 降至 1.92（前 2 step），内存稳定。

---

## 日志文件

| 文件 | 路径 |
|---|---|
| DeepSeek-V3 训练日志 | /mnt/raid/lpl/zero_test_output/dpsk3_671b/train.log |
| DeepSeek-V3 内存监控 | /mnt/raid/lpl/zero_test_output/dpsk3_671b/mem_monitor.log |
| Qwen3-235B 训练日志 | /mnt/raid/lpl/zero_test_output/qwen3_235b/train.log |
| Qwen3-235B 内存监控 | /mnt/raid/lpl/zero_test_output/qwen3_235b/mem_monitor.log |
