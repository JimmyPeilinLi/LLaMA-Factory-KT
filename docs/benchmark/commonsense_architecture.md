# Commonsense Benchmark - 架构分析文档

## 1. 系统架构概览

```
┌─────────────────────────────────────────────────────────────────┐
│                    Commonsense Benchmark System                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────────┐    ┌──────────────────┐                  │
│  │  原始数据集       │    │  转换脚本         │                  │
│  │  (HuggingFace)   │───▶│  convert_*.py    │                  │
│  └──────────────────┘    └────────┬─────────┘                  │
│                                   │                             │
│                                   ▼                             │
│  ┌──────────────────────────────────────────────────┐          │
│  │              data/commonsense/                    │          │
│  │  ├── arc_challenge_train.json                    │          │
│  │  ├── arc_challenge_val.json                      │          │
│  │  ├── ...                                         │          │
│  │  └── winogrande_xl_val.json                      │          │
│  └──────────────────────────────────────────────────┘          │
│                          │                                      │
│                          ▼                                      │
│  ┌──────────────────────────────────────────────────┐          │
│  │           data/dataset_info.json                 │          │
│  │  (数据集注册中心)                                   │          │
│  └──────────────────────────────────────────────────┘          │
│                          │                                      │
│                          ▼                                      │
│  ┌──────────────────────────────────────────────────┐          │
│  │     examples/commonsense_benchmark/*.yaml        │          │
│  │  ├── deepseek2_lora_sft.yaml                    │          │
│  │  ├── deepseek2_lora_sft_kt.yaml                 │          │
│  │  ├── deepseek3_lora_sft.yaml                    │          │
│  │  ├── deepseek3_lora_sft_kt.yaml                 │          │
│  │  ├── qwen3moe_lora_sft.yaml                     │          │
│  │  └── qwen3moe_lora_sft_kt.yaml                  │          │
│  └──────────────────────────────────────────────────┘          │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## 2. 数据流架构

```
┌─────────────────────────────────────────────────────────────────┐
│                        数据流                                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  原始数据                                                        │
│  /mnt/data/lpl/lora_datasets/commonsense_datasets/              │
│  └── {dataset_name}/                                            │
│      ├── train/  (HuggingFace Arrow 格式)                       │
│      └── validation/                                            │
│                     │                                            │
│                     │ convert_commonsense_datasets.py           │
│                     ▼                                            │
│  转换后数据                                                       │
│  data/commonsense/                                              │
│  ├── {dataset}_train.json  (Alpaca 格式)                        │
│  └── {dataset}_val.json                                         │
│                     │                                            │
│                     │ LLaMA-Factory DataLoader                  │
│                     ▼                                            │
│  ┌─────────────────────────────────────────────────┐            │
│  │              训练数据集                           │            │
│  │  - 8个数据集自动合并                             │            │
│  │  - 按配置的 max_samples 采样                    │            │
│  └─────────────────────────────────────────────────┘            │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## 3. 模型训练架构

### 3.1 非 KT 模式

```
┌─────────────────────────────────────────────────────────────────┐
│                    标准 LoRA 训练流程                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │   Base Model │    │  LoRA Adapter│    │   Trainer    │      │
│  │   (GPU)      │───▶│  (GPU)       │───▶│  (Standard)  │      │
│  └──────────────┘    └──────────────┘    └──────────────┘      │
│                                                                  │
│  训练配置:                                                       │
│  - 使用标准 Transformers Trainer                                 │
│  - 支持 gradient checkpointing                                  │
│  - 支持 predict_with_generate                                   │
│  - 支持 compute_accuracy                                        │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 KT 模式

```
┌─────────────────────────────────────────────────────────────────┐
│                KT (KTransformers) LoRA 训练流程                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                    MoE Model                              │  │
│  │  ┌────────────────┐    ┌────────────────────────────┐   │  │
│  │  │ Attention      │    │  MoE Experts               │   │  │
│  │  │ (GPU + LoRA)   │    │  (CPU + AMX + LoRA)        │   │  │
│  │  └────────────────┘    └────────────────────────────┘   │  │
│  └──────────────────────────────────────────────────────────┘  │
│                          │                                      │
│                          ▼                                      │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                   KTrainer                                │  │
│  │  - 管理 MoE LoRA + Attention LoRA 参数                    │  │
│  │  - TP 模式下同步 LoRA 指针                                │  │
│  │  - 合并保存 LoRA 权重                                     │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                  │
│  KT 配置:                                                        │
│  - use_kt: true                                                 │
│  - kt_backend: AMXBF16 / AMXINT8 / AMXINT4                     │
│  - kt_num_threads: 60                                           │
│  - kt_tp_enabled: false / true                                  │
│  - disable_gradient_checkpointing: true                         │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## 4. 验证集架构

```
┌─────────────────────────────────────────────────────────────────┐
│                     验证集评估流程                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  YAML 配置:                                                      │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ eval_dataset: cs_arc_challenge_val,cs_arc_easy_val,...   │  │
│  │ eval_strategy: steps                                      │  │
│  │ eval_steps: 500                                           │  │
│  │ per_device_eval_batch_size: 1                             │  │
│  └──────────────────────────────────────────────────────────┘  │
│                          │                                      │
│                          ▼                                      │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │               LLaMA-Factory Workflow                      │  │
│  │                                                           │  │
│  │  loader.py:                                               │  │
│  │  - _get_merged_dataset(eval_dataset, ...)                │  │
│  │  - split_dataset() (如果配置了 val_size)                  │  │
│  │                                                           │  │
│  │  workflow.py:                                             │  │
│  │  - trainer.evaluate(metric_key_prefix="eval")            │  │
│  │  - 输出 eval_loss 指标                                    │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                  │
│  KT 模式限制:                                                    │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ ✓ 支持: do_eval (eval loss)                              │  │
│  │ ✗ 不支持: predict_with_generate                          │  │
│  │ ✗ 不支持: compute_accuracy                               │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## 5. 文件结构

```
LLaMA-Factory-KT/
├── data/
│   ├── commonsense/                      # 转换后的数据集
│   │   ├── arc_challenge_train.json
│   │   ├── arc_challenge_val.json
│   │   ├── arc_easy_train.json
│   │   ├── arc_easy_val.json
│   │   ├── boolq_train.json
│   │   ├── boolq_val.json
│   │   ├── hellaswag_train.json
│   │   ├── hellaswag_val.json
│   │   ├── openbookqa_train.json
│   │   ├── openbookqa_val.json
│   │   ├── piqa_train.json
│   │   ├── piqa_val.json
│   │   ├── social_i_qa_train.json
│   │   ├── social_i_qa_val.json
│   │   ├── winogrande_xl_train.json
│   │   └── winogrande_xl_val.json
│   └── dataset_info.json                 # 数据集注册
├── examples/
│   └── commonsense_benchmark/            # 训练配置
│       ├── deepseek2_lora_sft.yaml
│       ├── deepseek2_lora_sft_kt.yaml
│       ├── deepseek3_lora_sft.yaml
│       ├── deepseek3_lora_sft_kt.yaml
│       ├── qwen3moe_lora_sft.yaml
│       └── qwen3moe_lora_sft_kt.yaml
├── scripts/
│   └── convert_commonsense_datasets.py   # 数据转换脚本
├── docs/
│   └── benchmark/                        # 文档
│       ├── commonsense_requirements.md
│       ├── commonsense_architecture.md
│       ├── commonsense_design.md
│       └── commonsense_testing.md
└── src/
    └── llamafactory/
        ├── data/
        │   └── loader.py                 # 数据加载
        └── train/
            └── sft/
                ├── workflow.py           # 训练流程
                └── kt_trainer.py         # KT 训练器
```

## 6. 关键代码路径

### 6.1 数据加载路径

```
loader.py:get_dataset()
    ├── _get_merged_dataset(data_args.dataset)      # 加载训练集
    ├── _get_merged_dataset(data_args.eval_dataset) # 加载验证集
    └── split_dataset()                              # 划分数据集
```

### 6.2 训练路径

```
workflow.py:run_sft()
    ├── load_tokenizer()
    ├── get_dataset()
    ├── load_model()
    ├── if use_kt:
    │   └── create_kt_trainer()  # 创建 KT Trainer
    │       └── KTrainer()
    │           ├── create_optimizer()    # 包含 MoE LoRA 参数
    │           └── training_step()       # 更新 LoRA 指针
    └── else:
        └── CustomSeq2SeqTrainer()
```

### 6.3 验证路径

```
workflow.py:run_sft()
    └── if training_args.do_eval:
        └── trainer.evaluate()
            ├── 计算 eval_loss
            ├── log_metrics("eval", metrics)
            └── save_metrics("eval", metrics)
```

## 7. 依赖关系

```
┌─────────────────────────────────────────────────────────────────┐
│                        依赖关系图                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  外部依赖:                                                       │
│  ├── transformers                                               │
│  ├── datasets (HuggingFace)                                     │
│  ├── peft (LoRA)                                                │
│  └── ktransformers (KT 后端)                                    │
│                                                                  │
│  内部依赖:                                                       │
│  ├── data/dataset_info.json ──▶ loader.py                       │
│  ├── loader.py ──▶ workflow.py                                  │
│  ├── workflow.py ──▶ kt_trainer.py (KT 模式)                    │
│  └── kt_trainer.py ──▶ ktransformers                            │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```
