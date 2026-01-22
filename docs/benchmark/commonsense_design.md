# Commonsense Benchmark - 功能设计文档

## 1. 数据集转换设计

### 1.1 转换格式

所有数据集统一转换为 LLaMA-Factory 的 alpaca 格式：

```json
{
  "instruction": "问题描述和选项",
  "input": "",
  "output": "正确答案"
}
```

### 1.2 各数据集转换逻辑

#### 1.2.1 ARC (Challenge/Easy) & OpenBookQA

**输入格式**:
```json
{
  "question": "问题文本",
  "choices": {
    "text": ["选项A", "选项B", "选项C", "选项D"],
    "label": ["A", "B", "C", "D"]
  },
  "answerKey": "A"
}
```

**输出格式**:
```json
{
  "instruction": "Question: 问题文本\nChoices:\nA. 选项A\nB. 选项B\nC. 选项C\nD. 选项D\nAnswer:",
  "input": "",
  "output": "A"
}
```

#### 1.2.2 BoolQ

**输入格式**:
```json
{
  "passage": "文章内容",
  "question": "问题",
  "answer": true/false
}
```

**输出格式**:
```json
{
  "instruction": "Passage: 文章内容\n\nQuestion: 问题\nAnswer (Yes/No):",
  "input": "",
  "output": "Yes/No"
}
```

#### 1.2.3 HellaSwag

**输入格式**:
```json
{
  "ctx": "上下文",
  "endings": ["续写0", "续写1", "续写2", "续写3"],
  "label": 0-3
}
```

**输出格式**:
```json
{
  "instruction": "Complete the sentence:\n上下文\nChoices:\nA. 续写0\nB. 续写1\nC. 续写2\nD. 续写3\nAnswer:",
  "input": "",
  "output": "A/B/C/D"
}
```

#### 1.2.4 PIQA

**输入格式**:
```json
{
  "goal": "目标",
  "sol1": "方案1",
  "sol2": "方案2",
  "label": 0/1
}
```

**输出格式**:
```json
{
  "instruction": "Goal: 目标\nWhich solution is correct?\nA. 方案1\nB. 方案2\nAnswer:",
  "input": "",
  "output": "A/B"
}
```

#### 1.2.5 Social IQA

**输入格式**:
```json
{
  "context": "场景",
  "question": "问题",
  "answerA": "选项A",
  "answerB": "选项B",
  "answerC": "选项C",
  "label": "1/2/3"
}
```

**输出格式**:
```json
{
  "instruction": "Context: 场景\nQuestion: 问题\nChoices:\nA. 选项A\nB. 选项B\nC. 选项C\nAnswer:",
  "input": "",
  "output": "A/B/C"
}
```

#### 1.2.6 Winogrande

**输入格式**:
```json
{
  "sentence": "句子（含 _ 占位符）",
  "option1": "选项1",
  "option2": "选项2",
  "answer": "1/2"
}
```

**输出格式**:
```json
{
  "instruction": "Fill in the blank with the correct option:\n句子\nOptions:\n1. 选项1\n2. 选项2\nAnswer:",
  "input": "",
  "output": "1/2"
}
```

## 2. YAML 配置设计

### 2.1 公共配置项

```yaml
### method
stage: sft
do_train: true
finetuning_type: lora
lora_rank: 8
lora_target: all

### dataset
dataset: cs_arc_challenge,cs_arc_easy,cs_boolq,cs_hellaswag,cs_openbookqa,cs_piqa,cs_social_iqa,cs_winogrande
cutoff_len: 2048
max_samples: 100000
overwrite_cache: true
preprocessing_num_workers: 16
dataloader_num_workers: 4

### train
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
learning_rate: 1.0e-4
num_train_epochs: 3.0
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true
ddp_timeout: 180000000

### eval
eval_dataset: cs_arc_challenge_val,cs_arc_easy_val,cs_boolq_val,cs_hellaswag_val,cs_openbookqa_val,cs_piqa_val,cs_social_iqa_val,cs_winogrande_val
per_device_eval_batch_size: 1
eval_strategy: steps
eval_steps: 500
```

### 2.2 模型特定配置

| 模型 | template | model_name_or_path |
|------|----------|-------------------|
| DeepSeek-V2-Lite | deepseek2 | /mnt/data3/models/DeepSeek-V2-Lite-Chat |
| DeepSeek-V3 | deepseek3 | /mnt/data3/models/DeepSeek-V3.1-Base-BF16 |
| Qwen3-30B-A3B | qwen3_nothink | /mnt/data3/models/Qwen3-30B-A3B |

### 2.3 KT 特定配置

```yaml
### ktransformers
use_kt: true
kt_backend: AMXBF16             # 后端类型
kt_num_threads: 60              # CPU 线程数
kt_tp_enabled: false            # NUMA 张量并行
kt_num_gpu_experts: 0           # GPU 专家数

### 必须禁用
disable_gradient_checkpointing: true
```

## 3. 数据集注册设计

### 3.1 命名规范

- 训练集: `cs_{dataset_name}` (如 `cs_arc_challenge`)
- 验证集: `cs_{dataset_name}_val` (如 `cs_arc_challenge_val`)

### 3.2 dataset_info.json 条目

```json
{
  "cs_arc_challenge": {
    "file_name": "commonsense/arc_challenge_train.json"
  },
  "cs_arc_challenge_val": {
    "file_name": "commonsense/arc_challenge_val.json"
  }
}
```

## 4. 训练输出设计

### 4.1 输出目录结构

```
saves/
├── deepseek2-lite/
│   └── lora/
│       └── commonsense/
│           ├── adapter_config.json
│           ├── adapter_model.safetensors
│           ├── trainer_state.json
│           └── training_loss.png
├── deepseek2-lite-kt/
│   └── lora/
│       └── commonsense/
│           └── ...
├── deepseek3/
│   └── lora/
│       └── commonsense/
│           └── ...
├── deepseek3-kt/
│   └── lora/
│       └── commonsense/
│           └── ...
├── qwen3-30b-a3b/
│   └── lora/
│       └── commonsense/
│           └── ...
└── qwen3-30b-a3b-kt/
    └── lora/
        └── commonsense/
            └── ...
```

### 4.2 日志输出

- `logging_steps: 10` - 每 10 步输出训练 loss
- `save_steps: 500` - 每 500 步保存 checkpoint
- `eval_steps: 500` - 每 500 步评估验证集
- `plot_loss: true` - 训练完成后生成 loss 图

## 5. 验证集评估设计

### 5.1 评估指标

| 模式 | 支持的指标 |
|------|-----------|
| 非 KT | eval_loss, predict_with_generate, compute_accuracy |
| KT | eval_loss |

### 5.2 评估配置

```yaml
# 基本评估配置
eval_dataset: cs_arc_challenge_val,cs_arc_easy_val,...
per_device_eval_batch_size: 1
eval_strategy: steps
eval_steps: 500

# 可选（仅非 KT 模式）
# predict_with_generate: true
# compute_accuracy: true
```

## 6. 错误处理设计

### 6.1 数据转换错误处理

```python
for example in ds:
    try:
        converted.append(converter(example))
    except Exception as e:
        print(f"Warning: Failed to convert example: {e}")
        continue
```

### 6.2 验证集路径检查

转换脚本会检查每个数据集的 train 和 validation 目录是否存在：

```python
if not source_path.exists():
    print(f"Warning: {source_path} does not exist, skipping...")
    return []
```

## 7. 扩展性设计

### 7.1 添加新数据集

1. 在 `convert_commonsense_datasets.py` 中添加转换函数
2. 将数据集名称添加到 `DATASETS` 列表
3. 在 `get_converter()` 中注册转换函数
4. 运行转换脚本
5. 更新 `dataset_info.json`
6. 更新 YAML 配置文件中的 `dataset` 和 `eval_dataset`

### 7.2 添加新模型

1. 确定模型路径和 template 名称
2. 复制现有 YAML 配置文件
3. 修改 `model_name_or_path` 和 `template`
4. 修改 `output_dir`
5. 如需 KT 版本，添加 KT 配置项

## 8. 性能优化设计

### 8.1 数据加载优化

```yaml
preprocessing_num_workers: 16    # 数据预处理并行数
dataloader_num_workers: 4        # DataLoader 工作进程数
overwrite_cache: true            # 强制重新处理数据
```

### 8.2 KT 训练优化

```yaml
kt_backend: AMXBF16              # 使用 AMX 加速
kt_num_threads: 60               # CPU 线程数（根据实际 CPU 核心数调整）
kt_num_gpu_experts: 0            # 专家全部在 CPU 上运行
```

### 8.3 内存优化

```yaml
per_device_train_batch_size: 1   # 小 batch size
gradient_accumulation_steps: 8   # 梯度累积补偿
bf16: true                       # 使用 BF16 精度
```
