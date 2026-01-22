# Commonsense Benchmark - 使用测试文档

## 1. 前置条件

### 1.1 环境要求

- Python 3.10+
- CUDA 11.8+ (非 KT 模式)
- Intel AMX 支持 (KT 模式)
- 足够的 GPU/CPU 内存

### 1.2 依赖安装

```bash
# 激活 conda 环境
source /mnt/data/lpl/anaconda3/bin/activate
conda activate ref-llama

# 确认依赖
pip show transformers datasets peft
```

### 1.3 模型路径检查

```bash
# 检查模型是否存在
ls -la /mnt/data3/models/DeepSeek-V2-Lite-Chat
ls -la /mnt/data3/models/DeepSeek-V3.1-Base-BF16
ls -la /mnt/data3/models/Qwen3-30B-A3B
```

## 2. 数据集准备

### 2.1 检查原始数据集

```bash
# 检查原始数据集目录
ls -la /mnt/data/lpl/lora_datasets/commonsense_datasets/
```

预期输出:
```
arc_challenge/
arc_easy/
boolq/
hellaswag/
openbookqa/
piqa/
social_i_qa/
winogrande_xl/
```

### 2.2 运行数据转换

```bash
cd /home/lpl/LLaMA-Factory-KT

# 运行转换脚本
python scripts/convert_commonsense_datasets.py
```

预期输出:
```
============================================================
Commonsense Dataset Conversion
============================================================
...
Conversion Summary
============================================================
Dataset              Train      Validation
----------------------------------------
arc_challenge        1119       299
arc_easy             2251       570
boolq                9427       3270
hellaswag            39905      10042
openbookqa           4957       500
piqa                 16113      1838
social_i_qa          33410      1954
winogrande_xl        40398      1267
----------------------------------------
Total                147580     19740

Conversion complete!
```

### 2.3 验证转换结果

```bash
# 检查转换后的数据集
ls -la data/commonsense/

# 查看样本数据
head -5 data/commonsense/arc_challenge_train.json
```

预期格式:
```json
[
  {
    "instruction": "Question: ...\nChoices:\nA. ...\nB. ...\nC. ...\nD. ...\nAnswer:",
    "input": "",
    "output": "A"
  },
  ...
]
```

## 3. 训练测试

### 3.1 非 KT 模式训练

#### DeepSeek-V2-Lite

```bash
cd /home/lpl/LLaMA-Factory-KT

# 单卡训练
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train \
    examples/commonsense_benchmark/deepseek2_lora_sft.yaml
```

#### DeepSeek-V3

```bash
# 单卡训练（需要大显存）
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train \
    examples/commonsense_benchmark/deepseek3_lora_sft.yaml
```

#### Qwen3-30B-A3B

```bash
# 单卡训练
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train \
    examples/commonsense_benchmark/qwen3moe_lora_sft.yaml
```

### 3.2 KT 模式训练

#### DeepSeek-V2-Lite (KT)

```bash
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train \
    examples/commonsense_benchmark/deepseek2_lora_sft_kt.yaml
```

#### DeepSeek-V3 (KT)

```bash
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train \
    examples/commonsense_benchmark/deepseek3_lora_sft_kt.yaml
```

#### Qwen3-30B-A3B (KT)

```bash
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train \
    examples/commonsense_benchmark/qwen3moe_lora_sft_kt.yaml
```

## 4. 验证测试

### 4.1 训练过程验证

训练日志中应包含以下信息：

```
[INFO] Loading dataset cs_arc_challenge...
[INFO] Loading dataset cs_arc_easy...
...
[INFO] training example:
input_ids: ...
labels: ...
...
[INFO] eval example:
...
```

### 4.2 评估输出验证

每 500 步应输出评估指标：

```
{'eval_loss': 1.234, 'eval_runtime': ...}
```

### 4.3 保存验证

检查输出目录：

```bash
ls -la saves/deepseek2-lite/lora/commonsense/
```

预期文件：
- `adapter_config.json`
- `adapter_model.safetensors`
- `trainer_state.json`
- `training_loss.png` (训练完成后)

## 5. 快速验证测试

### 5.1 小规模测试配置

创建测试配置（限制样本数）：

```bash
# 复制配置文件
cp examples/commonsense_benchmark/qwen3moe_lora_sft.yaml \
   examples/commonsense_benchmark/qwen3moe_lora_sft_test.yaml

# 修改配置（减少样本数）
sed -i 's/max_samples: 100000/max_samples: 100/' \
    examples/commonsense_benchmark/qwen3moe_lora_sft_test.yaml
sed -i 's/num_train_epochs: 3.0/num_train_epochs: 1.0/' \
    examples/commonsense_benchmark/qwen3moe_lora_sft_test.yaml
```

### 5.2 运行快速测试

```bash
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train \
    examples/commonsense_benchmark/qwen3moe_lora_sft_test.yaml
```

## 6. 常见问题排查

### 6.1 数据集加载失败

**问题**: `FileNotFoundError: ... commonsense/xxx.json`

**解决方案**:
```bash
# 重新运行数据转换
python scripts/convert_commonsense_datasets.py

# 检查 dataset_info.json 注册
grep "cs_arc" data/dataset_info.json
```

### 6.2 显存不足

**问题**: `CUDA out of memory`

**解决方案**:
```yaml
# 减少 batch size
per_device_train_batch_size: 1

# 增加梯度累积
gradient_accumulation_steps: 16

# 启用 gradient checkpointing (非 KT 模式)
gradient_checkpointing: true
```

### 6.3 KT 后端错误

**问题**: `KTransformers not available`

**解决方案**:
```bash
# 检查 KTransformers 安装
pip show ktransformers

# 检查 AMX 支持
lscpu | grep amx
```

### 6.4 评估失败

**问题**: `eval_dataset is empty`

**解决方案**:
```bash
# 检查验证集文件
ls -la data/commonsense/*_val.json

# 检查 dataset_info.json 注册
grep "_val" data/dataset_info.json
```

## 7. 测试检查清单

### 7.1 数据准备检查

- [ ] 原始数据集存在
- [ ] 转换脚本执行成功
- [ ] 16 个 JSON 文件生成
- [ ] dataset_info.json 已更新

### 7.2 配置检查

- [ ] 6 个 YAML 文件存在
- [ ] 模型路径正确
- [ ] 数据集名称正确

### 7.3 训练检查

- [ ] 训练正常启动
- [ ] 数据集加载成功
- [ ] 评估正常执行
- [ ] 模型保存成功

### 7.4 KT 模式检查

- [ ] KT 后端正常初始化
- [ ] MoE LoRA 参数正确添加
- [ ] 验证集评估正常（eval loss）

## 8. 性能基准

### 8.1 预期训练时间

| 模型 | 模式 | 100K 样本训练时间（估计） |
|------|------|--------------------------|
| DeepSeek-V2-Lite | 非 KT | ~4-6 小时 |
| DeepSeek-V2-Lite | KT | ~8-12 小时 |
| DeepSeek-V3 | 非 KT | ~12-18 小时 |
| DeepSeek-V3 | KT | ~24-36 小时 |
| Qwen3-30B-A3B | 非 KT | ~8-12 小时 |
| Qwen3-30B-A3B | KT | ~16-24 小时 |

### 8.2 显存使用（估计）

| 模型 | 模式 | 显存占用 |
|------|------|----------|
| DeepSeek-V2-Lite | 非 KT | ~20-30 GB |
| DeepSeek-V2-Lite | KT | ~8-12 GB |
| DeepSeek-V3 | 非 KT | ~60-80 GB |
| DeepSeek-V3 | KT | ~16-24 GB |
| Qwen3-30B-A3B | 非 KT | ~30-40 GB |
| Qwen3-30B-A3B | KT | ~10-15 GB |

## 9. 联系与支持

如遇到问题，请检查：
1. 日志文件中的错误信息
2. 本文档的常见问题排查部分
3. LLaMA-Factory 官方文档
