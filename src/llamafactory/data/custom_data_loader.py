"""
数据加载与预处理模块 (从 lora-without-regret 迁移)

支持的数据集:
- tulu3_coding: Tulu3 Coding 子集 (代码生成任务)
- gsm8k: GSM8K 数学推理数据集
"""

import os
import json
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizer


# 数据集路径配置 - 使用 lora-without-regret 仓库中的数据
LORA_WITHOUT_REGRET_PATH = "/home/lpl/lora-without-regret"

DATASET_PATHS = {
    "tulu3_coding": {
        "train": os.path.join(LORA_WITHOUT_REGRET_PATH, "datasets/allenai_tulu-3-sft-personas-code/train.jsonl"),
    },
    "gsm8k": {
        "train": os.path.join(LORA_WITHOUT_REGRET_PATH, "datasets/openai_gsm8k/train.jsonl"),
        "test": os.path.join(LORA_WITHOUT_REGRET_PATH, "datasets/openai_gsm8k/test.jsonl"),
    },
}


@dataclass
class DataConfig:
    """数据配置"""
    dataset_name: str = "tulu3_coding"
    max_length: int = 2048
    max_samples: Optional[int] = None
    eval_split_ratio: float = 0.05
    num_workers: int = 4


def load_jsonl(file_path: str) -> List[Dict]:
    """加载 JSONL 文件"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def load_dataset_by_name(
    name: str,
    split: str = "train",
    max_samples: Optional[int] = None
) -> List[Dict]:
    """
    统一数据加载接口

    Args:
        name: 数据集名称 ("tulu3_coding" | "gsm8k")
        split: 数据集划分 ("train" | "test")
        max_samples: 最大样本数 (用于数据规模实验)

    Returns:
        数据列表
    """
    if name not in DATASET_PATHS:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(DATASET_PATHS.keys())}")

    paths = DATASET_PATHS[name]
    if split not in paths:
        raise ValueError(f"Split '{split}' not available for dataset '{name}'. Available: {list(paths.keys())}")

    file_path = paths[split]
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found: {file_path}")

    data = load_jsonl(file_path)

    if max_samples is not None and max_samples < len(data):
        data = data[:max_samples]

    return data


def format_tulu3_coding_example(example: Dict) -> Dict[str, str]:
    """
    格式化 Tulu3 Coding 数据为 SFT 格式

    Tulu3 格式: {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
    """
    messages = example.get("messages", [])

    # 构建对话文本
    text_parts = []
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "user":
            text_parts.append(f"<|user|>\n{content}")
        elif role == "assistant":
            text_parts.append(f"<|assistant|>\n{content}")

    full_text = "\n".join(text_parts)

    # 分离 input 和 output (用于计算 loss)
    input_text = ""
    output_text = ""
    for i, msg in enumerate(messages):
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "user":
            input_text += f"<|user|>\n{content}\n"
        elif role == "assistant":
            if i == len(messages) - 1:
                input_text += "<|assistant|>\n"
                output_text = content
            else:
                input_text += f"<|assistant|>\n{content}\n"

    return {
        "input": input_text.strip(),
        "output": output_text,
        "full_text": full_text,
    }


def format_gsm8k_example(example: Dict) -> Dict[str, str]:
    """
    格式化 GSM8K 数据为 SFT 格式

    GSM8K 格式: {"question": "...", "answer": "..."}
    """
    question = example.get("question", "")
    answer = example.get("answer", "")

    # 构建 instruction 格式
    input_text = f"<|user|>\nSolve the following math problem step by step:\n{question}\n<|assistant|>\n"
    output_text = answer

    return {
        "input": input_text,
        "output": output_text,
        "full_text": input_text + output_text,
    }


def format_example(example: Dict, dataset_name: str) -> Dict[str, str]:
    """根据数据集类型格式化样本"""
    if dataset_name == "tulu3_coding":
        return format_tulu3_coding_example(example)
    elif dataset_name == "gsm8k":
        return format_gsm8k_example(example)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


class SFTDataset(Dataset):
    """SFT 数据集"""

    def __init__(
        self,
        data: List[Dict],
        tokenizer: PreTrainedTokenizer,
        dataset_name: str,
        max_length: int = 2048,
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.dataset_name = dataset_name
        self.max_length = max_length

        # 预处理数据
        self.processed_data = []
        for example in data:
            formatted = format_example(example, dataset_name)
            self.processed_data.append(formatted)

    def __len__(self) -> int:
        return len(self.processed_data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        example = self.processed_data[idx]

        # Tokenize input + output
        full_text = example["full_text"]
        input_text = example["input"]

        # Tokenize 完整序列
        full_encoding = self.tokenizer(
            full_text,
            max_length=self.max_length,
            truncation=True,
            padding=False,
            return_tensors=None,
        )

        # Tokenize input 部分 (用于确定 label mask)
        input_encoding = self.tokenizer(
            input_text,
            max_length=self.max_length,
            truncation=True,
            padding=False,
            return_tensors=None,
        )

        input_ids = full_encoding["input_ids"]
        attention_mask = full_encoding["attention_mask"]

        # 构建 labels: input 部分设为 -100 (不计算 loss)
        labels = input_ids.copy()
        input_len = len(input_encoding["input_ids"])
        labels[:input_len] = [-100] * input_len

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


class DataCollatorForSFT:
    """SFT 数据整理器 (支持动态 padding)"""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        padding: str = "longest",
        max_length: Optional[int] = None,
        pad_to_multiple_of: int = 8,
    ):
        self.tokenizer = tokenizer
        self.padding = padding
        self.max_length = max_length
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        # 提取各字段
        input_ids = [f["input_ids"] for f in features]
        attention_mask = [f["attention_mask"] for f in features]
        labels = [f["labels"] for f in features]

        # 计算 padding 长度
        max_len = max(len(ids) for ids in input_ids)
        if self.pad_to_multiple_of:
            max_len = ((max_len + self.pad_to_multiple_of - 1) // self.pad_to_multiple_of) * self.pad_to_multiple_of
        if self.max_length:
            max_len = min(max_len, self.max_length)

        # Padding
        pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0

        padded_input_ids = []
        padded_attention_mask = []
        padded_labels = []

        for ids, mask, lab in zip(input_ids, attention_mask, labels):
            pad_len = max_len - len(ids)
            # Right padding
            padded_input_ids.append(
                torch.cat([ids, torch.full((pad_len,), pad_token_id, dtype=torch.long)])
            )
            padded_attention_mask.append(
                torch.cat([mask, torch.zeros(pad_len, dtype=torch.long)])
            )
            padded_labels.append(
                torch.cat([lab, torch.full((pad_len,), -100, dtype=torch.long)])
            )

        return {
            "input_ids": torch.stack(padded_input_ids),
            "attention_mask": torch.stack(padded_attention_mask),
            "labels": torch.stack(padded_labels),
        }


# 任务名称到数据集名称的映射
TASK_TO_DATASET = {
    "coding": "tulu3_coding",
    "gsm8k": "gsm8k",
}


def load_dataset_for_task(
    task: str,
    tokenizer: PreTrainedTokenizer,
    max_length: int = 2048,
    max_samples: Optional[int] = None,
    eval_split_ratio: float = 0.05,
) -> Tuple[SFTDataset, SFTDataset]:
    """
    根据任务名称加载数据集

    Args:
        task: 任务名称 ("coding" | "gsm8k")
        tokenizer: Tokenizer
        max_length: 最大序列长度
        max_samples: 最大样本数 (可选)
        eval_split_ratio: 验证集比例

    Returns:
        (train_dataset, eval_dataset)
    """
    if task not in TASK_TO_DATASET:
        raise ValueError(f"Unknown task: {task}. Available: {list(TASK_TO_DATASET.keys())}")

    dataset_name = TASK_TO_DATASET[task]

    # 加载数据
    train_data = load_dataset_by_name(
        dataset_name,
        split="train",
        max_samples=max_samples
    )

    # 分割训练集和验证集
    if eval_split_ratio > 0:
        split_idx = int(len(train_data) * (1 - eval_split_ratio))
        eval_data = train_data[split_idx:]
        train_data = train_data[:split_idx]
    else:
        eval_data = []

    # 创建 SFTDataset
    train_dataset = SFTDataset(
        train_data,
        tokenizer,
        dataset_name,
        max_length
    )

    eval_dataset = SFTDataset(
        eval_data,
        tokenizer,
        dataset_name,
        max_length
    ) if eval_data else SFTDataset([], tokenizer, dataset_name, max_length)

    return train_dataset, eval_dataset
