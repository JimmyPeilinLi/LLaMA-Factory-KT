#!/usr/bin/env python3
"""
Convert commonsense reasoning datasets to LLaMA-Factory alpaca format.

This script converts 8 commonsense datasets from HuggingFace arrow format
to LLaMA-Factory's alpaca format for SFT training.

Datasets:
- arc_challenge, arc_easy: AI2 Reasoning Challenge
- boolq: Boolean Questions
- hellaswag: Sentence Completion
- openbookqa: Open Book QA
- piqa: Physical Intuition QA
- social_i_qa: Social Interaction QA
- winogrande_xl: Winograd Schema Challenge

Usage:
    python scripts/convert_commonsense_datasets.py
"""

import json
import os
from pathlib import Path
from datasets import load_from_disk


# Configuration
SOURCE_DIR = Path("/mnt/data/lpl/lora_datasets/commonsense_datasets")
OUTPUT_DIR = Path("/home/lpl/LLaMA-Factory-KT/data/commonsense")

# Dataset names
DATASETS = [
    "arc_challenge",
    "arc_easy",
    "boolq",
    "hellaswag",
    "openbookqa",
    "piqa",
    "social_i_qa",
    "winogrande_xl"
]


def convert_arc(example: dict, dataset_name: str) -> dict:
    """Convert ARC (arc_challenge, arc_easy) format to alpaca."""
    question = example["question"]
    choices = example["choices"]
    answer_key = example["answerKey"]

    # Build choices string
    choice_lines = []
    for label, text in zip(choices["label"], choices["text"]):
        choice_lines.append(f"{label}. {text}")
    choices_str = "\n".join(choice_lines)

    instruction = f"Question: {question}\nChoices:\n{choices_str}\nAnswer:"

    return {
        "instruction": instruction,
        "input": "",
        "output": answer_key
    }


def convert_openbookqa(example: dict) -> dict:
    """Convert OpenBookQA format to alpaca."""
    question = example["question_stem"]
    choices = example["choices"]
    answer_key = example["answerKey"]

    # Build choices string
    choice_lines = []
    for label, text in zip(choices["label"], choices["text"]):
        choice_lines.append(f"{label}. {text}")
    choices_str = "\n".join(choice_lines)

    instruction = f"Question: {question}\nChoices:\n{choices_str}\nAnswer:"

    return {
        "instruction": instruction,
        "input": "",
        "output": answer_key
    }


def convert_boolq(example: dict) -> dict:
    """Convert BoolQ format to alpaca."""
    passage = example["passage"]
    question = example["question"]
    answer = example["answer"]

    instruction = f"Passage: {passage}\n\nQuestion: {question}\nAnswer (Yes/No):"
    output = "Yes" if answer else "No"

    return {
        "instruction": instruction,
        "input": "",
        "output": output
    }


def convert_hellaswag(example: dict) -> dict:
    """Convert HellaSwag format to alpaca."""
    ctx = example["ctx"]
    endings = example["endings"]
    label = example["label"]

    # Map label to letter
    label_map = {0: "A", 1: "B", 2: "C", 3: "D"}
    answer = label_map.get(int(label), "A")

    # Build choices string
    choice_lines = []
    for i, ending in enumerate(endings):
        letter = label_map[i]
        choice_lines.append(f"{letter}. {ending}")
    choices_str = "\n".join(choice_lines)

    instruction = f"Complete the sentence:\n{ctx}\nChoices:\n{choices_str}\nAnswer:"

    return {
        "instruction": instruction,
        "input": "",
        "output": answer
    }


def convert_piqa(example: dict) -> dict:
    """Convert PIQA format to alpaca."""
    goal = example["goal"]
    sol1 = example["sol1"]
    sol2 = example["sol2"]
    label = example["label"]

    # Map label to letter
    answer = "A" if label == 0 else "B"

    instruction = f"Goal: {goal}\nWhich solution is correct?\nA. {sol1}\nB. {sol2}\nAnswer:"

    return {
        "instruction": instruction,
        "input": "",
        "output": answer
    }


def convert_social_iqa(example: dict) -> dict:
    """Convert Social IQA format to alpaca."""
    context = example["context"]
    question = example["question"]
    answer_a = example["answerA"]
    answer_b = example["answerB"]
    answer_c = example["answerC"]
    label = example["label"]

    # Map label to letter (label is 1-indexed as string)
    label_map = {"1": "A", "2": "B", "3": "C"}
    answer = label_map.get(str(label), "A")

    instruction = f"Context: {context}\nQuestion: {question}\nChoices:\nA. {answer_a}\nB. {answer_b}\nC. {answer_c}\nAnswer:"

    return {
        "instruction": instruction,
        "input": "",
        "output": answer
    }


def convert_winogrande(example: dict) -> dict:
    """Convert Winogrande format to alpaca."""
    sentence = example["sentence"]
    option1 = example["option1"]
    option2 = example["option2"]
    answer = example["answer"]

    instruction = f"Fill in the blank with the correct option:\n{sentence}\nOptions:\n1. {option1}\n2. {option2}\nAnswer:"

    return {
        "instruction": instruction,
        "input": "",
        "output": answer
    }


def get_converter(dataset_name: str):
    """Get the appropriate converter function for a dataset."""
    converters = {
        "arc_challenge": lambda x: convert_arc(x, "arc_challenge"),
        "arc_easy": lambda x: convert_arc(x, "arc_easy"),
        "boolq": convert_boolq,
        "hellaswag": convert_hellaswag,
        "openbookqa": convert_openbookqa,
        "piqa": convert_piqa,
        "social_i_qa": convert_social_iqa,
        "winogrande_xl": convert_winogrande,
    }
    return converters.get(dataset_name)


def convert_dataset(dataset_name: str, split: str) -> list[dict]:
    """Convert a single dataset split to alpaca format."""
    source_path = SOURCE_DIR / dataset_name / split

    if not source_path.exists():
        print(f"  Warning: {source_path} does not exist, skipping...")
        return []

    # Load dataset
    ds = load_from_disk(str(source_path))

    # Get converter
    converter = get_converter(dataset_name)
    if converter is None:
        print(f"  Error: No converter found for {dataset_name}")
        return []

    # Convert each example
    converted = []
    for example in ds:
        try:
            converted.append(converter(example))
        except Exception as e:
            print(f"  Warning: Failed to convert example: {e}")
            continue

    return converted


def save_dataset(data: list[dict], output_path: Path):
    """Save converted dataset to JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"  Saved {len(data)} examples to {output_path}")


def main():
    """Main conversion function."""
    print("=" * 60)
    print("Commonsense Dataset Conversion")
    print("=" * 60)
    print(f"Source: {SOURCE_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    print()

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Statistics
    stats = {}

    for dataset_name in DATASETS:
        print(f"\nProcessing {dataset_name}...")
        stats[dataset_name] = {"train": 0, "validation": 0}

        # Convert train split
        train_data = convert_dataset(dataset_name, "train")
        if train_data:
            output_path = OUTPUT_DIR / f"{dataset_name}_train.json"
            save_dataset(train_data, output_path)
            stats[dataset_name]["train"] = len(train_data)

        # Convert validation split
        val_data = convert_dataset(dataset_name, "validation")
        if val_data:
            output_path = OUTPUT_DIR / f"{dataset_name}_val.json"
            save_dataset(val_data, output_path)
            stats[dataset_name]["validation"] = len(val_data)

    # Print summary
    print("\n" + "=" * 60)
    print("Conversion Summary")
    print("=" * 60)
    print(f"{'Dataset':<20} {'Train':<10} {'Validation':<10}")
    print("-" * 40)

    total_train = 0
    total_val = 0
    for name, counts in stats.items():
        print(f"{name:<20} {counts['train']:<10} {counts['validation']:<10}")
        total_train += counts["train"]
        total_val += counts["validation"]

    print("-" * 40)
    print(f"{'Total':<20} {total_train:<10} {total_val:<10}")
    print()
    print("Conversion complete!")
    print(f"Output files are in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
