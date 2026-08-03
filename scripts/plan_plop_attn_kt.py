#!/usr/bin/env python3
# Copyright 2026 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

r"""Build a frozen PLoP-Attn NFN plan. This process must exit before the final SFT process starts."""

import argparse
import hashlib
import subprocess
from pathlib import Path
from typing import Any

import torch
from accelerate import Accelerator, DataLoaderConfiguration
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Subset

from llamafactory.data import SFTDataCollatorWith4DAttentionMask, get_dataset, get_template_and_fix_tokenizer
from llamafactory.hparams import get_train_args
from llamafactory.model import load_model, load_tokenizer
from llamafactory.model.lora_variants.common import (
    get_logical_weight,
    python_package_source_sha256,
    resolve_eligible_gpu_lora_targets,
    tensor_to_full,
    verify_kt_fused_expert_lora_contract,
)
from llamafactory.model.lora_variants.plop_attn import PLoPAttnScorer, write_plop_artifacts


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plan a frozen attention-only PLoP target set without backward.")
    parser.add_argument("config", help="Normal LLaMA-Factory SFT YAML used for exact tokenizer/data/model settings.")
    parser.add_argument("--output-dir", required=True, help="Directory for the four immutable PLoP artifacts.")
    parser.add_argument("--probe-examples", type=int, default=100)
    parser.add_argument("--probe-seed", type=int, default=20260803)
    parser.add_argument("--select-k", type=int, choices=[2, 3, 4, 5], default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    return parser.parse_args()


def _load_probe_config(path: str) -> dict[str, Any]:
    values = OmegaConf.to_container(OmegaConf.load(Path(path).absolute()), resolve=True)
    if not isinstance(values, dict):
        raise TypeError("PLoP-Attn planner YAML must contain a mapping.")
    values = dict(values)
    values.update(
        {
            "do_train": True,
            "do_eval": False,
            "do_predict": False,
            "kt_lora_variant": "vanilla",
            "use_lora_expert": False,
            "adapter_name_or_path": None,
            "create_new_adapter": False,
            "report_to": "none",
        }
    )
    return values


def _git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True, cwd=Path(__file__).parents[1]
    )
    return result.stdout.strip()


def _probe_data_hash(dataset, indices: list[int]) -> str:
    digest = hashlib.sha256()
    for index in indices:
        example = dataset[index]
        digest.update(str(index).encode("utf-8"))
        for key in sorted(example):
            value = example[key]
            digest.update(key.encode("utf-8"))
            if isinstance(value, str):
                digest.update(value.encode("utf-8"))
            else:
                digest.update(torch.as_tensor(value).contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _target_weight_hashes(model: torch.nn.Module, target_names: tuple[str, ...]) -> dict[str, str]:
    hashes = {}
    for name in target_names:
        weight = get_logical_weight(model.get_submodule(name))
        if weight is None:
            raise RuntimeError(f"PLoP-Attn cannot hash the logical base weight at {name}.")
        full_weight = tensor_to_full(weight).detach().contiguous().view(torch.uint8).cpu()
        hashes[name] = hashlib.sha256(full_weight.numpy().tobytes()).hexdigest()
    return hashes


def main() -> None:
    cli_args = _parse_args()
    raw_args = _load_probe_config(cli_args.config)
    model_args, data_args, training_args, finetuning_args, _ = get_train_args(raw_args)
    if not model_args.use_kt:
        raise ValueError("PLoP-Attn planning requires `use_kt: true`.")
    if cli_args.probe_examples <= 0 or cli_args.batch_size <= 0:
        raise ValueError("PLoP-Attn probe examples and batch size must be positive.")

    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    dataset_module = get_dataset(template, model_args, data_args, training_args, stage="sft", **tokenizer_module)
    train_dataset = dataset_module.get("train_dataset")
    if train_dataset is None:
        raise ValueError("PLoP-Attn planning requires the processed SFT training split.")
    probe_count = min(cli_args.probe_examples, len(train_dataset))
    probe_indices = list(range(probe_count))
    probe_dataset = Subset(train_dataset, probe_indices)

    model = load_model(tokenizer, model_args, finetuning_args, is_trainable=False)
    data_collator = SFTDataCollatorWith4DAttentionMask(
        template=template,
        model=model,
        pad_to_multiple_of=8,
        label_pad_token_id=-100,
        block_diag_attn=model_args.block_diag_attn,
        neat_packing=data_args.neat_packing,
        attn_implementation=getattr(model.config, "_attn_implementation", None),
        compute_dtype=model_args.compute_dtype,
        **tokenizer_module,
    )
    dataloader = DataLoader(
        probe_dataset,
        batch_size=cli_args.batch_size,
        shuffle=False,
        collate_fn=data_collator,
        num_workers=0,
    )
    accelerator = Accelerator(dataloader_config=DataLoaderConfiguration(even_batches=False))
    model, dataloader = accelerator.prepare(model, dataloader)
    model.eval()
    unwrapped_model = accelerator.unwrap_model(model)
    fused_wrapper_count = verify_kt_fused_expert_lora_contract(
        unwrapped_model, expected_rank=finetuning_args.lora_rank
    )
    eligible = resolve_eligible_gpu_lora_targets(unwrapped_model, require_gpu=True)
    before_hashes = _target_weight_hashes(unwrapped_model, eligible.exact_target_names)
    if any(parameter.grad is not None for parameter in model.parameters()):
        raise RuntimeError("PLoP-Attn probe model unexpectedly has gradients before scoring.")

    scorer = PLoPAttnScorer(unwrapped_model, eligible, seed=cli_args.probe_seed)
    try:
        for batch_index, batch in enumerate(dataloader):
            attention_mask = batch.get("attention_mask")
            model_inputs = {key: value for key, value in batch.items() if key != "labels"}
            with scorer.batch(batch_index, attention_mask):
                model(**model_inputs)
        scores = scorer.finalize()
    finally:
        scorer.close()

    if any(parameter.grad is not None for parameter in model.parameters()):
        raise RuntimeError("PLoP-Attn probe created parameter gradients.")
    after_hashes = _target_weight_hashes(unwrapped_model, eligible.exact_target_names)
    if before_hashes != after_hashes:
        raise RuntimeError("PLoP-Attn forward-only probe changed an eligible base weight.")

    probe_indices_sha256 = hashlib.sha256(",".join(str(index) for index in probe_indices).encode("utf-8")).hexdigest()
    ktransformers_source_sha256 = python_package_source_sha256("ktransformers")
    kt_kernel_source_sha256 = python_package_source_sha256("kt_kernel")
    trainable_parameter_count = sum(
        finetuning_args.lora_rank * (record.in_features + record.out_features) for record in eligible.records
    )
    probe_manifest = {
        "base_checkpoint": model_args.model_name_or_path,
        "base_revision": model_args.model_revision,
        "llamafactory_commit": _git_commit(),
        "dataset": data_args.dataset,
        "template": data_args.template,
        "cutoff_len": data_args.cutoff_len,
        "probe_examples": probe_count,
        "probe_indices_sha256": probe_indices_sha256,
        "probe_data_sha256": _probe_data_hash(train_dataset, probe_indices),
        "probe_seed": cli_args.probe_seed,
        "num_random_draws": 1,
        "lora_rank": finetuning_args.lora_rank,
        "lora_alpha": finetuning_args.lora_alpha,
        "lora_dropout": finetuning_args.lora_dropout,
        "eligible_lora_parameter_count": trainable_parameter_count,
        "kt_fused_wrapper_count": fused_wrapper_count,
        "base_weight_sha256": before_hashes,
        "ktransformers_source_sha256": ktransformers_source_sha256,
        "kt_kernel_source_sha256": kt_kernel_source_sha256,
    }
    if accelerator.is_main_process:
        write_plop_artifacts(
            cli_args.output_dir,
            eligible,
            scores,
            select_k=cli_args.select_k,
            probe_manifest=probe_manifest,
        )
    accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
