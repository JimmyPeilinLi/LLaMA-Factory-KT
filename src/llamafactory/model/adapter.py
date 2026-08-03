# Copyright 2025 the LlamaFactory team.
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

import re
from typing import TYPE_CHECKING

import torch
from peft import LoraConfig, LoraModel, OFTConfig, PeftModel, TaskType, get_peft_model
from transformers.integrations import is_deepspeed_zero3_enabled

from ..extras import logging
from .lora_variants.bilora_attn import register_bilora_eval_hooks
from .lora_variants.common import (
    KTLoraVariantConfig,
    TargetManifest,
    file_sha256,
    load_variant_config,
    python_package_source_sha256,
    resolve_eligible_gpu_lora_targets,
    resolve_injected_lora_pairs,
    select_target_manifest,
    set_variant_config,
    set_variant_manifests,
    verify_kt_fused_expert_lora_contract,
)
from .lora_variants.plop_attn import load_plop_training_selection
from .model_utils.lora_expert import (
    LORA_EXPERT_CONFIG_NAME,
    LORA_EXPERT_MODULE_NAME,
    attach_lora_experts,
    load_lora_expert_config,
    set_lora_expert_config,
)
from .model_utils.misc import find_all_linear_modules, find_expanded_modules
from .model_utils.quantization import QuantizationMethod
from .model_utils.unsloth import get_unsloth_peft_model, load_unsloth_peft_model
from .model_utils.visual import COMPOSITE_MODELS, get_forbidden_modules, patch_target_modules


if TYPE_CHECKING:
    from transformers import PretrainedConfig, PreTrainedModel

    from ..hparams import FinetuningArguments, ModelArguments


logger = logging.get_logger(__name__)


def _validate_variant_checkpoint_request(config: KTLoraVariantConfig, finetuning_args: "FinetuningArguments") -> None:
    if config.variant != finetuning_args.kt_lora_variant:
        raise ValueError(
            f"Requested KT LoRA variant {finetuning_args.kt_lora_variant!r} does not match "
            f"checkpoint variant {config.variant!r}."
        )
    if abs(config.lora_dropout - finetuning_args.lora_dropout) > 1.0e-12:
        raise ValueError("KT LoRA variant checkpoint dropout differs from the requested run.")
    method = config.method
    if method.get("deployment") == "primary_only":
        raise ValueError("A Bi-LoRA primary-only deployment adapter cannot resume Bi-LoRA training.")
    if config.variant in ["altlora_attn", "plop_attn"]:
        expected_rank, expected_alpha = finetuning_args.lora_rank, finetuning_args.lora_alpha
    else:
        expected_rank = finetuning_args.bilora_primary_rank + finetuning_args.bilora_aux_rank
        expected_alpha = round(finetuning_args.lora_alpha * expected_rank / finetuning_args.lora_rank)
    if (config.lora_rank, config.lora_alpha) != (expected_rank, expected_alpha):
        raise ValueError("KT LoRA variant checkpoint rank/scaling contract differs from the requested run.")

    requested_method = {
        "altlora_attn": (
            finetuning_args.altlora_reg,
            finetuning_args.altlora_beta1,
            finetuning_args.altlora_switch_every_optimizer_steps,
            finetuning_args.altlora_first_factor,
        ),
        "plop_attn": (finetuning_args.plop_attn_select_k,),
        "bilora_attn": (
            finetuning_args.bilora_primary_rank,
            finetuning_args.bilora_aux_rank,
            finetuning_args.bilora_rho,
            finetuning_args.bilora_aux_lr_ratio,
        ),
    }[config.variant]
    checkpoint_method = {
        "altlora_attn": (
            float(method["regularizer"]),
            float(method["beta1"]),
            int(method["switch_every_optimizer_steps"]),
            method["first_factor"],
        ),
        "plop_attn": (int(method["select_k"]),),
        "bilora_attn": (
            int(method["primary_rank"]),
            int(method["auxiliary_rank"]),
            float(method["rho"]),
            float(method["auxiliary_lr_ratio"]),
        ),
    }[config.variant]
    if checkpoint_method != requested_method:
        raise ValueError(
            f"KT LoRA variant checkpoint method parameters differ: {checkpoint_method} != {requested_method}."
        )


def _new_variant_config(
    finetuning_args: "FinetuningArguments",
    eligible: TargetManifest,
    selected: TargetManifest,
    selected_families: tuple[str, ...],
    fused_wrapper_count: int,
) -> KTLoraVariantConfig:
    variant = finetuning_args.kt_lora_variant
    source_hash = python_package_source_sha256("ktransformers")
    kt_kernel_source_hash = python_package_source_sha256("kt_kernel")
    if variant == "altlora_attn":
        rank, alpha = finetuning_args.lora_rank, finetuning_args.lora_alpha
        method = {
            "paper": "AltLoRA",
            "official_code_commit": "94b219882abcaa6004d2029fd8a09cd190a8a9aa",
            "scope": "eligible_gpu_attention_peft_linears",
            "algorithm": "ordinary_altlora_not_altlora_plus",
            "regularizer": finetuning_args.altlora_reg,
            "beta1": finetuning_args.altlora_beta1,
            "first_factor": finetuning_args.altlora_first_factor,
            "switch_every_optimizer_steps": finetuning_args.altlora_switch_every_optimizer_steps,
            "ktransformers_python_source_sha256": source_hash,
            "kt_kernel_python_source_sha256": kt_kernel_source_hash,
            "kt_fused_expert_lora": "independent_rank8_verified",
            "kt_fused_wrapper_count": fused_wrapper_count,
        }
    elif variant == "plop_attn":
        rank, alpha = finetuning_args.lora_rank, finetuning_args.lora_alpha
        method = {
            "paper": "PLoP",
            "official_code_commit": "561524d0e10bfd4380c0211366cc775a11819204",
            "scope": "eligible_gpu_attention_projection_families",
            "select_k": finetuning_args.plop_attn_select_k,
            "selected_families": list(selected_families),
            "target_artifact_sha256": file_sha256(finetuning_args.plop_attn_target_manifest_path),
            "score_artifact_sha256": (
                file_sha256(finetuning_args.plop_attn_score_manifest_path)
                if finetuning_args.plop_attn_score_manifest_path is not None
                else None
            ),
            "ktransformers_python_source_sha256": source_hash,
            "kt_kernel_python_source_sha256": kt_kernel_source_hash,
            "kt_fused_expert_lora": "independent_rank8_verified",
            "kt_fused_wrapper_count": fused_wrapper_count,
        }
    elif variant == "bilora_attn":
        rank = finetuning_args.bilora_primary_rank + finetuning_args.bilora_aux_rank
        alpha = round(finetuning_args.lora_alpha * rank / finetuning_args.lora_rank)
        method = {
            "paper": "Bi-LoRA",
            "official_code_commit": "e2db644c1c68f069af3bc8737222e9951ed61b3e",
            "scope": "eligible_gpu_attention_peft_linears",
            "primary_rank": finetuning_args.bilora_primary_rank,
            "auxiliary_rank": finetuning_args.bilora_aux_rank,
            "rho": finetuning_args.bilora_rho,
            "auxiliary_lr_ratio": finetuning_args.bilora_aux_lr_ratio,
            "rho_warmup": False,
            "eval_branch": "primary_only",
            "ktransformers_python_source_sha256": source_hash,
            "kt_kernel_python_source_sha256": kt_kernel_source_hash,
            "kt_fused_expert_lora": "independent_rank8_verified",
            "kt_fused_wrapper_count": fused_wrapper_count,
        }
    else:
        raise ValueError(f"Cannot create metadata for KT LoRA variant {variant!r}.")

    return KTLoraVariantConfig(
        variant=variant,
        eligible_sha256=eligible.sha256,
        selected_sha256=selected.sha256,
        exact_target_names=selected.exact_target_names,
        lora_rank=rank,
        lora_alpha=alpha,
        lora_dropout=finetuning_args.lora_dropout,
        method=method,
    )


def _resolve_variant_manifests(
    model: "PreTrainedModel",
    active_variant: str,
    finetuning_args: "FinetuningArguments",
    adapter_variant_config: KTLoraVariantConfig | None,
) -> tuple[TargetManifest, TargetManifest, tuple[str, ...], int]:
    fused_wrapper_count = verify_kt_fused_expert_lora_contract(model, expected_rank=finetuning_args.lora_rank)
    eligible = resolve_eligible_gpu_lora_targets(model, require_gpu=True)
    if adapter_variant_config is not None:
        source_hash = adapter_variant_config.method.get("ktransformers_python_source_sha256")
        if source_hash != python_package_source_sha256("ktransformers"):
            raise ValueError("KTransformers Python source hash differs from the KT LoRA variant checkpoint metadata.")
        kt_kernel_hash = adapter_variant_config.method.get("kt_kernel_python_source_sha256")
        if kt_kernel_hash != python_package_source_sha256("kt_kernel"):
            raise ValueError("kt_kernel Python source hash differs from the KT LoRA variant checkpoint metadata.")
        if adapter_variant_config.method.get("kt_fused_expert_lora") != "independent_rank8_verified":
            raise ValueError("KT LoRA variant checkpoint does not record the audited fused expert LoRA contract.")
        if int(adapter_variant_config.method.get("kt_fused_wrapper_count", -1)) != fused_wrapper_count:
            raise ValueError("KT fused expert wrapper count differs from the KT LoRA variant checkpoint metadata.")
        if adapter_variant_config.eligible_sha256 != eligible.sha256:
            raise ValueError("KT LoRA variant checkpoint eligible target hash differs from the current model.")
        selected = select_target_manifest(eligible, adapter_variant_config.exact_target_names)
        if selected.sha256 != adapter_variant_config.selected_sha256:
            raise ValueError("KT LoRA variant checkpoint selected target hash differs from the current model.")
        if adapter_variant_config.variant == "plop_attn" and finetuning_args.kt_lora_variant == "plop_attn":
            frozen_selected, frozen_families = load_plop_training_selection(
                eligible,
                finetuning_args.plop_attn_target_manifest_path,
                select_k=finetuning_args.plop_attn_select_k,
                score_manifest_path=finetuning_args.plop_attn_score_manifest_path,
            )
            if frozen_selected != selected:
                raise ValueError("PLoP-Attn resume artifact differs from the checkpoint's frozen target inventory.")
            method = adapter_variant_config.method
            if file_sha256(finetuning_args.plop_attn_target_manifest_path) != method["target_artifact_sha256"]:
                raise ValueError("PLoP-Attn resume target artifact file hash differs from the checkpoint metadata.")
            expected_score_hash = method.get("score_artifact_sha256")
            if expected_score_hash is not None and (
                finetuning_args.plop_attn_score_manifest_path is None
                or file_sha256(finetuning_args.plop_attn_score_manifest_path) != expected_score_hash
            ):
                raise ValueError("PLoP-Attn resume score artifact file hash differs from the checkpoint metadata.")
            return eligible, selected, frozen_families, fused_wrapper_count
        return eligible, selected, selected.families, fused_wrapper_count
    if active_variant == "plop_attn":
        selected, selected_families = load_plop_training_selection(
            eligible,
            finetuning_args.plop_attn_target_manifest_path,
            select_k=finetuning_args.plop_attn_select_k,
            score_manifest_path=finetuning_args.plop_attn_score_manifest_path,
        )
        return eligible, selected, selected_families, fused_wrapper_count
    selected = select_target_manifest(eligible, eligible.exact_target_names)
    return eligible, selected, selected.families, fused_wrapper_count


def _setup_full_tuning(
    model: "PreTrainedModel",
    finetuning_args: "FinetuningArguments",
    is_trainable: bool,
    cast_trainable_params_to_fp32: bool,
) -> None:
    if not is_trainable:
        return

    logger.info_rank0("Fine-tuning method: Full")
    forbidden_modules = get_forbidden_modules(model.config, finetuning_args)
    for name, param in model.named_parameters():
        if not any(forbidden_module in name for forbidden_module in forbidden_modules):
            if cast_trainable_params_to_fp32:
                param.data = param.data.to(torch.float32)
        else:
            param.requires_grad_(False)


def _setup_freeze_tuning(
    model: "PreTrainedModel",
    finetuning_args: "FinetuningArguments",
    is_trainable: bool,
    cast_trainable_params_to_fp32: bool,
) -> None:
    if not is_trainable:
        return

    logger.info_rank0("Fine-tuning method: Freeze")
    if hasattr(model.config, "text_config"):  # composite models
        config = getattr(model.config, "text_config")
    else:
        config = model.config

    num_layers = (
        getattr(config, "num_hidden_layers", None)
        or getattr(config, "num_layers", None)
        or getattr(config, "n_layer", None)
    )
    if not num_layers:
        raise ValueError("Current model does not support freeze tuning.")

    if finetuning_args.use_llama_pro:
        if num_layers % finetuning_args.freeze_trainable_layers != 0:
            raise ValueError(
                f"`num_layers` {num_layers} should be "
                f"divisible by `num_layer_trainable` {finetuning_args.freeze_trainable_layers}."
            )

        stride = num_layers // finetuning_args.freeze_trainable_layers
        trainable_layer_ids = range(stride - 1, num_layers + stride - 1, stride)
    elif finetuning_args.freeze_trainable_layers > 0:  # fine-tuning the last n layers if num_layer_trainable > 0
        trainable_layer_ids = range(max(0, num_layers - finetuning_args.freeze_trainable_layers), num_layers)
    else:  # fine-tuning the first n layers if num_layer_trainable < 0
        trainable_layer_ids = range(min(-finetuning_args.freeze_trainable_layers, num_layers))

    hidden_modules = set()
    non_hidden_modules = set()
    for name, _ in model.named_parameters():
        if ".0." in name:
            hidden_modules.add(name.split(".0.")[-1].split(".")[0])
        elif ".1." in name:  # MoD starts from layer 1
            hidden_modules.add(name.split(".1.")[-1].split(".")[0])

        if re.search(r"\.\d+\.", name) is None:
            non_hidden_modules.add(name.split(".")[-2])  # remove weight/bias

    trainable_layers = []
    for module_name in finetuning_args.freeze_trainable_modules:
        if module_name != "all" and module_name not in hidden_modules:
            raise ValueError(
                "Module {} is not found, please choose from {}".format(module_name, ", ".join(hidden_modules))
            )

        for idx in trainable_layer_ids:
            trainable_layers.append(".{:d}.{}".format(idx, module_name if module_name != "all" else ""))

    if finetuning_args.freeze_extra_modules:
        for module_name in finetuning_args.freeze_extra_modules:
            if module_name not in non_hidden_modules:
                raise ValueError(
                    "Module {} is not found, please choose from {}".format(module_name, ", ".join(non_hidden_modules))
                )

            trainable_layers.append(module_name)

    model_type = getattr(model.config, "model_type", None)
    if not finetuning_args.freeze_multi_modal_projector and model_type in COMPOSITE_MODELS:
        trainable_layers.extend(COMPOSITE_MODELS[model_type].projector_keys)

    forbidden_modules = get_forbidden_modules(model.config, finetuning_args)
    for name, param in model.named_parameters():
        if any(trainable_layer in name for trainable_layer in trainable_layers) and not any(
            forbidden_module in name for forbidden_module in forbidden_modules
        ):
            if cast_trainable_params_to_fp32:
                param.data = param.data.to(torch.float32)
        else:
            param.requires_grad_(False)

    logger.info_rank0("Set trainable layers: {}".format(",".join(trainable_layers)))


def _setup_lora_tuning(
    config: "PretrainedConfig",
    model: "PreTrainedModel",
    model_args: "ModelArguments",
    finetuning_args: "FinetuningArguments",
    is_trainable: bool,
    cast_trainable_params_to_fp32: bool,
) -> "PeftModel":
    if is_trainable:
        if finetuning_args.finetuning_type == "oft":
            logger.info_rank0("Fine-tuning method: OFT")
        else:
            logger.info_rank0("Fine-tuning method: {}".format("DoRA" if finetuning_args.use_dora else "LoRA"))

    adapter_to_resume = None
    lora_expert_config = None
    adapter_variant_config = None
    eligible_manifest = None
    selected_manifest = None
    selected_families: tuple[str, ...] = ()
    fused_wrapper_count = 0
    active_variant = finetuning_args.kt_lora_variant
    if active_variant != "vanilla" and not is_trainable and model_args.adapter_name_or_path is None:
        raise ValueError("KT LoRA variant evaluation requires an adapter checkpoint with variant metadata.")

    if model_args.adapter_name_or_path is not None:
        is_mergeable = True
        if getattr(model, "quantization_method", None):  # merge lora in quantized model is unstable
            assert len(model_args.adapter_name_or_path) == 1, "Quantized model only accepts a single adapter."
            is_mergeable = False

        if is_deepspeed_zero3_enabled():
            assert len(model_args.adapter_name_or_path) == 1, "Cannot use multiple adapters in DeepSpeed ZeRO-3."
            is_mergeable = False

        if model_args.use_kt:
            assert len(model_args.adapter_name_or_path) == 1, "KTransformers model only accepts a single adapter"
            is_mergeable = False

        if model_args.use_unsloth:
            assert len(model_args.adapter_name_or_path) == 1, "Unsloth model only accepts a single adapter."
            is_mergeable = False

        if (is_trainable and not finetuning_args.create_new_adapter) or (not is_mergeable):
            adapter_to_merge = model_args.adapter_name_or_path[:-1]
            adapter_to_resume = model_args.adapter_name_or_path[-1]
        else:
            adapter_to_merge = model_args.adapter_name_or_path

        for adapter in model_args.adapter_name_or_path:
            adapter_config = load_lora_expert_config(
                adapter,
                subfolder=model_args.adapter_folder,
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
                token=model_args.hf_hub_token,
            )
            if adapter_config is not None:
                if lora_expert_config is not None and adapter_config != lora_expert_config:
                    raise ValueError("Cannot combine adapters with different LoRA Expert metadata.")
                lora_expert_config = adapter_config

            variant_config = load_variant_config(
                adapter,
                subfolder=model_args.adapter_folder,
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
                token=model_args.hf_hub_token,
            )
            if variant_config is None:
                continue
            if adapter_variant_config is not None and variant_config != adapter_variant_config:
                raise ValueError("Cannot combine adapters with different KT LoRA variant metadata.")
            adapter_variant_config = variant_config

        if adapter_variant_config is not None:
            if not model_args.use_kt:
                raise ValueError("KT LoRA variant adapters require `use_kt: true` in the audited first version.")
            if active_variant == "vanilla":
                if is_trainable:
                    raise ValueError(
                        "Resuming a KT LoRA variant requires explicitly setting the checkpoint variant arguments."
                    )
                active_variant = adapter_variant_config.variant
            else:
                _validate_variant_checkpoint_request(adapter_variant_config, finetuning_args)
        elif adapter_to_resume is not None and active_variant != "vanilla":
            raise ValueError("Cannot resume a KT LoRA variant because its checkpoint metadata is missing.")

        if active_variant != "vanilla":
            (
                eligible_manifest,
                selected_manifest,
                selected_families,
                fused_wrapper_count,
            ) = _resolve_variant_manifests(model, active_variant, finetuning_args, adapter_variant_config)

        if lora_expert_config is not None:
            if finetuning_args.use_lora_expert and (
                finetuning_args.lora_expert_num != lora_expert_config.num_experts
                or finetuning_args.lora_expert_intermediate_size != lora_expert_config.intermediate_size
            ):
                raise ValueError("Requested LoRA Expert arguments do not match the adapter checkpoint metadata.")
            lora_expert_config = attach_lora_experts(
                model,
                config=lora_expert_config,
                use_kt=model_args.use_kt,
            )
            finetuning_args.use_lora_expert = True
        elif adapter_to_resume is not None and finetuning_args.use_lora_expert:
            raise ValueError(
                f"Cannot resume LoRA Expert training because the adapter has no {LORA_EXPERT_CONFIG_NAME} metadata."
            )

        init_kwargs = {
            "subfolder": model_args.adapter_folder,
            "offload_folder": model_args.offload_folder,
            "cache_dir": model_args.cache_dir,
            "revision": model_args.model_revision,
            "token": model_args.hf_hub_token,
        }

        for adapter in adapter_to_merge:
            model: LoraModel = PeftModel.from_pretrained(model, adapter, **init_kwargs)
            model = model.merge_and_unload()

        if len(adapter_to_merge) > 0:
            logger.info_rank0(f"Merged {len(adapter_to_merge)} adapter(s).")

        if adapter_to_resume is not None:  # resume lora training
            if isinstance(model, PeftModel):
                pass  # already loaded via load_unsloth_peft_model in loader.py
            else:
                if model_args.use_unsloth:
                    peft_model = load_unsloth_peft_model(
                        config, model_args, finetuning_args, is_trainable=is_trainable
                    )
                    if peft_model is not None:
                        model = peft_model

                if not model_args.use_unsloth:  # unsloth was disabled or fell back
                    model = PeftModel.from_pretrained(
                        model, adapter_to_resume, is_trainable=is_trainable, **init_kwargs
                    )

        logger.info_rank0("Loaded adapter(s): {}".format(",".join(model_args.adapter_name_or_path)))

    if active_variant != "vanilla" and eligible_manifest is None:
        (
            eligible_manifest,
            selected_manifest,
            selected_families,
            fused_wrapper_count,
        ) = _resolve_variant_manifests(model, active_variant, finetuning_args, adapter_variant_config)

    if is_trainable and adapter_to_resume is None:  # create new lora weights while training
        if len(finetuning_args.lora_target) == 1 and finetuning_args.lora_target[0] == "all":
            target_modules = find_all_linear_modules(model, finetuning_args.freeze_vision_tower)
        else:
            target_modules = finetuning_args.lora_target

        if finetuning_args.use_llama_pro:
            target_modules = find_expanded_modules(model, target_modules, finetuning_args.freeze_trainable_layers)

        target_modules = patch_target_modules(model, finetuning_args, target_modules)
        if active_variant != "vanilla":
            if selected_manifest is None:
                raise RuntimeError("KT LoRA variant selected targets were not resolved before PEFT injection.")
            target_modules = list(selected_manifest.exact_target_names)

        if finetuning_args.use_lora_expert and lora_expert_config is None:
            lora_expert_config = attach_lora_experts(
                model,
                num_experts=finetuning_args.lora_expert_num,
                intermediate_size=finetuning_args.lora_expert_intermediate_size,
                use_kt=model_args.use_kt,
            )

        if (
            finetuning_args.use_dora
            and getattr(model, "quantization_method", None) is not None
            and getattr(model, "quantization_method", None) != QuantizationMethod.BNB
        ):
            raise ValueError("DoRA is not compatible with PTQ-quantized models.")

        if model_args.resize_vocab and finetuning_args.additional_target is None:
            input_embeddings = model.get_input_embeddings()
            output_embeddings = model.get_output_embeddings()
            module_names = set()
            for name, module in model.named_modules():
                if module in [input_embeddings, output_embeddings]:
                    module_names.add(name.split(".")[-1])

            finetuning_args.additional_target = module_names
            logger.warning_rank0("Vocab has been resized, add {} to trainable params.".format(",".join(module_names)))

        modules_to_save = list(finetuning_args.additional_target or [])
        if lora_expert_config is not None and LORA_EXPERT_MODULE_NAME not in modules_to_save:
            modules_to_save.append(LORA_EXPERT_MODULE_NAME)
        if not modules_to_save:
            modules_to_save = None

        if finetuning_args.finetuning_type == "lora":
            peft_rank = finetuning_args.lora_rank
            peft_alpha = finetuning_args.lora_alpha
            if active_variant == "bilora_attn":
                peft_rank = finetuning_args.bilora_primary_rank + finetuning_args.bilora_aux_rank
                peft_alpha = round(finetuning_args.lora_alpha * peft_rank / finetuning_args.lora_rank)
            peft_kwargs = {
                "r": peft_rank,
                "target_modules": target_modules,
                "lora_alpha": peft_alpha,
                "lora_dropout": finetuning_args.lora_dropout,
                "use_rslora": finetuning_args.use_rslora,
                "use_dora": finetuning_args.use_dora,
                "modules_to_save": modules_to_save,
            }
        elif finetuning_args.finetuning_type == "oft":
            peft_kwargs = {
                "r": finetuning_args.oft_rank,
                "oft_block_size": finetuning_args.oft_block_size,
                "target_modules": target_modules,
                "module_dropout": finetuning_args.module_dropout,
                "modules_to_save": modules_to_save,
            }

        if model_args.use_kt:
            if finetuning_args.finetuning_type != "lora":
                raise ValueError("KTransformers only supports LoRA finetuning.")

            peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, **peft_kwargs)
            model = get_peft_model(model, peft_config)
        elif model_args.use_unsloth:
            if finetuning_args.finetuning_type == "oft":
                raise ValueError("Unsloth is currently not supported for OFT.")

            model = get_unsloth_peft_model(model, model_args, peft_kwargs)
        else:
            if finetuning_args.pissa_init:
                if finetuning_args.pissa_iter == -1:
                    logger.info_rank0("Using PiSSA initialization.")
                    peft_kwargs["init_lora_weights"] = "pissa"
                else:
                    logger.info_rank0(f"Using PiSSA initialization with FSVD steps {finetuning_args.pissa_iter}.")
                    peft_kwargs["init_lora_weights"] = f"pissa_niter_{finetuning_args.pissa_iter}"

            if finetuning_args.finetuning_type == "lora":
                peft_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    inference_mode=False,
                    **peft_kwargs,
                )
            elif finetuning_args.finetuning_type == "oft":
                peft_config = OFTConfig(
                    task_type=TaskType.CAUSAL_LM,
                    inference_mode=False,
                    **peft_kwargs,
                )
            model = get_peft_model(model, peft_config)

    if active_variant != "vanilla" and (is_trainable or adapter_variant_config is not None):
        if eligible_manifest is None or selected_manifest is None:
            raise RuntimeError("KT LoRA variant target manifests are unavailable after PEFT injection.")
        pairs = resolve_injected_lora_pairs(model, selected_manifest)
        variant_config = adapter_variant_config or _new_variant_config(
            finetuning_args,
            eligible_manifest,
            selected_manifest,
            selected_families,
            fused_wrapper_count,
        )
        if len(pairs) != len(selected_manifest.records):
            raise ValueError("KT LoRA variant PEFT pair count differs from the selected target manifest.")
        for pair in pairs:
            if pair.lora_a.shape[0] != variant_config.lora_rank:
                raise ValueError(
                    f"KT LoRA variant rank mismatch at {pair.name}: "
                    f"{pair.lora_a.shape[0]} != {variant_config.lora_rank}."
                )
            expected_scaling = variant_config.lora_alpha / variant_config.lora_rank
            if abs(pair.scaling - expected_scaling) > 1.0e-12:
                raise ValueError(
                    f"KT LoRA variant scaling mismatch at {pair.name}: {pair.scaling} != {expected_scaling}."
                )
        set_variant_config(model, variant_config)
        set_variant_manifests(model, eligible_manifest, selected_manifest)
        setattr(model, "_kt_lora_variant_pairs", pairs)
        if active_variant == "bilora_attn":
            register_bilora_eval_hooks(pairs, int(variant_config.method["primary_rank"]))
        logger.info_rank0(
            f"Configured {active_variant} on {len(pairs)} exact GPU attention PEFT pairs "
            f"across families {','.join(selected_families)}."
        )

    if lora_expert_config is not None:
        set_lora_expert_config(model, lora_expert_config)

    if is_trainable and cast_trainable_params_to_fp32:
        for param in filter(lambda p: p.requires_grad, model.parameters()):
            param.data = param.data.to(torch.float32)

    return model


def init_adapter(
    config: "PretrainedConfig",
    model: "PreTrainedModel",
    model_args: "ModelArguments",
    finetuning_args: "FinetuningArguments",
    is_trainable: bool,
) -> "PreTrainedModel":
    r"""Initialize the adapters.

    Support full-parameter, freeze and LoRA training.

    Note that the trainable parameters must be cast to float32.
    """
    if is_trainable and getattr(model, "quantization_method", None) is not None:
        if finetuning_args.finetuning_type not in ["lora", "oft"]:
            raise ValueError("Quantized models can only be used for the LoRA or OFT tuning.")

        if finetuning_args.pissa_init:
            raise ValueError("Cannot initialize PiSSA adapter on quantized models.")

    # cast trainable parameters to float32 if:
    # 1. is_trainable and not pure_bf16 and not badam and quantization_bit is not None (qlora)
    # 2. is_trainable and not pure_bf16 and not badam and not zero3 (zero3 already in fp32)
    cast_trainable_params_to_fp32 = False
    if not is_trainable:
        pass
    elif finetuning_args.pure_bf16 or finetuning_args.use_badam:
        logger.info_rank0("Pure bf16 / BAdam detected, remaining trainable params in half precision.")
    elif model_args.quantization_bit is None and is_deepspeed_zero3_enabled():
        logger.info_rank0("DeepSpeed ZeRO3 detected, remaining trainable params in float32.")
    else:
        logger.info_rank0("Upcasting trainable params to float32.")
        cast_trainable_params_to_fp32 = True

    if finetuning_args.finetuning_type == "full":
        _setup_full_tuning(model, finetuning_args, is_trainable, cast_trainable_params_to_fp32)
    elif finetuning_args.finetuning_type == "freeze":
        _setup_freeze_tuning(model, finetuning_args, is_trainable, cast_trainable_params_to_fp32)
    elif finetuning_args.finetuning_type in ["lora", "oft"]:
        model = _setup_lora_tuning(
            config, model, model_args, finetuning_args, is_trainable, cast_trainable_params_to_fp32
        )
    else:
        raise NotImplementedError(f"Unknown finetuning type: {finetuning_args.finetuning_type}.")

    return model
