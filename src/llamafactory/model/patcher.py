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

import gc
from types import MethodType
from typing import TYPE_CHECKING, Any

import torch
from peft import PeftModel
from transformers import GenerationMixin, PreTrainedModel, PreTrainedTokenizerBase
from transformers.integrations import is_deepspeed_zero3_enabled
from transformers.modeling_utils import is_fsdp_enabled

from ..extras import logging
from ..extras.misc import infer_optim_dtype
from ..extras.packages import is_transformers_version_greater_than
from .model_utils.attention import configure_attn_implementation, print_attn_implementation
from .model_utils.checkpointing import prepare_model_for_training
from .model_utils.embedding import resize_embedding_layer
from .model_utils.kv_cache import configure_kv_cache
from .model_utils.longlora import configure_longlora
from .model_utils.moe import add_z3_leaf_module, configure_moe
from .model_utils.packing import configure_packing
from .model_utils.quantization import configure_quantization
from .model_utils.rope import configure_rope
from .model_utils.valuehead import prepare_valuehead_model
from .model_utils.visual import autocast_projector_dtype, configure_visual_model


if TYPE_CHECKING:
    from transformers import PretrainedConfig, PreTrainedTokenizer, ProcessorMixin
    from trl import AutoModelForCausalLMWithValueHead

    from ..hparams import ModelArguments

if is_transformers_version_greater_than("4.57.0"):
    from transformers.models.qwen3_omni_moe import modeling_qwen3_omni_moe


logger = logging.get_logger(__name__)


def patch_qwen3_omni_moe_thinker_text_sparse_moe_block():
    if is_transformers_version_greater_than("4.57.0") and not is_transformers_version_greater_than("4.58.0"):
        from .model_utils.moe import Qwen3OmniMoeThinkerTextSparseMoeBlock

        logger.warning_rank0(
            "You are using transformers with 4.x version, the Qwen3OmniMoeThinkerTextSparseMoeBlock will have some issues about deepspeed zero2 and fsdp2 training, so that we patched this model to avoid it. Transformers v5.0.0rc0 has fixed the issue, you can also try to update the transformers to using qwen3_omni. See more information on https://github.com/hiyouga/LLaMA-Factory/issues/9628."
        )

        modeling_qwen3_omni_moe.Qwen3OmniMoeThinkerTextSparseMoeBlock = Qwen3OmniMoeThinkerTextSparseMoeBlock


def patch_youtu_vl_model(model: "PreTrainedModel") -> None:
    original_forward = model.forward

    def forward(self, *args, **kwargs):
        outputs = original_forward(*args, **kwargs)
        if "loss" not in outputs and "labels" in kwargs:
            logits = outputs.get("logits")
            labels = kwargs.get("labels")
            if logits is not None and labels is not None:
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss_fct = torch.nn.CrossEntropyLoss()
                loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))
                outputs["loss"] = loss

        return outputs

    model.forward = MethodType(forward, model)


_zero3_loading_patched = False


def _find_multi_shard_layers(checkpoint_files: list[str]) -> set[int]:
    r"""Read the safetensors index to find model layers whose keys span multiple shards.

    Returns a set of layer IDs that appear in more than one shard file.
    Returns an empty set if the index cannot be read.
    """
    import json
    import os
    import re
    from collections import defaultdict

    model_dir = os.path.dirname(checkpoint_files[0])
    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    if not os.path.exists(index_path):
        return set()

    with open(index_path) as f:
        index = json.load(f)

    weight_map = index.get("weight_map", {})
    if not weight_map:
        return set()

    layer_to_shards: dict[int, set[str]] = defaultdict(set)
    for key, shard_name in weight_map.items():
        match = re.search(r"\.layers\.(\d+)\.", key)
        if match:
            layer_to_shards[int(match.group(1))].add(shard_name)

    return {layer for layer, shards in layer_to_shards.items() if len(shards) > 1}


def _build_converter_pattern(weight_mapping) -> "re.Pattern | None":
    r"""Build a compiled regex that matches checkpoint keys needing weight conversion.

    Extracts source_patterns from WeightConverter entries in weight_mapping and converts
    glob-like patterns (e.g. ``mlp.experts.*.gate_proj.weight``) into a single regex.
    Returns None if no converter patterns exist.
    """
    import re

    from transformers.core_model_loading import WeightConverter

    raw_patterns: list[str] = []
    for entry in weight_mapping:
        if isinstance(entry, WeightConverter):
            patterns = entry.source_patterns
            if isinstance(patterns, str):
                patterns = [patterns]
            raw_patterns.extend(patterns)

    if not raw_patterns:
        return None

    # Convert glob-like patterns to regex: "mlp.experts.*.gate_proj.weight" →
    # r"\.mlp\.experts\.[^.]+\.gate_proj\.weight"
    # Also handle $ anchored patterns like "mlp.experts.gate_up_proj$"
    regex_parts = []
    for pat in raw_patterns:
        escaped = re.escape(pat).replace(r"\*", r"[^.]+").replace(r"\$", "$")
        regex_parts.append(escaped)

    combined = "|".join(f"(?:{p})" for p in regex_parts)
    return re.compile(combined)


def patch_zero3_model_loading() -> None:
    r"""Monkey-patch transformers to load model weights shard-by-shard for DeepSpeed ZeRO-3.

    The default behavior in transformers merges ALL checkpoint shards into a single state_dict
    before loading into the model. For large models (e.g. 30B+) with multiple processes, this
    causes a massive CPU memory spike (each process loads the full state_dict ~60GB).

    This patch reduces peak CPU memory by loading one shard at a time. For models with weight
    conversions (e.g. MoE expert fusion), it separates expert keys (which need cross-shard
    buffering for conversion) from non-expert keys (loaded directly per shard).
    """
    global _zero3_loading_patched
    if _zero3_loading_patched:
        return

    _zero3_loading_patched = True

    import copy
    import re

    import transformers.modeling_utils as mu
    from transformers.integrations.deepspeed import (
        _apply_weight_conversions_to_state_dict,
        _load_state_dict_into_zero3_model,
    )
    from transformers.modeling_utils import LoadStateDictInfo, load_state_dict

    _original_load = PreTrainedModel._load_pretrained_model

    @classmethod
    def _load_pretrained_model_shard_by_shard(
        cls,
        model: "PreTrainedModel",
        state_dict: dict | None,
        checkpoint_files: list[str] | None,
        load_config: "mu.LoadStateDictConfig",
    ) -> "LoadStateDictInfo":
        is_quantized = load_config.is_quantized

        # Only intercept the ZeRO-3 branch that loads from checkpoint files
        if not (is_deepspeed_zero3_enabled() and not is_quantized and state_dict is None and checkpoint_files):
            return _original_load.__func__(cls, model, state_dict, checkpoint_files, load_config)

        # Detect weight converters (e.g. MoE expert fusion)
        weight_mapping = getattr(load_config, "weight_mapping", None)
        has_weight_converters = False
        converter_re = None
        if weight_mapping:
            from transformers.core_model_loading import WeightConverter

            has_weight_converters = any(isinstance(v, WeightConverter) for v in weight_mapping)
            if has_weight_converters:
                converter_re = _build_converter_pattern(weight_mapping)

        # Find layers spanning multiple shards (for expert key buffering)
        multi_shard_layers: set[int] = set()
        if has_weight_converters:
            multi_shard_layers = _find_multi_shard_layers(checkpoint_files)

        if has_weight_converters and not multi_shard_layers:
            logger.info_rank0(
                "DeepSpeed ZeRO-3: weight converters detected but no shard index found, "
                "falling back to default loading."
            )
            return _original_load.__func__(cls, model, state_dict, checkpoint_files, load_config)

        logger.info_rank0(
            f"Loading model weights for DeepSpeed ZeRO-3 "
            f"({len(checkpoint_files)} shards, low CPU memory mode"
            f"{f', {len(multi_shard_layers)} boundary layers buffered' if multi_shard_layers else ''})."
        )

        # Create a load_config copy without weight_mapping so _load_state_dict_into_zero3_model
        # won't call _apply_weight_conversions_to_state_dict internally (it has a bug where
        # non-converter keys are dropped when converters are present).
        # We handle conversions ourselves: convert expert keys explicitly, load non-expert keys directly.
        no_wm_config = load_config
        if has_weight_converters:
            no_wm_config = copy.copy(load_config)
            object.__setattr__(no_wm_config, "weight_mapping", None)

        all_error_msgs = []
        all_missing_keys = None
        # Buffer for expert keys from layers that span shard boundaries
        expert_boundary_buffer: dict = {}

        def _track_missing(missing_keys_from_call):
            nonlocal all_missing_keys
            if all_missing_keys is None:
                all_missing_keys = missing_keys_from_call
            else:
                all_missing_keys = all_missing_keys.intersection(missing_keys_from_call)

        def _load_direct(keys_dict):
            """Load a state_dict directly into the model (no weight conversion)."""
            if not keys_dict:
                return
            error_msgs, missing = _load_state_dict_into_zero3_model(model, keys_dict, no_wm_config)
            all_error_msgs.extend(error_msgs)
            _track_missing(missing)

        def _convert_and_load(expert_keys_dict):
            """Apply weight conversions to expert keys, then load the converted result."""
            if not expert_keys_dict:
                return
            converted = _apply_weight_conversions_to_state_dict(model, expert_keys_dict, weight_mapping)
            _load_direct(converted)
            del converted

        for i, ckpt_file in enumerate(checkpoint_files):
            logger.info_rank0(f"Loading shard {i + 1}/{len(checkpoint_files)}")
            shard_state_dict = load_state_dict(
                ckpt_file, map_location="cpu", weights_only=load_config.weights_only
            )

            if not has_weight_converters:
                # Simple path: no conversions, load entire shard directly
                _load_direct(shard_state_dict)
                del shard_state_dict
            else:
                # Separate expert keys (need conversion) from non-expert keys (load directly)
                expert_keys = {}
                non_expert_keys = {}
                for key in list(shard_state_dict.keys()):
                    if converter_re and converter_re.search(key):
                        expert_keys[key] = shard_state_dict.pop(key)
                    else:
                        non_expert_keys[key] = shard_state_dict.pop(key)
                del shard_state_dict

                # Non-expert keys: load immediately (key names match model directly)
                _load_direct(non_expert_keys)
                del non_expert_keys

                # Expert keys: split into boundary-layer (buffer) vs complete-layer (convert now)
                complete_expert_keys = {}
                for key in list(expert_keys.keys()):
                    match = re.search(r"\.layers\.(\d+)\.", key)
                    if match and int(match.group(1)) in multi_shard_layers:
                        expert_boundary_buffer[key] = expert_keys.pop(key)
                    else:
                        complete_expert_keys[key] = expert_keys.pop(key)
                del expert_keys

                # Convert and load complete-layer expert keys immediately
                _convert_and_load(complete_expert_keys)
                del complete_expert_keys

            gc.collect()

        # Convert and load buffered boundary-layer expert keys (now complete across all shards)
        if expert_boundary_buffer:
            logger.info_rank0(
                f"Loading {len(expert_boundary_buffer)} buffered boundary-layer expert keys"
            )
            _convert_and_load(expert_boundary_buffer)
            del expert_boundary_buffer
            gc.collect()

        return LoadStateDictInfo(
            missing_keys=all_missing_keys or set(),
            unexpected_keys=set(),
            mismatched_keys=set(),
            disk_offload_index=None,
            error_msgs=all_error_msgs,
            conversion_errors=set(),
        )

    PreTrainedModel._load_pretrained_model = _load_pretrained_model_shard_by_shard
    logger.info_rank0("Patched model loading for DeepSpeed ZeRO-3 (shard-by-shard mode).")


def patch_tokenizer(tokenizer: "PreTrainedTokenizer", model_args: "ModelArguments") -> None:
    if "PreTrainedTokenizerBase" not in str(tokenizer._pad.__func__):
        tokenizer._pad = MethodType(PreTrainedTokenizerBase._pad, tokenizer)

    if model_args.model_max_length is not None and tokenizer.model_max_length < model_args.model_max_length:
        tokenizer.model_max_length = model_args.model_max_length  # enlarge the tokenizer max length

    if model_args.add_tokens is not None:
        num_added_tokens = tokenizer.add_tokens(new_tokens=model_args.add_tokens, special_tokens=False)
        logger.info_rank0("Add tokens {} to tokenizer's vocabulary.".format(",".join(model_args.add_tokens)))
        if num_added_tokens > 0 and not model_args.resize_vocab:
            model_args.resize_vocab = True
            logger.warning_rank0("New tokens have been added, changed `resize_vocab` to True.")

    if model_args.add_special_tokens is not None:
        num_added_special_tokens = tokenizer.add_tokens(new_tokens=model_args.add_special_tokens, special_tokens=True)
        logger.info_rank0(
            "Add special tokens {} to tokenizer's vocabulary.".format(",".join(model_args.add_special_tokens))
        )
        if num_added_special_tokens > 0 and not model_args.resize_vocab:
            model_args.resize_vocab = True
            logger.warning_rank0("New special tokens have been added, changed `resize_vocab` to True.")


def patch_processor(
    processor: "ProcessorMixin",
    tokenizer: "PreTrainedTokenizer",
    model_args: "ModelArguments",
) -> None:
    setattr(processor, "tokenizer", tokenizer)
    setattr(processor, "image_max_pixels", model_args.image_max_pixels)
    setattr(processor, "image_min_pixels", model_args.image_min_pixels)
    setattr(processor, "image_do_pan_and_scan", model_args.image_do_pan_and_scan)
    setattr(processor, "crop_to_patches", model_args.crop_to_patches)
    setattr(processor, "video_max_pixels", model_args.video_max_pixels)
    setattr(processor, "video_min_pixels", model_args.video_min_pixels)
    setattr(processor, "video_fps", model_args.video_fps)
    setattr(processor, "video_maxlen", model_args.video_maxlen)
    setattr(processor, "use_audio_in_video", model_args.use_audio_in_video)
    setattr(processor, "audio_sampling_rate", model_args.audio_sampling_rate)


def patch_config(
    config: "PretrainedConfig",
    tokenizer: "PreTrainedTokenizer",
    model_args: "ModelArguments",
    init_kwargs: dict[str, Any],
    is_trainable: bool,
) -> None:
    if model_args.compute_dtype is None:  # priority: bf16 > fp16 > fp32
        if model_args.infer_dtype != "auto" and not is_trainable:
            model_args.compute_dtype = getattr(torch, model_args.infer_dtype)
        else:
            model_args.compute_dtype = infer_optim_dtype(model_dtype=getattr(config, "torch_dtype", None))

    configure_attn_implementation(config, model_args)
    configure_rope(config, model_args)
    configure_longlora(config, model_args, is_trainable)
    configure_quantization(config, tokenizer, model_args, is_trainable, init_kwargs)
    configure_moe(config, model_args, is_trainable)
    configure_visual_model(config)
    configure_packing(model_args, is_trainable)
    configure_kv_cache(config, model_args, is_trainable)

    if getattr(config, "model_type", None) == "qwen":
        setattr(config, "use_flash_attn", model_args.flash_attn == "fa2")
        for dtype_name, dtype in [("fp16", torch.float16), ("bf16", torch.bfloat16), ("fp32", torch.float32)]:
            setattr(config, dtype_name, model_args.compute_dtype == dtype)

    if getattr(config, "model_type", None) == "minicpmo":
        setattr(config, "init_audio", True)
        setattr(config, "init_tts", False)

    # replace the top-k gating method
    if getattr(config, "model_type", None) == "kimi_vl" and is_trainable:
        setattr(config.text_config, "topk_method", "greedy")

    architectures = getattr(config, "architectures", None)
    if isinstance(architectures, list) and "InternVLChatModel" in architectures:
        raise ValueError(
            "Please download the internvl models in a Hugging Face–compatible format "
            "(for example, https://huggingface.co/OpenGVLab/InternVL3-8B-hf)."
        )

    if isinstance(architectures, list) and "LlavaLlamaForCausalLM" in architectures:
        raise ValueError("Please download llava models with hf-compatible format: https://huggingface.co/llava-hf")

    if getattr(config, "model_type", None) == "internlm3" and not is_transformers_version_greater_than("4.47.1"):
        raise RuntimeError("InternLM3 model requires transformers>=4.47.1, please upgrade it.")

    if getattr(config, "model_type", None) == "lfm2_vl" and not is_transformers_version_greater_than("4.58.0"):
        raise RuntimeError(
            "LFM2.5-VL model requires transformers>=4.58.0 or install from commit: "
            "pip install git+https://github.com/huggingface/transformers.git@3c2517727ce28a30f5044e01663ee204deb1cdbe"
        )

    if getattr(config, "model_type", None) == "qwen3_omni_moe":
        patch_qwen3_omni_moe_thinker_text_sparse_moe_block()

    # deepspeed zero3 is not compatible with low_cpu_mem_usage
    init_kwargs["low_cpu_mem_usage"] = model_args.low_cpu_mem_usage and (not is_deepspeed_zero3_enabled())

    # fsdp/deepspeed zero3 does not need device map
    if not (is_deepspeed_zero3_enabled() or is_fsdp_enabled()) and init_kwargs["low_cpu_mem_usage"]:
        if "device_map" not in init_kwargs and model_args.device_map:
            init_kwargs["device_map"] = model_args.device_map  # device map requires low_cpu_mem_usage=True

        if init_kwargs.get("device_map", None) == "auto":
            init_kwargs["offload_folder"] = model_args.offload_folder


def patch_model(
    model: "PreTrainedModel",
    tokenizer: "PreTrainedTokenizer",
    model_args: "ModelArguments",
    is_trainable: bool,
    add_valuehead: bool,
) -> None:
    gen_config = model.generation_config  # check and fix generation config
    if not gen_config.do_sample and (
        (gen_config.temperature is not None and gen_config.temperature != 1.0)
        or (gen_config.top_p is not None and gen_config.top_p != 1.0)
        or (gen_config.typical_p is not None and gen_config.typical_p != 1.0)
    ):
        gen_config.do_sample = True

    if getattr(model.config, "model_type", None) not in ["minicpmv", "minicpmo"] and "GenerationMixin" not in str(
        model.generate.__func__
    ):
        model.generate = MethodType(GenerationMixin.generate, model)

    if add_valuehead:
        prepare_valuehead_model(model)

    if model_args.resize_vocab:
        resize_embedding_layer(
            model,
            tokenizer,
            new_special_tokens_config=getattr(model_args, "_special_token_descriptions", None),
            init_special_tokens=model_args.init_special_tokens,
        )

    if is_trainable:
        if getattr(model.config, "model_type", None) == "gemma3n":
            setattr(model_args, "disable_gradient_checkpointing", True)

        if getattr(model.config, "model_type", None) == "youtu_vl":
            patch_youtu_vl_model(model)

        prepare_model_for_training(model, model_args)
        autocast_projector_dtype(model, model_args)
        add_z3_leaf_module(model)

    if not model_args.use_unsloth:
        print_attn_implementation(model.config)

    try:
        model.add_model_tags(["llama-factory"])
    except Exception:
        logger.warning_rank0("Cannot properly tag the model.")


def patch_valuehead_model(model: "AutoModelForCausalLMWithValueHead") -> None:
    def tie_weights(self: "AutoModelForCausalLMWithValueHead") -> None:
        if isinstance(self.pretrained_model, PreTrainedModel):
            self.pretrained_model.tie_weights()

    def get_input_embeddings(self: "AutoModelForCausalLMWithValueHead") -> torch.nn.Module:
        if isinstance(self.pretrained_model, PreTrainedModel):
            return self.pretrained_model.get_input_embeddings()

    def get_output_embeddings(self: "AutoModelForCausalLMWithValueHead") -> torch.nn.Module:
        if isinstance(self.pretrained_model, PreTrainedModel):
            return self.pretrained_model.get_output_embeddings()

    def create_or_update_model_card(self: "AutoModelForCausalLMWithValueHead", output_dir: str) -> None:
        if isinstance(self.pretrained_model, PeftModel):
            self.pretrained_model.create_or_update_model_card(output_dir)

    def get_rope_index_func(self: "AutoModelForCausalLMWithValueHead"):
        if isinstance(self.pretrained_model, PeftModel):
            base_model = self.pretrained_model.base_model.model
        else:
            base_model = self.pretrained_model

        if base_model and hasattr(base_model, "get_rope_index"):
            return base_model.get_rope_index
        elif base_model and hasattr(base_model, "model") and hasattr(base_model.model, "get_rope_index"):
            return base_model.model.get_rope_index
        else:
            return None

    ignore_modules = [name for name, _ in model.named_parameters() if "pretrained_model" in name]
    setattr(model, "_keys_to_ignore_on_save", ignore_modules)
    setattr(model, "tie_weights", MethodType(tie_weights, model))
    setattr(model, "get_input_embeddings", MethodType(get_input_embeddings, model))
    setattr(model, "get_output_embeddings", MethodType(get_output_embeddings, model))
    setattr(model, "get_rope_index", get_rope_index_func(model))
    setattr(model, "create_or_update_model_card", MethodType(create_or_update_model_card, model))
