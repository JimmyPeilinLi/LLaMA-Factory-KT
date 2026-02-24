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
_zero3_init_patched = False


def _get_mem_info() -> str:
    """Get process RSS and system available memory."""
    rss_gb = "N/A"
    avail_gb = "N/A"
    total_gb = "N/A"
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    rss_gb = f"{int(line.split()[1]) / 1024 / 1024:.2f}"
                    break
    except Exception:
        pass
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    total_gb = f"{int(line.split()[1]) / 1024 / 1024:.2f}"
                elif line.startswith("MemAvailable:"):
                    avail_gb = f"{int(line.split()[1]) / 1024 / 1024:.2f}"
    except Exception:
        pass
    return f"RSS={rss_gb}GB, Avail={avail_gb}GB, Total={total_gb}GB"


def _tensor_bytes(state_dict: dict) -> int:
    """Sum the byte size of all tensors in a state_dict."""
    total = 0
    for v in state_dict.values():
        if hasattr(v, "nbytes"):
            total += v.nbytes
        elif hasattr(v, "nelement") and hasattr(v, "element_size"):
            total += v.nelement() * v.element_size()
    return total


def _fmt_bytes(n: int) -> str:
    """Format bytes as human-readable string."""
    if n >= 1 << 30:
        return f"{n / (1 << 30):.2f}GB"
    elif n >= 1 << 20:
        return f"{n / (1 << 20):.2f}MB"
    else:
        return f"{n / (1 << 10):.2f}KB"


def _build_boundary_layer_info(checkpoint_files: list[str], converter_re: "re.Pattern") -> dict[int, int]:
    r"""For boundary layers (keys spanning multiple shards), count expected expert keys.

    Reads the safetensors index to identify which layers span multiple shard files,
    then counts how many expert keys (matching converter_re) belong to each such layer.

    Returns a dict mapping layer_id to the expected number of expert checkpoint keys.
    Returns an empty dict if the index cannot be read or no boundary layers exist.
    """
    import json
    import os
    import re
    from collections import defaultdict

    model_dir = os.path.dirname(checkpoint_files[0])
    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    if not os.path.exists(index_path):
        return {}

    with open(index_path) as f:
        index = json.load(f)

    weight_map = index.get("weight_map", {})
    if not weight_map:
        return {}

    layer_to_shards: dict[int, set[str]] = defaultdict(set)
    layer_expert_count: dict[int, int] = defaultdict(int)

    for key, shard_name in weight_map.items():
        match = re.search(r"\.layers\.(\d+)\.", key)
        if match:
            layer_id = int(match.group(1))
            layer_to_shards[layer_id].add(shard_name)
            if converter_re.search(key):
                layer_expert_count[layer_id] += 1

    return {
        layer: layer_expert_count[layer]
        for layer, shards in layer_to_shards.items()
        if len(shards) > 1 and layer in layer_expert_count
    }


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
    # r"\.mlp\.experts\.[^.]+\.gate_proj\.weight$"
    # Anchor with $ to avoid matching suffixed keys like "weight_scale_inv"
    regex_parts = []
    for pat in raw_patterns:
        escaped = re.escape(pat).replace(r"\*", r"[^.]+").replace(r"\$", "$")
        if not escaped.endswith("$"):
            escaped += "$"
        regex_parts.append(escaped)

    combined = "|".join(f"(?:{p})" for p in regex_parts)
    return re.compile(combined)


def patch_deepspeed_zero_init_memory() -> None:
    r"""Monkey-patch DeepSpeed zero.Init to reduce CPU memory during model creation.

    During ``deepspeed.zero.Init()``, each parameter is created at full size, then
    partitioned across ranks with the partition stored on CPU. When pin_memory=True
    (default for CPU offload), the partition goes through a double allocation:

      1. ``torch.empty(size, device='cpu')``  → regular malloc (~3.75GB for a 15GB param / 4 ranks)
      2. ``pin_memory()``                     → cudaHostAlloc (~3.75GB) + copy + free(original)

    The ``free()`` in step 2 often doesn't return memory to the OS (glibc arena fragmentation),
    causing ~2x the expected memory usage. For DeepSeek-V3 (671B, 256 experts), this means
    ~1.8TB instead of ~1TB, causing OOM on a 2TB machine at ~72% completion.

    This patch:
      1. Replaces the two-step malloc→pin with direct ``torch.empty(pin_memory=True)``
      2. Adds ``gc.collect()`` + ``malloc_trim(0)`` after each large parameter partition
      3. Adds diagnostic logging for RSS tracking during model creation
    """
    global _zero3_init_patched
    if _zero3_init_patched:
        return

    if not is_deepspeed_zero3_enabled():
        return

    _zero3_init_patched = True

    import ctypes
    import time

    from deepspeed.runtime.zero.partition_parameters import Init

    # Get libc for malloc_trim
    try:
        _libc = ctypes.CDLL("libc.so.6")
        _has_malloc_trim = hasattr(_libc, "malloc_trim")
    except Exception:
        _libc = None
        _has_malloc_trim = False

    # Threshold: only do aggressive cleanup for params larger than 100MB
    _CLEANUP_THRESHOLD_BYTES = 100 * 1024 * 1024

    _original_post_init = Init._post_init_method
    _param_count = [0]
    _last_log_time = [0.0]

    def _patched_post_init_method(self, module):
        r"""Wraps _post_init_method to add memory cleanup and diagnostics."""
        # Calculate total param bytes in this module (before partitioning)
        total_bytes = 0
        for _name, p in module.named_parameters(recurse=False):
            total_bytes += p.numel() * p.element_size()

        # Call original (creates, broadcasts, partitions params)
        _original_post_init(self, module)

        # Count params for progress tracking
        for _name, _p in module.named_parameters(recurse=False):
            _param_count[0] += 1

        # Aggressive cleanup after large modules (e.g. expert modules with 15GB+ params)
        if total_bytes > _CLEANUP_THRESHOLD_BYTES:
            gc.collect()
            if _has_malloc_trim:
                _libc.malloc_trim(0)

            # Log progress periodically (every 10 seconds)
            now = time.monotonic()
            if now - _last_log_time[0] > 10.0:
                _last_log_time[0] = now
                logger.info_rank0(
                    f"[DIAG] zero.Init progress: {_param_count[0]} params done, "
                    f"module={module.__class__.__name__}, "
                    f"module_params={_fmt_bytes(total_bytes)} | {_get_mem_info()}"
                )

    Init._post_init_method = _patched_post_init_method

    # --- Patch _partition_param to avoid double allocation ---
    _original_partition_param = Init._partition_param

    def _patched_partition_param(self, param, buffer=None, has_been_updated=False, free_data=True):
        r"""Wraps _partition_param to use direct pinned allocation.

        The original code does:
          1. torch.empty(size, device='cpu')        → malloc
          2. get_accelerator().pin_memory(tensor)    → cudaHostAlloc + copy + free(malloc'd)

        The free() in step 2 may not return memory to OS. We fix this by:
          - Allocating directly as pinned via torch.empty(pin_memory=True) when possible
          - Adding gc.collect + malloc_trim after freeing the full param
        """
        from deepspeed.runtime.zero.config import OffloadDeviceEnum
        from deepspeed.runtime.zero.partition_parameters import (
            PartitionedParamStatus,
            ZeroParamStatus,
            free_param,
            get_accelerator,
            print_rank_0,
            see_memory_usage,
        )

        assert param.ds_status is not ZeroParamStatus.INFLIGHT, f" {param} Cannot partition a param in flight"

        if param.ds_status is ZeroParamStatus.AVAILABLE:
            if param.ds_tensor is not None and not has_been_updated:
                see_memory_usage(f'Before partitioning param {param.ds_id} {param.shape}', force=False)
                if free_data:
                    free_param(param)
                see_memory_usage(f'After partitioning param {param.ds_id} {param.shape}', force=False)

                if param.ds_tensor.final_location == OffloadDeviceEnum.nvme:
                    print_rank_0(f"Param {param.ds_id} partition released since it exists in nvme", force=False)
                    param.nvme_swapper.remove_partition_and_release_buffers([param])
                return

            tensor_size = self._aligned_size(param)
            partition_size = tensor_size // self.num_partitions

            if param.ds_tensor is None:
                final_location = None
                if self.remote_device == OffloadDeviceEnum.nvme and self.param_swapper.swappable_tensor(
                        numel=partition_size):
                    # NVMe path: unchanged
                    final_location = OffloadDeviceEnum.nvme
                    buffer = self.param_swapper.get_buffer(param, partition_size)
                    partitioned_tensor = torch.empty(0, dtype=param.dtype, device=buffer.device)
                    partitioned_tensor.data = buffer.data
                    print_rank_0(f"ID {param.ds_id} Initializing partition for the first time for nvme offload.")
                else:
                    if param.ds_persist:
                        device = self.local_device
                    elif self.remote_device == OffloadDeviceEnum.nvme:
                        device = OffloadDeviceEnum.cpu
                    else:
                        device = self.remote_device

                    # KEY FIX: allocate directly as pinned to avoid double allocation
                    if device == OffloadDeviceEnum.cpu and self.pin_memory:
                        partitioned_tensor = torch.empty(partition_size, dtype=param.dtype,
                                                         device='cpu', pin_memory=True)
                    else:
                        partitioned_tensor = torch.empty(partition_size, dtype=param.dtype, device=device)

                    # quantize the tensor if it's not trainable
                    if not param.requires_grad and self.quantized_nontrainable_weights:
                        partitioned_tensor, partitioned_tensor.ds_quant_scale = self.quantizer_module.quantize(
                            partitioned_tensor)
                        # If quantized AND needs pinning, pin the quantized tensor (fallback to original path)
                        if device == OffloadDeviceEnum.cpu and self.pin_memory and not partitioned_tensor.is_pinned():
                            partitioned_tensor = get_accelerator().pin_memory(partitioned_tensor)

                partitioned_tensor.requires_grad = False
                param.ds_tensor = partitioned_tensor
                param.ds_tensor.ds_numel = partition_size
                param.ds_tensor.status = PartitionedParamStatus.AVAILABLE
                param.ds_tensor.final_location = final_location
                param.ds_numel_aligned = tensor_size

            start = partition_size * self.get_partition_rank()
            end = start + partition_size

            one_dim_param = param.contiguous().view(-1)

            if start < param.ds_numel and end <= param.ds_numel:
                src_tensor = one_dim_param.narrow(0, start, partition_size)
                with torch.no_grad():
                    param.ds_tensor.copy_(src_tensor)
            else:
                if start < param.ds_numel:
                    elems_to_copy = param.ds_numel - start
                    with torch.no_grad():
                        param.ds_tensor.narrow(0, 0, elems_to_copy).copy_(
                            one_dim_param.narrow(0, start, elems_to_copy))

            see_memory_usage(f'Before partitioning param {param.ds_id} {param.shape}', force=False)
            free_param(param)
            see_memory_usage(f'After partitioning param {param.ds_id} {param.shape}', force=False)

            # Force cleanup of any retained intermediate memory
            param_bytes = param.ds_numel * param.element_size()
            if param_bytes > _CLEANUP_THRESHOLD_BYTES:
                del one_dim_param
                gc.collect()
                if _has_malloc_trim:
                    _libc.malloc_trim(0)

            if param.ds_tensor.final_location == OffloadDeviceEnum.nvme:
                self.param_swapper.swap_out_and_release([param])
                print_rank_0(f"ID {param.ds_id} Offloaded to nvme offload and buffers released.")

            print_rank_0(
                f"ID {param.ds_id} partitioned type {param.dtype} dev {param.device} shape {param.shape}")

    Init._partition_param = _patched_partition_param

    logger.info_rank0(
        "Patched DeepSpeed zero.Init for reduced CPU memory "
        "(direct pinned allocation + gc/malloc_trim after large params)."
    )


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
        import os
        import time

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

        # Find boundary layers and count their expected expert keys (for per-layer tracking)
        boundary_layer_info: dict[int, int] = {}
        if has_weight_converters:
            boundary_layer_info = _build_boundary_layer_info(checkpoint_files, converter_re)
        multi_shard_layers = set(boundary_layer_info.keys())

        if has_weight_converters and not multi_shard_layers:
            logger.info_rank0(
                "DeepSpeed ZeRO-3: weight converters detected but no shard index found, "
                "falling back to default loading."
            )
            return _original_load.__func__(cls, model, state_dict, checkpoint_files, load_config)

        logger.info_rank0(
            f"[DIAG] ===== Shard-by-shard loading START ====="
            f"\n[DIAG] Shards: {len(checkpoint_files)}, "
            f"has_weight_converters: {has_weight_converters}, "
            f"boundary_layers: {len(multi_shard_layers)}"
            f"\n[DIAG] converter_re: {converter_re.pattern if converter_re else 'None'}"
            f"\n[DIAG] boundary_layer_info (layer_id: expected_expert_keys): "
            f"{dict(sorted(boundary_layer_info.items())) if boundary_layer_info else '{}'}"
            f"\n[DIAG] Memory BEFORE loading: {_get_mem_info()}"
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
        # Per-layer buffer for expert keys from boundary layers: layer_id → {key: tensor}
        per_layer_buffer: dict[int, dict] = {}
        completed_boundary_layers = 0
        total_buffer_bytes = 0  # track buffer memory

        def _track_missing(missing_keys_from_call):
            nonlocal all_missing_keys
            if all_missing_keys is None:
                all_missing_keys = missing_keys_from_call
            else:
                all_missing_keys = all_missing_keys.intersection(missing_keys_from_call)

        def _load_direct(keys_dict, label=""):
            """Load a state_dict directly into the model (no weight conversion)."""
            if not keys_dict:
                return
            n_keys = len(keys_dict)
            sz = _tensor_bytes(keys_dict)
            logger.info_rank0(
                f"[DIAG]   _load_direct({label}): {n_keys} keys, {_fmt_bytes(sz)} | {_get_mem_info()}"
            )
            t0 = time.monotonic()
            error_msgs, missing = _load_state_dict_into_zero3_model(model, keys_dict, no_wm_config)
            dt = time.monotonic() - t0
            all_error_msgs.extend(error_msgs)
            _track_missing(missing)
            logger.info_rank0(
                f"[DIAG]   _load_direct({label}) done in {dt:.1f}s, "
                f"errors: {len(error_msgs)} | {_get_mem_info()}"
            )

        def _convert_and_load(expert_keys_dict, label=""):
            """Apply weight conversions to expert keys, then load the converted result."""
            if not expert_keys_dict:
                return
            n_keys = len(expert_keys_dict)
            sz = _tensor_bytes(expert_keys_dict)
            logger.info_rank0(
                f"[DIAG]   _convert_and_load({label}): {n_keys} expert keys, "
                f"{_fmt_bytes(sz)} | {_get_mem_info()}"
            )
            t0 = time.monotonic()
            converted = _apply_weight_conversions_to_state_dict(model, expert_keys_dict, weight_mapping)
            conv_sz = _tensor_bytes(converted)
            logger.info_rank0(
                f"[DIAG]   conversion done: {len(converted)} converted keys, "
                f"{_fmt_bytes(conv_sz)} | {_get_mem_info()}"
            )
            _load_direct(converted, label=f"converted-{label}")
            del converted

        loading_start = time.monotonic()

        for i, ckpt_file in enumerate(checkpoint_files):
            shard_name = os.path.basename(ckpt_file)
            shard_file_size = os.path.getsize(ckpt_file) if os.path.exists(ckpt_file) else 0
            logger.info_rank0(
                f"[DIAG] --- Shard {i + 1}/{len(checkpoint_files)}: {shard_name} "
                f"(file: {_fmt_bytes(shard_file_size)}) | {_get_mem_info()}"
            )
            t_shard = time.monotonic()
            shard_state_dict = load_state_dict(
                ckpt_file, map_location="cpu", weights_only=load_config.weights_only
            )
            shard_keys = len(shard_state_dict)
            shard_bytes = _tensor_bytes(shard_state_dict)
            logger.info_rank0(
                f"[DIAG]   Shard loaded from disk: {shard_keys} keys, {_fmt_bytes(shard_bytes)} "
                f"in {time.monotonic() - t_shard:.1f}s | {_get_mem_info()}"
            )

            if not has_weight_converters:
                # Simple path: no conversions, load entire shard directly
                _load_direct(shard_state_dict, label=f"shard-{i+1}")
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

                expert_bytes = _tensor_bytes(expert_keys)
                non_expert_bytes = _tensor_bytes(non_expert_keys)
                logger.info_rank0(
                    f"[DIAG]   Split: {len(non_expert_keys)} non-expert keys ({_fmt_bytes(non_expert_bytes)}), "
                    f"{len(expert_keys)} expert keys ({_fmt_bytes(expert_bytes)})"
                )

                # Non-expert keys: load immediately (key names match model directly)
                _load_direct(non_expert_keys, label=f"shard-{i+1}-non-expert")
                del non_expert_keys

                # Expert keys: split into boundary-layer (per-layer buffer) vs complete-layer (convert now)
                complete_expert_keys = {}
                buffered_this_shard = 0
                buffered_this_shard_bytes = 0
                for key in list(expert_keys.keys()):
                    match = re.search(r"\.layers\.(\d+)\.", key)
                    if match and int(match.group(1)) in multi_shard_layers:
                        layer_id = int(match.group(1))
                        if layer_id not in per_layer_buffer:
                            per_layer_buffer[layer_id] = {}
                        tensor = expert_keys.pop(key)
                        buf_bytes = tensor.nbytes if hasattr(tensor, "nbytes") else tensor.nelement() * tensor.element_size()
                        total_buffer_bytes += buf_bytes
                        buffered_this_shard += 1
                        buffered_this_shard_bytes += buf_bytes
                        per_layer_buffer[layer_id][key] = tensor
                    else:
                        complete_expert_keys[key] = expert_keys.pop(key)
                del expert_keys

                if buffered_this_shard > 0:
                    logger.info_rank0(
                        f"[DIAG]   Buffered {buffered_this_shard} boundary expert keys "
                        f"({_fmt_bytes(buffered_this_shard_bytes)}) across {len(per_layer_buffer)} layers, "
                        f"total buffer: {_fmt_bytes(total_buffer_bytes)}"
                    )

                # Convert and load complete-layer expert keys immediately
                if complete_expert_keys:
                    _convert_and_load(complete_expert_keys, label=f"shard-{i+1}-complete")
                del complete_expert_keys

                # Check for completed boundary layers and load them immediately
                newly_completed = [
                    lid for lid in per_layer_buffer
                    if len(per_layer_buffer[lid]) >= boundary_layer_info[lid]
                ]
                for layer_id in sorted(newly_completed):
                    layer_keys = per_layer_buffer.pop(layer_id)
                    freed_bytes = _tensor_bytes(layer_keys)
                    total_buffer_bytes -= freed_bytes
                    logger.info_rank0(
                        f"[DIAG]   Boundary layer {layer_id} complete: "
                        f"{len(layer_keys)}/{boundary_layer_info[layer_id]} keys, "
                        f"freeing {_fmt_bytes(freed_bytes)} from buffer"
                    )
                    _convert_and_load(layer_keys, label=f"boundary-L{layer_id}")
                    del layer_keys
                    completed_boundary_layers += 1
                if newly_completed:
                    logger.info_rank0(
                        f"[DIAG]   Completed {len(newly_completed)} boundary layer(s) "
                        f"({completed_boundary_layers}/{len(boundary_layer_info)} total), "
                        f"remaining buffer: {len(per_layer_buffer)} layers, {_fmt_bytes(total_buffer_bytes)}"
                    )

            gc.collect()
            logger.info_rank0(
                f"[DIAG]   Shard {i + 1} done (gc.collect), elapsed: {time.monotonic() - t_shard:.1f}s | "
                f"{_get_mem_info()}"
            )

        # Handle any remaining incomplete boundary layers (shouldn't happen if index is accurate)
        if per_layer_buffer:
            remaining_keys = sum(len(keys) for keys in per_layer_buffer.values())
            logger.warning_rank0(
                f"[DIAG] Loading {remaining_keys} expert keys from {len(per_layer_buffer)} "
                f"incomplete boundary layers ({_fmt_bytes(total_buffer_bytes)}) "
                f"(may indicate index mismatch)"
            )
            remaining = {}
            for layer_keys in per_layer_buffer.values():
                remaining.update(layer_keys)
            per_layer_buffer.clear()
            total_buffer_bytes = 0
            _convert_and_load(remaining, label="remaining-boundary")
            del remaining
            gc.collect()

        total_time = time.monotonic() - loading_start
        logger.info_rank0(
            f"[DIAG] ===== Shard-by-shard loading DONE ====="
            f"\n[DIAG] Total time: {total_time:.1f}s, "
            f"boundary layers completed: {completed_boundary_layers}/{len(boundary_layer_info)}, "
            f"errors: {len(all_error_msgs)}"
            f"\n[DIAG] Memory AFTER loading: {_get_mem_info()}"
        )

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
