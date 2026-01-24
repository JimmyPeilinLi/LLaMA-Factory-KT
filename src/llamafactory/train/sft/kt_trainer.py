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

"""
KT Trainer for SFT (Supervised Fine-Tuning)

This trainer extends CustomSeq2SeqTrainer to support KTransformers MoE backend.
It handles:
- MoE LoRA parameters (managed by KT) + Attention LoRA parameters (managed by peft)
- LoRA weight pointer updates after optimizer.step() in TP mode
"""

from typing import TYPE_CHECKING, Any, Optional, Union

import torch
from transformers import Trainer, TrainerCallback
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.trainer_pt_utils import get_parameter_names
from typing_extensions import override

from ...extras import logging
from .trainer import CustomSeq2SeqTrainer


if TYPE_CHECKING:
    from torch.utils.data import Dataset
    from transformers import ProcessorMixin
    from transformers.trainer import PredictionOutput

    from ...hparams import FinetuningArguments, ModelArguments, TrainingArguments


logger = logging.get_logger(__name__)


class KTrainer(CustomSeq2SeqTrainer):
    """
    SFT Trainer with KTransformers AMX MOE backend support.

    This trainer extends CustomSeq2SeqTrainer to handle:
    - MoE LoRA parameters from KT wrappers
    - Attention LoRA parameters from peft
    - LoRA weight pointer synchronization in TP mode
    """

    def __init__(
        self,
        finetuning_args: "FinetuningArguments",
        processor: Optional["ProcessorMixin"],
        model_args: Optional["ModelArguments"] = None,
        gen_kwargs: Optional[dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        """
        Initialize KT Trainer.

        Args:
            finetuning_args: Finetuning arguments
            processor: Optional processor for multimodal models
            model_args: Model arguments containing KT configuration
            gen_kwargs: Generation keyword arguments
            **kwargs: Additional arguments passed to CustomSeq2SeqTrainer
        """
        super().__init__(
            finetuning_args=finetuning_args,
            processor=processor,
            model_args=model_args,
            gen_kwargs=gen_kwargs,
            **kwargs,
        )

        self.model_args = model_args

        # Get KT wrappers from model
        self._kt_wrappers = getattr(self.model, "_kt_wrappers", [])
        self._kt_tp_enabled = getattr(self.model, "_kt_tp_enabled", False)
        self._kt_use_lora_experts = getattr(self.model, "_kt_use_lora_experts", False)

        # Collect MoE LoRA parameters (per-expert LoRA mode)
        self._moe_lora_params = []
        # Collect LoRA Experts parameters (LoRA Experts mode)
        self._lora_expert_params = []

        for wrapper in self._kt_wrappers:
            if wrapper.lora_params is not None:
                trainable_lora_params = [p for p in wrapper.lora_params.values() if p.requires_grad]
                self._moe_lora_params.extend(trainable_lora_params)
            if wrapper.lora_experts is not None:
                self._lora_expert_params.extend(wrapper.lora_experts.parameters())

        if self._kt_wrappers:
            has_moe_lora = len(self._moe_lora_params) > 0
            has_lora_experts = len(self._lora_expert_params) > 0

            if has_moe_lora and has_lora_experts:
                mode_str = "LoRA Experts + LoRA (both trained)"
            elif has_lora_experts:
                mode_str = "LoRA Experts + SkipLoRA (only LoRA Experts trained)"
            elif has_moe_lora:
                mode_str = "Normal LoRA (per-expert LoRA trained)"
            else:
                mode_str = "SkipLoRA (MoE frozen)"

            logger.info_rank0(
                f"KT Trainer initialized with {len(self._kt_wrappers)} MoE layers, "
                f"mode={mode_str}, "
                f"moe_lora_params={len(self._moe_lora_params)}, "
                f"lora_expert_params={len(self._lora_expert_params)}, "
                f"TP mode: {self._kt_tp_enabled}"
            )

        # Disable cache for training
        self.model.config.use_cache = False

        # Flag for computation graph printing
        self._graph_printed = False

    @override
    def compute_loss(self, model, inputs, *args, **kwargs):
        """Compute loss and optionally print computation graph on first call."""
        loss = super().compute_loss(model, inputs, *args, **kwargs)

        # # Print computation graph on first step (before backward/detach)
        # if not self._graph_printed and loss.grad_fn is not None:
        #     self._graph_printed = True
        #     try:
        #         from torchviz import make_dot
        #         # Only include trainable parameters to avoid huge graph
        #         trainable_params = {k: v for k, v in model.named_parameters() if v.requires_grad}
        #         dot = make_dot(loss, params=trainable_params, show_attrs=True, show_saved=True)
        #         dot.render("computation_graph", format="svg")
        #         logger.info_rank0(f"Computation graph saved to computation_graph.png (with {len(trainable_params)} trainable params)")
        #     except ImportError:
        #         logger.warning_rank0("torchviz not installed. Install with: pip install torchviz graphviz")
        #     except Exception as e:
        #         logger.warning_rank0(f"Failed to generate computation graph: {e}")

        return loss

    @override
    def create_optimizer(self) -> "torch.optim.Optimizer":
        """
        Create optimizer that includes both MoE LoRA and Attention LoRA parameters.

        The optimizer handles:
        - MoE LoRA parameters (from KT wrappers, on CPU)
        - Attention LoRA parameters (from peft, on GPU)
        - Other trainable parameters

        Returns:
            Optimizer instance
        """
        if self.optimizer is not None:
            return self.optimizer

        # Check if custom optimizer is already created
        from ..trainer_utils import create_custom_optimizer
        self.optimizer = create_custom_optimizer(self.model, self.args, self.finetuning_args)

        if self.optimizer is not None:
            # Custom optimizer already handles all parameters
            # But we need to ensure MoE LoRA params are included
            self._ensure_moe_lora_in_optimizer()
            return self.optimizer

        # Create standard optimizer with MoE LoRA parameters included
        decay_parameters = self._get_decay_parameter_names()

        # Collect all trainable parameters
        param_groups = []

        # Group 1: MoE LoRA parameters (no weight decay for LoRA) - per-expert LoRA mode
        if self._moe_lora_params:
            moe_lora_param_group = {
                "params": self._moe_lora_params,
                "weight_decay": 0.0,
                "name": "moe_lora",
            }
            param_groups.append(moe_lora_param_group)
            logger.info_rank0(f"Added {len(self._moe_lora_params)} MoE LoRA parameters to optimizer")

        # Group 1b: LoRA Experts parameters (no weight decay) - LoRA Experts mode
        if self._lora_expert_params:
            lora_expert_param_group = {
                "params": self._lora_expert_params,
                "weight_decay": 0.0,
                "name": "lora_experts",
            }
            param_groups.append(lora_expert_param_group)
            logger.info_rank0(f"Added {len(self._lora_expert_params)} LoRA Expert parameters to optimizer")

        # Group 2: Other trainable parameters (Attention LoRA, etc.)
        moe_lora_param_ids = {id(p) for p in self._moe_lora_params}
        lora_expert_param_ids = {id(p) for p in self._lora_expert_params}
        kt_param_ids = moe_lora_param_ids | lora_expert_param_ids

        decay_params = []
        nodecay_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if id(param) in kt_param_ids:
                continue  # Already added to MoE LoRA or LoRA Experts group

            if name in decay_parameters:
                decay_params.append(param)
            else:
                nodecay_params.append(param)

        if decay_params:
            param_groups.append({
                "params": decay_params,
                "weight_decay": self.args.weight_decay,
                "name": "decay",
            })

        if nodecay_params:
            param_groups.append({
                "params": nodecay_params,
                "weight_decay": 0.0,
                "name": "no_decay",
            })

        # Get optimizer class and kwargs
        optim_class, optim_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)

        # Create optimizer
        self.optimizer = optim_class(param_groups, **optim_kwargs)

        logger.info_rank0(
            f"Created optimizer with {len(param_groups)} parameter groups, "
            f"total parameters: {sum(len(g['params']) for g in param_groups)}"
        )

        return self.optimizer

    def _get_decay_parameter_names(self) -> set[str]:
        """Get names of parameters that should have weight decay."""
        decay_parameters = get_parameter_names(self.model, ALL_LAYERNORM_LAYERS)
        decay_parameters = [name for name in decay_parameters if "bias" not in name]
        return set(decay_parameters)

    def _ensure_moe_lora_in_optimizer(self):
        """Ensure MoE LoRA and LoRA Experts parameters are in the optimizer (for custom optimizers)."""
        # Check if MoE LoRA params are already in optimizer
        optimizer_param_ids = set()
        for group in self.optimizer.param_groups:
            for param in group["params"]:
                optimizer_param_ids.add(id(param))

        # Check MoE LoRA params (per-expert LoRA mode)
        if self._moe_lora_params:
            missing_moe_lora = [p for p in self._moe_lora_params if id(p) not in optimizer_param_ids]
            if missing_moe_lora:
                self.optimizer.add_param_group({
                    "params": missing_moe_lora,
                    "weight_decay": 0.0,
                    "name": "moe_lora",
                })
                logger.info_rank0(f"Added {len(missing_moe_lora)} missing MoE LoRA parameters to optimizer")

        # Check LoRA Experts params (LoRA Experts mode)
        if self._lora_expert_params:
            missing_lora_experts = [p for p in self._lora_expert_params if id(p) not in optimizer_param_ids]
            if missing_lora_experts:
                self.optimizer.add_param_group({
                    "params": missing_lora_experts,
                    "weight_decay": 0.0,
                    "name": "lora_experts",
                })
                logger.info_rank0(f"Added {len(missing_lora_experts)} missing LoRA Expert parameters to optimizer")

    @override
    def training_step(
        self, model: "torch.nn.Module", inputs: dict[str, Union["torch.Tensor", Any]], num_items_in_batch: int = None
    ) -> "torch.Tensor":
        """
        Perform a training step.

        Args:
            model: The model to train
            inputs: The inputs and targets of the model
            num_items_in_batch: Number of items in the batch (for loss scaling)

        Returns:
            The training loss
        """
        loss = super().training_step(model, inputs, num_items_in_batch)
        return loss

    @override
    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        """
        Save model with both Attention LoRA (PEFT) and MoE LoRA (KT).

        This method:
        1. Calls parent's save_model() to save Attention LoRA via PEFT
        2. Merges MoE LoRA weights into the adapter file

        Args:
            output_dir: Directory to save the model. Uses args.output_dir if None.
            _internal_call: Internal flag from HuggingFace trainer
        """
        # 1. Call parent to save Attention LoRA via PEFT
        super().save_model(output_dir, _internal_call)

        # 2. Merge KT MoE weights (LoRA or LoRA Experts) into the adapter file
        if output_dir is None:
            output_dir = self.args.output_dir

        if self._kt_wrappers:
            from ...model.model_utils.kt_moe import save_kt_moe_to_adapter
            save_kt_moe_to_adapter(self.model, output_dir)
            logger.info_rank0(f"Saved KT MoE weights to {output_dir}")


class _KTLoRAPointerCallback(TrainerCallback):
    def __init__(self, model: "torch.nn.Module"):
        self.model = model

    def on_optimizer_step(self, args, state, control, **kwargs):
        from ...model.model_utils.kt_moe import update_kt_lora_pointers
        update_kt_lora_pointers(self.model)


def create_kt_trainer(
    model: "torch.nn.Module",
    training_args: "TrainingArguments",
    finetuning_args: "FinetuningArguments",
    model_args: "ModelArguments",
    data_collator: Any,
    tokenizer_module: dict,
    callbacks: Optional[list] = None,
    **kwargs,
) -> KTrainer:
    """
    Factory function to create KTrainer with proper configuration.

    Args:
        model: The model to train
        training_args: Training arguments
        finetuning_args: Finetuning arguments
        model_args: Model arguments
        data_collator: Data collator for batching
        tokenizer_module: Tokenizer module dict (tokenizer + processor)
        callbacks: Optional list of callbacks
        **kwargs: Additional arguments for trainer

    Returns:
        Configured KTrainer instance
    """
    trainer = KTrainer(
        model=model,
        args=training_args,
        finetuning_args=finetuning_args,
        model_args=model_args,
        data_collator=data_collator,
        callbacks=callbacks,
        **tokenizer_module,
        **kwargs,
    )

    # Add LoRA pointer callback if per-expert LoRA is trainable (Mode 1 or Mode 3)
    # Mode 1: Normal LoRA - per-expert LoRA trained
    # Mode 3: LoRA Experts + LoRA - both per-expert LoRA and LoRA Experts trained
    if trainer._moe_lora_params:
        trainer.add_callback(_KTLoRAPointerCallback(model))
        logger.info_rank0("Added KT LoRA pointer update callback (on_optimizer_step)")

    return trainer
