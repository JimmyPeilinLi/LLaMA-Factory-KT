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
from transformers import Trainer
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

        # Collect MoE LoRA parameters
        self._moe_lora_params = []
        for wrapper in self._kt_wrappers:
            self._moe_lora_params.extend(wrapper.lora_params.values())

        if self._kt_wrappers:
            logger.info_rank0(
                f"KT Trainer initialized with {len(self._kt_wrappers)} MoE layers, "
                f"{len(self._moe_lora_params)} MoE LoRA parameters, "
                f"TP mode: {self._kt_tp_enabled}"
            )

        # Disable cache for training
        self.model.config.use_cache = False

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

        # Group 1: MoE LoRA parameters (no weight decay for LoRA)
        if self._moe_lora_params:
            moe_lora_param_group = {
                "params": self._moe_lora_params,
                "weight_decay": 0.0,
                "name": "moe_lora",
            }
            param_groups.append(moe_lora_param_group)
            logger.info_rank0(f"Added {len(self._moe_lora_params)} MoE LoRA parameters to optimizer")

        # Group 2: Other trainable parameters (Attention LoRA, etc.)
        moe_lora_param_ids = {id(p) for p in self._moe_lora_params}

        decay_params = []
        nodecay_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if id(param) in moe_lora_param_ids:
                continue  # Already added to MoE LoRA group

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
        """Ensure MoE LoRA parameters are in the optimizer (for custom optimizers)."""
        if not self._moe_lora_params:
            return

        # Check if MoE LoRA params are already in optimizer
        optimizer_param_ids = set()
        for group in self.optimizer.param_groups:
            for param in group["params"]:
                optimizer_param_ids.add(id(param))

        missing_params = [p for p in self._moe_lora_params if id(p) not in optimizer_param_ids]

        if missing_params:
            # Add missing MoE LoRA parameters to optimizer
            self.optimizer.add_param_group({
                "params": missing_params,
                "weight_decay": 0.0,
                "name": "moe_lora",
            })
            logger.info_rank0(f"Added {len(missing_params)} missing MoE LoRA parameters to optimizer")

    @override
    def training_step(
        self, model: "torch.nn.Module", inputs: dict[str, Union["torch.Tensor", Any]], num_items_in_batch: int = None
    ) -> "torch.Tensor":
        """
        Perform a training step.

        This overrides the parent method to update LoRA weight pointers
        after optimizer.step() in TP mode.

        Args:
            model: The model to train
            inputs: The inputs and targets of the model
            num_items_in_batch: Number of items in the batch (for loss scaling)

        Returns:
            The training loss
        """
        loss = super().training_step(model, inputs, num_items_in_batch)

        # In TP mode, update LoRA weight pointers after gradient is computed
        # Note: The actual pointer update happens after optimizer.step() in _inner_training_loop
        # This is handled by the callback below

        return loss

    def _update_lora_pointers(self):
        """
        Update AMX operator's LoRA weight pointers.

        This must be called after optimizer.step() in TP mode because
        the optimizer may modify the underlying tensor storage.
        """
        if self._kt_tp_enabled and self._kt_wrappers:
            from ...model.model_utils.kt_moe import update_kt_lora_pointers
            update_kt_lora_pointers(self.model)

    def _maybe_log_save_evaluate(self, tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval, start_time, learning_rate=None):
        """Override to update LoRA pointers before evaluation."""
        # Update LoRA pointers before any evaluation
        self._update_lora_pointers()
        return super()._maybe_log_save_evaluate(tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval, start_time, learning_rate=learning_rate)

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

        # 2. Merge MoE LoRA into the adapter file
        if output_dir is None:
            output_dir = self.args.output_dir

        if self._kt_wrappers:
            from ...model.model_utils.kt_moe import save_moe_lora_to_adapter
            save_moe_lora_to_adapter(self.model, output_dir)
            logger.info_rank0(f"Saved MoE LoRA to {output_dir}")


class KTAMXOptimizerCallback:
    """
    Callback to handle LoRA weight pointer updates after optimizer.step().

    This is used internally by KTrainer to ensure LoRA pointers are
    synchronized after each optimizer update in TP mode.
    """

    def __init__(self, trainer: "KTrainer"):
        self.trainer = trainer

    def on_step_end(self, args, state, control, **kwargs):
        """Called after optimizer.step() and lr_scheduler.step()."""
        self.trainer._update_lora_pointers()

    def on_train_end(self, args, state, control, **kwargs):
        """Called at the end of training."""
        # Final pointer update
        self.trainer._update_lora_pointers()


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

    # Add optimizer callback for LoRA pointer updates in TP mode
    if getattr(model, "_kt_tp_enabled", False):
        from transformers import TrainerCallback

        class _KTAMXCallback(TrainerCallback):
            def __init__(self, kt_trainer):
                self.kt_trainer = kt_trainer

            def on_step_end(self, args, state, control, **kwargs):
                self.kt_trainer._update_lora_pointers()

        trainer.add_callback(_KTAMXCallback(trainer))
        logger.info_rank0("Added KT LoRA pointer update callback for TP mode")

    return trainer
