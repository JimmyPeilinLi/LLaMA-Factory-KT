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

import os
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import yaml
from accelerate.utils import KTransformersPlugin
from peft import LoraConfig, PeftModel, get_peft_model
from safetensors.torch import load_file
from torch import nn
from torch.utils.checkpoint import checkpoint
from transformers import HfArgumentParser, LlamaConfig, LlamaForCausalLM, PretrainedConfig, PreTrainedModel
from transformers.integrations.kt import _get_kt_config
from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeTextConfig
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeForCausalLM

from llamafactory.hparams import FinetuningArguments
from llamafactory.hparams.model_args import KTransformersArguments
from llamafactory.model.adapter import _setup_lora_tuning
from llamafactory.model.model_utils.lora_expert import (
    LORA_EXPERT_CONFIG_NAME,
    LORA_EXPERT_MODULE_NAME,
    LoRAExpertMLP,
    LoRAExperts,
    attach_lora_experts,
    get_lora_expert_config,
    load_lora_expert_config,
    save_lora_expert_config,
)
from llamafactory.train.callbacks import SaveLoRAExpertConfigCallback


class TinyConfig(PretrainedConfig):
    model_type = "tiny_lora_expert"

    def __init__(self, hidden_size=8, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size


class Qwen3MoeSparseMoeBlock(nn.Module):
    def forward(self, hidden_states):
        return hidden_states * 0.5


class Qwen3_5MoeSparseMoeBlock(nn.Module):
    def forward(self, hidden_states):
        return hidden_states * 0.5, hidden_states.new_tensor(7.0)


class TinyModel(PreTrainedModel):
    config_class = TinyConfig

    def __init__(self, config, tuple_output=False):
        super().__init__(config)
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.mlp = Qwen3_5MoeSparseMoeBlock() if tuple_output else Qwen3MoeSparseMoeBlock()

    def forward(self, hidden_states):
        return self.mlp(self.q_proj(hidden_states))


def randomize_down(module):
    with torch.no_grad():
        for expert in module.experts:
            expert.le_down.weight.normal_()


def test_formula_zero_init_gradients_and_parameter_contract():
    torch.manual_seed(0)
    module = LoRAExperts(hidden_size=5, intermediate_size=3, num_experts=2)
    names = {
        f"experts.{idx}.{projection}.weight" for idx in range(2) for projection in ("le_gate", "le_up", "le_down")
    }
    assert set(dict(module.named_parameters())) == names
    assert sum(param.numel() for param in module.parameters()) == 3 * 2 * 5 * 3

    hidden_states = torch.randn(2, 4, 5)
    base_output = torch.randn_like(hidden_states)
    assert torch.count_nonzero(module(hidden_states)).item() == 0
    assert torch.equal(base_output + module(hidden_states), base_output)

    module(hidden_states).sum().backward()
    for expert in module.experts:
        assert torch.count_nonzero(expert.le_down.weight.grad).item() > 0
        assert torch.count_nonzero(expert.le_gate.weight.grad).item() == 0
        assert torch.count_nonzero(expert.le_up.weight.grad).item() == 0

    optimizer = torch.optim.SGD(module.parameters(), lr=0.1)
    optimizer.step()
    optimizer.zero_grad()
    module(hidden_states).sum().backward()
    for expert in module.experts:
        assert torch.count_nonzero(expert.le_gate.weight.grad).item() > 0
        assert torch.count_nonzero(expert.le_up.weight.grad).item() > 0

    randomize_down(module)
    expected = sum(
        expert.le_down(torch.nn.functional.silu(expert.le_gate(hidden_states)) * expert.le_up(hidden_states))
        for expert in module.experts
    ) / len(module.experts)
    torch.testing.assert_close(module(hidden_states), expected, rtol=0, atol=0)


def test_e_experts_equal_one_wide_swiglu_mlp():
    torch.manual_seed(1)
    module = LoRAExperts(hidden_size=5, intermediate_size=3, num_experts=4)
    randomize_down(module)
    wide = LoRAExpertMLP(hidden_size=5, intermediate_size=12)
    with torch.no_grad():
        wide.le_gate.weight.copy_(torch.cat([expert.le_gate.weight for expert in module.experts]))
        wide.le_up.weight.copy_(torch.cat([expert.le_up.weight for expert in module.experts]))
        wide.le_down.weight.copy_(torch.cat([expert.le_down.weight for expert in module.experts], dim=1) / 4)

    hidden_states = torch.randn(2, 4, 5)
    torch.testing.assert_close(module(hidden_states), wide(hidden_states), rtol=1e-6, atol=1e-6)


def test_native_kt_tuple_and_gradient_checkpointing():
    torch.manual_seed(2)
    native = TinyModel(TinyConfig())
    kt = TinyModel(TinyConfig())
    kt.load_state_dict(native.state_dict())
    kt._kt_wrappers = [kt.mlp]

    torch.manual_seed(3)
    native_config = attach_lora_experts(native, num_experts=2, intermediate_size=3, use_kt=False)
    torch.manual_seed(3)
    kt_config = attach_lora_experts(kt, num_experts=2, intermediate_size=3, use_kt=True)
    randomize_down(getattr(native.mlp, LORA_EXPERT_MODULE_NAME))
    getattr(kt.mlp, LORA_EXPERT_MODULE_NAME).load_state_dict(getattr(native.mlp, LORA_EXPERT_MODULE_NAME).state_dict())
    hidden_states = torch.randn(2, 4, 8, requires_grad=True)
    assert native_config == kt_config
    torch.testing.assert_close(native(hidden_states), kt(hidden_states), rtol=0, atol=0)

    checkpoint(native, hidden_states, use_reentrant=False).sum().backward()
    assert all(
        expert.le_down.weight.grad is not None for expert in getattr(native.mlp, LORA_EXPERT_MODULE_NAME).experts
    )

    tuple_model = TinyModel(TinyConfig(), tuple_output=True)
    attach_lora_experts(tuple_model, num_experts=2, intermediate_size=3, use_kt=False)
    output, auxiliary = tuple_model(hidden_states.detach())
    assert output.shape == hidden_states.shape
    assert auxiliary.item() == 7.0


def test_real_qwen3_5_moe_native_zero_init_is_exact_noop():
    config = Qwen3_5MoeTextConfig(
        vocab_size=32,
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=32,
        moe_intermediate_size=8,
        shared_expert_intermediate_size=8,
        num_experts_per_tok=2,
        num_experts=4,
        layer_types=["full_attention"],
        use_cache=False,
    )
    model = Qwen3_5MoeForCausalLM(config).eval()
    input_ids = torch.randint(0, config.vocab_size, (2, 5))
    with torch.no_grad():
        original_logits = model(input_ids=input_ids, use_cache=False).logits

    expert_config = attach_lora_experts(model, num_experts=2, intermediate_size=4, use_kt=False)
    with torch.no_grad():
        injected_logits = model(input_ids=input_ids, use_cache=False).logits

    assert expert_config.target_modules == ("model.layers.0.mlp",)
    assert model.model.layers[0].mlp.__class__.__name__ == "Qwen3_5MoeSparseMoeBlock"
    assert torch.equal(injected_logits, original_logits)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_gpu_placement_and_fsdp2(tmp_path):
    model = TinyModel(TinyConfig()).cuda()
    attach_lora_experts(model, num_experts=2, intermediate_size=3, use_kt=False)
    module = getattr(model.mlp, LORA_EXPERT_MODULE_NAME)
    assert {parameter.device.type for parameter in module.parameters()} == {"cuda"}
    assert model(torch.randn(2, 4, 8, device="cuda")).device.type == "cuda"

    if not dist.is_available() or dist.is_initialized():
        return

    from torch.distributed.fsdp import fully_shard

    dist.init_process_group("nccl", init_method=f"file://{tmp_path / 'fsdp2_init'}", rank=0, world_size=1)
    try:
        model = fully_shard(model)
        model(torch.randn(2, 4, 8, device="cuda")).sum().backward()
        gradients = [
            parameter.grad
            for name, parameter in model.named_parameters()
            if LORA_EXPERT_MODULE_NAME in name and name.endswith("le_down.weight")
        ]
        assert gradients and all(gradient is not None for gradient in gradients)
    finally:
        dist.destroy_process_group()


def test_peft_trainable_keys_safetensor_and_reload(tmp_path):
    torch.manual_seed(4)
    base = TinyModel(TinyConfig())
    expert_config = attach_lora_experts(base, num_experts=2, intermediate_size=3, use_kt=False)
    model = get_peft_model(
        base,
        LoraConfig(
            r=2,
            lora_alpha=4,
            target_modules=["q_proj"],
            modules_to_save=[LORA_EXPERT_MODULE_NAME],
        ),
    )
    trainable_names = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    assert trainable_names == {
        "base_model.model.q_proj.lora_A.default.weight",
        "base_model.model.q_proj.lora_B.default.weight",
        *{
            f"base_model.model.mlp.{LORA_EXPERT_MODULE_NAME}.modules_to_save.default.experts.{idx}.{projection}.weight"
            for idx in range(2)
            for projection in ("le_gate", "le_up", "le_down")
        },
    }
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if parameter.requires_grad and ("lora_B" in name or "le_down" in name):
                parameter.normal_()

    hidden_states = torch.randn(2, 4, 8)
    expected = model(hidden_states)
    model.save_pretrained(tmp_path)
    save_lora_expert_config(expert_config, tmp_path)
    assert (tmp_path / LORA_EXPERT_CONFIG_NAME).is_file()
    state = load_file(tmp_path / "adapter_model.safetensors")
    for fragment in ("lora_A", "lora_B", "le_gate", "le_up", "le_down"):
        assert any(fragment in key for key in state)

    loaded_config = load_lora_expert_config(tmp_path)
    assert loaded_config == expert_config
    torch.manual_seed(4)
    reloaded_base = TinyModel(TinyConfig())
    attach_lora_experts(reloaded_base, config=loaded_config, use_kt=False)
    reloaded = PeftModel.from_pretrained(reloaded_base, tmp_path)
    torch.testing.assert_close(reloaded(hidden_states), expected, rtol=0, atol=0)


def make_tiny_llama_moe():
    config = LlamaConfig(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=32,
    )
    model = LlamaForCausalLM(config)
    model.model.layers[0].mlp = Qwen3MoeSparseMoeBlock()
    return model


def make_model_args(adapter_path=None):
    return SimpleNamespace(
        adapter_name_or_path=None if adapter_path is None else [str(adapter_path)],
        adapter_folder=None,
        offload_folder=None,
        cache_dir=None,
        model_revision="main",
        hf_hub_token=None,
        use_kt=False,
        use_unsloth=False,
        resize_vocab=False,
    )


def test_lf_adapter_automatically_creates_and_resumes_lora_expert(tmp_path):
    torch.manual_seed(7)
    base = make_tiny_llama_moe()
    finetuning_args = FinetuningArguments(
        lora_target="q_proj",
        lora_rank=2,
        lora_alpha=4,
        use_lora_expert=True,
        lora_expert_num=2,
        lora_expert_intermediate_size=3,
    )
    model = _setup_lora_tuning(
        base.config,
        base,
        make_model_args(),
        finetuning_args,
        is_trainable=True,
        cast_trainable_params_to_fp32=False,
    )
    expert_config = get_lora_expert_config(model)
    assert expert_config is not None
    assert expert_config.target_modules == ("model.layers.0.mlp",)
    assert model.peft_config["default"].modules_to_save == [LORA_EXPERT_MODULE_NAME]

    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if parameter.requires_grad and ("lora_B" in name or "le_down" in name):
                parameter.normal_()

    model.eval()
    input_ids = torch.randint(0, base.config.vocab_size, (2, 5))
    expected_logits = model(input_ids=input_ids).logits
    model.save_pretrained(tmp_path)
    save_lora_expert_config(expert_config, tmp_path)

    torch.manual_seed(7)
    resumed_base = make_tiny_llama_moe()
    resumed_args = FinetuningArguments(lora_target="q_proj")
    resumed = _setup_lora_tuning(
        resumed_base.config,
        resumed_base,
        make_model_args(tmp_path),
        resumed_args,
        is_trainable=True,
        cast_trainable_params_to_fp32=False,
    )
    resumed.eval()

    assert resumed_args.use_lora_expert is True
    assert get_lora_expert_config(resumed) == expert_config
    torch.testing.assert_close(resumed(input_ids=input_ids).logits, expected_logits, rtol=0, atol=0)

    torch.manual_seed(7)
    inference_base = make_tiny_llama_moe()
    inference_args = FinetuningArguments(lora_target="q_proj")
    merged = _setup_lora_tuning(
        inference_base.config,
        inference_base,
        make_model_args(tmp_path),
        inference_args,
        is_trainable=False,
        cast_trainable_params_to_fp32=False,
    )
    merged.eval()
    assert not isinstance(merged, PeftModel)
    assert get_lora_expert_config(merged) == expert_config
    torch.testing.assert_close(merged(input_ids=input_ids).logits, expected_logits, rtol=1e-6, atol=1e-8)


def test_lora_expert_callback_saves_checkpoint_and_final_metadata(tmp_path):
    model = make_tiny_llama_moe()
    expert_config = attach_lora_experts(model, num_experts=2, intermediate_size=3, use_kt=False)

    class DistributedWrapper(nn.Module):
        def __init__(self, wrapped_model):
            super().__init__()
            self.module = wrapped_model

    callback = SaveLoRAExpertConfigCallback()
    args = SimpleNamespace(should_save=True, output_dir=str(tmp_path))
    state = SimpleNamespace(global_step=9)
    control = SimpleNamespace()
    wrapped_model = DistributedWrapper(model)
    callback.on_save(args, state, control, model=wrapped_model)
    callback.on_train_end(args, state, control, model=wrapped_model)

    assert load_lora_expert_config(tmp_path / "checkpoint-9") == expert_config
    assert load_lora_expert_config(tmp_path) == expert_config


def test_standard_yaml_and_legacy_kt_validation(monkeypatch):
    values = yaml.safe_load(
        """
        finetuning_type: lora
        use_lora_expert: true
        lora_expert_num: 2
        lora_expert_intermediate_size: 16
        """
    )
    (args,) = HfArgumentParser(FinetuningArguments).parse_dict(values)
    assert (args.use_lora_expert, args.lora_expert_num, args.lora_expert_intermediate_size) == (True, 2, 16)

    with pytest.raises(ValueError, match="only valid for LoRA"):
        FinetuningArguments(finetuning_type="full", use_lora_expert=True)
    with pytest.raises(ValueError, match="greater than 0"):
        FinetuningArguments(use_lora_expert=True, lora_expert_num=0)
    with pytest.raises(ValueError, match="greater than 0"):
        FinetuningArguments(use_lora_expert=True, lora_expert_intermediate_size=0)

    training_args = SimpleNamespace(
        hf_kt_config=None,
        kt_config=None,
        accelerator_config=SimpleNamespace(kt_config=None),
        gradient_checkpointing=False,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=3,
        do_eval=True,
    )
    with pytest.raises(ValueError, match="use_lora_expert"):
        KTransformersArguments(use_kt=True, kt_use_lora_experts=True).apply_kt_config(args, training_args, 128)
    monkeypatch.setenv("ACCELERATE_KT_USE_LORA_EXPERTS", "true")
    with pytest.raises(ValueError, match="use_lora_expert"):
        KTransformersArguments(use_kt=True).apply_kt_config(args, training_args, 128)

    monkeypatch.delenv("ACCELERATE_KT_USE_LORA_EXPERTS")
    monkeypatch.setenv("WORLD_SIZE", "2")
    with pytest.raises(ValueError, match="greater than 0"):
        KTransformersArguments(use_kt=True, kt_model_max_length=0).apply_kt_config(args, training_args, 128)

    model_args = KTransformersArguments(use_kt=True)
    model_args.apply_kt_config(args, training_args, 128)
    assert model_args.get_kt_config_dict(args, 128)["kt_use_lora_experts"] is False
    assert training_args.kt_config["kt_skip_expert_loading"] is True
    assert training_args.kt_config["kt_use_lora_experts"] is False
    assert training_args.kt_config["kt_model_max_length"] == 1024
    KTransformersArguments(use_kt=True, kt_model_max_length=2048).apply_kt_config(args, training_args, 128)
    assert training_args.kt_config["kt_model_max_length"] == 2048
    assert "enabled" not in training_args.kt_config
    plugin_kwargs = training_args.accelerator_config.kt_config
    assert plugin_kwargs["enabled"] is True
    assert plugin_kwargs["kt_config"] is training_args.kt_config
    assert training_args.hf_kt_config._kt_config is training_args.kt_config
    assert _get_kt_config() is training_args.hf_kt_config
    assert os.environ["ACCELERATE_USE_KT"] == "true"
    assert os.environ["ACCELERATE_KT_USE_LORA_EXPERTS"] == "False"

    plugin = KTransformersPlugin(**plugin_kwargs)
    assert plugin.enabled is True
    assert plugin.kt_config.kt_lora_rank == args.lora_rank
    assert plugin.kt_config.kt_use_lora_experts is False

    legacy_config = {"kt_use_lora_experts": False, "kt_lora_expert_num": 2}
    training_args = SimpleNamespace(
        hf_kt_config=SimpleNamespace(_kt_config=legacy_config),
        kt_config=legacy_config,
        accelerator_config=None,
        gradient_checkpointing=False,
    )
    with pytest.raises(ValueError, match="use_lora_expert"):
        model_args.apply_kt_config(args, training_args, 128)
