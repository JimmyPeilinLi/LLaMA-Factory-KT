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

import copy
import json
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file, save_file
from torch import nn

from llamafactory.hparams import FinetuningArguments
from llamafactory.model.lora_variants.altlora_attn import AltLoraAttnOptimizer
from llamafactory.model.lora_variants.bilora_attn import (
    BiLoraAttnOptimizer,
    export_bilora_primary_adapter,
    register_bilora_eval_hooks,
)
from llamafactory.model.lora_variants.common import (
    KTLoraVariantConfig,
    LoraPair,
    TargetManifest,
    canonical_json_sha256,
    load_variant_config,
    resolve_eligible_gpu_lora_targets,
    resolve_injected_lora_pairs,
    select_target_manifest,
    verify_kt_fused_expert_lora_contract,
)
from llamafactory.model.lora_variants.plop_attn import (
    PLOP_EXACT_TARGETS_NAME,
    PLOP_NFN_SCORES_NAME,
    PLoPAttnScorer,
    compute_nfn_per_token,
    load_plop_training_selection,
    write_plop_artifacts,
)


class ToyLinearAttention(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        for family in ("in_proj_a", "in_proj_b", "in_proj_qkv", "in_proj_z", "out_proj"):
            setattr(self, family, nn.Linear(width, width, bias=False))


class ToyFullAttention(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        for family in ("q_proj", "k_proj", "v_proj", "o_proj"):
            setattr(self, family, nn.Linear(width, width, bias=False))


class ToyLayer(nn.Module):
    def __init__(self, attention: nn.Module, attribute: str):
        super().__init__()
        setattr(self, attribute, attention)


class ToyAttentionModel(nn.Module):
    def __init__(self, width: int = 4):
        super().__init__()
        self.config = SimpleNamespace(layer_types=["linear_attention", "full_attention"])
        self.layers = nn.ModuleList(
            [
                ToyLayer(ToyLinearAttention(width), "linear_attn"),
                ToyLayer(ToyFullAttention(width), "self_attn"),
            ]
        )

    def forward(self, hidden_states):
        for layer in self.layers:
            attention = getattr(layer, "linear_attn", None) or getattr(layer, "self_attn")
            for family in ("in_proj_a", "in_proj_b", "in_proj_qkv", "in_proj_z", "out_proj"):
                projection = getattr(attention, family, None)
                if projection is not None:
                    hidden_states = projection(hidden_states)
            for family in ("q_proj", "k_proj", "v_proj", "o_proj"):
                projection = getattr(attention, family, None)
                if projection is not None:
                    hidden_states = projection(hidden_states)
        return hidden_states


class FakePeftLinear(nn.Module):
    def __init__(self, width: int, rank: int):
        super().__init__()
        self.lora_A = nn.ModuleDict({"default": nn.Linear(width, rank, bias=False)})
        self.lora_B = nn.ModuleDict({"default": nn.Linear(rank, width, bias=False)})
        self.active_adapter = "default"
        self.scaling = {"default": 2.0}

    def forward(self, hidden_states):
        return self.lora_B["default"](self.lora_A["default"](hidden_states))


def make_pair(rank: int = 2, width: int = 2) -> tuple[FakePeftLinear, LoraPair]:
    module = FakePeftLinear(width, rank)
    return module, LoraPair(
        "layers.0.self_attn.q_proj",
        module,
        module.lora_A["default"].weight,
        module.lora_B["default"].weight,
        2.0,
    )


def test_exact_attention_inventory_and_selection_contract():
    model = ToyAttentionModel()
    eligible = resolve_eligible_gpu_lora_targets(model, require_gpu=False)
    assert eligible.expected_count == 9
    assert len(eligible.records) == 9
    assert all(record.weight_device == "cpu" for record in eligible.records)
    selected_names = [record.name for record in eligible.records if record.family in ("q_proj", "v_proj")]
    selected = select_target_manifest(eligible, selected_names)
    assert selected.families == ("q_proj", "v_proj")
    assert TargetManifest.from_dict(selected.to_dict(), expected_kind="selected") == selected

    with pytest.raises(ValueError, match="outside the eligible"):
        select_target_manifest(eligible, ["layers.0.mlp.q_proj"])
    with pytest.raises(ValueError, match="not on an accelerator"):
        resolve_eligible_gpu_lora_targets(model, require_gpu=True)

    staged = resolve_eligible_gpu_lora_targets(
        model,
        require_gpu=True,
        fsdp_staging_device="cuda",
    )
    assert all(record.weight_device == "cuda" for record in staged.records)
    with pytest.raises(ValueError, match="must name an accelerator"):
        resolve_eligible_gpu_lora_targets(
            model,
            require_gpu=True,
            fsdp_staging_device="cpu",
        )


def test_attention_resolver_rejects_fused_wrapper_duplicates():
    model = ToyAttentionModel()
    model.layers[0].linear_attn.generate_linear = nn.Module()
    model.layers[0].linear_attn.generate_linear.in_proj_a = nn.Linear(4, 4, bias=False)
    with pytest.raises(ValueError, match="Forbidden module path"):
        resolve_eligible_gpu_lora_targets(model, require_gpu=False)


def test_kt_fused_expert_lora_contract_is_independent_and_rank_locked():
    model = nn.Module()
    model._kt_wrappers = [
        SimpleNamespace(layer_idx=index, _fused_experts=True, _lora_rank=8, lora_experts=None) for index in range(2)
    ]
    assert verify_kt_fused_expert_lora_contract(model, expected_rank=8) == 2
    model._kt_wrappers[1]._lora_rank = 16
    with pytest.raises(ValueError, match="rank mismatch"):
        verify_kt_fused_expert_lora_contract(model, expected_rank=8)
    model._kt_wrappers[1]._lora_rank = 8
    model._kt_wrappers[1]._fused_experts = False
    with pytest.raises(ValueError, match="not fused"):
        verify_kt_fused_expert_lora_contract(model, expected_rank=8)


def test_injected_pair_resolver_rejects_any_extra_peft_target():
    model = nn.Module()
    model.layers = nn.ModuleList([nn.Module(), nn.Module()])
    model.layers[1].self_attn = nn.Module()
    model.layers[1].self_attn.q_proj = FakePeftLinear(2, 2)
    model.layers[1].self_attn.v_proj = FakePeftLinear(2, 2)
    eligible = resolve_eligible_gpu_lora_targets(ToyAttentionModel(), require_gpu=False)
    q_name = next(record.name for record in eligible.records if record.family == "q_proj")
    selected = select_target_manifest(eligible, [q_name])
    with pytest.raises(ValueError, match="unexpected"):
        resolve_injected_lora_pairs(model, selected)


def test_altlora_b_first_reference_and_baseline_parameter_ownership():
    _, pair = make_pair()
    with torch.no_grad():
        pair.lora_a.copy_(torch.eye(2))
        pair.lora_b.zero_()
    baseline_parameter = nn.Parameter(torch.tensor([1.0]))
    baseline = torch.optim.AdamW([pair.lora_a, pair.lora_b, baseline_parameter], lr=0.1, weight_decay=0.0)
    optimizer = AltLoraAttnOptimizer(baseline, (pair,), regularizer=1.0e-5, beta1=0.9, switch_every=1)
    pair.lora_a.grad = torch.ones_like(pair.lora_a)
    pair.lora_b.grad = torch.ones_like(pair.lora_b)
    baseline_parameter.grad = torch.ones_like(baseline_parameter)
    optimizer.step()

    raw_b_direction = torch.ones_like(pair.lora_b) / ((1.0 + 1.0e-5) * pair.scaling**2)
    expected_b = -0.1 * (1.0 - 0.9) * raw_b_direction
    torch.testing.assert_close(pair.lora_a, torch.eye(2), rtol=0, atol=0)
    torch.testing.assert_close(pair.lora_b, expected_b)
    assert baseline_parameter.item() < 1.0
    assert optimizer._active_factor() == "A"


def test_altlora_optimizer_resume_preserves_parity_and_momentum():
    def assign_gradients(pair, baseline_parameter, step):
        pair.lora_a.grad = torch.full_like(pair.lora_a, 0.1 + step)
        pair.lora_b.grad = torch.full_like(pair.lora_b, -0.2 - step)
        baseline_parameter.grad = torch.full_like(baseline_parameter, 0.3 + step)

    _, continuous_pair = make_pair()
    continuous_parameter = nn.Parameter(torch.tensor([1.0]))
    continuous_baseline = torch.optim.AdamW(
        [continuous_pair.lora_a, continuous_pair.lora_b, continuous_parameter], lr=0.01, weight_decay=0.01
    )
    continuous = AltLoraAttnOptimizer(
        continuous_baseline, (continuous_pair,), regularizer=1.0e-5, beta1=0.9, switch_every=1
    )
    for step in range(3):
        assign_gradients(continuous_pair, continuous_parameter, step)
        continuous.step()
    # Match a real save/load boundary: torch optimizer state_dict tensors otherwise share
    # storage with the live optimizer and are mutated by the continuous reference run.
    checkpoint = copy.deepcopy(continuous.state_dict())

    _, resumed_pair = make_pair()
    resumed_parameter = nn.Parameter(continuous_parameter.detach().clone())
    with torch.no_grad():
        resumed_pair.lora_a.copy_(continuous_pair.lora_a)
        resumed_pair.lora_b.copy_(continuous_pair.lora_b)
    resumed_baseline = torch.optim.AdamW(
        [resumed_pair.lora_a, resumed_pair.lora_b, resumed_parameter], lr=0.01, weight_decay=0.01
    )
    resumed = AltLoraAttnOptimizer(resumed_baseline, (resumed_pair,), regularizer=1.0e-5, beta1=0.9, switch_every=1)
    resumed.load_state_dict(checkpoint)

    for step in range(3, 5):
        assign_gradients(continuous_pair, continuous_parameter, step)
        assign_gradients(resumed_pair, resumed_parameter, step)
        continuous.step()
        resumed.step()
    torch.testing.assert_close(resumed_pair.lora_a, continuous_pair.lora_a, rtol=0, atol=0)
    torch.testing.assert_close(resumed_pair.lora_b, continuous_pair.lora_b, rtol=0, atol=0)
    torch.testing.assert_close(resumed_parameter, continuous_parameter, rtol=0, atol=0)
    assert resumed.variant_step == continuous.variant_step == 5


def test_bilora_rank_slices_ascent_projection_and_baseline_parameter():
    _, pair = make_pair()
    with torch.no_grad():
        pair.lora_a.copy_(torch.eye(2))
        pair.lora_b.zero_()
    baseline_parameter = nn.Parameter(torch.tensor([1.0]))
    baseline = torch.optim.AdamW([pair.lora_a, pair.lora_b, baseline_parameter], lr=0.1, weight_decay=0.0)
    optimizer = BiLoraAttnOptimizer(
        baseline,
        (pair,),
        primary_rank=1,
        auxiliary_rank=1,
        rho=100.0,
        auxiliary_lr_ratio=1.0,
    )
    pair.lora_a.grad = torch.ones_like(pair.lora_a)
    pair.lora_b.grad = torch.ones_like(pair.lora_b)
    baseline_parameter.grad = torch.ones_like(baseline_parameter)
    optimizer.step()

    torch.testing.assert_close(pair.lora_a[0], torch.tensor([0.9, -0.1]), atol=1.0e-6, rtol=0)
    torch.testing.assert_close(pair.lora_b[:, 0], torch.tensor([-0.1, -0.1]), atol=1.0e-6, rtol=0)
    torch.testing.assert_close(pair.lora_a[1], torch.tensor([0.1, 1.1]), atol=1.0e-6, rtol=0)
    torch.testing.assert_close(pair.lora_b[:, 1], torch.tensor([0.1, 0.1]), atol=1.0e-6, rtol=0)
    assert baseline_parameter.item() < 1.0

    pair.lora_a.grad = torch.ones_like(pair.lora_a)
    pair.lora_b.grad = torch.ones_like(pair.lora_b)
    optimizer.rho = 1.0e-3
    optimizer.step()
    perturbation_norm = torch.linalg.matrix_norm(pair.lora_b[:, 1:] @ pair.lora_a[1:], ord="fro")
    torch.testing.assert_close(perturbation_norm, torch.tensor(optimizer.rho), atol=1.0e-6, rtol=0)

    checkpoint = optimizer.state_dict()
    _, resumed_pair = make_pair()
    resumed_parameter = nn.Parameter(baseline_parameter.detach().clone())
    with torch.no_grad():
        resumed_pair.lora_a.copy_(pair.lora_a)
        resumed_pair.lora_b.copy_(pair.lora_b)
    resumed_baseline = torch.optim.AdamW(
        [resumed_pair.lora_a, resumed_pair.lora_b, resumed_parameter], lr=0.1, weight_decay=0.0
    )
    resumed = BiLoraAttnOptimizer(
        resumed_baseline,
        (resumed_pair,),
        primary_rank=1,
        auxiliary_rank=1,
        rho=optimizer.rho,
        auxiliary_lr_ratio=1.0,
    )
    resumed.load_state_dict(checkpoint)
    assert resumed.variant_step == optimizer.variant_step
    assert all(torch.is_tensor(state["step"]) for state in resumed.state.values() if state)


def test_bilora_eval_hook_removes_only_auxiliary_branch():
    module, pair = make_pair()
    with torch.no_grad():
        pair.lora_a.copy_(torch.eye(2))
        pair.lora_b.copy_(torch.eye(2))
    register_bilora_eval_hooks((pair,), primary_rank=1)
    hidden_states = torch.tensor([[2.0, 3.0]])
    module.train()
    torch.testing.assert_close(module(hidden_states), hidden_states)
    module.eval()
    torch.testing.assert_close(module(hidden_states), torch.tensor([[2.0, 0.0]]))


def test_bilora_primary_export_slices_only_attention_lora(tmp_path):
    names = ("layers.0.self_attn.q_proj", "layers.0.self_attn.v_proj")
    config = KTLoraVariantConfig(
        variant="bilora_attn",
        eligible_sha256="a" * 64,
        selected_sha256="b" * 64,
        exact_target_names=names,
        lora_rank=16,
        lora_alpha=32,
        lora_dropout=0.1,
        method={"primary_rank": 8, "auxiliary_rank": 8, "rho": 0.05},
    )
    state_dict = {}
    for index, name in enumerate(names):
        prefix = f"base_model.model.{name}"
        state_dict[f"{prefix}.lora_A.weight"] = torch.arange(16 * 3).reshape(16, 3) + index
        state_dict[f"{prefix}.lora_B.weight"] = torch.arange(4 * 16).reshape(4, 16) + index
    expert_name = "base_model.model.layers.0.mlp.fast_le.modules_to_save.default.weight"
    state_dict[expert_name] = torch.arange(5)
    save_file(state_dict, tmp_path / "adapter_model.safetensors")
    fused_payload = b"unchanged-kt-fused-expert-lora"
    (tmp_path / "fused_expert_lora.safetensors").write_bytes(fused_payload)
    with open(tmp_path / "adapter_config.json", "w", encoding="utf-8") as stream:
        json.dump({"r": 16, "lora_alpha": 32, "rank_pattern": {}, "alpha_pattern": {}}, stream)

    output_path = export_bilora_primary_adapter(tmp_path, config)
    assert output_path.name == "adapter_primary"
    assert (output_path / "fused_expert_lora.safetensors").read_bytes() == fused_payload
    exported = load_file(output_path / "adapter_model.safetensors")
    for name, tensor in state_dict.items():
        if ".lora_A." in name:
            torch.testing.assert_close(exported[name], tensor[:8])
        elif ".lora_B." in name:
            torch.testing.assert_close(exported[name], tensor[:, :8])
        else:
            torch.testing.assert_close(exported[name], tensor)
    with open(output_path / "adapter_config.json", encoding="utf-8") as stream:
        adapter_config = json.load(stream)
    assert (adapter_config["r"], adapter_config["lora_alpha"]) == (8, 16)
    exported_variant = load_variant_config(output_path)
    assert exported_variant is not None
    assert exported_variant.method["deployment"] == "primary_only"
    assert (exported_variant.lora_rank, exported_variant.lora_alpha) == (8, 16)


def test_plop_nfn_matches_tokenwise_reference_and_preserves_random_norm():
    weight = torch.tensor([[2.0, 0.0], [0.0, 1.0]])
    hidden_states = torch.tensor([[3.0, 4.0], [4.0, 3.0]])
    gaussian = torch.tensor([[1.0, -2.0], [-2.0, 1.0]])
    random_hidden_states = gaussian * (hidden_states.norm(dim=-1, keepdim=True) / gaussian.norm(dim=-1, keepdim=True))
    torch.testing.assert_close(random_hidden_states.norm(dim=-1), hidden_states.norm(dim=-1))
    expected = (hidden_states @ weight.mT).norm(dim=-1) / (random_hidden_states @ weight.mT).norm(dim=-1)
    torch.testing.assert_close(compute_nfn_per_token(weight, hidden_states, random_hidden_states), expected)


def test_plop_probe_uses_no_grad_without_masking_forward_errors():
    model = ToyAttentionModel()
    eligible = resolve_eligible_gpu_lora_targets(model, require_gpu=False)
    scorer = PLoPAttnScorer(model, eligible)
    hidden_states = torch.randn(2, 3, 4)
    attention_mask = torch.ones(2, 3, dtype=torch.bool)
    try:
        with scorer.batch(0, attention_mask):
            assert not torch.is_grad_enabled()
            assert not torch.is_inference_mode_enabled()
            model(hidden_states)
        assert all(record["probe_examples"] == 2 for record in scorer.finalize()["module_scores"])

        with pytest.raises(RuntimeError, match="forward sentinel"):
            with scorer.batch(1, attention_mask):
                raise RuntimeError("forward sentinel")
    finally:
        scorer.close()


def test_plop_artifacts_select_and_reload_exact_lowest_families(tmp_path):
    eligible = resolve_eligible_gpu_lora_targets(ToyAttentionModel(), require_gpu=False)
    family_rank = {family: index + 1 for index, family in enumerate(sorted(eligible.families))}
    module_scores = [
        {
            "name": record.name,
            "family": record.family,
            "nfn": float(family_rank[record.family]),
            "probe_calls": 2,
            "probe_examples": 2,
            "valid_tokens": 8,
            "weight_shape": list(record.weight_shape),
            "weight_dtype": record.weight_dtype,
            "weight_device": record.weight_device,
        }
        for record in eligible.records
    ]
    family_scores = {
        family: {
            "nfn": float(family_rank[family]),
            "module_count": sum(record.family == family for record in eligible.records),
        }
        for family in eligible.families
    }
    score_core = {
        "schema_version": 1,
        "method": "plop_attn",
        "eligible_sha256": eligible.sha256,
        "seed": 20260803,
        "num_random_draws": 1,
        "epsilon": 1.0e-8,
        "module_scores": module_scores,
        "family_scores": family_scores,
        "probe_wall_seconds": 1.0,
    }
    scores = {**score_core, "sha256": canonical_json_sha256(score_core)}
    selected = write_plop_artifacts(
        tmp_path,
        eligible,
        scores,
        select_k=2,
        probe_manifest={"lora_rank": 8, "probe_examples": 2},
    )
    reloaded, families = load_plop_training_selection(
        eligible,
        tmp_path / PLOP_EXACT_TARGETS_NAME,
        select_k=2,
        score_manifest_path=tmp_path / PLOP_NFN_SCORES_NAME,
    )
    assert reloaded == selected
    assert families == tuple(sorted(family_rank, key=family_rank.get)[:2])


@pytest.mark.parametrize("variant", ["altlora_attn", "plop_attn", "bilora_attn"])
@pytest.mark.parametrize("use_lora_expert", [False, True])
def test_variant_argument_contract_accepts_standalone_and_lora_expert(variant, use_lora_expert):
    args = FinetuningArguments(
        kt_lora_variant=variant,
        lora_target="kt_attention",
        lora_rank=8,
        lora_alpha=16,
        lora_dropout=0.1,
        plop_attn_target_manifest_path="frozen-plop-targets.json",
        use_lora_expert=use_lora_expert,
    )
    assert args.kt_lora_variant == variant
    assert args.use_lora_expert is use_lora_expert


def test_variant_argument_contract_rejects_broad_or_mixed_paths():
    with pytest.raises(ValueError, match="kt_attention"):
        FinetuningArguments(kt_lora_variant="altlora_attn", lora_target="all", lora_dropout=0.1)
    with pytest.raises(ValueError, match="rank=8"):
        FinetuningArguments(
            kt_lora_variant="bilora_attn",
            lora_target="kt_attention",
            lora_rank=4,
            lora_alpha=8,
            lora_dropout=0.1,
        )
    with pytest.raises(ValueError, match="cannot be combined"):
        FinetuningArguments(
            kt_lora_variant="altlora_attn",
            lora_target="kt_attention",
            lora_rank=8,
            lora_alpha=16,
            lora_dropout=0.1,
            use_dora=True,
        )
