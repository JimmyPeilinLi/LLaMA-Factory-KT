"""
异构 LoRA 注入模块 (从 lora-without-regret 迁移)

功能:
- 支持对不同模块使用不同的 LoRA rank
- 支持 hot/cold experts 差异化 LoRA 配置
- 手动注入 LoRA 层，绕过 PEFT 的统一 rank 限制

使用方式:
    from llamafactory.model.heterogeneous_lora import HeterogeneousLoRAInjector, create_lora_config_for_strategy

    config = create_lora_config_for_strategy('all_small_rank')
    injector = HeterogeneousLoRAInjector()
    model = injector.inject(model, config)
"""

import json
import math
import os
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LoRAConfig:
    """LoRA 配置"""
    rank: int = 16
    alpha: int = 32
    dropout: float = 0.05

    @property
    def scaling(self) -> float:
        return self.alpha / self.rank if self.rank > 0 else 0


@dataclass
class SharedLoRAConfig(LoRAConfig):
    """Shared 层 (Attention) 的 LoRA 配置"""
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj"
    ])


@dataclass
class ExpertLoRAConfig(LoRAConfig):
    """Expert 层的 LoRA 配置"""
    indices: Union[List[Tuple[int, int]], str] = "per_layer"
    hot_experts_file: Optional[str] = None
    num_hot_per_layer: Optional[int] = None
    hot_ratio: float = 0.25


@dataclass
class HeterogeneousLoRAConfig:
    """异构 LoRA 完整配置"""
    enabled: bool = True
    shared_config: SharedLoRAConfig = field(default_factory=SharedLoRAConfig)
    hot_experts_config: ExpertLoRAConfig = field(default_factory=lambda: ExpertLoRAConfig(rank=64, alpha=128))
    cold_experts_config: ExpertLoRAConfig = field(default_factory=lambda: ExpertLoRAConfig(rank=0))
    expert_target_modules: List[str] = field(default_factory=lambda: [
        "gate_proj", "up_proj", "down_proj"
    ])


class LoRALinear(nn.Module):
    """
    LoRA 线性层

    将原始线性层包装为 LoRA 形式:
    output = original(x) + (x @ A.T @ B.T) * scaling
    """

    def __init__(
        self,
        original_layer: nn.Linear,
        rank: int = 16,
        alpha: int = 32,
        dropout: float = 0.05,
    ):
        super().__init__()

        self.original_layer = original_layer
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank if rank > 0 else 0

        in_features = original_layer.in_features
        out_features = original_layer.out_features

        # LoRA 层
        if rank > 0:
            self.lora_A = nn.Linear(in_features, rank, bias=False)
            self.lora_B = nn.Linear(rank, out_features, bias=False)
            self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

            # 初始化
            nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B.weight)
        else:
            self.lora_A = None
            self.lora_B = None
            self.lora_dropout = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 原始层输出
        result = self.original_layer(x)

        # LoRA 增量
        if self.rank > 0 and self.lora_A is not None:
            lora_output = self.lora_B(self.lora_A(self.lora_dropout(x)))
            result = result + lora_output * self.scaling

        return result

    def merge_weights(self) -> nn.Linear:
        """将 LoRA 权重合并到原始层"""
        if self.rank == 0 or self.lora_A is None:
            return self.original_layer

        merged_weight = (
            self.original_layer.weight.data +
            (self.lora_B.weight.data @ self.lora_A.weight.data) * self.scaling
        )

        merged_layer = nn.Linear(
            self.original_layer.in_features,
            self.original_layer.out_features,
            bias=self.original_layer.bias is not None,
        )
        merged_layer.weight.data = merged_weight
        if self.original_layer.bias is not None:
            merged_layer.bias.data = self.original_layer.bias.data.clone()

        return merged_layer

    def get_lora_params(self) -> List[nn.Parameter]:
        """获取 LoRA 参数"""
        if self.rank == 0:
            return []
        return [self.lora_A.weight, self.lora_B.weight]

    def __repr__(self):
        return (
            f"LoRALinear(in={self.original_layer.in_features}, "
            f"out={self.original_layer.out_features}, "
            f"rank={self.rank}, alpha={self.alpha})"
        )


class HeterogeneousLoRAInjector:
    """
    异构 LoRA 注入器

    支持对不同模块使用不同的 LoRA rank:
    - Shared 层 (Attention): 统一的 rank
    - Hot experts: 大 rank
    - Cold experts: 小 rank 或不加 LoRA
    """

    def __init__(self):
        self.injected_modules: Dict[str, LoRALinear] = {}
        self.injection_stats = {
            "shared": {"count": 0, "rank": 0},
            "hot_experts": {"count": 0, "rank": 0},
            "cold_experts": {"count": 0, "rank": 0},
        }
        local_rank = int(os.environ.get("LOCAL_RANK", -1))
        self._is_main_process = local_rank in [-1, 0]

    def _log(self, message: str):
        """只在主进程打印日志"""
        if self._is_main_process:
            print(message)

    def inject(
        self,
        model: nn.Module,
        config: Union[HeterogeneousLoRAConfig, Dict],
        hot_experts_list: Optional[List[Tuple[int, int]]] = None,
        hot_experts_per_layer: Optional[Dict[int, List[int]]] = None,
    ) -> nn.Module:
        """
        注入异构 LoRA

        Args:
            model: 原始模型
            config: LoRA 配置
            hot_experts_list: 热门专家列表 [(layer_idx, expert_idx), ...] (旧格式)
            hot_experts_per_layer: 每层热门专家 {layer_idx: [expert_idx, ...]} (新格式)

        Returns:
            注入 LoRA 后的模型
        """
        if isinstance(config, dict):
            config = self._parse_config(config)

        if not config.enabled:
            self._log("[HeterogeneousLoRAInjector] 异构 LoRA 未启用，跳过注入")
            return model

        use_per_layer_mode = (
            hot_experts_per_layer is not None or
            config.hot_experts_config.indices == "per_layer"
        )

        if use_per_layer_mode:
            if hot_experts_per_layer is None:
                hot_experts_per_layer = self._load_hot_experts_per_layer(config.hot_experts_config)

            hot_config = config.hot_experts_config
            if hot_config.num_hot_per_layer is not None:
                num_hot_per_layer = hot_config.num_hot_per_layer
            else:
                model_config = getattr(model, 'config', None)
                if model_config is None and hasattr(model, 'model'):
                    model_config = getattr(model.model, 'config', None)
                num_experts = getattr(model_config, 'num_experts', 128) if model_config else 128
                num_hot_per_layer = int(num_experts * hot_config.hot_ratio)
                self._log(f"  从 hot_ratio={hot_config.hot_ratio} 计算 num_hot_per_layer={num_hot_per_layer}")

            total_hot = sum(len(experts) for experts in hot_experts_per_layer.values())

            self._log(f"\n[HeterogeneousLoRAInjector] 开始注入异构 LoRA (每层模式)")
            self._log(f"  每层 Hot experts 数量: {num_hot_per_layer}")
            self._log(f"  总 Hot experts 数量: {total_hot}")

            self._inject_shared_lora(model, config.shared_config)

            self._inject_expert_lora_per_layer(
                model,
                hot_experts_per_layer,
                config.hot_experts_config,
                config.cold_experts_config,
                config.expert_target_modules,
            )
        else:
            if hot_experts_list is None:
                hot_experts_list = self._load_hot_experts(config.hot_experts_config)

            hot_experts_set = set(hot_experts_list) if hot_experts_list else set()

            self._log(f"\n[HeterogeneousLoRAInjector] 开始注入异构 LoRA (全局模式)")
            self._log(f"  Hot experts 数量: {len(hot_experts_set)}")

            self._inject_shared_lora(model, config.shared_config)

            self._inject_expert_lora(
                model,
                hot_experts_set,
                config.hot_experts_config,
                config.cold_experts_config,
                config.expert_target_modules,
            )

        self._freeze_non_lora_params(model)
        self._print_injection_stats()
        return model

    def _freeze_non_lora_params(self, model: nn.Module):
        """冻结非 LoRA 参数"""
        frozen_count = 0
        for name, param in model.named_parameters():
            if ('lora_A' not in name and
                'lora_B' not in name and
                'mini_shared' not in name):
                param.requires_grad = False
                frozen_count += 1

        trainable_count = sum(1 for n, p in model.named_parameters() if p.requires_grad)
        self._log(f"  [Freeze] 冻结 {frozen_count} 个基础模型参数, 保留 {trainable_count} 个参数可训练 (LoRA + Mini-Shared)")

    def _parse_config(self, config_dict: Dict) -> HeterogeneousLoRAConfig:
        """从字典解析配置"""
        attention_cfg = config_dict.get("attention_config", {})
        if attention_cfg:
            shared_cfg = {
                "rank": attention_cfg.get("rank", 16),
                "alpha": attention_cfg.get("alpha", 32),
                "dropout": attention_cfg.get("dropout", 0.05),
                "target_modules": attention_cfg.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]),
            }
        else:
            shared_cfg = config_dict.get("shared_config", {})

        expert_cfg = config_dict.get("expert_config", {})
        expert_enabled = expert_cfg.get("enabled", True)

        if not expert_enabled:
            hot_cfg = {"rank": 0}
            cold_cfg = {"rank": 0}
        else:
            hot_cfg = expert_cfg.get("hot_experts", {})
            cold_cfg = expert_cfg.get("cold_experts", {})

        return HeterogeneousLoRAConfig(
            enabled=config_dict.get("heterogeneous", True),
            shared_config=SharedLoRAConfig(
                rank=shared_cfg.get("rank", 16),
                alpha=shared_cfg.get("alpha", 32),
                dropout=shared_cfg.get("dropout", 0.05),
                target_modules=shared_cfg.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]),
            ),
            hot_experts_config=ExpertLoRAConfig(
                rank=hot_cfg.get("rank", 32),
                alpha=hot_cfg.get("alpha", 64),
                dropout=hot_cfg.get("dropout", 0.05),
                indices=hot_cfg.get("indices", "per_layer"),
                hot_experts_file=hot_cfg.get("hot_experts_file"),
                num_hot_per_layer=hot_cfg.get("num_hot_per_layer"),
                hot_ratio=hot_cfg.get("hot_ratio", 0.25),
            ),
            cold_experts_config=ExpertLoRAConfig(
                rank=cold_cfg.get("rank", 0),
                alpha=cold_cfg.get("alpha", 0),
                dropout=cold_cfg.get("dropout", 0.0),
            ),
            expert_target_modules=config_dict.get("expert_target_modules", ["gate_proj", "up_proj", "down_proj"]),
        )

    def _load_hot_experts(self, config: ExpertLoRAConfig) -> List[Tuple[int, int]]:
        """加载热门专家列表 (旧格式)"""
        if isinstance(config.indices, list):
            return config.indices

        if config.hot_experts_file and os.path.exists(config.hot_experts_file):
            with open(config.hot_experts_file, "r") as f:
                data = json.load(f)
            return [tuple(item) for item in data.get("hot_experts_indices", [])]

        self._log("[HeterogeneousLoRAInjector] 警告: 未找到热门专家列表")
        return []

    def _load_hot_experts_per_layer(self, config: ExpertLoRAConfig) -> Dict[int, List[int]]:
        """加载每层热门专家列表 (新格式)"""
        if config.hot_experts_file and os.path.exists(config.hot_experts_file):
            with open(config.hot_experts_file, "r") as f:
                data = json.load(f)

            if "hot_experts_per_layer" in data:
                return {
                    int(layer_idx): expert_indices
                    for layer_idx, expert_indices in data["hot_experts_per_layer"].items()
                }
            else:
                self._log("[HeterogeneousLoRAInjector] 警告: 文件不含 'hot_experts_per_layer'")

        self._log("[HeterogeneousLoRAInjector] 警告: 未找到每层热门专家列表，将使用默认配置")
        return {}

    def _inject_shared_lora(self, model: nn.Module, config: SharedLoRAConfig):
        """注入 Shared 层 (Attention) LoRA"""
        if config.rank == 0:
            return

        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            layers = model.model.layers
        elif hasattr(model, 'layers'):
            layers = model.layers
        else:
            self._log("[HeterogeneousLoRAInjector] 警告: 无法找到模型层")
            return

        count = 0
        for layer_idx, layer in enumerate(layers):
            if hasattr(layer, 'self_attn'):
                attn = layer.self_attn
                for module_name in config.target_modules:
                    if hasattr(attn, module_name):
                        original = getattr(attn, module_name)
                        if isinstance(original, nn.Linear):
                            lora_layer = LoRALinear(
                                original,
                                rank=config.rank,
                                alpha=config.alpha,
                                dropout=config.dropout,
                            )
                            setattr(attn, module_name, lora_layer)
                            self.injected_modules[f"layer.{layer_idx}.attn.{module_name}"] = lora_layer
                            count += 1

        self.injection_stats["shared"]["count"] = count
        self.injection_stats["shared"]["rank"] = config.rank
        self._log(f"  [Shared] 注入 {count} 个 LoRA 层, rank={config.rank}")

    def _inject_expert_lora(
        self,
        model: nn.Module,
        hot_experts_set: set,
        hot_config: ExpertLoRAConfig,
        cold_config: ExpertLoRAConfig,
        target_modules: List[str],
    ):
        """注入 Expert 层 LoRA"""
        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            layers = model.model.layers
        elif hasattr(model, 'layers'):
            layers = model.layers
        else:
            return

        hot_count = 0
        cold_count = 0

        for layer_idx, layer in enumerate(layers):
            if not hasattr(layer, 'mlp') or not hasattr(layer.mlp, 'experts'):
                continue

            experts = layer.mlp.experts
            num_experts = len(experts)

            for expert_idx in range(num_experts):
                is_hot = (layer_idx, expert_idx) in hot_experts_set

                if is_hot:
                    config = hot_config
                else:
                    config = cold_config

                if config.rank == 0:
                    continue

                expert = experts[expert_idx]
                for module_name in target_modules:
                    if hasattr(expert, module_name):
                        original = getattr(expert, module_name)
                        if isinstance(original, nn.Linear):
                            lora_layer = LoRALinear(
                                original,
                                rank=config.rank,
                                alpha=config.alpha,
                                dropout=config.dropout,
                            )
                            setattr(expert, module_name, lora_layer)
                            key = f"layer.{layer_idx}.expert.{expert_idx}.{module_name}"
                            self.injected_modules[key] = lora_layer

                            if is_hot:
                                hot_count += 1
                            else:
                                cold_count += 1

        self.injection_stats["hot_experts"]["count"] = hot_count
        self.injection_stats["hot_experts"]["rank"] = hot_config.rank
        self.injection_stats["cold_experts"]["count"] = cold_count
        self.injection_stats["cold_experts"]["rank"] = cold_config.rank

        if hot_count > 0:
            self._log(f"  [Hot Experts] 注入 {hot_count} 个 LoRA 层, rank={hot_config.rank}")
        if cold_count > 0:
            self._log(f"  [Cold Experts] 注入 {cold_count} 个 LoRA 层, rank={cold_config.rank}")

    def _inject_expert_lora_per_layer(
        self,
        model: nn.Module,
        hot_experts_per_layer: Dict[int, List[int]],
        hot_config: ExpertLoRAConfig,
        cold_config: ExpertLoRAConfig,
        target_modules: List[str],
    ):
        """注入 Expert 层 LoRA (每层模式)"""
        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            layers = model.model.layers
        elif hasattr(model, 'layers'):
            layers = model.layers
        else:
            return

        hot_count = 0
        cold_count = 0

        num_experts_in_model = 128
        for layer in layers:
            if hasattr(layer, 'mlp') and hasattr(layer.mlp, 'experts'):
                num_experts_in_model = len(layer.mlp.experts)
                break

        if hot_config.num_hot_per_layer is not None:
            num_hot_per_layer = hot_config.num_hot_per_layer
        else:
            num_hot_per_layer = int(num_experts_in_model * hot_config.hot_ratio)

        for layer_idx, layer in enumerate(layers):
            if not hasattr(layer, 'mlp') or not hasattr(layer.mlp, 'experts'):
                continue

            experts = layer.mlp.experts
            num_experts = len(experts)

            if layer_idx in hot_experts_per_layer:
                hot_expert_indices = set(hot_experts_per_layer[layer_idx])
            else:
                hot_expert_indices = set(range(min(num_hot_per_layer, num_experts)))

            for expert_idx in range(num_experts):
                is_hot = expert_idx in hot_expert_indices

                if is_hot:
                    config = hot_config
                else:
                    config = cold_config

                if config.rank == 0:
                    continue

                expert = experts[expert_idx]
                for module_name in target_modules:
                    if hasattr(expert, module_name):
                        original = getattr(expert, module_name)
                        if isinstance(original, nn.Linear):
                            lora_layer = LoRALinear(
                                original,
                                rank=config.rank,
                                alpha=config.alpha,
                                dropout=config.dropout,
                            )
                            setattr(expert, module_name, lora_layer)
                            key = f"layer.{layer_idx}.expert.{expert_idx}.{module_name}"
                            self.injected_modules[key] = lora_layer

                            if is_hot:
                                hot_count += 1
                            else:
                                cold_count += 1

        self.injection_stats["hot_experts"]["count"] = hot_count
        self.injection_stats["hot_experts"]["rank"] = hot_config.rank
        self.injection_stats["cold_experts"]["count"] = cold_count
        self.injection_stats["cold_experts"]["rank"] = cold_config.rank

        if hot_count > 0:
            num_layers = len([l for l in layers if hasattr(l, 'mlp') and hasattr(l.mlp, 'experts')])
            hot_per_layer = hot_count // (3 * num_layers) if num_layers > 0 else 0
            self._log(f"  [Hot Experts] 注入 {hot_count} 个 LoRA 层 (每层约 {hot_per_layer} 个专家), rank={hot_config.rank}")
        if cold_count > 0:
            num_layers = len([l for l in layers if hasattr(l, 'mlp') and hasattr(l.mlp, 'experts')])
            cold_per_layer = cold_count // (3 * num_layers) if num_layers > 0 else 0
            self._log(f"  [Cold Experts] 注入 {cold_count} 个 LoRA 层 (每层约 {cold_per_layer} 个专家), rank={cold_config.rank}")

    def _print_injection_stats(self):
        """打印注入统计"""
        total = sum(s["count"] for s in self.injection_stats.values())
        self._log(f"\n[HeterogeneousLoRAInjector] 注入完成，共 {total} 个 LoRA 层")
        self._log(f"  Shared (Attention): {self.injection_stats['shared']['count']} 层, rank={self.injection_stats['shared']['rank']}")
        self._log(f"  Hot Experts: {self.injection_stats['hot_experts']['count']} 层, rank={self.injection_stats['hot_experts']['rank']}")
        self._log(f"  Cold Experts: {self.injection_stats['cold_experts']['count']} 层, rank={self.injection_stats['cold_experts']['rank']}")

    def get_lora_params(self) -> List[nn.Parameter]:
        """获取所有 LoRA 参数"""
        params = []
        for lora_layer in self.injected_modules.values():
            params.extend(lora_layer.get_lora_params())
        return params

    def count_lora_params(self) -> Dict[str, int]:
        """统计 LoRA 参数数量"""
        total = 0
        trainable = 0
        for lora_layer in self.injected_modules.values():
            for param in lora_layer.get_lora_params():
                total += param.numel()
                if param.requires_grad:
                    trainable += param.numel()

        return {
            "total": total,
            "trainable": trainable,
            "total_mb": total * 4 / (1024 * 1024),
        }

    def save_lora_weights(self, output_dir: str, zero3_enabled: bool = False):
        """保存 LoRA 权重"""
        os.makedirs(output_dir, exist_ok=True)

        state_dict = {}
        for name, lora_layer in self.injected_modules.items():
            if lora_layer.rank > 0:
                if zero3_enabled:
                    try:
                        import deepspeed
                        params = [lora_layer.lora_A.weight, lora_layer.lora_B.weight]
                        with deepspeed.zero.GatheredParameters(params, modifier_rank=0):
                            if self._is_main_process:
                                state_dict[f"{name}.lora_A"] = lora_layer.lora_A.weight.data.cpu().clone()
                                state_dict[f"{name}.lora_B"] = lora_layer.lora_B.weight.data.cpu().clone()
                    except Exception as e:
                        self._log(f"  [WARNING] 无法收集 LoRA 参数 {name}: {e}")
                else:
                    state_dict[f"{name}.lora_A"] = lora_layer.lora_A.weight.data.cpu().clone()
                    state_dict[f"{name}.lora_B"] = lora_layer.lora_B.weight.data.cpu().clone()

        if self._is_main_process:
            torch.save(state_dict, os.path.join(output_dir, "lora_weights.pt"))

            config = {
                "injection_stats": self.injection_stats,
                "module_names": list(self.injected_modules.keys()),
            }
            with open(os.path.join(output_dir, "lora_config.json"), "w") as f:
                json.dump(config, f, indent=2)

            self._log(f"[HeterogeneousLoRAInjector] LoRA 权重已保存到: {output_dir}")

    def load_lora_weights(self, input_dir: str):
        """加载 LoRA 权重"""
        state_dict = torch.load(os.path.join(input_dir, "lora_weights.pt"))

        for name, lora_layer in self.injected_modules.items():
            if lora_layer.rank > 0:
                if f"{name}.lora_A" in state_dict:
                    lora_layer.lora_A.weight.data = state_dict[f"{name}.lora_A"]
                if f"{name}.lora_B" in state_dict:
                    lora_layer.lora_B.weight.data = state_dict[f"{name}.lora_B"]

        self._log(f"[HeterogeneousLoRAInjector] LoRA 权重已从 {input_dir} 加载")


def create_lora_config_for_strategy(strategy: str) -> Dict:
    """
    为不同实验策略创建 LoRA 配置

    策略:
        - "hot_large_rank": 方案A - 每层 25% hot experts, rank=32; cold 无 LoRA
        - "all_small_rank": 方案B - 每层全部 experts, rank=8
        - "hot_small_rank": 方案C - 每层 25% hot experts, rank=8

    Args:
        strategy: 策略名称

    Returns:
        配置字典
    """
    base_config = {
        "heterogeneous": True,
        "shared_config": {
            "rank": 16,
            "alpha": 32,
            "dropout": 0.05,
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        },
        "expert_target_modules": ["gate_proj", "up_proj", "down_proj"],
    }

    if strategy == "hot_large_rank":
        base_config["expert_config"] = {
            "hot_experts": {
                "rank": 32,
                "alpha": 64,
                "dropout": 0.05,
                "indices": "per_layer",
                "hot_ratio": 0.25,
            },
            "cold_experts": {
                "rank": 0,
            },
        }

    elif strategy == "all_small_rank":
        base_config["expert_config"] = {
            "hot_experts": {
                "rank": 8,
                "alpha": 16,
                "dropout": 0.05,
                "indices": "per_layer",
                "hot_ratio": 1.0,
            },
            "cold_experts": {
                "rank": 0,
            },
        }

    elif strategy == "hot_small_rank":
        base_config["expert_config"] = {
            "hot_experts": {
                "rank": 8,
                "alpha": 16,
                "dropout": 0.05,
                "indices": "per_layer",
                "hot_ratio": 0.25,
            },
            "cold_experts": {
                "rank": 0,
            },
        }

    else:
        raise ValueError(f"未知策略: {strategy}")

    return base_config
