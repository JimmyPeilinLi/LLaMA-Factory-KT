"""
Router 控制模块 (从 lora-without-regret 迁移)

功能:
- 控制 Router (gate) 参数的冻结/解冻
- 支持 MoE 模型的路由器冻结

使用方式:
    from llamafactory.model.router_control import RouterController

    # 冻结 Router
    RouterController.set_router_trainable(model, trainable=False)
"""

import os
from typing import Dict, List

import torch
import torch.nn as nn


# Qwen3 MoE 的 Router 参数命名模式
ROUTER_PARAM_PATTERNS = [
    ".mlp.gate.",      # 主要模式: model.layers.X.mlp.gate.weight
    ".block_sparse_moe.gate.",  # 备用模式 (某些模型)
]


class RouterController:
    """
    Router 训练策略控制器

    支持三种策略:
    - R1: 完全冻结 (frozen)
    - R2: 全程可训练 (trainable)
    - R3: 两阶段 (two_stage)
    """

    @staticmethod
    def is_router_param(name: str) -> bool:
        """判断参数是否属于 Router"""
        return any(pattern in name for pattern in ROUTER_PARAM_PATTERNS)

    @staticmethod
    def set_router_trainable(model: nn.Module, trainable: bool = False) -> int:
        """
        设置所有 Router 参数的 requires_grad

        Args:
            model: 模型
            trainable: 是否可训练

        Returns:
            修改的参数数量
        """
        count = 0
        for name, param in model.named_parameters():
            if RouterController.is_router_param(name):
                param.requires_grad = trainable
                count += 1

        status = "可训练" if trainable else "冻结"
        local_rank = int(os.environ.get("LOCAL_RANK", -1))
        if local_rank in [-1, 0]:
            print(f"[RouterController] {count} 个 Router 参数已设置为{status}")
        return count

    @staticmethod
    def get_router_params(model: nn.Module) -> List[nn.Parameter]:
        """获取所有 Router 参数"""
        return [
            param for name, param in model.named_parameters()
            if RouterController.is_router_param(name)
        ]

    @staticmethod
    def get_router_param_names(model: nn.Module) -> List[str]:
        """获取所有 Router 参数名"""
        return [
            name for name, _ in model.named_parameters()
            if RouterController.is_router_param(name)
        ]

    @staticmethod
    def count_router_params(model: nn.Module) -> Dict[str, int]:
        """统计 Router 参数数量"""
        total = 0
        trainable = 0
        for name, param in model.named_parameters():
            if RouterController.is_router_param(name):
                total += param.numel()
                if param.requires_grad:
                    trainable += param.numel()

        return {
            "total": total,
            "trainable": trainable,
            "frozen": total - trainable,
        }

    @staticmethod
    def print_router_status(model: nn.Module):
        """打印 Router 参数状态"""
        stats = RouterController.count_router_params(model)
        print("\n" + "=" * 50)
        print("Router 参数状态")
        print("=" * 50)
        print(f"总参数量: {stats['total']:,}")
        print(f"可训练: {stats['trainable']:,}")
        print(f"冻结: {stats['frozen']:,}")
        print("=" * 50 + "\n")
