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
模型层输出调试模块

用于保存模型每一层（包含子层）的 forward 和 backward 输出，便于比对和调试。

使用示例:
    from llamafactory.extras.layer_debug import LayerDebugger

    # 创建调试器
    debugger = LayerDebugger(output_dir="./layer_outputs")

    # 注册 hooks
    debugger.register_hooks(model)

    # 执行前向传播
    output = model(input_ids)
    loss = criterion(output, labels)

    # 执行反向传播（会自动捕获梯度）
    loss.backward()

    # 保存结果
    debugger.save("run1")

    # 清理
    debugger.remove_hooks()
"""

import os
import re
from collections import OrderedDict
from datetime import datetime
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Set, Tuple, Union

import torch
import torch.nn as nn


# ANSI 颜色码
class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"

    # 前景色
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"

    # 亮色
    BRIGHT_RED = "\033[91m"
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_BLUE = "\033[94m"
    BRIGHT_MAGENTA = "\033[95m"
    BRIGHT_CYAN = "\033[96m"


# 全局调用链记录
_call_chain: List[str] = []
_call_chain_max_len = 10


def _push_call_chain(layer_name: str):
    """添加模块到调用链"""
    global _call_chain
    _call_chain.append(layer_name)
    if len(_call_chain) > _call_chain_max_len:
        _call_chain = _call_chain[-_call_chain_max_len:]


def _get_call_chain(current_layer: str, num_recent: int = 5) -> str:
    """获取最近的调用链（显示模块 name）"""
    global _call_chain
    if not _call_chain:
        return ""

    # 获取最近的 n 个（不包括当前层）
    recent = [name for name in _call_chain[-num_recent:] if name != current_layer]
    if not recent:
        return ""

    return " -> ".join(recent)


if TYPE_CHECKING:
    from transformers import PreTrainedModel


class LayerDebugger:
    """模型层输出调试器，用于保存 forward 和 backward 的输出"""

    def __init__(
        self,
        output_dir: str = "./layer_outputs",
        save_forward: bool = True,
        save_backward: bool = True,
        include_patterns: Optional[List[str]] = None,
        exclude_patterns: Optional[List[str]] = None,
        max_tensor_size: Optional[int] = None,
        save_input: bool = False,
        verbose: bool = False,
    ):
        """
        初始化层调试器

        Args:
            output_dir: 输出目录
            save_forward: 是否保存前向传播输出
            save_backward: 是否保存反向传播梯度
            include_patterns: 要包含的层名正则表达式列表（为空则包含所有）
            exclude_patterns: 要排除的层名正则表达式列表
            max_tensor_size: 最大张量大小（元素数），超过则不保存
            save_input: 是否同时保存输入
            verbose: 是否输出详细日志（包括调用链）
        """
        self.output_dir = output_dir
        self.save_forward = save_forward
        self.save_backward = save_backward
        self.include_patterns = include_patterns or []
        self.exclude_patterns = exclude_patterns or []
        self.max_tensor_size = max_tensor_size
        self.save_input = save_input
        self.verbose = verbose

        # 存储数据
        self.forward_outputs: Dict[str, Any] = OrderedDict()
        self.forward_inputs: Dict[str, Any] = OrderedDict()
        self.backward_grad_inputs: Dict[str, Any] = OrderedDict()
        self.backward_grad_outputs: Dict[str, Any] = OrderedDict()

        # Hook 句柄
        self._forward_handles: List = []
        self._backward_handles: List = []

        # 状态
        self._enabled = True
        self._step_count = 0

        # 模型引用（用于保存结构）
        self._model: Optional["PreTrainedModel"] = None

        os.makedirs(output_dir, exist_ok=True)

    def _get_model_structure(self) -> Optional[Dict[str, Any]]:
        """获取模型结构信息"""
        if self._model is None:
            return None

        structure = {
            "model_repr": str(self._model),  # 模型的字符串表示
            "modules": OrderedDict(),  # 所有模块信息
        }

        # 尝试获取模型配置
        if hasattr(self._model, "config"):
            try:
                structure["config"] = self._model.config.to_dict()
            except Exception as e:
                structure["config"] = {"error": str(e)}

        # 获取所有模块及其类型
        for name, module in self._model.named_modules():
            if not name:
                name = "root"
            module_info = {
                "class": module.__class__.__name__,
                "module_path": module.__class__.__module__,
            }
            # 获取模块的参数信息
            param_count = sum(p.numel() for p in module.parameters(recurse=False))
            if param_count > 0:
                module_info["param_count"] = param_count
            # 获取模块的额外属性
            if hasattr(module, "in_features"):
                module_info["in_features"] = module.in_features
            if hasattr(module, "out_features"):
                module_info["out_features"] = module.out_features
            if hasattr(module, "num_heads"):
                module_info["num_heads"] = module.num_heads
            if hasattr(module, "hidden_size"):
                module_info["hidden_size"] = module.hidden_size

            structure["modules"][name] = module_info

        return structure

    def _should_include_layer(self, layer_name: str) -> bool:
        """判断是否应该包含该层"""
        # 检查排除模式
        for pattern in self.exclude_patterns:
            if re.search(pattern, layer_name):
                return False

        # 如果没有包含模式，则默认包含所有
        if not self.include_patterns:
            return True

        # 检查包含模式
        for pattern in self.include_patterns:
            if re.search(pattern, layer_name):
                return True

        return False

    def _process_tensor(self, tensor: torch.Tensor, name: str) -> Optional[torch.Tensor]:
        """处理张量，转换为可保存的格式"""
        if tensor is None:
            return None

        try:
            # 检查大小限制
            if self.max_tensor_size and tensor.numel() > self.max_tensor_size:
                print(f"Warning: Tensor {name} size ({tensor.numel()}) exceeds limit, skipping")
                return None

            # 使用 .data 避免 autograd 追踪，然后安全地克隆到 CPU
            # 这样可以避免 "view is being modified inplace" 错误
            with torch.no_grad():
                return tensor.data.clone().cpu()
        except Exception as e:
            print(f"Warning: Failed to process tensor {name}: {e}")
            return None

    def _process_output(self, output: Any, name: str) -> Any:
        """处理输出，支持多种类型"""
        if output is None:
            return None
        elif isinstance(output, torch.Tensor):
            return self._process_tensor(output, name)
        elif isinstance(output, tuple):
            return tuple(self._process_output(o, f"{name}[{i}]") for i, o in enumerate(output))
        elif isinstance(output, list):
            return [self._process_output(o, f"{name}[{i}]") for i, o in enumerate(output)]
        elif isinstance(output, dict):
            return {k: self._process_output(v, f"{name}.{k}") for k, v in output.items()}
        elif hasattr(output, 'last_hidden_state'):
            # 处理 ModelOutput 类型
            result = {}
            for key in ['last_hidden_state', 'hidden_states', 'attentions', 'logits']:
                if hasattr(output, key) and getattr(output, key) is not None:
                    result[key] = self._process_output(getattr(output, key), f"{name}.{key}")
            return result if result else None
        else:
            return None

    def _log_hook(self, layer_name: str, hook_type: str, tensor_shape: Any = None):
        """输出带颜色的日志"""
        if not self.verbose:
            return

        # 获取调用链
        call_chain = _get_call_chain(layer_name)

        # 根据 hook 类型选择颜色
        if hook_type == "forward":
            type_color = Colors.BRIGHT_GREEN
            arrow = "→"
        elif hook_type == "forward_input":
            type_color = Colors.BRIGHT_CYAN
            arrow = "⇒"
        elif hook_type == "backward":
            type_color = Colors.BRIGHT_YELLOW
            arrow = "←"
        else:
            type_color = Colors.WHITE
            arrow = "-"

        # 格式化层名（高亮关键部分）
        layer_parts = layer_name.split(".")
        formatted_name = ""
        for i, part in enumerate(layer_parts):
            if part.isdigit():
                formatted_name += f"{Colors.BRIGHT_MAGENTA}{part}{Colors.RESET}"
            elif part in ["self_attn", "mlp", "embed_tokens", "lm_head", "norm"]:
                formatted_name += f"{Colors.BRIGHT_BLUE}{part}{Colors.RESET}"
            else:
                formatted_name += f"{Colors.DIM}{part}{Colors.RESET}"
            if i < len(layer_parts) - 1:
                formatted_name += f"{Colors.DIM}.{Colors.RESET}"

        # 格式化 shape
        shape_str = ""
        if tensor_shape is not None:
            if isinstance(tensor_shape, torch.Size):
                shape_str = f" {Colors.DIM}shape={list(tensor_shape)}{Colors.RESET}"
            elif isinstance(tensor_shape, (list, tuple)):
                shape_str = f" {Colors.DIM}shape={tensor_shape}{Colors.RESET}"

        # 格式化调用链
        chain_str = ""
        if call_chain:
            chain_parts = call_chain.split(" -> ")
            colored_chain = f" {Colors.CYAN}{arrow}{Colors.RESET} ".join(
                [f"{Colors.DIM}{p}{Colors.RESET}" for p in chain_parts]
            )
            chain_str = f"\n    {Colors.DIM}调用链:{Colors.RESET} {colored_chain}"

        print(f"{type_color}[{hook_type.upper()}]{Colors.RESET} {formatted_name}{shape_str}{chain_str}")

    def _create_forward_hook(self, layer_name: str) -> Callable:
        """创建前向传播 hook"""
        def hook(module: nn.Module, input: Tuple, output: Any):
            if not self._enabled or not self.save_forward:
                return

            # 记录模块到调用链
            _push_call_chain(layer_name)

            try:
                with torch.no_grad():
                    # 保存输出
                    processed_output = self._process_output(output, f"{layer_name}/forward/output")

                    if processed_output is not None:
                        self.forward_outputs[layer_name] = processed_output
                        # 输出日志
                        output_shape = output.shape if isinstance(output, torch.Tensor) else None
                        self._log_hook(layer_name, "forward", output_shape)

                    # 保存输入
                    if self.save_input:
                        processed_input = self._process_output(input, f"{layer_name}/forward/input")
                        if processed_input is not None:
                            self.forward_inputs[layer_name] = processed_input
                            # 输出日志
                            input_shape = input[0].shape if input and isinstance(input[0], torch.Tensor) else None
                            self._log_hook(layer_name, "forward_input", input_shape)
            except Exception as e:
                print(f"{Colors.RED}Warning: Failed to save forward for {layer_name}: {e}{Colors.RESET}")

        return hook

    def _create_backward_hook(self, layer_name: str) -> Callable:
        """创建反向传播 hook（full backward hook）"""
        def hook(module: nn.Module, grad_input: Tuple, grad_output: Tuple):
            if not self._enabled or not self.save_backward:
                return

            try:
                # 使用 torch.no_grad() 确保不会干扰计算图
                with torch.no_grad():
                    # 保存梯度输入（相对于该层输入的梯度）
                    processed_grad_input = self._process_output(
                        grad_input, f"{layer_name}/backward/grad_input"
                    )
                    if processed_grad_input is not None:
                        self.backward_grad_inputs[layer_name] = processed_grad_input

                    # 保存梯度输出（相对于该层输出的梯度）
                    processed_grad_output = self._process_output(
                        grad_output, f"{layer_name}/backward/grad_output"
                    )
                    if processed_grad_output is not None:
                        self.backward_grad_outputs[layer_name] = processed_grad_output
                        # 输出日志
                        grad_shape = grad_output[0].shape if grad_output and isinstance(grad_output[0], torch.Tensor) else None
                        self._log_hook(layer_name, "backward", grad_shape)
            except Exception as e:
                # 如果处理失败，跳过此层但不中断训练
                print(f"{Colors.RED}Warning: Failed to save backward for {layer_name}: {e}{Colors.RESET}")

        return hook

    def register_hooks(
        self,
        model: "PreTrainedModel",
        register_all_submodules: bool = True,
        use_legacy_backward_hook: bool = True
    ) -> "LayerDebugger":
        """
        为模型注册 hooks

        Args:
            model: 要注册的模型
            register_all_submodules: 是否注册所有子模块（包含子层）
            use_legacy_backward_hook: 是否使用旧版 backward hook API（更兼容自定义 Function）

        Returns:
            self，支持链式调用
        """
        self.remove_hooks()  # 先清理旧的 hooks

        # 保存模型引用
        self._model = model

        registered_count = 0
        backward_failed = 0

        for name, module in model.named_modules():
            # 跳过空名称（根模块）
            if not name:
                name = "root"

            # 检查是否应该包含该层
            if not self._should_include_layer(name):
                continue

            # 跳过一些不需要的基础模块类型
            if isinstance(module, (nn.Dropout, nn.Identity)):
                continue

            # 注册前向 hook
            if self.save_forward:
                handle = module.register_forward_hook(self._create_forward_hook(name))
                self._forward_handles.append(handle)

            # 注册反向 hook
            if self.save_backward:
                try:
                    if use_legacy_backward_hook:
                        # 使用旧版 API，更兼容自定义 autograd Function
                        handle = module.register_backward_hook(self._create_backward_hook(name))
                    else:
                        handle = module.register_full_backward_hook(self._create_backward_hook(name))
                    self._backward_handles.append(handle)
                except Exception as e:
                    backward_failed += 1
                    print(f"Warning: Failed to register backward hook for {name}: {e}")

            registered_count += 1

        print(f"LayerDebugger: Registered hooks for {registered_count} modules")
        print(f"  - Forward hooks: {len(self._forward_handles)}")
        print(f"  - Backward hooks: {len(self._backward_handles)}")
        if backward_failed > 0:
            print(f"  - Backward hook registration failed: {backward_failed}")

        return self

    def remove_hooks(self) -> "LayerDebugger":
        """移除所有注册的 hooks"""
        for handle in self._forward_handles:
            handle.remove()
        for handle in self._backward_handles:
            handle.remove()

        removed_count = len(self._forward_handles) + len(self._backward_handles)
        self._forward_handles.clear()
        self._backward_handles.clear()

        if removed_count > 0:
            print(f"LayerDebugger: Removed {removed_count} hooks")

        return self

    def enable(self) -> "LayerDebugger":
        """启用数据收集"""
        self._enabled = True
        return self

    def disable(self) -> "LayerDebugger":
        """禁用数据收集"""
        self._enabled = False
        return self

    def clear(self) -> "LayerDebugger":
        """清空收集的数据"""
        self.forward_outputs.clear()
        self.forward_inputs.clear()
        self.backward_grad_inputs.clear()
        self.backward_grad_outputs.clear()
        return self

    def step(self) -> "LayerDebugger":
        """递增步数计数器"""
        self._step_count += 1
        return self

    def save(
        self,
        prefix: str = "debug",
        include_timestamp: bool = False,
        include_step: bool = False
    ) -> str:
        """
        保存收集的数据到文件

        Args:
            prefix: 文件名前缀
            include_timestamp: 是否包含时间戳
            include_step: 是否包含步数

        Returns:
            保存的文件路径
        """
        # 构建文件名
        filename_parts = [prefix]
        if include_step:
            filename_parts.append(f"step{self._step_count}")
        if include_timestamp:
            filename_parts.append(datetime.now().strftime("%Y%m%d_%H%M%S"))
        filename_parts.append("layer_outputs.pt")

        filename = "_".join(filename_parts)
        save_path = os.path.join(self.output_dir, filename)

        # 准备保存的数据
        data = {
            "metadata": {
                "step": self._step_count,
                "timestamp": datetime.now().isoformat(),
                "save_forward": self.save_forward,
                "save_backward": self.save_backward,
                "save_input": self.save_input,
                "num_forward_outputs": len(self.forward_outputs),
                "num_backward_grad_inputs": len(self.backward_grad_inputs),
                "num_backward_grad_outputs": len(self.backward_grad_outputs),
            },
            "forward_outputs": dict(self.forward_outputs),
            "backward_grad_inputs": dict(self.backward_grad_inputs),
            "backward_grad_outputs": dict(self.backward_grad_outputs),
        }

        if self.save_input:
            data["forward_inputs"] = dict(self.forward_inputs)
            data["metadata"]["num_forward_inputs"] = len(self.forward_inputs)

        # 保存模型结构
        model_structure = self._get_model_structure()
        if model_structure is not None:
            data["model_structure"] = model_structure

        torch.save(data, save_path)
        print(f"LayerDebugger: Saved to {save_path}")
        print(f"  - Forward outputs: {len(self.forward_outputs)} layers")
        if self.save_input:
            print(f"  - Forward inputs: {len(self.forward_inputs)} layers")
        print(f"  - Backward grad_inputs: {len(self.backward_grad_inputs)} layers")
        print(f"  - Backward grad_outputs: {len(self.backward_grad_outputs)} layers")
        if model_structure is not None:
            print(f"  - Model structure: {len(model_structure['modules'])} modules")

        return save_path

    def get_data(self) -> Dict[str, Any]:
        """获取当前收集的所有数据"""
        data = {
            "forward_outputs": dict(self.forward_outputs),
            "backward_grad_inputs": dict(self.backward_grad_inputs),
            "backward_grad_outputs": dict(self.backward_grad_outputs),
        }
        if self.save_input:
            data["forward_inputs"] = dict(self.forward_inputs)
        return data

    def get_layer_names(self) -> Dict[str, List[str]]:
        """获取所有已收集数据的层名称"""
        return {
            "forward_outputs": list(self.forward_outputs.keys()),
            "forward_inputs": list(self.forward_inputs.keys()),
            "backward_grad_inputs": list(self.backward_grad_inputs.keys()),
            "backward_grad_outputs": list(self.backward_grad_outputs.keys()),
        }

    def __enter__(self) -> "LayerDebugger":
        """上下文管理器入口"""
        self.enable()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """上下文管理器出口"""
        self.disable()
        self.remove_hooks()


# 全局调试器实例（便于快速使用）
_global_debugger: Optional[LayerDebugger] = None


def get_global_debugger() -> Optional[LayerDebugger]:
    """获取全局调试器实例"""
    return _global_debugger


def create_global_debugger(**kwargs) -> LayerDebugger:
    """创建并设置全局调试器"""
    global _global_debugger
    _global_debugger = LayerDebugger(**kwargs)
    return _global_debugger


def remove_global_debugger() -> None:
    """移除全局调试器"""
    global _global_debugger
    if _global_debugger is not None:
        _global_debugger.remove_hooks()
        _global_debugger = None


# 便捷函数
def quick_debug(
    model: "PreTrainedModel",
    output_dir: str = "./layer_outputs",
    include_patterns: Optional[List[str]] = None,
    exclude_patterns: Optional[List[str]] = None,
) -> LayerDebugger:
    """
    快速创建调试器并注册到模型

    Args:
        model: 要调试的模型
        output_dir: 输出目录
        include_patterns: 要包含的层名模式
        exclude_patterns: 要排除的层名模式

    Returns:
        配置好的 LayerDebugger 实例
    """
    debugger = LayerDebugger(
        output_dir=output_dir,
        save_forward=True,
        save_backward=True,
        include_patterns=include_patterns,
        exclude_patterns=exclude_patterns,
    )
    debugger.register_hooks(model)
    return debugger
