#!/usr/bin/env python3
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
比对模型每层输出的脚本（支持 forward 和 backward）

使用方法:
    python compare_layer_outputs.py --file1 run1_layer_outputs.pt --file2 run2_layer_outputs.pt

可选参数:
    --threshold: 误差阈值（默认 1e-5）
    --output: 输出比对结果的文件路径
    --detailed: 显示详细的逐层比对信息
    --mode: 比对模式 (all, forward, backward)
    --layer-filter: 层名过滤正则表达式
    --show-structure: 显示模型结构信息
    --forward-order: 按 forward 顺序比较对应模块的输入输出
                     顺序: embed_tokens -> layers.X.input_layernorm ->
                           layers.X.self_attn.q_proj -> layers.X.self_attn.kv_a_proj_with_mqa ->
                           layers.X.self_attn.kv_a_layernorm -> layers.X.self_attn.kv_b_proj ->
                           layers.X.self_attn.o_proj -> layers.X.self_attn ->
                           layers.X.post_attention_layernorm -> layers.X.mlp ->
                           norm -> lm_head
    --backward-order: 按 backward 顺序比较对应模块的梯度（forward 的反向）
                      顺序: lm_head -> norm -> layers.X.mlp -> layers.X.post_attention_layernorm ->
                            layers.X.self_attn -> layers.X.self_attn.o_proj ->
                            layers.X.self_attn.kv_b_proj -> ... -> embed_tokens
    --mlp-mismatch: 比较 MLP 输出，专门处理 token 数量不匹配的情况
                    适用于 CPU 单进程 vs KT 分布式运行的对比，会:
                    1. 检测两个文件的配置差异（rank 数、token 数等）
                    2. 使用 rank_shapes 识别有效 token 区域
                    3. 只比较有效数据，忽略 padding
                    4. 诊断可能的问题（如 token 截断）
"""

import argparse
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

import torch
import numpy as np


def normalize_key_names(d: Dict[str, Any]) -> Dict[str, Any]:
    """将 key 中的 .orig_module 替换为空字符串"""
    if not isinstance(d, dict):
        return d
    return {k.replace(".orig_module", ""): v for k, v in d.items()}


def load_layer_outputs(file_path: str) -> Dict[str, Any]:
    """加载层输出文件"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在: {file_path}")

    try:
        import torch.distributed.tensor  # noqa: F401 — needed for DTensor deserialization
    except ImportError:
        pass
    data = torch.load(file_path, map_location='cpu', weights_only=False)

    # 规范化 rank_shapes 的 key（去掉 .orig_module）
    if "rank_shapes" in data:
        data["rank_shapes"] = normalize_key_names(data["rank_shapes"])

    # 兼容旧格式（只有 forward_outputs）
    if "metadata" not in data:
        # 旧格式，转换为新格式
        data = {
            "metadata": {
                "save_forward": True,
                "save_backward": False,
                "num_forward_outputs": len(data),
            },
            "forward_outputs": data,
            "backward_grad_inputs": {},
            "backward_grad_outputs": {},
        }

    metadata = data.get("metadata", {})
    print(f"加载文件: {file_path}")
    print(f"  - Forward outputs: {metadata.get('num_forward_outputs', 0)} 层")
    print(f"  - Backward grad_inputs: {metadata.get('num_backward_grad_inputs', 0)} 层")
    print(f"  - Backward grad_outputs: {metadata.get('num_backward_grad_outputs', 0)} 层")
    print(f"  - Param grads: {metadata.get('num_param_grads', len(data.get('param_grads', {})))} params")
    print(f"  - LoRA params: {metadata.get('num_lora_params', len(data.get('lora_params', {})))} params")
    if "model_structure" in data:
        num_modules = len(data["model_structure"].get("modules", {}))
        print(f"  - Model structure: {num_modules} modules")

    # 将 key 中的 .orig_module 替换为空字符串
    for key in ["forward_outputs", "forward_inputs", "backward_grad_inputs", "backward_grad_outputs"]:
        if key in data:
            data[key] = normalize_key_names(data[key])

    return data


def build_valid_mask(tensor: torch.Tensor, rank_shapes: list) -> torch.Tensor:
    """根据 per-rank 原始 shape 构建 bool mask，标记 padding 区域为 False。

    rank_shapes: list of list[int], e.g. [[1, 56, 2048], [1, 40, 2048]]
    tensor: gathered tensor, e.g. shape [2, 56, 2048]

    返回与 tensor 同 shape 的 bool tensor，真实数据位置为 True。
    """
    if not rank_shapes or len(rank_shapes) <= 1:
        return torch.ones(tensor.shape, dtype=torch.bool)

    mask = torch.ones(tensor.shape, dtype=torch.bool)
    # 沿 dim-0 按 rank 划分
    offset = 0
    for rs in rank_shapes:
        batch = rs[0]
        # 对每个非 batch 维度，将超出 real shape 的区域设为 False
        for d in range(1, tensor.dim()):
            if d < len(rs) and rs[d] < tensor.shape[d]:
                idx = [slice(None)] * tensor.dim()
                idx[0] = slice(offset, offset + batch)
                idx[d] = slice(rs[d], tensor.shape[d])
                mask[tuple(idx)] = False
        offset += batch
    return mask


def get_rank_shapes_for_key(data: Dict, key: str) -> Optional[list]:
    """从 data['rank_shapes'] 中查找匹配 key 的 per-rank shapes。"""
    rs = data.get("rank_shapes", {})
    if not rs:
        return None
    # 精确匹配
    if key in rs:
        return rs[key]
    # 尝试去掉 canonical 前缀匹配
    canonical = get_canonical_name(key)
    if canonical in rs:
        return rs[canonical]
    # 后缀匹配
    for rk, rv in rs.items():
        if rk.endswith(key) or key.endswith(rk):
            return rv
    return None


def apply_mask_to_tensors(t1: torch.Tensor, t2: torch.Tensor,
                          rs1: Optional[list], rs2: Optional[list]):
    """用 rank_shapes 构建 mask，取两个 mask 的交集，返回仅含有效数据的 1D tensor。"""
    mask = torch.ones(t1.shape, dtype=torch.bool)
    if rs1:
        mask = mask & build_valid_mask(t1, rs1)
    if rs2:
        mask = mask & build_valid_mask(t2, rs2)
    if mask.all():
        return t1, t2
    return t1[mask], t2[mask]


def print_model_structure(data: Dict[str, Any], label: str):
    """打印模型结构信息"""
    structure = data.get("model_structure")
    if structure is None:
        print(f"\n【{label}】没有模型结构信息")
        return

    print(f"\n{'=' * 70}")
    print(f"【{label}】模型结构")
    print("=" * 70)

    # 打印模型配置
    config = structure.get("config", {})
    if config and not config.get("error"):
        print("\n配置信息:")
        important_keys = [
            "model_type", "hidden_size", "num_hidden_layers", "num_attention_heads",
            "intermediate_size", "vocab_size", "max_position_embeddings",
            "num_key_value_heads", "rope_theta", "torch_dtype"
        ]
        for key in important_keys:
            if key in config:
                print(f"  {key}: {config[key]}")

    # 打印模块统计
    modules = structure.get("modules", {})
    if modules:
        print(f"\n模块总数: {len(modules)}")

        # 统计各类模块
        class_counts = {}
        for mod_info in modules.values():
            cls_name = mod_info.get("class", "Unknown")
            class_counts[cls_name] = class_counts.get(cls_name, 0) + 1

        print("\n模块类型统计:")
        for cls_name, count in sorted(class_counts.items(), key=lambda x: -x[1]):
            print(f"  {cls_name}: {count}")

    # 打印模型表示（前 50 行）
    model_repr = structure.get("model_repr", "")
    if model_repr:
        lines = model_repr.split("\n")
        print(f"\n模型结构 (共 {len(lines)} 行):")
        for line in lines[:50]:
            print(f"  {line}")
        if len(lines) > 50:
            print(f"  ... 还有 {len(lines) - 50} 行未显示")


def get_canonical_name(name: str) -> str:
    """将模块名规范化，去除 .orig_module 等包装层前缀"""
    # 去除 .orig_module
    name = name.replace(".orig_module", "")
    return name


def extract_forward_key(name: str) -> Optional[str]:
    """
    从完整模块名中提取 forward 顺序的关键路径
    主要模块：embed_tokens, layers.X.self_attn (展开一层), layers.X.mlp, norm, lm_head

    self_attn 子模块顺序:
    - q_proj (或 q_a_proj, q_b_proj)
    - kv_a_proj_with_mqa
    - kv_a_layernorm
    - kv_b_proj
    - rotary_emb
    - o_proj
    """
    name = get_canonical_name(name)

    # embed_tokens
    if re.search(r"\.embed_tokens$", name):
        return "embed_tokens"

    # layers.X.input_layernorm
    match = re.search(r"\.layers\.(\d+)\.input_layernorm$", name)
    if match:
        return f"layers.{match.group(1)}.input_layernorm"

    # layers.X.self_attn 子模块 (展开一层)
    # self_attn.q_proj
    match = re.search(r"\.layers\.(\d+)\.self_attn\.q_proj$", name)
    if match:
        return f"layers.{match.group(1)}.self_attn.q_proj"

    # self_attn.q_a_proj (用于 q_lora_rank 不为 None 的情况)
    match = re.search(r"\.layers\.(\d+)\.self_attn\.q_a_proj$", name)
    if match:
        return f"layers.{match.group(1)}.self_attn.q_a_proj"

    # self_attn.q_a_layernorm
    match = re.search(r"\.layers\.(\d+)\.self_attn\.q_a_layernorm$", name)
    if match:
        return f"layers.{match.group(1)}.self_attn.q_a_layernorm"

    # self_attn.q_b_proj
    match = re.search(r"\.layers\.(\d+)\.self_attn\.q_b_proj$", name)
    if match:
        return f"layers.{match.group(1)}.self_attn.q_b_proj"

    # self_attn.kv_a_proj_with_mqa
    match = re.search(r"\.layers\.(\d+)\.self_attn\.kv_a_proj_with_mqa$", name)
    if match:
        return f"layers.{match.group(1)}.self_attn.kv_a_proj_with_mqa"

    # self_attn.kv_a_layernorm
    match = re.search(r"\.layers\.(\d+)\.self_attn\.kv_a_layernorm$", name)
    if match:
        return f"layers.{match.group(1)}.self_attn.kv_a_layernorm"

    # self_attn.kv_b_proj
    match = re.search(r"\.layers\.(\d+)\.self_attn\.kv_b_proj$", name)
    if match:
        return f"layers.{match.group(1)}.self_attn.kv_b_proj"

    # self_attn.rotary_emb
    match = re.search(r"\.layers\.(\d+)\.self_attn\.rotary_emb$", name)
    if match:
        return f"layers.{match.group(1)}.self_attn.rotary_emb"

    # self_attn.o_proj
    match = re.search(r"\.layers\.(\d+)\.self_attn\.o_proj$", name)
    if match:
        return f"layers.{match.group(1)}.self_attn.o_proj"

    # layers.X.self_attn (整体，不含子模块 - 作为 fallback)
    match = re.search(r"\.layers\.(\d+)\.self_attn$", name)
    if match:
        return f"layers.{match.group(1)}.self_attn"

    # layers.X.post_attention_layernorm
    match = re.search(r"\.layers\.(\d+)\.post_attention_layernorm$", name)
    if match:
        return f"layers.{match.group(1)}.post_attention_layernorm"

    # layers.X.mlp (不含子模块)
    match = re.search(r"\.layers\.(\d+)\.mlp$", name)
    if match:
        return f"layers.{match.group(1)}.mlp"

    # norm
    if re.search(r"model\.norm$", name):
        return "norm"

    # lm_head
    if re.search(r"\.lm_head$", name):
        return "lm_head"

    return None


def build_forward_order_mapping(keys: List[str]) -> Dict[str, str]:
    """
    构建从原始 key 到 forward 关键路径的映射
    返回: {原始key: forward关键路径}
    """
    mapping = {}
    for key in keys:
        forward_key = extract_forward_key(key)
        if forward_key:
            mapping[key] = forward_key
    return mapping


def _print_tensor_values(output1: Any, output2: Any, prefix: str = "", num_elements: int = 10,
                         rs1: Optional[list] = None, rs2: Optional[list] = None):
    """打印两个 tensor 的值（展示部分元素）"""
    # 提取 tensor
    def get_tensor(output):
        if isinstance(output, torch.Tensor):
            return output
        elif isinstance(output, tuple) and len(output) > 0:
            for item in output:
                if isinstance(item, torch.Tensor):
                    return item
        elif isinstance(output, dict):
            for key in ['last_hidden_state', 'hidden_states', 'logits']:
                if key in output and isinstance(output[key], torch.Tensor):
                    return output[key]
        return None

    t1 = get_tensor(output1)
    t2 = get_tensor(output2)

    if t1 is None or t2 is None:
        return

    # 用 mask 去掉 padding 区域
    t1f, t2f = apply_mask_to_tensors(t1.float(), t2.float(), rs1, rs2)
    flat1 = t1f.flatten()
    flat2 = t2f.flatten()
    # 截取到相同大小再比较
    min_len = min(flat1.numel(), flat2.numel())
    flat1 = flat1[:min_len]
    flat2 = flat2[:min_len]
    diff = (flat1 - flat2).abs()

    n = min(num_elements, flat1.numel())

    # 打印前 N 个元素
    print(f"{prefix}Tensor values (first {n} elements):")
    print(f"{prefix}  文件1: [{', '.join([f'{flat1[i].item():.6f}' for i in range(n)])}]")
    print(f"{prefix}  文件2: [{', '.join([f'{flat2[i].item():.6f}' for i in range(n)])}]")
    print(f"{prefix}  diff:  [{', '.join([f'{diff[i].item():.6e}' for i in range(n)])}]")

    # Last N 
    print(f"{prefix}Tensor values (last {n} elements):")
    print(f"{prefix}  文件1: [{', '.join([f'{flat1[-n + i].item():.6f}' for i in range(n)])}]")
    print(f"{prefix}  文件2: [{', '.join([f'{flat2[-n + i].item():.6f}' for i in range(n)])}]")
    print(f"{prefix}  diff:  [{', '.join([f'{diff[-n + i].item():.6e}' for i in range(n)])}]")
    # 找到最大差异位置并打印
    if diff.numel() > 0:
        max_idx = diff.argmax().item()
        print(f"{prefix}Max diff at index {max_idx}:")
        print(f"{prefix}  文件1[{max_idx}]: {flat1[max_idx].item():.6f}")
        print(f"{prefix}  文件2[{max_idx}]: {flat2[max_idx].item():.6f}")
        print(f"{prefix}  diff: {diff[max_idx].item():.6e}")


def compare_outputs_forward_order(
    data1: Dict[str, Any],
    data2: Dict[str, Any],
    threshold: float = 1e-5,
    detailed: bool = False
) -> Dict:
    """
    按 forward 顺序比较两个模型的输入和输出
    只比较主要模块（embed_tokens, self_attn, mlp, norm, lm_head），不深入到 linear 层
    """
    results = {
        "summary": {
            "threshold": threshold,
            "total_passed": 0,
            "total_failed": 0,
        },
        "comparisons": {},
        "input_comparisons": {}
    }

    print(f"\n{'=' * 70}")
    print("按 Forward 顺序比较模块输入和输出")
    print("=" * 70)

    # 获取 forward_outputs 和 forward_inputs
    fwd1 = data1.get("forward_outputs", {})
    fwd2 = data2.get("forward_outputs", {})
    inp1 = data1.get("forward_inputs", {})
    inp2 = data2.get("forward_inputs", {})

    # 规范化 key 名称
    fwd1 = {get_canonical_name(k): v for k, v in fwd1.items()}
    fwd2 = {get_canonical_name(k): v for k, v in fwd2.items()}
    inp1 = {get_canonical_name(k): v for k, v in inp1.items()}
    inp2 = {get_canonical_name(k): v for k, v in inp2.items()}

    has_inputs = bool(inp1) and bool(inp2)
    if has_inputs:
        print(f"检测到输入数据，将同时比较输入和输出")
    else:
        print(f"未检测到输入数据，仅比较输出")

    # 构建 forward 顺序映射
    mapping1 = build_forward_order_mapping(list(fwd1.keys()))
    mapping2 = build_forward_order_mapping(list(fwd2.keys()))

    # 为输入也构建映射
    inp_mapping1 = build_forward_order_mapping(list(inp1.keys()))
    inp_mapping2 = build_forward_order_mapping(list(inp2.keys()))
    inp_reverse1 = {v: k for k, v in inp_mapping1.items()}
    inp_reverse2 = {v: k for k, v in inp_mapping2.items()}

    # 反转映射: forward_key -> original_key
    reverse1 = {v: k for k, v in mapping1.items()}
    reverse2 = {v: k for k, v in mapping2.items()}

    # 获取所有 forward 关键路径
    all_forward_keys = set(mapping1.values()) | set(mapping2.values())

    # 定义 forward 顺序
    def forward_sort_key(key: str) -> tuple:
        # embed_tokens 最先
        if key == "embed_tokens":
            return (0, 0, 0, "")
        # layers.X.xxx
        match = re.match(r"layers\.(\d+)\.(.*)", key)
        if match:
            layer_idx = int(match.group(1))
            submodule = match.group(2)

            # self_attn 子模块顺序 (展开一层)
            self_attn_sub_order = {
                "self_attn.q_proj": 0,
                "self_attn.q_a_proj": 1,
                "self_attn.q_a_layernorm": 2,
                "self_attn.q_b_proj": 3,
                "self_attn.kv_a_proj_with_mqa": 4,
                "self_attn.kv_a_layernorm": 5,
                "self_attn.kv_b_proj": 6,
                "self_attn.rotary_emb": 7,
                "self_attn.o_proj": 8,
                "self_attn": 9,  # self_attn 整体放在子模块之后
            }

            # 主模块顺序
            main_order = {
                "input_layernorm": 0,
                "post_attention_layernorm": 2,
                "mlp": 3,
            }

            # 检查是否是 self_attn 相关
            if submodule in self_attn_sub_order:
                return (1, layer_idx, 1, self_attn_sub_order[submodule])
            elif submodule in main_order:
                return (1, layer_idx, main_order[submodule], 0)
            else:
                return (1, layer_idx, 9, submodule)

        # norm
        if key == "norm":
            return (2, 0, 0, "")
        # lm_head
        if key == "lm_head":
            return (3, 0, 0, "")
        return (9, 0, 0, key)

    sorted_keys = sorted(all_forward_keys, key=forward_sort_key)

    print(f"\n共找到 {len(sorted_keys)} 个主要模块进行比较")
    print("-" * 70)

    passed = 0
    failed = 0

    for forward_key in sorted_keys:
        orig_key1 = reverse1.get(forward_key)
        orig_key2 = reverse2.get(forward_key)

        if orig_key1 is None:
            print(f"\n[{forward_key}]")
            print(f"  文件1: 未找到")
            print(f"  文件2: {orig_key2}")
            continue

        if orig_key2 is None:
            print(f"\n[{forward_key}]")
            print(f"  文件1: {orig_key1}")
            print(f"  文件2: 未找到")
            continue

        # 先比较输入（如果有）
        inp_orig_key1 = inp_reverse1.get(forward_key)
        inp_orig_key2 = inp_reverse2.get(forward_key)

        if has_inputs and inp_orig_key1 and inp_orig_key2:
            input1 = inp1[inp_orig_key1]
            input2 = inp2[inp_orig_key2]

            inp_comp_results = {}
            inp_p, inp_f = compare_tensors_recursive(input1, input2, f"{forward_key}_input", threshold, inp_comp_results, False)

            # 获取输入的关键指标
            inp_diff_stats = None
            if inp_comp_results:
                inp_first_result = list(inp_comp_results.values())[0]
                if inp_first_result.get("shape_mismatch"):
                    inp_status = "FAIL (形状不匹配)"
                    inp_symbol = "✗"
                elif inp_first_result.get("passed", False):
                    inp_abs_max = inp_first_result.get("abs_max", 0)
                    inp_cos_sim = inp_first_result.get("cosine_similarity", 1.0)
                    inp_status = f"PASS (abs_max={inp_abs_max:.2e}, cos_sim={inp_cos_sim:.6f})"
                    inp_symbol = "✓"
                    inp_diff_stats = inp_first_result
                else:
                    inp_abs_max = inp_first_result.get("abs_max", 0)
                    inp_cos_sim = inp_first_result.get("cosine_similarity", 0)
                    inp_status = f"FAIL (abs_max={inp_abs_max:.2e}, cos_sim={inp_cos_sim:.6f})"
                    inp_symbol = "✗"
                    inp_diff_stats = inp_first_result
            else:
                inp_status = "无数据"
                inp_symbol = "?"

            print(f"\n[{forward_key}] INPUT {inp_symbol} {inp_status}")
            if inp_diff_stats and not inp_diff_stats.get("shape_mismatch") and not inp_diff_stats.get("empty_tensor"):
                print(f"    shape: {inp_diff_stats.get('shape', 'N/A')}, abs_max: {inp_diff_stats.get('abs_max', 0):.6e}, cos_sim: {inp_diff_stats.get('cosine_similarity', 0):.6f}")
                _print_tensor_values(input1, input2, "  ")

            results["input_comparisons"][forward_key] = inp_comp_results

        # 比较输出
        output1 = fwd1[orig_key1]
        output2 = fwd2[orig_key2]

        rs1 = get_rank_shapes_for_key(data1, orig_key1)
        rs2 = get_rank_shapes_for_key(data2, orig_key2)

        comp_results = {}
        p, f = compare_tensors_recursive(output1, output2, forward_key, threshold, comp_results, False, rs1=rs1, rs2=rs2)
        passed += p
        failed += f

        # 获取关键指标
        diff_stats = None
        if comp_results:
            first_result = list(comp_results.values())[0]
            if first_result.get("shape_mismatch"):
                status = "FAIL (形状不匹配)"
                status_symbol = "✗"
            elif first_result.get("passed", False):
                abs_max = first_result.get("abs_max", 0)
                cos_sim = first_result.get("cosine_similarity", 1.0)
                status = f"PASS (abs_max={abs_max:.2e}, cos_sim={cos_sim:.6f})"
                status_symbol = "✓"
                diff_stats = first_result
            else:
                abs_max = first_result.get("abs_max", 0)
                cos_sim = first_result.get("cosine_similarity", 0)
                status = f"FAIL (abs_max={abs_max:.2e}, cos_sim={cos_sim:.6f})"
                status_symbol = "✗"
                diff_stats = first_result
        else:
            status = "无数据"
            status_symbol = "?"

        print(f"[{forward_key}] OUTPUT {status_symbol} {status}")
        print(f"  文件1: {orig_key1}")
        print(f"  文件2: {orig_key2}")

        # 打印 diff vector 统计信息和 tensor 值
        if diff_stats and not diff_stats.get("shape_mismatch") and not diff_stats.get("empty_tensor"):
            print(f"  Diff stats:")
            print(f"    shape: {diff_stats.get('shape', 'N/A')}")
            print(f"    abs_max: {diff_stats.get('abs_max', 0):.6e}, abs_mean: {diff_stats.get('abs_mean', 0):.6e}, abs_std: {diff_stats.get('abs_std', 0):.6e}")
            print(f"    rel_max: {diff_stats.get('rel_max', 0):.6e}, rel_mean: {diff_stats.get('rel_mean', 0):.6e}")
            print(f"    l2_norm: {diff_stats.get('l2_norm', 0):.6e}, l2_norm_relative: {diff_stats.get('l2_norm_relative', 0):.6e}")
            print(f"    close_ratio: <1e-3: {diff_stats.get('close_ratio_1e3', 0)*100:.2f}%, <1e-5: {diff_stats.get('close_ratio_1e5', 0)*100:.2f}%, <1e-7: {diff_stats.get('close_ratio_1e7', 0)*100:.2f}%")

            # 打印 tensor 值
            _print_tensor_values(output1, output2, "  ", rs1=rs1, rs2=rs2)

        results["comparisons"][forward_key] = comp_results

    results["summary"]["total_passed"] = passed
    results["summary"]["total_failed"] = failed

    print("\n" + "=" * 70)
    print(f"总结: 通过 {passed}, 失败 {failed}")
    if failed == 0 and passed > 0:
        print("结论: ✓ 所有主要模块输出在阈值范围内一致")
    elif failed > 0:
        print(f"结论: ✗ 有 {failed} 项输出超过阈值")

    # 打印各层 cos_sim 变化总结
    _print_layer_cos_sim_summary(results["comparisons"])

    print("=" * 70)

    return results


def compare_backward_order(
    data1: Dict[str, Any],
    data2: Dict[str, Any],
    threshold: float = 1e-5,
) -> Dict:
    """
    按 backward 顺序比较两个模型的梯度
    顺序：lm_head -> norm -> layers.N.mlp -> layers.N.self_attn -> ... -> embed_tokens
    """
    results = {
        "summary": {
            "threshold": threshold,
            "total_passed": 0,
            "total_failed": 0,
        },
        "grad_input_comparisons": {},
        "grad_output_comparisons": {}
    }

    print(f"\n{'=' * 70}")
    print("按 Backward 顺序比较梯度")
    print("=" * 70)

    # 获取 backward 梯度数据
    grad_in1 = data1.get("backward_grad_inputs", {})
    grad_in2 = data2.get("backward_grad_inputs", {})
    grad_out1 = data1.get("backward_grad_outputs", {})
    grad_out2 = data2.get("backward_grad_outputs", {})

    # 规范化 key 名称
    grad_in1 = {get_canonical_name(k): v for k, v in grad_in1.items()}
    grad_in2 = {get_canonical_name(k): v for k, v in grad_in2.items()}
    grad_out1 = {get_canonical_name(k): v for k, v in grad_out1.items()}
    grad_out2 = {get_canonical_name(k): v for k, v in grad_out2.items()}

    print(f"grad_inputs: 文件1={len(grad_in1)}, 文件2={len(grad_in2)}")
    print(f"grad_outputs: 文件1={len(grad_out1)}, 文件2={len(grad_out2)}")

    # 构建 forward 顺序映射（然后反转为 backward 顺序）
    all_keys = set(grad_in1.keys()) | set(grad_in2.keys()) | set(grad_out1.keys()) | set(grad_out2.keys())
    mapping = build_forward_order_mapping(list(all_keys))

    # 反转映射
    reverse_in1 = {}
    reverse_in2 = {}
    reverse_out1 = {}
    reverse_out2 = {}

    for k, v in mapping.items():
        if k in grad_in1:
            reverse_in1[v] = k
        if k in grad_in2:
            reverse_in2[v] = k
        if k in grad_out1:
            reverse_out1[v] = k
        if k in grad_out2:
            reverse_out2[v] = k

    # 获取所有 forward 关键路径并按 backward 顺序排序（反向）
    all_forward_keys = set(mapping.values())

    def backward_sort_key(key: str) -> tuple:
        # backward 顺序是 forward 的反向
        # lm_head 最先，embed_tokens 最后
        if key == "lm_head":
            return (0, 0, 0, "")
        if key == "norm":
            return (1, 0, 0, "")
        match = re.match(r"layers\.(\d+)\.(.*)", key)
        if match:
            layer_idx = int(match.group(1))
            submodule = match.group(2)

            # self_attn 子模块逆序（o_proj 最先，q_proj 最后）
            self_attn_sub_order = {
                "self_attn.o_proj": 0,
                "self_attn.rotary_emb": 1,
                "self_attn.kv_b_proj": 2,
                "self_attn.kv_a_layernorm": 3,
                "self_attn.kv_a_proj_with_mqa": 4,
                "self_attn.q_b_proj": 5,
                "self_attn.q_a_layernorm": 6,
                "self_attn.q_a_proj": 7,
                "self_attn.q_proj": 8,
                "self_attn": 9,  # self_attn 整体
            }

            # 主模块逆序
            main_order = {
                "mlp": 0,
                "post_attention_layernorm": 2,
                "input_layernorm": 3,
            }

            # 检查是否是 self_attn 相关
            if submodule in self_attn_sub_order:
                return (2, -layer_idx, 1, self_attn_sub_order[submodule])
            elif submodule in main_order:
                return (2, -layer_idx, main_order[submodule], 0)
            else:
                return (2, -layer_idx, 9, submodule)

        if key == "embed_tokens":
            return (3, 0, 0, "")
        return (9, 0, 0, key)

    sorted_keys = sorted(all_forward_keys, key=backward_sort_key)

    print(f"\n共找到 {len(sorted_keys)} 个主要模块进行比较")
    print("-" * 70)

    passed = 0
    failed = 0

    for forward_key in sorted_keys:
        # 比较 grad_output（该层输出的梯度）
        out_key1 = reverse_out1.get(forward_key)
        out_key2 = reverse_out2.get(forward_key)

        if out_key1 and out_key2:
            grad1 = grad_out1[out_key1]
            grad2 = grad_out2[out_key2]

            comp_results = {}
            p, f = compare_tensors_recursive(grad1, grad2, f"{forward_key}_grad_out", threshold, comp_results, False)
            passed += p
            failed += f

            # 获取关键指标
            if comp_results:
                first_result = list(comp_results.values())[0]
                if first_result.get("shape_mismatch"):
                    status = "FAIL (形状不匹配)"
                    symbol = "✗"
                elif first_result.get("passed", False):
                    abs_max = first_result.get("abs_max", 0)
                    cos_sim = first_result.get("cosine_similarity", 1.0)
                    status = f"PASS (abs_max={abs_max:.2e}, cos_sim={cos_sim:.6f})"
                    symbol = "✓"
                else:
                    abs_max = first_result.get("abs_max", 0)
                    cos_sim = first_result.get("cosine_similarity", 0)
                    status = f"FAIL (abs_max={abs_max:.2e}, cos_sim={cos_sim:.6f})"
                    symbol = "✗"
            else:
                status = "无数据"
                symbol = "?"

            print(f"\n[{forward_key}] GRAD_OUT {symbol} {status}")
            if comp_results and not first_result.get("shape_mismatch") and not first_result.get("empty_tensor"):
                _print_tensor_values(grad1, grad2, "  ")

            results["grad_output_comparisons"][forward_key] = comp_results

        # 比较 grad_input（该层输入的梯度）
        in_key1 = reverse_in1.get(forward_key)
        in_key2 = reverse_in2.get(forward_key)

        if in_key1 and in_key2:
            grad1 = grad_in1[in_key1]
            grad2 = grad_in2[in_key2]

            comp_results = {}
            p, f = compare_tensors_recursive(grad1, grad2, f"{forward_key}_grad_in", threshold, comp_results, False)
            passed += p
            failed += f

            if comp_results:
                first_result = list(comp_results.values())[0]
                if first_result.get("shape_mismatch"):
                    status = "FAIL (形状不匹配)"
                    symbol = "✗"
                elif first_result.get("passed", False):
                    abs_max = first_result.get("abs_max", 0)
                    cos_sim = first_result.get("cosine_similarity", 1.0)
                    status = f"PASS (abs_max={abs_max:.2e}, cos_sim={cos_sim:.6f})"
                    symbol = "✓"
                else:
                    abs_max = first_result.get("abs_max", 0)
                    cos_sim = first_result.get("cosine_similarity", 0)
                    status = f"FAIL (abs_max={abs_max:.2e}, cos_sim={cos_sim:.6f})"
                    symbol = "✗"
            else:
                status = "无数据"
                symbol = "?"

            print(f"[{forward_key}] GRAD_IN {symbol} {status}")
            if comp_results and not first_result.get("shape_mismatch") and not first_result.get("empty_tensor"):
                _print_tensor_values(grad1, grad2, "  ")

            results["grad_input_comparisons"][forward_key] = comp_results

    results["summary"]["total_passed"] = passed
    results["summary"]["total_failed"] = failed

    print("\n" + "=" * 70)
    print(f"总结: 通过 {passed}, 失败 {failed}")
    if failed == 0 and passed > 0:
        print("结论: ✓ 所有主要模块梯度在阈值范围内一致")
    elif failed > 0:
        print(f"结论: ✗ 有 {failed} 项梯度超过阈值")

    # 打印各层梯度 cos_sim 变化总结
    _print_layer_grad_cos_sim_summary(results)

    print("=" * 70)

    return results


def _print_layer_grad_cos_sim_summary(results: Dict[str, Any]):
    """打印各层梯度 cos_sim 变化总结"""
    module_types = [
        "input_layernorm",
        "self_attn.q_proj",
        "self_attn.q_a_proj",
        "self_attn.q_a_layernorm",
        "self_attn.q_b_proj",
        "self_attn.kv_a_proj_with_mqa",
        "self_attn.kv_a_layernorm",
        "self_attn.kv_b_proj",
        "self_attn.rotary_emb",
        "self_attn.o_proj",
        "self_attn",
        "post_attention_layernorm",
        "mlp"
    ]

    print("\n" + "-" * 70)
    print("各层梯度 cos_sim 变化总结 (grad_input):")
    print("-" * 70)

    type_data = {t: [] for t in module_types}

    for forward_key, comp_data in results.get("grad_input_comparisons", {}).items():
        match = re.match(r"layers\.(\d+)\.(.*)", forward_key)
        if match:
            layer_idx = int(match.group(1))
            module_type = match.group(2)

            if module_type in type_data and comp_data:
                first_result = list(comp_data.values())[0] if comp_data else {}
                cos_sim = first_result.get("cosine_similarity", None)
                if cos_sim is not None:
                    type_data[module_type].append((layer_idx, cos_sim))

    for module_type in module_types:
        data = sorted(type_data[module_type], key=lambda x: x[0])
        if not data:
            continue

        print(f"\n[{module_type}] ({len(data)} layers):")

        items_per_line = 4
        for i in range(0, len(data), items_per_line):
            batch = data[i:i+items_per_line]
            line = "  " + "  ".join([f"L{idx:02d}: {sim:.6f}" for idx, sim in batch])
            print(line)

        sims = [sim for _, sim in data]
        min_sim = min(sims)
        max_sim = max(sims)
        avg_sim = sum(sims) / len(sims)
        min_layer = [idx for idx, sim in data if sim == min_sim][0]

        print(f"  统计: min={min_sim:.6f} (L{min_layer:02d}), max={max_sim:.6f}, avg={avg_sim:.6f}")

        if min_sim < 0.999:
            declining_layers = [(idx, sim) for idx, sim in data if sim < 0.999]
            if declining_layers:
                print(f"  ⚠ cos_sim < 0.999 的层: {', '.join([f'L{idx}({sim:.4f})' for idx, sim in declining_layers])}")


def _print_layer_cos_sim_summary(comparisons: Dict[str, Any]):
    """打印各层 cos_sim 变化总结"""
    # 按模块类型分组（包括 self_attn 子模块）
    module_types = [
        "input_layernorm",
        "self_attn.q_proj",
        "self_attn.q_a_proj",
        "self_attn.q_a_layernorm",
        "self_attn.q_b_proj",
        "self_attn.kv_a_proj_with_mqa",
        "self_attn.kv_a_layernorm",
        "self_attn.kv_b_proj",
        "self_attn.rotary_emb",
        "self_attn.o_proj",
        "self_attn",  # 整体 self_attn
        "post_attention_layernorm",
        "mlp"
    ]

    # 收集各类型模块的 cos_sim
    type_data = {t: [] for t in module_types}

    for forward_key, comp_data in comparisons.items():
        # 提取层号和模块类型
        match = re.match(r"layers\.(\d+)\.(.*)", forward_key)
        if match:
            layer_idx = int(match.group(1))
            module_type = match.group(2)

            if module_type in type_data and comp_data:
                first_result = list(comp_data.values())[0] if comp_data else {}
                cos_sim = first_result.get("cosine_similarity", None)
                if cos_sim is not None:
                    type_data[module_type].append((layer_idx, cos_sim))

    # 打印总结
    print("\n" + "-" * 70)
    print("各层 cos_sim 变化总结:")
    print("-" * 70)

    for module_type in module_types:
        data = sorted(type_data[module_type], key=lambda x: x[0])
        if not data:
            continue

        print(f"\n[{module_type}] ({len(data)} layers):")

        # 每行显示多个层的数据
        items_per_line = 4
        for i in range(0, len(data), items_per_line):
            batch = data[i:i+items_per_line]
            line = "  " + "  ".join([f"L{idx:02d}: {sim:.6f}" for idx, sim in batch])
            print(line)

        # 统计信息
        sims = [sim for _, sim in data]
        min_sim = min(sims)
        max_sim = max(sims)
        avg_sim = sum(sims) / len(sims)
        min_layer = [idx for idx, sim in data if sim == min_sim][0]

        print(f"  统计: min={min_sim:.6f} (L{min_layer:02d}), max={max_sim:.6f}, avg={avg_sim:.6f}")

        # 如果有明显下降，标记出来
        if min_sim < 0.999:
            declining_layers = [(idx, sim) for idx, sim in data if sim < 0.999]
            if declining_layers:
                print(f"  ⚠ cos_sim < 0.999 的层: {', '.join([f'L{idx}({sim:.4f})' for idx, sim in declining_layers])}")


def compute_tensor_metrics(tensor1: torch.Tensor, tensor2: torch.Tensor,
                           rs1: Optional[list] = None, rs2: Optional[list] = None) -> Dict[str, float]:
    """计算两个张量之间的各种差异指标。rs1/rs2 为 per-rank shapes，用于忽略 padding。"""
    # 确保张量在同一设备上并且是浮点类型
    t1 = tensor1.float()
    t2 = tensor2.float()

    # 处理形状不匹配：截取到各维度的最小值再比较
    if t1.shape != t2.shape:
        if t1.dim() != t2.dim():
            # 尝试展平后比较（如果元素数量相同或接近）
            if t1.numel() == t2.numel():
                t1 = t1.reshape(-1)
                t2 = t2.reshape(-1)
            else:
                # 元素数量不同，尝试截取到最小元素数量
                min_numel = min(t1.numel(), t2.numel())
                t1 = t1.reshape(-1)[:min_numel]
                t2 = t2.reshape(-1)[:min_numel]
        else:
            slices = tuple(slice(0, min(t1.shape[d], t2.shape[d])) for d in range(t1.dim()))
            t1 = t1[slices]
            t2 = t2[slices]

    orig_shape = list(t1.shape)

    # 用 rank_shapes mask 去掉 padding 区域
    t1, t2 = apply_mask_to_tensors(t1, t2, rs1, rs2)

    diff = t1 - t2
    abs_diff = torch.abs(diff)

    # 避免空张量
    if t1.numel() == 0:
        return {
            "shape_mismatch": False,
            "shape": str(orig_shape),
            "empty_tensor": True,
        }

    metrics = {
        "shape_mismatch": False,
        "shape": str(orig_shape),
        "numel": t1.numel(),
        # 绝对差异
        "abs_max": abs_diff.max().item(),
        "abs_mean": abs_diff.mean().item(),
        "abs_std": abs_diff.std().item() if t1.numel() > 1 else 0.0,
        # 相对差异（避免除零）
        "rel_max": (abs_diff / (torch.abs(t1) + 1e-10)).max().item(),
        "rel_mean": (abs_diff / (torch.abs(t1) + 1e-10)).mean().item(),
        # L2 范数
        "l2_norm": torch.norm(diff).item(),
        "l2_norm_relative": (torch.norm(diff) / (torch.norm(t1) + 1e-10)).item(),
        # 余弦相似度（展平后计算）
        "cosine_similarity": torch.nn.functional.cosine_similarity(
            t1.flatten().unsqueeze(0), t2.flatten().unsqueeze(0)
        ).item(),
        # 是否完全相等
        "exact_match": torch.equal(t1, t2),
        # 接近的元素比例
        "close_ratio_1e3": (abs_diff < 1e-3).float().mean().item(),
        "close_ratio_1e5": (abs_diff < 1e-5).float().mean().item(),
        "close_ratio_1e7": (abs_diff < 1e-7).float().mean().item(),
    }

    return metrics


def compare_tensors_recursive(
    data1: Any,
    data2: Any,
    prefix: str,
    threshold: float,
    results: Dict,
    detailed: bool,
    rs1: Optional[list] = None,
    rs2: Optional[list] = None,
) -> Tuple[int, int]:
    """递归比较张量数据，返回 (passed, failed) 计数"""
    passed = 0
    failed = 0

    if isinstance(data1, torch.Tensor) and isinstance(data2, torch.Tensor):
        metrics = compute_tensor_metrics(data1, data2, rs1=rs1, rs2=rs2)

        if metrics.get("shape_mismatch", False):
            is_passed = False
        elif metrics.get("empty_tensor", False):
            is_passed = True
        else:
            is_passed = metrics["abs_max"] <= threshold

        metrics["passed"] = is_passed
        results[prefix] = metrics

        if is_passed:
            passed += 1
        else:
            failed += 1

        if detailed:
            status = "✓ PASS" if is_passed else "✗ FAIL"
            if metrics.get("shape_mismatch", False):
                print(f"  {status} {prefix}: 形状不匹配 {metrics['shape1']} vs {metrics['shape2']}")
            elif metrics.get("empty_tensor", False):
                print(f"  {status} {prefix}: 空张量")
            else:
                print(f"  {status} {prefix}: abs_max={metrics['abs_max']:.2e}, "
                      f"cos_sim={metrics['cosine_similarity']:.6f}")

    elif isinstance(data1, (tuple, list)) and isinstance(data2, (tuple, list)):
        for i, (d1, d2) in enumerate(zip(data1, data2)):
            if d1 is not None and d2 is not None:
                p, f = compare_tensors_recursive(d1, d2, f"{prefix}[{i}]", threshold, results, detailed, rs1=rs1, rs2=rs2)
                passed += p
                failed += f

    elif isinstance(data1, dict) and isinstance(data2, dict):
        common_keys = set(data1.keys()) & set(data2.keys())
        for key in sorted(common_keys):
            if data1[key] is not None and data2[key] is not None:
                p, f = compare_tensors_recursive(data1[key], data2[key], f"{prefix}.{key}", threshold, results, detailed, rs1=rs1, rs2=rs2)
                passed += p
                failed += f

    return passed, failed


def compare_layer_dict(
    dict1: Dict[str, Any],
    dict2: Dict[str, Any],
    category: str,
    threshold: float,
    layer_filter: Optional[str],
    detailed: bool
) -> Dict:
    """比对两个层输出字典"""
    results = {
        "category": category,
        "total_layers_file1": len(dict1),
        "total_layers_file2": len(dict2),
        "common_layers": 0,
        "only_in_file1": [],
        "only_in_file2": [],
        "passed": 0,
        "failed": 0,
        "comparisons": {}
    }

    layers1 = set(dict1.keys())
    layers2 = set(dict2.keys())

    # 应用过滤器
    if layer_filter:
        pattern = re.compile(layer_filter)
        layers1 = {l for l in layers1 if pattern.search(l)}
        layers2 = {l for l in layers2 if pattern.search(l)}

    common_layers = layers1 & layers2
    results["common_layers"] = len(common_layers)
    results["only_in_file1"] = sorted(list(layers1 - layers2))
    results["only_in_file2"] = sorted(list(layers2 - layers1))

    # 按层名排序
    def sort_key(x):
        # 尝试提取数字进行排序
        numbers = re.findall(r'\d+', x)
        return (tuple(int(n) for n in numbers), x)

    sorted_layers = sorted(common_layers, key=sort_key)

    if detailed and sorted_layers:
        print(f"\n--- {category} ---")

    for layer_name in sorted_layers:
        data1 = dict1[layer_name]
        data2 = dict2[layer_name]

        layer_results = {}
        p, f = compare_tensors_recursive(data1, data2, layer_name, threshold, layer_results, detailed)
        results["passed"] += p
        results["failed"] += f
        results["comparisons"].update(layer_results)

    return results


def compare_layer_outputs(
    data1: Dict[str, Any],
    data2: Dict[str, Any],
    threshold: float = 1e-5,
    mode: str = "all",
    layer_filter: Optional[str] = None,
    detailed: bool = False
) -> Dict:
    """比对两组层输出"""
    results = {
        "summary": {
            "threshold": threshold,
            "mode": mode,
            "layer_filter": layer_filter,
            "total_passed": 0,
            "total_failed": 0,
        },
        "categories": {}
    }

    # 比对 forward outputs
    if mode in ["all", "forward"]:
        fwd_results = compare_layer_dict(
            data1.get("forward_outputs", {}),
            data2.get("forward_outputs", {}),
            "forward_outputs",
            threshold,
            layer_filter,
            detailed
        )
        results["categories"]["forward_outputs"] = fwd_results
        results["summary"]["total_passed"] += fwd_results["passed"]
        results["summary"]["total_failed"] += fwd_results["failed"]

    # 比对 forward inputs（如果存在）
    if mode in ["all", "forward"] and ("forward_inputs" in data1 or "forward_inputs" in data2):
        fwd_in_results = compare_layer_dict(
            data1.get("forward_inputs", {}),
            data2.get("forward_inputs", {}),
            "forward_inputs",
            threshold,
            layer_filter,
            detailed
        )
        results["categories"]["forward_inputs"] = fwd_in_results
        results["summary"]["total_passed"] += fwd_in_results["passed"]
        results["summary"]["total_failed"] += fwd_in_results["failed"]

    # 比对 backward grad_inputs
    if mode in ["all", "backward"]:
        bwd_in_results = compare_layer_dict(
            data1.get("backward_grad_inputs", {}),
            data2.get("backward_grad_inputs", {}),
            "backward_grad_inputs",
            threshold,
            layer_filter,
            detailed
        )
        results["categories"]["backward_grad_inputs"] = bwd_in_results
        results["summary"]["total_passed"] += bwd_in_results["passed"]
        results["summary"]["total_failed"] += bwd_in_results["failed"]

    # 比对 backward grad_outputs
    if mode in ["all", "backward"]:
        bwd_out_results = compare_layer_dict(
            data1.get("backward_grad_outputs", {}),
            data2.get("backward_grad_outputs", {}),
            "backward_grad_outputs",
            threshold,
            layer_filter,
            detailed
        )
        results["categories"]["backward_grad_outputs"] = bwd_out_results
        results["summary"]["total_passed"] += bwd_out_results["passed"]
        results["summary"]["total_failed"] += bwd_out_results["failed"]

    return results


def print_summary(results: Dict):
    """打印比对结果摘要"""
    summary = results["summary"]

    print("\n" + "=" * 70)
    print("比对结果摘要")
    print("=" * 70)
    print(f"阈值: {summary['threshold']}")
    print(f"模式: {summary['mode']}")
    if summary['layer_filter']:
        print(f"层过滤: {summary['layer_filter']}")
    print(f"总通过数: {summary['total_passed']}")
    print(f"总失败数: {summary['total_failed']}")

    print("\n" + "-" * 70)
    print("各类别详情:")
    print("-" * 70)

    for cat_name, cat_data in results["categories"].items():
        print(f"\n【{cat_name}】")
        print(f"  文件1 层数: {cat_data['total_layers_file1']}")
        print(f"  文件2 层数: {cat_data['total_layers_file2']}")
        print(f"  共同层数: {cat_data['common_layers']}")
        print(f"  通过: {cat_data['passed']}, 失败: {cat_data['failed']}")

        if cat_data['only_in_file1']:
            print(f"  仅在文件1: {cat_data['only_in_file1']}")
        if cat_data['only_in_file2']:
            print(f"  仅在文件2: {cat_data['only_in_file2']}")

    # 显示失败的详情
    all_failed = []
    for cat_name, cat_data in results["categories"].items():
        for comp_name, comp_data in cat_data.get("comparisons", {}).items():
            if not comp_data.get("passed", True):
                all_failed.append((cat_name, comp_name, comp_data))

    if all_failed:
        print("\n" + "-" * 70)
        print(f"失败项详情 (共 {len(all_failed)} 项):")
        print("-" * 70)
        for cat_name, comp_name, metrics in all_failed[:20]:  # 最多显示20项
            if metrics.get("shape_mismatch", False):
                print(f"  [{cat_name}] {comp_name}: 形状不匹配 {metrics['shape1']} vs {metrics['shape2']}")
            else:
                print(f"  [{cat_name}] {comp_name}:")
                print(f"    最大绝对差异: {metrics.get('abs_max', 'N/A'):.2e}")
                print(f"    余弦相似度: {metrics.get('cosine_similarity', 'N/A'):.6f}")
        if len(all_failed) > 20:
            print(f"  ... 还有 {len(all_failed) - 20} 项未显示")

    print("\n" + "=" * 70)

    # 总体结论
    if summary['total_failed'] == 0 and summary['total_passed'] > 0:
        print("结论: ✓ 所有比对项均在阈值范围内")
    elif summary['total_passed'] == 0 and summary['total_failed'] == 0:
        print("结论: ⚠ 没有可比对的数据")
    else:
        print(f"结论: ✗ 有 {summary['total_failed']} 项超过阈值")


def save_results(results: Dict, output_path: str):
    """保存比对结果到文件"""
    # 转换不可序列化的类型
    def convert(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(i) for i in obj]
        return obj

    results = convert(results)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n比对结果已保存到: {output_path}")


def compare_kt_lora_grads(data1: Dict, data2: Dict, layer_filter: Optional[str] = None):
    """比较 KT LoRA 梯度（将 KT 的合并张量按 expert 拆开与 PEFT 的逐 expert 梯度对比）

    自动检测哪个文件是 ref（含 experts.X.proj.lora_A/B），哪个是 KT（含 lora_params.xxx_lora_a/b）。
    """
    import re
    import torch.nn.functional as F

    grads1 = data1.get("param_grads", {})
    grads2 = data2.get("param_grads", {})

    # Detect which file is ref (PEFT per-expert) and which is KT (merged lora_params)
    has_experts_1 = any("mlp.experts." in k and "lora_" in k for k in grads1)
    has_lora_params_1 = any("lora_params." in k for k in grads1)
    has_experts_2 = any("mlp.experts." in k and "lora_" in k for k in grads2)
    has_lora_params_2 = any("lora_params." in k for k in grads2)

    if has_experts_1 and has_lora_params_2:
        ref_grads, kt_grads = grads1, grads2
        print("检测: 文件1=ref(PEFT per-expert), 文件2=KT(lora_params)")
    elif has_experts_2 and has_lora_params_1:
        ref_grads, kt_grads = grads2, grads1
        print("检测: 文件2=ref(PEFT per-expert), 文件1=KT(lora_params)")
    else:
        print("错误: 无法检测 ref/KT 文件（需要一个含 experts.X.lora，一个含 lora_params）")
        return

    # KT key suffix -> (PEFT proj name, PEFT lora type)
    kt_to_peft = {
        "gate_lora_a": ("gate_proj", "lora_A"),
        "gate_lora_b": ("gate_proj", "lora_B"),
        "up_lora_a": ("up_proj", "lora_A"),
        "up_lora_b": ("up_proj", "lora_B"),
        "down_lora_a": ("down_proj", "lora_A"),
        "down_lora_b": ("down_proj", "lora_B"),
    }

    # Discover all MoE layer indices from KT keys
    layer_pat = re.compile(r"layers\.(\d+)\.mlp\.lora_params\.")
    layer_indices = sorted(set(int(m.group(1)) for k in kt_grads for m in [layer_pat.search(k)] if m))

    if layer_filter:
        filt = re.compile(layer_filter)
        layer_indices = [i for i in layer_indices if filt.search(f"layers.{i}")]

    print("\n" + "=" * 90)
    print("KT Expert LoRA 梯度比较 (KT merged tensor vs PEFT per-expert)")
    print("=" * 90)

    def _to_tensor(v):
        """Convert DTensor to plain Tensor if needed."""
        try:
            from torch.distributed.tensor import DTensor
            if isinstance(v, DTensor):
                return v._local_tensor
        except ImportError:
            pass
        return v

    all_results = []  # (layer, kt_suffix, expert, ref_norm, kt_norm, ratio, cos)

    for layer_idx in layer_indices:
        layer_results = []
        for kt_suffix, (proj, lora_type) in kt_to_peft.items():
            kt_key = f"base_model.model.model.layers.{layer_idx}.mlp.lora_params.{kt_suffix}"
            if kt_key not in kt_grads:
                continue
            kt_grad = _to_tensor(kt_grads[kt_key]).float()
            num_experts = kt_grad.shape[0]

            for exp_idx in range(num_experts):
                peft_key = (
                    f"base_model.model.model.layers.{layer_idx}.mlp.experts.{exp_idx}"
                    f".{proj}.{lora_type}.default.weight"
                )
                if peft_key not in ref_grads:
                    continue
                ref_g = _to_tensor(ref_grads[peft_key]).float()
                kt_e = kt_grad[exp_idx]

                rn = ref_g.norm().item()
                kn = kt_e.norm().item()
                # Skip all-zero pairs
                if rn < 1e-12 and kn < 1e-12:
                    continue
                ratio = kn / rn if rn > 1e-12 else float("inf")
                cos = F.cosine_similarity(
                    ref_g.flatten().unsqueeze(0), kt_e.flatten().unsqueeze(0)
                ).item()
                layer_results.append((kt_suffix, exp_idx, rn, kn, ratio, cos))

        if not layer_results:
            continue

        # Print per-layer summary
        cos_vals = [r[5] for r in layer_results if r[4] != float("inf")]
        ratio_vals = [r[4] for r in layer_results if r[4] != float("inf")]
        n_anomaly = sum(1 for r in layer_results if r[5] < 0.95 or r[4] < 0.8 or r[4] > 1.2)

        print(f"\nLayer {layer_idx:>2}: "
              f"compared={len(layer_results)}, anomalies={n_anomaly}, "
              f"cos=[{min(cos_vals):.4f}, {max(cos_vals):.4f}] mean={sum(cos_vals)/len(cos_vals):.4f}, "
              f"ratio=[{min(ratio_vals):.4f}, {max(ratio_vals):.4f}] mean={sum(ratio_vals)/len(ratio_vals):.4f}")

        # Show worst anomalies per layer (up to 5)
        anomalies = [(s, e, rn, kn, ra, c) for s, e, rn, kn, ra, c in layer_results
                     if c < 0.95 or ra < 0.8 or ra > 1.2]
        anomalies.sort(key=lambda x: x)  # worst cos first
        for s, e, rn, kn, ra, c in anomalies:
            print(f"    {s:<15} expert={e:<3} ref_norm={rn:.6e} kt_norm={kn:.6e} ratio={ra:.4f} cos={c:.6f}")

        all_results.extend(layer_results)

    # Global summary
    if all_results:
        cos_all = [r[5] for r in all_results if r[4] != float("inf")]
        ratio_all = [r[4] for r in all_results if r[4] != float("inf")]
        n_total = len(all_results)
        n_bad = sum(1 for r in all_results if r[5] < 0.95 or r[4] < 0.8 or r[4] > 1.2)
        print(f"\n{'=' * 90}")
        print(f"全局汇总: total={n_total}, anomalies={n_bad} ({100*n_bad/n_total:.1f}%)")
        print(f"  cos_sim:  min={min(cos_all):.6f}, mean={sum(cos_all)/len(cos_all):.6f}, max={max(cos_all):.6f}")
        print(f"  ratio:    min={min(ratio_all):.6f}, mean={sum(ratio_all)/len(ratio_all):.6f}, max={max(ratio_all):.6f}")

        # Group by lora type
        by_type = {}
        for s, e, rn, kn, ra, c in all_results:
            by_type.setdefault(s, []).append((ra, c))
        print(f"\n  按 LoRA 类型汇总:")
        for suffix in ["gate_lora_b", "up_lora_b", "down_lora_b", "gate_lora_a", "up_lora_a", "down_lora_a"]:
            vals = by_type.get(suffix, [])
            if not vals:
                continue
            ratios = [v[0] for v in vals if v[0] != float("inf")]
            coss = [v[1] for v in vals]
            if ratios:
                print(f"    {suffix:<15}: n={len(vals)}, "
                      f"cos=[{min(coss):.4f},{max(coss):.4f}] mean={sum(coss)/len(coss):.4f}, "
                      f"ratio=[{min(ratios):.4f},{max(ratios):.4f}] mean={sum(ratios)/len(ratios):.4f}")
    print("=" * 90)


def compare_attn_lora_grads(data1: Dict, data2: Dict, layer_filter: Optional[str] = None):
    """比较 Attention LoRA 梯度 (非 MoE 的 LoRA 参数)

    比较 self_attn 模块中的 LoRA 梯度:
    - q_proj.lora_A/B
    - kv_a_proj_with_mqa.lora_A/B
    - kv_b_proj.lora_A/B
    - o_proj.lora_A/B
    """
    import re
    import torch.nn.functional as F

    grads1 = data1.get("param_grads", {})
    grads2 = data2.get("param_grads", {})

    def _to_tensor(v):
        """Convert DTensor to plain Tensor if needed."""
        try:
            from torch.distributed.tensor import DTensor
            if isinstance(v, DTensor):
                return v.full_tensor()
        except ImportError:
            pass
        return v

    # Find attention LoRA keys (exclude mlp/experts)
    attn_pattern = re.compile(r"layers\.(\d+)\.self_attn\.([\w_]+)\.lora_([AB])\.default\.weight")

    def get_attn_lora_keys(grads):
        keys = {}
        for k in grads.keys():
            m = attn_pattern.search(k)
            if m:
                layer_idx, proj, lora_type = m.groups()
                key_tuple = (int(layer_idx), proj, lora_type)
                keys[key_tuple] = k
        return keys

    keys1 = get_attn_lora_keys(grads1)
    keys2 = get_attn_lora_keys(grads2)

    common_keys = set(keys1.keys()) & set(keys2.keys())

    if layer_filter:
        filt = re.compile(layer_filter)
        common_keys = {k for k in common_keys if filt.search(f"layers.{k[0]}")}

    print("\n" + "=" * 90)
    print("Attention LoRA 梯度比较 (self_attn 模块)")
    print("=" * 90)
    print(f"文件1 attention LoRA params: {len(keys1)}")
    print(f"文件2 attention LoRA params: {len(keys2)}")
    print(f"共同 params: {len(common_keys)}")
    print("-" * 90)

    if not common_keys:
        print("没有找到共同的 attention LoRA 参数")
        return

    # Group by layer
    layer_indices = sorted(set(k[0] for k in common_keys))

    all_results = []  # (layer, proj, lora_type, norm1, norm2, ratio, cos)

    for layer_idx in layer_indices:
        layer_results = []
        layer_keys = sorted([k for k in common_keys if k[0] == layer_idx], key=lambda x: (x[1], x[2]))

        for key_tuple in layer_keys:
            k1 = keys1[key_tuple]
            k2 = keys2[key_tuple]

            g1 = _to_tensor(grads1[k1]).float()
            g2 = _to_tensor(grads2[k2]).float()

            # Handle shape mismatch
            if g1.shape != g2.shape:
                print(f"  Shape mismatch: layer={layer_idx}, {key_tuple[1]}.lora_{key_tuple[2]}: "
                      f"{list(g1.shape)} vs {list(g2.shape)}")
                continue

            n1 = g1.norm().item()
            n2 = g2.norm().item()

            # Skip all-zero pairs
            if n1 < 1e-12 and n2 < 1e-12:
                continue

            ratio = n2 / n1 if n1 > 1e-12 else float("inf")
            cos = F.cosine_similarity(g1.flatten().unsqueeze(0), g2.flatten().unsqueeze(0)).item()

            # Also compute mean/sum ratios
            mean1 = g1.abs().mean().item()
            mean2 = g2.abs().mean().item()
            mean_ratio = mean2 / mean1 if mean1 > 1e-12 else float("inf")

            layer_results.append((key_tuple[1], key_tuple[2], n1, n2, ratio, cos, mean_ratio))

        if not layer_results:
            continue

        # Print per-layer summary
        cos_vals = [r[5] for r in layer_results if r[4] != float("inf")]
        ratio_vals = [r[4] for r in layer_results if r[4] != float("inf")]
        mean_ratio_vals = [r[6] for r in layer_results if r[6] != float("inf")]
        n_anomaly = sum(1 for r in layer_results if r[5] < 0.95 or r[4] < 0.8 or r[4] > 1.6)

        print(f"\nLayer {layer_idx:>2}: "
              f"compared={len(layer_results)}, anomalies={n_anomaly}, "
              f"cos=[{min(cos_vals):.4f}, {max(cos_vals):.4f}] mean={sum(cos_vals)/len(cos_vals):.4f}, "
              f"norm_ratio=[{min(ratio_vals):.4f}, {max(ratio_vals):.4f}] mean={sum(ratio_vals)/len(ratio_vals):.4f}, "
              f"mean_ratio=[{min(mean_ratio_vals):.4f}, {max(mean_ratio_vals):.4f}]")

        # Show details for each proj
        for proj, lora_type, n1, n2, ratio, cos, mean_ratio in layer_results:
            status = "✓" if cos > 0.95 and 0.8 < ratio < 1.6 else "✗"
            print(f"    [{status}] {proj}.lora_{lora_type}: "
                  f"norm1={n1:.4e}, norm2={n2:.4e}, ratio={ratio:.4f}, cos={cos:.4f}, mean_ratio={mean_ratio:.4f}")

        all_results.extend(layer_results)

    # Global summary
    if all_results:
        cos_all = [r[5] for r in all_results if r[4] != float("inf")]
        ratio_all = [r[4] for r in all_results if r[4] != float("inf")]
        mean_ratio_all = [r[6] for r in all_results if r[6] != float("inf")]
        n_total = len(all_results)
        n_bad = sum(1 for r in all_results if r[5] < 0.95 or r[4] < 0.8 or r[4] > 1.6)

        print(f"\n{'=' * 90}")
        print(f"全局汇总: total={n_total}, anomalies={n_bad} ({100*n_bad/n_total:.1f}%)")
        print(f"  cos_sim:    min={min(cos_all):.6f}, mean={sum(cos_all)/len(cos_all):.6f}, max={max(cos_all):.6f}")
        print(f"  norm_ratio: min={min(ratio_all):.6f}, mean={sum(ratio_all)/len(ratio_all):.6f}, max={max(ratio_all):.6f}")
        print(f"  mean_ratio: min={min(mean_ratio_all):.6f}, mean={sum(mean_ratio_all)/len(mean_ratio_all):.6f}, max={max(mean_ratio_all):.6f}")

        # Group by proj type
        by_proj = {}
        for proj, lora_type, n1, n2, ratio, cos, mean_ratio in all_results:
            key = f"{proj}.lora_{lora_type}"
            by_proj.setdefault(key, []).append((ratio, cos, mean_ratio))

        print(f"\n  按 Proj 类型汇总:")
        for proj_key in sorted(by_proj.keys()):
            vals = by_proj[proj_key]
            ratios = [v[0] for v in vals if v[0] != float("inf")]
            coss = [v[1] for v in vals]
            mean_ratios = [v[2] for v in vals if v[2] != float("inf")]
            if ratios:
                print(f"    {proj_key:<35}: n={len(vals):>3}, "
                      f"cos=[{min(coss):.4f},{max(coss):.4f}] mean={sum(coss)/len(coss):.4f}, "
                      f"ratio=[{min(ratios):.4f},{max(ratios):.4f}] mean={sum(ratios)/len(ratios):.4f}")

        # Print sqrt(2) check
        import math
        avg_ratio = sum(ratio_all) / len(ratio_all)
        print(f"\n  sqrt(2) = {math.sqrt(2):.6f}, 平均 ratio = {avg_ratio:.6f}, "
              f"差异 = {abs(avg_ratio - math.sqrt(2)):.6f}")

    print("=" * 90)


def compare_param_grads(
    data1: Dict[str, Any],
    data2: Dict[str, Any],
    threshold: float = 1e-5,
    layer_filter: Optional[str] = None,
) -> Dict:
    """
    比较两个文件中保存的参数梯度
    """
    grads1 = data1.get("param_grads", {})
    grads2 = data2.get("param_grads", {})

    # 规范化 key
    grads1 = {get_canonical_name(k): v for k, v in grads1.items()}
    grads2 = {get_canonical_name(k): v for k, v in grads2.items()}

    keys1 = set(grads1.keys())
    keys2 = set(grads2.keys())

    if layer_filter:
        pattern = re.compile(layer_filter)
        keys1 = {k for k in keys1 if pattern.search(k)}
        keys2 = {k for k in keys2 if pattern.search(k)}

    common = sorted(keys1 & keys2)
    only1 = keys1 - keys2
    only2 = keys2 - keys1

    print(f"\n{'=' * 70}")
    print("参数梯度比较")
    print("=" * 70)
    print(f"文件1 params: {len(keys1)}, 文件2 params: {len(keys2)}, 共同: {len(common)}")
    if only1:
        print(f"仅在文件1 ({len(only1)}): {sorted(only1)[:5]}{'...' if len(only1) > 5 else ''}")
    if only2:
        print(f"仅在文件2 ({len(only2)}): {sorted(only2)[:5]}{'...' if len(only2) > 5 else ''}")
    print("-" * 70)

    passed = 0
    failed = 0
    results_list = []

    for name in common:
        g1 = grads1[name]
        g2 = grads2[name]
        metrics = compute_tensor_metrics(g1, g2)
        metrics["passed"] = not metrics.get("shape_mismatch", False) and metrics.get("abs_max", float("inf")) <= threshold

        if metrics["passed"]:
            passed += 1
        else:
            failed += 1

        if metrics.get("shape_mismatch"):
            print(f"  ✗ {name}: SHAPE MISMATCH {metrics['shape1']} vs {metrics['shape2']}")
        elif not metrics["passed"]:
            results_list.append((name, metrics))

    # 按 cosine similarity 升序显示差异最大的
    results_list.sort(key=lambda x: x[1].get("cosine_similarity", 1.0))

    for name, metrics in results_list[:30]:
        cos = metrics.get("cosine_similarity", 0)
        abs_max = metrics.get("abs_max", 0)
        abs_mean = metrics.get("abs_mean", 0)
        symbol = "✓" if metrics["passed"] else "✗"
        print(f"  {symbol} {name}: abs_max={abs_max:.4e} abs_mean={abs_mean:.4e} cos={cos:.6f}")

    if len(results_list) > 30:
        print(f"  ... 还有 {len(results_list) - 30} 项未显示")

    print(f"\n总结: 通过 {passed}, 失败 {failed}")

    # 按模块类型汇总 cosine similarity
    type_cos = {}
    for name in common:
        metrics = compute_tensor_metrics(grads1[name], grads2[name])
        cos = metrics.get("cosine_similarity", None)
        if cos is None:
            continue
        # 提取类型: lora_A / lora_B / base_layer 等
        for tag in ["lora_A", "lora_B", "base_layer"]:
            if tag in name:
                type_cos.setdefault(tag, []).append(cos)
                break
        else:
            type_cos.setdefault("other", []).append(cos)

    if type_cos:
        print(f"\n按参数类型汇总 cosine similarity:")
        for tag in ["lora_A", "lora_B", "base_layer", "other"]:
            vals = type_cos.get(tag, [])
            if not vals:
                continue
            print(f"  [{tag}] count={len(vals)} mean={sum(vals)/len(vals):.6f} min={min(vals):.6f} max={max(vals):.6f}")

    print("=" * 70)
    return {"passed": passed, "failed": failed}


def compare_mlp_with_token_mismatch(
    data1: Dict[str, Any],
    data2: Dict[str, Any],
    threshold: float = 1e-5,
    layer_filter: Optional[str] = None,
) -> Dict:
    """
    比较两个文件的 MLP 输出，处理 token 数量不匹配的情况。

    适用于以下场景:
    - CPU 单进程运行 vs KT 分布式运行
    - 两个文件的 batch/seq 配置不同
    - 分布式训练时各 rank 的数据量不同

    比较策略:
    1. 使用 rank_shapes 识别每个 rank 的有效 token 数量
    2. 比较 embedding 以确认输入数据是否匹配
    3. 对于 gate 输出，按有效 token 对齐比较
    4. 对于 MLP hidden_states 输出，比较重叠部分
    """
    import torch.nn.functional as F

    fwd1 = data1.get("forward_outputs", {})
    fwd2 = data2.get("forward_outputs", {})
    rs1 = data1.get("rank_shapes", {})
    rs2 = data2.get("rank_shapes", {})

    print(f"\n{'=' * 70}")
    print("MLP 输出比较 (处理 token 数量不匹配)")
    print("=" * 70)

    # 1. 检测配置差异
    print("\n1. 配置检测:")

    # 获取 embedding shapes
    embed_key = None
    for key in fwd1.keys():
        if "embed_tokens" in key:
            embed_key = key
            break

    if embed_key:
        embed1 = fwd1.get(embed_key)
        embed2 = fwd2.get(embed_key)
        embed_rs1 = rs1.get(embed_key, [])
        embed_rs2 = rs2.get(embed_key, [])

        print(f"   文件1 embedding shape: {list(embed1.shape) if embed1 is not None else 'N/A'}")
        print(f"   文件1 rank_shapes: {embed_rs1}")
        print(f"   文件2 embedding shape: {list(embed2.shape) if embed2 is not None else 'N/A'}")
        print(f"   文件2 rank_shapes: {embed_rs2}")

        # 计算有效 token 数
        def count_valid_tokens(rs_list):
            if not rs_list:
                return None
            total = 0
            for rs in rs_list:
                if len(rs) >= 2:
                    total += rs[0] * rs[1]  # batch * seq
            return total

        valid1 = count_valid_tokens(embed_rs1)
        valid2 = count_valid_tokens(embed_rs2)

        if valid1 is not None or valid2 is not None:
            print(f"\n   有效 token 数:")
            print(f"   文件1: {valid1 if valid1 else 'unknown (single process)'}")
            print(f"   文件2: {valid2 if valid2 else 'unknown (single process)'}")

    # 2. 比较 embedding 以验证输入数据匹配
    print("\n2. Embedding 输入验证:")
    if embed_key and embed1 is not None and embed2 is not None:
        # 获取两者的有效区域
        # 使用 build_valid_mask 来获取有效数据
        mask1 = build_valid_mask(embed1, embed_rs1 if embed_rs1 else None)
        mask2 = build_valid_mask(embed2, embed_rs2 if embed_rs2 else None)

        # 只比较两者都有效的部分
        min_batch = min(embed1.shape[0], embed2.shape[0])
        min_seq = min(embed1.shape[1], embed2.shape[1])

        for b in range(min_batch):
            # 确定这个 batch 的有效 seq 长度
            seq1_valid = embed1.shape[1]
            seq2_valid = embed2.shape[1]

            # 从 rank_shapes 推断有效长度
            if embed_rs1 and b < len(embed_rs1):
                seq1_valid = embed_rs1[b][1] if len(embed_rs1[b]) > 1 else embed1.shape[1]
            if embed_rs2 and b < len(embed_rs2):
                seq2_valid = embed_rs2[b][1] if len(embed_rs2[b]) > 1 else embed2.shape[1]

            valid_seq = min(seq1_valid, seq2_valid)
            e1 = embed1[b, :valid_seq, :].flatten().float()
            e2 = embed2[b, :valid_seq, :].flatten().float()

            if e1.numel() > 0 and e2.numel() > 0:
                cos = F.cosine_similarity(e1.unsqueeze(0), e2.unsqueeze(0)).item()
                max_diff = (e1 - e2).abs().max().item()
                status = "✓" if cos > 0.999 else "✗"
                print(f"   Batch {b} (first {valid_seq} tokens): cos_sim={cos:.6f}, max_diff={max_diff:.6e} {status}")

    # 3. 比较 Gate 输出
    print("\n3. Gate (Router) 输出比较:")
    print("-" * 70)

    # 找到所有 gate keys
    gate_pattern = re.compile(r"layers\.(\d+)\.mlp\.gate$")
    gate_keys1 = {int(m.group(1)): k for k in fwd1.keys() for m in [gate_pattern.search(k)] if m}
    gate_keys2 = {int(m.group(1)): k for k in fwd2.keys() for m in [gate_pattern.search(k)] if m}

    common_layers = sorted(set(gate_keys1.keys()) & set(gate_keys2.keys()))

    if layer_filter:
        filt = re.compile(layer_filter)
        common_layers = [i for i in common_layers if filt.search(f"layers.{i}")]

    gate_results = []
    for layer_idx in common_layers[:10]:  # 只显示前 10 层
        k1, k2 = gate_keys1[layer_idx], gate_keys2[layer_idx]
        g1, g2 = fwd1[k1], fwd2[k2]

        # Gate 输出是 [num_tokens, num_experts]
        # 需要对齐有效 tokens

        # 获取 rank_shapes
        grs1 = rs1.get(k1, [])
        grs2 = rs2.get(k2, [])

        # 计算有效 token 数
        if grs1:
            # grs1 形如 [[48, 128], [40, 128]] 表示两次调用
            # 每次调用的第一维是 token 数
            valid_tokens1 = sum(rs[0] for rs in grs1)
        else:
            valid_tokens1 = g1.shape[0]

        if grs2:
            valid_tokens2 = sum(rs[0] for rs in grs2)
        else:
            valid_tokens2 = g2.shape[0]

        # 比较有效区域
        valid_tokens = min(valid_tokens1, valid_tokens2)
        g1_valid = g1[:valid_tokens].flatten().float()
        g2_valid = g2[:valid_tokens].flatten().float()

        if g1_valid.numel() > 0 and g2_valid.numel() > 0:
            cos = F.cosine_similarity(g1_valid.unsqueeze(0), g2_valid.unsqueeze(0)).item()
            max_diff = (g1_valid - g2_valid).abs().max().item()
            status = "✓" if cos > 0.99 else "✗"
            print(f"   Layer {layer_idx:2d}: shape1={list(g1.shape)}, shape2={list(g2.shape)}, "
                  f"valid={valid_tokens}, cos={cos:.6f}, max_diff={max_diff:.2e} {status}")
            gate_results.append((layer_idx, cos, max_diff))

    # 4. 比较 MLP hidden_states 输出
    print("\n4. MLP Hidden States 输出比较:")
    print("-" * 70)

    mlp_pattern = re.compile(r"layers\.(\d+)\.mlp$")
    mlp_keys1 = {int(m.group(1)): k for k in fwd1.keys() for m in [mlp_pattern.search(k)] if m}
    mlp_keys2 = {int(m.group(1)): k for k in fwd2.keys() for m in [mlp_pattern.search(k)] if m}

    common_mlp_layers = sorted(set(mlp_keys1.keys()) & set(mlp_keys2.keys()))

    if layer_filter:
        common_mlp_layers = [i for i in common_mlp_layers if filt.search(f"layers.{i}")]

    mlp_results = []
    for layer_idx in common_mlp_layers[:10]:  # 只显示前 10 层
        k1, k2 = mlp_keys1[layer_idx], mlp_keys2[layer_idx]
        m1, m2 = fwd1[k1], fwd2[k2]

        # MLP 输出可能是 tuple (hidden_states, router_logits) 或单个 tensor
        if isinstance(m1, (tuple, list)):
            m1 = m1[0]
        if isinstance(m2, (tuple, list)):
            m2 = m2[0]

        # 获取 rank_shapes
        mrs1 = rs1.get(k1, [])
        mrs2 = rs2.get(k2, [])

        # 使用 mask 来识别有效区域
        mask1 = build_valid_mask(m1, mrs1 if mrs1 else None)
        mask2 = build_valid_mask(m2, mrs2 if mrs2 else None)

        # 只比较两者都有效的部分
        min_shape = [min(m1.shape[d], m2.shape[d]) for d in range(m1.dim())]
        slices = tuple(slice(0, s) for s in min_shape)

        m1_sub = m1[slices]
        m2_sub = m2[slices]
        mask1_sub = mask1[slices]
        mask2_sub = mask2[slices]

        # 只保留两者都有效的数据
        combined_mask = mask1_sub & mask2_sub
        m1_valid = m1_sub[combined_mask].float()
        m2_valid = m2_sub[combined_mask].float()

        if m1_valid.numel() > 0 and m2_valid.numel() > 0:
            cos = F.cosine_similarity(m1_valid.unsqueeze(0), m2_valid.unsqueeze(0)).item()
            max_diff = (m1_valid - m2_valid).abs().max().item()
            mean_diff = (m1_valid - m2_valid).abs().mean().item()
            status = "✓" if cos > 0.99 else "✗"
            print(f"   Layer {layer_idx:2d}: shape1={list(m1.shape)}, shape2={list(m2.shape)}, "
                  f"valid_numel={m1_valid.numel()}, cos={cos:.6f}, max_diff={max_diff:.2e} {status}")
            mlp_results.append((layer_idx, cos, max_diff, mean_diff))
        else:
            print(f"   Layer {layer_idx:2d}: No valid overlapping data")

    # 5. 总结
    print("\n" + "=" * 70)
    print("总结")
    print("=" * 70)

    if gate_results:
        gate_cos = [r[1] for r in gate_results]
        print(f"\nGate 输出:")
        print(f"   cos_sim: min={min(gate_cos):.6f}, max={max(gate_cos):.6f}, mean={sum(gate_cos)/len(gate_cos):.6f}")
        bad_gates = [r for r in gate_results if r[1] < 0.99]
        if bad_gates:
            print(f"   ⚠ cos_sim < 0.99 的层: {', '.join([f'L{r[0]}({r[1]:.4f})' for r in bad_gates])}")

    if mlp_results:
        mlp_cos = [r[1] for r in mlp_results]
        print(f"\nMLP Hidden States:")
        print(f"   cos_sim: min={min(mlp_cos):.6f}, max={max(mlp_cos):.6f}, mean={sum(mlp_cos)/len(mlp_cos):.6f}")
        bad_mlps = [r for r in mlp_results if r[1] < 0.99]
        if bad_mlps:
            print(f"   ⚠ cos_sim < 0.99 的层: {', '.join([f'L{r[0]}({r[1]:.4f})' for r in bad_mlps])}")

    # 检测可能的问题
    print("\n诊断信息:")
    if embed_rs1 and embed_rs2:
        # 检测分布式配置差异
        if len(embed_rs1) != len(embed_rs2):
            print(f"   ⚠ Rank 数量不同: 文件1={len(embed_rs1)} ranks, 文件2={len(embed_rs2)} ranks")

        # 检测 token 数量差异
        tokens1 = sum(rs[0] * rs[1] if len(rs) >= 2 else rs[0] for rs in embed_rs1)
        tokens2 = sum(rs[0] * rs[1] if len(rs) >= 2 else rs[0] for rs in embed_rs2)
        if tokens1 != tokens2:
            print(f"   ⚠ 有效 token 数不同: 文件1={tokens1}, 文件2={tokens2}, 差异={abs(tokens1-tokens2)}")

        # 检测截断问题
        for i, (rs_1, rs_2) in enumerate(zip(embed_rs1, embed_rs2)):
            if len(rs_1) >= 2 and len(rs_2) >= 2:
                if rs_1[1] != rs_2[1]:
                    print(f"   ⚠ Rank {i} seq_len 不同: 文件1={rs_1[1]}, 文件2={rs_2[1]}")

    print("=" * 70)

    return {
        "gate_results": gate_results,
        "mlp_results": mlp_results,
    }


def compare_lora_params(
    data1: Dict[str, Any],
    data2: Dict[str, Any],
    threshold: float = 1e-5,
    layer_filter: Optional[str] = None,
) -> Dict:
    """
    比较两个文件中保存的 LoRA 参数值

    支持三种情况:
    1. 两个文件都是 PEFT LoRA (experts.X.proj.lora_A/B)
    2. 两个文件都是 KT LoRA (lora_params.xxx_lora_a/b)
    3. 一个 PEFT + 一个 KT (将 KT 合并张量按 expert 拆开对比)
    """
    import torch.nn.functional as F

    lora1 = data1.get("lora_params", {})
    lora2 = data2.get("lora_params", {})

    if not lora1 and not lora2:
        print("\n两个文件都没有 lora_params 数据")
        return {"passed": 0, "failed": 0}

    # Normalize keys
    lora1 = {get_canonical_name(k): v for k, v in lora1.items()}
    lora2 = {get_canonical_name(k): v for k, v in lora2.items()}

    keys1 = set(lora1.keys())
    keys2 = set(lora2.keys())

    if layer_filter:
        pattern = re.compile(layer_filter)
        keys1 = {k for k in keys1 if pattern.search(k)}
        keys2 = {k for k in keys2 if pattern.search(k)}

    # Detect types
    has_experts_1 = any("mlp.experts." in k and "lora_" in k for k in keys1)
    has_lora_params_1 = any("lora_params." in k for k in keys1)
    has_experts_2 = any("mlp.experts." in k and "lora_" in k for k in keys2)
    has_lora_params_2 = any("lora_params." in k for k in keys2)

    print(f"\n{'=' * 70}")
    print("LoRA 参数值比较")
    print("=" * 70)
    print(f"文件1: {len(keys1)} lora params (PEFT={has_experts_1}, KT={has_lora_params_1})")
    print(f"文件2: {len(keys2)} lora params (PEFT={has_experts_2}, KT={has_lora_params_2})")

    # Case: one PEFT + one KT -> cross-compare
    if (has_experts_1 and has_lora_params_2) or (has_experts_2 and has_lora_params_1):
        if has_experts_1 and has_lora_params_2:
            ref_lora, kt_lora = lora1, lora2
            print("模式: PEFT(文件1) vs KT(文件2) — 按 expert 拆开对比")
        else:
            ref_lora, kt_lora = lora2, lora1
            print("模式: PEFT(文件2) vs KT(文件1) — 按 expert 拆开对比")

        kt_to_peft = {
            "gate_lora_a": ("gate_proj", "lora_A"),
            "gate_lora_b": ("gate_proj", "lora_B"),
            "up_lora_a": ("up_proj", "lora_A"),
            "up_lora_b": ("up_proj", "lora_B"),
            "down_lora_a": ("down_proj", "lora_A"),
            "down_lora_b": ("down_proj", "lora_B"),
        }

        layer_pat = re.compile(r"layers\.(\d+)\.mlp\.lora_params\.")
        layer_indices = sorted(set(int(m.group(1)) for k in kt_lora for m in [layer_pat.search(k)] if m))

        if layer_filter:
            filt = re.compile(layer_filter)
            layer_indices = [i for i in layer_indices if filt.search(f"layers.{i}")]

        print("-" * 70)
        all_results = []

        for layer_idx in layer_indices:
            layer_results = []
            for kt_suffix, (proj, lora_type) in kt_to_peft.items():
                kt_key = f"base_model.model.model.layers.{layer_idx}.mlp.lora_params.{kt_suffix}"
                if kt_key not in kt_lora:
                    print(f"缺少 KT key: {kt_key}")
                    continue   
                kt_val = kt_lora[kt_key].float()
                num_experts = kt_val.shape[0]

                for exp_idx in range(num_experts):
                    peft_key = (
                        f"base_model.model.model.layers.{layer_idx}.mlp.experts.{exp_idx}"
                        f".{proj}.{lora_type}.default.weight"
                    )
                    if peft_key not in ref_lora:
                        print(f"缺少 PEFT key: {peft_key}")
                        continue
                    ref_v = ref_lora[peft_key].float()
                    kt_e = kt_val[exp_idx]

                    rn = ref_v.norm().item()
                    kn = kt_e.norm().item()
                    if rn < 1e-12 and kn < 1e-12:
                        cos = 1.0
                    else:
                        cos = F.cosine_similarity(
                            ref_v.flatten().unsqueeze(0), kt_e.flatten().unsqueeze(0)
                        ).item()
                    diff_max = (ref_v - kt_e).abs().max().item()
                    layer_results.append((kt_suffix, exp_idx, rn, kn, cos, diff_max))

            if not layer_results:
                continue

            cos_vals = [r[4] for r in layer_results]
            diff_vals = [r[5] for r in layer_results]
            print(f"\nLayer {layer_idx:>2}: compared={len(layer_results)}, "
                  f"cos=[{min(cos_vals):.4f}, {max(cos_vals):.4f}] mean={sum(cos_vals)/len(cos_vals):.4f}, "
                  f"abs_max_diff=[{min(diff_vals):.4e}, {max(diff_vals):.4e}]")

            # Show anomalies (cos < 0.99 or diff > threshold)
            anomalies = [(s, e, rn, kn, c, d) for s, e, rn, kn, c, d in layer_results
                         if d > threshold]
            for s, e, rn, kn, c, d in anomalies:
                print(f"    {s:<15} expert={e:<3} ref_norm={rn:.6e} kt_norm={kn:.6e} cos={c:.6f} abs_max={d:.4e}")

            all_results.extend(layer_results)

        if all_results:
            cos_all = [r[4] for r in all_results]
            diff_all = [r[5] for r in all_results]
            n_bad = sum(1 for r in all_results if r[4] < 0.99 or r[5] > threshold)
            print(f"\nMLP Expert LoRA 全局汇总: total={len(all_results)}, anomalies={n_bad}")
            print(f"  cos: min={min(cos_all):.6f}, mean={sum(cos_all)/len(cos_all):.6f}, max={max(cos_all):.6f}")
            print(f"  abs_max_diff: min={min(diff_all):.4e}, max={max(diff_all):.4e}")

        # ============================================================
        # 比较 Attention LoRA (self_attn 模块的 q/k/v/o_proj LoRA)
        # 两个文件都应该有相同格式的 attention LoRA，直接比较
        # ============================================================
        print(f"\n{'=' * 70}")
        print("Attention LoRA 参数值比较")
        print("=" * 70)

        # Find attention LoRA keys (exclude mlp/experts)
        attn_pattern = re.compile(r"layers\.(\d+)\.self_attn\.([\w_]+)\.lora_([AB])\.default\.weight")

        def get_attn_lora_keys(lora_dict):
            keys = {}
            for k in lora_dict.keys():
                m = attn_pattern.search(k)
                if m:
                    layer_idx, proj, lora_type = m.groups()
                    key_tuple = (int(layer_idx), proj, lora_type)
                    keys[key_tuple] = k
            return keys

        attn_keys1 = get_attn_lora_keys(lora1)
        attn_keys2 = get_attn_lora_keys(lora2)

        common_attn_keys = set(attn_keys1.keys()) & set(attn_keys2.keys())

        if layer_filter:
            filt = re.compile(layer_filter)
            common_attn_keys = {k for k in common_attn_keys if filt.search(f"layers.{k[0]}")}

        print(f"文件1 attention LoRA params: {len(attn_keys1)}")
        print(f"文件2 attention LoRA params: {len(attn_keys2)}")
        print(f"共同 params: {len(common_attn_keys)}")
        print("-" * 70)

        attn_results = []
        if common_attn_keys:
            # Group by layer
            attn_layer_indices = sorted(set(k[0] for k in common_attn_keys))

            for layer_idx in attn_layer_indices:
                layer_results = []
                layer_keys = sorted([k for k in common_attn_keys if k[0] == layer_idx], key=lambda x: (x[1], x[2]))

                for key_tuple in layer_keys:
                    k1 = attn_keys1[key_tuple]
                    k2 = attn_keys2[key_tuple]

                    v1 = lora1[k1].float()
                    v2 = lora2[k2].float()

                    # Handle shape mismatch
                    if v1.shape != v2.shape:
                        print(f"  Shape mismatch: layer={layer_idx}, {key_tuple[1]}.lora_{key_tuple[2]}: "
                              f"{list(v1.shape)} vs {list(v2.shape)}")
                        continue

                    n1 = v1.norm().item()
                    n2 = v2.norm().item()

                    if n1 < 1e-12 and n2 < 1e-12:
                        cos = 1.0
                        diff_max = 0.0
                    else:
                        cos = F.cosine_similarity(v1.flatten().unsqueeze(0), v2.flatten().unsqueeze(0)).item()
                        diff_max = (v1 - v2).abs().max().item()

                    layer_results.append((key_tuple[1], key_tuple[2], n1, n2, cos, diff_max))

                if layer_results:
                    cos_vals = [r[4] for r in layer_results]
                    diff_vals = [r[5] for r in layer_results]
                    n_anomaly = sum(1 for r in layer_results if r[4] < 0.99 or r[5] > threshold)

                    print(f"\nLayer {layer_idx:>2}: compared={len(layer_results)}, anomalies={n_anomaly}, "
                          f"cos=[{min(cos_vals):.4f}, {max(cos_vals):.4f}], "
                          f"abs_max_diff=[{min(diff_vals):.4e}, {max(diff_vals):.4e}]")

                    # Show details for each proj
                    for proj, lora_type, n1, n2, cos, diff_max in layer_results:
                        status = "✓" if cos > 0.99 and diff_max <= threshold else "✗"
                        print(f"    [{status}] {proj}.lora_{lora_type}: "
                              f"norm1={n1:.4e}, norm2={n2:.4e}, cos={cos:.6f}, abs_max={diff_max:.4e}")

                    attn_results.extend(layer_results)

            # Attention global summary
            if attn_results:
                attn_cos_all = [r[4] for r in attn_results]
                attn_diff_all = [r[5] for r in attn_results]
                attn_n_bad = sum(1 for r in attn_results if r[4] < 0.99 or r[5] > threshold)
                print(f"\nAttention LoRA 全局汇总: total={len(attn_results)}, anomalies={attn_n_bad}")
                print(f"  cos: min={min(attn_cos_all):.6f}, mean={sum(attn_cos_all)/len(attn_cos_all):.6f}, max={max(attn_cos_all):.6f}")
                print(f"  abs_max_diff: min={min(attn_diff_all):.4e}, max={max(attn_diff_all):.4e}")

        # ============================================================
        # 比较 Gate (Router) LoRA
        # ============================================================
        print(f"\n{'=' * 70}")
        print("Gate (Router) LoRA 参数值比较")
        print("=" * 70)

        gate_pattern = re.compile(r"layers\.(\d+)\.mlp\.gate\.lora_([AB])\.default\.weight")

        def get_gate_lora_keys(lora_dict):
            keys = {}
            for k in lora_dict.keys():
                m = gate_pattern.search(k)
                if m:
                    layer_idx, lora_type = m.groups()
                    key_tuple = (int(layer_idx), lora_type)
                    keys[key_tuple] = k
            return keys

        gate_keys1 = get_gate_lora_keys(lora1)
        gate_keys2 = get_gate_lora_keys(lora2)

        common_gate_keys = set(gate_keys1.keys()) & set(gate_keys2.keys())

        if layer_filter:
            filt = re.compile(layer_filter)
            common_gate_keys = {k for k in common_gate_keys if filt.search(f"layers.{k[0]}")}

        print(f"文件1 gate LoRA params: {len(gate_keys1)}")
        print(f"文件2 gate LoRA params: {len(gate_keys2)}")
        print(f"共同 params: {len(common_gate_keys)}")
        print("-" * 70)

        gate_results = []
        if common_gate_keys:
            gate_layer_indices = sorted(set(k[0] for k in common_gate_keys))

            for layer_idx in gate_layer_indices:
                layer_keys = sorted([k for k in common_gate_keys if k[0] == layer_idx], key=lambda x: x[1])

                for key_tuple in layer_keys:
                    k1 = gate_keys1[key_tuple]
                    k2 = gate_keys2[key_tuple]

                    v1 = lora1[k1].float()
                    v2 = lora2[k2].float()

                    if v1.shape != v2.shape:
                        print(f"  Shape mismatch: layer={layer_idx}, gate.lora_{key_tuple[1]}: "
                              f"{list(v1.shape)} vs {list(v2.shape)}")
                        continue

                    n1 = v1.norm().item()
                    n2 = v2.norm().item()

                    if n1 < 1e-12 and n2 < 1e-12:
                        cos = 1.0
                        diff_max = 0.0
                    else:
                        cos = F.cosine_similarity(v1.flatten().unsqueeze(0), v2.flatten().unsqueeze(0)).item()
                        diff_max = (v1 - v2).abs().max().item()

                    status = "✓" if cos > 0.99 and diff_max <= threshold else "✗"
                    print(f"  [{status}] Layer {layer_idx:>2} gate.lora_{key_tuple[1]}: "
                          f"norm1={n1:.4e}, norm2={n2:.4e}, cos={cos:.6f}, abs_max={diff_max:.4e}")

                    gate_results.append((layer_idx, key_tuple[1], n1, n2, cos, diff_max))

            if gate_results:
                gate_cos_all = [r[4] for r in gate_results]
                gate_diff_all = [r[5] for r in gate_results]
                gate_n_bad = sum(1 for r in gate_results if r[4] < 0.99 or r[5] > threshold)
                print(f"\nGate LoRA 全局汇总: total={len(gate_results)}, anomalies={gate_n_bad}")
                print(f"  cos: min={min(gate_cos_all):.6f}, mean={sum(gate_cos_all)/len(gate_cos_all):.6f}, max={max(gate_cos_all):.6f}")
                print(f"  abs_max_diff: min={min(gate_diff_all):.4e}, max={max(gate_diff_all):.4e}")

        print("=" * 70)

        # Calculate total results
        total_passed = 0
        total_failed = 0
        if all_results:
            total_passed += sum(1 for r in all_results if r[4] >= 0.99 and r[5] <= threshold)
            total_failed += sum(1 for r in all_results if r[4] < 0.99 or r[5] > threshold)
        if attn_results:
            total_passed += sum(1 for r in attn_results if r[4] >= 0.99 and r[5] <= threshold)
            total_failed += sum(1 for r in attn_results if r[4] < 0.99 or r[5] > threshold)
        if gate_results:
            total_passed += sum(1 for r in gate_results if r[4] >= 0.99 and r[5] <= threshold)
            total_failed += sum(1 for r in gate_results if r[4] < 0.99 or r[5] > threshold)

        return {"passed": total_passed, "failed": total_failed}

    # Case: same type -> direct compare on common keys
    common = sorted(keys1 & keys2)
    only1 = keys1 - keys2
    only2 = keys2 - keys1

    print(f"共同: {len(common)}")
    if only1:
        s = sorted(only1)
        print(f"仅在文件1 ({len(only1)}): {s[:5]}{'...' if len(only1) > 5 else ''}")
    if only2:
        s = sorted(only2)
        print(f"仅在文件2 ({len(only2)}): {s[:5]}{'...' if len(only2) > 5 else ''}")
    print("-" * 70)

    passed = 0
    failed = 0
    results_list = []

    for name in common:
        v1 = lora1[name]
        v2 = lora2[name]
        metrics = compute_tensor_metrics(v1, v2)
        is_passed = not metrics.get("shape_mismatch", False) and metrics.get("abs_max", float("inf")) <= threshold
        metrics["passed"] = is_passed

        if is_passed:
            passed += 1
        else:
            failed += 1
            results_list.append((name, metrics))

    # Sort by cosine similarity ascending (worst first)
    results_list.sort(key=lambda x: x[1].get("cosine_similarity", 1.0))

    for name, metrics in results_list[:30]:
        cos = metrics.get("cosine_similarity", 0)
        abs_max = metrics.get("abs_max", 0)
        abs_mean = metrics.get("abs_mean", 0)
        print(f"  ✗ {name}: abs_max={abs_max:.4e} abs_mean={abs_mean:.4e} cos={cos:.6f}")

    if len(results_list) > 30:
        print(f"  ... 还有 {len(results_list) - 30} 项未显示")

    print(f"\n总结: 通过 {passed}, 失败 {failed}")

    # Group by lora type
    type_cos = {}
    for name in common:
        metrics = compute_tensor_metrics(lora1[name], lora2[name])
        cos = metrics.get("cosine_similarity", None)
        if cos is None:
            continue
        for tag in ["lora_A", "lora_B", "lora_params"]:
            if tag in name:
                type_cos.setdefault(tag, []).append(cos)
                break
        else:
            type_cos.setdefault("other", []).append(cos)

    if type_cos:
        print(f"\n按参数类型汇总 cosine similarity:")
        for tag in ["lora_A", "lora_B", "lora_params", "other"]:
            vals = type_cos.get(tag, [])
            if not vals:
                continue
            print(f"  [{tag}] count={len(vals)} mean={sum(vals)/len(vals):.6f} min={min(vals):.6f} max={max(vals):.6f}")

    print("=" * 70)
    return {"passed": passed, "failed": failed}


def main():
    parser = argparse.ArgumentParser(description="比对模型每层输出（支持 forward/backward）")
    parser.add_argument("--file1", "-f1", required=True, help="第一个层输出文件路径")
    parser.add_argument("--file2", "-f2", required=True, help="第二个层输出文件路径")
    parser.add_argument("--threshold", "-t", type=float, default=1e-5,
                        help="误差阈值（默认 1e-5）")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="输出比对结果的JSON文件路径")
    parser.add_argument("--detailed", "-d", action="store_true",
                        help="显示详细的逐层比对信息")
    parser.add_argument("--mode", "-m", choices=["all", "forward", "backward"],
                        default="all", help="比对模式（默认 all）")
    parser.add_argument("--layer-filter", "-l", type=str, default=None,
                        help="层名过滤正则表达式")
    parser.add_argument("--show-structure", "-s", action="store_true",
                        help="显示模型结构信息")
    parser.add_argument("--forward-order", "-F", action="store_true",
                        help="按 forward 顺序比较对应模块的输入输出（不深入 linear 层）")
    parser.add_argument("--backward-order", "-B", action="store_true",
                        help="按 backward 顺序比较对应模块的梯度（不深入 linear 层）")
    parser.add_argument("--param-grads", "-G", action="store_true",
                        help="比较参数梯度（param.grad）")
    parser.add_argument("--kt-lora-grads", "-K", action="store_true",
                        help="比较 KT expert LoRA 梯度（将 KT 合并张量拆开与 PEFT 逐 expert 对比）")
    parser.add_argument("--attn-lora-grads", "-A", action="store_true",
                        help="比较 Attention LoRA 梯度（self_attn 模块的 q/kv/o_proj LoRA）")
    parser.add_argument("--lora-params", "-L", action="store_true",
                        help="比较 LoRA 参数值（支持 PEFT vs PEFT, KT vs KT, PEFT vs KT）")
    parser.add_argument("--mlp-mismatch", "-M", action="store_true",
                        help="比较 MLP 输出（处理 token 数量不匹配的情况，如 CPU vs KT 分布式）")

    args = parser.parse_args()

    # 加载层输出
    try:
        data1 = load_layer_outputs(args.file1)
        data2 = load_layer_outputs(args.file2)
    except FileNotFoundError as e:
        print(f"错误: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"加载文件时出错: {e}")
        sys.exit(1)

    # 显示模型结构
    if args.show_structure:
        print_model_structure(data1, "文件1")
        print_model_structure(data2, "文件2")

    # 按 forward 顺序比较对应模块的输入输出
    if args.forward_order:
        compare_outputs_forward_order(data1, data2, threshold=args.threshold, detailed=args.detailed)

    # 按 backward 顺序比较对应模块的梯度
    if args.backward_order:
        compare_backward_order(data1, data2, threshold=args.threshold)

    # 比较参数梯度
    if args.param_grads:
        compare_param_grads(data1, data2, threshold=args.threshold, layer_filter=args.layer_filter)

    # 比较 KT expert LoRA 梯度
    if args.kt_lora_grads:
        compare_kt_lora_grads(data1, data2, layer_filter=args.layer_filter)

    # 比较 Attention LoRA 梯度
    if args.attn_lora_grads:
        compare_attn_lora_grads(data1, data2, layer_filter=args.layer_filter)

    # 比较 LoRA 参数值
    if args.lora_params:
        compare_lora_params(data1, data2, threshold=args.threshold, layer_filter=args.layer_filter)

    # 比较 MLP 输出（处理 token 数量不匹配）
    if args.mlp_mismatch:
        compare_mlp_with_token_mismatch(data1, data2, threshold=args.threshold, layer_filter=args.layer_filter)

    # 如果使用了 forward/backward order 或 param-grads 模式，就不再进行常规比对
    if args.forward_order or args.backward_order or args.param_grads or args.kt_lora_grads or args.attn_lora_grads or args.lora_params or args.mlp_mismatch:
        sys.exit(0)

    # 执行比对
    results = compare_layer_outputs(
        data1, data2,
        threshold=args.threshold,
        mode=args.mode,
        layer_filter=args.layer_filter,
        detailed=args.detailed
    )

    # 打印摘要
    print_summary(results)

    # 保存结果
    if args.output:
        save_results(results, args.output)

    # 返回退出码
    if results["summary"]["total_failed"] > 0:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
