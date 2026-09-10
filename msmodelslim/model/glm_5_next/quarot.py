#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
-------------------------------------------------------------------------
This file is part of the MindStudio project.
Copyright (c) 2026 Huawei Technologies Co.,Ltd.

MindStudio is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:

         http://license.coscl.org.cn/MulanPSL2

THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details.
-------------------------------------------------------------------------
"""

import torch
from ..interface_hub import QuaRotInterface


def _get_full_expert_range(config):
    """获取完整的 expert 范围（0 ~ n_routed_experts），不按 rank 切分。
    用于配置生成阶段，确保各 rank 生成的配置一致，避免 DTS 校验失败。
    """
    if hasattr(config, 'n_routed_experts') and isinstance(config.n_routed_experts, int):
        return 0, config.n_routed_experts
    return 0, 0


def get_ln_fuse_map(config, num_hidden_layers=None):
    ln_linear_map = {}
    if num_hidden_layers is None:
        num_hidden_layers = config.num_hidden_layers

    expert_start, expert_end = _get_full_expert_range(config)
    layer_types = config.layer_types

    for layer_idx in range(num_hidden_layers):
        # MTP 层（idx >= len(layer_types）使用 DSA
        if layer_idx < len(layer_types):
            layer_type = layer_types[layer_idx]
        else:
            layer_type = "deepseek_sparse_attention"

        if layer_type == "deepseek_sparse_attention":
            # DSA/MLA attention: q_a_proj, kv_a_proj_with_mqa
            ln_linear_map[f"model.layers.{layer_idx}.input_layernorm"] = [
                f"model.layers.{layer_idx}.self_attn.q_a_proj",
                f"model.layers.{layer_idx}.self_attn.kv_a_proj_with_mqa",
            ]
            ln_linear_map[f"model.layers.{layer_idx}.self_attn.q_a_layernorm"] = [
                f"model.layers.{layer_idx}.self_attn.q_b_proj"
            ]
            ln_linear_map[f"model.layers.{layer_idx}.self_attn.kv_a_layernorm"] = [
                f"model.layers.{layer_idx}.self_attn.kv_b_proj"
            ]
        else:
            # Linear attention: q_proj, k_proj, v_proj 及 KDA 三个门控投影。
            # b_proj / forget_gate.f_a_proj / g_a_proj 与 q/k/v 同吃 input_layernorm
            # 输出（参考 kimi_k3 input_layernorm_targets），必须一并 fold γ，
            # 否则导出权重缺失 norm 折叠（sglang 端表现为 KDA gate 数值错误）。
            ln_linear_map[f"model.layers.{layer_idx}.input_layernorm"] = [
                f"model.layers.{layer_idx}.self_attn.q_proj",
                f"model.layers.{layer_idx}.self_attn.k_proj",
                f"model.layers.{layer_idx}.self_attn.v_proj",
                f"model.layers.{layer_idx}.self_attn.b_proj",
                f"model.layers.{layer_idx}.self_attn.forget_gate.f_a_proj",
                f"model.layers.{layer_idx}.self_attn.g_a_proj",
            ]

        # FFN post-attention norm targets
        # MTP 层（idx >= len(mlp_layer_types）按 MoE（sparse）处理
        if config.mlp_layer_types and layer_idx < len(config.mlp_layer_types):
            mlp_layer_type = config.mlp_layer_types[layer_idx]
        else:
            mlp_layer_type = "sparse"
        if mlp_layer_type == "dense":
            ln_linear_map[f"model.layers.{layer_idx}.post_attention_layernorm"] = [
                f"model.layers.{layer_idx}.mlp.gate_proj",
                f"model.layers.{layer_idx}.mlp.up_proj",
            ]
        else:
            # MoE layer: per-expert nn.Linear (after unstack)
            ln_linear_map[f"model.layers.{layer_idx}.post_attention_layernorm"] = [
                f"model.layers.{layer_idx}.mlp.experts.{i}.{proj}"
                for proj in ["gate_proj", "up_proj"]
                for i in range(expert_start, expert_end)
            ]
            # shared experts
            ln_linear_map[f"model.layers.{layer_idx}.post_attention_layernorm"] += [
                f"model.layers.{layer_idx}.mlp.shared_experts.{proj}" for proj in ["gate_proj", "up_proj"]
            ]
            # expert gate
            ln_linear_map[f"model.layers.{layer_idx}.post_attention_layernorm"] += [
                f"model.layers.{layer_idx}.mlp.gate"
            ]

    ln_linear_map["model.norm"] = ['lm_head']
    return ln_linear_map


def get_rotate_map(config, block_size, num_hidden_layers=None):
    if num_hidden_layers is None:
        num_hidden_layers = config.num_hidden_layers
    rot = QuaRotInterface.get_rotate_command(
        size=config.hidden_size,
        mode=QuaRotInterface.QuaRotMode.HADAMARD,
        block_size=block_size,
    )
    rot_b_proj = QuaRotInterface.get_rotate_command(
        size=config.q_lora_rank,
        mode=QuaRotInterface.QuaRotMode.BLOCK_HADAMARD_SHIFTED,
        block_size=block_size,
    )
    rot_uv = QuaRotInterface.get_rotate_command(
        size=config.v_head_dim,
        mode=QuaRotInterface.QuaRotMode.HADAMARD,
        block_size=block_size,
    )
    rot_kv_b_proj = QuaRotInterface.get_rotate_command(
        size=config.kv_lora_rank,
        mode=QuaRotInterface.QuaRotMode.HADAMARD,
        block_size=block_size,
    )

    # pre run - embed_tokens rotation
    left_rot = {}
    right_rot = {}
    right_rot["model.embed_tokens"] = rot
    pre_run = QuaRotInterface.RotatePair(left_rot=left_rot, right_rot=right_rot)

    rot_pairs = {}
    layer_types = config.layer_types
    expert_start, expert_end = _get_full_expert_range(config)

    # rot
    left_rot = {}
    right_rot = {}
    right_rot["lm_head"] = rot

    for layer_idx in range(num_hidden_layers):
        layer_type = layer_types[layer_idx]

        if layer_type == "deepseek_sparse_attention":
            right_rot[f"model.layers.{layer_idx}.self_attn.q_a_proj"] = rot
            right_rot[f"model.layers.{layer_idx}.self_attn.kv_a_proj_with_mqa"] = rot
            left_rot[f"model.layers.{layer_idx}.self_attn.o_proj"] = rot
        else:
            right_rot[f"model.layers.{layer_idx}.self_attn.q_proj"] = rot
            right_rot[f"model.layers.{layer_idx}.self_attn.k_proj"] = rot
            right_rot[f"model.layers.{layer_idx}.self_attn.v_proj"] = rot
            left_rot[f"model.layers.{layer_idx}.self_attn.o_proj"] = rot
            # Linear attention 的 gate 分支输入与 q/k/v 同源（旋转域），需配套旋转：
            # - f_a_proj / g_a_proj: 输入侧右旋 R，吸收旋转使 head_dim 输出回到原始域
            # - b_proj: 输入侧右旋 R（输出 num_heads 维度，逐头使用）
            # - f_b_proj / g_b_proj: 输入来自 f_a/g_a（已是原始域），且 gate 需与
            #   core_attn_out（原始域）逐元素对齐，故不可旋转
            right_rot[f"model.layers.{layer_idx}.self_attn.forget_gate.f_a_proj"] = rot
            right_rot[f"model.layers.{layer_idx}.self_attn.g_a_proj"] = rot
            right_rot[f"model.layers.{layer_idx}.self_attn.b_proj"] = rot

        # mHC fn 从旋转残差流取输入，右旋 block_diag(R×hc_mult) 每流独立消 R
        right_rot[f"model.layers.{layer_idx}.attn_hc.fn"] = rot
        right_rot[f"model.layers.{layer_idx}.ffn_hc.fn"] = rot

        # MLP rotation (same for all layers)
        mlp_layer_type = config.mlp_layer_types[layer_idx] if config.mlp_layer_types else "dense"
        if mlp_layer_type == "dense":
            right_rot[f"model.layers.{layer_idx}.mlp.gate_proj"] = rot
            right_rot[f"model.layers.{layer_idx}.mlp.up_proj"] = rot
            left_rot[f"model.layers.{layer_idx}.mlp.down_proj"] = rot
        else:
            # MoE: per-expert nn.Linear (after unstack)
            for i in range(expert_start, expert_end):
                right_rot[f"model.layers.{layer_idx}.mlp.experts.{i}.gate_proj"] = rot
                right_rot[f"model.layers.{layer_idx}.mlp.experts.{i}.up_proj"] = rot
                left_rot[f"model.layers.{layer_idx}.mlp.experts.{i}.down_proj"] = rot
            right_rot[f"model.layers.{layer_idx}.mlp.shared_experts.gate_proj"] = rot
            right_rot[f"model.layers.{layer_idx}.mlp.shared_experts.up_proj"] = rot
            left_rot[f"model.layers.{layer_idx}.mlp.shared_experts.down_proj"] = rot
            right_rot[f"model.layers.{layer_idx}.mlp.gate"] = rot

    rot_pairs['rot'] = QuaRotInterface.RotatePair(left_rot=left_rot, right_rot=right_rot)

    # rot_b_proj (only for DSA layers with q_a_proj/q_b_proj)
    left_rot_b_proj = {}
    right_rot_b_proj = {}
    for layer_idx in range(num_hidden_layers):
        if layer_types[layer_idx] == "deepseek_sparse_attention":
            left_rot_b_proj[f"model.layers.{layer_idx}.self_attn.q_a_proj"] = rot_b_proj
            right_rot_b_proj[f"model.layers.{layer_idx}.self_attn.q_b_proj"] = rot_b_proj
    if left_rot_b_proj and right_rot_b_proj:
        rot_pairs["rot_b_proj"] = QuaRotInterface.RotatePair(left_rot=left_rot_b_proj, right_rot=right_rot_b_proj)

    # rot_uv (only for DSA layers with kv_b_proj)
    left_rot_uv = {}
    right_rot_uv = {}
    for layer_idx in range(num_hidden_layers):
        if layer_types[layer_idx] == "deepseek_sparse_attention":
            left_rot_uv[f"model.layers.{layer_idx}.self_attn.kv_b_proj"] = [
                torch.eye(config.qk_nope_head_dim, dtype=rot_uv.dtype, device=rot_uv.device),
                rot_uv,
            ]
            right_rot_uv[f"model.layers.{layer_idx}.self_attn.o_proj"] = rot_uv
    if left_rot_uv and right_rot_uv:
        rot_pairs["rot_uv"] = QuaRotInterface.RotatePair(left_rot=left_rot_uv, right_rot=right_rot_uv)

    # rot_kv_b_proj (only for DSA layers)
    left_rot_kv_b_proj = {}
    right_rot_kv_b_proj = {}
    for layer_idx in range(num_hidden_layers):
        if layer_types[layer_idx] == "deepseek_sparse_attention":
            left_rot_kv_b_proj[f"model.layers.{layer_idx}.self_attn.kv_a_proj_with_mqa"] = [
                rot_kv_b_proj,
                torch.eye(config.qk_rope_head_dim, dtype=rot_kv_b_proj.dtype, device=rot_kv_b_proj.device),
            ]
            right_rot_kv_b_proj[f"model.layers.{layer_idx}.self_attn.kv_b_proj"] = rot_kv_b_proj
    if left_rot_kv_b_proj and right_rot_kv_b_proj:
        rot_pairs["rot_kv_b_proj"] = QuaRotInterface.RotatePair(
            left_rot=left_rot_kv_b_proj, right_rot=right_rot_kv_b_proj
        )

    rotate_matrix = {
        'rot': rot,
        'rot_b_proj': rot_b_proj,
        'rot_uv': rot_uv,
        'rot_kv_b_proj': rot_kv_b_proj,
    }
    return pre_run, rot_pairs, rotate_matrix
