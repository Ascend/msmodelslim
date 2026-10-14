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
from msmodelslim.model.common.utils import _get_expert_range


def get_ln_fuse_map(config, num_hidden_layers=None):
    ln_linear_map = {}
    expert_start, expert_end = _get_expert_range(config)
    if num_hidden_layers is None:
        num_hidden_layers = config.num_hidden_layers + 1
    # +1 for mtp
    # pylint: disable=duplicate-code
    for layer_idx in range(num_hidden_layers):
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
        if layer_idx < config.first_k_dense_replace:
            ln_linear_map[f"model.layers.{layer_idx}.post_attention_layernorm"] = [
                f"model.layers.{layer_idx}.mlp.gate_proj",
                f"model.layers.{layer_idx}.mlp.up_proj",
            ]
        else:
            # routed experts
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
    # MTP
    ln_linear_map[(f"model.layers.{num_hidden_layers - 1}.enorm", f"model.layers.{num_hidden_layers - 1}.hnorm")] = [
        f"model.layers.{num_hidden_layers - 1}.eh_proj",
    ]
    ln_linear_map[f"model.layers.{num_hidden_layers - 1}.shared_head.norm"] = [
        f"model.layers.{num_hidden_layers - 1}.shared_head.head",
    ]
    return ln_linear_map

    # pylint: disable=duplicate-code


def get_rotate_map(config, block_size, num_hidden_layers=None):
    if num_hidden_layers is None:
        num_hidden_layers = config.num_hidden_layers + 1
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
    expert_start, expert_end = _get_expert_range(config)
    # pre run
    # pylint: disable=duplicate-code
    left_rot = {}
    right_rot = {}
    right_rot["model.embed_tokens"] = rot
    pre_run = QuaRotInterface.RotatePair(left_rot=left_rot, right_rot=right_rot)
    rot_pairs = {}
    # rot
    left_rot = {}
    right_rot = {}
    right_rot["lm_head"] = rot
    for layer_idx in range(num_hidden_layers):
        right_rot[f"model.layers.{layer_idx}.self_attn.q_a_proj"] = rot
        right_rot[f"model.layers.{layer_idx}.self_attn.kv_a_proj_with_mqa"] = rot
        left_rot[f"model.layers.{layer_idx}.self_attn.o_proj"] = rot
        if layer_idx < config.first_k_dense_replace:
            right_rot[f"model.layers.{layer_idx}.mlp.gate_proj"] = rot
            right_rot[f"model.layers.{layer_idx}.mlp.up_proj"] = rot
            left_rot[f"model.layers.{layer_idx}.mlp.down_proj"] = rot
        else:
            for i in range(expert_start, expert_end):
                right_rot[f"model.layers.{layer_idx}.mlp.experts.{i}.gate_proj"] = rot
                right_rot[f"model.layers.{layer_idx}.mlp.experts.{i}.up_proj"] = rot
                left_rot[f"model.layers.{layer_idx}.mlp.experts.{i}.down_proj"] = rot
            right_rot[f"model.layers.{layer_idx}.mlp.shared_experts.gate_proj"] = rot
            right_rot[f"model.layers.{layer_idx}.mlp.shared_experts.up_proj"] = rot
            left_rot[f"model.layers.{layer_idx}.mlp.shared_experts.down_proj"] = rot
            right_rot[f"model.layers.{layer_idx}.mlp.gate"] = rot
    right_rot[f"model.layers.{num_hidden_layers - 1}.embed_tokens"] = rot
    # concat output of enorm and hnorm
    right_rot[f"model.layers.{num_hidden_layers - 1}.eh_proj"] = torch.block_diag(*[rot] * 2)
    left_rot[f"model.layers.{num_hidden_layers - 1}.eh_proj"] = rot
    right_rot[f"model.layers.{num_hidden_layers - 1}.shared_head.head"] = rot
    rot_pairs['rot'] = QuaRotInterface.RotatePair(left_rot=left_rot, right_rot=right_rot)
    # rot_b_proj
    left_rot_b_proj = {}
    right_rot_b_proj = {}
    for layer_idx in range(num_hidden_layers):
        left_rot_b_proj[f"model.layers.{layer_idx}.self_attn.q_a_proj"] = rot_b_proj
        right_rot_b_proj[f"model.layers.{layer_idx}.self_attn.q_b_proj"] = rot_b_proj
    rot_pairs["rot_b_proj"] = QuaRotInterface.RotatePair(left_rot=left_rot_b_proj, right_rot=right_rot_b_proj)
    # rot_uv
    left_rot_uv = {}
    right_rot_uv = {}
    for layer_idx in range(num_hidden_layers):
        # split output of kv_b_proj
        left_rot_uv[f"model.layers.{layer_idx}.self_attn.kv_b_proj"] = [
            torch.eye(config.qk_nope_head_dim, dtype=rot_uv.dtype, device=rot_uv.device),
            rot_uv,
        ]
        right_rot_uv[f"model.layers.{layer_idx}.self_attn.o_proj"] = rot_uv
    rot_pairs["rot_uv"] = QuaRotInterface.RotatePair(left_rot=left_rot_uv, right_rot=right_rot_uv)
    # rot_kv_b_proj
    left_rot_kv_b_proj = {}
    right_rot_kv_b_proj = {}
    for layer_idx in range(num_hidden_layers):
        # split output of kv_a_proj_with_mqa
        left_rot_kv_b_proj[f"model.layers.{layer_idx}.self_attn.kv_a_proj_with_mqa"] = [
            rot_kv_b_proj,
            torch.eye(config.qk_rope_head_dim, dtype=rot_kv_b_proj.dtype, device=rot_kv_b_proj.device),
        ]
        right_rot_kv_b_proj[f"model.layers.{layer_idx}.self_attn.kv_b_proj"] = rot_kv_b_proj
    rot_pairs["rot_kv_b_proj"] = QuaRotInterface.RotatePair(left_rot=left_rot_kv_b_proj, right_rot=right_rot_kv_b_proj)
    rotate_matrix = {'rot': rot, 'rot_b_proj': rot_b_proj, 'rot_uv': rot_uv, 'rot_kv_b_proj': rot_kv_b_proj}
    return pre_run, rot_pairs, rotate_matrix
