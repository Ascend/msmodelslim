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
# pylint: disable=logging-fstring-interpolation,relative-beyond-top-level,unnecessary-comprehension,useless-return

import typing
from dataclasses import dataclass

import sys  # noqa: F401
import os  # noqa: F401
import math  # noqa: F401
import ctypes
import gc
import random
import torch
import numpy as np
import tqdm


from ascend_utils import ResListToRelease
from msmodelslim.pytorch.llm_ptq.accelerate_adapter import PrepareWeight
from msmodelslim.pytorch.llm_ptq.accelerate_adapter import replace_device_align_hook_if_needed
from msmodelslim import logger

from .hadamard_utils import random_hadamard_matrix

GLOBAL_DTYPE = torch.float32


@dataclass
class RotConfig:
    """create_rot函数的配置类"""

    size: int
    group_size: int = -1
    mode: str = "hadamard"
    rot_step: int = 1
    eye_step: tuple = (-1,)
    device: str = "npu"


@dataclass
class CreateRotationMatrixConfig:
    """create_rotation_matrix函数的配置类"""

    group_size: int = -1
    rotate_kv: bool = False
    device: str = "npu"
    r2_mode: str = "hadamard"
    r2_step: int = 1
    eye_step: tuple = (-1,)


class GraphOpt:
    @staticmethod
    def set_module(model, submodule_key, module):
        tokens = submodule_key.split('.')
        sub_tokens = tokens[:-1]
        cur_mod = model
        for s in sub_tokens:
            cur_mod = getattr(cur_mod, s)
        setattr(cur_mod, tokens[-1], module)


def has_mtp(model):
    return hasattr(model, 'mtp_layer')


def get_mtp(model):
    return model.mtp_layer, model.mtp_decoder


def set_bias_to_ones(norm):
    shape, dtype, device = norm.weight.data.shape, norm.weight.data.dtype, norm.weight.data.device
    norm.weight.data = torch.ones(shape).to(device=device, dtype=dtype)


def fuse_ln_linear(layernorm: torch.nn.Module, linear_layers: typing.Iterable[torch.nn.Linear]) -> None:
    for linear in linear_layers:
        linear_dtype = linear.weight.dtype
        current_weight = linear.weight.data.to(dtype=GLOBAL_DTYPE)

        linear.weight.data = (current_weight * layernorm.weight.to(dtype=GLOBAL_DTYPE)).to(linear_dtype)
        del current_weight
    return


def fuse_layer_norms_module(module, config_hidden_size, fuse_kv_ln=False, idx=None, model=None):
    modules = [
        module.self_attn.q_a_proj,
        module.self_attn.kv_a_proj_with_mqa,
        module.self_attn.kv_b_proj,
        module.self_attn.q_b_proj,
    ]
    modules += [module.input_layernorm, module.self_attn.q_a_layernorm]
    prepare_list = [PrepareWeight(ws, post_force=True) for ws in modules]
    with ResListToRelease(*prepare_list):
        if hasattr(module.self_attn, "q_proj"):
            fuse_ln_linear(module.input_layernorm, [module.self_attn.q_proj, module.self_attn.kv_a_proj_with_mqa])
        else:
            fuse_ln_linear(module.input_layernorm, [module.self_attn.q_a_proj, module.self_attn.kv_a_proj_with_mqa])
            fuse_ln_linear(module.self_attn.q_a_layernorm, [module.self_attn.q_b_proj])

        if fuse_kv_ln:
            fuse_ln_linear(module.self_attn.kv_a_layernorm, [module.self_attn.kv_b_proj])
            ln_weight = module.self_attn.kv_a_layernorm.weight.data
            shape, dtype, device = ln_weight.shape, ln_weight.dtype, ln_weight.device
            module.self_attn.kv_a_layernorm.weight.data = torch.ones(shape).to(device=device, dtype=dtype)

        set_bias_to_ones(module.self_attn.q_a_layernorm)

        set_bias_to_ones(module.input_layernorm)

    if hasattr(module.mlp, "experts"):
        modules = [module.mlp.experts[i].up_proj for i in range(len(module.mlp.experts))]
        modules += [module.mlp.experts[i].gate_proj for i in range(len(module.mlp.experts))]
        modules += [module.mlp.shared_experts.up_proj, module.mlp.shared_experts.gate_proj, module.mlp.gate]
    else:
        modules = [module.mlp.up_proj, module.mlp.gate_proj]
    modules += [module.post_attention_layernorm]
    prepare_list = [PrepareWeight(ws, post_force=True) for ws in modules]

    with ResListToRelease(*prepare_list):
        if hasattr(module.mlp, "experts"):
            for expert in module.mlp.experts:
                fuse_ln_linear(module.post_attention_layernorm, [expert.up_proj, expert.gate_proj])
            fuse_ln_linear(
                module.post_attention_layernorm,
                [module.mlp.shared_experts.up_proj, module.mlp.shared_experts.gate_proj, module.mlp.gate],
            )
        else:
            fuse_ln_linear(module.post_attention_layernorm, [module.mlp.up_proj, module.mlp.gate_proj])

        set_bias_to_ones(module.post_attention_layernorm)

    gc.collect()
    torch.npu.empty_cache()

    return


def fuse_layer_norms_head(model):
    m_norm = model.model.norm
    m_linear = [model.lm_head]

    prepare_list = [PrepareWeight(m_norm, post_force=True, post_recurse=False)]
    prepare_list += [PrepareWeight(m_linear, post_force=True, post_recurse=False)]
    with ResListToRelease(*prepare_list):
        fuse_ln_linear(m_norm, m_linear)
        set_bias_to_ones(m_norm)

    return


def fuse_layer_shared_head(model):
    m_norm = model.mtp_layer.shared_head.norm
    m_linear = [model.mtp_layer.shared_head.head]

    prepare_list = [PrepareWeight(m_norm, post_force=True, post_recurse=False)]
    prepare_list += [PrepareWeight(m_linear, post_force=True, post_recurse=False)]
    with ResListToRelease(*prepare_list):
        fuse_ln_linear(m_norm, m_linear)
        set_bias_to_ones(m_norm)

    return


def fuse_ln_linear_eh(e_norm, h_norm: torch.nn.Module, linear_layers: typing.Iterable[torch.nn.Linear]) -> None:
    for linear in linear_layers:
        linear_dtype = linear.weight.dtype
        current_weight = linear.weight.data.to(dtype=GLOBAL_DTYPE)

        # Get norm weight and handle dimension mismatch
        e_w = e_norm.weight.data.to(dtype=GLOBAL_DTYPE)
        h_w = h_norm.weight.data.to(dtype=GLOBAL_DTYPE)
        w = torch.concat([e_w, h_w], dim=0)

        linear.weight.data = (current_weight * w.to(dtype=GLOBAL_DTYPE)).to(linear_dtype)
        del current_weight
    return


def fuse_layer_eh_norm(model):
    enorm = model.mtp_layer.enorm
    hnorm = model.mtp_layer.hnorm
    m_linear = [model.mtp_layer.eh_proj]
    prepare_list = [PrepareWeight(hnorm, post_force=True, post_recurse=True)]
    prepare_list += [PrepareWeight(m_linear, post_force=True, post_recurse=True)]
    with ResListToRelease(*prepare_list):
        fuse_ln_linear_eh(enorm, hnorm, m_linear)
        set_bias_to_ones(enorm)
        set_bias_to_ones(hnorm)
    return


def fuse_layer_norms(model, fuse_kv_ln=False):
    layers = [layer for layer in model.model.layers]
    # Fuse the linear operations in Layernorm into the adjacent linear blocks.
    for i, layer in tqdm.tqdm(enumerate(layers)):
        fuse_layer_norms_module(layer, model.config.hidden_size, fuse_kv_ln, idx=i, model=model)
        try:
            libc = ctypes.CDLL("libc.so.6")
            libc.malloc_trim(0)
        except Exception as e:
            logger.warning(f"Failed to call malloc_trim: {e}, please notice your system memory usage")
    fuse_layer_norms_head(model)
    if has_mtp(model):
        fuse_layer_eh_norm(model)
        fuse_layer_norms_module(model.mtp_decoder, model.config.hidden_size, fuse_kv_ln, idx=None, model=model)
        fuse_layer_shared_head(model)
    return


def rotate_embeddings(model, rot: torch.Tensor) -> None:
    for weight in [model.model.embed_tokens]:
        dtype = weight.weight.data.dtype
        device = weight.weight.device
        weight_data = weight.weight.data.to(device=device, dtype=GLOBAL_DTYPE)
        weight.weight.data = torch.matmul(weight_data, rot.to(dtype=weight_data.dtype)).to(dtype=dtype)
        del weight_data
    return


def rotate_attention_inputs(layer, rot, q_b_proj_rot=None, kv_b_proj_rot=None) -> None:
    if hasattr(layer.self_attn, "q_proj"):
        modules = [layer.self_attn.q_proj, layer.self_attn.kv_a_proj_with_mqa]
    else:
        modules = [layer.self_attn.q_a_proj, layer.self_attn.kv_a_proj_with_mqa]

    prepare_list = [PrepareWeight(ws, True, True) for ws in modules]
    with ResListToRelease(*prepare_list):
        for weight in modules:
            dtype = weight.weight.dtype
            device = weight.weight.device

            weight_data = weight.weight.to(device=device, dtype=GLOBAL_DTYPE)
            weight.weight.data = torch.matmul(weight_data, rot).to(device=device, dtype=dtype)

            del weight_data

    if hasattr(layer.self_attn, "q_b_proj"):
        modules = [layer.self_attn.q_a_proj, layer.self_attn.q_b_proj]
        prepare_list = [PrepareWeight(ws, True, True) for ws in modules]
        with ResListToRelease(*prepare_list):
            weight = layer.self_attn.q_b_proj
            device = weight.weight.device
            weight_data = weight.weight.to(device=device, dtype=GLOBAL_DTYPE)
            weight.weight.data = torch.matmul(weight_data, q_b_proj_rot).to(device=device, dtype=dtype)

            weight = layer.self_attn.q_a_proj
            weight_data = weight.weight.to(device=device, dtype=GLOBAL_DTYPE)

            weight.weight.data = torch.matmul(q_b_proj_rot.T, weight_data).to(device=device, dtype=dtype)
            del weight_data

    if hasattr(layer.self_attn, "kv_b_proj") and kv_b_proj_rot is not None:
        modules = [layer.self_attn.kv_b_proj, layer.self_attn.kv_a_proj_with_mqa]
        prepare_list = [PrepareWeight(ws, True, True) for ws in modules]
        with ResListToRelease(*prepare_list):
            weight = layer.self_attn.kv_a_proj_with_mqa
            dtype, device = weight.weight.dtype, weight.weight.device

            rot_blocks = (kv_b_proj_rot, torch.diag(torch.ones(layer.self_attn.qk_rope_head_dim, device=device)))
            kv_b_proj_rot_reshaped = torch.block_diag(*rot_blocks)

            weight_data = weight.weight.to(device=device, dtype=GLOBAL_DTYPE)
            weight.weight.data = torch.matmul(kv_b_proj_rot_reshaped.T, weight_data).to(device=device, dtype=dtype)
            del weight_data

            weight = layer.self_attn.kv_b_proj
            dtype, device = weight.weight.dtype, weight.weight.device
            weight_data = weight.weight.to(device=device, dtype=GLOBAL_DTYPE)
            weight.weight.data = torch.matmul(weight_data, kv_b_proj_rot).to(device=device, dtype=dtype)

            del weight_data
    return


def rotate_attention_output(layer, rot) -> None:
    with PrepareWeight(layer.self_attn.o_proj, True, True):
        weight = layer.self_attn.o_proj

        dtype = weight.weight.data.dtype
        device = weight.weight.device

        weight_data = weight.weight.data.to(device=device, dtype=GLOBAL_DTYPE)

        weight.weight.data = torch.matmul(rot.T, weight_data).to(device=device, dtype=dtype)

        if weight.bias is not None:
            b = weight.bias.data.to(device=device, dtype=GLOBAL_DTYPE)
            weight.bias.data = torch.matmul(rot.T, b).to(device=device, dtype=dtype)

        del weight_data
        return


def rotate_oproj_input(layer, rot) -> None:
    with PrepareWeight(layer.self_attn.o_proj, True, True):
        weight = layer.self_attn.o_proj

        dtype = weight.weight.data.dtype
        device = weight.weight.device

        rot = torch.block_diag(*[rot] * layer.self_attn.num_heads)

        weight_data = weight.weight.to(device=device, dtype=GLOBAL_DTYPE)
        weight.weight.data = torch.matmul(weight_data, rot).to(device=device, dtype=dtype)

        del weight_data
        return


def rotate_uv_output(layer, rot) -> None:
    with PrepareWeight(layer.self_attn.kv_b_proj, True, True):
        weight = layer.self_attn.kv_b_proj

        dtype = weight.weight.data.dtype
        device = weight.weight.device

        # Create expanded rot
        rot_transformed = torch.cat(
            [
                torch.cat(
                    [
                        torch.eye(layer.self_attn.qk_nope_head_dim, device=device),
                        torch.zeros(layer.self_attn.qk_nope_head_dim, layer.self_attn.v_head_dim, device=device),
                    ],
                    dim=1,
                ),
                torch.cat(
                    [torch.zeros(layer.self_attn.v_head_dim, layer.self_attn.qk_nope_head_dim, device=device), rot],
                    dim=1,
                ),
            ],
            dim=0,
        )

        rot_transformed = torch.block_diag(*[rot_transformed] * layer.self_attn.num_heads)

        weight_data = weight.weight.data.to(device=device, dtype=GLOBAL_DTYPE)
        weight.weight.data = torch.matmul(rot_transformed.T, weight_data).to(device=device, dtype=dtype)

        if weight.bias is not None:
            b = weight.bias.data.to(device=device, dtype=GLOBAL_DTYPE)
            weight.bias.data = torch.matmul(rot_transformed.T, b).to(device=device, dtype=dtype)

        del weight_data
        return


def rotate_mlp_input(layer, rot):
    mlp_inputs = []
    if hasattr(layer.mlp, "experts"):
        for expert in layer.mlp.experts:
            mlp_inputs += [expert.up_proj, expert.gate_proj]
        mlp_inputs += [layer.mlp.shared_experts.up_proj, layer.mlp.shared_experts.gate_proj]
        mlp_inputs += [layer.mlp.gate]
    else:
        mlp_inputs = [layer.mlp.up_proj, layer.mlp.gate_proj]

    prepare_list = [PrepareWeight(ws, True, True) for ws in mlp_inputs]
    with ResListToRelease(*prepare_list):
        for weight in mlp_inputs:
            dtype = weight.weight.dtype
            device = weight.weight.device
            weight_data = weight.weight.data.to(device=device, dtype=GLOBAL_DTYPE)

            weight.weight.data = torch.matmul(weight_data, rot).to(device=device, dtype=dtype)
            del weight_data
        return


def rotate_mlp_output(layer, rot):
    cur_dtype = torch.float
    modules = []
    if hasattr(layer.mlp, "experts"):
        for expert in layer.mlp.experts:
            modules += [expert.down_proj]

        modules += [layer.mlp.shared_experts.down_proj]
    else:
        modules = [layer.mlp.down_proj]

    prepare_list = [PrepareWeight(ws, True, True) for ws in modules]
    with ResListToRelease(*prepare_list):
        for weight in modules:
            dtype = weight.weight.data.dtype
            device = weight.weight.device

            weight_data = weight.weight.data.to(device=device, dtype=GLOBAL_DTYPE)
            weight.weight.data = torch.matmul(rot.T, weight_data).to(dtype=dtype)

            if weight.bias is not None:
                b = weight.bias.data.to(device=device, dtype=GLOBAL_DTYPE)
                weight.bias.data = torch.matmul(rot.T, b.to(cur_dtype)).to(dtype=dtype)

            del weight_data
        return


def rotate_head(model, rot: torch.Tensor) -> None:
    """
    Rotate the head.
    """

    weight = model.lm_head
    dtype = weight.weight.data.dtype
    device = weight.weight.device

    weight_data = weight.weight.data.to(device=device, dtype=GLOBAL_DTYPE)
    weight.weight.data = torch.matmul(weight_data, rot.to(dtype=weight_data.dtype)).to(dtype=dtype)
    del weight_data
    return


def random_hadamard_matrix_block(in_channel, stride, eye_step, device):
    """生成对角线上放置Hadamard矩阵的in_channel维矩阵"""
    # 生成stride维的Hadamard矩阵
    h_stride = random_hadamard_matrix(stride, device)

    # 创建in_channel维的全零张量，并确保它在指定的设备上
    h_in_channel = torch.zeros((in_channel, in_channel), device=device, dtype=torch.float32)
    # 将stride维的Hadamard矩阵的副本放置在in_channel维矩阵的对角线上
    num_blocks = in_channel // stride
    for i in range(num_blocks):
        start_idx = i * stride
        end_idx = start_idx + stride
        if i in eye_step:
            h_in_channel[start_idx:end_idx, start_idx:end_idx] = torch.eye(stride).to(
                device=device, dtype=torch.float32
            )
        else:
            h_in_channel[start_idx:end_idx, start_idx:end_idx] = h_stride

    return h_in_channel


def create_rot(config: RotConfig):
    """
    创建旋转矩阵

    Args:
        config (RotConfig): 包含所有参数的配置类

    Returns:
        torch.Tensor: 旋转矩阵
    """
    block_size = 32  # noqa: F841
    shift = 16

    size = config.size
    group_size = config.group_size
    mode = config.mode
    rot_step = config.rot_step
    eye_step = config.eye_step
    device = config.device

    if group_size == -1:
        transformation_dim = size
    else:
        transformation_dim = group_size

    if mode == "hadamard":
        rot = random_hadamard_matrix(transformation_dim, device)
    elif mode == 'block_hadamard':
        rot = random_hadamard_matrix_block(size, 32, eye_step, device)
    elif mode == 'block_hadamard_shifted':
        rot = random_hadamard_matrix_block(size, 32, eye_step, device)
        identity = torch.eye(size, dtype=torch.float32)
        p = torch.cat((identity[:, -shift:], identity[:, :-shift]), dim=1).to(rot.device)
        if rot_step == 1:
            rot = rot @ p @ rot
        elif rot_step == 2:
            rot = rot @ p @ rot @ p.T @ rot.T
        elif rot_step == 3:
            rot = rot @ p @ rot @ p.T @ rot.T @ p @ rot @ p.T @ rot.T
        else:
            raise ValueError("rot_step must be 1, 2, or 3")

    if group_size != -1:
        rot = rot.repeat(size // group_size, 1, 1)
        rot = torch.block_diag(*rot)

    return rot


@torch.no_grad()
def rotate_module(module, rot, rot_att_uv, q_b_proj_rot, kv_b_proj_rot):
    rotate_attention_inputs(module, rot, q_b_proj_rot=q_b_proj_rot, kv_b_proj_rot=kv_b_proj_rot)

    rotate_uv_output(module, rot_att_uv)
    rotate_oproj_input(module, rot_att_uv)
    rotate_attention_output(module, rot)
    rotate_mlp_input(module, rot)
    rotate_mlp_output(module, rot)

    del rot
    gc.collect()
    torch.cuda.empty_cache()
    return


@torch.no_grad()
def create_rotation_matrix(model, layer, config: CreateRotationMatrixConfig):
    group_size = config.group_size
    rotate_kv = config.rotate_kv
    device = config.device
    r2_mode = config.r2_mode
    r2_step = config.r2_step
    eye_step = config.eye_step

    v_head_dim = model.config.v_head_dim
    rot_att_uv_config = RotConfig(size=v_head_dim, group_size=-1, mode=r2_mode, device=device)
    rot_att_uv = create_rot(rot_att_uv_config)

    q_b_proj_rot = None
    if hasattr(layer.self_attn, "q_b_proj"):
        q_b_proj_config = RotConfig(
            size=layer.self_attn.q_lora_rank,
            group_size=group_size,
            mode=r2_mode,
            rot_step=r2_step,
            eye_step=eye_step,
            device=device,
        )
        q_b_proj_rot = create_rot(q_b_proj_config)

    kv_b_proj_rot = None
    if hasattr(layer.self_attn, "kv_b_proj") and rotate_kv:
        kv_b_proj_config = RotConfig(
            size=layer.self_attn.kv_lora_rank,
            group_size=group_size,
            mode=r2_mode,
            rot_step=r2_step,
            eye_step=eye_step,
            device=device,
        )
        kv_b_proj_rot = create_rot(kv_b_proj_config)

    return rot_att_uv, q_b_proj_rot, kv_b_proj_rot


def right_rotate(model, rot: torch.Tensor) -> None:
    weight = model
    dtype = weight.weight.data.dtype
    device = weight.weight.device

    weight_data = weight.weight.data.to(device=device, dtype=GLOBAL_DTYPE)
    weight.weight.data = torch.matmul(weight_data, rot.to(dtype=weight_data.dtype)).to(dtype=dtype)
    del weight_data
    return


def left_rotate(model, rot: torch.Tensor) -> None:
    weight = model
    dtype = weight.weight.data.dtype
    device = weight.weight.device

    weight_data = weight.weight.data.to(device=device, dtype=GLOBAL_DTYPE)
    weight.weight.data = torch.matmul(rot.T.to(dtype=weight_data.dtype), weight_data).to(dtype=dtype)
    del weight_data
    return


@torch.no_grad()
def rotate_model(model, group_size=-1, rotate_kv=False, args=None):
    device = model.lm_head.weight.device
    # R1
    r1_config = RotConfig(
        size=model.config.hidden_size,
        group_size=group_size,
        mode=args.r1_mode,
        rot_step=args.r1_step,
        eye_step=args.eye_step,
        device=device,
    )
    rot = create_rot(r1_config)

    layers = [layer for layer in model.model.layers]
    # R2
    rotation_matrix_config = CreateRotationMatrixConfig(
        group_size=group_size,
        rotate_kv=rotate_kv,
        device=device,
        r2_mode=args.r2_mode,
        r2_step=args.r2_step,
        eye_step=args.eye_step,
    )
    rot_att_uv, q_b_proj_rot, kv_b_proj_rot = create_rotation_matrix(model, layers[0], rotation_matrix_config)
    rotate_embeddings(model, rot)
    for idx, _ in enumerate(tqdm.tqdm(layers, unit="layer", desc="Rotating")):
        with PrepareWeight(layers[idx], True, True):
            rotate_module(layers[idx], rot, rot_att_uv, q_b_proj_rot, kv_b_proj_rot)
            try:
                libc = ctypes.CDLL("libc.so.6")
                libc.malloc_trim(0)
            except Exception as e:
                logger.warning(f"Failed to call malloc_trim: {e}, please notice your system memory usage")
    rotate_head(model, rot)
    if has_mtp(model):
        mtp, mtp_decoder = get_mtp(model)
        left_rotate(mtp.eh_proj, rot)
        right_rotate(mtp.eh_proj, torch.block_diag(*[rot] * 2))
        rotate_module(mtp_decoder, rot, rot_att_uv, q_b_proj_rot, kv_b_proj_rot)
        right_rotate(mtp.shared_head.head, rot)
        right_rotate(mtp.embed_tokens, rot)
    return rot, rot_att_uv, q_b_proj_rot, kv_b_proj_rot


def rot_model(model, seed=42):
    # 设置随机数种子以确保旋转矩阵可复现
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # 设置确定性模式（可选，可能影响性能）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    from collections import namedtuple

    rotate_kv = False
    args = namedtuple('args', ['r1_mode', 'r1_step', 'r2_mode', 'r2_step', 'eye_step'])
    rot_args = args('block_hadamard', 1, 'block_hadamard_shifted', 1, [0, 1])

    replace_device_align_hook_if_needed(model)
    fuse_layer_norms(model, fuse_kv_ln=rotate_kv)
    rot, rot_att_uv, q_b_proj_rot, kv_b_proj_rot = rotate_model(
        model, group_size=-1, rotate_kv=rotate_kv, args=rot_args
    )
    return rot, rot_att_uv, q_b_proj_rot, kv_b_proj_rot
