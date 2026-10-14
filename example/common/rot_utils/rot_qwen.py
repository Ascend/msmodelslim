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

import ctypes
import gc
import os
import sys
import typing
from dataclasses import dataclass

import torch
import tqdm

current_directory = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.abspath(os.path.join(current_directory, '..', ".."))
sys.path.append(parent_directory)

from ascend_utils import ResListToRelease  # noqa: E402
from msmodelslim.pytorch.llm_ptq.accelerate_adapter import PrepareWeight  # noqa: E402
from msmodelslim.pytorch.llm_ptq.accelerate_adapter import replace_device_align_hook_if_needed  # noqa: E402
from msmodelslim import logger  # noqa: E402

from .hadamard_utils import random_hadamard_matrix  # noqa: E402

GLOBAL_DTYPE = torch.float32


@dataclass
class RotConfig:
    """create_Q函数的配置类"""

    size: int
    group_size: int = -1
    mode: str = "hadamard"
    rot_step: int = 1
    eye_step: tuple = (-1,)
    device: str = "cuda"


@dataclass
class CreateRotationMatrixConfig:
    """create_rotation_matrix函数的配置类"""

    group_size: int = -1
    rotate_kv: bool = False
    device: str = "cuda"
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


def set_bias_to_ones(norm):
    """设置 LayerNorm 的权重为全1"""
    shape, dtype, device = norm.weight.data.shape, norm.weight.data.dtype, norm.weight.data.device
    norm.weight.data = torch.ones(shape).to(device=device, dtype=dtype)


def fuse_ln_linear(layernorm: torch.nn.Module, linear_layers: typing.Iterable[torch.nn.Linear]) -> None:
    """
    This function modifies the weights and biases of the provided linear layers
    by incorporating the scaling of the given LayerNorm module.

    Parameters:
    - layernorm (torch.nn.Module): The LayerNorm module whose parameters will be fused.
    - linear_layers (typing.Iterable[torch.nn.Linear]): A list of linear layers adjacent to the LayerNorm module
      that will be updated with the fused parameters.

    Returns:
    - None: The function modifies the linear layers in place.
    """

    for linear in linear_layers:
        linear_dtype = linear.weight.dtype
        current_weight = linear.weight.data.to(dtype=GLOBAL_DTYPE)
        # Calculating new weight and bias
        if hasattr(layernorm, 'bias'):
            if linear.bias is None:
                linear.bias = torch.nn.Parameter(
                    torch.zeros(linear.out_features, dtype=linear_dtype, device=linear.weight.device)
                )
            ln_bias = linear.bias.data.to(dtype=GLOBAL_DTYPE)
            linear.bias.data = ln_bias + torch.matmul(current_weight, ln_bias)
            linear.bias.data.to(linear_dtype)

        linear.weight.data = (current_weight * layernorm.weight.to(dtype=GLOBAL_DTYPE)).to(linear_dtype)
        del current_weight
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


def fuse_layer_norms_module(module, config_hidden_size, fuse_kv_ln=False, idx=None, model=None):
    """
    Fuse the input layernorms into the linear layers
    """
    modules = [module.self_attn.q_proj, module.self_attn.k_proj, module.self_attn.v_proj]
    modules += [module.input_layernorm]
    prepare_list = [PrepareWeight(ws, post_force=True) for ws in modules]
    with ResListToRelease(*prepare_list):
        fuse_ln_linear(
            module.input_layernorm, [module.self_attn.q_proj, module.self_attn.k_proj, module.self_attn.v_proj]
        )

        set_bias_to_ones(module.input_layernorm)

    if hasattr(module.mlp, "experts"):
        modules = []
        for expert in module.mlp.experts:
            modules += [expert.up_proj, expert.gate_proj]
        modules += [module.mlp.gate]
    else:
        modules = [module.mlp.up_proj, module.mlp.gate_proj]
    modules += [module.post_attention_layernorm]
    prepare_list = [PrepareWeight(ws, post_force=True) for ws in modules]

    with ResListToRelease(*prepare_list):
        if hasattr(module.mlp, "experts"):
            for expert in module.mlp.experts:
                fuse_ln_linear(module.post_attention_layernorm, [expert.up_proj, expert.gate_proj])
            fuse_ln_linear(module.post_attention_layernorm, [module.mlp.gate])
        else:
            fuse_ln_linear(module.post_attention_layernorm, [module.mlp.up_proj, module.mlp.gate_proj])

        set_bias_to_ones(module.post_attention_layernorm)

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
    return


def rotate_embeddings(model, rot: torch.Tensor) -> None:
    """
    Rotates the embedding weights of the model using a given rotation matrix.

    Parameters:
    - model: The model whose embedding weights need to be rotated.
    - rot (torch.Tensor): The rotation matrix to apply to the embeddings.

    Returns:
    - None: The function modifies the embedding weights in place.
    """
    for weight in [model.model.embed_tokens]:
        dtype = weight.weight.data.dtype
        device = weight.weight.device
        weight_data = weight.weight.data.to(device=device, dtype=GLOBAL_DTYPE)
        weight.weight.data = torch.matmul(weight_data, rot.to(dtype=weight_data.dtype)).to(dtype=dtype)
        del weight_data
    return


def rotate_attention_inputs(layer, rot, q_b_proj_rot=None, kv_b_proj_rot=None) -> None:
    """
    Rotate the W q_proj, W q_a_proj, W q_b_proj and W kv_a_proj_with_mqa matrices of the self-attention layer.
    """
    modules = [layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj]

    prepare_list = [PrepareWeight(ws, post_force=True) for ws in modules]
    with ResListToRelease(*prepare_list):
        for weight in modules:
            dtype = weight.weight.dtype
            device = weight.weight.device

            weight_data = weight.weight.to(device=device, dtype=GLOBAL_DTYPE)
            weight.weight.data = torch.matmul(weight_data, rot).to(device=device, dtype=dtype)

            del weight_data
    return


def rotate_attention_output(layer, rot) -> None:
    """
    Rotate output matrix of the self-attention layer.
    """
    with PrepareWeight(layer.self_attn.o_proj, post_force=True):
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
    """
    Rotate the W o_proj, of the self-attention layer.
    self.o_proj = nn.Linear(
            self.num_heads * self.v_head_dim,
            self.hidden_size,
            bias=config.attention_bias,
        )
    """
    with PrepareWeight(layer.self_attn.o_proj, post_force=True):
        weight = layer.self_attn.o_proj

        dtype = weight.weight.data.dtype
        device = weight.weight.device

        rot = torch.block_diag(*[rot] * layer.self_attn.config.num_attention_heads)

        weight_data = weight.weight.to(device=device, dtype=GLOBAL_DTYPE)
        weight.weight.data = torch.matmul(weight_data, rot).to(device=device, dtype=dtype)

        del weight_data
        return


def rotate_uv_output(layer, rot) -> None:
    with PrepareWeight(layer.self_attn.v_proj, post_force=True):
        weight = layer.self_attn.v_proj

        dtype = weight.weight.data.dtype
        device = weight.weight.device

        rot_transformed = torch.block_diag(*[rot] * layer.self_attn.config.num_key_value_heads)

        weight_data = weight.weight.data.to(device=device, dtype=GLOBAL_DTYPE)
        weight.weight.data = torch.matmul(rot_transformed.T, weight_data).to(device=device, dtype=dtype)

        if weight.bias is not None:
            b = weight.bias.data.to(device=device, dtype=GLOBAL_DTYPE)
            weight.bias.data = torch.matmul(rot_transformed.T, b).to(device=device, dtype=dtype)

        del weight_data
        return


def rotate_mlp_input(layer, rot):
    """
    Rotates the input weights of the MLP (Multi-Layer Perceptron) layer.

    Parameters:
    - layer: The transformer layer containing the MLP whose input weights need to be rotated.
    - rot: A torch.Tensor representing the rotation matrix.

    Returns:
    - None
    """
    mlp_inputs = []
    if hasattr(layer.mlp, "experts"):
        for expert in layer.mlp.experts:
            mlp_inputs += [expert.up_proj, expert.gate_proj]
        mlp_inputs += [layer.mlp.gate]
    else:
        mlp_inputs = [layer.mlp.up_proj, layer.mlp.gate_proj]

    prepare_list = [PrepareWeight(ws, post_force=True) for ws in mlp_inputs]
    with ResListToRelease(*prepare_list):
        for weight in mlp_inputs:
            dtype = weight.weight.dtype
            device = weight.weight.device
            weight_data = weight.weight.data.to(device=device, dtype=GLOBAL_DTYPE)

            weight.weight.data = torch.matmul(weight_data, rot).to(device=device, dtype=dtype)
            del weight_data
        return


def rotate_mlp_output(layer, rot):
    """
    Rotate the MLP output weights and bias.
    """
    cur_dtype = torch.float
    modules = []
    if hasattr(layer.mlp, "experts"):
        for expert in layer.mlp.experts:
            modules += [expert.down_proj]
    else:
        modules = [layer.mlp.down_proj]

    prepare_list = [PrepareWeight(ws, post_force=True) for ws in modules]
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
    """
    生成对角线上放置Hadamard矩阵的in_channel维矩阵

    Args:
        in_channel: 输入通道数
        stride: 块大小
        eye_step: 哪些块使用单位矩阵
        device: 设备

    Returns:
        torch.Tensor: 生成的矩阵
    """
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
def rotate_emb_head(model, rot):
    rotate_embeddings(model, rot)
    rotate_head(model, rot)
    return


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
    group_size = config.group_size  # noqa: F841
    rotate_kv = config.rotate_kv  # noqa: F841
    device = config.device
    r2_mode = config.r2_mode
    r2_step = config.r2_step  # noqa: F841
    eye_step = config.eye_step  # noqa: F841

    v_head_dim = model.config.head_dim
    rot_att_uv_config = RotConfig(size=v_head_dim, group_size=-1, mode=r2_mode, device=device)
    rot_att_uv = create_rot(rot_att_uv_config)

    q_b_proj_rot = None
    kv_b_proj_rot = None
    return rot_att_uv, q_b_proj_rot, kv_b_proj_rot


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
    return rot, rot_att_uv, q_b_proj_rot, kv_b_proj_rot


class RMSN(torch.nn.Module):
    """
    This class implements the Root Mean Square Normalization (RMSN) layer.
    We use the implementation from LLAMARMSNorm here:
    https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L75
    """

    def __init__(self, mean_dim: int, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.mean_dim = mean_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        if x.dtype == torch.float16:
            x = x.to(torch.float32)
        variance = x.pow(2).sum(-1, keepdim=True) / self.mean_dim
        x = x * torch.rsqrt(variance + self.eps)
        return x.to(input_dtype)


def rot_model(model, seed=42):
    # 设置随机数种子以确保旋转矩阵可复现
    torch.manual_seed(seed)
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
