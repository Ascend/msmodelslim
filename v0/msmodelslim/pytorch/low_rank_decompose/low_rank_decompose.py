#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
-------------------------------------------------------------------------
This file is part of the MindStudio project.
Copyright (c) 2025 Huawei Technologies Co.,Ltd.

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


import copy

import torch
from torch import nn

from ascend_utils.common.security.type import check_int, check_type
from msmodelslim.common.low_rank_decompose import (
    is_hidden_channels_valid,
    get_hidden_channels_by_layer_name,
    get_decompose_channels_2d,
    decompose_weight_2d_svd,
    get_decompose_channels_4d,
    decompose_weight_4d_tucker,
)
from msmodelslim.common.low_rank_decompose import export
from msmodelslim import logger
from msmodelslim.utils.logging import progress_bar


def weight_as_numpy(weight):
    return weight.detach().cpu().numpy().astype("float32")


class SaveInput(nn.Module):
    def __init__(self, in_features: int, source_name: str, is_channel_first=True):
        super().__init__()
        self.register_buffer("num_samples", torch.zeros(1))
        self.register_buffer("input_data", torch.zeros([in_features, in_features]))
        self.source_name = source_name
        self.is_channel_first = is_channel_first

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.dim() != 2:
            perm = [0] + list(range(2, inputs.dim())) + [1]
            input_data = inputs.permute(perm) if self.is_channel_first else inputs
            input_data = input_data.reshape([-1, input_data.shape[-1]])
        else:
            input_data = inputs

        cur_num_samples = input_data.shape[0]
        input_data = torch.matmul(input_data.T, input_data)
        input_data = self.input_data * self.num_samples + input_data
        num_samples = self.num_samples + cur_num_samples
        self.input_data.data.copy_(torch.div(input_data, num_samples))
        self.num_samples.data.copy_(num_samples)

        return inputs


def replace_module_with_saving_input(network, decompose_config=None, prefix_name=""):
    for name, child_module in network.named_children():
        full_name = prefix_name + name
        if isinstance(child_module, nn.Linear) or isinstance(child_module, nn.Conv2d):
            if decompose_config is not None and full_name not in decompose_config:
                continue  # Convert only those need to decompose

            if isinstance(child_module, nn.Linear):
                layer_with_input = nn.Sequential(SaveInput(child_module.in_features, full_name, False), child_module)
            else:
                layer_with_input = nn.Sequential(SaveInput(child_module.in_channels, full_name, True), child_module)
            setattr(network, name, layer_with_input)
        elif hasattr(child_module, "named_children"):
            cur_prefix_name = full_name + "."
            replace_module_with_saving_input(child_module, decompose_config, prefix_name=cur_prefix_name)


def get_input_data_for_each_layer(network, decompose_config, datasets, max_iter=-1):
    network = copy.deepcopy(network)

    """ Convert Linear to a block with SaveInput, saving input_data for each Linear layer """
    _ = network.eval()
    replace_module_with_saving_input(network, decompose_config)
    is_npu_available = False
    if hasattr(network, "npu"):
        network = network.npu()
        is_npu_available = True

    """ Run Inference """
    input_data_dict = {}
    for iter_id, inputs in progress_bar(enumerate(datasets), desc='Running Inference'):
        if isinstance(inputs, dict):
            inputs = {kk: vv.npu() for kk, vv in inputs.items()} if is_npu_available else inputs
            network(**inputs)
        elif isinstance(inputs, (list, tuple)):
            inputs = [ii.npu() for ii in inputs] if is_npu_available else inputs
            network(*inputs)
        else:
            logger.error("Provided datasets element is not a dict or list or tuple")
            return input_data_dict

        if max_iter > 0 and iter_id + 1 >= max_iter:
            break

    """ Take out input_data for each block with SaveInput layer """
    for _, child_module in network.named_modules():
        if isinstance(child_module, SaveInput):
            input_data = weight_as_numpy(child_module.input_data)
            input_data_dict.update({child_module.source_name: input_data})
    return input_data_dict


def decompose_conv2d(source_layer, hidden_channels, input_data=None, do_decompose_weight=True):
    if not is_hidden_channels_valid(hidden_channels, digit_only=True):
        raise ValueError(f"Provided hidden_channels={hidden_channels} is not valid")

    if isinstance(hidden_channels, (int, float)):
        hidden_out, hidden_in = hidden_channels, hidden_channels
    else:
        hidden_out, hidden_in = hidden_channels[0], hidden_channels[1]

    out_channels, in_channels = source_layer.out_channels, source_layer.in_channels
    kernel_size = source_layer.kernel_size
    has_bias = source_layer.bias is not None
    decomposed = nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=hidden_in, kernel_size=1, padding=0, bias=False),
        nn.Conv2d(
            in_channels=hidden_in,
            out_channels=hidden_out,
            kernel_size=kernel_size,
            stride=source_layer.stride,
            padding_mode=source_layer.padding_mode,
            padding=source_layer.padding,
            dilation=source_layer.dilation,
            groups=source_layer.groups,
            bias=False,
        ),
        nn.Conv2d(in_channels=hidden_out, out_channels=out_channels, kernel_size=1, padding=0, bias=has_bias),
    )

    if do_decompose_weight:
        original_weight = weight_as_numpy(source_layer.weight)  # (out_channels, in_channels, kernel_size, kernel_size)
        res = decompose_weight_4d_tucker(
            original_weight, hidden_out, hidden_in, input_data
        )
        first = res.get('first', None)
        core = res.get('core', None)
        last = res.get('last', None)
        (actual_hidden_out, actual_hidden_in) = res.get('out_in', None)

        if first is None or actual_hidden_out != hidden_out or actual_hidden_in != hidden_in:
            logger.error(f"Decompose weight failed for Conv2d layer, shape: {source_layer.shape}")
            return decomposed

        decomposed[0].weight.data = torch.from_numpy(first)
        decomposed[1].weight.data = torch.from_numpy(core)
        decomposed[-1].weight.data = torch.from_numpy(last)
        if has_bias:
            decomposed[-1].bias.data = source_layer.bias
    return decomposed


def decompose_embedding(source_layer, hidden_channels, do_decompose_weight=True):
    if not is_hidden_channels_valid(hidden_channels, digit_only=True):
        raise ValueError(f"Provided hidden_channels={hidden_channels} is not valid")

    num_embeddings, embedding_dim = source_layer.num_embeddings, source_layer.embedding_dim
    decomposed = nn.Sequential(
        nn.Embedding(
            num_embeddings=num_embeddings,
            embedding_dim=hidden_channels,
            padding_idx=source_layer.padding_idx,
            max_norm=source_layer.max_norm,
            norm_type=source_layer.norm_type,
            scale_grad_by_freq=source_layer.scale_grad_by_freq,
            sparse=source_layer.sparse,
        ),
        nn.Linear(in_features=hidden_channels, out_features=embedding_dim, bias=False)
    )

    if do_decompose_weight:
        original_weight = weight_as_numpy(source_layer.weight)
        # uu: (num_embeddings, hidden_channels), vv: (hidden_channels, embedding_dim)
        svd_uu, svd_vv, actual_hidden_channels = decompose_weight_2d_svd(original_weight, hidden_channels)
        if svd_uu is None or actual_hidden_channels != hidden_channels:
            logger.error(f"Decompose weight failed for Embedding layer, shape: {source_layer.shape}")
            return decomposed

        decomposed[0].weight.data = torch.from_numpy(svd_uu)
        decomposed[-1].weight.data = torch.from_numpy(svd_vv.T)
    return decomposed


def decompose_linear(source_layer, hidden_channels, input_data=None, do_decompose_weight=True):
    if not is_hidden_channels_valid(hidden_channels, digit_only=True):
        raise ValueError(f"Provided hidden_channels={hidden_channels} is not valid")

    out_channels, in_channels = source_layer.out_features, source_layer.in_features
    has_bias = source_layer.bias is not None
    decomposed = nn.Sequential(
        nn.Linear(in_features=in_channels, out_features=hidden_channels, bias=False),
        nn.Linear(in_features=hidden_channels, out_features=out_channels, bias=has_bias)
    )

    if do_decompose_weight:
        original_weight = weight_as_numpy(source_layer.weight)
        svd_uu, svd_vv, actual_hidden_channels = decompose_weight_2d_svd(original_weight, hidden_channels, input_data)
        if svd_uu is None or actual_hidden_channels != hidden_channels:
            logger.error(f"Decompose weight failed for Linear layer, shape: {source_layer.shape}")
            return decomposed

        decomposed[0].weight.data = torch.from_numpy(svd_vv)
        decomposed[-1].weight.data = torch.from_numpy(svd_uu)
        if has_bias:
            decomposed[-1].bias.data = source_layer.bias
    return decomposed


def get_decomposed_config(network, hidden_channels=0.5, excludes=None, divisor=64):
    if not isinstance(network, torch.nn.Module):
        raise ValueError("Provided network is not a torch.nn.Module")

    decomposed_info = {}
    for name, child_module in network.named_modules():
        cur_hidden_channels = get_hidden_channels_by_layer_name(name, hidden_channels, excludes=excludes)
        if not is_hidden_channels_valid(cur_hidden_channels):
            continue

        if isinstance(child_module, nn.Linear) or isinstance(child_module, nn.Embedding):
            actual_hidden_channels = get_decompose_channels_2d(
                child_module.weight, cur_hidden_channels, divisor, as_numpy_func=weight_as_numpy
            )
            if actual_hidden_channels > 0:
                decomposed_info[name] = actual_hidden_channels
        elif isinstance(child_module, nn.Conv2d):
            actual_hidden_out, actual_hidden_in = get_decompose_channels_4d(
                child_module.weight, cur_hidden_channels, divisor, as_numpy_func=weight_as_numpy
            )
            if actual_hidden_out > 0 and actual_hidden_in > 0:
                decomposed_info[name] = (actual_hidden_out, actual_hidden_in)
    return decomposed_info


def decompose_network_recursion(
        network,
        decompose_config,
        input_data_dict=None,
        do_decompose_weight=True,
        prefix_name="",
):
    for name, child_module in network.named_children():
        full_name = prefix_name + name

        if full_name in decompose_config:
            decomposed = None
            cur_hidden_channels = decompose_config[full_name]
            if isinstance(child_module, nn.Linear):
                input_data = input_data_dict.get(full_name) if isinstance(input_data_dict, dict) else None
                decomposed = decompose_linear(child_module, cur_hidden_channels, input_data, do_decompose_weight)
            elif isinstance(child_module, nn.Embedding):
                decomposed = decompose_embedding(child_module, cur_hidden_channels, do_decompose_weight)
            elif isinstance(child_module, nn.Conv2d):
                input_data = input_data_dict.get(full_name) if isinstance(input_data_dict, dict) else None
                decomposed = decompose_conv2d(child_module, cur_hidden_channels, input_data, do_decompose_weight)

            if decomposed is None:
                continue

            logger.debug(f"[{child_module.__class__.__name__}] Layer name: {full_name}")
            logger.debug(f"  Before: {child_module}")
            logger.debug(f"  After: {decomposed}")
            setattr(network, name, decomposed)
        elif hasattr(child_module, "named_children"):
            cur_prefix_name = full_name + "."
            decompose_network_recursion(
                network=child_module,
                decompose_config=decompose_config,
                input_data_dict=input_data_dict,
                do_decompose_weight=do_decompose_weight,
                prefix_name=cur_prefix_name,
            )


def decompose_network(network, decompose_config, do_decompose_weight=True, datasets=None, max_iter=-1):
    if not isinstance(network, torch.nn.Module):
        raise TypeError("Provided network is not a torch.nn.Module")
    check_type(do_decompose_weight, bool, param_name="do_decompose_weight")
    check_int(max_iter, min_value=-1, param_name="max_iter")

    network = copy.deepcopy(network)
    if datasets is not None:
        input_data_dict = get_input_data_for_each_layer(network, decompose_config, datasets, max_iter)
    else:
        input_data_dict = None

    decompose_network_recursion(network, decompose_config, input_data_dict, do_decompose_weight)
    return network


class Decompose(export.Decompose):
    def __init__(self, model: torch.nn.Module, config_file: str = None):
        """
        Create a low rank decomposer instance.

        Args:
          model: PyTorch or MindSpore model.
          config_file: json file name for saving model decomposed layer config.
              Also as a restore path if calling `from_file`.

        Examples:
        >>> from msmodelslim.pytorch import low_rank_decompose
        >>> from ascend_utils.common.utils import count_parameters
        >>> from torchvision.models import resnet50
        >>> model = resnet50()
        >>> decomposer = low_rank_decompose.Decompose(model).from_ratio(0.5)
        >>> decomposed_model = decomposer.decompose_network()
        >>> print("Original model parameters:", count_parameters(model))
        >>> print("decompose_config:", decomposer.decompose_config)
        >>> print("Decomposed model parameters:", count_parameters(decomposed_model))
        """
        if not isinstance(model, torch.nn.Module):
            raise ValueError("Provided network is not a torch.nn.Module")

        super().__init__(model, config_file)
        self.get_decomposed_config_backend = get_decomposed_config
        self.decompose_network_backend = decompose_network
