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

import mindspore as ms
from mindspore import nn

from ascend_utils.common.mindspore_utils import SaveInput, update_cell
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
    return weight.asnumpy().astype("float32")


def replace_module_with_saving_input(network, decompose_config=None):
    for name, cell in network.cells_and_names():
        if decompose_config is not None and name not in decompose_config:  # Convert only those need to decompose
            continue

        if isinstance(cell, nn.Dense):
            layer_with_input = nn.SequentialCell([SaveInput(cell.in_channels, name, is_channel_first=False), cell])
            update_cell(network, cell, name, layer_with_input)
        elif isinstance(cell, nn.Conv2d):
            layer_with_input = nn.SequentialCell([SaveInput(cell.in_channels, name, is_channel_first=True), cell])
            update_cell(network, cell, name, layer_with_input)


def get_input_data_for_each_layer(network, decompose_config, datasets, max_iter=-1):
    network = copy.deepcopy(network)

    """ Convert Linear to a block with SaveInput, saving input_data for each Linear layer """
    replace_module_with_saving_input(network, decompose_config)
    """ Run Inference """
    input_data_dict = {}
    for iter_id, inputs in progress_bar(enumerate(datasets), desc='Running Inference'):
        if isinstance(inputs, dict):
            network(**inputs)
        elif isinstance(inputs, (list, tuple)):
            network(*inputs)
        else:
            logger.error("Provided datasets element is not a dict or list or tuple")
            return input_data_dict

        if max_iter > 0 and iter_id + 1 >= max_iter:
            break

    """ Take out input_data for each block with SaveInput layer """
    for _, cell in network.cells_and_names():
        if isinstance(cell, SaveInput):
            input_data_dict.update({cell.source_name: cell.input_data.asnumpy().astype("float32")})
    return input_data_dict


def copy_shard_strategy(source_layer, target_layer):
    if hasattr(source_layer, "_shard_fn"):
        setattr(target_layer, "_shard_fn", getattr(source_layer, "_shard_fn"))

    for sub_attr_name in source_layer.__dict__.keys():
        if sub_attr_name.startswith('_') or not hasattr(target_layer, sub_attr_name):
            continue

        sub_attr = getattr(source_layer, sub_attr_name)
        in_strategy = getattr(sub_attr, "in_strategy", None)
        out_strategy = getattr(sub_attr, "out_strategy", None)
        target_sub_attr = getattr(target_layer, sub_attr_name)
        if hasattr(target_sub_attr, "shard") and (in_strategy or out_strategy):
            target_sub_attr.shard(in_strategy, out_strategy)


def has_shard_strategy(source_layer):
    if getattr(source_layer, "in_strategy", None) is not None \
            or getattr(source_layer, "out_strategy", None) is not None:
        return True

    for sub_attr in source_layer.__dict__.values():
        in_strategy = getattr(sub_attr, "in_strategy", None)
        out_strategy = getattr(sub_attr, "out_strategy", None)
        if in_strategy is not None or out_strategy is not None:
            return True

    return False


def decompose_conv2d(source_layer, hidden_channels, input_data=None, do_decompose_weight=True):
    if not is_hidden_channels_valid(hidden_channels, digit_only=True):
        raise ValueError(f"Provided hidden_channels={hidden_channels} is not valid")

    if isinstance(hidden_channels, (list, tuple)):
        hidden_out, hidden_in = hidden_channels[0], hidden_channels[1]
    else:
        hidden_out, hidden_in = hidden_channels, hidden_channels

    out_channels, in_channels = source_layer.out_channels, source_layer.in_channels
    has_bias, kernel_size = source_layer.has_bias, source_layer.kernel_size
    decomposed = nn.SequentialCell([
        nn.Conv2d(in_channels=in_channels, out_channels=hidden_in, kernel_size=1, padding=0, has_bias=False),
        nn.Conv2d(
            in_channels=hidden_in,
            out_channels=hidden_out,
            kernel_size=kernel_size,
            stride=source_layer.stride,
            pad_mode=source_layer.pad_mode,
            padding=source_layer.padding,
            dilation=source_layer.dilation,
            group=source_layer.group,
            has_bias=False,
        ),
        nn.Conv2d(in_channels=hidden_out, out_channels=out_channels, kernel_size=1, padding=0, has_bias=has_bias),
    ])

    if do_decompose_weight:
        original_weight = source_layer.weight.asnumpy()  # (out_channels, in_channels, kernel_size, kernel_size)
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

        decomposed[0].weight.set_data(ms.Tensor(first))
        decomposed[1].weight.set_data(ms.Tensor(core))
        decomposed[-1].weight.set_data(ms.Tensor(last))
        if has_bias:
            decomposed[-1].bias.set_data(source_layer.bias)
    return decomposed


def decompose_embedding(source_layer, hidden_channels, do_decompose_weight=True):
    if not is_hidden_channels_valid(hidden_channels, digit_only=True):
        raise ValueError(f"Provided hidden_channels={hidden_channels} is not valid")

    vocab_size, embedding_size = source_layer.vocab_size, source_layer.embedding_size
    decomposed = nn.SequentialCell([
        nn.Embedding(
            vocab_size=vocab_size,
            embedding_size=hidden_channels,
            use_one_hot=source_layer.use_one_hot,
            dtype=source_layer.dtype,
            padding_idx=source_layer.padding_idx,
        ),
        nn.Dense(in_channels=hidden_channels, out_channels=embedding_size, has_bias=False)
    ])

    copy_shard_strategy(source_layer, decomposed[0])
    if has_shard_strategy(source_layer):
        decomposed[1].matmul.shard(((1, 1), (1, 1)))
        decomposed[1].weight.parallel_optimizer = False

    if do_decompose_weight:
        original_weight = source_layer.embedding_table.asnumpy()  # (vocab_size, embedding_size)
        # uu: (vocab_size, hidden_channels), vv: (hidden_channels, embedding_size)
        svd_uu, svd_vv, actual_hidden_channels = decompose_weight_2d_svd(original_weight, hidden_channels)
        if svd_uu is None or actual_hidden_channels != hidden_channels:
            logger.error(f"Decompose weight failed for Embedding layer, shape: {source_layer.shape}")
            return decomposed

        decomposed[0].embedding_table.set_data(ms.Tensor(svd_uu))
        decomposed[-1].weight.set_data(ms.Tensor(svd_vv.T))
    return decomposed


def decompose_linear(source_layer, hidden_channels, input_data=None, do_decompose_weight=True):
    if not is_hidden_channels_valid(hidden_channels, digit_only=True):
        raise ValueError(f"Provided hidden_channels={hidden_channels} is not valid")

    out_channels, in_channels = source_layer.out_channels, source_layer.in_channels
    is_fp16_cell = source_layer.get_mixed_precision_type() == getattr(ms, "_c_expression").MixedPrecisionType.FP16
    decomposed = nn.SequentialCell([
        nn.Dense(in_channels=in_channels, out_channels=hidden_channels, has_bias=False),
        nn.Dense(
            in_channels=hidden_channels,
            out_channels=out_channels,
            has_bias=source_layer.has_bias,
            activation=source_layer.activation,
        )
    ])

    if is_fp16_cell:
        decomposed[0] = decomposed[0].to_float(ms.float16)
        decomposed[1] = decomposed[1].to_float(ms.float16)

    if do_decompose_weight:
        original_weight = source_layer.weight.asnumpy()
        svd_uu, svd_vv, actual_hidden_channels = decompose_weight_2d_svd(original_weight, hidden_channels, input_data)
        if svd_uu is None or actual_hidden_channels != hidden_channels:
            logger.error(f"Decompose weight failed for Dense layer, shape: {source_layer.shape}")
            return decomposed

        decomposed[0].weight.set_data(ms.Tensor(svd_vv).astype(source_layer.weight.dtype))
        decomposed[-1].weight.set_data(ms.Tensor(svd_uu).astype(source_layer.weight.dtype))
        if source_layer.has_bias:
            decomposed[-1].bias.set_data(source_layer.bias)
    return decomposed


def get_decomposed_config(network, hidden_channels=0.5, excludes=None, divisor=64):
    if not isinstance(network, ms.nn.Cell):
        raise ValueError("Provided network is not a mindspore.nn.Cell")

    decomposed_info = {}
    for name, cell in network.cells_and_names():
        cur_hidden_channels = get_hidden_channels_by_layer_name(name, hidden_channels, excludes=excludes)
        if not is_hidden_channels_valid(cur_hidden_channels):
            continue

        if isinstance(cell, nn.Dense):
            actual_hidden_channels = get_decompose_channels_2d(
                cell.weight, cur_hidden_channels, divisor, as_numpy_func=weight_as_numpy
            )
            if actual_hidden_channels > 0:
                decomposed_info[name] = actual_hidden_channels
        elif isinstance(cell, nn.Embedding):
            actual_hidden_channels = get_decompose_channels_2d(
                cell.embedding_table, cur_hidden_channels, divisor, as_numpy_func=weight_as_numpy
            )
            if actual_hidden_channels > 0:
                decomposed_info[name] = actual_hidden_channels
        elif isinstance(cell, nn.Conv2d):
            actual_hidden_out, actual_hidden_in = get_decompose_channels_4d(
                cell.weight, cur_hidden_channels, divisor, as_numpy_func=weight_as_numpy
            )
            if actual_hidden_out > 0 and actual_hidden_in > 0:
                decomposed_info[name] = (actual_hidden_out, actual_hidden_in)
    return decomposed_info


def decompose_network(network, decompose_config, do_decompose_weight=True, datasets=None, max_iter=-1):
    if not isinstance(network, ms.nn.Cell):
        raise TypeError("Provided network is not a ms.nn.Cell")

    if datasets is not None:
        input_data_dict = get_input_data_for_each_layer(network, decompose_config, datasets, max_iter)
    else:
        input_data_dict = None

    for name, cell in network.cells_and_names():
        if name not in decompose_config:
            continue

        decomposed = None
        cur_hidden_channels = decompose_config[name]
        if isinstance(cell, nn.Conv2d):
            input_data = input_data_dict.get(name) if isinstance(input_data_dict, dict) else None
            decomposed = decompose_conv2d(cell, cur_hidden_channels, input_data, do_decompose_weight)
        elif isinstance(cell, nn.Embedding):
            decomposed = decompose_embedding(cell, cur_hidden_channels, do_decompose_weight)
        elif isinstance(cell, nn.Dense):
            input_data = input_data_dict.get(name) if isinstance(input_data_dict, dict) else None
            decomposed = decompose_linear(cell, cur_hidden_channels, input_data, do_decompose_weight)

        if decomposed is None:
            continue

        logger.debug(f"[{cell.__class__.__name__}] Layer name: {name}")
        logger.debug(f"  Before: {cell}")
        logger.debug(f"  After: {decomposed}")
        update_cell(network, cell, name, decomposed)
    return network


class Decompose(export.Decompose):
    def __init__(self, model: ms.nn.Cell, config_file: str = None):
        """
        Create a low rank decomposer instance.

        Args:
          model: PyTorch or MindSpore model.
          config_file: json file name for saving model decomposed layer config.
              Also as a restore path if calling `from_file`.

        Examples:
        >>> from msmodelslim.mindspore import low_rank_decompose
        >>> from ascend_utils.common.utils import count_parameters
        >>> from mindvision.classification.models import resnet50
        >>> model = resnet50()
        >>> decomposer = low_rank_decompose.Decompose(model).from_ratio(0.5)
        >>> decomposed_model = decomposer.decompose_network()
        >>> print("Original model parameters:", count_parameters(model))
        >>> print("decompose_config:", decomposer.decompose_config)
        >>> print("Decomposed model parameters:", count_parameters(decomposed_model))
        """
        if not isinstance(model, ms.nn.Cell):
            raise ValueError("Provided network is not a mindspore.nn.Cell")

        super().__init__(model, config_file)
        self.get_decomposed_config_backend = get_decomposed_config
        self.decompose_network_backend = decompose_network
