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
# pylint: disable=consider-merging-isinstance,consider-using-from-import,no-else-continue

from collections import OrderedDict
import copy
import logging
import numpy as np

from mindspore import context
import mindspore.nn as nn

try:
    from mindspore.nn.transformer.layers import _Linear
except ModuleNotFoundError:
    try:
        from mindspore.nn.layer.transformer import _Linear
    except ImportError:
        from mindspore.nn import Dense as _Linear

from ascend_utils.mindspore.quant.ptq_quant.featuremap_quantization import quant_activation
from ascend_utils.mindspore.quant.ptq_quant.weight_quantization import quant_weight
from ascend_utils.mindspore.quant.ptq_quant.simulated_quant import SimulatedQuant

QUANT_OPERATORS = (nn.Conv2d, nn.Dense, _Linear, nn.Conv1d)


def measure_model(network, input_data, quant_layer_indics):
    ori_mode = context.get_context("mode")
    context.set_context(mode=context.PYNATIVE_MODE)
    feature_maps = OrderedDict()
    cell_ids = []

    def pre_forward_hook(cell_id, inputs):
        _input = inputs[0]
        cell_ids.append(cell_id)
        feature_size = _input.shape[-2] * _input.shape[-1]
        if cell_id not in feature_maps:
            feature_maps.setdefault(cell_id, feature_size)

    all_index = OrderedDict()
    test_cell_ids = []
    for index, (_, cell) in enumerate(network.cells_and_names()):
        all_index.setdefault(index, cell)
        if index in quant_layer_indics:
            cell_id = cell.cls_name + "(" + str(id(cell)) + ")"
            test_cell_ids.append(cell_id)
            cell.register_forward_pre_hook(pre_forward_hook)
    network(*input_data)
    diff_cell_ids = set(test_cell_ids).difference(set(cell_ids))
    # 有的cell在构造函数中定义了，但是construct时并没有调用，这类算子不量化
    for index, cell in enumerate(network.cells_and_names()):
        _, cell = cell
        cell_id = cell.cls_name + "(" + str(id(cell)) + ")"
        if cell_id in diff_cell_ids:
            quant_layer_indics.remove(index)
    context.set_context(mode=ori_mode)
    return list(feature_maps.values())


def find_input_feature(nodes, node, input_data):
    for value in node.input._values:
        if value.name != 'x' and nodes[int(value.name) - 1].op_type == 'Load':
            continue
        elif value.name != 'x':
            tmp = nodes[int(value.name) - 1].output_type.tensor_type
            dims = tmp.shape.dim._values
            if len(dims) == 4:
                dim = dims[2].size * dims[3].size
            else:
                dim = dims[0].size * dims[1].size
            return dim
        else:
            tmp_size = input_data[0].shape[2] * input_data[0].shape[3]
            return tmp_size
    raise ValueError("cannot find input feature")


def quant_module(model, strategy):
    network = custom_deepcopy(model)
    for _, (name, cell) in enumerate(network.cells_and_names()):
        if not strategy:
            break
        if is_quant_cell(cell):
            if strategy[0] == 8:
                quant_weight(cell)
                quant_activation(network, name, cell)
            strategy.pop(0)
    return network


def is_quant_cell(cell):
    return (
        isinstance(cell, nn.Conv2d)
        and cell.in_channels > 16
        or isinstance(cell, nn.Conv1d)
        or isinstance(cell, nn.Dense)
        or (isinstance(cell, _Linear) and hasattr(cell, "expert_num") and cell.expert_num == 1)
    )


def get_compression_rate(network):
    origin_num, quant_num, unquant_num = calc_quant_nums(network)
    # The model is quantized from float32 to int8, which decreases by 4 times.
    origin_size = origin_num * 4 * 1e-3
    quanted_model_size = (unquant_num * 4 + quant_num) * 1e-3
    try:
        compress_ratio = (origin_size - quanted_model_size) / origin_size * 100
    except ZeroDivisionError as ex:
        logging.error('origin_size cannot be zero. %s', str(ex))
        raise ex
    logging.info(
        "Origin model size = %.2f KB, Quanted model size  = %.2f KB, compress_ratio = %.2f%%",
        origin_size,
        quanted_model_size,
        compress_ratio,
    )
    return compress_ratio, origin_size, quanted_model_size


def calc_quant_nums(network):
    quant_num = 0
    origin_num = 0
    unquant_num = 0
    quant_modules = []
    add_moduels = []
    for _, cell in enumerate(network.cells_and_names()):
        module_name, module = cell
        if isinstance(module, QUANT_OPERATORS):
            quant_modules.append((module_name, module))
        elif isinstance(module, SimulatedQuant):
            add_moduels.append(module)
    for q_module_name, q_module in quant_modules:
        if isinstance(q_module, nn.Conv2d):
            weight_num = np.prod(q_module.weight.shape)
        else:
            weight_num = q_module.in_channels * q_module.out_channels
        if q_module_name.endswith('compute_cell'):
            quant_num += weight_num
            origin_num += weight_num
            if hasattr(q_module, "bias") and q_module.bias is not None:
                quant_num += np.prod(q_module.bias.shape)
                origin_num += np.prod(q_module.bias.shape)
            if hasattr(q_module, 'scale_weight'):
                unquant_num += np.prod(q_module.scale_weight.shape)
        else:
            origin_num += weight_num
            unquant_num += weight_num
            if hasattr(q_module, "bias") and q_module.bias is not None:
                origin_num += np.prod(q_module.bias.shape)
                unquant_num += np.prod(q_module.bias.shape)
    for _module in add_moduels:
        unquant_num += np.prod(_module.scale_featuremap.shape)
        unquant_num += np.prod(_module.offset_featuremap.shape)
    return origin_num, quant_num, unquant_num


def get_flops(model, input_data):
    stats = ModelStatistics(model, input_data)
    flops, _ = stats.get_stats()
    flops *= 1e-9
    params = 0
    for _, param in model.parameters_and_names():
        params += np.prod(param.shape)
    params *= 1e-3 * 4
    return flops, params


class ModelStatistics:
    def __init__(self, model, input_data):
        self.model = custom_deepcopy(model)
        self.input_data = input_data
        self.wsize_list = []
        self.flops_list = []
        self.has_weight_idx = []

    @classmethod
    def measure_layer_for_quant(cls, layer, input_x):
        multi_add = 1
        dim = layer.weight.ndim if hasattr(layer, 'weight') else 0
        # ops_conv2d & conv1d
        if dim == 4:
            if isinstance(layer, nn.Conv1d):
                temp_input = layer.expand_dims(input_x, 2)
            else:
                temp_input = input_x
            params_dict = layer.parameters_dict()

            value_h0 = (layer.kernel_size[0] - 1) * (layer.dilation[0] - 1) + layer.kernel_size[0]
            value_w0 = (layer.kernel_size[1] - 1) * (layer.dilation[1] - 1) + layer.kernel_size[1]
            if isinstance(layer.padding, tuple):
                h_padding = layer.padding[0] + layer.padding[1]
                w_padding = layer.padding[2] + layer.padding[3]
            else:
                h_padding = 2 * layer.padding
                w_padding = 2 * layer.padding
            try:
                out_h = int((temp_input.shape[2] + h_padding - value_h0) / layer.stride[0] + 1)
                out_w = int((temp_input.shape[3] + w_padding - value_w0) / layer.stride[1] + 1)
                layer.flops = (
                    layer.in_channels
                    * layer.out_channels
                    * layer.kernel_size[0]
                    * layer.kernel_size[1]
                    * out_h
                    * out_w
                    / layer.group
                    * multi_add
                )
            except ZeroDivisionError:
                logging.error("division by zero! %s", ZeroDivisionError)
            layer.params = sum(p.size for p in params_dict.values())
        # ops_linear
        elif dim == 2:
            params = layer._params
            weight_ops = params['weight'].size * multi_add
            if 'bias' not in params:
                layer.flops = weight_ops
            else:
                bias_ops = params['bias'].size
                layer.flops = weight_ops + bias_ops
            layer.params = sum(p.size for p in params.values())

    def get_stats(self):
        def new_construct(cell):
            def lambda_forward(input_x):
                self.measure_layer_for_quant(cell, input_x)
                value_y = cell.old_construct(input_x)
                return value_y

            return lambda_forward

        context.set_context(mode=context.PYNATIVE_MODE)
        c_list = []
        for _, cell in self.model.cells_and_names():
            c_list.append(cell)

        self.has_weight_idx = []
        for i, cell in enumerate(c_list):
            params = cell._params
            if not ('weight' in params and params['weight'].ndim in [2, 4]):
                continue
            self.has_weight_idx.append(i)

        for idx in self.has_weight_idx:  # get all
            cell = c_list[idx]
            cell.old_construct = cell.construct
            cell.construct = new_construct(cell)

        self.wsize_list = []
        self.flops_list = []

        self.model.set_train(False)
        if isinstance(self.input_data, (list, tuple)):
            self.model(*self.input_data)
        else:
            self.model(self.input_data)

        for idx in self.has_weight_idx:
            if hasattr(c_list[idx], 'params'):
                self.wsize_list.append(c_list[idx].params)
            else:
                logging.warning('Network has redundancy cell: %r', c_list[idx])
            if hasattr(c_list[idx], 'flops'):
                self.flops_list.append(c_list[idx].flops)
            else:
                logging.warning('Network has redundancy cell: %r', c_list[idx])

        context.set_context(mode=0)
        del self.model
        return sum(self.flops_list), sum(self.wsize_list)


def custom_deepcopy(model):
    if "config" in dir(model):
        if "logger" in dir(model.config):
            del model.config.logger
    copy_model = copy.deepcopy(model)
    from mindspore._c_expression import MixedPrecisionType

    ori_cells = {}
    for name, cell in model.cells_and_names():
        ori_cells.setdefault(name, cell)
    new_cells = {}
    for name, cell in copy_model.cells_and_names():
        new_cells.setdefault(name, cell)
    for name, cell in new_cells.items():
        if name in ori_cells:
            ori_cell = ori_cells.get(name)
            mixed_type = ori_cell.get_mixed_precision_type()
            if mixed_type == MixedPrecisionType.FP16:
                cell._set_mixed_precision_type_recursive(MixedPrecisionType.FP16)
            elif mixed_type == MixedPrecisionType.FP32:
                cell._set_mixed_precision_type_recursive(MixedPrecisionType.FP32)
    return copy_model
