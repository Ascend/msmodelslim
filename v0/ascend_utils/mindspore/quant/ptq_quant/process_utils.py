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
# pylint: disable=consider-using-from-import,possibly-used-before-assignment,used-before-assignment

import copy
import collections
import numpy as np

import mindspore.nn as nn
from mindspore import Tensor
from mindspore.common.parameter import Parameter
from mindspore.nn.cell import Cell
from mindspore.ops import functional as F
from mindspore.train.anf_ir_pb2 import ModelProto

from ascend_utils.mindspore.quant.ptq_quant.utils import custom_deepcopy

FusedObj = collections.namedtuple(
    "FusedObj", ["network", "cell_name", "bn_name", "param_weights", "fused_weights", "param_bias", "fused_bias"]
)
FuseBiasObj = collections.namedtuple("FuseBiasObj", "conv_bias, bn_gamma, bn_beta, bn_mean, bn_variance, bn_epsilon")


class Identity(Cell):
    """The identity Layer, simply pass throught the input"""

    def construct(self, input_x):
        '''Definition of subgraph.'''
        return F.identity(input_x)


def is_invalid(value_array):
    '''whether there's inf or nan in value_array'''
    is_array_invalid = np.isnan(value_array) | np.isinf(value_array)
    return is_array_invalid.any()


def find_node_byname(nodes, _name):
    for node in nodes:
        if node.name == _name:
            return node
    return None


def match_fusebn_pattern(node):
    if node.op_type not in ['FusedBatchNorm', 'BatchNorm']:
        return False
    return True


def get_np_array_from_conv(param_weights, param_bias, conv_type):
    if len(param_weights.data.shape) != 4:
        raise RuntimeError("Convolution weights' shape should be 4 dims: N,C,H,W")
    weights_array = None
    if param_weights.data is not None:
        weights_array = param_weights.data.asnumpy()
    bias_array = None
    if param_bias is not None:
        if len(param_bias.data.shape) != 1:
            raise RuntimeError("Convolution bias' shape should be 1 dims: N")
        if param_bias.data is not None:
            bias_array = param_bias.data.asnumpy()
    else:
        if conv_type == 'DepthwiseConv2dNative':
            bias_array = np.zeros((param_weights.data.shape[1]), dtype=np.float32)
        else:
            bias_array = np.zeros((param_weights.data.shape[0]), dtype=np.float32)
    return weights_array, bias_array


def get_np_array_from_bn(param_mean, param_variance):
    if len(param_mean.data.shape) != 1:
        raise RuntimeError('BatchNorm mean\' shape should be [N]')
    mean_array = None
    if param_mean.data is not None:
        mean_array = param_mean.data.asnumpy()

    # Get array of variance from ms parameter
    if len(param_variance.data.shape) != 1:
        raise RuntimeError('BatchNorm variance\' shape should be [N]')
    variance_array = None
    if param_variance.data is not None:
        variance_array = param_variance.data.asnumpy()

    return mean_array, variance_array


def reshape_bn_params(compute_type, bn_params):
    '''Reshape parameters of BN layer before fusing.
    The bn_params is with size matching the 'Channel' dimension.
    '''
    if compute_type == "Conv2D":
        # Channel of output is dim[0], expand shape to 4 dims based on dim[0]
        shape = [-1, 1, 1, 1]
    elif compute_type == 'DepthwiseConv2dNative':
        # Channel of output is dim[1], expand shape to 4 dims based on dim[1]
        shape = [1, -1, 1, 1]
    else:
        raise TypeError('{} is not supported yet.'.format(compute_type))

    return bn_params.reshape(shape)


def reshape_scale_params(compute_type, scale_params):
    '''Reshape parameters of Scale layer before fusing.
    The scale_params is with size matching the 'Channel' dimension.
    '''
    if compute_type == "Conv2D":
        shape = [-1, 1, 1, 1]
    elif compute_type == 'DepthwiseConv2dNative':
        shape = [1, -1, 1, 1]
    else:
        raise TypeError('{} is not supported yet.'.format(compute_type))

    return scale_params.reshape(shape)


def fuse_weight(conv_weight, bn_gamma, bn_variance, bn_epsilon, conv_type):
    '''Inputs:
        conv_weight: a np.array, the weight to be fused.
        bn_variance: a np.array, the variance of BN layer.
        bn_epsilon: a small value, the epsilon of BN layer.
    Returns:
        fused_weight: a np.array, the fused weight.
    '''
    variance = np.add(bn_variance, bn_epsilon)
    variance = reshape_bn_params(conv_type, variance)
    bn_gamma = reshape_scale_params(conv_type, bn_gamma)

    stdev = np.sqrt(variance)
    fused_weight = np.divide(conv_weight, stdev)
    fused_weight = np.multiply(fused_weight, bn_gamma)
    return fused_weight


def fuse_bias(fuse_obj):
    """Fuse bias with BN layer's parameters.

    Inputs:
        conv_bias: a np.array, the bias to be fused.
        bn_mean: a np.array, the mean of BN layer.
        bn_variance: a np.array, the variance of BN layer.
    Returns:
        fused_bias: a np.array, the fused bias.
    """
    conv_bias, bn_gamma, bn_beta, bn_mean, bn_variance = (
        fuse_obj.conv_bias,
        fuse_obj.bn_gamma,
        fuse_obj.bn_beta,
        fuse_obj.bn_mean,
        fuse_obj.bn_variance,
    )
    if conv_bias is None:
        tmp_bias = np.multiply(bn_mean, -1)
    else:
        tmp_bias = conv_bias - bn_mean
    variance = np.add(bn_variance, fuse_obj.bn_epsilon)
    stdev = np.sqrt(variance)
    fused_bias = np.divide(tmp_bias, stdev)

    fused_bias = np.multiply(bn_gamma, fused_bias)
    fused_bias = np.add(fused_bias, bn_beta)
    return fused_bias


def fuse_conv_bn(conv_params, bn_params, conv_type):
    """check the input conv bn params

    Arguments:
        conv_params {list} -- list of conv_weights, conv_bias
        bn_params {list} -- list bn_mean, bn_variance,
                            bn_scale_factor, bn_epsilon

    return:
        the fused conv weights and bias
    """
    conv_weight = conv_params[0]
    conv_bias = conv_params[1]
    bn_mean = bn_params[0]
    bn_variance = bn_params[1]
    bn_gamma = bn_params[2]
    bn_beta = bn_params[3]
    bn_epsilon = bn_params[4]

    fused_weight = fuse_weight(conv_weight, bn_gamma, bn_variance, bn_epsilon, conv_type)
    fused_bias = fuse_bias(FuseBiasObj(conv_bias, bn_gamma, bn_beta, bn_mean, bn_variance, bn_epsilon))
    return fused_weight, fused_bias


def find_cell_by_name(network, layer_name):
    """find the specific cell from network use the input layer_name
    args:
        network: the network of mindspore
        layer_name: the name_prefix of parameters
    """

    for name, cell in network.cells_and_names():
        if name == layer_name:
            return cell
    raise ValueError("cannot find layer by name")


def convert_bn_with_identity(network, bn_name, identity):
    """
    convet sub cell to quant cell
    """
    cells = network.name_cells()
    change = False
    for name in cells:
        subcell = cells[name]
        if subcell == network:
            continue
        if isinstance(subcell, (nn.BatchNorm2d)) and subcell.gamma.name.startswith(bn_name):
            new_subcell = identity
            network.insert_child_to_cell(name, new_subcell)
            change = True
        else:
            convert_bn_with_identity(subcell, bn_name, identity)
    if isinstance(network, nn.SequentialCell) and change:
        network.cell_list = list(network.cells())


def write_fused_weights_bias_back(fused_obj):
    """
    Function: Write fused weights, bias to conv_node
    Parameters: conv_node: target conv_node to write fused weights,
                bias back
                param_weights: weights that in Parameter format
                fused_weights: fused weights that in numpy.array format
                param_bias: bias that in Parameter format
                fused_bias: fused bias that in numpy.array format
    Return: None
    """
    network, cell_name, bn_name, param_weights, fused_weights = (
        fused_obj.network,
        fused_obj.cell_name,
        fused_obj.bn_name,
        fused_obj.param_weights,
        fused_obj.fused_weights,
    )
    param_bias, fused_bias = fused_obj.param_bias, fused_obj.fused_bias
    fused_weights_tensor = Tensor(fused_weights.reshape(param_weights.data.shape), param_weights.data.dtype)
    if fused_bias is not None:
        fused_bias_tensor = Tensor(fused_bias, param_weights.data.dtype)
        if param_bias is not None:
            param_bias.set_data(fused_bias_tensor)
        else:
            param_bias = Parameter(fused_bias_tensor, name='bias')
    conv_cell = find_cell_by_name(network, cell_name)
    conv_cell.weight.set_data(fused_weights_tensor)
    conv_cell.has_bias = fused_bias is not None
    conv_cell.bias = param_bias

    conv_cell.weight.name = cell_name + '.weight'
    conv_cell.bias.name = cell_name + '.bias'

    new_identity_cell = Identity()
    convert_bn_with_identity(network, bn_name, new_identity_cell)


def fuse_convbn_node_pair(network, nodes, node):
    param_dict = network.parameters_dict()
    node_bn = node
    node_bn_inputs = node_bn.input._values
    node_conv = find_node_byname(nodes, node_bn_inputs[0].name)
    if node_conv.op_type == "BiasAdd":
        return  # for conv2d which has bias, does not fuse bn
    param_name = ""
    for _ in node_conv.input._values:
        _node = find_node_byname(nodes, _.name)
        if _node and _node.op_type == 'Load':
            param_name = _node.input._values[0].name
    if param_name.endswith(".weight"):
        cell_name = param_name[: -len('.weight')]
    else:
        cell_name = param_name[: -len('.bias')]
    conv_cell = find_cell_by_name(network, cell_name)
    if is_invalid(conv_cell.weight.data.asnumpy()):
        raise ValueError('Invalid value(nan or inf) in weight of layer {}'.format(cell_name))

    weight_param_name = cell_name + '.weight'
    bias_param_name = cell_name + '.bias'
    if weight_param_name not in conv_cell.parameters_dict():
        return
    param_weights = conv_cell.parameters_dict()[weight_param_name]
    param_weights = copy.deepcopy(param_weights)
    param_bias = None
    if conv_cell.has_bias:
        if is_invalid(conv_cell.bias.data.asnumpy()):
            raise ValueError('Invalid value(nan or inf) in bias of layer {}'.format(node_conv.name_prefix))
        param_bias = conv_cell.parameters_dict()[bias_param_name]
        param_bias = copy.deepcopy(param_bias)

    for _ in node_bn_inputs[1:]:
        _node = find_node_byname(nodes, _.name)
        input_name = _node.input._values[0].name
        if input_name.endswith('gamma'):
            param_gamma = param_dict[input_name]
        elif input_name.endswith('beta'):
            param_beta = param_dict[input_name]
        elif input_name.endswith('moving_mean'):
            param_mean = param_dict[input_name]
        elif input_name.endswith('moving_variance'):
            param_variance = param_dict[input_name]
    bn_name = input_name[: -len('.moving_variance')]
    bn_params = [param_gamma, param_beta, param_mean, param_variance]
    for param in bn_params:
        if is_invalid(param.data.asnumpy()):
            raise ValueError('Invalid value(nan or inf) in {}'.format(node.name))

    gamma_array = param_gamma.data.asnumpy()
    beta_array = param_beta.data.asnumpy()
    weights_array, bias_array = get_np_array_from_conv(param_weights, param_bias, node_conv.op_type)
    mean_array, variance_array = get_np_array_from_bn(param_mean, param_variance)
    bn_cell = find_cell_by_name(network, bn_name)
    eps = bn_cell.eps if hasattr(bn_cell, "eps") else 0.00001
    fused_weights, fused_bias = fuse_conv_bn(
        [weights_array, bias_array], [mean_array, variance_array, gamma_array, beta_array, eps], node_conv.op_type
    )
    weights_shape = param_weights.data.shape

    fused_weights.shape = weights_shape[3] * weights_shape[2] * weights_shape[1] * weights_shape[0]

    if node_conv.op_type == 'DepthwiseConv2dNative':
        fused_bias.shape = weights_shape[1]
    else:
        fused_bias.shape = weights_shape[0]

    write_fused_weights_bias_back(
        FusedObj(network, cell_name, bn_name, param_weights, fused_weights, param_bias, fused_bias)
    )


def get_model_nodes(model, input_data):
    network = custom_deepcopy(model)

    # Specific jit config for quant after the prune
    if hasattr(network, "set_jit_for_quant_after_prune"):
        if hasattr(network, "set_jit_config"):
            from mindspore.common.jit_config import JitConfig

            jit_config = JitConfig(jit_level="O0")
            network.set_jit_config(jit_config)
        else:
            from mindspore.common.api import _cell_graph_executor

            if hasattr(_cell_graph_executor, "set_jit_config"):
                _cell_graph_executor.set_jit_config(jit_config={"jit_level": "o0"})

    network.compile(*input_data)
    graph_proto = network.get_func_graph_proto()
    anf_ir = ModelProto()
    anf_ir.ParseFromString(graph_proto)
    nodes = anf_ir.graph.node._values
    return nodes


def fuse_bn(network, *input_data):
    nodes = get_model_nodes(network, *input_data)
    matched_nodes = []
    flag = False
    for node in nodes:
        if match_fusebn_pattern(node):
            matched_nodes.append(node)
        if node.op_type == 'ReLU6':
            flag = True
    for _, node in enumerate(matched_nodes):
        fuse_convbn_node_pair(network, nodes, node)
    return network, flag


def convert_equact_to_relu(network):
    first_cell = None
    first_name = None
    silu_names = []
    relu6_names = []
    for _, cell in enumerate(network.cells_and_names()):
        name, cell = cell
        if not cell._cells or cell._cell_tag.endswith('SimulatedQuant'):
            if cell.cls_name == 'SimulatedQuant' and first_cell and first_cell.cls_name == 'SiLU':
                silu_names.append(first_name)

            elif cell.cls_name == 'SimulatedQuant' and first_cell and first_cell.cls_name == 'ReLU6':
                relu6_names.append(first_name)
            first_cell = cell
            first_name = name
    new_subcell = nn.ReLU()
    for cell in network.cells_and_names():
        name, cell = cell
        if name + '.act' in silu_names:
            cell.insert_child_to_cell('act', new_subcell)
        elif name + '.2' in relu6_names:
            cell.insert_child_to_cell('2', new_subcell)
    return network
