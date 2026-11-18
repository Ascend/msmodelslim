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
import onnx
from onnx.helper import get_attribute_value

from msmodelslim.onnx.post_training_quant.dag.graph import OnnxGraph
from msmodelslim.onnx.post_training_quant.dag.param import Tensor, NodeParam
from msmodelslim.onnx.post_training_quant.dag.node import OnnxNode, QuantizableOnnxNode
from msmodelslim.onnx.post_training_quant.util import parse_tensor_proto
from msmodelslim import logger


def parse_model(model, quantized_nodes=None) -> OnnxGraph:
    logger.info("Parse model...")
    graph = model.graph
    onnx_graph = OnnxGraph(graph.input, graph.output,
                           model.opset_import, model.ir_version,
                           graph.name, model.domain)

    for param in graph.initializer:
        value = parse_tensor_proto(param)
        onnx_param = Tensor(value=value, dtype=param.data_type,
                            shape=param.dims, name=param.name)
        onnx_graph.params.setdefault(param.name, onnx_param)

    all_input_tensor, all_output_tensor = build_relations(graph, onnx_graph)

    for node in graph.node:
        attrs = {}
        for attr in node.attribute:
            value = get_attribute_value(attr)
            attrs.setdefault(attr.name, value)

        if quantized_nodes and node.name in quantized_nodes:
            onnx_node = QuantizableOnnxNode(name=node.name, op_type=node.op_type,
                                            attrs=attrs, domain=node.domain)
        else:
            onnx_node = OnnxNode(name=node.name, op_type=node.op_type,
                                 attrs=attrs, domain=node.domain)

        for idx, input_x in enumerate(node.input):
            if input_x in onnx_graph.params:
                onnx_param = onnx_graph.params.get(input_x)
                onnx_node.params.append(NodeParam(tensor=onnx_param, idx=idx))
            else:
                if input_x in all_output_tensor:
                    onnx_node.add_parent(all_output_tensor.get(input_x))
                onnx_node.inputs.append(input_x)

        for output_y in node.output:
            if output_y in all_input_tensor:
                onnx_node.add_child(all_input_tensor.get(output_y))
            onnx_node.outputs.append(output_y)

        logger.debug(onnx_node)
        onnx_graph.node_map.setdefault(node.name, onnx_node)
    logger.info("Finish to build DAG graph.")
    return onnx_graph


def build_relations(graph, onnx_graph):
    all_input_tensor = {}
    all_output_tensor = {}
    for node in graph.node:
        name = node.name
        for input_x in node.input:
            if input_x in onnx_graph.params:
                continue
            if input_x in all_input_tensor:
                node_list = all_input_tensor.get(input_x)
                node_list.append(name)
                all_input_tensor.update({input_x: node_list})
            else:
                all_input_tensor.setdefault(input_x, [name])

        for output_y in node.output:
            if output_y in all_output_tensor:
                node_list = all_output_tensor.get(output_y)
                node_list.append(name)
                all_output_tensor.update({output_y: node_list})
            else:
                all_output_tensor.setdefault(output_y, [name])
    return all_input_tensor, all_output_tensor


def get_depthwise_node(model_path):
    model = onnx.load(model_path)
    onnx_graph = parse_model(model)
    depthwise_nodes = []
    for node_name, node in onnx_graph.node_map.items():
        if node.op_type == "Conv" and node.attrs.get("group") > 1:
            depthwise_nodes.append(node_name)
    return depthwise_nodes
