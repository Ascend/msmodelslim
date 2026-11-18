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
from collections import OrderedDict

import onnx
import numpy as np
from onnx import helper

from ascend_utils.common.security import SafeWriteUmask
from msmodelslim.onnx.post_training_quant.dag.node import OnnxNode, QuantizableOnnxNode
from msmodelslim.onnx.post_training_quant.dag.param import NodeParam, Tensor
from msmodelslim import logger


class OnnxGraph:
    def __init__(self, inputs, outputs, opset_import, ir_version, name, domain):
        self._inputs = inputs
        self._outputs = outputs
        self._node_map = OrderedDict()
        self._opset_import = opset_import
        self._ir_version = ir_version
        self._name = name
        self._domain = domain
        self._params = {}

    @property
    def params(self):
        return self._params

    @property
    def inputs(self):
        return self._inputs

    @property
    def outputs(self):
        return self._outputs

    @property
    def node_map(self):
        return self._node_map

    @property
    def opset_import(self):
        return self._opset_import

    @property
    def ir_version(self):
        return self._ir_version

    @property
    def name(self):
        return self._name

    @property
    def domain(self):
        return self._domain

    @property
    def nodes(self):
        logger.info("Export nodes...")
        return [dag_node.node for _, dag_node in self.node_map.items()]

    @property
    def initializers(self):
        logger.info("Export initializers...")
        return [tensor.initializer for _, tensor in self.params.items()]

    @staticmethod
    def gen_ascend_quant_onnx_node(node_name: str, quant_bit: int, scale: float, offset: float):
        if scale == 0:
            raise ValueError("scale cannot be zero")
        quant_node_attrs = {
            "offset": offset,
            "quant_bit": quant_bit,
            "scale": 1 / scale
        }
        quant_node = OnnxNode(name=node_name, op_type="AscendQuant", attrs=quant_node_attrs, domain="")
        quant_node.outputs = [node_name + "_output"]
        return quant_node

    @staticmethod
    def gen_ascend_dequant_onnx_node(node_name, d_scale, w_scale):
        deq_scale = np.multiply(np.array(w_scale),
                                np.array(d_scale)).reshape([-1])
        deq_scale = np.squeeze(deq_scale)
        deq_scale = deq_scale.astype(np.float32)

        uint32_deq_scale = np.frombuffer(deq_scale, np.uint32)
        uint64_deq_scale = uint32_deq_scale.astype(np.uint64)

        deq_scale_name = node_name + "_dequant_scale"
        deq_scale_param = Tensor(name=deq_scale_name, dtype=onnx.TensorProto.UINT64,
                                 shape=uint64_deq_scale.shape, value=uint64_deq_scale)
        param = NodeParam(tensor=deq_scale_param, idx=1)

        dequant_node = OnnxNode(name=node_name, op_type="AscendDequant")
        dequant_node.outputs = [node_name + "_output"]
        dequant_node.params = [param]

        return dequant_node

    @node_map.setter
    def node_map(self, value):
        self._node_map = value

    def convert_graph_df_quant(self):
        """
        For data-free situation, we should insert the AscendQuant node befor the quantized conv or matmul node, and
        insert the AscendDequant node after the the quantized conv or matmul node
        """
        # To solve the problem that dict cannot be changed when iterated
        for name in list(self.node_map.keys()):
            dag_node = self.node_map.get(name)
            if not isinstance(dag_node, QuantizableOnnxNode):
                continue

            if len(dag_node.params) > 0:
                w_param = dag_node.params[0]
                w_param.tensor.value = dag_node.quant_weight
                w_param.tensor.dtype = onnx.TensorProto.INT8
                if dag_node.op_type == "Gemm" and dag_node.attrs.get("transB"):
                    w_param.tensor.value = w_param.tensor.value.T
                    w_param.shape = list(w_param.tensor.value.shape)
                    dag_node.attrs['transB'] = 0

            # quantize bias
            deq_scale = np.multiply(np.array(dag_node.weight_scale),
                                    np.array(dag_node.activation_scale)).reshape([-1])
            deq_scale = np.squeeze(deq_scale)
            if len(dag_node.params) > 1:  # has bias
                b_param = dag_node.params[1]
                bias = b_param.tensor.value
                quant_bias = (bias / deq_scale).squeeze().astype(np.int32)
                b_param.tensor.value = quant_bias
                b_param.dtype = onnx.TensorProto.INT32

            quant_node_name = dag_node.name + "_quant"
            quant_node = self.gen_ascend_quant_onnx_node(node_name=quant_node_name, quant_bit=8,
                                                         scale=dag_node.activation_scale,
                                                         offset=dag_node.activation_offset)
            self.insert_node(node=dag_node, inserted_node=quant_node, mode="before")

            dequant_node_name = dag_node.name + "_dequant"
            dequant_node = self.gen_ascend_dequant_onnx_node(node_name=dequant_node_name,
                                                             d_scale=dag_node.activation_scale,
                                                             w_scale=dag_node.weight_scale)
            self.insert_node(node=dag_node, inserted_node=dequant_node, mode="after")

    def convert_graph_lf_quant(self):
        """
        For label-free situation, We use the onnxruntime quantization api to quantize the model.
        After quantization, Conv is replaced with the QuantizeLinear -> QLinearConv ->DequantizeLinear block.
        It can run on cpu, but not support on npu. So we must convert block
        QuantizeLinear -> QLinearConv ->DequantizeLinear to block AscendQuant -> Conv -> AscendDequant.
        """

        model_outputs = set(output.name for output in self.outputs)

        # To solve the problem that dict cannot be changed when iterated
        for name in list(self.node_map.keys()):
            node = self.node_map.get(name)
            if node and node.op_type == "QLinearConv":
                res = self.convert_qconv2conv(node)
                quantized_node, x_scale, x_offset, w_scale = (
                    res.get('conv_node', None), res.get('x_scale', None),
                    res.get('x_offset', None), res.get('w_scale', None)
                )
            elif node and node.op_type == "QGemm":
                res = self.convert_qgemm2gemm(node)
                quantized_node, x_scale, x_offset, w_scale = (
                    res.get('gemm_node', None), res.get('x_scale', None),
                    res.get('x_offset', None), res.get('b_scale', None)
                )
            elif node and node.op_type == "QLinearMatMul":
                res = self.convert_qmm2mm(node)
                quantized_node, x_scale, x_offset, w_scale = (
                    res.get('mm_node', None), res.get('x_scale', None),
                    res.get('x_offset', None), res.get('w_scale', None)
                )
            else:
                continue

            parent_node_name = quantized_node.parent_nodes[0]
            parent_node = self.node_map.get(parent_node_name)
            if parent_node.op_type == "QuantizeLinear":
                self.remove_node(parent_node)

            quant_node_name = quantized_node.name + "_quant"
            quant_node = self.gen_ascend_quant_onnx_node(node_name=quant_node_name, quant_bit=8,
                                                         scale=x_scale.tensor.value, offset=x_offset.tensor.value)
            self.insert_node(quantized_node, quant_node, mode="before")

            dequantize_linear_node = None
            for child in quantized_node.child_nodes:
                child_node = self.node_map.get(child)
                if child_node.op_type == "DequantizeLinear":
                    self.remove_node(child_node)
                    dequantize_linear_node = child_node

            node_name = quantized_node.name + "_dequant"
            dequant_node = self.gen_ascend_dequant_onnx_node(node_name=node_name, d_scale=x_scale.tensor.value,
                                                             w_scale=w_scale.tensor.value)

            if dequantize_linear_node and set(dequantize_linear_node.outputs).intersection(model_outputs):
                dequant_node.outputs = dequantize_linear_node.outputs

            self.insert_node(quantized_node, dequant_node, mode="after")

            if quantized_node.op_type == "MatMul":
                self.optimize_mm_dequant_add_subgraph(quantized_node, w_scale.tensor.value, x_scale.tensor.value)

        self.remove_unused_params()

    def remove_unused_params(self):
        graph_params = set(self.params.keys())
        node_params = set()
        for _, node in self.node_map.items():
            for param in node.params:
                node_params.add(param.tensor.name)
        unused_params = graph_params - node_params
        for unused_param in unused_params:
            self.params.pop(unused_param)

    def optimize_mm_dequant_add_subgraph(self, matmul_node, w_scale, x_scale):
        """
        For subgraph AscendQuant->MatMul->AscendDequant->Add, convert it to AscendQuant->MatMul->Add->AscendDequant.
        And quantize the param of Add.
        """
        child_node = self.node_map.get(matmul_node.child_nodes[0])
        if child_node.op_type == "AscendDequant" and len(child_node.child_nodes) == 1:
            grandchild_node = self.node_map.get(child_node.child_nodes[0])
            if grandchild_node.op_type == "Add" and len(grandchild_node.params) == 1:
                # quantize bias
                deq_scale = np.multiply(w_scale, x_scale).reshape([-1])
                b_param = grandchild_node.params[0]
                bias = b_param.tensor.value
                quant_bias = np.true_divide(bias, deq_scale)
                quant_bias = quant_bias.reshape(bias.shape).astype(np.int32)
                b_param.tensor.value = quant_bias

                b_param.tensor.dtype = onnx.TensorProto.INT32
                b_param.idx = 1
                grandchild_node.params = [b_param]
                self._exchange_two_continuous_nodes(child_node, grandchild_node)

    def convert_qconv2conv(self, q_linear_conv_node):
        """
        convert QLinearConv node to Conv node
        """
        if len(q_linear_conv_node.params) < 7:
            raise ValueError("The size of q_linear_conv_node's params couldn't be smaller than 7.")
        x_scale = q_linear_conv_node.params[0]
        x_offset = q_linear_conv_node.params[1]
        weight = q_linear_conv_node.params[2]
        w_scale = q_linear_conv_node.params[3]
        has_bias = len(q_linear_conv_node.params) == 8

        weight.idx = 1
        conv_node = OnnxNode(name=q_linear_conv_node.name, op_type="Conv", attrs=q_linear_conv_node.attrs)

        if has_bias:
            bias = q_linear_conv_node.params[7]
            bias.idx = 2
            conv_node.params = [weight, bias]
        else:
            conv_node.params = [weight]

        self.replace_node(q_linear_conv_node, conv_node)
        return {'conv_node': conv_node, 'x_scale': x_scale, 'x_offset': x_offset, 'w_scale': w_scale}

    def convert_qgemm2gemm(self, q_gemm_node):
        """
        convert QGemm node to Gemm node
        """
        if len(q_gemm_node.params) < 7:
            raise ValueError("The size of q_gemm_node's params couldn't be smaller than 7.")
        x_scale = q_gemm_node.params[0]
        x_offset = q_gemm_node.params[1]
        param_b = q_gemm_node.params[2]
        b_scale = q_gemm_node.params[3]
        has_c = len(q_gemm_node.params) == 8

        param_b.idx = 1
        attrs = q_gemm_node.attrs
        if attrs.get("transB"):
            input_b = param_b.tensor.value.T
            param_b.tensor.value = input_b
            param_b.tensor.shape = list(input_b.shape)
            attrs['transB'] = 0
        gemm_node = OnnxNode(name=q_gemm_node.name, op_type="Gemm", attrs=attrs)
        if has_c:
            param_c = q_gemm_node.params[5]
            param_c.idx = 2
            gemm_node.params = [param_b, param_c]
        else:
            gemm_node.params = [param_b]

        self.replace_node(q_gemm_node, gemm_node)
        return {'gemm_node': gemm_node, 'x_scale': x_scale, 'x_offset': x_offset, 'b_scale': b_scale}

    def convert_qmm2mm(self, q_matmul_node):
        if len(q_matmul_node.params) != 7:
            raise ValueError("The size of q_linear_conv_node's params isn't equal to 7.")
        x_scale = q_matmul_node.params[0]
        x_offset = q_matmul_node.params[1]
        weight = q_matmul_node.params[2]
        w_scale = q_matmul_node.params[3]
        weight.idx = 1
        mm_node = OnnxNode(name=q_matmul_node.name, op_type="MatMul", attrs=q_matmul_node.attrs)
        mm_node.params = [weight]

        self.replace_node(q_matmul_node, mm_node)

        return {'mm_node': mm_node, 'x_scale': x_scale, 'x_offset': x_offset, 'w_scale': w_scale}

    def reduce_redundant_quant_node(self):

        def get_equal_ascend_quant_groups(cur_node):
            """
            Divide AscendQuant nodes with the same inputs into a group.
            Only AscendQuant nodes in the same group can be simplified.
            """
            quant_node_list = []
            for child in cur_node.child_nodes:
                if self.node_map.get(child).op_type == "AscendQuant":
                    quant_node_list.append(child)
            inputs_and_name_dic = {}
            for node_name in quant_node_list:
                # node input  as key
                inputs_key = self.node_map.get(node_name).inputs[0]
                # node name as value
                inputs_and_name_dic.setdefault(inputs_key, []).append(node_name)
            groups = list(inputs_and_name_dic.values())
            return groups

        def reduce_ascend_quant_group(group):
            """
            Simplify AscendQuant nodes in the same group by DAG. Save first one and remove others.
            """
            next_node_list = []  # save children of redundant AscendQuant
            head_quant_node = group[0]
            for node_name in group[1:]:
                redundant_quant_node = self.node_map.pop(node_name)
                next_node_list.extend(redundant_quant_node.child_nodes)
                # remove redundant_quant_node
                for parent in redundant_quant_node.parent_nodes:
                    parent_node = self.node_map.get(parent)
                    parent_node.child_nodes.remove(redundant_quant_node.name)
                for child in redundant_quant_node.child_nodes:
                    child_node = self.node_map.get(child)
                    child_node.parent_nodes.remove(redundant_quant_node.name)
                    # change child inputs
                    redundant_input = redundant_quant_node.outputs[0]
                    redundant_input_idx = child_node.inputs.index(redundant_input)
                    child_node.inputs[redundant_input_idx] = self.node_map.get(head_quant_node).outputs[0]
            # change connection in DAG
            self.node_map.get(head_quant_node).child_nodes.extend(next_node_list)
            for next_node in next_node_list:
                self.node_map.get(next_node).parent_nodes.append(head_quant_node)

        # change DAG by child/parent and inputs/outputs
        for node in list(self.node_map.values()):
            if not node or node.op_type == "AscendQuant":
                continue
            ascend_quant_group_list = get_equal_ascend_quant_groups(node)
            for ascend_quant_group in ascend_quant_group_list:
                if len(ascend_quant_group) >= 2:
                    reduce_ascend_quant_group(ascend_quant_group)

    def build_model(self):
        graph = onnx.helper.make_graph(nodes=self.nodes, name=self.name, inputs=self.inputs,
                                       outputs=self.outputs, initializer=self.initializers)
        logger.info("Build onnx model...")
        onnx_model = onnx.helper.make_model(graph,
                                            opset_imports=[helper.make_opsetid(domain="", version=11)],
                                            ir_version=self.ir_version)
        return onnx_model

    def save_model(self, model_path):
        onnx_model = self.build_model()
        with SafeWriteUmask():
            onnx.save(onnx_model, model_path)

    def remove_node(self, node: OnnxNode):
        logger.debug("Remove the node %s", node.name)
        if len(node.parent_nodes) > 1:
            raise NotImplementedError("Node that has more than 1 parent node can not be removed!")
        if len(node.inputs) != len(node.outputs):
            raise NotImplementedError("Node that the length of inputs and outputs are not equal can not be removed!")

        # 假设要移除节点A，把A的子节点全部加到A父节点的子节点中，父节点的输出不用改
        for parent in node.parent_nodes:
            parent_node = self.node_map.get(parent)
            if node.name in parent_node.child_nodes:
                parent_node.child_nodes.remove(node.name)
                parent_node.add_child(node.child_nodes)

        # 假设要移除节点A，把A的父节点全部加到A的子节点的父节点中，并把子节点的输入更改为A的输入
        for child in node.child_nodes:
            child_node = self.node_map.get(child)
            if node.name in child_node.parent_nodes:
                child_node.parent_nodes.remove(node.name)
                child_node.add_parent(node.parent_nodes)

            for input_x, output in zip(node.inputs, node.outputs):
                if output in child_node.inputs:
                    idx = child_node.inputs.index(output)
                    child_node.inputs[idx] = input_x

        self.node_map.pop(node.name)

    def replace_node(self, node, replaced_node):
        replaced_node.inputs = node.inputs
        replaced_node.outputs = node.outputs
        replaced_node.parent_nodes = node.parent_nodes
        replaced_node.child_nodes = node.child_nodes

        for parent in node.parent_nodes:
            parent_node = self.node_map.get(parent)
            if node.name in parent_node.child_nodes:
                idx = parent_node.child_nodes.index(node.name)
                parent_node.child_nodes[idx] = replaced_node.name

        for child in node.child_nodes:
            child_node = self.node_map.get(child)
            if node.name in child_node.parent_nodes:
                idx = child_node.parent_nodes.index(node.name)
                child_node.parent_nodes[idx] = replaced_node.name

        self.node_map.pop(node.name)
        self.node_map.setdefault(replaced_node.name, replaced_node)

    def insert_node(self, node: OnnxNode, inserted_node: OnnxNode, mode):
        logger.debug("Insert node %s %s the node %s", inserted_node.name, mode, node.name)
        if mode == "before":
            self._insert_forward_node(node, inserted_node)
        elif mode == "after":
            if len(node.outputs) != len(inserted_node.outputs):
                raise ValueError("The number of outputs of the node and inserted node must be same when mode is after.")
            self._insert_backward_node(node, inserted_node)
        else:
            raise ValueError("The value of mode must be before or after.")

        self.node_map.setdefault(inserted_node.name, inserted_node)
        for param in inserted_node.params:
            self.params.setdefault(param.tensor.name, param.tensor)

    def _insert_backward_node(self, node, inserted_node):
        if len(node.outputs) != len(inserted_node.outputs):
            raise ValueError("The length of inserted node outputs is not equal to the length of node outputs!")
        for child in node.child_nodes:
            child_node = self.node_map.get(child)
            if node.name in child_node.parent_nodes:
                idx = child_node.parent_nodes.index(node.name)
                child_node.parent_nodes[idx] = inserted_node.name
            # 更新子节点的输入
            for node_output, inserted_node_output in zip(node.outputs, inserted_node.outputs):
                if node_output in child_node.inputs:
                    idx = child_node.inputs.index(node_output)
                    child_node.inputs[idx] = inserted_node_output

        inserted_node.inputs = node.outputs
        inserted_node.parent_nodes = [node.name]
        inserted_node.child_nodes = node.child_nodes

        node.child_nodes = [inserted_node.name]

        for i, n_output in enumerate(node.outputs):
            for g_output in self.outputs:
                if n_output == g_output.name:  # node is the last node of the graph.
                    node.outputs[i] = node.name + "_output" + str(i)
                    inserted_node.outputs[i] = n_output

    def _insert_forward_node(self, node, inserted_node):
        if len(node.inputs) != len(inserted_node.outputs):
            raise ValueError("The length of inserted node outputs is not equal to the length of node inputs!")
        for parent in node.parent_nodes:
            parent_node = self.node_map.get(parent)
            if node.name in parent_node.child_nodes:
                idx = parent_node.child_nodes.index(node.name)
                parent_node.child_nodes[idx] = inserted_node.name

        inserted_node.add_parent(node.parent_nodes)
        inserted_node.add_child(node.name)
        inserted_node.inputs = node.inputs

        node.inputs = inserted_node.outputs
        node.parent_nodes = [inserted_node.name]

    def _exchange_two_continuous_nodes(self, node1: OnnxNode, node2: OnnxNode):
        node2.inputs = node1.inputs
        node1.inputs = node2.outputs

        # update children nodes and parent nodes
        if node2.name in node1.child_nodes:
            node1.child_nodes.remove(node2.name)
        if node1.name in node2.parent_nodes:
            node2.parent_nodes.remove(node1.name)

        for child_name in node2.child_nodes:
            child_node = self.node_map.get(child_name)
            if node2.outputs[0] in child_node.inputs:
                index = child_node.inputs.index(node2.outputs[0])
                child_node.inputs[index] = node1.outputs[0]

            if node2.name in child_node.parent_nodes:
                child_node.parent_nodes.remove(node2.name)
            child_node.add_parent(node1.name)
            node1.add_child(child_name)

        for parent_name in node1.parent_nodes:
            parent_node = self.node_map.get(parent_name)
            parent_node.add_child(node2.name)
            if node1.name in parent_node.child_nodes:
                parent_node.child_nodes.remove(node1.name)
            node2.add_parent(parent_name)

        node1.parent_nodes = []
        node1.add_parent(node2.name)

        node1.child_nodes = []
        node2.add_child(node1.name)
