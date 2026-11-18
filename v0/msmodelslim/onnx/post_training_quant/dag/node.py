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
import onnx.helper

from msmodelslim import logger


class OnnxNode:
    def __init__(self, name, op_type, attrs=None, domain=None):
        self._name = name
        self._op_type = op_type
        self._attrs = attrs or {}
        self._domain = domain or ""
        self._parent_nodes = []
        self._child_nodes = []
        self._params = []
        self._inputs = []
        self._outputs = []

    def __str__(self):
        return f'Node({self.name}): \tparent_nodes={self.parent_nodes}\tchild_nodes={self.child_nodes}\t' \
               f'inputs={self.inputs}\toutputs={self.outputs}\tattrs={self.attrs}\tparams={self.params}'

    def __repr__(self):
        return self.__str__()

    @property
    def op_type(self):
        return self._op_type

    @property
    def name(self):
        return self._name

    @property
    def attrs(self):
        return self._attrs

    @property
    def domain(self):
        return self._domain

    @property
    def params(self):
        return self._params

    @property
    def parent_nodes(self):
        return self._parent_nodes

    @property
    def child_nodes(self):
        return self._child_nodes

    @property
    def inputs(self):
        return self._inputs

    @property
    def outputs(self):
        return self._outputs

    @property
    def node(self):
        logger.debug("Export node: %r", self.name)
        inputs = self.inputs + [param.tensor.name for param in self.params]
        input_size = len(inputs)

        param_indices = [param.idx for param in self.params]
        for i, idx in enumerate(param_indices):
            inputs[idx] = self.params[i].tensor.name

        input_dices = list(set(range(input_size)).difference(set(param_indices)))
        for i, idx in enumerate(input_dices):
            inputs[idx] = self.inputs[i]

        return onnx.helper.make_node(
            op_type=self.op_type,
            inputs=inputs,
            outputs=self.outputs,
            name=self.name,
            domain=self.domain,
            **self.attrs
        )

    @op_type.setter
    def op_type(self, value):
        self._op_type = value

    @name.setter
    def name(self, value):
        self._name = value

    @attrs.setter
    def attrs(self, value):
        self._attrs = value

    @domain.setter
    def domain(self, value):
        self._domain = value

    @params.setter
    def params(self, value):
        self._params = value

    @parent_nodes.setter
    def parent_nodes(self, value):
        self._parent_nodes = value

    @child_nodes.setter
    def child_nodes(self, value):
        self._child_nodes = value

    @inputs.setter
    def inputs(self, value):
        self._inputs = value

    @outputs.setter
    def outputs(self, value):
        self._outputs = value

    def add_parent(self, nodes):
        if isinstance(nodes, list):
            self._parent_nodes.extend(nodes)
        else:
            self._parent_nodes.append(nodes)

    def add_child(self, node):
        if isinstance(node, list):
            self._child_nodes.extend(node)
        else:
            self._child_nodes.append(node)


class QuantizableOnnxNode(OnnxNode):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._activation = None
        self._activation_offset = None
        self._activation_scale = None
        self._weight_scale = None
        self._weight_offset = None
        self._quant_weight = None

    @property
    def activation(self):
        return self._activation

    @property
    def activation_offset(self):
        return self._activation_offset

    @property
    def activation_scale(self):
        return self._activation_scale

    @property
    def weight_offset(self):
        return self._weight_offset

    @property
    def weight_scale(self):
        return self._weight_scale

    @property
    def quant_weight(self):
        return self._quant_weight

    @activation.setter
    def activation(self, value):
        self._activation = value

    @activation_offset.setter
    def activation_offset(self, value):
        self._activation_offset = value

    @activation_scale.setter
    def activation_scale(self, value):
        self._activation_scale = value

    @weight_offset.setter
    def weight_offset(self, value):
        self._weight_offset = value

    @weight_scale.setter
    def weight_scale(self, value):
        self._weight_scale = value

    @quant_weight.setter
    def quant_weight(self, value):
        self._quant_weight = value
