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
# pylint: disable=reimported

from collections.abc import Generator, Iterable
from typing import Optional, Any, List, Iterable, Generator, Union, Set  # noqa: F811
import torch

from ascend_utils.core.dag.dag_node_io import DagNodeIO


class DagNode:
    def __init__(
        self,
        node: Optional[Any] = None,
        name: Optional[str] = None,
        op_type: Optional[Union[str, Set[Optional[str]]]] = None,
        inputs: Optional[List[DagNodeIO]] = None,
        outputs: Optional[List[DagNodeIO]] = None,
    ):
        # node info
        self._node = node
        self._name_in_network: Optional[str] = name
        self._op_type: [Union[str, Set[str]]] = op_type if op_type is not None else type(node).__name__

        # node relationship info
        self._inputs: List[DagNodeIO] = []
        self._outputs: List[DagNodeIO] = []
        self.set_node_io(inputs, outputs)
        self.input_param = []

    def __repr__(self):
        return "{} [{}] ({}) >> * >> ({}) ".format(
            self.name,
            self.op_type,
            ",".join((x.name for x in self.input_nodes)),
            ",".join((x.name for x in self.output_nodes)),
        )

    def __rshift__(self, nodes: Union["DagNode", Iterable["DagNode"]]):
        self.add_next_node(nodes)
        return nodes

    @property
    def node(self) -> Any:
        return self._node

    @property
    def name(self) -> str:
        return self._name_in_network

    @property
    def name_in_network(self) -> str:
        return self._name_in_network

    @property
    def op_type(self) -> str:
        return self._op_type if isinstance(self._op_type, str) else "|".join((str(op) for op in self._op_type))

    @property
    def op_types(self) -> Set[str]:
        # Set type are only possible when searching for sub graph.
        return {self._op_type} if isinstance(self._op_type, str) else self._op_type

    @property
    def inputs(self) -> List[DagNodeIO]:
        return self._inputs

    @property
    def outputs(self) -> List[DagNodeIO]:
        return self._outputs

    @property
    def input_nodes(self) -> Generator["DagNode", None, None]:
        return (input_info.dag_node_from for input_info in self._inputs if input_info.dag_node_from is not None)

    @property
    def output_nodes(self) -> Generator["DagNode", None, None]:
        for output_info in self._outputs:
            for output_node in output_info.dag_nodes_to:
                if output_node is not None:
                    yield output_node

    def add_next_node(self, nodes: Union["DagNode", Iterable["DagNode"]], output_name: Optional[str] = None) -> None:
        node_list = nodes
        if not isinstance(nodes, (Generator, Iterable)):
            node_list = [nodes]
        for node in node_list:
            if not isinstance(node, DagNode):
                raise TypeError("add next node must be type DagNode")
            if output_name is None:
                output_name = "output" + str(len(self.outputs))
            output_io = DagNodeIO(output_name, output_name, node_from=self)
            output_io.add_node_to(node)
            node.inputs.append(output_io)
            self.outputs.append(output_io)

    def replace(self, node: Any, op_type: Optional[str] = None) -> None:
        if node is None:
            raise ValueError("node must be not None")
        if op_type is not None and not isinstance(op_type, str):
            raise TypeError("op_type must be type str")
        self._node = node
        self._op_type = op_type

    def set_node_io(self, inputs: Optional[List[DagNodeIO]], outputs: Optional[List[DagNodeIO]]) -> None:
        self._inputs: List[DagNodeIO] = inputs if inputs is not None else []
        for one_input in self._inputs:
            one_input.add_node_to(self)

        self._outputs: List[DagNodeIO] = outputs if outputs is not None else []
        for one_output in self._outputs:
            one_output.set_node_from(self)

    def set_input_param(self, input_param) -> None:
        for param in input_param:
            if not torch.is_tensor(param):
                self.input_param.append(param)
