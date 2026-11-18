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
# pylint: disable=unnecessary-comprehension,use-yield-from

import copy
from abc import abstractmethod
from itertools import chain
from queue import Queue
from typing import List, Any, Optional, Type, Generator, Dict, Union

from ascend_utils.common.security import check_element_type
from ascend_utils.common.security import check_type
from ascend_utils.common.utils import FullPermutation
from ascend_utils.core.dag.dag_node import DagNode


class DirectedAcyclicGraph:
    def __init__(self, network: Any):
        self._network = network
        self._dag_node_list: List[DagNode] = []

    @property
    def network(self):
        return self._network

    @property
    def dag_node_list(self) -> List[DagNode]:
        return self._dag_node_list

    @staticmethod
    def _clone_sub_graph(node_list_in_calc_order: List[DagNode]) -> List[DagNode]:
        if node_list_in_calc_order is None or not isinstance(node_list_in_calc_order, list):
            return []
        dag_node_list = [DagNode(op_type=node.op_types, name=node.name) for node in node_list_in_calc_order]
        dag_node_map = {node.name: node for node in dag_node_list}
        for node in node_list_in_calc_order:
            node_clone = dag_node_map[node.name]
            node_clone.add_next_node(dag_node_map[node_output.name] for node_output in node.output_nodes)
        return dag_node_list

    @staticmethod
    def _get_node_input_node(sub_graph):
        # check 图只有一个输入和一个输出
        input_node = None
        output_node = None
        for node in sub_graph:
            if len(node.inputs) == 0:
                if input_node is not None:
                    raise ValueError("There can only be one input node.")
                else:
                    input_node = node
            if len(node.outputs) == 0:
                if output_node is not None:
                    raise ValueError("There can only be one output node.")
                else:
                    output_node = node

        for node in sub_graph:
            for linked_node in chain(node.input_nodes, node.output_nodes):
                if linked_node not in sub_graph:
                    raise ValueError("There can only be one input/output node.")

        node_names = set()
        for node in sub_graph:
            if node.name in node_names:
                raise ValueError("The node name must be different.")
            node_names.add(node.name)

        if input_node is None or output_node is None:
            raise ValueError("You must have an input and an output.")

        return input_node

    @staticmethod
    def _get_node_list_in_calc_order(sub_graph, input_node):
        # Sort by Calculation Order
        node_list_in_calc_order = []
        calculating_nodes: Queue[DagNode] = Queue()
        calculating_nodes.put(input_node)
        while len(node_list_in_calc_order) < len(sub_graph) and not calculating_nodes.empty():
            calculating_node = calculating_nodes.get()
            node_list_in_calc_order.append(calculating_node)
            for next_will_calc_node in calculating_node.output_nodes:
                if all((needed_input in node_list_in_calc_order for needed_input in next_will_calc_node.input_nodes)):
                    calculating_nodes.put(next_will_calc_node)

        return node_list_in_calc_order

    @staticmethod
    def _get_proper_combination_of_outputs(ori_graph_outputs: List[DagNode], sub_graph_outputs: List[DagNode]):
        # check output count
        if len(ori_graph_outputs) != len(sub_graph_outputs):
            return

        # Preprocessing op type classes
        for seq in FullPermutation().get_all_permutations(len(sub_graph_outputs)):
            for index_sub_node, index_ori_node in enumerate(seq):
                if ori_graph_outputs[index_ori_node].op_type not in sub_graph_outputs[index_sub_node].op_types:
                    break
            else:
                yield {
                    sub_graph_outputs[index_sub_node].name: ori_graph_outputs[index_ori_node]
                    for index_sub_node, index_ori_node in enumerate(seq)
                }

    def search_nodes_by_class(self, cls: Type) -> Generator[DagNode, None, None]:
        check_type(cls, type, param_name="cls")
        for dag_node in self._dag_node_list:
            if isinstance(dag_node.node, cls):
                yield dag_node

    def search_nodes_by_op_type(self, op_types: Union[str, List[str]]) -> Generator[DagNode, None, None]:
        if isinstance(op_types, str):
            op_type_list = [op_types]
        else:
            check_element_type(op_types, element_type=str, value_type=list, param_name="op_types")
            op_type_list = op_types
        for dag_node in self._dag_node_list:
            if dag_node.op_type in op_type_list:
                yield dag_node

    def get_node_by_name(self, name: str) -> Optional[DagNode]:
        check_type(name, str, param_name="name")
        dag_node_get = None
        for dag_node in self._dag_node_list:
            if dag_node.name == name:
                dag_node_get = dag_node
                break

        return dag_node_get

    def get_nodes_by_name_prefix(self, name_prefix: str) -> Generator[DagNode, None, None]:
        check_type(name_prefix, str, param_name="name_prefix")
        for dag_node in self._dag_node_list:
            if dag_node.name.startswith(name_prefix):
                yield dag_node

    def search_sub_graph(self, sub_graph: List[DagNode]) -> Generator[Dict[str, DagNode], None, None]:
        """
        search sub graph
        Args:
            sub_graph: nodes in sub graph
                    Only one node input and one node output graph are supported.
                    Each node name must be unique.

        Returns:
            generator. The element is a dictionary, and the key of the dictionary is the node name in sub graph.

        Examples:
        >>> conv = DagNode(op_type="Conv2d", name="c")
        >>> bn = DagNode(op_type="BatchNorm2d", name="b")
        >>> conv >> bn
        >>> dag.search_sub_graph([conv, bn])
        """
        if sub_graph is None or not isinstance(sub_graph, list):
            raise TypeError("sub graph must be list of DagNode.")
        if len(sub_graph) == 0:
            return

        for input_node, node_list_in_calc_order in self._parse_sub_graph(sub_graph):
            for node in self.dag_node_list:
                if node.op_type not in input_node.op_types:
                    continue
                for search_out in self._search_by_calc_order({input_node.name: node}, node_list_in_calc_order, 0):
                    yield search_out

    @abstractmethod
    def get_params(self) -> int:
        pass

    def _parse_sub_graph(self, sub_graph: List[DagNode]):
        for node in sub_graph:
            if None in node.op_types and (len(list(node.input_nodes)) > 1 or len(list(node.output_nodes)) > 1):
                raise ValueError("The node whose op type is None must have only one input and one output now.")

        input_node = self._get_node_input_node(sub_graph)
        node_list_in_calc_order = self._get_node_list_in_calc_order(sub_graph, input_node)
        for one_sub_graph in self._get_possible_sub_graph(node_list_in_calc_order):
            if len(one_sub_graph) == 0:
                continue
            input_node = self._get_node_input_node(one_sub_graph)
            node_list_in_calc_order = self._get_node_list_in_calc_order(one_sub_graph, input_node)
            if len(node_list_in_calc_order) == 0:
                raise ValueError("graph must has node.")
            yield node_list_in_calc_order[0], node_list_in_calc_order

    def _get_possible_sub_graph(self, node_list_in_calc_order: List[DagNode]) -> Generator[List[DagNode], None, None]:
        none_names = [node.name for node in node_list_in_calc_order if None in node.op_types]
        if len(none_names) == 0:
            yield node_list_in_calc_order
            return
        for idxes in FullPermutation.get_all_combinations([2] * len(none_names)):
            now_is_none_names = [name for index, name in enumerate(none_names) if idxes[index] == 0]
            yield self._create_new_sub_graph(node_list_in_calc_order, now_is_none_names)

    def _create_new_sub_graph(self, node_list_in_calc_order: List[DagNode], none_names: List[str]) -> List[DagNode]:
        dag_node_list = self._clone_sub_graph(node_list_in_calc_order)
        for node in reversed(dag_node_list):
            if node.name not in none_names:
                continue
            dag_node_list.remove(node)
            if len(node.inputs) == 0:
                continue
            if len(node.inputs) > 1:
                raise ValueError("The node whose op type is None must have only one input.")
            node_input = node.inputs[0]
            node_input.dag_nodes_to.remove(node)
            node_input.dag_nodes_to.extend(list(node.output_nodes))
            for output_of_node in node.outputs:
                for node_output in output_of_node.dag_nodes_to:
                    node_output.inputs.remove(output_of_node)
                    node_output.inputs.append(node_input)
        return dag_node_list

    def _search_by_calc_order(
        self, matched_nodes: Dict[str, DagNode], node_list_in_calc_order: List[DagNode], index: int
    ):
        if index >= len(node_list_in_calc_order):
            return

        sub_graph_node = node_list_in_calc_order[index]
        node_name = sub_graph_node.name
        ori_graph_node = matched_nodes.get(node_name, None)

        # check op type
        if ori_graph_node is None or ori_graph_node.op_type not in sub_graph_node.op_types:
            return

        # check node inputs, The first node does not need check
        if index != 0:
            sub_graph_input_list = [matched_nodes.get(in_sub.name, None) for in_sub in sub_graph_node.input_nodes]
            ori_graph_input_list = list(ori_graph_node.input_nodes)
            sub_graph_input_list.sort(key=lambda x: x.name)
            ori_graph_input_list.sort(key=lambda x: x.name)
            if sub_graph_input_list != ori_graph_input_list:
                return

        # last node, just return dict
        if index == len(node_list_in_calc_order) - 1:
            yield copy.copy(matched_nodes)
            return

        # check output
        ori_graph_outputs = [node for node in ori_graph_node.output_nodes]
        sub_graph_outputs = [node for node in sub_graph_node.output_nodes]

        # If a proper combination of output parameters is available.
        for proper_combination in self._get_proper_combination_of_outputs(ori_graph_outputs, sub_graph_outputs):
            matched_nodes.update(proper_combination)
            # the search continues based on the calculation order.
            for search_out in self._search_by_calc_order(matched_nodes, node_list_in_calc_order, index + 1):
                yield search_out

    def _remove_one_node(self, node: DagNode):
        if node in self.dag_node_list:
            self.dag_node_list.remove(node)

        for input_info in node.inputs:
            for output_info in node.outputs:
                input_info.add_node_to(output_info.dag_nodes_to)
                if node in input_info.dag_nodes_to:
                    input_info.dag_nodes_to.remove(node)

        for output_node in node.output_nodes:
            for output_info in node.outputs:
                if output_info in output_node.inputs:
                    output_node.inputs.remove(output_info)
            output_node.inputs.extend(node.inputs)
