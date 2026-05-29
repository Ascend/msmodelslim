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

from typing import Any, Tuple, Dict
from abc import ABC, abstractmethod

import torch

from msmodelslim.core.graph.fast_ops_grapher.exec_observer.exec_dag import (
    ComputationGraph,
    trace_to_graph,
    GraphNode,
    GraphEdge,
)
from msmodelslim.core.graph.fast_ops_grapher.exec_observer.exec_trace import trace_module

OPERATIONS_TO_REMOVE = [
    'detach',
    'view',
    '_unsafe_view',
    'unsqueeze',
    'transpose',
    't',
    'slice',
    'select',
    'expand',
    'clone',
    'alias',
    'lift_fresh',
    '_to_copy',
    'to',
]


def _remove_simple_node(node: GraphNode) -> None:
    """移除简单操作节点，并重新连接其前驱和后继节点。

    当前只处理单输出的简单操作（如 view、transpose 等）。
    多输出操作（如 split、chunk）保留在图中，因为它们会影响数据流拓扑。

    Args:
        node: 要移除的节点（必须属于某个 graph）
    """
    graph = node.graph

    predecessors = node.get_predecessors()
    successors = node.get_successors()

    if predecessors and successors and len(node.operator.outputs) == 1:
        tensor_to_reuse = node.operator.outputs[0]
        next_edge_id = graph.next_edge_id
        for pred in predecessors:
            for succ in successors:
                new_edge = GraphEdge(
                    id_=next_edge_id, source_node_id=pred.id, target_node_id=succ.id, tensor=tensor_to_reuse
                )
                graph.add_edge(new_edge)
                next_edge_id += 1

    graph.remove_node(node.id)


class BaseExtractor(ABC):
    """BaseExtractor 是一个抽象基类，定义了从 PyTorch 模型中提取计算图的接口和基本流程。"""

    def __init__(self):
        pass

    @staticmethod
    @abstractmethod
    def create(*args, **kwargs) -> 'BaseExtractor':
        pass

    @property
    @abstractmethod
    def target_module(self) -> torch.nn.Module:
        pass

    @property
    @abstractmethod
    def dummy_inputs(self) -> Tuple[Tuple[Any], Dict[str, Any]]:
        pass

    def _extract_raw_dag(self) -> ComputationGraph:
        args, kwargs = self.dummy_inputs
        execution_trace = trace_module(self.target_module, args=args, kwargs=kwargs)
        graph = trace_to_graph(execution_trace)
        return graph

    def _post_process_dag(self, graph: ComputationGraph) -> ComputationGraph:
        nodes_to_remove = []
        for node in graph.iter_nodes():
            op_name = node.operator.op_name
            base_op_name = op_name.split('.')[0] if '.' in op_name else op_name
            if base_op_name in OPERATIONS_TO_REMOVE:
                nodes_to_remove.append(node)

        for node in nodes_to_remove:
            _remove_simple_node(node)
        graph.reconcile_edges()

        return graph

    def extract_dag(self) -> ComputationGraph:
        raw_graph = self._extract_raw_dag()
        processed_graph = self._post_process_dag(raw_graph)
        return processed_graph
