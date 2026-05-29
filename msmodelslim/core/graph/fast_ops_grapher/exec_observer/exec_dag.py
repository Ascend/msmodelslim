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

from __future__ import annotations
from typing import List, Optional, Iterator, Dict, Tuple, Callable
import networkx as nx

from msmodelslim.core.graph.fast_ops_grapher.formatters.formatter_mng import format_graph
from msmodelslim.core.graph.fast_ops_grapher.exec_observer.exec_trace import TensorInfo, OperatorRecord, ExecutionTrace


class GraphNode:
    """计算图节点类，表示一个算子。

    Attributes:
        id: 节点的唯一标识符
        operator: 算子执行记录
        graph: 所属的计算图对象（用于查询关联节点和边）
    """

    def __init__(self, id_: int, operator: OperatorRecord) -> None:
        self.id = id_
        self.operator = operator
        self.graph: Optional[ComputationGraph] = None

    def get_outgoing_edges(self) -> List[GraphEdge]:
        if self.graph is None:
            return []
        return [edge for edge in self.graph.iter_edges() if edge.source_node_id == self.id]

    def get_incoming_edges(self) -> List[GraphEdge]:
        if self.graph is None:
            return []
        return [edge for edge in self.graph.iter_edges() if edge.target_node_id == self.id]

    def get_successors(self) -> List[GraphNode]:
        if self.graph is None:
            return []
        successors = []
        for successor_id in self.graph.successors(self.id):
            node = self.graph.get_node(successor_id)
            if node is not None:
                successors.append(node)
        return successors

    def get_predecessors(self) -> List[GraphNode]:
        if self.graph is None:
            return []
        predecessors = []
        for predecessor_id in self.graph.predecessors(self.id):
            node = self.graph.get_node(predecessor_id)
            if node is not None:
                predecessors.append(node)
        return predecessors

    def format_info(self) -> dict[str, str]:
        call_stack_lines = []
        for frame in self.operator.traceback:
            call_stack_lines.append(
                f"{frame['filename']}:{frame['lineno']} [{frame['module']}.{frame['function']}] {frame['code_context'] or ''}"
            )
        return {"op_name": self.operator.op_name, "call_stack": "\n".join(call_stack_lines)}


class GraphEdge:
    """计算图边类，表示 tensor 数据流。

    Attributes:
        id: 边的唯一标识符
        source_node_id: 源节点（输出该 tensor 的算子）的 ID
        target_node_id: 目标节点（使用该 tensor 的算子）的 ID
        tensor: 流通的 tensor 的元信息
        graph: 所属的计算图对象（用于查询关联节点）
    """

    def __init__(
        self,
        id_: int,
        source_node_id: int,
        target_node_id: int,
        tensor: TensorInfo,
    ) -> None:
        self.id = id_
        self.source_node_id = source_node_id
        self.target_node_id = target_node_id
        self.tensor = tensor
        self.graph: Optional[ComputationGraph] = None

    def get_source_node(self) -> Optional[GraphNode]:
        if self.graph is None:
            return None
        return self.graph.get_node(self.source_node_id)

    def get_target_node(self) -> Optional[GraphNode]:
        if self.graph is None:
            return None
        return self.graph.get_node(self.target_node_id)

    def format_info(self) -> dict[str, str]:
        return {"varname": self.tensor.varname, "dtype": str(self.tensor.dtype), "shape": str(self.tensor.shape)}


class ComputationGraph(nx.DiGraph):
    """计算图类，继承自 networkx.DiGraph。

    **重要**: 请使用封装方法（iter_nodes、iter_edges、get_node、get_edge）
    访问节点和边，避免直接使用 nx.DiGraph 原生方法。

    原生方法（如 nodes()、edges()）返回的是 node_id，而非 GraphNode 对象。

    Attributes:
        _nodes: 节点 ID 到 GraphNode 的映射
        _edges: 边 ID 到 GraphEdge 的映射
    """

    def __init__(self) -> None:
        super().__init__()
        self._nodes: Dict[int, GraphNode] = {}
        self._edges: Dict[int, GraphEdge] = {}
        self._next_edge_id = 1

    def add_node(self, node: GraphNode) -> None:
        self._nodes[node.id] = node
        super().add_node(node.id)
        node.graph = self

    def add_edge(self, edge: GraphEdge) -> None:
        self._edges[edge.id] = edge
        super().add_edge(edge.source_node_id, edge.target_node_id, key=edge.id)
        edge.graph = self
        self._next_edge_id += 1

    def get_node(self, node_id: int) -> Optional[GraphNode]:
        return self._nodes.get(node_id)

    def get_edge(self, edge_id: int) -> Optional[GraphEdge]:
        return self._edges.get(edge_id)

    def iter_nodes(self) -> Iterator[GraphNode]:
        yield from self._nodes.values()

    def iter_edges(self) -> Iterator[GraphEdge]:
        yield from self._edges.values()

    @property
    def node_count(self) -> int:
        return len(self._nodes)

    @property
    def edge_count(self) -> int:
        return len(self._edges)

    @property
    def next_edge_id(self) -> int:
        return self._next_edge_id

    def remove_node(self, node_id: int) -> None:
        if node_id not in self._nodes:
            raise ValueError(f"Node with id {node_id} does not exist in the graph.")

        edges_to_remove = []
        for edge in self._edges.values():
            if node_id in (edge.source_node_id, edge.target_node_id):
                edges_to_remove.append(edge.id)

        for edge_id in edges_to_remove:
            removed_edge = self._edges.pop(edge_id)
            removed_edge.graph = None

        if node_id in self._nodes:
            removed_node = self._nodes.pop(node_id)
            removed_node.graph = None

        super().remove_node(node_id)

    def reconcile_edges(self) -> None:
        existing_edge_ids = set()
        for _, _, attr in self.edges(data=True):
            if 'key' in attr:
                existing_edge_ids.add(attr['key'])

        edge_ids_to_remove = []
        for edge_id in self._edges:
            if edge_id not in existing_edge_ids:
                edge_ids_to_remove.append(edge_id)

        for edge_id in edge_ids_to_remove:
            del self._edges[edge_id]

    def format(self, format_preset: str) -> str:
        return format_graph(format_preset, self)


def tensor_identifier() -> Tuple[Callable[[TensorInfo], None], Callable[[TensorInfo], None]]:
    tensor_cnt = 0
    tensor_id_map = {}

    def _assign_new_id(tensor_info: TensorInfo) -> None:
        nonlocal tensor_cnt
        mem_id = tensor_info.id
        unique_id = tensor_cnt
        tensor_info.id = unique_id
        tensor_id_map[mem_id] = unique_id
        tensor_cnt += 1

    def for_input(tensor_info: TensorInfo) -> None:
        unique_id = tensor_id_map.get(tensor_info.id, None)
        if unique_id is None:
            _assign_new_id(tensor_info)
        else:
            tensor_info.id = unique_id

    def for_output(tensor_info: TensorInfo) -> None:
        _assign_new_id(tensor_info)

    return for_input, for_output


def tensor_matcher() -> Tuple[Callable[[TensorInfo], Optional[int]], Callable[[TensorInfo, int], None]]:
    output_tensor_map = {}

    def for_input(tensor_info: TensorInfo) -> Optional[int]:
        return output_tensor_map.get(tensor_info.id, None)

    def for_output(tensor_info: TensorInfo, node_id: int) -> None:
        output_tensor_map[tensor_info.id] = node_id

    return for_input, for_output


def trace_to_graph(trace: ExecutionTrace) -> ComputationGraph:
    graph = ComputationGraph()

    tensor_identifier_input, tensor_identifier_output = tensor_identifier()
    tensor_matcher_input, tensor_matcher_output = tensor_matcher()

    for idx, operator in enumerate(trace.operators):
        node = GraphNode(id_=idx, operator=operator)
        graph.add_node(node)

        for input_tensor in {t.id: t for t in operator.inputs}.values():
            tensor_identifier_input(input_tensor)
            source_node_id = tensor_matcher_input(input_tensor)
            if source_node_id is not None:
                edge = GraphEdge(
                    id_=graph.next_edge_id,
                    source_node_id=source_node_id,
                    target_node_id=node.id,
                    tensor=input_tensor,
                )
                graph.add_edge(edge)
        for output_tensor in {t.id: t for t in operator.outputs}.values():
            tensor_identifier_output(output_tensor)
            tensor_matcher_output(output_tensor, node.id)

    return graph
