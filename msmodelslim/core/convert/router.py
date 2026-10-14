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

IR graph router: shortest-path (or constrained) routing between IR nodes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from msmodelslim.core.convert.edges import RouteConstraints, TransformEdge
from msmodelslim.core.convert.types import IRKind, LossLevel
from msmodelslim.utils.exception import UnsupportedError

if TYPE_CHECKING:
    from msmodelslim.core.convert.protocol import IIRTransformProcessor


class IRRouter:
    """
    Resolve transform paths on a registered IR graph.

    Usage::

        router = IRRouter.default()
        edges = router.resolve(src_ir=IRKind.FP8_BLOCK, dst_ir=IRKind.W8A8_MXFP8)
    """

    _DEFAULT: IRRouter | None = None

    def __init__(self) -> None:
        self._edges: list[TransformEdge] = []
        self._processors: dict[str, IIRTransformProcessor] = {}

    @classmethod
    def default(cls) -> IRRouter:
        if cls._DEFAULT is None:
            from msmodelslim.processor.convert.registry import register_convert_processors

            cls._DEFAULT = register_convert_processors(cls())
        return cls._DEFAULT

    def register_edge(self, edge: TransformEdge) -> None:
        self._edges.append(edge)

    def register_processor(self, processor: IIRTransformProcessor) -> None:
        if processor.requires_forward or processor.requires_calibration:
            raise UnsupportedError(
                f"Convert cannot register processor {processor.name}: "
                "requires_forward and requires_calibration must be False.",
            )
        self._processors[processor.name] = processor
        self.register_edge(
            TransformEdge(
                src_ir=processor.src_ir,
                dst_ir=processor.dst_ir,
                processor_name=processor.name,
                loss_level=LossLevel(processor.loss_level)
                if processor.loss_level in LossLevel._value2member_map_
                else LossLevel.LOSSY,
            ),
        )

    def resolve(
        self,
        src_ir: IRKind,
        dst_ir: IRKind,
        constraints: RouteConstraints | None = None,
    ) -> list[TransformEdge]:
        """
        Return ordered edges from ``src_ir`` to ``dst_ir``.

        If ``constraints.explicit_route`` is set, validate and map IR names to edges.
        Otherwise run shortest-path on the registered graph.
        """
        if src_ir == dst_ir:
            return []

        if constraints and constraints.explicit_route:
            return self._route_from_explicit(constraints.explicit_route)

        path = self._shortest_path(src_ir, dst_ir, constraints)
        if path is None:
            raise UnsupportedError(
                f"No route from {src_ir.value} to {dst_ir.value}. "
                "Register processors or specify an explicit route in convert_rules.",
            )
        return path

    def validate_route(self, route_ir_names: list[IRKind]) -> None:
        """Ensure consecutive IR kinds are connected by a registered edge."""
        for i in range(len(route_ir_names) - 1):
            src, dst = route_ir_names[i], route_ir_names[i + 1]
            if not any(e.src_ir == src and e.dst_ir == dst for e in self._edges):
                raise UnsupportedError(f"Missing edge {src.value} -> {dst.value}")

    def get_processor(self, name: str) -> IIRTransformProcessor:
        if name not in self._processors:
            raise UnsupportedError(f"Processor {name!r} is not registered on IRRouter.")
        return self._processors[name]

    def _route_from_explicit(self, route_ir_names: list[IRKind]) -> list[TransformEdge]:
        self.validate_route(route_ir_names)
        edges: list[TransformEdge] = []
        for i in range(len(route_ir_names) - 1):
            src, dst = route_ir_names[i], route_ir_names[i + 1]
            matched = [e for e in self._edges if e.src_ir == src and e.dst_ir == dst]
            if not matched:
                raise UnsupportedError(f"No edge for explicit step {src.value} -> {dst.value}")
            edges.append(matched[0])
        return edges

    def _shortest_path(
        self,
        src: IRKind,
        dst: IRKind,
        constraints: RouteConstraints | None,  # noqa: F821
    ) -> list[TransformEdge] | None:
        # Dijkstra on small IR graph
        import heapq

        forbidden = set(constraints.forbidden_edges) if constraints else set()
        required = set(constraints.required_nodes) if constraints else set()

        dist: dict[IRKind, int] = {src: 0}
        prev: dict[IRKind, tuple[IRKind, TransformEdge] | None] = {src: None}
        heap: list[tuple[int, IRKind]] = [(0, src)]

        while heap:
            d, node = heapq.heappop(heap)
            if d > dist.get(node, 10**9):
                continue
            if node == dst:
                break
            for edge in self._edges:
                if edge.src_ir != node:
                    continue
                if (edge.src_ir, edge.dst_ir) in forbidden:
                    continue
                new_dist = d + edge.cost
                if new_dist < dist.get(edge.dst_ir, 10**9):
                    dist[edge.dst_ir] = new_dist
                    prev[edge.dst_ir] = (node, edge)
                    heapq.heappush(heap, (new_dist, edge.dst_ir))

        if dst not in prev:
            return None

        if required and not required.issubset(dist.keys()):
            return None

        # Reconstruct edge list
        edges_rev: list[TransformEdge] = []
        cur = dst
        while cur != src:
            if cur not in prev or prev[cur] is None:
                return None
            _, edge = prev[cur]
            edges_rev.append(edge)
            cur = edge.src_ir
        edges_rev.reverse()
        return edges_rev
