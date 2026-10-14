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
# pylint: disable=consider-merging-isinstance

from typing import List, Any


class DagNodeIO:
    """
    DagNode input output
    It is usually used to record where the Tensor comes from and where the Tensor goes to (who uses the Tensor).
    """

    def __init__(self, io_index: Any, name: str, io_info: Any = None, node_from=None):
        if io_index is None:
            raise ValueError("io_index must be not None")
        if not isinstance(name, str):
            raise TypeError("name must be type str")

        self._io_index: str = str(io_index)
        self._name: str = name
        self._dag_node_from = node_from  # where the Tensor comes from
        self._dag_nodes_to: List = []  # where the Tensor goes to
        self._io_info = io_info

    @property
    def io_index(self) -> str:
        return self._io_index

    @property
    def name(self) -> str:
        return self._name

    @property
    def dag_node_from(self):
        return self._dag_node_from

    @property
    def dag_nodes_to(self) -> List:
        return self._dag_nodes_to

    @property
    def info(self):
        return self._io_info

    def add_node_to(self, node):
        if node is None:
            return
        if isinstance(node, list) or isinstance(node, tuple):
            self._dag_nodes_to.extend(node)
        else:
            self._dag_nodes_to.append(node)

    def set_node_from(self, node):
        if self._dag_node_from is not None:
            raise ValueError("Repeated settings are not allowed.")
        self._dag_node_from = node
