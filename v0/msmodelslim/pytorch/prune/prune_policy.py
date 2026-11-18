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


import inspect
import itertools
from abc import abstractmethod, ABC
from typing import List, Dict, Optional

import torch.nn
from torch.nn import Module
from torch.nn.modules import activation
from torch.nn.modules import batchnorm
from torch.nn.modules import dropout
from torch.nn.modules import instancenorm
from torch.nn.modules import normalization
from torch.nn.modules import padding
from torch.nn.modules import pooling
from torch.nn.modules import upsampling

from ascend_utils.core.dag.dag_node import DagNode


class PrunePolicyGraph:
    @staticmethod
    def get_module_attr_name(py_module):
        for attr_name in dir(py_module):
            attr = getattr(py_module, attr_name)
            if inspect.isclass(attr) and issubclass(attr, Module):
                yield attr_name

    @staticmethod
    def get_chn_eq_node(name: str):
        chn_eq_op_types = {None}
        for act in PrunePolicyGraph.get_module_attr_name(activation):
            chn_eq_op_types.add(act)
        for pool in PrunePolicyGraph.get_module_attr_name(pooling):
            chn_eq_op_types.add(pool)
        for batch_name in PrunePolicyGraph.get_module_attr_name(batchnorm):
            chn_eq_op_types.add(batch_name)
        for drop_name in PrunePolicyGraph.get_module_attr_name(dropout):
            chn_eq_op_types.add(drop_name)
        for padding_name in PrunePolicyGraph.get_module_attr_name(padding):
            chn_eq_op_types.add(padding_name)
        for normal_name in PrunePolicyGraph.get_module_attr_name(normalization):
            chn_eq_op_types.add(normal_name)
        for ins_name in PrunePolicyGraph.get_module_attr_name(instancenorm):
            chn_eq_op_types.add(ins_name)
        for up_name in PrunePolicyGraph.get_module_attr_name(upsampling):
            chn_eq_op_types.add(up_name)

        return DagNode(op_type=chn_eq_op_types, name=name)

    @staticmethod
    def get_conv2d_to_conv2d_graph(first_conv_name, second_conv_name):
        conv1 = DagNode(op_type="Conv2d", name=first_conv_name)
        chn_eq1 = PrunePolicyGraph.get_chn_eq_node("chn_eq1")
        chn_eq2 = PrunePolicyGraph.get_chn_eq_node("chn_eq2")
        conv2 = DagNode(op_type="Conv2d", name=second_conv_name)

        conv1 >> chn_eq1 >> chn_eq2 >> conv2
        return [conv1, chn_eq1, chn_eq2, conv2]

    @staticmethod
    def get_linear_to_linear_graph(first_conv_name, second_conv_name):
        linear1 = DagNode(op_type="Linear", name=first_conv_name)
        chn_eq1 = PrunePolicyGraph.get_chn_eq_node("chn_eq1")
        chn_eq2 = PrunePolicyGraph.get_chn_eq_node("chn_eq2")
        linear2 = DagNode(op_type="Linear", name=second_conv_name)

        linear1 >> chn_eq1 >> chn_eq2 >> linear2
        return [linear1, chn_eq1, chn_eq2, linear2]


class ImportanceInfo(dict):
    def __init__(self, importance: float, params: int, out_weight_idxes: List, policy, **kwargs):
        if any((ele is None for ele in (importance, params, out_weight_idxes, policy))):
            raise ValueError("all input param of ImportanceInfo must not None")
        super().__init__(importance=importance, params=params, out_weight_idxes=out_weight_idxes, policy=policy,
                         **kwargs)

    @property
    def importance(self) -> float:
        return self.get("importance")

    @property
    def params(self) -> int:
        return self.get("params")

    @property
    def out_weight_idxes(self) -> List:
        return self.get("out_weight_idxes")

    @property
    def policy(self):
        return self.get("policy")


class PrunePolicy:
    def __init__(self, node_out, node_in, chn_eq1, chn_eq2):
        self._node_out = node_out
        self._node_in = node_in
        self._chn_eq1 = chn_eq1
        self._chn_eq2 = chn_eq2
        self._importance_infos: Optional[List[ImportanceInfo]] = None

    @property
    def name(self) -> str:
        return self._node_out.name

    @property
    def node_out(self) -> DagNode:
        # Node of the prune weight output channel
        return self._node_out

    @property
    def node_in(self) -> DagNode:
        # Node of the prune weight input channel
        return self._node_in

    @property
    @abstractmethod
    def out_chn(self) -> int:
        """ return out channel count """

    @property
    @abstractmethod
    def out_weight_dims(self) -> int:
        """ return out node weight dim count """
        return self.out_chn

    @property
    @abstractmethod
    def in_weight_dims(self) -> int:
        """ return in node weight dim count """
        return self.out_chn

    @property
    def importance_infos(self) -> List[ImportanceInfo]:
        if self._importance_infos is None:
            self._calc_importance_infos()
        return self._importance_infos

    @staticmethod
    def create_item_in_desc(desc, name, put_type, indexes, count):
        if name not in desc:
            desc[name] = {}
        if put_type not in desc[name]:
            desc[name][put_type] = [count, ['-'] * count]
        desc[name][put_type][0] -= len(indexes)
        for index in indexes:
            desc[name][put_type][1][index] = "x"

    def write_desc(self, desc, importance_info):
        out_weight_idxes = importance_info.get("out_weight_idxes")
        in_weight_idxes = importance_info.get("in_weight_idxes")
        out_chn_idxes = importance_info.get("out_chn_idxes")
        self.create_item_in_desc(desc, self._node_out.name, "output", out_weight_idxes, self.out_weight_dims)
        self.create_item_in_desc(desc, self._node_in.name, "input", in_weight_idxes, self.in_weight_dims)
        if self._chn_eq1 is not None and hasattr(self._chn_eq1.node, "weight"):
            self.create_item_in_desc(desc, self._chn_eq1.name, "input", out_chn_idxes, self.out_chn)
        if self._chn_eq2 is not None and hasattr(self._chn_eq2.node, "weight"):
            self.create_item_in_desc(desc, self._chn_eq2.name, "input", out_chn_idxes, self.out_chn)

    @abstractmethod
    def _calc_importance_infos(self) -> List[ImportanceInfo]:
        pass


class PrunePolicyGraphConv2D(PrunePolicy):
    def __init__(self, graph: Dict[str, DagNode], importance_eval):
        if graph.get("conv_out", None) is None or graph.get("conv_in", None) is None:
            raise ValueError("inner Error, search nothing")
        super().__init__(graph.get("conv_out"), graph.get("conv_in"),
                         graph.get("chn_eq1", None), graph.get("chn_eq2", None))
        self._conv_out_node: torch.nn.Conv2d = self._node_out.node
        self._conv_in_node: torch.nn.Conv2d = self._node_in.node
        self.importance_eval = importance_eval

    @property
    def out_chn(self) -> int:
        return self._conv_out_node.out_channels

    @property
    def in_weight_dims(self) -> int:
        """ return out channel count """
        if self._conv_in_node.groups == 0:
            raise ValueError("Check whether the node is a normal Conv2d operator.")
        return int(self.out_chn / self._conv_in_node.groups)

    @staticmethod
    def get_search_graph():
        return PrunePolicyGraph.get_conv2d_to_conv2d_graph("conv_out", "conv_in")

    def _calc_importance_infos(self):
        self._importance_infos = []
        in_groups = self._conv_in_node.groups
        out_groups = self._conv_out_node.groups

        if in_groups == 0 or out_groups == 0:
            raise ValueError("Check whether the node is a normal Conv2d operator.")

        out_group_dim_count = int(self.out_chn / out_groups)
        in_group_dim_count = int(self.out_chn / in_groups)
        if out_groups % in_groups == 0:
            group_mapping = int(out_groups / in_groups)
            is_out_groups_bigger = True
        elif in_groups % out_groups == 0:
            group_mapping = int(in_groups / out_groups)
            is_out_groups_bigger = False
        else:
            return

        for index in range(min(in_group_dim_count, out_group_dim_count)):
            if is_out_groups_bigger:
                out_weight_idxes = [index + out_group_dim_count * i for i in range(out_groups)]
                in_weight_idxes = [out_group_dim_count * i + index for i in range(group_mapping)]
                out_chn_idxes = [index + out_group_dim_count * i for i in range(out_groups)]
            else:
                in_weight_idxes = [index]
                out_weight_idxes = [in_group_dim_count * i + index for i in range(group_mapping)]
                out_chn_idxes = ((x + out_group_dim_count * i for x in out_weight_idxes) for i in range(out_groups))
                out_chn_idxes = list(itertools.chain.from_iterable(out_chn_idxes))

            out_weight: torch.Tensor = self._conv_out_node.weight.data[out_weight_idxes]
            in_weight: torch.Tensor = self._conv_in_node.weight.data[:, in_weight_idxes]
            self._importance_infos.append(ImportanceInfo(
                importance=self.importance_eval(out_weight),
                params=out_weight.nelement() + in_weight.nelement(),
                out_weight_idxes=out_weight_idxes,
                policy=self,
                out_chn_idxes=out_chn_idxes,
                in_weight_idxes=in_weight_idxes,
            ))


class PrunePolicyGraphLinear(PrunePolicy, ABC):
    def __init__(self, graph: Dict[str, DagNode], importance_eval):
        if graph.get("linear_out", None) is None or graph.get("linear_in", None) is None:
            raise ValueError("inner Error, search nothing")
        super().__init__(graph.get("linear_out"), graph.get("linear_in"),
                         graph.get("chn_eq1", None), graph.get("chn_eq2", None))
        self.linear_out_node: torch.nn.Linear = self._node_out.node
        self.linear_in_node: torch.nn.Linear = self._node_in.node
        self.importance_eval = importance_eval

    @property
    def out_chn(self) -> int:
        return self.linear_out_node.out_features

    @staticmethod
    def get_search_graph():
        return PrunePolicyGraph.get_linear_to_linear_graph("linear_out", "linear_in")

    def _calc_importance_infos(self):
        self._importance_infos = []
        for index in range(self.out_chn):
            out_weight: torch.Tensor = self.linear_out_node.weight.data[index]
            in_weight: torch.Tensor = self.linear_in_node.weight.data[:, index]
            self._importance_infos.append(ImportanceInfo(
                importance=self.importance_eval(out_weight),
                params=out_weight.nelement() + in_weight.nelement(),
                out_weight_idxes=[index],
                policy=self,
                out_chn_idxes=[index],
                in_weight_idxes=[index],
            ))
