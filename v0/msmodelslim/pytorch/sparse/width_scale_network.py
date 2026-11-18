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


from typing import Union, List, Tuple, Set

import torch
from torch import Tensor
from torch.functional import F

from ascend_utils.common.utils import CallParams
from ascend_utils.core.dag.dag_node import DagNode
from ascend_utils.pytorch.dag.dag_torch_hook import DagTorchHook
from msmodelslim import logger


class WidthScaleNetwork:
    """
    Args:
        network: initialized pytorch model. Will perform in-place pruning / enlarging
        inputs: model actual inputs. Better with batch_size=1

    Returns:
        desc
    Examples:
    >>> WidthScaleNetwork(network).scale()
    >>> # then training
    """
    def __init__(self, network: Union[torch.nn.Module, DagTorchHook],
                 inputs: Union[Tensor, List[Tensor], Tuple[Tensor], CallParams] = None):
        if isinstance(network, torch.nn.Module):
            self._network = network
            self._dag_network = DagTorchHook(network, inputs)
        elif isinstance(network, DagTorchHook):
            self._network = network.network
            self._dag_network = network
        else:
            raise ValueError("network must be torch.nn.Module or Dag")

    @property
    def dag(self):
        return self._dag_network

    @property
    def network(self):
        return self._network

    @staticmethod
    def get_node_in_out_channel_count(dag_node) -> Tuple[int, int, int]:
        in_chn, out_chn, groups = 0, 0, 1
        if not hasattr(dag_node.node, "weight"):
            return in_chn, out_chn, 1

        if hasattr(dag_node.node, "num_features"):
            in_chn, out_chn = dag_node.node.num_features, dag_node.node.num_features
        if hasattr(dag_node.node, "num_channels"):
            in_chn, out_chn = dag_node.node.num_channels, dag_node.node.num_channels
        if hasattr(dag_node.node, "normalized_shape"):
            in_chn, out_chn = dag_node.node.normalized_shape[0], dag_node.node.normalized_shape[0]

        if hasattr(dag_node.node, "in_features"):
            in_chn = dag_node.node.in_features
        if hasattr(dag_node.node, "out_features"):
            out_chn = dag_node.node.out_features

        if hasattr(dag_node.node, "in_channels"):
            in_chn = dag_node.node.in_channels
        if hasattr(dag_node.node, "out_channels"):
            out_chn = dag_node.node.out_channels
        if hasattr(dag_node.node, "groups"):
            groups = dag_node.node.groups if dag_node.node.groups > 0 else 1

        return in_chn, out_chn, groups

    @staticmethod
    def rescale_weight(weight, target_shape):
        if weight is None or len(weight.shape) == 0:
            return weight

        num_dims = len(weight.shape)
        while weight.shape[0] < target_shape[0] or (num_dims > 1 and weight.shape[1] < target_shape[1]):
            source_shape = weight.shape
            if num_dims == 1:
                if source_shape[0] < target_shape[0]:
                    weight = torch.stack([weight, weight], dim=0).contiguous().view([-1])
            elif source_shape[0] < target_shape[0] and source_shape[1] < target_shape[1]:
                left_weight = F.pad(weight, [0, 0] * (num_dims - 2) + [0, 0, 0, source_shape[0]])
                right_weight = F.pad(weight, [0, 0] * (num_dims - 2) + [0, 0, source_shape[0], 0])
                weight = torch.stack([left_weight, right_weight], dim=1)
                weight = weight.contiguous().view([left_weight.shape[0], -1, *source_shape[2:]])
            elif source_shape[0] < target_shape[0]:
                weight = torch.stack([weight, weight], dim=0)
                weight = weight.contiguous().view([-1, *source_shape[1:]])
            elif source_shape[1] < target_shape[1]:
                weight = torch.stack([weight, weight], dim=1)
                weight = weight.contiguous().view([source_shape[0], -1, *source_shape[2:]])
        new_weight = weight[:target_shape[0]] if num_dims == 1 else weight[:target_shape[0], :target_shape[1]]
        return new_weight.clone()

    @staticmethod
    def _is_all_node_not_in_graph(node_name, in_output_nodes, all_nodes_set):
        in_graph_count = 0
        all_count = 0
        for in_out_node in in_output_nodes:
            all_count += 1
            if in_out_node in all_nodes_set:
                in_graph_count += 1
        if in_graph_count == 0:
            return True
        elif all_count != in_graph_count:
            raise ValueError(f"Some node {node_name} upper or lower nodes are in the"
                             f" graph to be prune, and some are not.")
        else:
            return False
            
    @staticmethod
    def _dfs_search_till_node_with_weight(start_node_set, mode="out") -> Set[DagNode]:
        list_nodes = list(start_node_set)
        seen_node_set = set()
        while list_nodes:
            cur_node = list_nodes.pop()
            if hasattr(cur_node.node, "weight"):
                continue

            next_nodes = cur_node.output_nodes if mode == "out" else cur_node.input_nodes
            for node in next_nodes:
                if node in start_node_set or node in seen_node_set:
                    continue
                list_nodes.append(node)
                seen_node_set.add(node)
        return seen_node_set
    
    @classmethod
    def prune_node(cls, dag_node, in_chn, out_chn, groups):
        if hasattr(dag_node.node, "num_features"):
            dag_node.node.num_features = in_chn
        if hasattr(dag_node.node, "num_channels"):
            dag_node.node.num_channels = in_chn

        if hasattr(dag_node.node, "normalized_shape"):
            dag_node.node.normalized_shape = (in_chn,) + dag_node.node.normalized_shape[1:]

        if hasattr(dag_node.node, "in_features"):
            dag_node.node.in_features = in_chn
        if hasattr(dag_node.node, "out_features"):
            dag_node.node.out_features = out_chn

        if hasattr(dag_node.node, "in_channels"):
            dag_node.node.in_channels = in_chn
        if hasattr(dag_node.node, "out_channels"):
            dag_node.node.out_channels = out_chn
        if hasattr(dag_node.node, "groups"):
            dag_node.node.groups = groups if groups > 0 else 1

        if hasattr(dag_node.node, "running_mean") and dag_node.node.running_mean is not None:
            dag_node.node.running_mean.data = cls.rescale_weight(dag_node.node.running_mean.data, [out_chn])
        if hasattr(dag_node.node, "running_var") and dag_node.node.running_var is not None:
            dag_node.node.running_var.data = cls.rescale_weight(dag_node.node.running_var.data, [out_chn])
        if hasattr(dag_node.node, "weight") and dag_node.node.weight is not None:
            target_shape = [out_chn, int(in_chn / groups)] + list(dag_node.node.weight.shape[2:])
            dag_node.node.weight = torch.nn.Parameter(cls.rescale_weight(dag_node.node.weight.data, target_shape))
        if hasattr(dag_node.node, "bias") and dag_node.node.bias is not None:
            dag_node.node.bias = torch.nn.Parameter(cls.rescale_weight(dag_node.node.bias.data, [out_chn]))

    def get_nodes_input_output(self, list_nodes) -> Tuple[Set[DagNode], Set[DagNode]]:
        set_input_nodes = set()
        set_output_nodes = set()
        set_nodes = set(list_nodes)

        for node in list_nodes:
            if self._is_all_node_not_in_graph(node.name, node.input_nodes, set_nodes):
                set_input_nodes.add(node)
            if self._is_all_node_not_in_graph(node.name, node.output_nodes, set_nodes):
                set_output_nodes.add(node)

        set_input_nodes.update(self._dfs_search_till_node_with_weight(set_input_nodes, mode="out"))
        set_output_nodes.update(self._dfs_search_till_node_with_weight(set_output_nodes, mode="in"))
        return set_input_nodes, set_output_nodes

    def scale(self, scale_multiple: int = 2, node_prefix: str = ""):
        """
        prune / enlarge network on channel dimension.

        Args:
            scale_multiple: non-zero value for model channel scale ratio. > 0 for enlarging, < 0 for pruning
            node_prefix: str value for filtering node be prefix. Default "" for not filtering

        Returns:
            desc
        """
        if scale_multiple <= 0:
            raise ValueError("scale_multiple should be value > 0")

        prune_multiple = 1 / scale_multiple
        list_nodes = list(self.dag.get_nodes_by_name_prefix(node_prefix))
        if len(list_nodes) == 0:
            logger.warning("No node is pruned.")
            return

        # get input output nodes
        input_nodes, output_nodes = self.get_nodes_input_output(list_nodes)

        # check ratio
        for dag_node in list_nodes:
            in_chn, out_chn, _ = self.get_node_in_out_channel_count(dag_node)
            if dag_node in input_nodes:
                in_chn = 0
            if dag_node in output_nodes:
                out_chn = 0
            if in_chn % prune_multiple != 0 or out_chn % prune_multiple != 0:
                raise ValueError(f"{dag_node.name} channels number({in_chn},{out_chn}) "
                                 f"cannot be exactly divisible by {prune_multiple}.")

        # prune
        for dag_node in list_nodes:
            in_chn, out_chn, groups = self.get_node_in_out_channel_count(dag_node)
            if in_chn == 0 or out_chn == 0:
                continue
            if dag_node not in input_nodes:
                in_chn = int(in_chn / prune_multiple)
            if dag_node not in output_nodes:
                out_chn = int(out_chn / prune_multiple)
            if in_chn < groups:
                groups = in_chn

            self.prune_node(dag_node, in_chn, out_chn, groups)
