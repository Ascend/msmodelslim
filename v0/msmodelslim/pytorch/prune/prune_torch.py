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


from collections import Counter
from typing import Union, List, Tuple, Dict, Set

import torch
from torch import Tensor

from ascend_utils.common.utils import CallParams
from ascend_utils.common.security.type import check_dict_character
from ascend_utils.pytorch.dag.dag_torch_hook import DagTorchHook

from msmodelslim.pytorch.prune.prune_policy import PrunePolicyGraphConv2D, \
    ImportanceInfo, PrunePolicyGraphLinear, PrunePolicy
from msmodelslim import logger


def chn_weight(x):
    return torch.abs(x).mean().item()


class PruneTorch:
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

        self._importance_evaluation_function = chn_weight
        self._node_reserved_ratio = 0.5
        self._align = 16

    @property
    def dag(self):
        return self._dag_network

    @property
    def network(self):
        return self._network

    @staticmethod
    def _check_desc_input_output(node_x_put):
        if len(node_x_put) != 2 and not isinstance(node_x_put[0], int) and not isinstance(node_x_put[1], str):
            raise ValueError("node channels prune setting in desc is wrong.")

    def set_importance_evaluation_function(self, importance_evaluation_function):
        """
        set importance evaluation function

        Args:
            importance_evaluation_function: importance evaluation function

        Returns: self

        Examples:
        >>> eval_func_l2 = lambda chn_weight: torch.norm(chn_weight).item() / chn_weight.nelement()
        >>> PruneTorch(dag).set_importance_evaluation_function(eval_func_l2).prune()

        """

        if not callable(importance_evaluation_function):
            raise ValueError("importance evaluation function must callable")
        self._importance_evaluation_function = importance_evaluation_function
        return self

    def set_node_reserved_ratio(self, node_reserved_ratio: float):
        """
        set node not prune ratio, minimum reserved parameter ratio

        Args:
            node_reserved_ratio: ratio

        Returns: self

        Examples:
        >>> PruneTorch(dag).set_node_reserved_ratio(0.5).prune()
        """
        if not isinstance(node_reserved_ratio, float):
            raise TypeError("node max prune ratio must be float")
        if node_reserved_ratio >= 1 or node_reserved_ratio <= 0:
            raise ValueError("node max prune ratio must in (0, 1)")
        self._node_reserved_ratio = node_reserved_ratio
        return self

    def prune(self, reserved_ratio: float = 0.75, un_prune_list: List[Union[str, int]] = None):
        """
        prune network

        Args:
            reserved_ratio: reserved params ratio
            un_prune_list: unprune list, accept int(locate) and str(name), default [0, -1]

        Returns:
            desc

        Examples:
        >>> desc = PruneTorch(dag).prune()
        >>> # then training
        """
        _, desc = self.analysis(reserved_ratio, un_prune_list)
        self.prune_by_desc(desc)
        return desc

    def analysis(self, reserved_ratio: float = 0.75, un_prune_list: List[Union[str, int]] = None):
        """
        analysis network and return desc of this network

        Args:
            reserved_ratio: left params ratio
            un_prune_list: unprune list, accept int(locate) and str(name), default [0, -1]

        Returns:
            after_pruned_params, desc

        Examples:
        >>> left_params, desc = PruneTorch(dag).analysis()
        >>> print(desc)
        >>> print(left_params)
        """
        if not isinstance(reserved_ratio, float):
            raise TypeError("prune ratio must be float")

        if reserved_ratio < 0.4 or reserved_ratio >= 1:
            logger.warning("The prune ratio is abnormal. Please check.")

        un_prune_name_set = self._preprocess_un_prune_list(un_prune_list)

        logger.debug(f"un_prune_name_set: {un_prune_name_set}")
        all_importance_for_sort: List[ImportanceInfo] = []
        self._assessment_importance_conv(all_importance_for_sort, un_prune_name_set)
        self._assessment_importance_linear(all_importance_for_sort, un_prune_name_set)

        all_importance_for_sort.sort(key=lambda x: x.importance)

        all_params = self.dag.get_params()
        reserved_params = all_params * reserved_ratio
        left_params = all_params
        will_prune_outputs: Dict[str, List[ImportanceInfo]] = {}
        node_left_output_dims: Dict[str, List[int, int]] = {}
        desc = {}

        for importance_info in all_importance_for_sort:
            policy: PrunePolicy = importance_info.policy
            node_name = policy.name

            if policy.out_weight_dims == 0:
                raise ValueError("weight dims is zero.")

            if node_name not in will_prune_outputs:
                will_prune_outputs[node_name] = []
            if node_name not in node_left_output_dims:
                node_left_output_dims[node_name] = [policy.out_weight_dims, policy.out_weight_dims]

            # calc will prune output count this time
            will_prune_outputs[node_name].append(importance_info)
            node_will_prune_output_count = sum((len(x.out_weight_idxes) for x in will_prune_outputs[node_name]))

            # align
            if (policy.out_weight_dims - node_will_prune_output_count) % self._align != 0:
                continue

            # node_reserved_ratio
            pruned_left_cnt = node_left_output_dims[node_name][0] - node_will_prune_output_count
            if (policy.out_weight_dims - pruned_left_cnt) / policy.out_weight_dims > self._node_reserved_ratio:
                continue

            for prune_importance_info in will_prune_outputs[node_name]:
                left_params -= prune_importance_info.params
                prune_importance_info.policy.write_desc(desc, prune_importance_info)

            node_left_output_dims[node_name][0] -= node_will_prune_output_count
            will_prune_outputs[node_name].clear()
            if left_params <= reserved_params:
                break

        for one_desc in desc.values():
            for x_put in one_desc.values():
                x_put[1] = "".join(x_put[1])

        for name, chn_info in node_left_output_dims.items():
            left_chn, ori_chn = chn_info
            if left_chn != ori_chn:
                logger.info(f"node name: {name} chn: {ori_chn} -> {left_chn}; ")
            else:
                logger.debug(f"node name: {name} chn: {ori_chn} -> {left_chn}; ")
        logger.debug(f"desc: {desc}")
        return left_params, desc

    def prune_by_desc(self, desc: Dict[str, Dict[str, Tuple[int, str]]]):
        """
        prune network by desc, use in inference

        Args:
            desc: prune info

        Returns: None

        Examples:
        >>> desc = PruneTorch(dag).prune()
        >>> # training
        >>> # then inference
        >>> PruneTorch(dag).prune_by_desc(desc)
        """
        if not isinstance(desc, dict):
            raise TypeError("parameter desc must be dict")
        check_dict_character(desc, param_name="desc")

        ori_params = self.dag.get_params()
        if ori_params == 0:
            raise ValueError("network has no params")
        try:
            with self._dag_network:
                for node_name, node_desc in desc.items():
                    dag_node = self._dag_network.get_node_by_name(node_name)
                    self._prune_one_node(dag_node, node_desc)
        except ValueError as ex:
            logger.error(ex)
            raise ValueError("Please Check whether desc belongs to the network.") from ex

        pruned_params = self.dag.get_params()
        prune_off_ratio = (ori_params - pruned_params) / ori_params
        logger.info(f"Number of original parameters: {ori_params}; ")
        logger.info(f"Number of pruned parameters: {pruned_params}; ")
        logger.info(f"Pruning off the ratio {prune_off_ratio * 100:.3f} %")
        
    def _preprocess_un_prune_list(self, un_prune_list: List[Union[str, int]] = None) -> Set[str]:
        if un_prune_list is None:
            un_prune_name_list = [0, -1]
        else:
            if not isinstance(un_prune_list, list):
                raise TypeError("un_prune_list must be a list.")
            un_prune_name_list = un_prune_list

        prune_list = list(self.dag.search_nodes_by_op_type(["Conv2d", "Linear"]))
        for index, un_prune in enumerate(un_prune_name_list):
            if isinstance(un_prune, int):
                if un_prune >= len(prune_list) or un_prune < - len(prune_list):
                    logger.warning(f"error index({un_prune}), length of prune list is {len(prune_list)}")
                else:
                    un_prune_name_list[index] = prune_list[un_prune].name
            elif not isinstance(un_prune, str):
                raise ValueError("Element of un_prune_list must be str or int.")

        reuse_names = filter(lambda x: x[1] > 1, Counter((node.name for node in self.dag.dag_node_list)).items())
        return set(un_prune_name_list).union(map(lambda x: x[0], reuse_names))

    def _assessment_importance_conv(self, all_importance_for_sort: List, un_prune_name_set: Set[str]):
        node_analysis = set()
        conv_search_sub_graph = PrunePolicyGraphConv2D.get_search_graph()
        for conv_sub_graph in self.dag.search_sub_graph(conv_search_sub_graph):
            policy = PrunePolicyGraphConv2D(conv_sub_graph, self._importance_evaluation_function)
            conv_out = policy.node_out
            conv_in = policy.node_in
            if conv_out in node_analysis or conv_out.name in un_prune_name_set or conv_in.name in un_prune_name_set:
                continue

            node_analysis.add(conv_out)
            all_importance_for_sort.extend(policy.importance_infos)

    def _assessment_importance_linear(self, all_importance_for_sort: List, un_prune_name_set: Set[str]):
        node_analysis = set()
        linear_search_sub_graph = PrunePolicyGraphLinear.get_search_graph()
        for linear_sub_graph in self.dag.search_sub_graph(linear_search_sub_graph):
            policy = PrunePolicyGraphLinear(linear_sub_graph, self._importance_evaluation_function)
            if policy.node_out in node_analysis:
                continue
            if policy.node_out.name in un_prune_name_set or policy.node_in.name in un_prune_name_set:
                continue
            node_analysis.add(policy.node_out)
            all_importance_for_sort.extend(policy.importance_infos)

    def _prune_one_node(self, dag_node, node_desc):
        if dag_node is None:
            raise ValueError("node_desc key must exists")
        if not isinstance(node_desc, dict):
            raise TypeError("node_desc must be dict")
        if "input" not in node_desc and "output" not in node_desc:
            raise ValueError("node_desc must be list and have input or output")
        node_input, node_output = node_desc.get("input", (-1, [])), node_desc.get("output", (-1, []))

        if isinstance(dag_node.node, torch.nn.Conv2d):
            self._prune_conv2d(dag_node.node, node_input, node_output)
        elif isinstance(dag_node.node, torch.nn.Linear):
            self._prune_linear(dag_node.node, node_input, node_output)
        else:
            self._prune_batchnorm(dag_node.node, node_input)

    def _prune_conv2d(self, conv: torch.nn.Conv2d, node_input, node_output):
        self._check_desc_input_output(node_input)
        self._check_desc_input_output(node_output)

        if conv.groups <= 0:
            raise ValueError("unexpected error, this conv node groups is less than or equal to 0")

        in_weight_dims, in_channels_accept = node_input if node_input[0] > 0 else \
            (conv.in_channels / conv.groups, '-' * (conv.in_channels // conv.groups))
        out_weight_dims, out_channels_accept = node_output if node_output[0] > 0 else \
            (conv.out_channels, '-' * conv.out_channels)

        in_channels = in_weight_dims * conv.groups
        out_channels = out_weight_dims

        if in_channels > conv.in_channels or len(in_channels_accept) * conv.groups != conv.in_channels:
            raise ValueError("node in channels prune setting is wrong.")
        if out_channels > conv.out_channels or len(out_channels_accept) != conv.out_channels:
            raise ValueError("node out channels prune setting is wrong.")

        in_channels_idx = [idx for idx, value in enumerate(in_channels_accept) if value == '-']
        out_channels_idx = [idx for idx, value in enumerate(out_channels_accept) if value == '-']

        if in_weight_dims != len(in_channels_idx):
            raise ValueError("node in channels prune setting is wrong.")
        if out_weight_dims != len(out_channels_idx):
            raise ValueError("node out channels prune setting is wrong.")

        if hasattr(conv, "weight") and conv.weight is not None:
            conv.weight.data = conv.weight.data[out_channels_idx, :, :, :][:, in_channels_idx, :, :]
        if hasattr(conv, "bias") and conv.bias is not None:
            conv.bias.data = conv.bias.data[out_channels_idx]

        conv.in_channels = in_channels
        conv.out_channels = out_channels

    def _prune_batchnorm(self, batchnorm, node_feature_num):
        if isinstance(batchnorm, torch.nn.LayerNorm):
            return

        if hasattr(batchnorm, "num_channels"):
            bn_num_features = batchnorm.num_channels
        else:
            bn_num_features = batchnorm.num_features

        self._check_desc_input_output(node_feature_num)

        num_features, num_features_accept = node_feature_num if node_feature_num[0] > 0 else \
            (bn_num_features, '-' * bn_num_features)

        if num_features > bn_num_features or len(num_features_accept) != bn_num_features:
            raise ValueError("node in channels prune setting is wrong.")

        num_features_idx = [idx for idx, value in enumerate(num_features_accept) if value == '-']

        if num_features != len(num_features_idx):
            raise ValueError("node in channels prune setting is wrong.")

        if hasattr(batchnorm, "running_mean") and batchnorm.running_mean is not None:
            batchnorm.running_mean.data = batchnorm.running_mean.data[num_features_idx]
        if hasattr(batchnorm, "running_var") and batchnorm.running_var is not None:
            batchnorm.running_var.data = batchnorm.running_var.data[num_features_idx]
        if hasattr(batchnorm, "weight") and batchnorm.weight is not None:
            batchnorm.weight.data = batchnorm.weight.data[num_features_idx]
        if hasattr(batchnorm, "bias") and batchnorm.bias is not None:
            batchnorm.bias.data = batchnorm.bias.data[num_features_idx]

        if hasattr(batchnorm, "num_channels"):
            batchnorm.num_channels = num_features
        else:
            batchnorm.num_features = num_features

    def _prune_linear(self, linear, node_input, node_output):
        self._check_desc_input_output(node_input)
        self._check_desc_input_output(node_output)

        in_features, in_features_accept = node_input if node_input[0] > 0 else \
            (linear.in_features, '-' * linear.in_features)
        out_features, out_features_accept = node_output if node_output[0] > 0 else \
            (linear.out_features, '-' * linear.out_features)

        if in_features > linear.in_features or len(in_features_accept) != linear.in_features:
            raise ValueError("node in channels prune setting is wrong.")
        if out_features > linear.out_features or len(out_features_accept) != linear.out_features:
            raise ValueError("node out channels prune setting is wrong.")

        in_features_idx = [idx for idx, value in enumerate(in_features_accept) if value == '-']
        out_features_idx = [idx for idx, value in enumerate(out_features_accept) if value == '-']

        if in_features != len(in_features_idx):
            raise ValueError(
                "node in channels prune setting is wrong." + str(in_features) + " " + str(len(in_features_idx)))
        if out_features != len(out_features_idx):
            raise ValueError("node out channels prune setting is wrong.")

        if hasattr(linear, "weight") and linear.weight is not None:
            linear.weight.data = linear.weight.data[out_features_idx, :][:, in_features_idx]
        if hasattr(linear, "bias") and linear.bias is not None:
            linear.bias.data = linear.bias.data[out_features_idx]

        linear.in_features = in_features
        linear.out_features = out_features
