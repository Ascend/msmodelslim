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


"""
Adapter for DAG, 
including DAG info, conv-bn pattern etc. 
"""
from typing import List, Any, Union, Tuple, OrderedDict
from collections import defaultdict, deque

import torch.nn as nn
import torch
import transformers

from ascend_utils.common.utils import CallParams
from ascend_utils.core.dag.dag_node import DagNode

from .dag_torch_hook import DagTorchHook
from .model_infos import ModuleType
from .model_structure_process import StructureProcess


class DagNodeInfo:
    def __init__(self,
                 name: str,
                 class_type: str,
                 input_nodes: List[str],
                 output_nodes: List[str]):
        self.name = name
        self.class_type = class_type
        self.input_nodes = input_nodes
        self.output_nodes = output_nodes

    def __eq__(self, other):
        return other and self.name == other.name and \
               self.class_type == other.class_type and \
               self.input_nodes == other.input_nodes and \
               self.output_nodes == other.output_nodes


class TorchDAGAdapter(object):
    """ dag adapter for AI model pattern recognition """

    def __init__(self,
                 model: nn.Module,
                 dummy_input: Union[
                     torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor],
                     CallParams, None
                 ] = None,
                 hook_nodes: list = None,
                 anti_method=None):

        self._model = model
        self._model.eval()
        self._conv_bn_pattern = {}

        if dummy_input is None:
            self._dummy_input = torch.ones([1, 3, 224, 224]).type(torch.float32)
        else:
            self._dummy_input = dummy_input

        if isinstance(self._dummy_input, torch.Tensor):
            self._dummy_input = self._dummy_input.to(next(model.parameters()).device)

        if not hook_nodes:
            hook_nodes = []
        self.norm_nodes = [m.__name__.lower() for m in hook_nodes if "norm" in m.__name__.lower()]

        self._dag_hook = DagTorchHook(self._model, self._dummy_input, hook_nodes, anti_method=anti_method)

        self.node_list = self._dag_hook.dag_node_list

        self.norm_node_list = [m for m in self.node_list if m.op_type.lower() in self.norm_nodes][:-1]
        
        self.v_up_name_list = []

        self.dark_name_list = (
            '_TensorBase.__getitem__', '_TensorBase.__setitem__', '_TensorBase.size',
            'Tensor.__hash__', '_TensorBase.is_floating_point', 'view', 'unsqueeze',
            'expand', 'dim', '__getitem__', 'unbind', 'size'
        )

        self.add_op_type = '__add__'

    @staticmethod
    def _get_ffn_pattern_and_ln_list(matmul_tuple, ln_list, stop_node, ffn_pattern, ffn_ln_list):
        matmul_num, matmul_list = matmul_tuple
        if StructureProcess.is_ffn_matmul(matmul_list, matmul_num):
            # NOTICE: this condition only work in bert related model for dag parse bug
            if stop_node.op_type.title() != ModuleType.LINEAR:
                matmul_list.reverse()
            ffn_pattern.append([item.name_in_network for item in matmul_list])
            ffn_ln_list.append(ln_list[0].name_in_network)

    @staticmethod
    def _get_trav_node(add_node, node):
        trav_node = None
        for item in add_node[0].input_nodes:
            if item == node:
                continue
            trav_node = item
        return trav_node

    def add_mhsa_norm_linears(self, mhsa_linear, mhsa_o, mhsa_ln, interval):
        start, end = interval[0], interval[1]
        inter_linears = self.node_list[start+1:end]
        if len(inter_linears) <= 4:
            mhsa_linear.append([node.name_in_network for node in self.node_list[start+1:end-1]])
            mhsa_o.append(self.node_list[end-1].name_in_network)
            mhsa_ln.append(self.node_list[start].name_in_network)
        else:
            mhsa_linear.append([node.name_in_network for node in self.node_list[start+1:start+4]])
            mhsa_o.append(self.node_list[start+4].name_in_network)
            mhsa_ln.append(self.node_list[start].name_in_network)

    def get_llm_network_pattern_auto(self):
        mhsa_linear = []
        mhsa_o = []  # multi head self attention中的o的linear
        mhsa_ln = []  # multi head self attention中的norm
        ffn_linear = []  
        ffn_ln = []  # mlp的post layernorm
        norm_positions = [i for i, x in enumerate(self.node_list) if x.op_type.lower() in self.norm_nodes]
        num_norm = len(norm_positions)

        for i in range(num_norm - 1):
            start, end = norm_positions[i], norm_positions[i + 1]
            interval = [start, end]
            if i % 2 == 0:
                self.add_mhsa_norm_linears(mhsa_linear, mhsa_o, mhsa_ln, interval)
            else:
                ffn_linear.append([node.name_in_network for node in self.node_list[start+1:end]])
                ffn_ln.append(self.node_list[start].name_in_network)

        ret = mhsa_linear, mhsa_o, mhsa_ln, ffn_linear, ffn_ln
        return ret

    def get_node_name_in_order(self):
        return [node.name_in_network for node in self.node_list]

    def get_name_type_dict_in_order(self):
        ret = {}
        for node in self.node_list:
            ret[node.name_in_network] = node.op_type

        return ret

    def get_network_topology(self):
        dag_info_list = []
        for node in self.node_list:
            node_input_names = [item.name_in_network if item else None for item in node.input_nodes]
            node_output_names = [item.name_in_network if item else None for item in node.output_nodes]
            dag_info_list.append(DagNodeInfo(node.name_in_network,
                                             node.op_type,
                                             node_input_names,
                                             node_output_names))

        return dag_info_list

    def get_mhsa_pattern(self):
        qkv_list = []
        proj_list = []

        for node in self.node_list:
            add_node, other_branch = self._split_by_add_op_type(node)
            if len(add_node) != 1 or len(add_node[0].inputs) != 2 or not other_branch:
                continue
            trav_node = self._get_trav_node(add_node, node)

            matmul_list = self._get_all_interval_nodes_by_type(
                node, trav_node, cls_type=ModuleType.LINEAR, ignore_stop_node=True
            )
            StructureProcess.mhsa_matmul_process(matmul_list, qkv_list, proj_list)

        return qkv_list, proj_list

    def get_ffn_pattern(self):
        ffn_pattern = []
        ffn_matmul_num = 2

        for node in self.node_list:
            add_node, other_branch = self._split_by_add_op_type(node)
            if len(add_node) != 1 or len(other_branch) != 1 or len(add_node[0].inputs) != 2:
                continue
            stop_node = other_branch[0]
            trav_node = self._get_trav_node(add_node, node)
            matmul_list = self._get_all_interval_nodes_by_type(stop_node, trav_node, cls_type=ModuleType.LINEAR)

            if StructureProcess.is_ffn_matmul(matmul_list, ffn_matmul_num):
                # NOTICE: this condition only work in bert related model for dag parse bug
                if stop_node.op_type.title() != ModuleType.LINEAR:
                    matmul_list.reverse()
                ffn_pattern.append([item.name_in_network for item in matmul_list])

        return ffn_pattern

    def get_mhsa_ln_pattern(self, ln_type=ModuleType.LAYERNORM):
        qkv_list = []
        proj_list = []
        mhsa_ln_list = []
        for node in self.node_list:
            add_node, other_branch = self._split_by_add_op_type(node)
            if len(add_node) != 1 or not other_branch or len(add_node[0].inputs) != 2:
                continue
            trav_node = self._get_trav_node(add_node, node)

            matmul_list = self._get_all_interval_nodes_by_type(
                node, trav_node, cls_type=ModuleType.LINEAR, ignore_stop_node=True
            )

            ln_list = self._get_all_interval_nodes_by_type(node, trav_node, cls_type=ln_type, ignore_stop_node=True)
            StructureProcess.mhsa_matmul_ln_process(matmul_list, ln_list, qkv_list, proj_list, mhsa_ln_list)
        return qkv_list, proj_list, mhsa_ln_list

    def get_ffn_ln_pattern(self):
        matmul_num = 2
        ffn_pattern, ffn_ln_list = [], []

        for node in self.node_list:
            add_node, other_branch = self._split_by_add_op_type(node)
            if len(add_node) != 1 or len(other_branch) != 1 or len(add_node[0].inputs) != 2:
                continue
            trav_node = self._get_trav_node(add_node, node)
            ln_list = self._get_all_interval_nodes_by_type(
                node, trav_node, cls_type=ModuleType.LAYERNORM, ignore_stop_node=True
            )
            stop_node = other_branch[0]

            matmul_list = self._get_all_interval_nodes_by_type(stop_node, trav_node, cls_type=ModuleType.LINEAR)
            matmul_tuple = (matmul_num, matmul_list)

            self._get_ffn_pattern_and_ln_list(matmul_tuple, ln_list, stop_node, ffn_pattern, ffn_ln_list)
        return ffn_pattern, ffn_ln_list

    def get_llama_mhsa_ln_pattern(self):
        """get_mhsa_ln_pattern that only work for LLAMA"""
        return self.get_mhsa_ln_pattern(ln_type='Llamarmsnormbias')

    def get_llama_ffn_ln_pattern(self, cls_type='Llamarmsnormbias'):
        """get_ffn_ln_pattern that only valid for LLAMA"""
        matmul_num = 3
        ffn_pattern, ffn_ln_list = [], []

        for node in self.node_list:
            add_node, other_branch = self._split_by_add_op_type(node)
            if len(add_node) != 1 or len(other_branch) != 1 or len(add_node[0].inputs) != 2:
                continue
            trav_node = self._get_trav_node(add_node, node)

            stop_node = other_branch[0]
            matmul_list = self._get_all_interval_nodes_by_type(
                stop_node, trav_node, ModuleType.LINEAR, branch_aware=True
            )
            ln_list = self._get_all_interval_nodes_by_type(node, trav_node, cls_type=cls_type)
            matmul_tuple = (matmul_num, matmul_list)
            self._get_ffn_pattern_and_ln_list(matmul_tuple, ln_list, stop_node, ffn_pattern, ffn_ln_list)
        return ffn_pattern, ffn_ln_list

    def get_norm_linear_subgraph(self):
        norm_linear_subgraph = defaultdict(list)
        norm_positions = [i for i, x in enumerate(self.node_list) if x.op_type.lower() in self.norm_nodes]
        num_norm = len(norm_positions)
 
        for i in range(num_norm - 1):
            start = norm_positions[i]
            end = norm_positions[i + 1]
            interval_linears = [node.name_in_network for node in self.node_list[start+1:end-1]]
            norm_node = self.node_list[start].name_in_network
            
            if len(interval_linears) <= 4:
                norm_linear_subgraph[norm_node].extend(interval_linears)
            else:
                qkv_linears = [node.name_in_network for node in self.node_list[start+1:start+4]]
                norm_linear_subgraph[norm_node].extend(qkv_linears)
        return norm_linear_subgraph

    def get_kv_linears(self):
        '''
        purpose: get the linear names of key and value in the model;
        find the index of norm class in the self.node_list, choose the interval_linears between two norm classes,
        select the wanted the linear names of key and value in the interval_linears while leaving the qury linear.
        '''
        kv_linears = []
        norm_positions = [i for i, x in enumerate(self.node_list) if x.op_type.lower() in self.norm_nodes]
        num_norm = len(norm_positions)

        for i in range(num_norm - 1):
            start = norm_positions[i]
            end = norm_positions[i + 1]
            interval_linears = [node.name_in_network for node in self.node_list[start+1:end-1]]
            # 选取attention block的两个norm class偶数区间（区间0,2...）内的qkv linear，并去除q linear
            if i % 2 == 0:
                if len(interval_linears) == 1:
                    kv_linears.extend(interval_linears)
                    num_kv = 0
                else:
                    kv_linears.extend(interval_linears[1:])
                    num_kv = len(interval_linears[1:])
        return kv_linears, num_kv

    def get_linear_linear_subgraph(self, structured_linear_order=True):
 
        linear_linear_subgraph = defaultdict(list)
        norm_positions = [i for i, x in enumerate(self.node_list) if x.op_type.lower() in self.norm_nodes]
        num_norm = len(norm_positions)
        for i in range(num_norm - 1):
            # Different cases of linear blocks in LLM:
            # 4 linears - q k v o
            # 3 linears - gate up down
            # 2 linears - qkv o || gate down
            num_of_linears = norm_positions[i + 1] - norm_positions[i] - 1
            origin_node = self.node_list[norm_positions[i + 1] - 1]
            origin_node_name = origin_node.name_in_network # o_proj_name or down_proj_name
            target_node_name = None # (qk)v_proj_name or up_proj_name
            if num_of_linears > 4:
                origin_node = self.node_list[norm_positions[i] + 4]
                origin_node_name = origin_node.name_in_network
                target_node_name = self.node_list[norm_positions[i] + 3].name_in_network
            elif num_of_linears > 2: # q k v o || gate up down
                if structured_linear_order:
                    target_node_name = self.node_list[norm_positions[i + 1] - 2].name_in_network
                else:
                    target_node_name = self._find_latest_module_by_type_bfs(origin_node, "Linear")
            elif num_of_linears == 2: # qkv o || gate down
                target_node_name = self.node_list[norm_positions[i + 1] - 2].name_in_network
                target_layer = self._find_module_by_name(target_node_name)
                origin_layer = self._find_module_by_name(origin_node_name)
                if origin_layer.weight.size(1) == target_layer.weight.size(0): # gate down
                    target_node_name = None
            else:
                target_node_name = None
            if target_node_name is not None:
                linear_linear_subgraph[target_node_name].extend([origin_node_name])
        return linear_linear_subgraph

    def get_allreduce_linear(self):
        down_o_linear = list()
        norm_pos = [i for i, x in enumerate(self.node_list) if x.op_type.lower() in self.norm_nodes]
        num_norm = len(norm_pos)

        for i in range(num_norm - 1):
            start = norm_pos[i]
            end = norm_pos[i + 1]
            down_o_linear.append(self.node_list[end - 1].name_in_network)

        return down_o_linear

    def _find_module_by_name(self, name):
        if name is None:
            return name
        tokens = name.split('.')
        cur_mod = self._model
        for s in tokens:
            cur_mod = getattr(cur_mod, s, None)
        return cur_mod
       
    def _find_latest_module_by_type_bfs(self,
                                        search_node: Any,
                                        node_type: str = None,
                                        stop_node: DagNode = None,
                                        is_input: bool = True):
        if search_node is None:
            return []

        if is_input:
            nodes = search_node.input_nodes
        else:
            nodes = search_node.output_nodes

        traverse_order = deque(list(nodes))
        while len(traverse_order) > 0:
            node = traverse_order.popleft()
            if not node:
                continue

            if node == stop_node:
                return []

            if node.op_type.title() == node_type:
                return [node]
            
            # add childs
            if is_input:
                nodes = search_node.input_nodes
            else:
                nodes = search_node.output_nodes
            
            for node in nodes:
                if node not in traverse_order:
                    traverse_order.append(node)

    def _split_by_add_op_type(self, node):
        add_node, other_branch = [], []
        for out_node in node.output_nodes:
            if out_node.op_type == self.add_op_type:
                add_node.append(out_node)
            else:
                other_branch.append(out_node)
        return add_node, other_branch

    def _get_all_interval_nodes_by_type(self,
                                        stop_node,
                                        trav_node,
                                        cls_type=ModuleType.LINEAR,
                                        ignore_stop_node=False,
                                        branch_aware=False):
        matmul_list = []
        if not ignore_stop_node and stop_node.op_type.title() == cls_type:
            matmul_list.append(stop_node)
        if trav_node is not None:
            trav_node = [trav_node]
        while trav_node:
            for trav_node_ in trav_node:
                if trav_node_.op_type.title() == cls_type and trav_node_ not in matmul_list:
                    matmul_list.append(trav_node_)

            trav_node = self._find_latest_module_by_type(
                trav_node[0], cls_type, stop_node, is_input=True, branch_aware=branch_aware)
        return matmul_list

    def _find_input_nodes(self):

        def is_all_inputs_from_none(io_nodes):
            for io_node in io_nodes:
                if io_node.dag_node_from is not None:
                    return False
            return True

        input_nodes = []
        for node in self.node_list:
            if (not node.op_type.startswith('_TensorBase')
                    and node.op_type not in self.dark_name_list
                    and is_all_inputs_from_none(node.inputs)):
                input_nodes.append(node)

        return input_nodes

    def _find_output_nodes(self):
        output_nodes = []
        for node in self.node_list:
            if node.op_type not in self.dark_name_list and \
                    node.outputs and not node.outputs[0].dag_nodes_to:
                output_nodes.append(node)
        return output_nodes

    def _find_latest_module_by_type(self,
                                    search_node: Any,
                                    node_type: str = None,
                                    stop_node: DagNode = None,
                                    is_input: bool = True,
                                    branch_aware: bool = False):
        ret_node = []
        if search_node is None:
            return ret_node

        if is_input:
            nodes = search_node.input_nodes
        else:
            nodes = search_node.output_nodes

        nodes = list(nodes)

        for node in nodes:
            if not node:
                continue

            if node.op_type == ModuleType.CONV2D or node.op_type == ModuleType.GETITEM:
                return ret_node

            if stop_node in list(node.output_nodes):
                continue

            if node == stop_node:
                return ret_node

            if node.op_type.title() == node_type:
                ret_node = [node]
                return ret_node

            found_node = self._find_latest_module_by_type(node,
                                                          node_type,
                                                          stop_node,
                                                          is_input=is_input,
                                                          branch_aware=branch_aware)
            if found_node:
                if branch_aware:
                    found_node = found_node + \
                                 [node for node in nodes if node != found_node[0] and node.op_type.title() == node_type]
                    return found_node
                else:
                    return found_node

        return ret_node