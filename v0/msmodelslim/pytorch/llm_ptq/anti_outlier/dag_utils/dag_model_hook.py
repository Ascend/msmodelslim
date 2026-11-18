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


from typing import Dict

import accelerate

from ascend_utils.core.dag.dag_node import DagNode
from ascend_utils.core.dag.dag_node_io import DagNodeIO


class DagModelHook(accelerate.hooks.ModelHook):
    """
    用于获取输入输出的特征
    """

    def __init__(self, ops_name, replace_stack, node_io_dict, parsed_node_list, dag_hook):
        super().__init__()
        self.ops_type = ops_name
        self.replace_stack = replace_stack
        self.node_io_dict = node_io_dict
        self.parsed_node_list = parsed_node_list
        self.dag_hook = dag_hook
        self.infos = {}

    def get_pre_forward_hook(self):
        return lambda model, args, kwargs: self.pre_forward(model, *args, **kwargs)

    def get_post_forward_hook(self):
        return lambda model, args, output: self.post_forward(model, output)

    def pre_forward(self, model, *args, **kwargs):
        if len(self.replace_stack) > 0:
            return args, kwargs

        # before call node
        self.replace_stack.append(model)
        # record input info
        node_inputs = self.dag_hook.get_node_input(self.node_io_dict, *args, **kwargs)
        node_struct_info = self.dag_hook.structure_tree.get(id(model), None)
        name_in_network = self.dag_hook.get_node_name(node_struct_info, model)
        self.infos[id(model)] = (node_inputs, name_in_network)
        return args, kwargs

    def post_forward(self, model, output):
        if len(self.replace_stack) == 0 or self.replace_stack[-1] is not model:
            return output

        node_inputs, name_in_network = self.infos[id(model)]
        # record output info
        outputs_dict: Dict[int, DagNodeIO] = self.dag_hook.get_node_output(output, [], name_in_network + ":output")
        self.node_io_dict.update(outputs_dict)
        node_outputs = list(outputs_dict.values())

        # record node info
        if isinstance(model, self.dag_hook.get_module_cls()) and model in self.parsed_node_list:
            dag_node = self.parsed_node_list[model]
            dag_node.set_node_io(node_inputs, node_outputs)
        else:
            dag_node: DagNode = DagNode(model, name_in_network, self.ops_type, node_inputs, node_outputs)
        self.dag_hook.dag_node_list.append(dag_node)
        self.replace_stack.pop()

        # 清理内存
        del self.infos[id(model)]

        return output
