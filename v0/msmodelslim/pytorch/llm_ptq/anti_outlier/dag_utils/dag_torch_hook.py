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
from abc import ABC
from typing import Union, Any, List, Tuple

import torch
from torch import Tensor
from torch.nn import Module, functional

from ascend_utils.common.utils import CallParams
from ascend_utils.core.dag.dag_node import DagNode
from .dag_hook import DagHook


class DagTorchHook(DagHook, ABC):
    """
        hook方式获取torch的DAG图

        优势：
            1. 简单好理解
        劣势：
            1. 无法替换函数类节点
            2. 无法跟踪到多个 if 分支
            3. 如果存在共享卷积（在不同的父Module 中）无法正常替换
            4. 如果一个变量直接穿透算子（既是输入又是输出），以计算顺序为准，以最后一次计算输出作为后面使用的输入，可能与代码写法不一致
    """

    def __init__(self,
                 network: Module,
                 inputs: Union[Tensor, List[Tensor], Tuple[Tensor], CallParams],
                 hook_ops: Union[List[Any], None] = None,
                 anti_method=None):
        if not isinstance(network, torch.nn.Module):
            raise TypeError("network must be type torch.nn.Module")
        if not isinstance(inputs, (Tensor, list, tuple, dict, CallParams)):
            raise TypeError("inputs must be type Tensor, list, tuple, CallParams")
        if hook_ops is not None and not isinstance(hook_ops, list):
            raise TypeError("hook_nodes must be type list")

        self._tmp_device = "CPU"

        super().__init__(network, inputs, hook_ops, anti_method=anti_method)

    @staticmethod
    def get_obj_module_attrs(obj):
        attrs = (getattr(obj, attr_name) for attr_name in dir(obj))

        module_attrs = (attr for attr in attrs if inspect.isclass(attr) and issubclass(attr, torch.nn.Module))
        return module_attrs

    @staticmethod
    def _parse_network_device(network: torch.nn.Module):
        for _, param in network.named_parameters():
            _device = param.device
            return _device

    @staticmethod
    def _input_item_to_cpu(input_item):
        if isinstance(input_item, Tensor):
            return input_item.cpu()
        else:
            return input_item

    @classmethod
    def input_to_cpu(cls, inputs):
        if isinstance(inputs, CallParams):
            input_args = []
            input_kwargs = dict()
            for input_item in inputs.args:
                input_args.append(cls._input_item_to_cpu(input_item))
            for input_key, input_item in inputs.kwargs.items():
                input_kwargs[input_key] = cls._input_item_to_cpu(input_item)
            input_data = CallParams(*input_args, **input_kwargs)
        elif isinstance(inputs, tuple) or isinstance(inputs, list):
            input_data = []
            for input_item in inputs:
                input_data.append(cls._input_item_to_cpu(input_item))
        elif isinstance(inputs, dict):
            input_data = {}
            for key, input_item in inputs.items():
                input_data[key] = cls._input_item_to_cpu(input_item)
        else:
            input_data = cls._input_item_to_cpu(inputs)
        return input_data

    def get_params(self) -> int:
        return sum([param.nelement() for param in self.network.parameters()])

    def replace_node(self, dag_node: DagNode, new_node: torch.nn.Module):
        """
        replace one node by torch.nn.Module
        Args:
            dag_node: replaced node
            new_node: new node

        Returns:
            None

        Examples:
        >>> with DagTorchHook(model, inputs) as dag:
        >>>     conv = torch.nn.Conv2d(4, 64, kernel_size=11, stride=4, padding=2)
        >>>     dag.replace_node(dag.get_node_by_name("feature.0"), conv)
        """
        if not isinstance(dag_node, DagNode):
            raise TypeError("dag_node must be type DagNode")
        if not isinstance(new_node, torch.nn.Module):
            raise TypeError("new_node must be type torch.nn.Module")

        super(DagTorchHook, self).replace_node(dag_node, new_node)

    def add_node_before(self, dag_node: DagNode, preprocess_module: Module):
        """
        add preprocess node before
        Args:
            dag_node: dag node need preprocess
            preprocess_module: preprocess module

        Returns:
            None

        Examples:
        >>> with DagTorchHook(model, inputs) as dag:
        >>>     some = PreprocessModule()
        >>>     dag.add_node_before(dag.get_node_by_name("feature.0"), some)
        """
        module = torch.nn.Sequential(preprocess_module, dag_node.node)
        self.replace_node(dag_node, module)

    def add_node_after(self, dag_node: DagNode, postprocess_module: Module):
        """
        add postprocess node before
        Args:
            dag_node: dag node need preprocess
            postprocess_module: postprocess module

        Returns:
            None

        Examples:
        >>> with DagTorchHook(model, inputs) as dag:
        >>>     some = PostprocessModule()
        >>>     dag.add_node_after(dag.get_node_by_name("feature.0"), some)
        """
        module = torch.nn.Sequential(dag_node.node, postprocess_module)
        self.replace_node(dag_node, module)

    def remove_node(self, dag_node: DagNode, check_io=True):
        """
        remove node
        Args:
            dag_node: dag node need preprocess
            check_io: check input count and output count

        Returns:
            None

        Examples:
        >>> with DagTorchHook(model, inputs) as dag:
        >>>     dag.remove_node(dag.get_node_by_name("feature.0"))
        """
        if check_io:
            if len(dag_node.inputs) != len(dag_node.outputs):
                raise ValueError("remove node must input eq output")

        class Eq(torch.nn.Module):
            def forward(self, *args, **kwargs):
                if len(args) == 1:
                    return args[0]
                return args if len(args) > 0 else kwargs

        self.replace_node(dag_node, Eq())
        
    def get_module_cls(self):
        return torch.nn.Module

    def _after_parse(self):
        self.network.to(self._tmp_device)

    def _before_parse(self):
        self._tmp_device = self._parse_network_device(self.network)
        self.network.cpu()

    def _get_module_children(self, module):
        return module.named_children()

    def _collecting_feature_map_info(self, output):
        io_info = {}
        if isinstance(output, Tensor):
            io_info["shape"] = output.shape
            io_info["dtype"] = output.dtype
        return io_info

    def _get_all_hook_ops(self, user_hook_ops) -> List[Tuple[Any, Any, str, str]]:
        forward_str = "forward"
        nn_modules_hook_infos = self._get_class_hook_infos(torch.nn, torch.nn.Module)
        hook_modules = [torch.nn.Linear]
        linear_hook_infos = [(getattr(cell, forward_str), (cell, forward_str), name)
                                 for cell, _, name in nn_modules_hook_infos
                                 if cell in hook_modules]
        
        user_hook_infos = []
        if user_hook_ops is not None:
            for hook_op in user_hook_ops:
                if inspect.isclass(hook_op) and issubclass(hook_op, torch.nn.Module):
                    user_hook_infos.append((getattr(hook_op, forward_str), (hook_op, forward_str), hook_op.__name__))
                else:
                    user_hook_infos.append((hook_op, hook_op, hook_op.__qualname__))
        return linear_hook_infos + user_hook_infos

    def _get_hook_ops(self, user_hook_ops) -> List[Tuple[Any, Any, str, str]]:
        forward_str = "forward"
        torch_func_hook_infos = self._get_function_hook_infos(torch)
        tensor_func_hook_infos = self._get_function_hook_infos(Tensor)
        functional_func_hook_infos = self._get_function_hook_infos(functional)
        tensor_operator_hook_infos = self._get_operator_hook_infos(Tensor)

        unhook_modules = [
                              torch.nn.Sequential, torch.nn.Container, torch.nn.ModuleList, torch.nn.ModuleDict,
                              torch.nn.ParameterList, torch.nn.ParameterDict, torch.nn.modules.module.Module, 
                              torch.nn.modules.dropout.Dropout, torch.nn.modules.activation.SiLU
        ]
        nn_modules_hook_infos = self._get_class_hook_infos(torch.nn, torch.nn.Module)
        nn_modules_hook_infos = [(getattr(cell, forward_str), (cell, forward_str), name)
                                 for cell, _, name in nn_modules_hook_infos
                                 if cell not in unhook_modules]

        user_hook_infos = []
        if user_hook_ops is not None:
            for hook_op in user_hook_ops:
                if inspect.isclass(hook_op) and issubclass(hook_op, torch.nn.Module):
                    user_hook_infos.append((getattr(hook_op, forward_str), (hook_op, forward_str), hook_op.__name__))
                else:
                    user_hook_infos.append((hook_op, hook_op, hook_op.__qualname__))
        return torch_func_hook_infos + tensor_func_hook_infos + functional_func_hook_infos + \
               tensor_operator_hook_infos + nn_modules_hook_infos + user_hook_infos
