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


import functools
import inspect
from abc import ABC, abstractmethod
from typing import Union, Optional, Callable, Any, List, Tuple, Dict, Iterable, NoReturn, Set

import torch

from ascend_utils.common.hook import FunctionReplace
from ascend_utils.common.utils import (
    CallParams, ResListToRelease, concatenate_name_in_network, OperatorAttrName
)
from ascend_utils.core.dag.dag import DirectedAcyclicGraph
from ascend_utils.core.dag.dag_node import DagNode
from ascend_utils.core.dag.dag_node_io import DagNodeIO

from msmodelslim import logger

from .dag_model_hook import DagModelHook


class DagHook(DirectedAcyclicGraph, ABC):
    """
        hook方式获取的DAG图

        优势：
            1. 简单好理解
        劣势：
            1. 无法替换函数类节点
            2. 不支持多个 if 分支
            3. 如果存在共享卷积（在不同的父Module 中）无法正常替换
            4. 如果一个变量直接穿透算子（既是输入又是输出），以计算顺序为准，以最后一次计算输出作为后面使用的输入，可能与代码写法不一致
    """

    def __init__(self,
                 network,
                 inputs,
                 hook_ops: Union[List[Any], None] = None,
                 anti_method=None):
        super().__init__(network)
        self._inputs = inputs
        self._hook_ops = self._get_all_hook_ops(hook_ops)
        self._structure_tree: Dict[int, Dict] = {}
        self._replaced_nodes: Set[DagNode] = set()

        self._parse_network_structure_tree(self.network, "", None, "")
        self._parse_network_with_hook(self._inputs)

    def __enter__(self):
        """
        enter to change this network

        Returns: self
        """
        self._before_parse()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._reparse_network()
        self._after_parse()

    @property
    def structure_tree(self):
        return self._structure_tree

    @staticmethod
    def _call_ori_func(func, *args, **kwargs):
        return func(*args, **kwargs)

    @staticmethod
    def _get_attr_names(obj: Any, filter_func: Optional[Callable]) -> List[str]:
        if obj is None:
            return []
        if filter_func is not None:
            return [attr_name for attr_name in dir(obj) if filter_func(getattr(obj, attr_name), attr_name)]
        else:
            return [attr_name for attr_name in dir(obj)]

    @staticmethod
    def _get_ops_hook_info(obj, attr_names) -> List[Tuple[Any, Tuple, str]]:
        return [(getattr(obj, name), (obj, name), name) for name in attr_names]

    @staticmethod
    def _replace_node(parent_module, name_in_parent, new_node):
        setattr(parent_module, name_in_parent, new_node)

    @classmethod
    def _get_operator_hook_infos(cls, obj) -> List[Tuple[Any, Tuple, str]]:
        def is_operator(_, name):
            return name in OperatorAttrName.attr_names

        return cls._get_ops_hook_info(obj, cls._get_attr_names(obj, is_operator))

    @classmethod
    def _get_class_hook_infos(cls, obj, cls_type) -> List[Tuple[Any, Tuple, str]]:
        def is_class(attr, _):
            return inspect.isclass(attr) and issubclass(attr, cls_type)

        return cls._get_ops_hook_info(obj, cls._get_attr_names(obj, is_class))

    @classmethod
    def _get_function_hook_infos(cls, obj) -> List[Tuple[Any, Tuple, str]]:
        def is_function(attr, name):
            return not inspect.isclass(attr) and not name.startswith("_") and callable(attr)

        return cls._get_ops_hook_info(obj, cls._get_attr_names(obj, is_function))

    def get_params(self) -> int:
        return sum([param.nelement() for param in self.network.parameters()])

    def replace_node(self, dag_node: DagNode, new_node):
        """
        replace one node by mindspore.nn.Cell/ torch.nn.Module
        Args:
            dag_node: replaced node
            new_node: new node

        Returns:
            None
        """
        old_node = dag_node.node
        node_struct_info = self._structure_tree.get(id(old_node), None)
        parent_module_infos = None if node_struct_info is None else node_struct_info.get("parent_module_info", None)
        if not isinstance(parent_module_infos, list) or len(parent_module_infos) != 1:
            raise ValueError("node must has just 1 parent")
        if parent_module_infos is None:
            raise ValueError("cannot replace this node")

        parent_module, name_in_parent = parent_module_infos[0]
        self._replace_node(parent_module, name_in_parent, new_node)
        # 更新网络，并标记需要重新解析
        dag_node.replace(new_node, type(new_node).__name__)
        self._structure_tree[id(new_node)] = node_struct_info
        self._replaced_nodes.add(dag_node)

    @abstractmethod
    def _before_parse(self):
        pass

    @abstractmethod
    def _after_parse(self):
        pass

    @abstractmethod
    def _get_module_children(self, module):
        pass

    @abstractmethod
    def get_module_cls(self):
        pass

    @abstractmethod
    def _collecting_feature_map_info(self, output):
        pass

    @abstractmethod
    def _get_all_hook_ops(self, user_hook_ops) -> List[Tuple[Any, Any, str]]:
        """
        get all need hook ops, include default ops and user ops

        Args:
            user_hook_ops: user hook ops

        Returns:
            [(ops instances, ops owner module, attr name, ops name)]
        """
        pass

    def get_node_name(self, node_struct_info, node_hook):
        if node_struct_info is not None and "name_in_network" in node_struct_info:
            return node_struct_info.get("name_in_network")
        else:
            if hasattr(node_hook, "__name__"):
                node_type_name = node_hook.__name__
            elif hasattr(node_hook, "name"):
                node_type_name = node_hook.name
            else:
                node_type_name = ""

            return node_type_name.strip("_") + "_" + str(len(self._dag_node_list))

    def get_node_input(self, node_io_dict: Dict[int, DagNodeIO], *args, **kwargs) -> List[DagNodeIO]:
        inputs: List[DagNodeIO] = []

        inputs.extend(self._get_node_input_in_gen(node_io_dict, enumerate(args)))
        inputs.extend(self._get_node_input_in_gen(node_io_dict, kwargs.items()))

        return inputs

    def get_node_output(self, output, deduplicate: List[int], name: str = "output") -> Dict[int, DagNodeIO]:
        node_output_dict: Dict[int, DagNodeIO] = {}
        id_output = id(output)
        if id_output in deduplicate:
            return node_output_dict
        else:
            deduplicate.append(id_output)

        io_info = self._collecting_feature_map_info(output)
        dag_node_output = DagNodeIO(id_output, name, io_info)
        node_output_dict[id_output] = dag_node_output

        if isinstance(output, dict):
            for key, sub_output in output.items():
                node_output_dict.update(
                    self.get_node_output(sub_output, deduplicate, name + '.' + str(key)))
        elif isinstance(output, list) or isinstance(output, tuple):
            for index, sub_output in enumerate(output):
                node_output_dict.update(
                    self.get_node_output(sub_output, deduplicate, name + '.' + str(index)))

        return node_output_dict

    def _parse_network_structure_tree(self, module,
                                      name: str,
                                      parent_module: Optional,
                                      name_in_network: str) -> NoReturn:
        index_module = id(module)
        if index_module in self._structure_tree:
            if "parent_module_info" in self._structure_tree[index_module]:
                self._structure_tree[index_module]["parent_module_info"] = [(parent_module, name)]
            else:
                self._structure_tree[index_module]["parent_module_info"].append((parent_module, name))
            return

        self._structure_tree[index_module] = {
            "name_in_network": name_in_network, "parent_module_info": [(parent_module, name)]
        }

        for sub_name, sub_module in self._get_module_children(module):
            sub_name_in_network = concatenate_name_in_network(name_in_network, sub_name)
            self._parse_network_structure_tree(sub_module, sub_name, module, sub_name_in_network)

    def _parse_network_with_hook(self, inputs,
                                 parsed_nodes: Optional[Dict[Any, DagNode]] = None):
        self._dag_node_list.clear()
        # prepare hook function
        replace_stack: List[Any] = []
        node_io_dict: Dict[int, DagNodeIO] = {}
        parsed_node_list = [] if parsed_nodes is None else parsed_nodes

        # create hook for special module types
        hooks = {}
        for op_hook_info in self._hook_ops:
            _, ops_location, ops_name = op_hook_info
            module_class, _ = ops_location
            hooks[module_class] = DagModelHook(ops_name=ops_name,
                                               replace_stack=replace_stack,
                                               node_io_dict=node_io_dict,
                                               parsed_node_list=parsed_node_list,
                                               dag_hook=self)

        # register hook to modules of special types
        registered_hooks = []
        for _, m in self.network.named_modules():
            for module_class, hook in hooks.items():
                if isinstance(m, module_class):
                    registered_hooks.append(m.register_forward_pre_hook(hook.get_pre_forward_hook(), with_kwargs=True))
                    registered_hooks.append(m.register_forward_hook(hook.get_post_forward_hook()))

        # network construct and parse network
        try:
            if isinstance(inputs, CallParams):
                self.network(*inputs.args, **inputs.kwargs)
            elif isinstance(inputs, tuple) or isinstance(inputs, list):
                self.network(*inputs)
            elif isinstance(inputs, dict):
                self.network(**inputs)
            else:
                self.network(inputs)
        except RuntimeError as ex:
            raise ValueError("Check whether the input is of the current network.") from ex

        # remove dag hook
        for hook in registered_hooks:
            hook.remove()

        logger.info("parse network over")

    def _parse_network(self, inputs,
                       parsed_nodes: Optional[Dict[Any, DagNode]] = None):
        self._dag_node_list.clear()
        # prepare hook function
        replace_stack: List[Any] = []
        node_io_dict: Dict[int, DagNodeIO] = {}
        replace_functions: List[FunctionReplace] = []
        parsed_node_list = [] if parsed_nodes is None else parsed_nodes

        for op_hook_info in self._hook_ops:
            node_ins, ops_location, ops_name = op_hook_info
            node_wrapper = self._get_node_wrapper(op_hook_info, replace_stack, node_io_dict, parsed_node_list)
            replace_functions.append(FunctionReplace(ops_location, node_wrapper))

        # 在多卡的情况下，deepcopy的这个模型需要去掉之前accelerate备份的每个module的forward函数，来保持正常的forward流程
        init_with_accelerate = hasattr(self.network, 'hf_device_map')
        forward_str = 'forward'
        if init_with_accelerate:
            ops_list = []
            for op_hook_info in self._hook_ops:
                node_ins, ops_location, ops_name = op_hook_info
                if ops_location[1] == forward_str:
                    ops_list.append(ops_location[0])
            ops_tuple = tuple(ops_list)

            for mod in self.network.modules():
                values = set(self.network.hf_device_map.values())
                init_with_accelerate_parallel = False if len(values) == 1 else True
                is_target_ops = isinstance(mod, ops_tuple)
                if hasattr(mod, forward_str) and is_target_ops and init_with_accelerate_parallel:
                    delattr(mod, forward_str)

        # network construct and parse network
        with ResListToRelease(*replace_functions):
            try:
                if isinstance(inputs, CallParams):
                    self.network(*inputs.args, **inputs.kwargs)
                elif isinstance(inputs, tuple) or isinstance(inputs, list):
                    self.network(*inputs)
                elif isinstance(inputs, dict):
                    self.network(**inputs)
                else:
                    self.network(inputs)
            except RuntimeError as ex:
                raise ValueError("Check whether the input is of the current network.") from ex

        logger.info("parse network over")

    def _get_node_wrapper(self, op_hook_info: Tuple[Callable, Any, str], replace_stack: List[Any],
                          node_io_dict: Dict[int, DagNodeIO],
                          parsed_nodes: Dict[Any, DagNode]) -> Callable:
        node_hook, _, ops_type = op_hook_info

        @functools.wraps(node_hook)
        def wrapper(*args, **kwargs):
            if len(replace_stack) > 0:
                return self._call_ori_func(node_hook, *args, **kwargs)

            # before call node
            replace_stack.append(node_hook)
            # record input info
            if len(args) > 0 and isinstance(args[0], self.get_module_cls()):  # param self is Module
                module_self = args[0]
                inputs = self.get_node_input(node_io_dict, *args[1:], **kwargs)
                node_struct_info = self._structure_tree.get(id(module_self), None)
                name_in_network = self.get_node_name(node_struct_info, node_hook)
                # prevent the absence of accelarate hook cause device error
                if hasattr(module_self, "_old_module"):
                    dev = next(module_self.parameters()).device
                    new_args = [aa.to(dev) if isinstance(aa, torch.Tensor) and dev != aa.device else aa for aa in args]
                    args = tuple(new_args)  # args is a tuple, so reconstruct one
            else:
                module_self = node_hook
                inputs = self.get_node_input(node_io_dict, *args, **kwargs)
                name_in_network = self.get_node_name(None, node_hook)

            # call node
            output = self._call_ori_func(node_hook, *args, **kwargs)

            # after call node

            # record output info
            outputs_dict: Dict[int, DagNodeIO] = self.get_node_output(output, [], name_in_network + ":output")
            node_io_dict.update(outputs_dict)
            outputs = list(outputs_dict.values())

            # record node info
            if isinstance(module_self, self.get_module_cls()) and module_self in parsed_nodes:
                dag_node = parsed_nodes[module_self]
                dag_node.set_node_io(inputs, outputs)
            else:
                dag_node: DagNode = DagNode(module_self, name_in_network, ops_type, inputs, outputs)
            self._dag_node_list.append(dag_node)
            replace_stack.pop()
            return output

        return wrapper

    def _get_node_input_in_gen(self, node_io_dict: Dict[int, DagNodeIO], gen: Iterable) -> List[DagNodeIO]:
        inputs: List[DagNodeIO] = []
        for idx, argument in gen:
            id_input = id(argument)
            if id_input not in node_io_dict:
                io_info = self._collecting_feature_map_info(id_input)
                new_node_io = DagNodeIO(id_input, str(idx), io_info)
                node_io_dict[id_input] = new_node_io

            inputs.append(node_io_dict[id_input])
            if isinstance(argument, list) or isinstance(argument, tuple):
                inputs.extend(self.get_node_input(node_io_dict, *argument))
            if isinstance(argument, dict):
                inputs.extend(self.get_node_input(node_io_dict, **argument))

        return inputs

    def _reparse_network(self):
        for node in self._replaced_nodes:
            if node in self.dag_node_list:
                self._dag_node_list.remove(node)

        parsed_nodes = {}
        for dag_node in self.dag_node_list:
            if dag_node.node in parsed_nodes:
                parsed_nodes[dag_node.node] = None  # 去重，Relu等共享算子的，都全部重新解析，否则会存在问题
            else:
                parsed_nodes[dag_node.node] = dag_node

        parsed_nodes = {k: v for k, v in parsed_nodes.items() if v is not None}

        self._structure_tree: Dict[int, Dict] = {}
        self._parse_network_structure_tree(self.network, "", None, "")
        self._parse_network_with_hook(self._inputs, parsed_nodes)
