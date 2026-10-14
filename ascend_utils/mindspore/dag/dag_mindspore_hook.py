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
# pylint: disable=consider-using-generator,super-with-arguments

from abc import ABC
from typing import Union, Any, List, Tuple

import mindspore
import numpy
from mindspore import Tensor
from mindspore import nn

from ascend_utils.common.utils import CallParams
from ascend_utils.core.dag.dag_hook import DagHook
from ascend_utils.core.dag.dag_node import DagNode


class DagMindSporeHook(DagHook, ABC):
    def __init__(
        self,
        network: nn.Cell,
        inputs: Union[Tensor, List[Tensor], Tuple[Tensor], CallParams],
        hook_ops: Union[List[Any], None] = None,
    ):
        if not isinstance(network, mindspore.nn.Cell):
            raise TypeError("network must be type mindspore.nn.Cell")
        if not isinstance(inputs, (Tensor, list, tuple, CallParams)):
            raise TypeError("inputs must be type Tensor, list, tuple, CallParams")
        if hook_ops is not None and not isinstance(hook_ops, list):
            raise TypeError("hook_nodes must be type list")

        self._tmp_device = dict(device_target="CPU", mode=mindspore.PYNATIVE_MODE)
        super(DagMindSporeHook, self).__init__(network, inputs, hook_ops)

    @staticmethod
    def _parse_network_device():
        return dict(
            device_target=mindspore.context.get_context("device_target"), mode=mindspore.context.get_context("mode")
        )

    def get_params(self) -> int:
        model: mindspore.nn.Cell = self.network
        return sum([numpy.prod(param.shape) for param in model.get_parameters()])

    def replace_node(self, dag_node: DagNode, new_node: mindspore.nn.Cell):
        """
        replace one node by mindspore.nn.Cell
        Args:
            dag_node: replaced node
            new_node: new node

        Returns:
            None

        Examples:
        >>> with DagMindSporeHook(model, inputs) as dag:
        >>>     conv = mindspore.nn.Conv2d(4, 64, kernel_size=11, stride=4, padding=2)
        >>>     dag.replace_node(dag.get_node_by_name("feature.0"), conv)
        """
        if not isinstance(dag_node, DagNode):
            raise TypeError("dag_node must be type DagNode")
        if not isinstance(new_node, mindspore.nn.Cell):
            raise TypeError("new_node must be type mindspore.nn.Cell")

        super(DagMindSporeHook, self).replace_node(dag_node, new_node)

    def add_node_before(self, dag_node: DagNode, preprocess_module: mindspore.nn.Cell):
        """
        add preprocess node before
        Args:
            dag_node: dag node need preprocess
            preprocess_module: preprocess module

        Returns:
            None

        Examples:
        >>> with DagMindSporeHook(model, inputs) as dag:
        >>>     some = PreprocessModule()
        >>>     dag.add_node_before(dag.get_node_by_name("feature.0"), some)
        """
        module = mindspore.nn.SequentialCell(preprocess_module, dag_node.node)
        self.replace_node(dag_node, module)

    def add_node_after(self, dag_node: DagNode, postprocess_module: mindspore.nn.Cell):
        """
        add postprocess node before
        Args:
            dag_node: dag node need preprocess
            postprocess_module: postprocess module

        Returns:
            None

        Examples:
        >>> with DagMindSporeHook(model, inputs) as dag:
        >>>     some = PostprocessModule()
        >>>     dag.add_node_after(dag.get_node_by_name("feature.0"), some)
        """
        module = mindspore.nn.SequentialCell(dag_node.node, postprocess_module)
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
        >>> with DagMindSporeHook(model, inputs) as dag:
        >>>     dag.remove_node(dag.get_node_by_name("feature.0"))
        """
        if check_io:
            if len(dag_node.inputs) != len(dag_node.outputs):
                raise ValueError("remove node must input eq output")

        class Eq(mindspore.nn.Cell):
            def construct(self, *args, **kwargs):
                if len(args) == 1:
                    return args[0]
                return args if len(args) > 0 else kwargs

        self.replace_node(dag_node, Eq())

    def _before_parse(self):
        self._tmp_device = self._parse_network_device()
        mindspore.context.set_context(device_target="CPU", mode=mindspore.PYNATIVE_MODE)

    def _after_parse(self):
        # __new__ 节点全部和操作合并掉
        for _, dag_node in enumerate(list(self.dag_node_list)):
            if dag_node.op_type == "__new__":
                self._remove_one_node(dag_node)

        mindspore.context.set_context(**self._tmp_device)

    def _get_module_children(self, module):
        return module.name_cells().items()

    def _get_module_cls(self):
        return mindspore.nn.Cell

    def _replace_node(self, parent_module, name_in_parent, new_node):
        setattr(parent_module, name_in_parent, new_node)
        try:
            if isinstance(parent_module, (nn.SequentialCell, nn.CellList)) and name_in_parent:
                index_in_parent = int(name_in_parent)
                if 0 <= index_in_parent < len(parent_module):
                    parent_module[index_in_parent] = new_node
                else:
                    raise ValueError("error index in parent module")
        except ValueError as ex:
            raise ValueError("error index in parent module") from ex

    def _collecting_feature_map_info(self, output):
        io_info = {}
        if isinstance(output, Tensor):
            io_info["shape"] = output.shape
            io_info["dtype"] = output.dtype
        return io_info

    def _get_all_hook_ops(self, user_hook_ops) -> List[Tuple[Any, Any, str, str]]:
        mindspore_func_hook_infos = self._get_function_hook_infos(mindspore)
        tensor_func_hook_infos = [x for x in self._get_function_hook_infos(Tensor) if not x[2].startswith("set_")]
        ops_func_hook_infos = self._get_function_hook_infos(mindspore.ops)

        tensor_operator_hook_infos = self._get_operator_hook_infos(Tensor)

        unhook_modules = [mindspore.nn.SequentialCell, mindspore.nn.CellList]
        nn_modules_hook_infos = self._get_class_hook_infos(mindspore.nn, mindspore.nn.Cell)
        nn_modules_hook_infos = [
            (getattr(cell, "construct"), (cell, "construct"), name)
            for cell, _, name in nn_modules_hook_infos
            if cell not in unhook_modules
        ]

        ops_cells = self._get_class_hook_infos(mindspore.ops, mindspore.ops.primitive.Primitive)
        ops_call_hook_infos = [
            (getattr(cell, "__call__"), (cell, "__call__"), name)
            for cell, _, name in ops_cells
            if cell not in unhook_modules
        ]
        ops_new_hook_infos = [
            (getattr(cell, "__new__"), (cell, "__new__"), "__new__")
            for cell, _, name in ops_cells
            if cell not in unhook_modules
        ]

        user_hook_infos = []
        if user_hook_ops is not None:
            for hook_op in user_hook_ops:
                if issubclass(hook_op, mindspore.nn.Cell):
                    user_hook_infos.append((getattr(hook_op, "construct"), (hook_op, "construct"), hook_op.__name__))
                else:
                    user_hook_infos.append((hook_op, hook_op, hook_op.__qualname__))
        return (
            mindspore_func_hook_infos
            + tensor_func_hook_infos
            + tensor_operator_hook_infos
            + ops_func_hook_infos
            + ops_call_hook_infos
            + ops_new_hook_infos
            + nn_modules_hook_infos
            + user_hook_infos
        )
