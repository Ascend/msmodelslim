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
# pylint: disable=consider-using-generator,consider-using-set-comprehension

import copy
from typing import Callable, Any, List, Optional, Generator
import operator

import numpy as np


def get_attrs_of_obj(obj, filter_func=None) -> List[Any]:
    if filter_func is None:
        return [getattr(obj, attr_name) for attr_name in dir(obj)]
    else:
        return [attr for attr in (getattr(obj, attr_name) for attr_name in dir(obj)) if filter_func(attr)]


def concatenate_name_in_network(name_in_network: Optional[str], sub_name: str) -> str:
    if name_in_network is None or name_in_network == "":
        return sub_name
    else:
        return name_in_network + "." + sub_name


class FullPermutation:
    cache = [[[]], None, None, None, None, None, None, None]

    @classmethod
    def get_all_permutations(cls, max_index: int) -> Generator[List, None, None]:
        """
        Get all possible permutations.

        Args:
            max_index: max index.

        Examples:
            >>> x = FullPermutation.get_all_permutations(3)
            >>> print(list(x))
            [[2, 1, 0], [1, 2, 0], [1, 0, 2], [2, 0, 1], [0, 2, 1], [0, 1, 2]]
        """
        if not isinstance(max_index, int) or max_index < 0:
            raise ValueError("index must be int")
        if max_index < len(FullPermutation.cache):
            if FullPermutation.cache[max_index] is not None:
                return cls.cache[max_index]
            return_lists = list(cls._get_all_permutations(max_index))
            FullPermutation.cache[max_index] = return_lists
            return return_lists
        else:
            return cls._get_all_permutations(max_index)

    @classmethod
    def get_all_combinations(cls, cnt_list: List[int]) -> Generator[List, None, None]:
        """
        Get all possible combinations.

        Args:
            cnt_list: max index list.

        Examples:
            >>> x = FullPermutation.get_all_combinations([1, 2, 3])
            >>> print(list(x))
            [[0, 0, 0], [0, 1, 0], [0, 0, 1], [0, 1, 1], [0, 0, 2], [0, 1, 2]]
        """
        if cnt_list is None or len(cnt_list) == 0:
            yield []
            return
        ret_combination = [0] * len(cnt_list)
        while True:
            yield copy.copy(ret_combination)
            now_x = ret_combination[0]
            if now_x + 1 < cnt_list[0]:
                ret_combination[0] = now_x + 1
                continue

            for i, max_in_list in enumerate(cnt_list):
                ret_combination[i] += 1
                if ret_combination[i] >= max_in_list:
                    ret_combination[i] = 0
                else:
                    break
            else:
                return

    @classmethod
    def _get_all_permutations(cls, max_index) -> Generator[List, None, None]:
        for index_1_seq in cls.get_all_permutations(max_index - 1):
            for location in range(max_index):
                seq_return = copy.copy(index_1_seq)
                seq_return.insert(location, max_index - 1)
                yield seq_return


class CallParams:
    """
    to save function call params
    """

    def __init__(self, *args, **kwargs):
        self._args = args
        self._kwargs = kwargs

    @property
    def args(self):
        return self._args

    @property
    def kwargs(self):
        return self._kwargs

    def call(self, func: Callable):
        return func(*self.args, **self.kwargs)


class ResListToRelease:
    def __init__(self, *args):
        self.res_list = args

    def __enter__(self):
        for res in self.res_list:
            res.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for res in self.res_list:
            res.__exit__(exc_type, exc_val, exc_tb)


def amp_enabled():
    try:
        from apex import amp
    except ImportError:
        return False
    return hasattr(amp._amp_state, 'handle')


class OperatorAttrName:
    attr_names = set([f"__{x}__" for x in dir(operator) if not x.startswith("__")])


def check_model_backend(model):
    """
    Check model is a MindSpore or PyTorch model.

    Args:
        model: model instance

    Returns: backend name
    """
    if model is None:
        raise ValueError("The model can't be None!")

    try:
        from torch.nn.modules import Module
    except ModuleNotFoundError:
        pass
    else:
        if isinstance(model, Module):
            return "pytorch"

    try:
        from mindspore.nn.cell import Cell
    except ModuleNotFoundError:
        pass
    else:
        if isinstance(model, Cell):
            return "mindspore"

    raise ValueError("The model must be a MindSpore or PyTorch model, and with MindSpore or PyTorch environment!")


def count_parameters(network):
    if hasattr(network, "parameters_dict"):
        import mindspore as ms

        if not isinstance(network, ms.nn.Cell):
            raise TypeError("Provided network is not a mindspore.nn.Cell")

        param_dict = network.parameters_dict(recurse=True)
        return sum([np.prod(param.shape) for param in param_dict.values() if isinstance(param, ms.Parameter)])
    elif hasattr(network, "state_dict"):
        import torch

        if not isinstance(network, torch.nn.Module):
            raise TypeError("Provided network is not a torch.nn.Module")

        # if use state_dict(), will also include non-trainable parameters
        return sum([np.prod(param.shape) for param in network.parameters()])
    else:
        raise AttributeError("network should be an instance of torch.nn.Module or mindspore.nn.Cell")
