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
# pylint: disable=attribute-defined-outside-init,unnecessary-dunder-call

import pytest
import operator  # noqa: F401
import numpy as np  # noqa: F401
from unittest.mock import patch, MagicMock
from ascend_utils.common.utils import (
    get_attrs_of_obj,
    concatenate_name_in_network,
    FullPermutation,
    CallParams,
    ResListToRelease,
    amp_enabled,
    OperatorAttrName,
    check_model_backend,
    count_parameters,
)


class TestObj:
    def __init__(self):
        self.attr1 = "value1"
        self.attr2 = 42
        self._private_attr = "private"


# Test cases for get_attrs_of_obj
def test_get_attrs_of_obj_given_obj_when_no_filter_then_return_all_attrs():
    obj = TestObj()
    attrs = get_attrs_of_obj(obj)
    assert "attr1" in str(attrs)
    assert "attr2" in str(attrs)
    assert "_private_attr" in str(attrs)


def test_get_attrs_of_obj_given_obj_when_filter_then_return_filtered_attrs():
    obj = TestObj()
    attrs = get_attrs_of_obj(obj, filter_func=lambda x: isinstance(x, str))
    assert all(isinstance(attr, str) for attr in attrs)


# Test cases for concatenate_name_in_network
def test_concatenate_name_in_network_given_none_and_name_when_called_then_return_subname():
    result = concatenate_name_in_network(None, "subname")
    assert result == "subname"


def test_concatenate_name_in_network_given_empty_and_name_when_called_then_return_subname():
    result = concatenate_name_in_network("", "subname")
    assert result == "subname"


def test_concatenate_name_in_network_given_name_and_subname_when_called_then_return_concatenated():
    result = concatenate_name_in_network("parent", "child")
    assert result == "parent.child"


# Test cases for FullPermutation
def test_fullpermutation_get_all_permutations_given_negative_when_called_then_raise_error():
    with pytest.raises(ValueError):
        list(FullPermutation.get_all_permutations(-1))


def test_fullpermutation_get_all_permutations_given_non_int_when_called_then_raise_error():
    with pytest.raises(ValueError):
        list(FullPermutation.get_all_permutations("3"))


def test_fullpermutation_get_all_permutations_given_zero_when_called_then_return_empty_list():
    result = list(FullPermutation.get_all_permutations(0))
    assert result == [[]]


def test_fullpermutation_get_all_permutations_given_three_when_called_then_return_all_permutations():
    result = list(FullPermutation.get_all_permutations(3))
    assert len(result) == 6
    assert [2, 1, 0] in result
    assert [0, 1, 2] in result


def test_fullpermutation_get_all_combinations_given_empty_list_when_called_then_return_empty_list():
    result = list(FullPermutation.get_all_combinations([]))
    assert result == [[]]


def test_fullpermutation_get_all_combinations_given_single_item_when_called_then_return_single_combinations():
    result = list(FullPermutation.get_all_combinations([2]))
    assert result == [[0], [1]]


def test_fullpermutation_get_all_combinations_given_multiple_items_when_called_then_return_all_combinations():
    result = list(FullPermutation.get_all_combinations([1, 2]))
    assert len(result) == 2
    assert [0, 0] in result
    assert [0, 1] in result


# Test cases for CallParams
def test_callparams_init_given_args_kwargs_when_called_then_store_values():
    params = CallParams(1, 2, a=3, b=4)
    assert params.args == (1, 2)
    assert params.kwargs == {"a": 3, "b": 4}


def test_callparams_call_given_function_when_called_then_execute_function():
    def test_func(a, b, c=0):
        return a + b + c

    params = CallParams(1, 2, c=3)
    result = params.call(test_func)
    assert result == 6


# Test cases for ResListToRelease
class MockResource:
    def __enter__(self):
        self.entered = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.exited = True


def test_reslisttorelease_enter_given_resources_when_called_then_enter_all():
    res1 = MockResource()
    res2 = MockResource()
    with ResListToRelease(res1, res2) as rl:  # noqa: F841
        assert res1.entered
        assert res2.entered


def test_reslisttorelease_exit_given_resources_when_called_then_exit_all():
    res1 = MockResource()
    res2 = MockResource()
    rl = ResListToRelease(res1, res2)
    rl.__enter__()
    rl.__exit__(None, None, None)
    assert hasattr(res1, 'exited')
    assert hasattr(res2, 'exited')


# Test cases for amp_enabled
def test_amp_enabled_given_apex_available_when_amp_enabled_then_return_true():
    with patch.dict('sys.modules', {'apex': MagicMock()}):
        with patch('apex.amp._amp_state.handle', True):
            assert amp_enabled()


def test_amp_enabled_given_apex_not_available_when_called_then_return_false():
    with patch.dict('sys.modules', {'apex': None}):
        assert not amp_enabled()


# Test cases for OperatorAttrName
def test_operatorattrname_attr_names_when_accessed_then_contains_operator_methods():
    assert "__add__" in OperatorAttrName.attr_names
    assert "__sub__" in OperatorAttrName.attr_names


# Test cases for check_model_backend
def test_check_model_backend_given_none_when_called_then_raise_error():
    with pytest.raises(ValueError):
        check_model_backend(None)


def test_check_model_backend_given_pytorch_model_when_called_then_return_pytorch():
    mock_torch = MagicMock()
    mock_module = MagicMock()
    mock_module.Module = object
    with patch.dict('sys.modules', {'torch.nn.modules': mock_module}):
        model = mock_torch.nn.Module()
        assert check_model_backend(model) == "pytorch"


def test_check_model_backend_given_mindspore_model_when_called_then_return_mindspore():
    mock_ms = MagicMock()
    mock_cell = MagicMock()
    mock_cell.Cell = object
    with patch.dict('sys.modules', {'mindspore.nn.cell': mock_cell}):
        model = mock_ms.nn.Cell()
        assert check_model_backend(model) == "mindspore"


def test_check_model_backend_given_invalid_model_when_called_then_raise_error():
    with pytest.raises(ValueError):
        check_model_backend("not a model")


# Test cases for count_parameters
def test_count_parameters_given_invalid_network_when_called_then_raise_error():
    with pytest.raises(AttributeError):
        count_parameters("not a network")
