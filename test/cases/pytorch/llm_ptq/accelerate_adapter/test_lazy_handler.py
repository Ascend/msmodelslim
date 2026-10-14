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

from typing import MutableMapping

import pytest
import torch

from msmodelslim.pytorch.llm_ptq.accelerate_adapter.lazy_handler import LazyTensor, get_tensor_size, handle_lazy_tensor


class TestLazyTensor:
    @staticmethod
    @pytest.mark.parametrize(('x', 'y'), [(torch.ones([2, 2]), LazyTensor(lambda: torch.ones([2, 2])))])
    def test_generate_value(x: torch.Tensor, y: LazyTensor):
        assert torch.equal(x, y.value), f"LazyTensor value not functional:\n{x}\n{y}"

    @staticmethod
    @pytest.mark.parametrize(
        'y', [LazyTensor(lambda: torch.ones([2, 2])), LazyTensor(lambda: torch.ones([2, 2]), torch.ones([2, 2]))]
    )
    @pytest.mark.parametrize('x', [torch.ones([2, 2])])
    def test_get_size(x: torch.Tensor, y: LazyTensor):
        assert get_tensor_size(x) == y.size, f"LazyTensor size not functional:\n{x}\n{y}"


@pytest.mark.parametrize(
    'tensor_dict',
    [
        {
            1: torch.ones([2, 2]),
            2: LazyTensor(lambda: torch.ones([2, 2])),
            3: LazyTensor(lambda: torch.ones([2, 2]), torch.ones([2, 2])),
        },
    ],
)
def test_handle_lazy_tensor(tensor_dict: MutableMapping):
    handle_lazy_tensor(tensor_dict)
    for k in tensor_dict:
        assert isinstance(tensor_dict[k], torch.Tensor)
