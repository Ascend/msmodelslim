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

Unit tests for anti_outlier.common.fused_linear base-class and error contracts.
"""

import unittest

import torch
from torch import nn

from msmodelslim.processor.anti_outlier.common.fused_linear import (
    VirtualVModuleBase,
    VirtualVModuleFromKVFused,
    VirtualVModuleFromQKVFused,
)
from msmodelslim.utils.exception import SpecError, ToDoError


class TestVirtualVModuleBase(unittest.TestCase):
    """VirtualVModuleBase refuses forward and abstract update_weights."""

    def test_forward_raises_spec_error(self):
        with self.assertRaises(SpecError):
            VirtualVModuleBase().forward(torch.ones(2))

    def test_update_weights_raises_todo_error(self):
        with self.assertRaises(ToDoError):
            VirtualVModuleBase().update_weights()


class TestVirtualVModuleFromQKVFused(unittest.TestCase):
    """QKV-fused virtual module validates the head configuration."""

    def test_unknown_attention_type_raises_value_error(self):
        # 8 heads not divisible by 3 key/value heads -> UNKNOWN -> ValueError on index lookup.
        with self.assertRaises(ValueError):
            VirtualVModuleFromQKVFused(nn.Linear(512, 512), num_attention_heads=8, num_key_value_heads=3)


class TestVirtualVModuleFromKVFused(unittest.TestCase):
    """KV-fused virtual module validates the fused weight dimension."""

    def test_dimension_mismatch_raises_spec_error(self):
        kv_module = nn.Linear(64, 100, bias=False)
        # per-head out (100 // 2 = 50) != qk_nope_head_dim + v_head_dim (32 + 32).
        with self.assertRaises(SpecError):
            VirtualVModuleFromKVFused(
                kv_module,
                num_attention_heads=2,
                qk_nope_head_dim=32,
                v_head_dim=32,
            )


if __name__ == "__main__":
    unittest.main()
