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

Unit tests for msmodelslim.processor.anti_outlier.default.model_structure_retriever.
Covers: is_layernorm, collect_shared_input_modules.
"""
# pylint: disable=consider-using-from-import

import unittest

from collections import defaultdict  # noqa: F401
import torch
import torch.nn as nn

from msmodelslim.core.base.protocol import BatchProcessRequest
from msmodelslim.processor.anti_outlier.default.model_structure_retriever import (
    is_layernorm,
    collect_shared_input_modules,
)


class TestIsLayernorm(unittest.TestCase):
    """Tests for is_layernorm."""

    def test_is_layernorm_returns_true_when_module_is_layer_norm(self):
        module = nn.LayerNorm(10)
        self.assertTrue(is_layernorm(module))

    def test_is_layernorm_returns_true_when_module_is_rms_norm(self):
        # Use a class that has RMSNorm in the name (e.g. from msmodelslim or a minimal one)
        class RMSNorm(nn.Module):
            def __init__(self, hidden_size):
                super().__init__()
                self.weight = nn.Parameter(torch.ones(hidden_size))

            def forward(self, x):
                return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * self.weight

        module = RMSNorm(10)
        self.assertTrue(is_layernorm(module))

    def test_is_layernorm_returns_false_when_module_is_linear(self):
        module = nn.Linear(10, 20)
        self.assertFalse(is_layernorm(module))

    def test_is_layernorm_returns_false_when_module_is_sequential(self):
        module = nn.Sequential(nn.Linear(10, 10), nn.LayerNorm(10))
        self.assertFalse(is_layernorm(module))


class _NormLinearBlock(nn.Module):
    """Block with one norm and one linear sharing the same flow (norm -> linear)."""

    def __init__(self, dim=4):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.linear = nn.Linear(dim, dim)

    def forward(self, x):
        x = self.norm(x)
        x = self.linear(x)
        return x


class _TwoLinearBlock(nn.Module):
    """Block with norm then two linears (same input to linears in practice from one norm output)."""

    def __init__(self, dim=4):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.linear1 = nn.Linear(dim, dim)
        self.linear2 = nn.Linear(dim, dim)

    def forward(self, x):
        h = self.norm(x)
        y1 = self.linear1(h)
        y2 = self.linear2(h)
        return y1 + y2


class TestCollectSharedInputModules(unittest.TestCase):
    """Tests for collect_shared_input_modules."""

    def test_collect_shared_input_modules_returns_empty_when_no_linear_or_layernorm(self):
        model = nn.Sequential(nn.ReLU())
        # 构造没有 Linear/LayerNorm 的 BatchProcessRequest
        request = BatchProcessRequest(
            name="no_linear_norm",
            module=model,
            datas=[((torch.ones(1, 4),), {})],
            outputs=None,
        )
        input_to_linear, output_to_layernorm = collect_shared_input_modules(request)
        self.assertEqual(len(input_to_linear), 0)
        self.assertEqual(len(output_to_layernorm), 0)

    def test_collect_shared_input_modules_collects_norm_and_linear_when_norm_linear_block(self):
        model = _NormLinearBlock(dim=4)
        request = BatchProcessRequest(
            name="norm_linear_block",
            module=model,
            datas=[((torch.ones(2, 4),), {})],
            outputs=None,
        )
        input_to_linear, output_to_layernorm = collect_shared_input_modules(request)
        # One linear (model.1 or similar), so one entry in input_to_linear
        self.assertGreaterEqual(len(input_to_linear), 1)
        self.assertGreaterEqual(len(output_to_layernorm), 1)

    def test_collect_shared_input_modules_assigns_name_when_traversing_modules(self):
        model = _NormLinearBlock(dim=4)
        request = BatchProcessRequest(
            name="norm_linear_block",
            module=model,
            datas=[((torch.ones(2, 4),), {})],
            outputs=None,
        )
        collect_shared_input_modules(request)
        for name, module in model.named_modules():
            self.assertTrue(hasattr(module, "name"), f"module {name} should have .name")
            self.assertEqual(module.name, name)

    def test_collect_shared_input_modules_runs_model_forward_when_datas_provided(self):
        class _RecordForwardLinear(nn.Linear):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.called = False

            def forward(self, x):
                self.called = True
                return super().forward(x)

        model = _RecordForwardLinear(4, 4)
        request = BatchProcessRequest(
            name="record_forward",
            module=model,
            datas=[((torch.ones(1, 4),), {})],
            outputs=None,
        )
        collect_shared_input_modules(request)
        self.assertTrue(model.called)

    def test_collect_shared_input_modules_removes_hooks_when_forward_finished(self):
        model = _NormLinearBlock(dim=4)
        request = BatchProcessRequest(
            name="norm_linear_block",
            module=model,
            datas=[((torch.ones(2, 4),), {})],
            outputs=None,
        )
        collect_shared_input_modules(request)
        for _, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.LayerNorm)):
                self.assertEqual(len(module._forward_hooks), 0)

    def test_collect_shared_input_modules_removes_handles_and_reraises_when_dummy_forward_raises(self):
        class _FailForwardLinear(nn.Linear):
            def forward(self, x):
                raise RuntimeError("dummy failure")

        model = _FailForwardLinear(4, 4)
        request = BatchProcessRequest(
            name="fail_forward",
            module=model,
            datas=[((torch.ones(1, 4),), {})],
            outputs=None,
        )
        with self.assertRaises(RuntimeError):
            collect_shared_input_modules(request)
        # 确保即使 forward 抛异常也会移除 hooks
        for _, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.LayerNorm)):
                self.assertEqual(len(module._forward_hooks), 0)
