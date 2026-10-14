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
# pylint: disable=consider-using-from-import

import unittest
from typing import Dict, Any

from typing import Tuple  # noqa: F401
import torch
import torch.nn as nn

from msmodelslim.model.common.model_wise_forward import (
    model_wise_forward_func,
    model_wise_visit_func,
)
from msmodelslim.core.base.protocol import ProcessRequest


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class TestModelWiseForward(unittest.TestCase):
    def setUp(self):
        self.model = DummyModel()
        torch.manual_seed(0)

    def test_forward_func_with_list_inputs(self):
        inputs = [torch.randn(2, 4)]
        gen = model_wise_forward_func(self.model, inputs)
        req = self._collect(gen)
        self.assertEqual(req.name, "")
        self.assertIs(req.module, self.model)
        self.assertIsInstance(req.args, list)
        self.assertEqual(len(req.args), 1)
        self.assertIs(req.args[0], inputs[0])
        self.assertEqual(req.kwargs, {})

    def test_forward_func_with_tuple_inputs(self):
        inputs = (torch.randn(2, 4),)
        gen = model_wise_forward_func(self.model, inputs)
        req = self._collect(gen)
        self.assertIs(req.args[0], inputs[0])

    def test_forward_func_with_dict_inputs(self):
        inputs: Dict[str, Any] = {"x": torch.randn(2, 4)}
        gen = model_wise_forward_func(self.model, inputs)
        req = self._collect(gen)
        self.assertEqual(req.args, [inputs])
        self.assertEqual(req.kwargs, inputs)

    def test_visit_func(self):
        gen = model_wise_visit_func(self.model)
        req = next(gen)
        self.assertEqual(req.name, "")
        self.assertIs(req.module, self.model)
        self.assertEqual(req.args, tuple())
        self.assertEqual(req.kwargs, {})

    def _collect(self, gen) -> ProcessRequest:
        req = next(gen)
        self.assertIsInstance(req, ProcessRequest)
        return req
