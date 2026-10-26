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

Unit tests for AttentionMSEAnalysisMethod numeric scoring and hook collection.
"""

import unittest

import torch
from torch import nn

from msmodelslim.processor.analysis.binary_operator.metrics.attention_mse.impl import (
    AttentionMSEAnalysisMethod,
)
from msmodelslim.processor.analysis.binary_operator.metrics.attention_mse.interface import (
    AttentionMSEAnalysisInterface,
)
from msmodelslim.utils.exception import UnexpectedError, UnsupportedError


class _Adapter(AttentionMSEAnalysisInterface):
    def get_attention_module_cls(self):
        return "_TinyAttention"

    def get_attention_output_extractor(self):
        return lambda output: output[0] if isinstance(output, tuple) else output


class _TinyAttention(nn.Module):
    def forward(self, x):
        return (x, x)


class TestAttentionMSEAnalysisMethod(unittest.TestCase):
    def test_init_rejects_adapter_without_interface(self):
        with self.assertRaises(UnsupportedError) as ctx:
            AttentionMSEAnalysisMethod(object())
        self.assertIn("object", str(ctx.exception))

    def test_compute_score_averages_elementwise_mse(self):
        method = AttentionMSEAnalysisMethod(_Adapter())
        before = {"attn_output": [torch.zeros(1, 2), torch.ones(1, 2)]}
        after = {"attn_output": [torch.ones(1, 2), torch.full((1, 2), 2.0)]}
        self.assertAlmostEqual(method.compute_score(before, after), 1.0, places=6)

    def test_compute_score_raises_when_activation_missing(self):
        method = AttentionMSEAnalysisMethod(_Adapter())
        with self.assertRaises(UnexpectedError):
            method.compute_score({"attn_output": []}, {"attn_output": [torch.zeros(2, 2)]})

    def test_hook_rows_and_matches_attention_class(self):
        method = AttentionMSEAnalysisMethod(_Adapter())
        stats = {}
        module = _TinyAttention()
        method.get_hook()(module, None, (torch.arange(8.0).reshape(2, 4), None), "block.attn", stats)
        collected = stats["block.attn"]["attn_output"]
        self.assertEqual(len(collected), 1)
        self.assertEqual(tuple(collected[0].shape), (2, 4))
        self.assertTrue(torch.equal(collected[0], torch.arange(8.0).reshape(2, 4)))
        self.assertTrue(method._matches(module))
        self.assertFalse(method._matches(nn.Linear(2, 2)))


if __name__ == "__main__":
    unittest.main()
