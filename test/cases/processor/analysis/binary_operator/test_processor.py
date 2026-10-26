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

Unit tests for analysis.binary_operator.processor covering the DP flag
and the empty-target-layer branch not exercised by the aggregate suite.
"""

import unittest
from unittest.mock import MagicMock, patch

from torch import nn

from msmodelslim.core.base.protocol import BatchProcessRequest
from msmodelslim.processor.analysis.binary_operator.processor import (
    BinaryAnalysisProcessor,
    BinaryAnalysisProcessorConfig,
)


class _TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = nn.Identity()
        self.linear = nn.Linear(4, 4)

    def forward(self, x):
        return self.linear(self.attn(x))


class TestBinaryAnalysisProcessorExtra(unittest.TestCase):
    """补充 BinaryAnalysisProcessor 中聚合用例未覆盖的行为。"""

    @staticmethod
    def _fake_method():
        fake = MagicMock()
        fake.name = "mse"
        fake.get_hook.return_value = lambda module, inputs, output, layer_name, stats: None
        return fake

    def _build_processor(self, fake_method):
        config = BinaryAnalysisProcessorConfig(metrics="mse", patterns=["*"], configs=[])
        with patch(
            "msmodelslim.processor.analysis.binary_operator.processor.BinaryAnalysisMethodFactory.create_method",
            return_value=fake_method,
        ):
            return BinaryAnalysisProcessor(_TinyModel(), config)

    def test_support_distributed_requires_all_quant_processors(self):
        processor = self._build_processor(self._fake_method())
        qp_true = MagicMock()
        qp_true.support_distributed.return_value = True
        qp_false = MagicMock()
        qp_false.support_distributed.return_value = False

        processor.quant_processors = [qp_true, qp_true]
        self.assertTrue(processor.support_distributed())

        processor.quant_processors = [qp_true, qp_false]
        self.assertFalse(processor.support_distributed())

    def test_preprocess_still_runs_float_forward_when_no_target_layer_matched(self):
        fake_method = self._fake_method()
        fake_method.get_target_layers.return_value = ["block.linear"]
        fake_method.filter_layers_by_patterns.return_value = []
        processor = self._build_processor(fake_method)
        request = BatchProcessRequest(name="block", module=_TinyModel(), datas=[])

        with patch(
            "msmodelslim.processor.analysis.binary_operator.processor.AutoSessionProcessor.process"
        ) as mock_process:
            processor.preprocess(request)

        mock_process.assert_called_once_with(request)
        self.assertEqual(processor._target_layers, [])
        self.assertEqual(processor._hook_handles, {})


if __name__ == "__main__":
    unittest.main()
