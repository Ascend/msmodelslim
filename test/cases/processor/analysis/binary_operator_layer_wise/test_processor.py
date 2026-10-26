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

Unit tests for BinaryOperatorLayerWiseProcessor DP flag and restore failure.
"""

import unittest
from unittest.mock import MagicMock, patch

from torch import nn

from msmodelslim.core.base.protocol import BatchProcessRequest
from msmodelslim.processor.analysis.binary_operator_layer_wise.processor import (
    BinaryOperatorLayerWiseProcessor,
    BinaryOperatorLayerWiseProcessorConfig,
)
from msmodelslim.utils.exception import UnexpectedError


class TestBinaryOperatorLayerWiseProcessorExtra(unittest.TestCase):
    @staticmethod
    def _fake_method():
        fake = MagicMock()
        fake.name = "mse_layer_wise"
        return fake

    def _build_processor(self, fake_method):
        config = BinaryOperatorLayerWiseProcessorConfig(metrics="mse_layer_wise", quant_modules=["*"], configs=[])
        with patch(
            "msmodelslim.processor.analysis.binary_operator_layer_wise.processor.LayerWiseMethodFactory.create_method",
            return_value=fake_method,
        ):
            return BinaryOperatorLayerWiseProcessor(nn.Linear(2, 2), config)

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

    def test_process_raises_unexpected_error_when_float_state_restore_fails(self):
        processor = self._build_processor(self._fake_method())
        model = nn.Linear(2, 2)
        request = BatchProcessRequest(name="block.layers.0", module=model, datas=[])

        with patch.object(model, "load_state_dict", side_effect=RuntimeError("restore boom")):
            with self.assertRaises(UnexpectedError) as ctx:
                processor.process(request)

        self.assertIn("restore float state_dict", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
