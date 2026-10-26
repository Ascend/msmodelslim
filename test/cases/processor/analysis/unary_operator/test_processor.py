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

Unit tests for unary_operator.processor helpers and DP support flag.
"""

import unittest
from unittest.mock import MagicMock, patch

from torch import nn

from msmodelslim.processor.analysis.unary_operator.processor import (
    UnaryAnalysisProcessor,
    UnaryAnalysisProcessorConfig,
    _accepts_block_name,
)


class TestAcceptsBlockName(unittest.TestCase):
    def test_true_when_setter_takes_datas_and_block_name(self):
        def setter(datas, name):
            return (datas, name)

        self.assertTrue(_accepts_block_name(setter))

    def test_false_when_setter_takes_only_datas(self):
        def setter(datas):
            return datas

        self.assertFalse(_accepts_block_name(setter))

    def test_false_for_single_parameter_builtin(self):
        self.assertFalse(_accepts_block_name(len))


class TestUnaryAnalysisProcessorSupportDistributed(unittest.TestCase):
    def test_support_distributed_is_always_true(self):
        config = UnaryAnalysisProcessorConfig(metrics="std")
        with patch(
            "msmodelslim.processor.analysis.unary_operator.processor.UnaryAnalysisMethodFactory.create_method",
            return_value=MagicMock(),
        ):
            processor = UnaryAnalysisProcessor(nn.Linear(2, 2), config)

        self.assertTrue(processor.support_distributed())


if __name__ == "__main__":
    unittest.main()
