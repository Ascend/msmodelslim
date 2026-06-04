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

import torch

from msmodelslim.core.quantizer.base import QConfig
from msmodelslim.core.quantizer.impl.none import ActPerTensorNone
from msmodelslim.ir.qal.qbase import QDType, QScope


class TestActPerTensorNone:
    """Tests for ActPerTensorNone."""

    def _make_quantizer(self) -> ActPerTensorNone:
        config = QConfig(
            dtype=QDType.FLOAT,
            scope=QScope.PER_TENSOR,
            symmetric=True,
            method="none",
        )
        return ActPerTensorNone(config)

    def test_forward_returns_input_unchanged_when_called(self):
        """场景：forward 输入张量。预期：原样返回。"""
        q = self._make_quantizer()
        x = torch.randn(3, 4)
        out = q.forward(x)
        assert torch.equal(out, x)

    def test_is_data_free_returns_true_when_queried(self):
        """场景：查询 is_data_free。预期：True。"""
        assert self._make_quantizer().is_data_free() is True

    def test_get_q_param_reflects_config_when_called(self):
        """场景：get_q_param。预期：scheme 与 config 一致。"""
        q = self._make_quantizer()
        q_param = q.get_q_param()
        assert q_param.scheme.dtype == QDType.FLOAT
        assert q_param.scheme.scope == QScope.PER_TENSOR
        assert q_param.scheme.symmetric is True
