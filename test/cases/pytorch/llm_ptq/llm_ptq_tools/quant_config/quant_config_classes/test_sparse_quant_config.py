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

import pytest  # noqa: F401
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.quant_config.quant_config_classes import SparseQuantConfig
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.quant_config.quant_config_classes import BaseConfig
from ..old_quant_config import OldQuantConfig
from ..compare_old_and_new_config import compare_config_parameters


class TestSparseQuantConfig:
    def test_sparse_quant_config_should_equal_to_old_quant_config_when_default_value(self):
        base_config = BaseConfig()
        sparse_quant_config = SparseQuantConfig(base_config)
        old_quant_config = OldQuantConfig(co_sparse=True)
        assert compare_config_parameters(sparse_quant_config, old_quant_config) is True
