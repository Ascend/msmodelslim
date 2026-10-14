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

import pytest

from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.quant_config.quant_config_factory import QuantConfigFactory
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.quant_config.quant_config_classes import BaseConfig
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.quant_config.quant_config_classes import WeightQuantConfig
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.quant_config.quant_config_classes import WeightActivationQuantConfig
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.quant_config.quant_config_classes import SparseQuantConfig
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.quant_config.quant_config_classes import KVQuantConfig


class TestQuantConfigFactory:
    def test_get_base_quant_config_should_return_base_config_given_base_description(self):
        description = 'base'
        config = QuantConfigFactory.get_quant_config(description)
        assert isinstance(config, BaseConfig)

    def test_get_weight_quant_config_should_return_base_config_given_weight_description(self):
        description = 'weight'
        config = QuantConfigFactory.get_quant_config(description, last_config=BaseConfig())
        assert isinstance(config, WeightQuantConfig)

    def test_get_weight_activation_quant_config_should_return_base_config_given_weight_activation_description(self):
        description = 'weight_activation'
        config = QuantConfigFactory.get_quant_config(description, last_config=BaseConfig())
        assert isinstance(config, WeightActivationQuantConfig)

    def test_get_sparse_quant_config_should_return_base_config_given_sparse_description(self):
        description = 'sparse'
        config = QuantConfigFactory.get_quant_config(description, last_config=BaseConfig())
        assert isinstance(config, SparseQuantConfig)

    def test_get_kv_quant_config_should_return_base_config_given_kv_description(self):
        description = 'kv'
        config = QuantConfigFactory.get_quant_config(description, last_config=BaseConfig())
        assert isinstance(config, KVQuantConfig)

    def test_get_base_quant_config_should_raise_error_given_error_description(self):
        description = 'xx'
        with pytest.raises(ValueError):
            config = QuantConfigFactory.get_quant_config(description)  # noqa: F841
