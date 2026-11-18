#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
-------------------------------------------------------------------------
This file is part of the MindStudio project.
Copyright (c) 2025 Huawei Technologies Co.,Ltd.

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


from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.quant_config.quant_config_classes import BaseConfig
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.quant_config.quant_config_classes import WeightQuantConfig
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.quant_config.quant_config_classes import WeightActivationQuantConfig
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.quant_config.quant_config_classes import SparseQuantConfig
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.quant_config.quant_config_classes import KVQuantConfig
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.quant_config.quant_config_classes import FAQuantConfig
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.quant_config.quant_config_classes import SimulateTPConfig
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.quant_config.quant_config_classes import TimestepQuantConfig


class QuantConfigFactory:
    """
    工厂模式，返回不同的量化Config类
    """
    QuantConfigMapper = {
        'base': BaseConfig,
        'weight': WeightQuantConfig,
        'weight_activation': WeightActivationQuantConfig,
        'sparse': SparseQuantConfig,
        'kv': KVQuantConfig,
        'fa_quant': FAQuantConfig,
        'simulate_tp': SimulateTPConfig,
        'timestep_quant': TimestepQuantConfig,
    }

    @classmethod
    def get_quant_config(cls, description: str, **kwargs) -> BaseConfig:
        if description in cls.QuantConfigMapper:
            return cls.QuantConfigMapper[description](**kwargs)
        raise ValueError(f"QuantConfig {description} does not support, please check your QuantConfig.")
