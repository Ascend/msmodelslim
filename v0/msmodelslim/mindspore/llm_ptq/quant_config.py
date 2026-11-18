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

from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.quant_config.quant_config import QuantConfig as BaseQuantConfig


class QuantConfig(object):
    def __init__(self, 
                 disable_names: object = None,
                 fraction: float = 0.01):
        base_quant_cfg = BaseQuantConfig(
            disable_names=disable_names,
            fraction=fraction
        )
        config_attribute = base_quant_cfg.__dict__
        self.__dict__.update(**config_attribute)
