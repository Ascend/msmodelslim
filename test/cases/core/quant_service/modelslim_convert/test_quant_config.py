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

msmodelslim.core.quant_service.modelslim_convert.quant_config 模块的单元测试
"""

from msmodelslim.core.quant_service.interface import BaseQuantConfig
from msmodelslim.core.quant_service.modelslim_convert.config_mapper import ModelslimConvertServiceConfig
from msmodelslim.core.quant_service.modelslim_convert.quant_config import ModelslimConvertQuantConfig


class TestModelslimConvertQuantConfig:
    """测试 ModelslimConvertQuantConfig 类"""

    def test_from_base_return_typed_config_when_base_given(self):
        base = BaseQuantConfig.model_validate(
            {
                "apiversion": "modelslim_convert",
                "spec": {"model_path": "/m", "save_path": "/o", "linears": []},
            }
        )
        cfg = ModelslimConvertQuantConfig.from_base(base)
        assert cfg.apiversion == "modelslim_convert"  # 校验 apiversion 保留
        assert isinstance(cfg.spec, ModelslimConvertServiceConfig)  # 校验 spec 类型化
        assert cfg.spec.model_path == "/m"
