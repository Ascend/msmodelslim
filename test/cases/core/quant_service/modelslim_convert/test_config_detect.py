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

msmodelslim.core.quant_service.modelslim_convert.config_detect 模块的单元测试
"""

from pathlib import Path

from msmodelslim.core.quant_service.modelslim_convert.config_detect import is_modelslim_convert_config


class TestIsModelslimConvertConfig:
    """测试 is_modelslim_convert_config 函数"""

    def test_is_modelslim_convert_config_return_true_when_apiversion_matches(self, tmp_path: Path):
        cfg = tmp_path / "convert.yaml"
        cfg.write_text("apiversion: modelslim_convert\nspec: {}\n", encoding="utf-8")
        assert is_modelslim_convert_config(cfg) is True

    def test_is_modelslim_convert_config_return_false_when_apiversion_other(self, tmp_path: Path):
        cfg = tmp_path / "quant.yaml"
        cfg.write_text("apiversion: modelslim_v1\nspec: {}\n", encoding="utf-8")
        assert is_modelslim_convert_config(cfg) is False
