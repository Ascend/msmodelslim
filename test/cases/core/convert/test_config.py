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

msmodelslim.core.convert.config 模块的单元测试
"""

import pytest

from msmodelslim.core.convert.config import ConvertConfig, ConvertRule, ModuleRule
from msmodelslim.core.convert.types import IRKind
from msmodelslim.utils.exception import SchemaValidateError


class TestConvertConfig:
    """测试 ConvertConfig 校验逻辑"""

    def test_convert_config_raise_error_when_mxfp8_with_non_ascend_dst(self):
        with pytest.raises(SchemaValidateError, match="W8A8_MXFP8 requires dst_format ascendv1"):
            ConvertConfig(
                model_path="/m",
                save_path="/o",
                dst_format="huggingface",
                convert_rules=[
                    ConvertRule(match="layers.*", target_ir=IRKind.W8A8_MXFP8),
                ],
            )

    def test_convert_config_pass_when_mxfp8_with_ascendv1_dst(self):
        cfg = ConvertConfig(
            model_path="/m",
            save_path="/o",
            dst_format="ascendv1",
            module_rules=[ModuleRule(match="layers.*", source_format="bf16")],
            convert_rules=[ConvertRule(match="layers.*", target_ir=IRKind.W8A8_MXFP8)],
        )
        assert cfg.dst_format == "ascendv1"

    def test_convert_config_pass_when_valid_module_and_convert_rules(self):
        cfg = ConvertConfig(
            model_path="/data/model",
            save_path="/data/out",
            module_rules=[
                ModuleRule(
                    match="layers.*.q_proj",
                    source_format="bf16",
                    tensor_map={"weight": "{module}.weight"},
                ),
            ],
            convert_rules=[ConvertRule(match="layers.*.q_proj", target_ir=IRKind.FLOAT)],
        )
        assert cfg.parallel.max_workers == 1  # 校验默认并行配置

    def test_convert_config_raise_error_when_save_path_empty(self):
        with pytest.raises(SchemaValidateError):
            ConvertConfig(model_path="/m", save_path="")

    def test_convert_config_raise_error_when_path_empty(self):
        with pytest.raises(SchemaValidateError):
            ConvertConfig(model_path="", save_path="/o")
