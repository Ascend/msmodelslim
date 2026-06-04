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

from msmodelslim.core.quant_service.interface import BaseQuantConfig
from msmodelslim.core.quant_service.modelslim_v0.quant_config import (
    ModelslimV0QuantConfig,
    QuantSpec,
    load_specific_config,
)
from msmodelslim.utils.exception import SchemaValidateError


class TestModelslimV0QuantConfig:
    """Tests for ModelslimV0QuantConfig."""

    def test_from_base_parses_spec_when_dict_provided(self):
        """场景：BaseQuantConfig.spec 为 dict。预期：from_base 填充 QuantSpec 字段。"""
        base = BaseQuantConfig(
            apiversion="modelslim_v0",
            spec={
                "batch_size": 8,
                "calib_dataset": "my_calib.jsonl",
                "anti_dataset": "anti.jsonl",
            },
        )
        cfg = ModelslimV0QuantConfig.from_base(base)
        assert cfg.spec.batch_size == 8
        assert cfg.spec.calib_dataset == "my_calib.jsonl"
        assert cfg.spec.anti_dataset == "anti.jsonl"

    def test_load_specific_config_returns_quant_spec_when_already_typed(self):
        """场景：spec 已是 QuantSpec。预期：原样返回。"""
        spec = QuantSpec(batch_size=2)
        assert load_specific_config(spec) is spec

    def test_load_specific_config_raises_schema_validate_error_when_not_dict(self):
        """场景：spec 类型非法。预期：SchemaValidateError。"""
        with pytest.raises(SchemaValidateError, match="task spec must be dict"):
            load_specific_config(123)

    def test_load_specific_config_uses_defaults_when_dict_empty(self):
        """场景：空 dict spec。预期：默认 batch_size 与 calib_dataset。"""
        spec = load_specific_config({})
        assert spec.batch_size == 4
        assert spec.calib_dataset == "teacher_qualification.jsonl"
