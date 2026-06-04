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

from msmodelslim.core.const import RunnerType
from msmodelslim.core.quant_service.interface import BaseQuantConfig
from msmodelslim.core.quant_service.modelslim_v1.quant_config import (
    ModelslimV1QuantConfig,
    ModelslimV1ServiceConfig,
    load_specific_config,
)


class TestModelslimV1QuantConfig:
    """Tests for ModelslimV1QuantConfig."""

    def test_from_base_converts_dict_spec_when_base_provided(self):
        """场景：BaseQuantConfig.spec 为 dict。预期：from_base 得到 ModelslimV1ServiceConfig。"""
        base = BaseQuantConfig(
            apiversion="modelslim_v1",
            spec={"runner": RunnerType.LAYER_WISE, "dataset": "custom.jsonl"},
        )
        cfg = ModelslimV1QuantConfig.from_base(base)
        assert isinstance(cfg, ModelslimV1QuantConfig)
        assert isinstance(cfg.spec, ModelslimV1ServiceConfig)
        assert cfg.spec.runner == RunnerType.LAYER_WISE
        assert cfg.spec.dataset == "custom.jsonl"

    def test_load_specific_config_returns_same_instance_when_already_typed(self):
        """场景：spec 已是 ModelslimV1ServiceConfig。预期：原样返回。"""
        typed = ModelslimV1ServiceConfig()
        assert load_specific_config(typed) is typed

    def test_load_specific_config_raises_value_error_when_not_dict(self):
        """场景：spec 非 dict 且非 ModelslimV1ServiceConfig。预期：ValueError。"""
        with pytest.raises(ValueError, match="task spec must be dict"):
            load_specific_config("invalid")
