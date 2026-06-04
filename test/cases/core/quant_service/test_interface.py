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

from msmodelslim.core.quant_service.interface import BaseQuantConfig, QuantServiceConfig
from msmodelslim.utils.exception import SchemaValidateError


class TestBaseQuantConfig:
    """Tests for BaseQuantConfig."""

    def test_defaults_when_minimal_constructed(self):
        """场景：最小构造 BaseQuantConfig。预期：apiversion 与空 spec。"""
        cfg = BaseQuantConfig(apiversion="test_v1")
        assert cfg.apiversion == "test_v1"
        assert cfg.spec == {}

    def test_allows_extra_fields_when_provided(self):
        """场景：传入额外字段。预期：extra allow 保留字段。"""
        cfg = BaseQuantConfig(apiversion="v1", custom_field=42)
        assert cfg.custom_field == 42


class TestQuantServiceConfig:
    """Tests for QuantServiceConfig plugin config."""

    def test_requires_apiversion_when_instantiated_directly(self):
        """场景：直接实例化 QuantServiceConfig。预期：缺少 apiversion 时校验失败。"""
        with pytest.raises(SchemaValidateError):
            QuantServiceConfig()
