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

Unit tests for `multimodal_vlm_v1.quant_config`.

These tests focus on:
- Service config validation (especially `default_text`).
- Conversion from base quant config to the concrete multimodal VLM config.
- Robustness of YAML spec loading and error wrapping.
"""

from unittest.mock import Mock

import pytest

from msmodelslim.core.quant_service.multimodal_vlm_v1.quant_config import (
    MultimodalVLMServiceConfig,
    MultimodalVLMModelslimV1QuantConfig,
    load_specific_config,
)
from msmodelslim.core.quant_service.interface import BaseQuantConfig
from msmodelslim.utils.exception import SchemaValidateError


def test_multimodal_vlm_service_config_default_text_validation_success():
    """验证 `default_text` 为空格以外的正常字符串可以通过校验。"""
    cfg = MultimodalVLMServiceConfig.model_validate(
        {
            # 只关心 default_text 字段，其余字段沿用父类默认值
            "default_text": "Describe this image in detail.",
        }
    )
    assert cfg.default_text == "Describe this image in detail."


def test_multimodal_vlm_service_config_default_text_validation_fail():
    """验证 `default_text` 为空字符串时会触发 SchemaValidateError。"""
    with pytest.raises(SchemaValidateError):
        MultimodalVLMServiceConfig.model_validate(
            {
                "default_text": "",
            }
        )


def test_load_specific_config_success_with_dict_spec():
    """验证 `load_specific_config` 能够从 dict 规范正常生成配置。"""
    yaml_spec = {
        # 只需要提供当前文件中有自定义校验的字段即可
        "default_text": "valid prompt",
    }

    cfg = load_specific_config(yaml_spec)

    assert isinstance(cfg, MultimodalVLMServiceConfig)
    assert cfg.default_text == "valid prompt"


def test_load_specific_config_invalid_type_raises_value_error():
    """
    当传入的 yaml_spec 不是 dict 时，
    由于异常消息不包含 "validation error" 关键字，装饰器不会转换异常，
    直接抛出原始的 ValueError。
    """
    with pytest.raises(ValueError) as exc_info:
        load_specific_config(["not-a-dict"])  # type: ignore[arg-type]

    # 验证异常消息内容
    assert "task spec must be dict" in str(exc_info.value)


def test_multimodal_vlm_modelslim_v1_quant_config_from_base():
    """验证 `from_base` 能够正确地从 BaseQuantConfig 构造具体配置。"""
    # 构造一个最小化的 BaseQuantConfig mock，仅包含 `apiversion` 和 `spec` 字段
    base_cfg = Mock(spec=BaseQuantConfig)
    base_cfg.apiversion = "multimodal_vlm_modelslim_v1"
    base_cfg.spec = {
        "default_text": "prompt from base",
    }

    quant_cfg = MultimodalVLMModelslimV1QuantConfig.from_base(base_cfg)

    assert isinstance(quant_cfg, MultimodalVLMModelslimV1QuantConfig)
    assert quant_cfg.apiversion == "multimodal_vlm_modelslim_v1"
    # spec 应当被转换为 MultimodalVLMServiceConfig
    assert isinstance(quant_cfg.spec, MultimodalVLMServiceConfig)
    assert quant_cfg.spec.default_text == "prompt from base"
