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

from msmodelslim.core.practice import PracticeConfig
from msmodelslim.core.quant_service.interface import BaseQuantConfig
from msmodelslim.core.quant_service.modelslim_v0.quant_config import (
    ModelslimV0QuantConfig,
    QuantSpec,
    load_specific_config,
)
from msmodelslim.utils.exception import SchemaValidateError


class TestModelslimV0QuantConfig:
    """Tests for ModelslimV0QuantConfig.

    看护范围限定在 modelslim_v0 实际生效的路径：YAML -> PracticeConfig.model_validate
    -> BaseQuantConfig 插件判别 -> ModelslimV0QuantConfig(spec=QuantSpec)。
    该路径下 spec 由 pydantic 依据 QuantSpec 类字段默认值构造，随后
    ModelslimV0QuantConfig.from_base 里 load_specific_config 走 isinstance 早返回，
    因此"默认值契约"落在 QuantSpec 类字段上，而不在 load_specific_config 的 dict 分支。
    """

    def test_quant_spec_defaults_cover_all_fields(self):
        """QuantSpec 全部字段默认值看护。

        默认值即 modelslim_v0 的规格契约：未显式书写的字段必须稳定落到既定默认值上——
        anti_cfg/anti_dataset 默认 None（anti-outlier 不开启），anti_params/calib_cfg/
        calib_params/calib_save_params 默认空 dict，batch_size 默认 4，calib_dataset 默认
        teacher_qualification.jsonl。任一项被改动，都会让未书写该字段的 v0 配置行为
        静默漂移（如 anti-outlier 默认开启、校准数据集变化）。
        """
        spec = QuantSpec()
        assert spec.anti_cfg is None
        assert spec.anti_params == {}
        assert spec.calib_cfg == {}
        assert spec.calib_params == {}
        assert spec.calib_save_params == {}
        assert spec.batch_size == 4
        assert spec.anti_dataset is None
        assert spec.calib_dataset == "teacher_qualification.jsonl"

    def test_practice_config_validate_uses_all_defaults_when_spec_empty(self):
        """场景：YAML 只书写 apiversion，spec 为空（生产入口 PracticeConfig.model_validate）。
        预期：spec 被判别构造为 QuantSpec 并落到全字段默认值，anti-outlier 默认不开启。
        """
        practice = PracticeConfig.model_validate({"apiversion": "modelslim_v0", "spec": {}})
        spec = practice.extract_quant_config().spec
        assert isinstance(spec, QuantSpec)
        assert spec == QuantSpec()
        # 回归关键字段在真实入口上单独断言，避免只靠与 QuantSpec() 相等而自证
        assert spec.anti_cfg is None
        assert spec.calib_dataset == "teacher_qualification.jsonl"
        assert spec.batch_size == 4

    def test_practice_config_validate_keeps_explicit_values_when_spec_partial(self):
        """场景：spec 只书写部分字段。
        预期：已书写字段生效，未书写字段落到 QuantSpec 类默认值，二者互不污染。
        """
        practice = PracticeConfig.model_validate(
            {
                "apiversion": "modelslim_v0",
                "metadata": {"config_id": "ut-v0-spec-partial"},
                "spec": {"calib_cfg": {"w_bit": 8, "a_bit": 8}, "calib_dataset": "boolq.jsonl"},
            }
        )
        spec = practice.extract_quant_config().spec
        assert spec.calib_cfg == {"w_bit": 8, "a_bit": 8}
        assert spec.calib_dataset == "boolq.jsonl"
        assert spec.anti_cfg is None
        assert spec.anti_params == {}
        assert spec.calib_params == {}
        assert spec.calib_save_params == {}
        assert spec.batch_size == 4
        assert spec.anti_dataset is None

    def test_from_base_returns_same_spec_when_already_typed(self):
        """场景：spec 已是 QuantSpec（生产路径下 pydantic 已构造完成）。
        预期：load_specific_config 走 isinstance 早返回，from_base 原样透传同一实例，
        不会再由 dict 分支重新填默认值。
        """
        spec = QuantSpec(batch_size=2)
        assert load_specific_config(spec) is spec
        base = BaseQuantConfig.model_validate({"apiversion": "modelslim_v0", "spec": spec})
        assert ModelslimV0QuantConfig.from_base(base).spec is spec

    def test_quant_spec_rejects_apiversion_other_than_modelslim_v0(self):
        """场景：apiversion 非 modelslim_v0。
        预期：SchemaValidateError（Literal 判别字段锁定 v0 插件归属）。
        """
        with pytest.raises(SchemaValidateError):
            ModelslimV0QuantConfig(apiversion="modelslim_v1", spec={})
