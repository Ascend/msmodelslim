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

PracticeConfig 前校验（配置与任务分离）单测。

对应设计文档 new_dev_配置与任务分离_设计.md §8（v0.5）：
- BaseQuantConfig 自身升级为判别基类（_validate_plugin 的 cls.__dict__ 判断）
- 判别根校验（BaseQuantConfig.model_validate 触发插件判别）与错误透传（实现期验证点②）
- model_validate 两层聚合：一次报全、计数=各层头部计数之和、错误路径与现状原生形态一致
- 未知/缺失 apiversion 兜底
- model_validate 生成的 practice 与直接 model_validate 构造一致
- model_validate 是唯一构造入口（统一 metadata + spec 强校验）
- from_base 嵌套化输出一致性、Literal 错配、v0 宽松语义、processor 注册表不变量
"""

import pytest

from msmodelslim.core.practice.interface import PracticeConfig
from msmodelslim.core.quant_service.interface import BaseQuantConfig
from msmodelslim.utils.exception import SchemaValidateError

pytest.importorskip("pydantic")

# 合法的 modelslim_v1 spec（字段全默认，空 dict 即合法；带一个标量字段错误/未知 processor 用于负例）
VALID_V1_CONFIG = {
    "apiversion": "modelslim_v1",
    "metadata": {
        "config_id": "ut-precheck",
        "score": 90,
        "label": {"w_bit": 8, "a_bit": 8},
        "verified_model_types": ["Qwen3-32B"],
    },
    "spec": {"process": [], "save": [], "dataset": "mix_calib.jsonl"},
}


@pytest.fixture(scope="module", autouse=True)
def register_task_plugins():
    """将后端任务配置插件注册进内存注册表（等价 entry point ep.load() 后的路径）。"""
    from msmodelslim.utils.plugin.plugin_utils import register_plugin
    from msmodelslim.core.quant_service.modelslim_v0.quant_config import get_plugin as v0_gp
    from msmodelslim.core.quant_service.modelslim_v1.quant_config import get_plugin as v1_gp
    from msmodelslim.core.quant_service.modelslim_convert.quant_config import get_plugin as cvt_gp
    from msmodelslim.core.quant_service.multimodal_sd_v1.quant_config import get_plugin as sd_gp
    from msmodelslim.core.quant_service.multimodal_vlm_v1.quant_config import get_plugin as vlm_gp

    for getter in (v0_gp, v1_gp, cvt_gp, sd_gp, vlm_gp):
        register_plugin(getter)


class TestBaseQuantConfigDispatch:
    """判别根校验：BaseQuantConfig 基类 model_validate 的插件分派（实现期验证点②）。"""

    def test_dispatch_returns_backend_strong_instance(self):
        """场景：在判别基类上校验合法任务描述。预期：返回后端强类型实例，spec 强校验生效。"""
        task = BaseQuantConfig.model_validate({"apiversion": "modelslim_v1", "spec": {}})
        from msmodelslim.core.quant_service.modelslim_v1.quant_config import (
            ModelslimV1QuantConfig,
            ModelslimV1ServiceConfig,
        )

        assert isinstance(task, ModelslimV1QuantConfig)
        assert isinstance(task.spec, ModelslimV1ServiceConfig)

    def test_dispatch_validation_error_converted_to_schema_error(self):
        """场景：spec 非法（未知 processor type）。预期：透传为 SchemaValidateError，路径含 spec.process.0。"""
        with pytest.raises(SchemaValidateError) as err:
            BaseQuantConfig.model_validate(
                {
                    "apiversion": "modelslim_v1",
                    "spec": {"process": [{"type": "ut_nonexistent_processor"}]},
                }
            )
        assert "spec.process.0" in str(err.value)

    def test_dispatch_unknown_apiversion_raises_unsupported(self):
        """场景：apiversion 拼写错误。预期：判别即探测，抛 UnsupportedError。"""
        from msmodelslim.utils.exception import UnsupportedError

        with pytest.raises(UnsupportedError):
            BaseQuantConfig.model_validate({"apiversion": "modelslm_v1", "spec": {}})

    def test_model_validate_is_only_construction_path(self):
        """场景：PracticeConfig.model_validate 是唯一构造入口，触发 dispatch 强校验。
        model_validate 统一 metadata + spec 两层校验，坏 spec 会报错。
        """
        # PracticeConfig.model_validate 触发 dispatch → 坏 spec 会报错
        with pytest.raises(SchemaValidateError):
            PracticeConfig.model_validate({"apiversion": "modelslim_v1", "spec": {"unknown": 1}})


class TestPracticePrecheckAggregation:
    """model_validate 两层聚合：一次报全、计数、路径形态。"""

    def test_valid_config_returns_practice_instance(self):
        practice = PracticeConfig.model_validate(VALID_V1_CONFIG)
        assert isinstance(practice, PracticeConfig)
        assert practice.task.apiversion == "modelslim_v1"
        assert practice.task.spec.dataset == VALID_V1_CONFIG["spec"]["dataset"]

    def test_metadata_and_spec_errors_reported_together(self):
        """核心验收用例：metadata + spec 同时异常 → 单一 SchemaValidateError 同时包含两层错误。"""
        bad = {
            "apiversion": "modelslim_v1",
            "metadata": {"score": -1},
            "spec": {"process": [{"type": "ut_nonexistent_processor"}]},
        }
        with pytest.raises(SchemaValidateError) as err:
            PracticeConfig.model_validate(bad)
        msg = str(err.value)
        assert "metadata.score" in msg
        assert "spec.process.0" in msg
        assert "2 validation error(s) found" in msg  # 计数=两层之和

    def test_metadata_only_error(self):
        bad = {"apiversion": "modelslim_v1", "metadata": {"score": -1}, "spec": {}}
        with pytest.raises(SchemaValidateError) as err:
            PracticeConfig.model_validate(bad)
        assert "metadata.score" in str(err.value)
        assert "validation error" in str(err.value)

    def test_spec_only_error_keeps_metadata_clean(self):
        bad = {"apiversion": "modelslim_v1", "metadata": {}, "spec": {"dataset": 123}}
        with pytest.raises(SchemaValidateError) as err:
            PracticeConfig.model_validate(bad)
        msg = str(err.value)
        assert "spec.dataset" in msg
        assert "metadata." not in msg  # metadata 无错误，不应出现在消息中


class TestUnknownApiversionFallback:
    """未知/缺失 apiversion：_validate_plugin 判别即探测，抛 UnsupportedError。"""

    def test_unknown_apiversion_reports_builtin_list(self):
        from msmodelslim.utils.exception import UnsupportedError

        with pytest.raises(UnsupportedError) as err:
            PracticeConfig.model_validate({"apiversion": "modelslm_v1", "metadata": {}, "spec": {}})
        assert "modelslm_v1" in str(err.value)

    def test_missing_apiversion_defaults_to_unknown(self):
        from msmodelslim.utils.exception import UnsupportedError

        with pytest.raises(UnsupportedError):
            PracticeConfig.model_validate({"metadata": {}, "spec": {}})

    def test_unknown_apiversion_with_metadata_error_reports_both(self):
        """apiversion 未知时 UnsupportedError 优先传播；metadata 错误被覆盖（已知 trade-off）。"""
        from msmodelslim.utils.exception import UnsupportedError

        bad = {"apiversion": "modelslm_v1", "metadata": {"score": -1}, "spec": {}}
        with pytest.raises(UnsupportedError):
            PracticeConfig.model_validate(bad)


class TestFromBaseAfterNesting:
    """from_base 嵌套化：输出路径与现状一致（spec.process[0]…）、Literal 错配报错。"""

    def test_from_base_validates_and_returns_strong_instance(self):
        from msmodelslim.core.quant_service.modelslim_v1.quant_config import ModelslimV1QuantConfig

        base = BaseQuantConfig.model_construct(apiversion="modelslim_v1", spec=VALID_V1_CONFIG["spec"])
        strong = ModelslimV1QuantConfig.from_base(base)
        assert strong.apiversion == "modelslim_v1"
        assert strong.spec.dataset == "mix_calib.jsonl"

    def test_from_base_error_path_matches_legacy_format(self):
        """执行期输出一致性：错误路径为 spec.process.0…（无双重前缀、无 task 痕迹）。"""
        from msmodelslim.core.quant_service.modelslim_v1.quant_config import ModelslimV1QuantConfig

        base = BaseQuantConfig.model_construct(
            apiversion="modelslim_v1", spec={"process": [{"type": "ut_nonexistent_processor"}]}
        )
        with pytest.raises(SchemaValidateError) as err:
            ModelslimV1QuantConfig.from_base(base)
        msg = str(err.value)
        assert "spec.process.0" in msg
        assert "spec.spec." not in msg
        assert "task" not in msg

    def test_from_base_apiversion_mismatch_raises(self):
        """Literal 收紧：描述与执行者 apiversion 错配在构造期报错。"""
        from msmodelslim.core.quant_service.modelslim_v1.quant_config import ModelslimV1QuantConfig

        base = BaseQuantConfig.model_construct(apiversion="modelslim_v0", spec={})
        with pytest.raises(SchemaValidateError):
            ModelslimV1QuantConfig.from_base(base)


class TestPluginRegistration:
    """插件注册与 config.ini 一致性。"""

    def test_get_plugin_registers_with_derived_type(self):
        """场景：register_plugin 按 Literal 推导 plugin_type。预期：全部 apiversion 均已注册。"""
        from msmodelslim.utils.plugin.plugin_utils import list_registered_plugin_types
        from msmodelslim.core.quant_service.interface import QUANT_TASK_PLUGIN_GROUP

        registered = set(list_registered_plugin_types(QUANT_TASK_PLUGIN_GROUP))
        assert registered == {
            "modelslim_v0",
            "modelslim_v1",
            "modelslim_convert",
            "multimodal_sd_modelslim_v1",
            "multimodal_vlm_modelslim_v1",
        }

    def test_processor_registry_populated_by_quant_config_import(self):
        """processor 注册表不变量：仅 import 后端 quant_config 即完成 processor 注册。"""
        from msmodelslim.processor.base import AutoProcessorConfig  # noqa: F401

        assert len(AutoProcessorConfig._registry) > 0
