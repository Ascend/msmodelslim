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

配置文档目录、路径、分类和已知事实源注册表。
"""

from __future__ import annotations


MANUAL_KEEP = {"processor_group.md", "auto_precision_tuning.md"}

SLUG_OVERRIDES = {
    "ModelslimV1QuantConfig": "modelslim_v1",
    "ModelslimV1ServiceConfig": "modelslim_v1_spec",
    "MultimodalVLMModelslimV1QuantConfig": "multimodal_vlm_modelslim_v1",
    "MultimodalVLMServiceConfig": "multimodal_vlm_modelslim_v1_spec",
    "MultimodalSDModelslimV1QuantConfig": "multimodal_sd_modelslim_v1",
    "MultimodalSDServiceConfig": "multimodal_sd_modelslim_v1_spec",
    "ModelslimConvertQuantConfig": "modelslim_convert",
    "ModelslimConvertServiceConfig": "modelslim_convert_spec",
    "AdaptRotationStage1ProcessorConfig": "adapt_rotation_stage1",
    "AdaptRotationStage2ProcessorConfig": "adapt_rotation_stage2",
    "TuningPlanConfig": "tuning_plan",
    "StandingHighStrategyConfig": "strategy_standing_high",
    "StandingHighWithExperienceStrategyConfig": "strategy_standing_high_with_experience",
    "BinaryFallbackStrategyConfig": "strategy_binary_fallback",
    "ServiceOrientedEvaluateServiceConfig": "evaluation_service_oriented",
}

QUALNAME_SLUGS = {
    "msmodelslim.processor.quant.autoround.QuantStrategyConfig": "autoround_quant_strategy_config",
    "msmodelslim.processor.flat_quant.flat_quant.QuantStrategyConfig": "flatquant_quant_strategy_config",
    "msmodelslim.processor.trainable_linear_quant.config.processor_config.QuantStrategyConfig": "tlq_quant_strategy_config",
}

APIVERSION_BY_ROOT = {
    "ModelslimV1QuantConfig": "modelslim_v1",
    "MultimodalVLMModelslimV1QuantConfig": "multimodal_vlm_modelslim_v1",
    "MultimodalSDModelslimV1QuantConfig": "multimodal_sd_modelslim_v1",
    "ModelslimConvertQuantConfig": "modelslim_convert",
}

SKIP_CLASS_NAMES = {
    "BaseModel",
    "BaseQuantConfig",
    "AutoProcessorConfig",
    "QuantFormatConfig",
    "AutoSaverBaseConfig",
    "TypedConfig",
    "QuantServiceConfig",
}

TARGET_GROUP_KEYWORDS = {
    "task": "任务配置",
    "root": "任务配置",
    "spec": "服务规格",
    "service": "服务规格",
    "processor": "处理器",
    "format": "保存格式",
    "save": "保存格式",
    "tuning": "自动调优",
    "auto_tuning": "自动调优",
    "nested": "嵌套配置",
}

CATEGORY_DIRS = {
    "任务配置": "quant",
    "服务规格": "spec",
    "处理器": "processor",
    "保存格式": "format",
    "自动调优": "tuning",
    "嵌套配置": "nested",
}

BASE_CLASS_REPRESENTATIVE_TYPE_TAGS = {
    "AutoProcessorConfig": "linear_quant",
    "ProcessorConfig": "linear_quant",
    "QuantFormatConfig": "ascendv1_saver",
}

SOURCE_REFERENCE_PATHS = {
    "QuantFormatConfig": "msmodelslim/format/base.py",
}

CONFIG_SOURCE_MODULES = (
    "msmodelslim.processor",
    "msmodelslim.processor.kv_smooth.processor",
    "msmodelslim.processor.quarot",
    "msmodelslim.format.registry",
    "msmodelslim.core.quant_service.modelslim_v1.quant_config",
    "msmodelslim.core.quant_service.multimodal_vlm_v1.quant_config",
    "msmodelslim.core.quant_service.multimodal_sd_v1.quant_config",
    "msmodelslim.core.quant_service.modelslim_convert.quant_config",
    "msmodelslim.core.quant_service.modelslim_convert.config_mapper",
    "msmodelslim.app.auto_tuning.plan_manager_infra",
    "msmodelslim.app.auto_tuning.evaluation_service_infra",
    "msmodelslim.core.tune_strategy.interface",
    "msmodelslim.core.tune_strategy.standing_high.strategy",
    "msmodelslim.core.tune_strategy.standing_high_with_experience.strategy",
    "msmodelslim.core.tune_strategy.binary_fallback.strategy",
    "msmodelslim.infra.service_oriented_evaluate_service",
    "msmodelslim.infra.evaluation.aisbench_server",
    "msmodelslim.infra.evaluation.precheck.base",
    "msmodelslim.infra.evaluation.precheck.expected_answer_rule",
    "msmodelslim.infra.evaluation.precheck.garbled_text_rule",
    "msmodelslim.infra.vllm_ascend_server",
)
