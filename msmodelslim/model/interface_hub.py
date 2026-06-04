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
"""

# Interface hub collects all the interfaces in the project to assist the model adapter's development.
#
# Each interface represents a series of demands to model for specific component dedicated to specific domain.
# It is unnecessary to implement all the interfaces in a model adapter.
# Just implement the interfaces for components you need.

from msmodelslim.app.auto_tuning.model_info_interface import ModelInfoInterface as atmi
from msmodelslim.app.naive_quantization.model_info_interface import ModelInfoInterface as nami
from msmodelslim.core.quant_service.modelslim_v0.pipeline_interface import (
    PipelineInterface as ModelSlimPipelineInterfaceV0,
)
from msmodelslim.core.quant_service.modelslim_v1.save.interface import (
    AscendV1GlobalModelDtypeInterface,
    AscendV1SaveInterface,
)
from msmodelslim.core.quant_service.multimodal_sd_v1.legacy_pipeline_interface import (
    LegacyMultimodalPipelineInterface,
)
from msmodelslim.core.quant_service.multimodal_sd_v1.pipeline_interface import MultimodalPipelineInterface
from msmodelslim.core.runner.optional_interface import LayerWiseOffloadOptionalInterface
from msmodelslim.core.runner.pipeline_interface import PipelineInterface as ModelSlimPipelineInterfaceV1
from msmodelslim.core.tune_strategy.standing_high.standing_high_interface import StandingHighInterface
from msmodelslim.core.tune_strategy.standing_high_with_experience.standing_high_with_experience_interface import (
    StandingHighWithExperienceInterface,
)
from msmodelslim.model import IModel
from msmodelslim.processor.adapt_rotation import AdaptRotationInterface
from msmodelslim.processor.analysis.binary_operator.metrics.attention_mse.interface import (
    AttentionMSEAnalysisInterface as amseai,
)
from msmodelslim.processor.anti_outlier.flex_smooth.interface import FlexSmoothQuantInterface
from msmodelslim.processor.anti_outlier.iter_smooth.interface import IterSmoothInterface
from msmodelslim.processor.anti_outlier.smooth_quant.interface import SmoothQuantInterface
from msmodelslim.processor.kv_smooth import KVSmoothFusedInterface
from msmodelslim.processor.quant.fa3.interface import FA3QuantAdapterInterface, FA3QuantPlaceHolder
from msmodelslim.processor.quarot import LAOSOnlineRotationInterface, OnlineQuaRotInterface, QuaRotInterface


class AnalyzePipelineInterface(StandingHighInterface, StandingHighWithExperienceInterface): ...


class ModelInfoInterface(nami, atmi): ...


class AttentionAnalysisInterface(amseai): ...


# 迁移期兼容
MultimodalSDPipelineInterface = LegacyMultimodalPipelineInterface


__all__ = [
    # base interface
    'IModel',  # Necessary, including model_type, model_path, trust_remote_code, etc.
    # app interface
    'ModelInfoInterface',  # For Naive Quantization, get model info from model.
    'AnalyzePipelineInterface',  # For Analysis, describing the pipeline of model inference.
    # analysis_method interface
    'AttentionAnalysisInterface',  # For Attention Analysis, describing the attention structure of model.
    # algorithm interface
    'KVSmoothFusedInterface',  # For KV Smooth, describing the architecture of model.
    'SmoothQuantInterface',  # For Smooth Quant, describing the architecture of model.
    'IterSmoothInterface',  # For Iter Smooth, describing the architecture of model.
    'FlexSmoothQuantInterface',  # For Flex Smooth Quant, describing the architecture of model.
    # remaining interface
    'ModelSlimPipelineInterfaceV0',  # For ModelSlim V0 quant service, describing the pipeline of model inference.
    'LegacyMultimodalPipelineInterface',  # 主仓兼容的多模态生成 pipeline 接口。
    'MultimodalSDPipelineInterface',  # 迁移期别名，与 LegacyMultimodalPipelineInterface 相同。
    'MultimodalPipelineInterface',  # 多模态生成量化 pipeline（当前标准接口）。
    'ModelSlimPipelineInterfaceV1',  # For ModelSlim V1 quant service, describing the pipeline of model inference.
    'LayerWiseOffloadOptionalInterface',  # Optional interface for layer-wise runner offload customization.
    'StandingHighInterface',  # For Standing High strategy, describing the pipeline of model inference.
    'StandingHighWithExperienceInterface',  # For Standing High with Experience strategy, describing the pipeline of model inference.
    # FA3 activation quantization interface
    'FA3QuantAdapterInterface',  # For FA3 activation quantization, inject placeholders.
    'FA3QuantPlaceHolder',  # For FA3 activation quantization, placeholders.
    # QuaRot interface
    'QuaRotInterface',  # For QuaRot.
    'AdaptRotationInterface',  # For AdaptRotation stage1/stage2.
    'LAOSOnlineRotationInterface',  # For LAOS online rotation.
    'OnlineQuaRotInterface',  # For QuaRotOnline.
    # save interface
    'AscendV1SaveInterface',  # For AscendV1 save.
    'AscendV1GlobalModelDtypeInterface',  # For global model torch dtype (e.g. Saver deq_scale int64).
]
