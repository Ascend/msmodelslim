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

from abc import abstractmethod

from msmodelslim.core.runner.pipeline_interface import PipelineInterface
from msmodelslim.utils.exception import ToDoError


class LegacyMultimodalPipelineInterface(PipelineInterface):
    """
    主仓兼容的多模态生成 pipeline 接口（旧版）。

    供 flux1、wan2_1、wan2_2/model_adapter、qwen_image_edit 及基于主仓自行接入的新模型使用。
    与 ``pipeline_interface.MultimodalPipelineInterface`` 并存，由量化服务按适配器类型分发。
    """

    @abstractmethod
    def run_calib_inference(self):
        raise ToDoError(
            "This model does not support run_calib_inference.",
            action="Please implement run_calib_inference for your model.",
        )

    @abstractmethod
    def apply_quantization(self, quant_model_func):
        """
        应用模型量化的抽象方法，子类需实现具体逻辑。

        参数:
            quant_model_func: 量化函数（即 api 中的 quant_model 函数）
        """
        raise ToDoError(
            "This model does not support apply_quantization.",
            action="Please implement apply_quantization for your model.",
        )

    @abstractmethod
    def load_pipeline(self):
        raise ToDoError(
            "This model does not support load_pipeline.",
            action="Please implement load_pipeline for your model.",
        )

    @abstractmethod
    def set_model_args(self, override_model_config: object):
        raise ToDoError(
            "This model does not support set_model_args.",
            action="Please implement set_model_args for your model.",
        )
