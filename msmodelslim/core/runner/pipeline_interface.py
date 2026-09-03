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
from typing import Generator, Any, List

from torch import nn

from msmodelslim.core.base.protocol import ProcessRequest
from msmodelslim.core.const import DeviceType
from msmodelslim.model import IModel
from msmodelslim.utils.exception import ToDoError


class PipelineInterface(IModel):
    """
    Interface for determining the pipeline of model inference.
    Runner schedules the process of model in finer granularity.
    The granularity of the pipeline is determined by users via the implementation of this interface.
    """

    @abstractmethod
    def handle_dataset(self, dataset: Any, device: DeviceType = DeviceType.NPU) -> List[Any]:
        """
        Handle the dataset for model inference.
        The dataset should be converted into a List of data
            that can be directly used for model inference(model(*data) or model(**data)).
        Returns:
            List[Any]: The processed dataset.
        """
        raise ToDoError(
            "This model does not support generate dataset.",
            action="Please implement generate_dataset for PipelineInterface.",
        )

    @abstractmethod
    def init_model(self, device: DeviceType = DeviceType.NPU) -> nn.Module:
        """
        Init the model to specified device which may be different from execution device.
        If the model is large, just load a part of the model.
        You can extend the model when generating model visit and forward.

        Returns:
            nn.Module: The loaded model.
        """
        raise ToDoError(
            "This model does not support init model to specified device.",
            action="Please implement init_model for PipelineInterface.",
        )

    @abstractmethod
    def generate_model_visit(self, model: nn.Module) -> Generator[ProcessRequest, Any, None]:
        """
        Determine the model visit pipeline for data-free quantization.
        Used when **all** processors in the pipeline are data-free.

        The model visit pipeline is a generator of ProcessRequest,
            which decomposes the model modification into a list of module visit.

        Returns:
            Generator[ProcessRequest, Any, None]: The generator of model visit.
        """
        raise ToDoError(
            "This model does not support generate_model_visit, which is required for model modification.",
            action="Please implement generate_model_visit for PipelineInterface.",
        )

    @abstractmethod
    def generate_model_forward(
        self,
        model: nn.Module,
        inputs: Any,
    ) -> Generator[ProcessRequest, Any, None]:
        """
        Determine the model forward pipeline for calibration-based quantization.
        Used when **any** processor in the pipeline requires calibration data.
        All process units share this forward schedule (data-free processors skip ``process()``).

        The model forward pipeline is a generator of ProcessRequest,
            which decomposes the model forward into a list of module forward.

        Returns:
            Generator[ProcessRequest, Any, None]: The generator of model forward.
        """
        raise ToDoError(
            "This model does not support generate_model_forward, which is required for model calibration.",
            action="Please implement generate_model_forward for PipelineInterface.",
        )

    @abstractmethod
    def enable_kv_cache(self, model: nn.Module, need_kv_cache: bool) -> None:
        """
        Enable/Disable the kv cache for the model.
        Quantization Calibration probably does NOT requires the kv cache, which depends on specific processor.
        Disable kv cache can reduce the memory usage.
        """
        raise ToDoError(
            "This model does not support enable_kv_cache, which is required for model calibration.",
            action="Please implement enable_kv_cache for PipelineInterface.",
        )
