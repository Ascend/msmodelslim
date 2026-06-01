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
from typing import Any, List

from torch import nn

from msmodelslim.core.const import DeviceType
from msmodelslim.model import IModel
from msmodelslim.utils.exception import UnsupportedError


class PipelineInterface(IModel):
    """
    Interface for determining the pipeline of model inference.
    ModelSlim V0 is a simple quant service, which does NOT schedule the model inference pipeline in finer granularity.
    Just show how to handle dataset and load model. Remaining parts are left to the model itself.
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
        raise UnsupportedError(
            "This model does not support generate dataset.",
            action="Please implement generate_dataset in PipelineInterface.",
        )

    @abstractmethod
    def handle_dataset_by_batch(self, dataset: Any, batch_size: int, device: DeviceType = DeviceType.NPU) -> List[Any]:
        """
        Handle the dataset for model inference with certain batch size.
        The dataset should be converted into a List of data
            that can be directly used for model inference(model(*data) or model(**data)).
        Returns:
            List[Any]: The processed dataset.
        """
        raise UnsupportedError(
            "This model does not support generate dataset by batch.",
            action="Please implement generate_dataset_by_batch in PipelineInterface.",
        )

    @abstractmethod
    def load_model(self, device: DeviceType = DeviceType.NPU) -> nn.Module:
        """
        Load the model to specified device.
        After loading, the model should be ready for inference.

        Returns:
            nn.Module: The loaded model.
        """
        raise UnsupportedError(
            "This model does not support load model to specified device and torch dtype.",
            action="Please implement load_model in PipelineInterface.",
        )
