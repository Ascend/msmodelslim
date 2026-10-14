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

from pathlib import Path
from typing import List, Any, Generator

from torch import nn

from msmodelslim.core.base.protocol import ProcessRequest
from msmodelslim.core.const import DeviceType
from msmodelslim.model.common.layer_wise_forward import (
    transformers_generated_forward_func,
    generated_decoder_layer_visit_func,
)
from msmodelslim.utils.exception import InvalidModelError
from msmodelslim.utils.exception_decorator import exception_handler
from msmodelslim.utils.logging import logger_setter
from ..common.transformers import TransformersModel
from ..interface_hub import (
    ModelInfoInterface,
    ModelSlimPipelineInterfaceV0,
    ModelSlimPipelineInterfaceV1,
    StandingHighWithExperienceInterface,
)


@logger_setter()
class DefaultModelAdapter(  # pylint: disable=too-many-ancestors
    TransformersModel,
    ModelInfoInterface,  # support naive quantization
    ModelSlimPipelineInterfaceV0,  # support modelslim v0
    ModelSlimPipelineInterfaceV1,  # support modelslim v1
    StandingHighWithExperienceInterface,  # support standing_high_with_experience
):
    """
    Default model adapter which implements some widely used interface.
    You can try to quant new unadapted model by using this model adapter.
    HOWEVER, it may be not functional.
    You can treat this model adapter as a reference to implement your own model adapter.
    """

    def __init__(self, model_type: str, model_path: Path, trust_remote_code: bool = False):
        with exception_handler(
            'You are creating default model adapter but failed',
            ms_err_cls=InvalidModelError,
            action='Please ensure default model adapter match your model',
        ):
            super().__init__(model_type, model_path, trust_remote_code)

    def get_model_type(self) -> str:
        return self.model_type

    def get_model_pedigree(self) -> str:
        return 'default'

    def load_model(self, device: DeviceType = DeviceType.NPU) -> nn.Module:
        with exception_handler(
            'You are creating default model adapter but failed',
            ms_err_cls=InvalidModelError,
            action='Please ensure default model adapter match your model',
        ):
            return self._load_model(device)

    def handle_dataset(self, dataset: Any, device: DeviceType = DeviceType.NPU) -> List[Any]:
        with exception_handler(
            'You are creating default model adapter but failed',
            ms_err_cls=InvalidModelError,
            action='Please ensure default model adapter match your model',
        ):
            return self._get_tokenized_data(dataset, device)

    def handle_dataset_by_batch(self, dataset: Any, batch_size: int, device: DeviceType = DeviceType.NPU) -> List[Any]:
        with exception_handler(
            'You are creating default model adapter but failed',
            ms_err_cls=InvalidModelError,
            action='Please ensure default model adapter match your model',
        ):
            return self._get_batch_tokenized_data(calib_list=dataset, batch_size=batch_size, device=device)

    def init_model(self, device: DeviceType = DeviceType.NPU) -> nn.Module:
        with exception_handler(
            'You are creating default model adapter but failed',
            ms_err_cls=InvalidModelError,
            action='Please ensure default model adapter match your model',
        ):
            return self._load_model(device)

    def generate_model_visit(self, model: nn.Module) -> Generator[ProcessRequest, Any, None]:
        with exception_handler(
            'You are creating default model adapter but failed',
            ms_err_cls=InvalidModelError,
            action='Please ensure default model adapter match your model',
        ):
            yield from generated_decoder_layer_visit_func(model)

    def generate_model_forward(
        self,
        model: nn.Module,
        inputs: Any,
    ) -> Generator[ProcessRequest, Any, None]:
        with exception_handler(
            'You are creating default model adapter but failed',
            ms_err_cls=InvalidModelError,
            action='Please ensure default model adapter match your model',
        ):
            yield from transformers_generated_forward_func(model, inputs)

    def enable_kv_cache(self, model: nn.Module, need_kv_cache: bool) -> None:
        with exception_handler(
            'You are creating default model adapter but failed',
            ms_err_cls=InvalidModelError,
            action='Please ensure default model adapter match your model',
        ):
            return self._enable_kv_cache(model, need_kv_cache)
