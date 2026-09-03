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

from typing import Any, Generator, List

from torch import nn

from msmodelslim.core.base.protocol import ProcessRequest
from msmodelslim.core.const import DeviceType
from msmodelslim.utils.logging import logger_setter
from msmodelslim.utils.exception import InvalidModelError
from msmodelslim.utils.exception_decorator import exception_handler
from ..common.layer_wise_forward import (
    generated_decoder_layer_visit_func,
    transformers_generated_forward_func,
)
from ..common.transformers import TransformersModel
from ..interface_hub import (
    ModelInfoInterface,
    ModelSlimPipelineInterfaceV1,
)


@logger_setter()
class LLMTransformersModel(  # pylint: disable=too-many-ancestors
    TransformersModel,
    ModelInfoInterface,
    ModelSlimPipelineInterfaceV1,
):
    """Generic HuggingFace LLM adapter for ``--model_type transformers``.

    Suitable for standard decoder-layer LLMs. MoE / VLM / custom structures
    still need dedicated adapters.
    """

    def get_model_type(self) -> str:
        return self.model_type

    def get_model_pedigree(self) -> str:
        return "llm_transformers"

    def handle_dataset(self, dataset: Any, device: DeviceType = DeviceType.NPU) -> List[Any]:
        with exception_handler(
            'You are handling dataset with llm transformers model adapter but failed',
            ms_err_cls=InvalidModelError,
            action='Please ensure llm transformers model adapter match your model',
        ):
            return self._get_tokenized_data(dataset, device)

    def init_model(self, device: DeviceType = DeviceType.NPU) -> nn.Module:
        with exception_handler(
            'You are initializing model with llm transformers model adapter but failed',
            ms_err_cls=InvalidModelError,
            action='Please ensure llm transformers model adapter match your model. '
            'For multimodal model (VLM/DiT) quantization, please implement a dedicated model adapter',
        ):
            return self._load_model(device)

    def generate_model_visit(self, model: nn.Module) -> Generator[ProcessRequest, Any, None]:
        with exception_handler(
            'You are generating model visit with llm transformers model adapter but failed',
            ms_err_cls=InvalidModelError,
            action='Please ensure llm transformers model adapter match your model',
        ):
            yield from generated_decoder_layer_visit_func(model)

    def generate_model_forward(
        self,
        model: nn.Module,
        inputs: Any,
    ) -> Generator[ProcessRequest, Any, None]:
        with exception_handler(
            'You are generating model forward with llm transformers model adapter but failed',
            ms_err_cls=InvalidModelError,
            action='Please ensure llm transformers model adapter match your model',
        ):
            yield from transformers_generated_forward_func(model, inputs)

    def enable_kv_cache(self, model: nn.Module, need_kv_cache: bool) -> None:
        with exception_handler(
            'You are enabling kv cache with llm transformers model adapter but failed',
            ms_err_cls=InvalidModelError,
            action='Please ensure llm transformers model adapter match your model',
        ):
            return self._enable_kv_cache(model, need_kv_cache)
