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

from typing import List, Any, Generator

from torch import nn
from transformers import PreTrainedTokenizerBase

from msmodelslim.core.base.protocol import ProcessRequest
from msmodelslim.core.const import DeviceType
from msmodelslim.processor.kv_smooth import KVSmoothFusedType, KVSmoothFusedUnit
from msmodelslim.utils.exception import InvalidModelError
from msmodelslim.utils.logging import logger_setter
from msmodelslim.utils.security.model import SafeGenerator
from ..common.layer_wise_forward import generated_decoder_layer_visit_func, transformers_generated_forward_func
from ..default.model_adapter import DefaultModelAdapter
from ..interface_hub import (
    ModelInfoInterface,
    ModelSlimPipelineInterfaceV0,
    ModelSlimPipelineInterfaceV1,
    StandingHighWithExperienceInterface,
    KVSmoothFusedInterface,
)


@logger_setter()
class Qwen25ModelAdapter(  # pylint: disable=too-many-ancestors
    DefaultModelAdapter,
    ModelInfoInterface,
    ModelSlimPipelineInterfaceV0,
    ModelSlimPipelineInterfaceV1,
    StandingHighWithExperienceInterface,
    KVSmoothFusedInterface,
):
    def get_model_type(self) -> str:
        return self.model_type

    def get_model_pedigree(self) -> str:
        return 'qwen2_5'

    def load_model(self, device: DeviceType = DeviceType.NPU) -> nn.Module:
        return self._load_model(device)

    def handle_dataset(self, dataset: Any, device: DeviceType = DeviceType.NPU) -> List[Any]:
        return self._get_tokenized_data(dataset, device)

    def handle_dataset_by_batch(self, dataset: Any, batch_size: int, device: DeviceType = DeviceType.NPU) -> List[Any]:
        return self._get_batch_tokenized_data(calib_list=dataset, batch_size=batch_size, device=device)

    def init_model(self, device: DeviceType = DeviceType.NPU) -> nn.Module:
        return self._load_model(device)

    def generate_model_visit(self, model: nn.Module) -> Generator[ProcessRequest, Any, None]:
        yield from generated_decoder_layer_visit_func(model)

    def generate_model_forward(
        self,
        model: nn.Module,
        inputs: Any,
    ) -> Generator[ProcessRequest, Any, None]:
        yield from transformers_generated_forward_func(model, inputs)

    def enable_kv_cache(self, model: nn.Module, need_kv_cache: bool) -> None:
        return self._enable_kv_cache(model, need_kv_cache)

    def get_kvcache_smooth_fused_subgraph(self) -> List[KVSmoothFusedUnit]:
        return [
            KVSmoothFusedUnit(
                attention_name=f"model.layers.{i}.self_attn",
                layer_idx=i,
                fused_from_query_states_name="q_proj",
                fused_from_key_states_name="k_proj",
                fused_type=KVSmoothFusedType.StateViaRopeToLinear,
            )
            for i in range(self.config.num_hidden_layers)
        ]

    def get_head_dim(self) -> int:
        if not hasattr(self.config, 'hidden_size'):
            raise InvalidModelError(
                "hidden_size is not found in config.json", action="Please check the model config.json"
            )
        if not hasattr(self.config, 'num_attention_heads'):
            raise InvalidModelError(
                "num_attention_heads is not found in config.json", action="Please check the model config.json"
            )
        if self.config.num_attention_heads == 0:
            raise InvalidModelError(
                "num_attention_heads is 0 in config.json, which should be greater than 0",
                action="Please check the model config.json",
            )
        return self.config.hidden_size // self.config.num_attention_heads

    def get_num_key_value_groups(self) -> int:
        if not hasattr(self.config, 'num_attention_heads'):
            raise InvalidModelError(
                "num_attention_heads is not found in config.json", action="Please check the model config.json"
            )
        if not hasattr(self.config, 'num_key_value_heads'):
            raise InvalidModelError(
                "num_key_value_heads is not found in config.json", action="Please check the model config.json"
            )
        if self.config.num_key_value_heads == 0:
            raise InvalidModelError(
                "num_key_value_heads is 0 in config.json, which should be greater than 0",
                action="Please check the model config.json",
            )
        return self.config.num_attention_heads // self.config.num_key_value_heads

    def get_num_key_value_heads(self) -> int:
        if not hasattr(self.config, 'num_key_value_heads'):
            raise InvalidModelError(
                "num_key_value_heads is not found in config.json", action="Please check the model config.json"
            )
        return self.config.num_key_value_heads

    def _load_tokenizer(self, trust_remote_code=False) -> PreTrainedTokenizerBase:
        return SafeGenerator.get_tokenizer_from_pretrained(
            model_path=str(self.model_path),
            use_fast=False,
            legacy=False,
            padding_side='left',
            pad_token='<|extra_0|>',  # nosec B106
            eos_token='<|endoftext|>',  # nosec B106,
            trust_remote_code=trust_remote_code,
        )
