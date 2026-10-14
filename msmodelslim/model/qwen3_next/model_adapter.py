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
# pylint: disable=no-name-in-module,too-many-ancestors

from typing import List, Any, Optional, Generator, Tuple

from torch import nn
from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm
from transformers.models.qwen3_next.modeling_qwen3_next import Qwen3NextRMSNorm

from msmodelslim.core.base.protocol import ProcessRequest
from msmodelslim.core.const import DeviceType
from msmodelslim.core.graph import AdapterConfig, MappingConfig
from msmodelslim.utils.logging import logger_setter
from ..common.layer_wise_forward import generated_decoder_layer_visit_func, transformers_generated_forward_func
from ..default.model_adapter import DefaultModelAdapter
from ..interface_hub import (
    ModelInfoInterface,
    ModelSlimPipelineInterfaceV0,
    ModelSlimPipelineInterfaceV1,
    IterSmoothInterface,
    FlexSmoothQuantInterface,
    AscendV1SaveInterface,
)


@logger_setter()
class Qwen3NextModelAdapter(
    DefaultModelAdapter,
    ModelInfoInterface,
    ModelSlimPipelineInterfaceV0,
    ModelSlimPipelineInterfaceV1,
    IterSmoothInterface,
    FlexSmoothQuantInterface,
    AscendV1SaveInterface,
):
    def get_model_type(self) -> str:
        return self.model_type

    def get_model_pedigree(self) -> str:
        return 'qwen3_next'

    def load_model(self, device: DeviceType = DeviceType.NPU) -> nn.Module:
        return self._load_model(device)

    def handle_dataset(self, dataset: Any, device: DeviceType = DeviceType.NPU) -> List[Any]:
        return self._get_tokenized_data(dataset, device)

    def handle_dataset_by_batch(self, dataset: Any, batch_size: int, device: DeviceType = DeviceType.NPU) -> List[Any]:
        return self._get_batch_tokenized_data(calib_list=dataset, batch_size=batch_size, device=device)

    def init_model(self, device: DeviceType = DeviceType.NPU) -> nn.Module:
        loaded_model = self._load_model(device)

        for name, module in loaded_model.named_modules():
            if 'input_layernorm' in name and module.__class__.__name__ == 'Qwen3NextRMSNorm':
                new_module = Qwen3RMSNorm(module.weight.shape[0], module.eps)
                new_module.weight.data = module.weight.data + 1
                loaded_model.set_submodule(name, new_module)

        return loaded_model

    def generate_model_visit(
        self,
        model: nn.Module,
        transformer_blocks: Optional[List[Tuple[str, nn.Module]]] = None,
    ) -> Generator[ProcessRequest, Any, None]:
        yield from generated_decoder_layer_visit_func(model, transformer_blocks)

    def generate_model_forward(
        self,
        model: nn.Module,
        inputs: Any,
    ) -> Generator[ProcessRequest, Any, None]:
        yield from transformers_generated_forward_func(model, inputs)

    def enable_kv_cache(self, model: nn.Module, need_kv_cache: bool) -> None:
        return self._enable_kv_cache(model, need_kv_cache)

    def get_adapter_config_for_subgraph(self) -> List[AdapterConfig]:
        adapter_config = []
        for layer_idx in range(
            self.config.full_attention_interval - 1, self.config.num_hidden_layers, self.config.full_attention_interval
        ):
            # Norm-Linear融合的映射配置：输入层归一化到QKV投影
            norm_linear_mapping_config = MappingConfig(
                source=f"model.layers.{layer_idx}.input_layernorm",  # 第一个LayerNorm
                targets=[
                    f"model.layers.{layer_idx}.self_attn.k_proj",
                    f"model.layers.{layer_idx}.self_attn.q_proj",
                    f"model.layers.{layer_idx}.self_attn.v_proj",
                ],  # 注意力层的QKV投影
            )

            # 为当前layer添加配置
            adapter_config.extend(
                [
                    AdapterConfig(subgraph_type="norm-linear", mapping=norm_linear_mapping_config),
                ]
            )
        return adapter_config

    def ascendv1_save_module_preprocess(
        self, prefix: str, module: nn.Module, model: nn.Module
    ) -> Tuple[str, nn.Module]:
        if 'input_layernorm' in prefix and module.__class__.__name__ == 'Qwen3RMSNorm':
            new_module = Qwen3NextRMSNorm(module.weight.shape[0], module.variance_epsilon)
            new_module.weight.data = module.weight.data - 1
            model.set_submodule(prefix, new_module)
            return prefix, new_module
        return prefix, module
