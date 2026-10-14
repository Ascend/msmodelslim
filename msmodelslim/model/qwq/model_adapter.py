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
# pylint: disable=no-name-in-module

from typing import Any, List, Generator

from torch import nn
from transformers import PreTrainedTokenizerBase

from msmodelslim.core.base.protocol import ProcessRequest
from msmodelslim.core.const import DeviceType
from msmodelslim.core.graph.adapter_types import AdapterConfig, MappingConfig
from msmodelslim.processor.quarot import QuaRotInterface, LAOSOnlineRotationInterface
from msmodelslim.utils.exception import InvalidModelError
from msmodelslim.utils.logging import logger_setter, get_logger
from msmodelslim.utils.security.model import SafeGenerator
from ..common.layer_wise_forward import generated_decoder_layer_visit_func, transformers_generated_forward_func
from ..default.model_adapter import DefaultModelAdapter
from ..interface_hub import (
    ModelInfoInterface,
    ModelSlimPipelineInterfaceV0,
    ModelSlimPipelineInterfaceV1,
    StandingHighWithExperienceInterface,
    IterSmoothInterface,
)


@logger_setter()
class QwqModelAdapter(  # pylint: disable=too-many-ancestors
    DefaultModelAdapter,
    ModelInfoInterface,
    ModelSlimPipelineInterfaceV0,
    ModelSlimPipelineInterfaceV1,
    StandingHighWithExperienceInterface,
    IterSmoothInterface,
    QuaRotInterface,
    LAOSOnlineRotationInterface,
):
    def get_model_type(self) -> str:
        return self.model_type

    def get_model_pedigree(self) -> str:
        return 'qwq'

    def load_model(self, device: DeviceType = DeviceType.NPU) -> nn.Module:
        return self._load_model(device)

    def _load_tokenizer(self, trust_remote_code=False) -> PreTrainedTokenizerBase:
        return SafeGenerator.get_tokenizer_from_pretrained(
            model_path=str(self.model_path),
            use_fast=False,
            legacy=False,
            padding_side='left',
            pad_token='<|extra_0|>',  # nosec B106
            eos_token='<|endoftext|>',  # nosec B106
            trust_remote_code=trust_remote_code,
        )

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

    def get_hidden_dim(self):
        return self.config.hidden_size

    def get_head_dim(self) -> int:
        if hasattr(self.config, 'head_dim'):
            return self.config.head_dim

        get_logger().warning('head_dim is not found in config.json, use hidden_size // num_attention_heads instead')
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

    def get_num_attention_heads(self):
        if not hasattr(self.config, 'num_attention_heads'):
            raise InvalidModelError(
                "num_attention_heads is not found in config.json",
                action=f"Please check config.json in {self.model_path}",
            )
        return self.config.num_attention_heads

    def get_layer_wise_ov_pair(self, decoder_module: nn.Module):
        ov_pairs = {decoder_module.self_attn.o_proj: decoder_module.self_attn.v_proj}
        return ov_pairs

    def get_layer_wise_up_down_pair(self, decoder_module: nn.Module):
        up_down_pairs = {decoder_module.mlp.up_proj: decoder_module.mlp.down_proj}
        return up_down_pairs

    def get_ln_fuse_map(self):
        return {}, qwq_get_ln_fuse_map(self.config)

    def get_bake_names(self):
        return [], []

    def get_adapter_config_for_subgraph(self) -> List[AdapterConfig]:
        adapter_config = []
        for layer_idx in range(self.config.num_hidden_layers):
            norm_linear_mapping_config1 = MappingConfig(
                source=f"model.layers.{layer_idx}.input_layernorm",
                targets=[
                    f"model.layers.{layer_idx}.self_attn.k_proj",
                    f"model.layers.{layer_idx}.self_attn.q_proj",
                    f"model.layers.{layer_idx}.self_attn.v_proj",
                ],
            )

            norm_linear_mapping_config2 = MappingConfig(
                source=f"model.layers.{layer_idx}.post_attention_layernorm",
                targets=[
                    f"model.layers.{layer_idx}.mlp.gate_proj",
                    f"model.layers.{layer_idx}.mlp.up_proj",
                ],
            )

            up_down_mapping_config = MappingConfig(
                source=f"model.layers.{layer_idx}.mlp.up_proj",
                targets=[f"model.layers.{layer_idx}.mlp.down_proj"],
            )

            adapter_config.extend(
                [
                    AdapterConfig(
                        subgraph_type="norm-linear",
                        mapping=norm_linear_mapping_config1,
                    ),
                    AdapterConfig(
                        subgraph_type="norm-linear",
                        mapping=norm_linear_mapping_config2,
                    ),
                    AdapterConfig(
                        subgraph_type="up-down",
                        mapping=up_down_mapping_config,
                    ),
                ]
            )
        return adapter_config

    def get_rotate_map(self, block_size):
        pre_run, rot_pairs, _, _ = qwq_get_rotate_map(self.config, block_size)
        return [pre_run], list(rot_pairs.values())


def qwq_get_ln_fuse_map(config):
    ln_linear_map = {}
    for layer_idx in range(config.num_hidden_layers):
        ln_linear_map[f"model.layers.{layer_idx}.input_layernorm"] = [
            f"model.layers.{layer_idx}.self_attn.q_proj",
            f"model.layers.{layer_idx}.self_attn.k_proj",
            f"model.layers.{layer_idx}.self_attn.v_proj",
        ]

        ln_linear_map[f"model.layers.{layer_idx}.post_attention_layernorm"] = [
            f"model.layers.{layer_idx}.mlp.{proj}" for proj in ["gate_proj", "up_proj"]
        ]
    ln_linear_map["model.norm"] = ['lm_head']
    return ln_linear_map


def qwq_get_rotate_map(config, block_size):
    head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
    rot = QuaRotInterface.get_rotate_command(
        size=config.hidden_size,
        block_size=block_size,
        mode=QuaRotInterface.QuaRotMode.HADAMARD,
    )
    rot_uv = QuaRotInterface.get_rotate_command(
        size=head_dim,
        block_size=block_size,
        mode=QuaRotInterface.QuaRotMode.BLOCK_HADAMARD_SHIFTED,
    )

    left_rot = {}
    right_rot = {
        "model.embed_tokens": rot,
    }
    pre_run = QuaRotInterface.RotatePair(left_rot=left_rot, right_rot=right_rot)

    rot_pairs = {}
    left_rot = {}
    right_rot = {
        "lm_head": rot,
    }
    for layer_idx in range(config.num_hidden_layers):
        right_rot[f"model.layers.{layer_idx}.self_attn.q_proj"] = rot
        right_rot[f"model.layers.{layer_idx}.self_attn.k_proj"] = rot
        right_rot[f"model.layers.{layer_idx}.self_attn.v_proj"] = rot
        left_rot[f"model.layers.{layer_idx}.self_attn.o_proj"] = rot

        right_rot[f"model.layers.{layer_idx}.mlp.gate_proj"] = rot
        right_rot[f"model.layers.{layer_idx}.mlp.up_proj"] = rot
        left_rot[f"model.layers.{layer_idx}.mlp.down_proj"] = rot
    rot_pairs['rot'] = QuaRotInterface.RotatePair(left_rot=left_rot, right_rot=right_rot)

    left_rot_uv = {}
    right_rot_uv = {}
    for layer_idx in range(config.num_hidden_layers):
        left_rot_uv[f"model.layers.{layer_idx}.self_attn.v_proj"] = rot_uv
        right_rot_uv[f"model.layers.{layer_idx}.self_attn.o_proj"] = rot_uv
    rot_pairs["rot_uv"] = QuaRotInterface.RotatePair(left_rot=left_rot_uv, right_rot=right_rot_uv)

    return pre_run, rot_pairs, rot, rot_uv
