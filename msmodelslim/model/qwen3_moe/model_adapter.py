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

from typing import List, Any, Generator

from torch import nn

from msmodelslim.core.base.protocol import ProcessRequest
from msmodelslim.core.const import DeviceType
from msmodelslim.core.graph import AdapterConfig, MappingConfig
from msmodelslim.processor.quarot import QuaRotInterface
from msmodelslim.utils.logging import logger_setter
from ..common.layer_wise_forward import generated_decoder_layer_visit_func, transformers_generated_forward_func
from ..default.model_adapter import DefaultModelAdapter
from ..interface_hub import (
    ModelInfoInterface,
    ModelSlimPipelineInterfaceV0,
    ModelSlimPipelineInterfaceV1,
    IterSmoothInterface,
    FlexSmoothQuantInterface,
    AdaptRotationInterface,
)


# pylint: disable=too-many-ancestors
@logger_setter()
class Qwen3MoeModelAdapter(
    DefaultModelAdapter,
    ModelInfoInterface,
    ModelSlimPipelineInterfaceV0,
    ModelSlimPipelineInterfaceV1,
    IterSmoothInterface,
    FlexSmoothQuantInterface,
    AdaptRotationInterface,
):
    def get_model_type(self) -> str:
        return self.model_type

    def get_model_pedigree(self) -> str:
        return 'qwen3_moe'

    def get_hidden_dim(self):
        return self.config.hidden_size

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

    def get_adapter_config_for_subgraph(self) -> List[AdapterConfig]:
        adapter_config = []
        expert_num = getattr(self.config, 'num_experts', None)
        if expert_num is None:
            return adapter_config
        for layer_idx in range(self.config.num_hidden_layers):
            # Norm-Linear融合的映射配置：输入层归一化到QKV投影
            norm_linear_mapping_config = MappingConfig(
                source=f"model.layers.{layer_idx}.input_layernorm",  # 第一个LayerNorm
                targets=[
                    f"model.layers.{layer_idx}.self_attn.k_proj",
                    f"model.layers.{layer_idx}.self_attn.q_proj",
                    f"model.layers.{layer_idx}.self_attn.v_proj",
                ],  # 注意力层的QKV投影
            )

            # OV融合的映射配置（QKV到输出投影）
            ov_mapping_config = MappingConfig(
                source=f"model.layers.{layer_idx}.self_attn.v_proj",  # V投影层
                targets=[f"model.layers.{layer_idx}.self_attn.o_proj"],  # 输出投影层
            )

            # 为当前layer添加2个配置
            adapter_config.extend(
                [
                    AdapterConfig(subgraph_type="norm-linear", mapping=norm_linear_mapping_config),
                    AdapterConfig(subgraph_type="ov", mapping=ov_mapping_config, extra_config={'group_method': 'max'}),
                ]
            )

            # up-down 子图
            for expert in range(expert_num):
                up_down_mapping_config = MappingConfig(
                    source=f"model.layers.{layer_idx}.mlp.experts.{expert}.up_proj",  # 上投影层
                    targets=[f"model.layers.{layer_idx}.mlp.experts.{expert}.down_proj"],  # 下投影层
                )
                adapter_config.extend([AdapterConfig(subgraph_type="up-down", mapping=up_down_mapping_config)])

        return adapter_config

    def get_ln_fuse_map(self):
        return {}, qwen3_moe_get_ln_fuse_map(self.config)

    def get_bake_names(self):
        return [], []

    def get_rotate_map(self, block_size):
        pre_run, rot_pairs, _, _ = qwen3_moe_get_rotate_map(self.config, block_size)
        return [pre_run], list(rot_pairs.values())


def qwen3_moe_get_ln_fuse_map(config):
    # for quarot rotate interface
    ln_linear_map = {}
    for layer_idx in range(config.num_hidden_layers):
        ln_linear_map[f"model.layers.{layer_idx}.input_layernorm"] = [
            f"model.layers.{layer_idx}.self_attn.q_proj",
            f"model.layers.{layer_idx}.self_attn.k_proj",
            f"model.layers.{layer_idx}.self_attn.v_proj",
        ]

        # routed experts
        ln_linear_map[f"model.layers.{layer_idx}.post_attention_layernorm"] = [
            f"model.layers.{layer_idx}.mlp.experts.{i}.{proj}"
            for proj in ["gate_proj", "up_proj"]
            for i in range(config.num_experts)
        ]
        # expert gate
        ln_linear_map[f"model.layers.{layer_idx}.post_attention_layernorm"] += [f"model.layers.{layer_idx}.mlp.gate"]
    ln_linear_map["model.norm"] = ['lm_head']
    return ln_linear_map


def qwen3_moe_get_rotate_map(config, block_size):
    rot = QuaRotInterface.get_rotate_command(
        size=config.hidden_size,
        mode=QuaRotInterface.QuaRotMode.HADAMARD,
        block_size=block_size,
    )
    rot_uv = QuaRotInterface.get_rotate_command(
        size=config.head_dim,
        mode=QuaRotInterface.QuaRotMode.BLOCK_HADAMARD_SHIFTED,
        block_size=block_size,
    )
    # pre run
    left_rot = {}
    right_rot = {}
    # embedding weight is transposed, right is output channel
    right_rot["model.embed_tokens"] = rot
    pre_run = QuaRotInterface.RotatePair(left_rot=left_rot, right_rot=right_rot)
    rot_pairs = {}
    # rot
    left_rot = {}
    right_rot = {}
    right_rot["lm_head"] = rot
    for layer_idx in range(config.num_hidden_layers):
        right_rot[f"model.layers.{layer_idx}.self_attn.q_proj"] = rot
        right_rot[f"model.layers.{layer_idx}.self_attn.k_proj"] = rot
        right_rot[f"model.layers.{layer_idx}.self_attn.v_proj"] = rot
        left_rot[f"model.layers.{layer_idx}.self_attn.o_proj"] = rot
        # routed experts
        for i in range(config.num_experts):
            right_rot[f"model.layers.{layer_idx}.mlp.experts.{i}.gate_proj"] = rot
            right_rot[f"model.layers.{layer_idx}.mlp.experts.{i}.up_proj"] = rot
            left_rot[f"model.layers.{layer_idx}.mlp.experts.{i}.down_proj"] = rot
        # expert gate
        right_rot[f"model.layers.{layer_idx}.mlp.gate"] = rot
    rot_pairs['rot'] = QuaRotInterface.RotatePair(left_rot=left_rot, right_rot=right_rot)

    # rot_uv
    left_rot_uv = {}
    right_rot_uv = {}
    for layer_idx in range(config.num_hidden_layers):
        left_rot_uv[f"model.layers.{layer_idx}.self_attn.v_proj"] = rot_uv
        right_rot_uv[f"model.layers.{layer_idx}.self_attn.o_proj"] = rot_uv
    rot_pairs["rot_uv"] = QuaRotInterface.RotatePair(left_rot=left_rot_uv, right_rot=right_rot_uv)

    return pre_run, rot_pairs, rot, rot_uv
