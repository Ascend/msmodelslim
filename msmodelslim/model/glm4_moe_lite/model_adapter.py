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

import torch
from torch import nn
from transformers import PreTrainedTokenizerBase

from msmodelslim.core.const import DeviceType
from msmodelslim.core.base.protocol import ProcessRequest
from msmodelslim.core.graph import AdapterConfig, MappingConfig, FusionConfig
from msmodelslim.processor.quarot import QuaRotInterface
from msmodelslim.model.interface_hub import (
    ModelSlimPipelineInterfaceV1,
    ModelInfoInterface,
    IterSmoothInterface,
    FlexSmoothQuantInterface,
)
from msmodelslim.model.common.layer_wise_forward import (
    generated_decoder_layer_visit_func,
    transformers_generated_forward_func,
)
from msmodelslim.model.common.transformers import TransformersModel
from msmodelslim.model.common.utils import _get_expert_range
from msmodelslim.utils.logging import logger_setter, get_logger
from msmodelslim.utils.security.model import SafeGenerator

from .moe_utils import UnstackedGlm4MoeLiteMoE


# 当前支持的量化 / 离群值抑制算法：
#   - QuaRot (QuaRotInterface)         : 旋转量化预处理，平滑激活值分布
#   - SmoothQuant (IterSmoothInterface)  : 迭代平滑量化
#   - FlexSmoothQuant (FlexSmoothQuantInterface) : 灵活平滑量化，支持 norm-linear / ov 子图
#   - LinearQuant (ModelSlimPipelineInterfaceV1) : 线性量化 (W8A8 / W4A8 等)，由 YAML pipeline 指定
#
# 如需新增 / 移除算法支持，在此增减对应的 Interface 继承。
@logger_setter()
class GLM4MoeLiteFlashModelAdapter(  # pylint: disable=too-many-ancestors
    TransformersModel,
    ModelSlimPipelineInterfaceV1,
    ModelInfoInterface,
    IterSmoothInterface,
    FlexSmoothQuantInterface,
    QuaRotInterface,
):
    def get_model_type(self) -> str:
        return self.model_type

    def get_model_pedigree(self) -> str:
        return 'glm4_moe_lite'

    def handle_dataset(self, dataset: Any, device: DeviceType = DeviceType.NPU) -> List[Any]:
        return self._get_tokenized_data(dataset, device)

    def init_model(self, device: DeviceType = DeviceType.NPU) -> nn.Module:
        model = self._load_model(device)
        self._unstack_moe_layers(model)
        return model

    def _is_moe_layer(self, layer_idx: int) -> bool:
        return layer_idx >= self.config.first_k_dense_replace

    def _unstack_moe_layers(self, model: nn.Module) -> None:
        for layer_idx in range(self.config.num_hidden_layers):
            if not self._is_moe_layer(layer_idx):
                continue
            layer = model.model.layers[layer_idx]
            mlp = layer.mlp
            if hasattr(mlp, 'experts') and hasattr(mlp.experts, 'gate_up_proj'):
                get_logger().info("Unstacking MoE layer %s...", layer_idx)
                layer.mlp = UnstackedGlm4MoeLiteMoE(self.config, mlp)

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
        expert_start, expert_end = _get_expert_range(self.config)

        for layer_idx in range(self.config.num_hidden_layers):
            okv_b_mapping_config = MappingConfig(
                source=f"model.layers.{layer_idx}.self_attn.kv_b_proj",
                targets=[f"model.layers.{layer_idx}.self_attn.o_proj"],
            )

            input_norm_mapping_config = MappingConfig(
                source=f"model.layers.{layer_idx}.input_layernorm",
                targets=[
                    f"model.layers.{layer_idx}.self_attn.q_a_proj",
                    f"model.layers.{layer_idx}.self_attn.kv_a_proj_with_mqa",
                ],
            )

            qa_norm_mapping_config = MappingConfig(
                source=f"model.layers.{layer_idx}.self_attn.q_a_layernorm",
                targets=[f"model.layers.{layer_idx}.self_attn.q_b_proj"],
            )

            adapter_config.extend(
                [
                    AdapterConfig(
                        subgraph_type="ov",
                        mapping=okv_b_mapping_config,
                        extra_config={'group_method': 'max'},
                        fusion=FusionConfig(
                            fusion_type="kv",
                            num_attention_heads=self.config.num_attention_heads,
                            num_key_value_heads=self.config.num_key_value_heads,
                            custom_config={
                                'qk_nope_head_dim': self.config.qk_nope_head_dim,
                                'v_head_dim': self.config.v_head_dim,
                            },
                        ),
                    ),
                    AdapterConfig(subgraph_type="norm-linear", mapping=input_norm_mapping_config),
                    AdapterConfig(subgraph_type="norm-linear", mapping=qa_norm_mapping_config),
                ]
            )

            if layer_idx < self.config.first_k_dense_replace:
                up_proj = f'model.layers.{layer_idx}.mlp.up_proj'
                down_proj = f'model.layers.{layer_idx}.mlp.down_proj'
                adapter_config.append(
                    AdapterConfig(
                        subgraph_type="up-down",
                        mapping=MappingConfig(source=up_proj, targets=[down_proj]),
                    )
                )
            else:
                shared_up = f'model.layers.{layer_idx}.mlp.shared_experts.up_proj'
                shared_down = f'model.layers.{layer_idx}.mlp.shared_experts.down_proj'
                adapter_config.append(
                    AdapterConfig(
                        subgraph_type="up-down",
                        mapping=MappingConfig(source=shared_up, targets=[shared_down]),
                    )
                )

                for expert in range(expert_start, expert_end):
                    up_proj = f'model.layers.{layer_idx}.mlp.experts.{expert}.up_proj'
                    down_proj = f'model.layers.{layer_idx}.mlp.experts.{expert}.down_proj'
                    adapter_config.append(
                        AdapterConfig(
                            subgraph_type="up-down",
                            mapping=MappingConfig(source=up_proj, targets=[down_proj]),
                        )
                    )

        return adapter_config

    def get_ln_fuse_map(self):
        return {}, _get_ln_fuse_map(self.config)

    def get_bake_names(self):
        return [], _get_bake_names(self.config)

    def get_rotate_map(self, block_size):
        pre_run, rot_pairs, _ = _get_rotate_map(self.config, block_size)
        return [pre_run], list(rot_pairs.values())

    def _load_tokenizer(self, trust_remote_code=False) -> PreTrainedTokenizerBase:
        return SafeGenerator.get_tokenizer_from_pretrained(
            model_path=str(self.model_path), legacy=False, trust_remote_code=trust_remote_code
        )


def _get_bake_names(config):
    return []


def _get_ln_fuse_map(config):
    ln_linear_map = {}
    expert_start, expert_end = _get_expert_range(config)

    for layer_idx in range(config.num_hidden_layers):
        ln_linear_map[f"model.layers.{layer_idx}.input_layernorm"] = [
            f"model.layers.{layer_idx}.self_attn.q_a_proj",
            f"model.layers.{layer_idx}.self_attn.kv_a_proj_with_mqa",
        ]
        ln_linear_map[f"model.layers.{layer_idx}.self_attn.q_a_layernorm"] = [
            f"model.layers.{layer_idx}.self_attn.q_b_proj",
        ]
        ln_linear_map[f"model.layers.{layer_idx}.self_attn.kv_a_layernorm"] = [
            f"model.layers.{layer_idx}.self_attn.kv_b_proj",
        ]
        if layer_idx < config.first_k_dense_replace:
            ln_linear_map[f"model.layers.{layer_idx}.post_attention_layernorm"] = [
                f"model.layers.{layer_idx}.mlp.gate_proj",
                f"model.layers.{layer_idx}.mlp.up_proj",
            ]
        else:
            ln_linear_map[f"model.layers.{layer_idx}.post_attention_layernorm"] = [
                f"model.layers.{layer_idx}.mlp.experts.{i}.{proj}"
                for proj in ["gate_proj", "up_proj"]
                for i in range(expert_start, expert_end)
            ]
            ln_linear_map[f"model.layers.{layer_idx}.post_attention_layernorm"] += [
                f"model.layers.{layer_idx}.mlp.shared_experts.{proj}" for proj in ["gate_proj", "up_proj"]
            ]
            ln_linear_map[f"model.layers.{layer_idx}.post_attention_layernorm"] += [
                f"model.layers.{layer_idx}.mlp.gate",
            ]
    ln_linear_map["model.norm"] = ['lm_head']

    return ln_linear_map


def _get_rotate_map(config, block_size):
    rot = QuaRotInterface.get_rotate_command(
        size=config.hidden_size,
        mode=QuaRotInterface.QuaRotMode.HADAMARD,
        block_size=block_size,
    )
    rot_b_proj = QuaRotInterface.get_rotate_command(
        size=config.q_lora_rank,
        mode=QuaRotInterface.QuaRotMode.BLOCK_HADAMARD_SHIFTED,
        block_size=block_size,
    )
    rot_uv = QuaRotInterface.get_rotate_command(
        size=config.v_head_dim,
        mode=QuaRotInterface.QuaRotMode.HADAMARD,
        block_size=block_size,
    )
    rot_kv_b_proj = QuaRotInterface.get_rotate_command(
        size=config.kv_lora_rank,
        mode=QuaRotInterface.QuaRotMode.HADAMARD,
        block_size=block_size,
    )

    left_rot = {}
    right_rot = {}
    right_rot["model.embed_tokens"] = rot
    pre_run = QuaRotInterface.RotatePair(left_rot=left_rot, right_rot=right_rot)

    rot_pairs = {}
    expert_start, expert_end = _get_expert_range(config)

    left_rot = {}
    right_rot = {}
    right_rot["lm_head"] = rot
    for layer_idx in range(config.num_hidden_layers):
        right_rot[f"model.layers.{layer_idx}.self_attn.q_a_proj"] = rot
        right_rot[f"model.layers.{layer_idx}.self_attn.kv_a_proj_with_mqa"] = rot
        left_rot[f"model.layers.{layer_idx}.self_attn.o_proj"] = rot
        if layer_idx < config.first_k_dense_replace:
            right_rot[f"model.layers.{layer_idx}.mlp.gate_proj"] = rot
            right_rot[f"model.layers.{layer_idx}.mlp.up_proj"] = rot
            left_rot[f"model.layers.{layer_idx}.mlp.down_proj"] = rot
        else:
            for i in range(expert_start, expert_end):
                right_rot[f"model.layers.{layer_idx}.mlp.experts.{i}.gate_proj"] = rot
                right_rot[f"model.layers.{layer_idx}.mlp.experts.{i}.up_proj"] = rot
                left_rot[f"model.layers.{layer_idx}.mlp.experts.{i}.down_proj"] = rot
            right_rot[f"model.layers.{layer_idx}.mlp.shared_experts.gate_proj"] = rot
            right_rot[f"model.layers.{layer_idx}.mlp.shared_experts.up_proj"] = rot
            left_rot[f"model.layers.{layer_idx}.mlp.shared_experts.down_proj"] = rot
            right_rot[f"model.layers.{layer_idx}.mlp.gate"] = rot
    rot_pairs['rot'] = QuaRotInterface.RotatePair(left_rot=left_rot, right_rot=right_rot)

    left_rot_b_proj = {}
    right_rot_b_proj = {}
    for layer_idx in range(config.num_hidden_layers):
        left_rot_b_proj[f"model.layers.{layer_idx}.self_attn.q_a_proj"] = rot_b_proj
        right_rot_b_proj[f"model.layers.{layer_idx}.self_attn.q_b_proj"] = rot_b_proj
    rot_pairs["rot_b_proj"] = QuaRotInterface.RotatePair(left_rot=left_rot_b_proj, right_rot=right_rot_b_proj)

    left_rot_uv = {}
    right_rot_uv = {}
    for layer_idx in range(config.num_hidden_layers):
        left_rot_uv[f"model.layers.{layer_idx}.self_attn.kv_b_proj"] = [
            torch.eye(config.qk_nope_head_dim, dtype=rot_uv.dtype, device=rot_uv.device),
            rot_uv,
        ]
        right_rot_uv[f"model.layers.{layer_idx}.self_attn.o_proj"] = rot_uv
    rot_pairs["rot_uv"] = QuaRotInterface.RotatePair(left_rot=left_rot_uv, right_rot=right_rot_uv)

    left_rot_kv_b_proj = {}
    right_rot_kv_b_proj = {}
    for layer_idx in range(config.num_hidden_layers):
        left_rot_kv_b_proj[f"model.layers.{layer_idx}.self_attn.kv_a_proj_with_mqa"] = [
            rot_kv_b_proj,
            torch.eye(config.qk_rope_head_dim, dtype=rot_kv_b_proj.dtype, device=rot_kv_b_proj.device),
        ]
        right_rot_kv_b_proj[f"model.layers.{layer_idx}.self_attn.kv_b_proj"] = rot_kv_b_proj
    rot_pairs["rot_kv_b_proj"] = QuaRotInterface.RotatePair(left_rot=left_rot_kv_b_proj, right_rot=right_rot_kv_b_proj)

    rotate_matrix = {
        'rot': rot,
        'rot_b_proj': rot_b_proj,
        'rot_uv': rot_uv,
        'rot_kv_b_proj': rot_kv_b_proj,
    }
    return pre_run, rot_pairs, rotate_matrix
