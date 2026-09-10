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

from typing import List, Any, Generator, Dict

import torch
from torch import nn

from msmodelslim.core.base.protocol import ProcessRequest
from msmodelslim.core.const import DeviceType
from msmodelslim.core.graph.adapter_types import AdapterConfig, MappingConfig
from msmodelslim.processor.anti_outlier.awq.interface import AWQInterface
from msmodelslim.processor.kv_smooth import KVSmoothFusedType, KVSmoothFusedUnit
from msmodelslim.processor.quarot import QuaRotInterface, LAOSOnlineRotationInterface
from msmodelslim.processor.quant.fa3.interface import FA3QuantAdapterInterface, FA3QuantPlaceHolder
from msmodelslim.utils.exception import InvalidModelError
from msmodelslim.utils.logging import logger_setter, get_logger
from ..common.layer_wise_forward import generated_decoder_layer_visit_func, transformers_generated_forward_func
from ..default.model_adapter import DefaultModelAdapter
from ..interface_hub import (
    ModelInfoInterface,
    ModelSlimPipelineInterfaceV0,
    ModelSlimPipelineInterfaceV1,
    StandingHighWithExperienceInterface,
    KVSmoothFusedInterface,
    SmoothQuantInterface,
    IterSmoothInterface,
    FlexSmoothQuantInterface,
    AdaptRotationInterface,
    FakeQuantInferenceInterface,
)

from msmodelslim.processor.flat_quant import FlatQuantInterface
from msmodelslim.processor.flat_quant.flat_quant_utils.structure_pair import (
    AttnNormLinearPair,
    AttnLinearLinearPair,
    MLPNormLinearPair,
    MLPLinearLinearPair,
)


@logger_setter()
class Qwen3ModelAdapter(  # pylint: disable=too-many-ancestors
    DefaultModelAdapter,
    ModelInfoInterface,
    ModelSlimPipelineInterfaceV0,
    ModelSlimPipelineInterfaceV1,
    StandingHighWithExperienceInterface,
    KVSmoothFusedInterface,
    SmoothQuantInterface,
    IterSmoothInterface,
    FlexSmoothQuantInterface,
    AdaptRotationInterface,
    LAOSOnlineRotationInterface,
    FlatQuantInterface,
    AWQInterface,
    FakeQuantInferenceInterface,
    FA3QuantAdapterInterface,
):
    def get_flatquant_subgraph(self) -> List[Dict[str, object]]:  # pylint: disable=arguments-differ
        """分析Qwen模型结构并注册所有相关的结构对。"""
        attn_norm_linear_names = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"]
        attn_linear_linear_names = ["self_attn.o_proj"]
        mlp_norm_linear_names = ["mlp.gate_proj", "mlp.up_proj"]
        mlp_linear_linear_names = ["mlp.down_proj"]
        head_dim = getattr(self.config, "head_dim", self.config.hidden_size // self.config.num_attention_heads)
        num_attention_heads = self.config.num_attention_heads
        structure_configs = [
            {"source": "input_layernorm", "targets": attn_norm_linear_names, "pair_class": AttnNormLinearPair},
            {
                "source": "self_attn.v_proj",
                "targets": attn_linear_linear_names,
                "pair_class": AttnLinearLinearPair,
                "extra_config": {'head_dim': head_dim, 'num_attention_heads': num_attention_heads},
            },
            {"source": "post_attention_layernorm", "targets": mlp_norm_linear_names, "pair_class": MLPNormLinearPair},
            {"source": "mlp.up_proj", "targets": mlp_linear_linear_names, "pair_class": MLPLinearLinearPair},
        ]
        return structure_configs

    def get_model_type(self) -> str:
        return self.model_type

    def get_model_pedigree(self) -> str:
        return 'qwen3'

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

    # ===== FA3QuantAdapterInterface =====
    def inject_fa3_placeholders(
        self,
        root_name: str,
        root_module: nn.Module,
        should_inject,
    ) -> None:
        """Inject fa_q / fa_k / fa_v placeholders into Qwen3 attention modules and wrap forward.

        For FA3 activation quantization, the placeholders are later replaced by
        FakeQuantActivation IR modules.  For KV-cache quantization, fa_k / fa_v are
        replaced by FakeQuantDynamicCache.  fa_q remains a passthrough placeholder
        when only KV-cache quantization is active.

        The wrapped forward replicates ``Qwen3Attention.forward`` exactly, inserting
        ``self.fa_q / fa_k / fa_v`` calls after RoPE and (optional) cache update,
        before the attention interface — the correct point for both FA3 activation
        and KV-cache fake quantization.
        """
        from transformers.models.qwen3.modeling_qwen3 import (
            apply_rotary_pos_emb,
            eager_attention_forward,
        )
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

        def _wrap_attention_forward(attn_mod: nn.Module):
            def new_forward(
                self,
                hidden_states: torch.Tensor,
                position_embeddings,
                attention_mask,
                past_key_value=None,
                cache_position=None,
                **kwargs,
            ):
                input_shape = hidden_states.shape[:-1]
                hidden_shape = (*input_shape, -1, self.head_dim)

                query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
                key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
                value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

                cos, sin = position_embeddings
                query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

                if past_key_value is not None:
                    cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
                    key_states, value_states = past_key_value.update(
                        key_states, value_states, self.layer_idx, cache_kwargs
                    )

                # ===== fa_q / fa_k / fa_v placeholder calls =====
                # fa_k / fa_v may be replaced by FakeQuantDynamicCache for KV-cache
                # quantization, or by FakeQuantActivation for FA3 activation quant.
                # fa_q stays as FA3QuantPlaceHolder (passthrough) when only KV-cache.
                if hasattr(self, "fa_q"):
                    query_states = self.fa_q(query_states)
                if hasattr(self, "fa_k"):
                    key_states = self.fa_k(key_states)
                if hasattr(self, "fa_v"):
                    value_states = self.fa_v(value_states)
                # ================================================

                # Dispatch attention interface the same way as the original
                # ``Qwen3Attention.forward`` so that ``sdpa`` (default) and
                # ``flash_attention_2`` keep working.  Hardcoding eager here would
                # break when ``_update_causal_mask`` returns ``None`` for sdpa.
                attention_interface = eager_attention_forward
                if self.config._attn_implementation != "eager":
                    attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

                attn_output, attn_weights = attention_interface(
                    self,
                    query_states,
                    key_states,
                    value_states,
                    attention_mask,
                    scaling=self.scaling,
                    dropout=0.0 if not self.training else self.attention_dropout,
                    sliding_window=self.sliding_window,
                    **kwargs,
                )

                attn_output = attn_output.reshape(*input_shape, -1).contiguous()
                attn_output = self.o_proj(attn_output)
                return attn_output, attn_weights

            # pylint: disable=no-value-for-parameter
            attn_mod.forward = new_forward.__get__(attn_mod, attn_mod.__class__)

        for name, module in root_module.named_modules():
            if "Attention" not in module.__class__.__name__:
                continue
            full_name = f"{root_name}.{name}" if root_name else name
            if not should_inject(full_name):
                continue
            root_module.set_submodule(f"{name}.fa_q", FA3QuantPlaceHolder(ratio=0.9999))
            root_module.set_submodule(f"{name}.fa_k", FA3QuantPlaceHolder(ratio=0.9999))
            root_module.set_submodule(f"{name}.fa_v", FA3QuantPlaceHolder(ratio=1.0))
            _wrap_attention_forward(module)

    def get_kvcache_smooth_fused_subgraph(self) -> List[KVSmoothFusedUnit]:
        return [
            KVSmoothFusedUnit(
                attention_name=f"model.layers.{i}.self_attn",
                layer_idx=i,
                fused_from_query_states_name="q_norm",
                fused_from_key_states_name="k_norm",
                fused_type=KVSmoothFusedType.StateViaRopeToNorm,
            )
            for i in range(self.config.num_hidden_layers)
        ]

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

    def get_num_key_value_groups(self) -> int:
        if not hasattr(self.config, 'num_attention_heads'):
            raise InvalidModelError(
                "num_attention_heads is not found in config.json",
                action=f"Please check config.json in {self.model_path}",
            )
        if not hasattr(self.config, 'num_key_value_heads'):
            raise InvalidModelError(
                "num_key_value_heads is not found in config.json",
                action=f"Please check config.json in {self.model_path}",
            )
        if self.config.num_key_value_heads == 0:
            raise InvalidModelError(
                "num_key_value_heads is 0 in config.json, which should be greater than 0",
                action=f"Please check config.json in {self.model_path}",
            )
        return self.config.num_attention_heads // self.config.num_key_value_heads

    def get_num_key_value_heads(self) -> int:
        if not hasattr(self.config, 'num_key_value_heads'):
            raise InvalidModelError(
                "num_key_value_heads is not found in config.json",
                action=f"Please check config.json in {self.model_path}",
            )
        return self.config.num_key_value_heads

    def get_adapter_config_for_subgraph(self) -> List[AdapterConfig]:
        adapter_config = []
        for layer_idx in range(self.config.num_hidden_layers):
            # Norm-Linear的映射配置1：输入层归一化到QKV投影
            norm_linear_mapping_config1 = MappingConfig(
                source=f"model.layers.{layer_idx}.input_layernorm",  # 第一个LayerNorm
                targets=[
                    f"model.layers.{layer_idx}.self_attn.k_proj",
                    f"model.layers.{layer_idx}.self_attn.q_proj",
                    f"model.layers.{layer_idx}.self_attn.v_proj",
                ],  # 注意力层的QKV投影
            )

            # Norm-Linear的映射配置2：后注意力层归一化到MLP投影
            norm_linear_mapping_config2 = MappingConfig(
                source=f"model.layers.{layer_idx}.post_attention_layernorm",  # 第二个LayerNorm
                targets=[
                    f"model.layers.{layer_idx}.mlp.gate_proj",
                    f"model.layers.{layer_idx}.mlp.up_proj",
                ],  # MLP层的门控和上投影
            )

            # OV的映射配置（QKV到输出投影）
            ov_mapping_config = MappingConfig(
                source=f"model.layers.{layer_idx}.self_attn.v_proj",  # V投影层
                targets=[f"model.layers.{layer_idx}.self_attn.o_proj"],  # 输出投影层
            )

            # Up-Down的映射配置
            up_down_mapping_config = MappingConfig(
                source=f"model.layers.{layer_idx}.mlp.up_proj",  # 上投影层
                targets=[f"model.layers.{layer_idx}.mlp.down_proj"],  # 下投影层
            )

            # 为当前layer添加4个配置
            adapter_config.extend(
                [
                    AdapterConfig(subgraph_type="norm-linear", mapping=norm_linear_mapping_config1),
                    AdapterConfig(subgraph_type="norm-linear", mapping=norm_linear_mapping_config2),
                    AdapterConfig(subgraph_type="ov", mapping=ov_mapping_config, extra_config={'group_method': 'max'}),
                    AdapterConfig(subgraph_type="up-down", mapping=up_down_mapping_config),
                ]
            )
        return adapter_config

    def get_hidden_dim(self):
        return self.config.hidden_size

    def get_num_attention_heads(self):
        return self.config.num_attention_heads

    def get_lm_head(self) -> str:
        return "lm_head"

    def get_pre_head_layernorm(self) -> str:
        return "model.norm"

    def build_meta_model(self) -> nn.Module:
        """Build a full-depth Qwen3 meta CausalLM skeleton for fake-quant.

        Parameters live on meta; non-persistent buffers (e.g. RoPE ``inv_freq``) stay on CPU
        so they can be snapshotted and restored after decoder ``.to(meta)``.
        Weights are filled later by AscendV1 hydrate.
        """
        try:
            from transformers import AutoModelForCausalLM
        except ImportError as exc:
            raise InvalidModelError(
                "Failed to import AutoModelForCausalLM for fake-quant inference shell",
                action="Please install a transformers version that provides AutoModelForCausalLM.",
            ) from exc

        origin_layers = int(self.config.num_hidden_layers)
        if hasattr(self.config, "use_cache"):
            self.config.use_cache = False
        from accelerate import init_empty_weights

        with init_empty_weights(include_buffers=False):
            model = AutoModelForCausalLM.from_config(
                self.config,
                trust_remote_code=self.trust_remote_code,
            )

        get_logger().info(
            "Built Qwen3 fake-quant inference shell: %d decoder layers (meta params, CPU non-persistent buffers)",
            origin_layers,
        )
        return model

    def get_embedding(self) -> str:
        return "model.embed_tokens"

    def get_layer_wise_norm_liner_pair(self, decoder_module: nn.Module):
        norm_linear_pairs = {
            decoder_module.input_layernorm: [
                decoder_module.self_attn.q_proj,
                decoder_module.self_attn.k_proj,
                decoder_module.self_attn.v_proj,
            ],
            decoder_module.post_attention_layernorm: [decoder_module.mlp.gate_proj, decoder_module.mlp.up_proj],
        }
        return norm_linear_pairs

    def get_layer_wise_ov_pair(self, decoder_module: nn.Module):
        ov_pairs = {decoder_module.self_attn.o_proj: decoder_module.self_attn.v_proj}
        return ov_pairs

    def get_layer_wise_up_down_pair(self, decoder_module: nn.Module):
        up_down_pairs = {decoder_module.mlp.up_proj: decoder_module.mlp.down_proj}
        return up_down_pairs

    def get_ln_fuse_map(self):
        return {}, qwen3_get_ln_fuse_map(self.config)

    def get_bake_names(self):
        return [], []

    def get_rotate_map(self, block_size):
        pre_run, rot_pairs, _, _ = qwen3_get_rotate_map(self.config, block_size)
        return [pre_run], list(rot_pairs.values())


def qwen3_get_ln_fuse_map(config):
    # for quarot rotate interface
    ln_linear_map = {}
    for layer_idx in range(config.num_hidden_layers):
        ln_linear_map[f"model.layers.{layer_idx}.input_layernorm"] = [
            f"model.layers.{layer_idx}.self_attn.q_proj",
            f"model.layers.{layer_idx}.self_attn.k_proj",
            f"model.layers.{layer_idx}.self_attn.v_proj",
        ]

        # mlp
        ln_linear_map[f"model.layers.{layer_idx}.post_attention_layernorm"] = [
            f"model.layers.{layer_idx}.mlp.{proj}" for proj in ["gate_proj", "up_proj"]
        ]
    ln_linear_map["model.norm"] = ['lm_head']
    return ln_linear_map


def qwen3_get_rotate_map(config, block_size):
    rot = QuaRotInterface.get_rotate_command(
        size=config.hidden_size,
        block_size=block_size,
        mode=QuaRotInterface.QuaRotMode.HADAMARD,
    )
    rot_uv = QuaRotInterface.get_rotate_command(
        size=config.head_dim,
        block_size=block_size,
        mode=QuaRotInterface.QuaRotMode.HADAMARD,
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
        # mlp
        right_rot[f"model.layers.{layer_idx}.mlp.gate_proj"] = rot
        right_rot[f"model.layers.{layer_idx}.mlp.up_proj"] = rot
        left_rot[f"model.layers.{layer_idx}.mlp.down_proj"] = rot
    rot_pairs['rot'] = QuaRotInterface.RotatePair(left_rot=left_rot, right_rot=right_rot)

    # rot_uv
    left_rot_uv = {}
    right_rot_uv = {}
    for layer_idx in range(config.num_hidden_layers):
        left_rot_uv[f"model.layers.{layer_idx}.self_attn.v_proj"] = rot_uv
        right_rot_uv[f"model.layers.{layer_idx}.self_attn.o_proj"] = rot_uv
    rot_pairs["rot_uv"] = QuaRotInterface.RotatePair(left_rot=left_rot_uv, right_rot=right_rot_uv)

    return pre_run, rot_pairs, rot, rot_uv
