#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
-------------------------------------------------------------------------
Copyright 2023 DeepSeek-AI and The HuggingFace Inc. team. All rights reserved.
Copyright (c) 2025 Huawei Technologies Co.,Ltd.

This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
and OPT implementations in this library. It has been modified from its
original forms to accommodate minor architectural differences compared
to GPT-NeoX and OPT used by the Meta AI team that trained the model.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-------------------------------------------------------------------------
"""

import os.path
from collections import defaultdict
from contextlib import contextmanager
from functools import lru_cache
from typing import Any, List, Optional, Tuple, Generator, Dict
from unittest.mock import patch

import torch
from safetensors import safe_open
from torch import distributed as dist
from torch import nn
from tqdm import tqdm

import msmodelslim.core.quantization_schema as qir
from msmodelslim.core.base.protocol import ProcessRequest
from msmodelslim.core.const import DeviceType
from msmodelslim.core.graph import AdapterConfig, MappingConfig, FusionConfig
from msmodelslim.model.common.layer_wise_forward import generated_decoder_layer_visit_func, TransformersForwardBreak
from msmodelslim.utils.exception import InvalidModelError
from msmodelslim.utils.logging import logger_setter, get_logger
from msmodelslim.utils.security import json_safe_load, json_safe_dump, get_valid_read_path, MAX_READ_FILE_SIZE_32G
from msmodelslim.utils.security.model import SafeGenerator
from .convert_fp8_to_bf16 import auto_convert_module_fp8_to_bf16
from .mtp_quant_module import get_mtp_layer, wrap_mtp_decoder
from .quarot import get_ln_fuse_map, get_rotate_map
from ..default.model_adapter import DefaultModelAdapter
from ..interface_hub import (
    ModelInfoInterface,
    ModelSlimPipelineInterfaceV1,
    IterSmoothInterface,
    FlexSmoothQuantInterface,
    QuaRotInterface,
    AscendV1SaveInterface,
)


@contextmanager
def default_dtype(dtype):
    """自定义默认 dtype 上下文管理器"""
    original_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(original_dtype)


@logger_setter("msmodelslim.model.kimi_k2")
class KimiK2ModelAdapter(  # pylint: disable=too-many-ancestors
    DefaultModelAdapter,
    ModelInfoInterface,  # support naive quantization
    ModelSlimPipelineInterfaceV1,  # support modelslim v1
    IterSmoothInterface,  # support iter smooth
    FlexSmoothQuantInterface,  # support flex smooth quant
    QuaRotInterface,
    AscendV1SaveInterface,
):
    def get_model_type(self) -> str:
        return self.model_type

    def get_model_pedigree(self) -> str:
        return 'kimi_k2'

    def init_model(self, device: DeviceType = DeviceType.NPU) -> nn.Module:
        with default_dtype(torch.bfloat16):
            # 保存原始层数
            origin_layers = self.config.num_hidden_layers
            get_logger().info("Model with %s layers totally", origin_layers)

            # 临时设置为1层进行初始化
            self.config.num_hidden_layers = 1

            # 加载只有一层的模型
            model = SafeGenerator.get_model_from_pretrained(
                model_path=str(self.model_path),
                config=self.config,
                trust_remote_code=self.trust_remote_code,
                device_map="cpu",
                torch_dtype="auto",
                attn_implementation='eager',
            )

            # 恢复原始层数
            self.config.num_hidden_layers = origin_layers

            # 加载权重
            state_dict = self.get_state_dict(model)
            model.load_state_dict(state_dict)

            # auto convert fp8 to bf16
            auto_convert_module_fp8_to_bf16("", model, str(self.model_path))
            model.eval()
            get_logger().info(
                "Create model with %s layers successfully at first",
                self.config.num_hidden_layers,
            )
            return model

    def handle_dataset(self, dataset: Any, device: DeviceType = DeviceType.NPU) -> List[Any]:
        return self._get_tokenized_data(dataset, device)

    def generate_model_visit(self, model: nn.Module) -> Generator[ProcessRequest, Any, None]:
        return generated_decoder_layer_visit_func(model, transformer_blocks=self.generate_decoder_layer(model))

    def generate_model_forward(
        self,
        model: nn.Module,
        inputs: Any,
    ) -> Generator[ProcessRequest, Any, None]:
        # 存储第一个transformer block的输入
        first_block_input: Optional[Tuple] = None

        def break_hook(module: nn.Module, hook_args: Tuple[Any, ...], hook_kwargs: Dict[str, Any]):
            nonlocal first_block_input
            first_block_input = (
                hook_args,
                hook_kwargs,
            )
            raise TransformersForwardBreak()

        remove_handler = model.model.layers[0].register_forward_pre_hook(break_hook, with_kwargs=True, prepend=True)

        # 执行一次前向传播以获取输入
        try:
            if isinstance(inputs, (list, tuple)):
                model(inputs[0])
            elif isinstance(inputs, dict):
                model(**inputs)
            else:
                model(inputs)
        except TransformersForwardBreak:
            pass
        finally:
            remove_handler.remove()

        if first_block_input is None:
            raise InvalidModelError("Can't get first block input.", action="Please check the model and input")

        # 循环处理每个transformer block
        current_inputs = first_block_input

        if dist.is_initialized():
            dist.barrier()

        for name, block in self.generate_decoder_layer(model):
            args, kwargs = current_inputs
            outputs = yield ProcessRequest(name, block, args, kwargs)
            hidden_states = outputs[0]
            current_inputs = ((hidden_states,), current_inputs[1])

    def enable_kv_cache(self, model: nn.Module, need_kv_cache: bool) -> None:
        def pre_forward_hook(module, args, kwargs):
            kwargs['use_cache'] = need_kv_cache
            return args, kwargs

        model.model.register_forward_pre_hook(pre_forward_hook, with_kwargs=True)

    def get_adapter_config_for_subgraph(self) -> List[AdapterConfig]:
        adapter_config = []
        # mtp layer does not apply smooth due to the compatible with pre-refactor
        for layer_idx in range(self.config.num_hidden_layers - 1):
            # OKV_b融合的映射配置：o_proj -> kv_b_proj
            okv_b_mapping_config = MappingConfig(
                source=f"model.layers.{layer_idx}.self_attn.kv_b_proj",  # KV_b投影层
                targets=[f"model.layers.{layer_idx}.self_attn.o_proj"],  # 输出投影层
            )

            # Norm-Linear融合的映射配置1：q_a_proj, kv_a_proj_with_mqa -> input_layernorm
            norm_linear_mapping_config1 = MappingConfig(
                source=f"model.layers.{layer_idx}.input_layernorm",  # 第一个LayerNorm
                targets=[
                    f"model.layers.{layer_idx}.self_attn.q_a_proj",
                    f"model.layers.{layer_idx}.self_attn.kv_a_proj_with_mqa",
                ],  # 注意力层的Q_a,KV_a投影
            )

            # Norm-Linear融合的映射配置2：q_b_proj -> q_a_layernorm
            norm_linear_mapping_config2 = MappingConfig(
                source=f"model.layers.{layer_idx}.self_attn.q_a_layernorm",  # q_a_layernorm
                targets=[f"model.layers.{layer_idx}.self_attn.q_b_proj"],  # q_b投影
            )

            # 为当前layer添加4个配置
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
                    AdapterConfig(subgraph_type="norm-linear", mapping=norm_linear_mapping_config1),
                    AdapterConfig(subgraph_type="norm-linear", mapping=norm_linear_mapping_config2),
                ]
            )

        return adapter_config

    def get_ln_fuse_map(self):
        return {}, get_ln_fuse_map(self.config, num_hidden_layers=self.config.num_hidden_layers)

    def get_bake_names(self):
        return [], []

    def get_rotate_map(self, block_size):
        pre_run, rot_pairs, _ = get_rotate_map(self.config, block_size, num_hidden_layers=self.config.num_hidden_layers)
        return [pre_run], list(rot_pairs.values())

    @lru_cache(maxsize=1)
    def get_weight_map(self):
        model_index_path = os.path.join(self.model_path, "model.safetensors.index.json")
        model_index = json_safe_load(model_index_path)
        weight_map = model_index['weight_map']
        return weight_map

    def get_state_dict(self, module: nn.Module, prefix: str = ""):
        weight_map = self.get_weight_map()
        names = map(lambda x: x[0], module.named_parameters())

        groups = defaultdict(list)
        for name in names:
            file_name = weight_map[f'{prefix}.{name}' if prefix else name]
            groups[file_name].append(name)

        state_dict = {}
        for file_name in tqdm(groups, desc=f'Loading {prefix}'):
            file_path = os.path.join(self.model_path, file_name)
            file_path = get_valid_read_path(file_path, extensions='safetensors', size_max=MAX_READ_FILE_SIZE_32G)
            with safe_open(file_path, framework='pt', device='cpu') as f:
                for name in tqdm(groups[file_name], desc=f'Loading {file_path}'):
                    state_dict[name] = f.get_tensor(f'{prefix}.{name}' if prefix else name)
        return state_dict

    def load_decoder_if_not_exist(self, model: nn.Module, name: str, idx: int):
        try:
            decoder = model.get_submodule(name)
        except AttributeError:
            # disable reset_parameters so that the weights will not be initialized
            # these initializations is not necessary because we will load it from the state_dict
            # and these initializations will cost too much time because the DeepSeekV3's decoder layer is too large
            with patch.object(nn.Linear, 'reset_parameters', lambda _self: None), default_dtype(torch.bfloat16):
                get_logger().info('Creating decoder layer %s', idx)
                module_list: nn.ModuleList = model.model.layers
                template_module = module_list[0]
                decoder = template_module.__class__(layer_idx=idx, config=self.config)

                state_dict = self.get_state_dict(decoder, prefix=name)
                decoder.load_state_dict(state_dict)
                auto_convert_module_fp8_to_bf16(name, decoder, str(self.model_path))
                decoder.eval()
                module_list.append(decoder)
                get_logger().info('Create decoder layer %s successfully', idx)
        return decoder

    def load_mtp_if_not_load(self, mtp_decoder: nn.Module):
        try:
            mtp_decoder.get_submodule('shared_head')
        except AttributeError:
            get_logger().info('Creating MTP layer')
            mtp_layer = get_mtp_layer(config=self.config, model_path=self.model_path)
            wrap_mtp_decoder(mtp_decoder=mtp_decoder, mtp_layer=mtp_layer)
            get_logger().info('Create MTP successfully')

    def generate_decoder_layer(self, model: nn.Module):
        for idx in range(self.config.num_hidden_layers):
            name = f"model.layers.{idx}"
            decoder = self.load_decoder_if_not_exist(model, name=name, idx=idx)
            # if idx == self.config.num_hidden_layers - 1:
            #     self.load_mtp_if_not_load(decoder)
            yield name, decoder

    def ascendv1_save_postprocess(self, model: nn.Module, save_directory: str) -> None:
        """
        根据 MideIE 要求在deepseek w4a8和w4a8c8场景下，config.json 中添加以下字段
        - mtp_quantize: w8a8_dynamic
        - quantize: w8a8_dynamic
        - moe_quantize: w4a8_dynamic
        - mla_quantize: w8a8(w4a8c8场景下使用w8a8) or w8a8_dynamic(w4a8场景下使用w8a8_dynamic)

        Args:
            model: 模型
            save_directory: 导出件的保存目录
        """
        use_w4a8 = False
        use_c8 = False

        for _, module in model.named_modules():
            if isinstance(module, qir.W4A8DynamicFakeQuantLinear):
                use_w4a8 = True
            if isinstance(module, qir.FakeQuantActivationPerHead):
                use_c8 = True
            if use_w4a8 and use_c8:
                break

        if use_w4a8:
            config_file = os.path.join(save_directory, "config.json")
            config_data = json_safe_load(config_file, check_user_stat=False)

            # config_data["mtp_quantize"] = "w8a8_dynamic"
            config_data["quantize"] = "w8a8_dynamic"
            config_data["moe_quantize"] = "w4a8_dynamic"
            if use_c8:
                config_data["mla_quantize"] = "w8a8"
            else:
                config_data["mla_quantize"] = "w8a8_dynamic"

            json_safe_dump(config_data, config_file, indent=2, check_user_stat=False)
