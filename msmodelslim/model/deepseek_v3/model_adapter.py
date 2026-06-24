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
from importlib import import_module
from typing import Any, List, Optional, Tuple, Generator, Dict, Union, Callable
from unittest.mock import patch

import torch
from torch import nn
from torch import distributed as dist
from safetensors import safe_open
from tqdm import tqdm

from msmodelslim import ir as qir
from msmodelslim.core.base.protocol import ProcessRequest
from msmodelslim.core.const import DeviceType
from msmodelslim.core.graph import AdapterConfig, MappingConfig, FusionConfig
from msmodelslim.model.common.layer_wise_forward import generated_decoder_layer_visit_func, TransformersForwardBreak
from msmodelslim.model.common.utils import _get_expert_range
from msmodelslim.utils.exception import InvalidModelError
from msmodelslim.utils.logging import logger_setter, get_logger
from msmodelslim.utils.security import json_safe_load, json_safe_dump, get_valid_read_path, MAX_READ_FILE_SIZE_32G
from msmodelslim.utils.security.model import SafeGenerator
from .convert_fp8_to_bf16 import auto_convert_module_fp8_to_bf16
from .mtp_quant_module import remove_zero_and_shift, get_mtp_layer, wrap_mtp_decoder, get_mtp_position_ids
from .quarot import get_ln_fuse_map, get_rotate_map
from ..default.model_adapter import DefaultModelAdapter
from ..interface_hub import (
    ModelInfoInterface,
    ModelSlimPipelineInterfaceV1,
    IterSmoothInterface,
    FlexSmoothQuantInterface,
    FA3QuantAdapterInterface,
    FA3QuantPlaceHolder,
    QuaRotInterface,
    AscendV1SaveInterface,
    AttentionAnalysisInterface,
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


@logger_setter("msmodelslim.model.deepseek_v3")
# pylint: disable=too-many-ancestors
class DeepSeekV3ModelAdapter(
    DefaultModelAdapter,
    AttentionAnalysisInterface,
    ModelInfoInterface,  # support naive quantization
    ModelSlimPipelineInterfaceV1,  # support modelslim v1
    IterSmoothInterface,  # support iter smooth
    FlexSmoothQuantInterface,  # support flex smooth quant
    FA3QuantAdapterInterface,  # support FA3 activation quant placeholders
    QuaRotInterface,
    AscendV1SaveInterface,
):
    def get_model_type(self) -> str:
        return self.model_type

    def get_model_pedigree(self) -> str:
        return 'deepseek_v3'

    def init_model(self, device: DeviceType = DeviceType.NPU) -> nn.Module:
        with default_dtype(torch.bfloat16):
            # 保存原始层数
            self.config.num_hidden_layers += 1
            origin_layers = self.config.num_hidden_layers
            get_logger().info("Model with %s layers totally", origin_layers)

            if dist.is_initialized():
                self.config.ep_size = dist.get_world_size()

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
            # pylint: disable=duplicate-code
            state_dict = self.get_state_dict(model)
            model.load_state_dict(state_dict)

            # auto convert fp8 to bf16
            auto_convert_module_fp8_to_bf16("", model, str(self.model_path))
            model.eval()
            get_logger().info("Create model with %s layers successfully at first", self.config.num_hidden_layers)
            return model

    def handle_dataset(self, dataset: Any, device: DeviceType = DeviceType.NPU) -> List[Any]:
        return self._get_tokenized_data(dataset, device)

    def generate_model_visit(self, model: nn.Module) -> Generator[ProcessRequest, Any, None]:
        return generated_decoder_layer_visit_func(model, transformer_blocks=self.generate_decoder_layer(model))

    # pylint: disable=duplicate-code
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
        # pylint: disable=duplicate-code
        try:
            if isinstance(inputs, (list, tuple)):
                model(inputs[0])
            elif isinstance(inputs, dict):
                model(**inputs)
            else:
                model(inputs)
        except TransformersForwardBreak:
            pass
        except Exception as e:
            raise e
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
            if name == f'model.layers.{self.config.num_hidden_layers - 1}':
                args, kwargs = self.mtp_preprocess(model, mtp_decoder=block, inputs=inputs, args=args, kwargs=kwargs)
            outputs = yield ProcessRequest(name, block, args, kwargs)
            hidden_states = outputs[0]
            current_inputs = ((hidden_states,), current_inputs[1])

    def mtp_preprocess(
        self,
        model: nn.Module,
        mtp_decoder: nn.Module,
        inputs: Union[List[Any], Dict[str, Any]],
        args: Tuple[Any, Any],
        kwargs: Dict[str, Any],
    ) -> Tuple[Tuple[Any, Any], Dict[str, Any]]:
        def wrap_device(module: nn.Module):
            # pylint: disable=duplicate-code
            def auto_module(arg):
                module.to('npu')
                result = module(arg.to('npu'))
                module.to('cpu')
                return result

            return auto_module

        pre_hidden_states = args[0]
        hidden_states = model.model.norm(pre_hidden_states)
        logits = wrap_device(model.lm_head)(hidden_states)
        logits = logits.float()

        ####################### MTP LAYER ######################
        input_ids = inputs['input_ids'] if isinstance(inputs, dict) else inputs[0]
        # pylint: disable=duplicate-code
        input_ids_mtp = remove_zero_and_shift(input_ids)
        position_ids = get_mtp_position_ids(input_ids_mtp.shape[-1], input_ids.device)
        input_ids_mtp[:, -1] = logits[:, -1, :].argmax(dim=1)

        input_embeds_mtp = wrap_device(mtp_decoder.embed_tokens)(input_ids_mtp)
        input_embeds_mtp = wrap_device(mtp_decoder.enorm)(input_embeds_mtp)
        hidden_states_mtp = wrap_device(mtp_decoder.hnorm)(pre_hidden_states)
        hidden_states_mtp = torch.cat([input_embeds_mtp, hidden_states_mtp], dim=-1)
        hidden_states_mtp = wrap_device(mtp_decoder.eh_proj)(hidden_states_mtp)

        attention_mask = inputs['attention_mask'] if isinstance(inputs, dict) else inputs[1]

        # transformers==4.48.2
        from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask

        attention_mask_mtp = _prepare_4d_causal_attention_mask(
            attention_mask,
            (input_ids.shape[:2]),
            input_embeds_mtp,
            0,
        )

        return (
            (hidden_states_mtp,),
            {
                "attention_mask": attention_mask_mtp,
                "position_ids": position_ids,
                "past_key_value": None,
                "output_attentions": False,
                "use_cache": False,
            },
        )

    def enable_kv_cache(self, model: nn.Module, need_kv_cache: bool) -> None:
        def pre_forward_hook(module, args, kwargs):
            kwargs['use_cache'] = need_kv_cache
            return args, kwargs

        model.model.register_forward_pre_hook(pre_forward_hook, with_kwargs=True)

    def get_adapter_config_for_subgraph(self) -> List[AdapterConfig]:
        adapter_config = []
        expert_start, expert_end = _get_expert_range(self.config)

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
            # pylint: disable=duplicate-code
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

            # 添加up-down的FFN配置
            if layer_idx < self.config.first_k_dense_replace:
                # Dense FFN 层
                up_proj = 'model.layers.' + str(layer_idx) + '.mlp.up_proj'
                down_proj = 'model.layers.' + str(layer_idx) + '.mlp.down_proj'
                up_down_mapping_config = MappingConfig(
                    source=up_proj,  # 上投影层
                    targets=[down_proj],  # 下投影层
                )
                adapter_config.extend(
                    [
                        AdapterConfig(subgraph_type="up-down", mapping=up_down_mapping_config),
                    ]
                )
            else:
                # MOE FFN 层：Shared Experts
                expert_up_proj = 'model.layers.' + str(layer_idx) + '.mlp.shared_experts.up_proj'
                expert_down_proj = 'model.layers.' + str(layer_idx) + '.mlp.shared_experts.down_proj'
                up_down_mapping_config_shared = MappingConfig(source=expert_up_proj, targets=[expert_down_proj])
                adapter_config.extend([AdapterConfig(subgraph_type="up-down", mapping=up_down_mapping_config_shared)])

                # MOE FFN 层：Routed Experts
                for expert in range(expert_start, expert_end):
                    up_proj = 'model.layers.' + str(layer_idx) + '.mlp.experts.' + str(expert) + '.up_proj'
                    down_proj = 'model.layers.' + str(layer_idx) + '.mlp.experts.' + str(expert) + '.down_proj'
                    up_down_mapping_config_expert = MappingConfig(source=up_proj, targets=[down_proj])
                    adapter_config.extend(
                        [AdapterConfig(subgraph_type="up-down", mapping=up_down_mapping_config_expert)]
                    )

        return adapter_config

    # ===== FA3QuantAdapterInterface =====
    def inject_fa3_placeholders(self, root_name: str, root_module: nn.Module, should_inject) -> None:
        """为 DeepSeekV3 注意力模块安装 FA3 占位，并包裹 forward 调用这些占位。

        - 在每个 Attention 模块下注入子模块：fa3_q, fa3_k, fa3_v
        - 包裹其 forward，在计算 q_nope 与 compressed_kv 后，依次调用占位：
            q_nope = self.fa3_q(q_nope)
            compressed_kv = self.fa3_k(compressed_kv.unsqueeze(1)).squeeze(1)
            _ = self.fa3_v(compressed_kv.unsqueeze(1)).squeeze(1)
        """

        def _wrap_attention_forward(attn_mod: nn.Module):
            # 动态导入以获取 apply_rotary_pos_emb（采用 import_module，更清晰稳健）
            deepseek_module = import_module(attn_mod.forward.__module__)
            apply_rotary_pos_emb = deepseek_module.apply_rotary_pos_emb

            def new_forward(
                self,
                hidden_states: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.LongTensor] = None,
                past_key_value: Optional[Any] = None,
                output_attentions: bool = False,
                use_cache: bool = False,
                **kwargs,
            ):
                # 参考默认适配器 deepseek mla 前向的关键路径（保留原计算流，仅插入占位调用）
                bsz, q_len, _ = hidden_states.size()

                if getattr(self, 'q_lora_rank', None) is None:
                    q = self.q_proj(hidden_states)
                else:
                    q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
                q = q.view(bsz, q_len, self.num_heads, self.q_head_dim).transpose(1, 2)
                q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

                compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
                compressed_kv, k_pe = torch.split(compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
                compressed_kv = self.kv_a_layernorm(compressed_kv)
                k_pe = k_pe.view(bsz, q_len, 1, self.qk_rope_head_dim).transpose(1, 2)
                kv_seq_len = k_pe.shape[-2]

                if past_key_value is not None:
                    if getattr(self, 'layer_idx', None) is None:
                        raise ValueError(
                            f"The cache structure has changed since version v4.36. "
                            f"If you are using {self.__class__.__name__} "
                            f"for auto-regressive decoding with k/v caching, "
                            f"please make sure to initialize the attention class "
                            "with a layer index."
                        )
                    kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)

                cos, sin = self.rotary_emb(q_pe, seq_len=kv_seq_len)
                q_pe, k_pe = apply_rotary_pos_emb(q_pe, k_pe, cos, sin, position_ids)

                if past_key_value is not None:
                    cache_kwargs = {"sin": sin, "cos": cos}
                    compressed_kv = compressed_kv.unsqueeze(1)
                    k_pe, compressed_kv = past_key_value.update(k_pe, compressed_kv, self.layer_idx, cache_kwargs)
                    compressed_kv = compressed_kv.squeeze(1)

                kv_b_proj = self.kv_b_proj.weight.view(self.num_heads, -1, self.kv_lora_rank)

                q_absorb = kv_b_proj[:, : self.qk_nope_head_dim, :]
                out_absorb = kv_b_proj[:, self.qk_nope_head_dim :, :]

                q_nope = torch.matmul(q_nope, q_absorb)

                # ===== 插入 FA3 占位 =====
                if hasattr(self, 'fa_q'):
                    q_nope = self.fa_q(q_nope)
                if hasattr(self, 'fa_k'):
                    compressed_kv = self.fa_k(compressed_kv.unsqueeze(1)).squeeze(1)
                if hasattr(self, 'fa_v'):
                    _ = self.fa_v(compressed_kv.unsqueeze(1)).squeeze(1)
                # ========================

                attn_weights = torch.matmul(q_pe, k_pe.mT) + torch.matmul(q_nope, compressed_kv.unsqueeze(-3).mT)
                attn_weights = attn_weights * self.softmax_scale

                if attention_mask is None:
                    raise ValueError("Attention mask cannot be None")
                attn_weights = attn_weights + attention_mask

                attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q_pe.dtype)
                attn_weights = torch.nn.functional.dropout(
                    attn_weights, p=self.attention_dropout, training=self.training
                )
                attn_output = torch.einsum('bhql,blc->bhqc', attn_weights, compressed_kv)
                attn_output = torch.matmul(attn_output, out_absorb.mT)
                attn_output = attn_output.transpose(1, 2).contiguous()
                attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.v_head_dim)
                attn_output = self.o_proj(attn_output)
                if not output_attentions:
                    attn_weights_out = None
                else:
                    attn_weights_out = attn_weights
                return attn_output, attn_weights_out, past_key_value

            # pylint: disable=no-value-for-parameter
            attn_mod.forward = new_forward.__get__(attn_mod, attn_mod.__class__)

        # 遍历并注入
        for name, module in root_module.named_modules():
            if 'Attention' in module.__class__.__name__ and should_inject(f"{root_name}.{name}" if root_name else name):
                # 为该注意力模块注入占位
                root_module.set_submodule(f"{name}.fa_q", FA3QuantPlaceHolder(ratio=0.9999))
                root_module.set_submodule(f"{name}.fa_k", FA3QuantPlaceHolder(ratio=0.9999))
                root_module.set_submodule(f"{name}.fa_v", FA3QuantPlaceHolder(ratio=1.0))
                _wrap_attention_forward(module)

    def get_ln_fuse_map(self):
        return {}, get_ln_fuse_map(self.config, num_hidden_layers=self.config.num_hidden_layers)

    def get_bake_names(self):
        return [], []

    def get_rotate_map(self, block_size):
        pre_run, rot_pairs, _ = get_rotate_map(self.config, block_size, num_hidden_layers=self.config.num_hidden_layers)
        return [pre_run], list(rot_pairs.values())

    @lru_cache(maxsize=1)
    def get_weight_map(self):
        # pylint: disable=duplicate-code
        model_index_path = os.path.join(self.model_path, "model.safetensors.index.json")
        model_index = json_safe_load(model_index_path)
        weight_map = model_index['weight_map']
        return weight_map

    def get_state_dict(self, module: nn.Module, prefix: str = ""):
        weight_map = self.get_weight_map()
        names = map(lambda x: x[0], module.named_parameters())

        groups = defaultdict(list)
        for name in names:
            # pylint: disable=duplicate-code
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
                get_logger().info("Creating decoder layer %s", idx)
                module_list: nn.ModuleList = model.model.layers
                template_module = module_list[0]
                decoder = template_module.__class__(layer_idx=idx, config=self.config)
                # pylint: disable=duplicate-code
                state_dict = self.get_state_dict(decoder, prefix=name)
                decoder.load_state_dict(state_dict)
                auto_convert_module_fp8_to_bf16(name, decoder, str(self.model_path))
                decoder.eval()
                module_list.append(decoder)
                get_logger().info("Create decoder layer %s successfully", idx)
        return decoder

    def load_mtp_if_not_load(self, mtp_decoder: nn.Module):
        # pylint: disable=duplicate-code
        try:
            mtp_decoder.get_submodule('shared_head')
        except AttributeError:
            get_logger().info('Creating MTP layer')
            mtp_layer = get_mtp_layer(config=self.config, model_path=self.model_path)
            wrap_mtp_decoder(mtp_decoder=mtp_decoder, mtp_layer=mtp_layer)
            get_logger().info('Create MTP successfully')

    # pylint: disable=duplicate-code
    def generate_decoder_layer(self, model: nn.Module):
        for idx in range(self.config.num_hidden_layers):
            name = f"model.layers.{idx}"
            decoder = self.load_decoder_if_not_exist(model, name=name, idx=idx)
            if idx == self.config.num_hidden_layers - 1:
                self.load_mtp_if_not_load(decoder)
            yield name, decoder

    # pylint: disable=useless-return
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

            config_data["mtp_quantize"] = "w8a8_dynamic"
            config_data["quantize"] = "w8a8_dynamic"
            config_data["moe_quantize"] = "w4a8_dynamic"
            if use_c8:
                config_data["mla_quantize"] = "w8a8"
            else:
                config_data["mla_quantize"] = "w8a8_dynamic"

            json_safe_dump(config_data, config_file, indent=2, check_user_stat=False)

        return

    def get_attention_module_cls(self) -> str:
        return "DeepseekV3Attention"

    def get_attention_output_extractor(self) -> Callable[[Union[tuple, torch.Tensor]], torch.Tensor]:
        """
        DeepseekV3Attention.forward 返回 (attn_output, attn_weights, past_key_value)
        注意力结构的敏感分析需要使用attn_output，因此返回lambda x: x[0]
        """
        return lambda x: x[0]
