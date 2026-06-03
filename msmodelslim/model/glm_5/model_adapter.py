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

import os.path
import types
from collections import defaultdict
from functools import lru_cache
from typing import List, Any, Generator, Optional, Tuple, Dict, Union, Callable
from unittest.mock import patch

import torch
from einops import rearrange
from safetensors import safe_open
from torch import distributed as dist
from torch import nn
from tqdm import tqdm

from msmodelslim import ir as qir
from msmodelslim.app.naive_quantization.model_info_interface import ModelInfoInterface
from msmodelslim.core.base.protocol import ProcessRequest
from msmodelslim.core.const import DeviceType
from msmodelslim.core.graph import AdapterConfig, MappingConfig, FusionConfig
from msmodelslim.ir import QuaRotExtraInfoWrapperIR
from msmodelslim.processor.quarot import QuaRotInterface
from msmodelslim.utils.exception import InvalidModelError, UnsupportedError
from msmodelslim.utils.logging import logger_setter, get_logger
from msmodelslim.utils.security import get_valid_read_path, json_safe_load, json_safe_dump, MAX_READ_FILE_SIZE_32G
from .quarot import get_ln_fuse_map, get_rotate_map
from .convert_fp8_to_bf16 import auto_convert_module_fp8_to_bf16
from .model import Transformer, ModelArgs
from .mtp_quant_module import MTPLayer, wrap_mtp_decoder, remove_zero_and_shift
from ..common.layer_wise_forward import generated_decoder_layer_visit_func, TransformersForwardBreak
from ..common.transformers import TransformersModel
from ..interface_hub import (
    ModelSlimPipelineInterfaceV1,
    FlexSmoothQuantInterface,
    FA3QuantAdapterInterface,
    FA3QuantPlaceHolder,
    OnlineQuaRotInterface,
    AscendV1SaveInterface,
)
from msmodelslim.model.common.utils import _get_expert_range


@logger_setter("msmodelslim.model.glm_5")
class GLM5ModelAdapter(  # pylint: disable=too-many-ancestors
    TransformersModel,
    ModelInfoInterface,
    ModelSlimPipelineInterfaceV1,
    FlexSmoothQuantInterface,
    QuaRotInterface,
    FA3QuantAdapterInterface,
    OnlineQuaRotInterface,
    AscendV1SaveInterface,
):
    def get_model_pedigree(self) -> str:
        return 'glm_5'

    def get_model_type(self) -> str:
        return self.model_type

    def handle_dataset(self, dataset: Any, device: DeviceType = DeviceType.NPU) -> List[Any]:
        return self._get_tokenized_data(dataset, device)

    def init_model(self, device: DeviceType = DeviceType.NPU) -> nn.Module:
        torch.set_default_dtype(torch.bfloat16)
        self.config.num_hidden_layers = 79
        get_logger().info("Model with %s layers totally", self.config.num_hidden_layers)

        origin = self.config.num_hidden_layers

        self.config.num_hidden_layers = 1
        with torch.device("cpu"):
            model: nn.Module = Transformer(self.config)

        self.config.num_hidden_layers = origin

        state_dict = self.get_state_dict(model)
        model.load_state_dict(state_dict)
        auto_convert_module_fp8_to_bf16("", model, str(self.model_path))
        model.eval()
        get_logger().info("Create model with %s layers successfully at first", self.config.num_hidden_layers)
        return model

    def generate_model_visit(self, model: nn.Module) -> Generator[ProcessRequest, Any, None]:
        return generated_decoder_layer_visit_func(model, transformer_blocks=self.generate_decoder_layer(model))

    def generate_model_forward(self, model: nn.Module, inputs: Any) -> Generator[ProcessRequest, Any, None]:
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

        args, kwargs = current_inputs
        for name, block in self.generate_decoder_layer(model):
            if name == f'model.layers.{self.config.num_hidden_layers - 1}':
                args, kwargs = self.mtp_preprocess(model, mtp_decoder=block, inputs=inputs, args=args, kwargs=kwargs)
            h, residual = yield ProcessRequest(name, block, args, kwargs)
            args = (h, residual)

    def mtp_preprocess(
        self,
        model: nn.Module,
        mtp_decoder: nn.Module,
        inputs: Union[List[Any], Dict[str, Any]],
        args: Tuple[Any, Any],
        kwargs: Dict[str, Any],
    ) -> Tuple[Tuple[Any, Any], Dict[str, Any]]:
        def wrap_device(module: nn.Module):
            def auto_module(arg):
                module.to('npu')
                result = module(arg.to('npu'))
                module.to('cpu')
                return result

            return auto_module

        pre_hidden_states, residual = args
        hidden_states = model.model.norm(pre_hidden_states)
        logits = wrap_device(model.lm_head)(hidden_states)
        logits = logits.float()

        ####################### MTP LAYER ######################
        input_ids = inputs['input_ids'] if isinstance(inputs, dict) else inputs[0]
        input_ids_mtp = remove_zero_and_shift(input_ids)
        position_ids = (
            torch.arange(
                0,
                input_ids_mtp.shape[-1],
                dtype=torch.long,
                device=input_ids.device,
            )
            + 1
        )
        position_ids = position_ids.unsqueeze(0)
        logits[:, -1, :].argmax(dim=1)
        input_ids_mtp[:, -1] = logits[:, -1, :].argmax(dim=1)

        input_embeds_mtp = wrap_device(mtp_decoder.embed_tokens)(input_ids_mtp)
        input_embeds_mtp = wrap_device(mtp_decoder.enorm)(input_embeds_mtp)
        hidden_states_mtp = wrap_device(mtp_decoder.hnorm)(pre_hidden_states)
        hidden_states_mtp = torch.cat([input_embeds_mtp, hidden_states_mtp], dim=-1)
        hidden_states_mtp = wrap_device(mtp_decoder.eh_proj)(hidden_states_mtp)

        attention_mask = inputs['attention_mask'] if isinstance(inputs, dict) else inputs[1]

        from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask

        attention_mask_mtp = _prepare_4d_causal_attention_mask(
            attention_mask,
            (input_ids.shape[:2]),
            input_embeds_mtp,
            0,
        )

        start_pos = kwargs['start_pos'] + 1
        seq_len = len(kwargs['freqs_cis'])
        kwargs['mask'] = attention_mask_mtp.squeeze(1)
        kwargs['freqs_cis'] = model.model.freqs_cis[start_pos : start_pos + seq_len]
        return (hidden_states_mtp, residual), kwargs

    def enable_kv_cache(self, model: nn.Module, need_kv_cache: bool) -> None:
        pass

    def get_adapter_config_for_subgraph(self) -> List[AdapterConfig]:
        adapter_config = []
        expert_start, expert_end = _get_expert_range(self.config)

        for layer_idx in range(self.config.num_hidden_layers):
            # OKV_b融合的映射配置：o_proj -> kv_b_proj
            okv_b_mapping_config = MappingConfig(
                source=f"model.layers.{layer_idx}.self_attn.kv_b_proj",  # KV_b投影层
                targets=[f"model.layers.{layer_idx}.self_attn.o_proj"],  # 输出投影层
            )

            # Norm-Linear融合的映射配置1：q_a_proj, kv_a_proj_with_mqa -> input_layernorm
            input_norm_mapping_config = MappingConfig(
                source=f"model.layers.{layer_idx}.input_layernorm",  # 第一个LayerNorm
                targets=[
                    f"model.layers.{layer_idx}.self_attn.q_a_proj",
                    f"model.layers.{layer_idx}.self_attn.kv_a_proj_with_mqa",
                    f"model.layers.{layer_idx}.self_attn.indexer.wk",
                    f"model.layers.{layer_idx}.self_attn.indexer.weights_proj",
                ],  # 注意力层的Q_a,KV_a投影
            )

            # Norm-Linear融合的映射配置2：q_b_proj -> q_a_layernorm
            qa_norm_mapping_config = MappingConfig(
                source=f"model.layers.{layer_idx}.self_attn.q_a_layernorm",  # q_a_layernorm
                targets=[
                    f"model.layers.{layer_idx}.self_attn.q_b_proj",
                    f"model.layers.{layer_idx}.self_attn.indexer.wq_b",
                ],  # q_b投影
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
                    AdapterConfig(subgraph_type="norm-linear", mapping=input_norm_mapping_config),
                    AdapterConfig(subgraph_type="norm-linear", mapping=qa_norm_mapping_config),
                ]
            )

            # 根据层类型添加不同的FFN配置
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
            full_name = f'{prefix}.{name}' if prefix else name
            if full_name not in weight_map:
                continue
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

    def get_mtp_layer(
        self,
    ):
        get_logger().debug('Start to load mtp')
        mtp_layer = MTPLayer(self.config)

        # lm_head.weight ——> MTPLayer.shared_head.head.weight
        # model.embed_tokens.weight ——> MTPLayer.embed_tokens.weight
        embed_tokends = nn.Embedding(self.config.vocab_size, self.config.hidden_size)
        embed_state_dict = self.get_state_dict(embed_tokends, prefix='model.embed_tokens')
        head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
        head_state_dict = self.get_state_dict(head, prefix='lm_head')

        state_dict = self.get_state_dict(mtp_layer, prefix=f'model.layers.{self.config.num_hidden_layers - 1}')
        state_dict['shared_head.head.weight'] = head_state_dict['weight']
        state_dict['embed_tokens.weight'] = embed_state_dict['weight']

        mtp_layer.load_state_dict(state_dict)
        auto_convert_module_fp8_to_bf16(
            f'model.layers.{self.config.num_hidden_layers - 1}', mtp_layer, str(self.model_path)
        )

        get_logger().debug('Success to load mtp')
        return mtp_layer

    def load_mtp_if_not_load(self, mtp_decoder: nn.Module):
        try:
            mtp_decoder.get_submodule('shared_head')
        except AttributeError:
            get_logger().info('Creating MTP layer')
            mtp_layer = self.get_mtp_layer()
            wrap_mtp_decoder(mtp_decoder=mtp_decoder, mtp_layer=mtp_layer)
            get_logger().info('Create MTP successfully')

    def load_decoder_if_not_exist(self, model: nn.Module, name: str, idx: int):
        try:
            decoder = model.get_submodule(name)
        except AttributeError:
            # disable reset_parameters so that the weights will not be initialized
            # these initializations is not necessary because we will load it from the state_dict
            # and these initializations will cost too much time because the GLM-5's decoder layer is too large
            with patch.object(nn.Linear, 'reset_parameters', lambda _self: None):
                get_logger().info("Creating decoder layer %s", idx)
                module_list: nn.ModuleList = model.model.layers
                template_module = module_list[0]
                decoder = template_module.__class__(layer_id=idx, args=self.config)

                state_dict = self.get_state_dict(decoder, prefix=name)
                decoder.load_state_dict(state_dict)
                auto_convert_module_fp8_to_bf16(name, decoder, str(self.model_path))
                decoder.eval()
                module_list.append(decoder)
                get_logger().info("Create decoder layer %s successfully", idx)
        return decoder

    def generate_decoder_layer(self, model: nn.Module):
        for idx in range(self.config.num_hidden_layers):
            name = f"model.layers.{idx}"
            decoder = self.load_decoder_if_not_exist(model, name=name, idx=idx)
            if idx == self.config.num_hidden_layers - 1:
                self.load_mtp_if_not_load(decoder)
            yield name, decoder

    def get_ln_fuse_map(self):
        ln_linear_map = get_ln_fuse_map(self.config, num_hidden_layers=self.config.num_hidden_layers)
        for layer_idx in range(self.config.num_hidden_layers):
            ln_linear_map[f"model.layers.{layer_idx}.input_layernorm"].append(
                f"model.layers.{layer_idx}.self_attn.indexer.wk",
            )
            ln_linear_map[f"model.layers.{layer_idx}.input_layernorm"].append(
                f"model.layers.{layer_idx}.self_attn.indexer.weights_proj",
            )
            ln_linear_map[f"model.layers.{layer_idx}.self_attn.q_a_layernorm"].append(
                f"model.layers.{layer_idx}.self_attn.indexer.wq_b"
            )
        return {}, ln_linear_map

    def get_bake_names(self):
        return [], []

    def get_rotate_map(self, block_size):
        pre_run, rot_pairs, rotate_matrix = get_rotate_map(
            self.config, block_size, num_hidden_layers=self.config.num_hidden_layers
        )
        for layer_idx in range(self.config.num_hidden_layers):
            rot_pairs['rot'].right_rot[f"model.layers.{layer_idx}.self_attn.indexer.wk"] = rotate_matrix['rot']
            rot_pairs['rot'].right_rot[f"model.layers.{layer_idx}.self_attn.indexer.weights_proj"] = rotate_matrix[
                'rot'
            ]
            rot_pairs['rot_b_proj'].right_rot[f"model.layers.{layer_idx}.self_attn.indexer.wq_b"] = rotate_matrix[
                'rot_b_proj'
            ]
        return [pre_run], list(rot_pairs.values())

    def get_online_rotation_configs(self, model: Optional[nn.Module] = None):
        configs = {}
        shared_seed = 1234
        head_dim = self.config.index_head_dim

        for layer_idx in range(self.config.num_hidden_layers):
            name = f"model.layers.{layer_idx}.self_attn.indexer"

            configs[f"{name}.q_rot"] = OnlineQuaRotInterface.RotationConfig(
                rotation_type="replace",
                rotation_size=head_dim,
                rotation_mode=OnlineQuaRotInterface.QuaRotMode.HADAMARD,
                block_size=-1,
                seed=shared_seed,
                dtype=torch.bfloat16,
            )

            configs[f"{name}.k_rot"] = OnlineQuaRotInterface.RotationConfig(
                rotation_type="replace",
                rotation_size=head_dim,
                rotation_mode=OnlineQuaRotInterface.QuaRotMode.HADAMARD,
                block_size=-1,
                seed=shared_seed,
                dtype=torch.bfloat16,
            )

        return configs

    def inject_fa3_placeholders(self, root_name: str, root_module: nn.Module, should_inject) -> None:
        from importlib import import_module
        from .model import rotate_activation, fp8_index

        def _wrap_indexer_forward(indexer_mod: nn.Module):
            glm5_module = import_module(indexer_mod.forward.__module__)
            apply_rotary_emb = glm5_module.apply_rotary_emb

            def new_indexer_forward(
                self,
                x: torch.Tensor,
                qr: torch.Tensor,
                start_pos: int,
                freqs_cis: torch.Tensor,
                mask: Optional[torch.Tensor],
            ):
                bsz, seqlen, _ = x.size()
                end_pos = start_pos + seqlen

                q = self.wq_b(qr)
                q = rearrange(q, 'b s (h d) -> b s h d', d=self.head_dim)
                q_pe, q_nope = torch.split(q, [self.rope_head_dim, self.head_dim - self.rope_head_dim], dim=-1)
                q_pe = apply_rotary_emb(q_pe, freqs_cis)
                q = torch.cat([q_pe, q_nope], dim=-1)

                k = self.wk(x)
                k = self.k_norm(k)
                k_pe, k_nope = torch.split(k, [self.rope_head_dim, self.head_dim - self.rope_head_dim], dim=-1)
                k_pe = apply_rotary_emb(k_pe.unsqueeze(2), freqs_cis).squeeze(2)
                k = torch.cat([k_pe, k_nope], dim=-1)

                if hasattr(self, 'q_rot'):
                    q = self.q_rot(q)
                else:
                    q = rotate_activation(q)
                if hasattr(self, 'k_rot'):
                    k = self.k_rot(k)
                else:
                    k = rotate_activation(k)

                if hasattr(self, 'fa3_q'):
                    q = self.fa3_q(q)
                if hasattr(self, 'fa3_k'):
                    k = self.fa3_k(k)

                q_scale = torch.ones(*q.size()[:-1], q.size(-1) // 128, dtype=torch.float32).npu()
                weights = self.weights_proj(x) * self.n_heads**-0.5
                weights = weights.unsqueeze(-1) * q_scale * self.softmax_scale

                k = k.view(bsz, -1, 1, self.head_dim)
                index_score = fp8_index(q.contiguous(), weights, k)

                if mask is not None:
                    index_score += mask
                topk_indices = index_score.topk(min(self.index_topk, end_pos), dim=-1)[1]
                return topk_indices.clone()

            indexer_mod.forward = types.MethodType(new_indexer_forward, indexer_mod)

        for name, module in root_module.named_modules():
            module_type = module.__class__.__name__

            if module_type not in ["Indexer"]:
                continue

            full_name = f"{root_name}.{name}" if root_name else name
            if not should_inject(full_name):
                continue

            if name != "":
                root_module.set_submodule(f'{name}.fa3_q', FA3QuantPlaceHolder(ratio=0.9999))
                root_module.set_submodule(f'{name}.fa3_k', FA3QuantPlaceHolder(ratio=0.9999))
            else:
                root_module.set_submodule('fa3_q', FA3QuantPlaceHolder(ratio=0.9999))
                root_module.set_submodule('fa3_k', FA3QuantPlaceHolder(ratio=0.9999))
            _wrap_indexer_forward(module)

    def get_attention_module_cls(self) -> str:
        return "MLA"

    def get_attention_output_extractor(self) -> Callable[[Union[tuple, torch.Tensor]], torch.Tensor]:
        return lambda x: x

    def ascendv1_save_postprocess(self, model: nn.Module, save_directory: str) -> None:
        global_rotation, norm_weight = None, None

        # catch the global rotation
        for _, module in model.named_modules():
            if isinstance(module, QuaRotExtraInfoWrapperIR):
                offline_info = module.rotation_info
                global_rotation = offline_info.global_rotation
        if global_rotation is None:
            return

        # catch the original model.norm.weight
        origin_index_path = os.path.join(self.model_path, "model.safetensors.index.json")
        origin_index_data = json_safe_load(origin_index_path)

        weight_path = os.path.join(self.model_path, origin_index_data["weight_map"]["model.norm.weight"])
        with safe_open(weight_path, framework='pt', device='cpu') as f:
            norm_weight = f.get_tensor("model.norm.weight")
        if norm_weight is None:
            raise UnsupportedError("model.norm.weight is not found.")

        def _apply_rot_transform(w: torch.Tensor, Q: torch.Tensor) -> torch.Tensor:
            """
            计算 Q.T @ w @ Q。
            - 约定 w 为 1D (d,) 或 (d, 1)，视为对角矩阵 diag(w)，Q 为 (d, d)，返回 (d, d)。
            """
            if w.dim() != 1:
                raise ValueError(f"Weight w must be 1D, got shape {w.shape}")
            dtype = torch.float32
            device = w.device
            w = w.to(dtype=dtype, device=device)
            Q = Q.to(dtype=dtype, device=device)

            w = w.flatten()
            d = w.shape[0]
            if Q.shape[0] != d or Q.shape[1] != d:
                raise ValueError(f"Q must be ({d}, {d}) when w is 1D length {d}, got Q {Q.shape}")
            return Q.T * w @ Q

        # keep the output dtype same as the original model.norm.weight
        original_dtype = norm_weight.dtype
        rot_weight = _apply_rot_transform(norm_weight, global_rotation).to(original_dtype)

        from safetensors.torch import save_file

        save_file({"rot.weight": rot_weight}, os.path.join(save_directory, "rot.safetensors"))

        # update quant_model_description.json
        description_path = os.path.join(save_directory, "quant_model_description.json")
        description_data = json_safe_load(description_path)
        description_data["is_rot_used"] = True
        json_safe_dump(description_data, description_path, indent=2)

        # update quant_model_weights.safetensors.index.json
        index_path = os.path.join(save_directory, "quant_model_weights.safetensors.index.json")
        index_data = json_safe_load(index_path)
        index_data["weight_map"]["rot.weight"] = "rot.safetensors"
        json_safe_dump(index_data, index_path, indent=2)

        use_per_token_c8 = False
        for _, module in model.named_modules():
            if isinstance(module, qir.FakeQuantActivationPerToken):
                use_per_token_c8 = True
                break

        if use_per_token_c8:
            description_file = os.path.join(save_directory, "quant_model_description.json")
            description_data = json_safe_load(description_file, check_user_stat=False)
            description_data["indexer_quant_type"] = "INT8_DYNAMIC"
            json_safe_dump(description_data, description_file, indent=2, check_user_stat=False)

    def _load_config(self, trust_remote_code=False) -> object:
        return ModelArgs()
