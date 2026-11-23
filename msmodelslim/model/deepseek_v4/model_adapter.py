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

import os.path
from functools import wraps
from importlib import import_module
from types import MethodType
from typing import List, Any, Generator, Optional, Tuple, Dict, Callable
from unittest.mock import patch

import torch
import torch.distributed as dist
from torch import nn

from msmodelslim.app.naive_quantization.model_info_interface import ModelInfoInterface
from msmodelslim.core.base.protocol import ProcessRequest
from msmodelslim.core.const import DeviceType
from msmodelslim.core.infer_engine.interface import FakeQuantInferenceInterface
from msmodelslim.core.graph import AdapterConfig, MappingConfig
from msmodelslim.core.quantization_algorithm.quarot import QuaRotInterface
from msmodelslim.utils.exception import InvalidModelError
from msmodelslim.utils.logging import logger_setter, get_logger
from msmodelslim.utils.security import json_safe_load
from .convert_fp8_to_bf16 import auto_dequant_state_dict
from .model import Transformer, ModelArgs, Block
from .mtp_quant_module import get_mtp_layer, wrap_mtp_decoder, remove_zero_and_shift
from ..common.layer_wise_forward import generated_decoder_layer_visit_func, TransformersForwardBreak
from ..common.transformers import TransformersModel
from ..common.weight_helper import get_state_dict
from ..interface_hub import (
    ModelSlimPipelineInterfaceV1,
    FlexSmoothQuantInterface,
    IterSmoothInterface,
    AscendV1SaveInterface,
    FA3QuantAdapterInterface,
)


@logger_setter("msmodelslim.model.deepseek_v4")
class DeepSeekV4ModelAdapter(  # pylint: disable=too-many-ancestors
    TransformersModel,
    ModelInfoInterface,
    ModelSlimPipelineInterfaceV1,
    IterSmoothInterface,
    FlexSmoothQuantInterface,
    QuaRotInterface,
    AscendV1SaveInterface,
    FA3QuantAdapterInterface,
    FakeQuantInferenceInterface,
):
    def get_model_pedigree(self) -> str:
        return 'deepseek_v4'

    def get_model_type(self) -> str:
        return self.model_type

    def handle_dataset(self, dataset: Any, device: DeviceType = DeviceType.NPU) -> List[Any]:
        return self._get_tokenized_data(dataset, device)

    def init_model(self, device: DeviceType = DeviceType.NPU) -> nn.Module:
        torch.set_default_dtype(torch.bfloat16)
        n_mtp = getattr(self.config, 'n_mtp_layers', 0)
        total = self.config.num_hidden_layers + n_mtp
        get_logger().info("Model with %s layers + %s MTP = %s total", self.config.num_hidden_layers, n_mtp, total)

        origin = self.config.num_hidden_layers

        self.config.num_hidden_layers = 1
        with torch.device("cpu"):
            model: nn.Module = Transformer(self.config)

        self.config.num_hidden_layers = origin

        state_dict = get_state_dict(self.model_path, model)
        auto_dequant_state_dict("", state_dict, str(self.model_path))
        model.load_state_dict(state_dict)
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

        remove_handler = model.layers[0].register_forward_pre_hook(break_hook, with_kwargs=True, prepend=True)

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
        h, start_pos, input_ids = args
        for name, block in self.generate_decoder_layer(model):
            if name.startswith('mtp.'):
                args, kwargs = self.mtp_preprocess(model, mtp_decoder=block, args=args, kwargs=kwargs)
            h = yield ProcessRequest(name, block, args, kwargs)
            args = (h, start_pos, input_ids)

    def mtp_preprocess(
        self, model: nn.Module, mtp_decoder: nn.Module, args: Tuple[Any, Any], kwargs: Dict[str, Any]
    ) -> Tuple[Tuple[Any, Any], Dict[str, Any]]:
        def wrap_device(module: nn.Module):
            def auto_module(arg):
                module.to('npu')
                result = module(arg.to('npu'))
                module.to('cpu')
                return result

            return auto_module

        pre_hidden_states, start_pos, input_ids = args
        pre_hidden_states = model.hc_head(pre_hidden_states, model.hc_head_fn, model.hc_head_scale, model.hc_head_base)
        # hidden_states = pre_hidden_states = wrap_device(model.norm)(pre_hidden_states)
        hidden_states = wrap_device(model.norm)(pre_hidden_states)
        logits = wrap_device(model.head)(hidden_states[:, -1])
        logits = logits.float()

        ####################### MTP LAYER ######################
        # input_ids = inputs['input_ids'] if isinstance(inputs, dict) else inputs[0]
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
        input_ids_mtp[:, -1] = logits.argmax(dim=1)

        input_embeds_mtp = wrap_device(mtp_decoder.emb.tok_emb)(input_ids_mtp)
        input_embeds_mtp = wrap_device(mtp_decoder.enorm)(input_embeds_mtp)
        input_embeds_mtp = wrap_device(mtp_decoder.e_proj)(input_embeds_mtp)
        # input_embeds_mtp = wrap_device(mtp_decoder.enorm)(input_embeds_mtp)

        hidden_states_mtp = wrap_device(mtp_decoder.hnorm)(pre_hidden_states)
        hidden_states_mtp = wrap_device(mtp_decoder.h_proj)(hidden_states_mtp)
        # hidden_states_mtp = wrap_device(mtp_decoder.hnorm)(hidden_states_mtp)

        hidden_states_mtp = torch.add(input_embeds_mtp, hidden_states_mtp)
        # hidden_states_mtp = wrap_device(mtp_decoder.norm)(hidden_states_mtp)
        hc_mult = mtp_decoder.hc_head_base.shape[0]
        hidden_states_mtp = hidden_states_mtp.unsqueeze(2).repeat(1, 1, hc_mult, 1)  # [b, s, hc_mult, d]

        return (hidden_states_mtp, start_pos + 1, input_ids), kwargs

    def enable_kv_cache(self, model: nn.Module, need_kv_cache: bool) -> None:
        pass

    def get_adapter_config_for_subgraph(self) -> List[AdapterConfig]:
        adapter_config = []
        expert_start, expert_end, _ = self._get_local_expert_range()

        for prefix, layer_or_mtp_idx, ratio, is_mtp in self._iter_layers_and_mtp():
            # ============================ MOE =========================
            # Shared Experts
            adapter_config.extend(
                [
                    AdapterConfig(
                        subgraph_type="up-down",
                        mapping=MappingConfig(
                            source=f"{prefix}.{layer_or_mtp_idx}.ffn.shared_experts.w3",
                            targets=[f"{prefix}.{layer_or_mtp_idx}.ffn.shared_experts.w2"],
                        ),
                    )
                ]
            )

            # Routed Experts (only local experts for EP)
            for expert in range(expert_start, expert_end):
                adapter_config.extend(
                    [
                        AdapterConfig(
                            subgraph_type="up-down",
                            mapping=MappingConfig(
                                source=f"{prefix}.{layer_or_mtp_idx}.ffn.experts.{expert}.w3",
                                targets=[f"{prefix}.{layer_or_mtp_idx}.ffn.experts.{expert}.w2"],
                            ),
                        )
                    ]
                )

            # ======================== Attention ========================
            # Linear-Linear
            adapter_config.extend(
                [
                    AdapterConfig(
                        subgraph_type="linear-linear",
                        mapping=MappingConfig(
                            source=f"{prefix}.{layer_or_mtp_idx}.attn.wo_a",
                            targets=[f"{prefix}.{layer_or_mtp_idx}.attn.wo_b"],
                        ),
                    )
                ]
            )

            # Norm-Linear by ratio
            if ratio <= 1:
                input_norm_mapping_config = MappingConfig(
                    source=f"{prefix}.{layer_or_mtp_idx}.attn_norm",
                    targets=[f"{prefix}.{layer_or_mtp_idx}.attn.wq_a", f"{prefix}.{layer_or_mtp_idx}.attn.wkv"],
                )
                qa_norm_mapping_config = MappingConfig(
                    source=f"{prefix}.{layer_or_mtp_idx}.attn.q_norm",
                    targets=[f"{prefix}.{layer_or_mtp_idx}.attn.wq_b"],
                )
            elif ratio == 4:
                input_norm_mapping_config = MappingConfig(
                    source=f"{prefix}.{layer_or_mtp_idx}.attn_norm",
                    targets=[
                        f"{prefix}.{layer_or_mtp_idx}.attn.wq_a",
                        f"{prefix}.{layer_or_mtp_idx}.attn.wkv",
                        f"{prefix}.{layer_or_mtp_idx}.attn.compressor.wgate",
                        f"{prefix}.{layer_or_mtp_idx}.attn.compressor.wkv",
                        f"{prefix}.{layer_or_mtp_idx}.attn.indexer.weights_proj",
                        f"{prefix}.{layer_or_mtp_idx}.attn.indexer.compressor.wgate",
                        f"{prefix}.{layer_or_mtp_idx}.attn.indexer.compressor.wkv",
                    ],
                )
                qa_norm_mapping_config = MappingConfig(
                    source=f"{prefix}.{layer_or_mtp_idx}.attn.q_norm",
                    targets=[
                        f"{prefix}.{layer_or_mtp_idx}.attn.wq_b",
                        f"{prefix}.{layer_or_mtp_idx}.attn.indexer.wq_b",
                    ],
                )
            else:
                input_norm_mapping_config = MappingConfig(
                    source=f"{prefix}.{layer_or_mtp_idx}.attn_norm",
                    targets=[
                        f"{prefix}.{layer_or_mtp_idx}.attn.wq_a",
                        f"{prefix}.{layer_or_mtp_idx}.attn.wkv",
                        f"{prefix}.{layer_or_mtp_idx}.attn.compressor.wgate",
                        f"{prefix}.{layer_or_mtp_idx}.attn.compressor.wkv",
                    ],
                )
                qa_norm_mapping_config = MappingConfig(
                    source=f"{prefix}.{layer_or_mtp_idx}.attn.q_norm",
                    targets=[f"{prefix}.{layer_or_mtp_idx}.attn.wq_b"],
                )

            adapter_config.extend(
                [
                    AdapterConfig(subgraph_type="norm-linear", mapping=input_norm_mapping_config),
                    AdapterConfig(subgraph_type="norm-linear", mapping=qa_norm_mapping_config),
                ]
            )

        return adapter_config

    def load_decoder_if_not_exist(self, model: nn.Module, layer_prefix: str, idx: int):
        try:
            decoder = model.get_submodule(layer_prefix)
        except AttributeError:
            # disable reset_parameters so that the weights will not be initialized
            # these initializations is not necessary because we will load it from the state_dict
            # and these initializations will cost too much time because the DeepSeekV3's decoder layer is too large
            with patch.object(nn.Linear, 'reset_parameters', lambda _self: None):
                get_logger().info('Creating decoder layer %s', idx)
                module_list: nn.ModuleList = model.layers
                template_module = module_list[0]
                decoder = template_module.__class__(layer_id=idx, args=self.config)

                state_dict = get_state_dict(self.model_path, decoder, prefix=layer_prefix)
                auto_dequant_state_dict(layer_prefix, state_dict, str(self.model_path))
                decoder.load_state_dict(state_dict)
                decoder.eval()
                module_list.append(decoder)
                get_logger().info('Create decoder layer %s successfully', idx)
        return decoder

    def load_mtp_decoder_if_not_exist(self, model: nn.Module, layer_prefix: str, mtp_idx: int):
        try:
            mtp_block = model.mtp[mtp_idx]
        except (IndexError, AttributeError):
            with patch.object(nn.Linear, 'reset_parameters', lambda _self: None):
                get_logger().info('Creating MTP decoder layer %s', mtp_idx)
                layer_id = self.config.num_hidden_layers + mtp_idx
                mtp_block = Block(layer_id, self.config)

                state_dict = get_state_dict(self.model_path, mtp_block, prefix=layer_prefix)
                auto_dequant_state_dict(layer_prefix, state_dict, str(self.model_path))
                mtp_block.load_state_dict(state_dict, strict=False)

                mtp_extra = get_mtp_layer(
                    config=self.config,
                    model_path=str(self.model_path),
                    layer_prefix=layer_prefix,
                )
                wrap_mtp_decoder(mtp_block, mtp_extra)

                mtp_block.eval()
                model.mtp.append(mtp_block)
                get_logger().info('Create MTP decoder layer %s successfully', mtp_idx)
        return mtp_block

    def generate_decoder_layer(self, model: nn.Module):
        for idx in range(self.config.num_hidden_layers):
            layer_prefix = f"layers.{idx}"
            decoder = self.load_decoder_if_not_exist(model, layer_prefix=layer_prefix, idx=idx)
            yield layer_prefix, decoder
        for mtp_idx in range(getattr(self.config, 'n_mtp_layers', 0)):
            layer_prefix = f"mtp.{mtp_idx}"
            decoder = self.load_mtp_decoder_if_not_exist(model, layer_prefix=layer_prefix, mtp_idx=mtp_idx)
            yield layer_prefix, decoder

    def _get_local_expert_range(self):
        """Get the local expert index range for EP mode."""
        if dist.is_initialized():
            world_size = dist.get_world_size()
            rank = dist.get_rank()
        else:
            world_size = 1
            rank = 0
        if hasattr(self.config, 'n_routed_experts'):
            expert_num = self.config.n_routed_experts
        elif hasattr(self.config, 'num_experts'):
            expert_num = self.config.num_experts
        else:
            expert_num = 0
        n_local = expert_num // world_size
        start = rank * n_local
        end = start + n_local
        return start, end, expert_num

    def _iter_layers_and_mtp(self):
        """
        Iterate over model blocks in a stable order: first all `layers.*` then all `mtp.*`.

        Yields a 4-tuple:
        - `prefix` (str): `"layers"` or `"mtp"`.
        - `layer_or_mtp_idx` (int): index within `prefix` (`layers.{idx}` / `mtp.{idx}`).
        - `ratio` (int/float): for `layers.*` uses `self.config.compress_ratios[idx]` (fallback to `1`);
          for `mtp.*` it is always `1` (the original code used `ratio = 1` for mtp).
        - `is_mtp` (bool): `True` only for the `mtp.*` range.
        """
        num_layers = self.config.num_hidden_layers
        n_mtp = getattr(self.config, 'n_mtp_layers', 0)
        # layers.* blocks
        for layer_idx in range(num_layers):
            ratio = self.config.compress_ratios[layer_idx] if layer_idx < len(self.config.compress_ratios) else 1
            yield "layers", layer_idx, ratio, False
        # mtp.* blocks
        for mtp_idx in range(n_mtp):
            yield "mtp", mtp_idx, 1, True

    def get_ln_fuse_map(self):
        pre_ln_linear_map = {}
        # =========================== GLOBAL ==============================
        pre_ln_linear_map['norm'] = ['head']

        expert_start, expert_end, _ = self._get_local_expert_range()
        ln_linear_map = {}
        for prefix, layer_or_mtp_idx, ratio, is_mtp in self._iter_layers_and_mtp():
            # ============================ MOE =========================
            # Only include local experts for EP mode
            ln_linear_map[f"{prefix}.{layer_or_mtp_idx}.ffn_norm"] = [
                f"{prefix}.{layer_or_mtp_idx}.ffn.experts.{i}.{proj}"
                for proj in ["w1", "w3"]
                for i in range(expert_start, expert_end)
            ]
            # shared experts
            ln_linear_map[f"{prefix}.{layer_or_mtp_idx}.ffn_norm"] += [
                f"{prefix}.{layer_or_mtp_idx}.ffn.shared_experts.{proj}" for proj in ["w1", "w3"]
            ]
            # expert gate
            ln_linear_map[f"{prefix}.{layer_or_mtp_idx}.ffn_norm"] += [f"{prefix}.{layer_or_mtp_idx}.ffn.gate"]

            # ============================ Attention =========================
            # 根据层类型添加不同的Attention配置
            if ratio <= 1:
                ln_linear_map[f"{prefix}.{layer_or_mtp_idx}.attn_norm"] = [
                    f"{prefix}.{layer_or_mtp_idx}.attn.wq_a",
                    f"{prefix}.{layer_or_mtp_idx}.attn.wkv",
                ]
                ln_linear_map[f"{prefix}.{layer_or_mtp_idx}.attn.q_norm"] = [
                    f"{prefix}.{layer_or_mtp_idx}.attn.wq_b",
                ]
            elif ratio == 4:
                ln_linear_map[f"{prefix}.{layer_or_mtp_idx}.attn_norm"] = [
                    f"{prefix}.{layer_or_mtp_idx}.attn.wq_a",
                    f"{prefix}.{layer_or_mtp_idx}.attn.wkv",
                    f"{prefix}.{layer_or_mtp_idx}.attn.compressor.wgate",  # 2,3,...
                    f"{prefix}.{layer_or_mtp_idx}.attn.compressor.wkv",  # 2,3,...
                    f"{prefix}.{layer_or_mtp_idx}.attn.indexer.weights_proj",  # 2,4,...
                    f"{prefix}.{layer_or_mtp_idx}.attn.indexer.compressor.wgate",  # 2,4,...
                    f"{prefix}.{layer_or_mtp_idx}.attn.indexer.compressor.wkv",  # 2,4,...
                ]
                ln_linear_map[f"{prefix}.{layer_or_mtp_idx}.attn.q_norm"] = [
                    f"{prefix}.{layer_or_mtp_idx}.attn.wq_b",
                    f"{prefix}.{layer_or_mtp_idx}.attn.indexer.wq_b",
                ]  # 2,4,...
            else:
                ln_linear_map[f"{prefix}.{layer_or_mtp_idx}.attn_norm"] = [
                    f"{prefix}.{layer_or_mtp_idx}.attn.wq_a",
                    f"{prefix}.{layer_or_mtp_idx}.attn.wkv",
                    f"{prefix}.{layer_or_mtp_idx}.attn.compressor.wgate",  # 2,3,...
                    f"{prefix}.{layer_or_mtp_idx}.attn.compressor.wkv",  # 2,3,...
                ]
                ln_linear_map[f"{prefix}.{layer_or_mtp_idx}.attn.q_norm"] = [
                    f"{prefix}.{layer_or_mtp_idx}.attn.wq_b",
                ]

            # ============================ MTP 专有结构 =========================
            if is_mtp:
                ln_linear_map[f"{prefix}.{layer_or_mtp_idx}.enorm"] = [
                    f"{prefix}.{layer_or_mtp_idx}.e_proj",
                ]
                ln_linear_map[f"{prefix}.{layer_or_mtp_idx}.hnorm"] = [
                    f"{prefix}.{layer_or_mtp_idx}.h_proj",
                ]
                ln_linear_map[f"{prefix}.{layer_or_mtp_idx}.norm"] = [
                    f"{prefix}.{layer_or_mtp_idx}.head",
                ]

        return pre_ln_linear_map, ln_linear_map

    def get_bake_names(self):
        return [], []

    def get_rotate_map(self, block_size):
        expert_start, expert_end, _ = self._get_local_expert_range()
        rot_pairs = {}

        # chain rot
        rot = QuaRotInterface.get_rotate_command(
            size=self.config.dim,
            mode=QuaRotInterface.QuaRotMode.HADAMARD,
            block_size=block_size,
        )
        # ============================= GLOBAL ==============================
        pre_left_rot = {}
        pre_right_rot = {}
        pre_right_rot['embed'] = rot
        pre_right_rot['hc_head_fn'] = rot
        pre_right_rot['head'] = rot
        pre_run = QuaRotInterface.RotatePair(left_rot=pre_left_rot, right_rot=pre_right_rot)

        left_rot = {}
        right_rot = {}
        # ================================== 主模型 ==============================
        for prefix, layer_or_mtp_idx, ratio, is_mtp in self._iter_layers_and_mtp():
            # ============================ MOE =========================
            for i in range(expert_start, expert_end):
                left_rot[f"{prefix}.{layer_or_mtp_idx}.ffn.experts.{i}.w2"] = rot
            left_rot[f"{prefix}.{layer_or_mtp_idx}.ffn.shared_experts.w2"] = rot

            right_rot[f"{prefix}.{layer_or_mtp_idx}.hc_ffn_fn"] = rot
            for i in range(expert_start, expert_end):
                right_rot[f"{prefix}.{layer_or_mtp_idx}.ffn.experts.{i}.w1"] = rot
                right_rot[f"{prefix}.{layer_or_mtp_idx}.ffn.experts.{i}.w3"] = rot
            right_rot[f"{prefix}.{layer_or_mtp_idx}.ffn.shared_experts.w1"] = rot
            right_rot[f"{prefix}.{layer_or_mtp_idx}.ffn.shared_experts.w3"] = rot
            right_rot[f"{prefix}.{layer_or_mtp_idx}.ffn.gate"] = rot

            # ============================ Attention =========================
            left_rot[f"{prefix}.{layer_or_mtp_idx}.attn.wo_b"] = rot
            right_rot[f"{prefix}.{layer_or_mtp_idx}.hc_attn_fn"] = rot
            if ratio <= 1:
                right_rot[f"{prefix}.{layer_or_mtp_idx}.attn.wq_a"] = rot
                right_rot[f"{prefix}.{layer_or_mtp_idx}.attn.wkv"] = rot
            elif ratio == 4:
                right_rot[f"{prefix}.{layer_or_mtp_idx}.attn.wq_a"] = rot
                right_rot[f"{prefix}.{layer_or_mtp_idx}.attn.wkv"] = rot
                right_rot[f"{prefix}.{layer_or_mtp_idx}.attn.compressor.wgate"] = rot
                right_rot[f"{prefix}.{layer_or_mtp_idx}.attn.compressor.wkv"] = rot
                right_rot[f"{prefix}.{layer_or_mtp_idx}.attn.indexer.weights_proj"] = rot
                right_rot[f"{prefix}.{layer_or_mtp_idx}.attn.indexer.compressor.wgate"] = rot
                right_rot[f"{prefix}.{layer_or_mtp_idx}.attn.indexer.compressor.wkv"] = rot
            else:
                right_rot[f"{prefix}.{layer_or_mtp_idx}.attn.wq_a"] = rot
                right_rot[f"{prefix}.{layer_or_mtp_idx}.attn.wkv"] = rot
                right_rot[f"{prefix}.{layer_or_mtp_idx}.attn.compressor.wgate"] = rot
                right_rot[f"{prefix}.{layer_or_mtp_idx}.attn.compressor.wkv"] = rot

            # ============================= MTP 专有结构 ========================
            if is_mtp:
                right_rot[f"{prefix}.{layer_or_mtp_idx}.h_proj"] = rot
                right_rot[f"{prefix}.{layer_or_mtp_idx}.head"] = rot
                right_rot[f"{prefix}.{layer_or_mtp_idx}.emb.tok_emb"] = rot
                right_rot[f"{prefix}.{layer_or_mtp_idx}.e_proj"] = rot

                left_rot[f"{prefix}.{layer_or_mtp_idx}.e_proj"] = rot
                left_rot[f"{prefix}.{layer_or_mtp_idx}.h_proj"] = rot

        rot_pairs['rot'] = QuaRotInterface.RotatePair(left_rot=left_rot, right_rot=right_rot)

        # q_b_proj rot
        rot_b_proj = QuaRotInterface.get_rotate_command(
            size=self.config.q_lora_rank,
            mode=QuaRotInterface.QuaRotMode.BLOCK_HADAMARD_SHIFTED,
            block_size=block_size,
        )
        left_rot_b_proj = {}
        right_rot_b_proj = {}
        for prefix, layer_or_mtp_idx, ratio, is_mtp in self._iter_layers_and_mtp():
            # =============================== Attention =========================
            left_rot_b_proj[f"{prefix}.{layer_or_mtp_idx}.attn.wq_a"] = rot_b_proj
            if ratio <= 1:
                right_rot_b_proj[f"{prefix}.{layer_or_mtp_idx}.attn.wq_b"] = rot_b_proj
            elif ratio == 4:
                right_rot_b_proj[f"{prefix}.{layer_or_mtp_idx}.attn.wq_b"] = rot_b_proj
                right_rot_b_proj[f"{prefix}.{layer_or_mtp_idx}.attn.indexer.wq_b"] = rot_b_proj
            else:
                right_rot_b_proj[f"{prefix}.{layer_or_mtp_idx}.attn.wq_b"] = rot_b_proj
        rot_pairs["rot_b_proj"] = QuaRotInterface.RotatePair(left_rot=left_rot_b_proj, right_rot=right_rot_b_proj)
        return [pre_run], list(rot_pairs.values())

    def ascendv1_save_module_preprocess(
        self, prefix: str, module: nn.Module, model: nn.Module
    ) -> Tuple[str, nn.Module]:
        return prefix, module

    def build_meta_model(self) -> nn.Module:
        """Build the layer-offloaded evaluator for the DeepSeek-V4 backbone."""
        from .fake_quant_inference import build_meta_model

        model = build_meta_model(self.config)
        model.eos_token_id = self.tokenizer.eos_token_id
        return model

    # ===== FA3QuantAdapterInterface =====
    def inject_fa3_placeholders(
        self,
        root_name: str,
        root_module: nn.Module,
        should_inject: Callable[[str], bool],
    ) -> None:
        """在真实 Q 和共享 KV 边界安装占位模块，供 FA3 Processor 采集和量化。

        Attention 在注意力计算前量化 Q 和拼接后的共享 KV，Indexer 使用 Q/压缩 KV。
        同一份 latent KV 同时参与 QK/PV，保留原有缓存写入流程。
        """
        from msmodelslim.core.quantization_algorithm.quant.fa3 import FA3QuantPlaceHolder
        from .model import Attention, Indexer, rotate_activation

        def _wrap_indexer_forward(indexer_mod: nn.Module):
            deepseek_module = import_module(indexer_mod.forward.__module__)
            apply_rotary_emb = deepseek_module.apply_rotary_emb

            @wraps(indexer_mod.forward)
            def new_indexer_forward(
                self,
                x: torch.Tensor,
                qr: torch.Tensor,
                start_pos: int,
                freqs_cis: torch.Tensor,
                offset: int,
            ):
                bsz, seqlen, _ = x.size()
                ratio = self.compress_ratio
                rd = self.rope_head_dim
                end_pos = start_pos + seqlen
                if self.compressor.kv_cache is None:
                    self.compressor.kv_cache = self.kv_cache
                q = self.wq_b(qr)
                n_heads = getattr(self, "n_local_heads", self.n_heads)
                q = q.reshape(bsz, seqlen, n_heads, self.head_dim)
                apply_rotary_emb(q[..., -rd:], freqs_cis)
                q = rotate_activation(q)

                # ===== 插入 FA3 占位 =====
                if hasattr(self, 'fa_q'):
                    q_input = q.transpose(1, 2).contiguous()
                    q_output = self.fa_q(q_input)
                    q = q_output.transpose(1, 2).contiguous()

                # =========================

                self.compressor(x, start_pos, freqs_cis)
                kv_len = end_pos // ratio
                k = self.kv_cache[:bsz, :kv_len]
                if kv_len > 0:
                    k_input = k.unsqueeze(1)
                    if hasattr(self, "fa_v"):
                        _ = self.fa_v(k_input)
                    if hasattr(self, "fa_k"):
                        k_output = self.fa_k(k_input)
                        k = k_output.squeeze(1)

                weights = self.weights_proj(x) * (self.softmax_scale * self.n_heads**-0.5)
                index_score = torch.einsum("bshd,btd->bsht", q, k)
                index_score = (index_score.relu_() * weights.unsqueeze(-1)).sum(dim=2)
                if start_pos == 0:
                    device = index_score.device
                    mask = (
                        torch.arange(seqlen // ratio, device=device).repeat(seqlen, 1)
                        >= torch.arange(1, seqlen + 1, device=device).unsqueeze(1) // ratio
                    )
                    index_score += torch.where(mask, float("-inf"), 0)
                topk_idxs = index_score.topk(min(self.index_topk, end_pos // ratio), dim=-1)[1]
                if start_pos == 0:
                    mask = topk_idxs >= torch.arange(1, seqlen + 1, device=topk_idxs.device).unsqueeze(1) // ratio
                    topk_idxs = torch.where(mask, -1, topk_idxs + offset)
                else:
                    topk_idxs += offset
                return topk_idxs

            # Bind the original A6 forward; this does not invoke it.
            # pylint: disable-next=no-value-for-parameter
            indexer_mod.forward = new_indexer_forward.__get__(indexer_mod, indexer_mod.__class__)

        def _wrap_attention_forward(attn_mod: nn.Module):
            from .model import apply_rotary_emb, get_window_topk_idxs, get_compress_topk_idxs, sparse_attn

            attention_interface = sparse_attn

            def new_forward(self, x: torch.Tensor, start_pos: int):
                """
                Forward pass for the Multi-Head Latent Attention (MLA) Layer.

                Args:
                    x (torch.Tensor): Input tensor of shape (batch_size, seq_len, dim).
                    start_pos (int): Starting position in the sequence for caching.

                Returns:
                    torch.Tensor: Output tensor with the same shape as the input.
                """
                bsz, seqlen, _ = x.size()
                freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]
                win = self.window_size
                ratio = self.compress_ratio
                rd = self.rope_head_dim
                if self.compress_ratio > 1 and self.compressor.kv_cache is None:
                    self.compressor.kv_cache = self.kv_cache[:, win:]
                # q
                qr = q = self.q_norm(self.wq_a(x))
                q = self.wq_b(q).unflatten(-1, (self.n_local_heads, self.head_dim))
                q *= torch.rsqrt(q.square().mean(-1, keepdim=True) + self.eps)
                apply_rotary_emb(q[..., -rd:], freqs_cis)

                # win kv & topk_idxs
                kv = self.wkv(x)
                kv = self.kv_norm(kv)
                apply_rotary_emb(kv[..., -rd:], freqs_cis)
                topk_idxs = get_window_topk_idxs(win, bsz, seqlen, start_pos).to(x.device)
                if self.compress_ratio > 1:
                    offset = kv.size(1) if start_pos == 0 else win
                    if self.indexer is not None:
                        compress_topk_idxs = self.indexer(x, qr, start_pos, freqs_cis, offset)
                    else:
                        compress_topk_idxs = get_compress_topk_idxs(ratio, bsz, seqlen, start_pos, offset).to(x.device)
                    topk_idxs = torch.cat([topk_idxs, compress_topk_idxs], dim=-1)
                topk_idxs = topk_idxs.int()

                # compress kv & attn
                # if start_pos == 0:
                # Preserve the native Attention prefill branch.
                if True:  # pylint: disable=using-constant-test
                    self.kv_cache[:bsz, : min(win, seqlen)] = kv[:, -win:]
                    if self.compress_ratio > 1:
                        if (kv_compress := self.compressor(x, start_pos, freqs_cis)) is not None:
                            kv = torch.cat([kv, kv_compress], dim=1)
                    # We performed QAT here, kv could also use fp8 format, though current implementation uses bf16
                    # ===== FA3 placeholders (after RoPE / cache, before attention kernel) =====
                    if hasattr(self, "fa_q"):
                        q = self.fa_q(q.transpose(1, 2).contiguous()).transpose(1, 2).contiguous()
                    if hasattr(self, "fa_k"):
                        # K and V share the same latent: (B, T, D) -> (B, 1, T, D).
                        kv = self.fa_k(kv.unsqueeze(1)).squeeze(1)
                    # ========================================================================
                    o = attention_interface(q, kv, self.attn_sink, topk_idxs, self.softmax_scale)
                else:
                    self.kv_cache[:bsz, start_pos % win] = kv.squeeze(1)
                    if self.compress_ratio > 1:
                        self.compressor(x, start_pos, freqs_cis)
                    kv = self.kv_cache[:bsz]
                    # ===== FA3 placeholders (after RoPE / cache, before attention kernel) =====
                    if hasattr(self, "fa_q"):
                        q = self.fa_q(q.transpose(1, 2).contiguous()).transpose(1, 2).contiguous()
                    if hasattr(self, "fa_k"):
                        # K and V share the same latent: (B, T, D) -> (B, 1, T, D).
                        kv = self.fa_k(kv.unsqueeze(1)).squeeze(1)
                    # ========================================================================
                    o = attention_interface(q, kv, self.attn_sink, topk_idxs, self.softmax_scale)
                apply_rotary_emb(o[..., -rd:], freqs_cis, True)

                # o
                o = o.view(bsz, seqlen, self.n_local_groups, -1)
                # self.wo_a.weight [8192, 4096]; self.n_local_groups=8; self.o_lora_rank=1024
                wo_a = self.wo_a.weight.view(self.n_local_groups, self.o_lora_rank, -1)
                o = torch.einsum("bsgd,grd->bsgr", o, wo_a)
                x = self.wo_b(o.flatten(2))
                return x

            attn_mod.forward = MethodType(new_forward, attn_mod)

        for name, module in list(root_module.named_modules()):
            full_name = ".".join(part for part in (root_name, name) if part)
            if isinstance(module, Attention):
                boundaries = (("fa_q", 1.0), ("fa_k", 1.0))
            elif isinstance(module, Indexer):
                # Retain the legacy V export slot when the recipe selects it.
                boundaries = (("fa_q", 0.9999), ("fa_k", 0.9999), ("fa_v", 1.0))
            else:
                continue

            parent_selected = should_inject(full_name)
            selected = False
            for boundary, ratio in boundaries:
                target_name = f"{full_name}.{boundary}" if full_name else boundary
                if not parent_selected and not should_inject(target_name):
                    continue
                selected = True
                # Preserve observers and IR installed by an earlier FA3 processor.
                if not hasattr(module, boundary):
                    num_heads = 1
                    if boundary == "fa_q":
                        num_heads = (
                            module.n_local_heads
                            if isinstance(module, Attention)
                            else getattr(module, "n_local_heads", module.n_heads)
                        )
                    module.add_module(
                        boundary,
                        FA3QuantPlaceHolder(
                            ratio=ratio,
                            num_heads=num_heads,
                        ),
                    )
            if not selected:
                continue
            if isinstance(module, Indexer):
                _wrap_indexer_forward(module)
            else:
                _wrap_attention_forward(module)

    def _load_config(self, trust_remote_code=False) -> object:
        config_data = json_safe_load(os.path.join(self.model_path, "config.json"))
        args = ModelArgs()
        args_mapping = self._get_args_mapping()
        for arg_name, mapping_name in args_mapping.items():
            setattr(args, arg_name, config_data[mapping_name])

        # args.num_hidden_layers = 4
        # args.compress_ratios = [128,128, 4, 128, 0]
        # args.compress_ratios [0, 0, 4, 128, 0] -> [1, 1, 4, 128, 1]
        if getattr(args, "compress_ratios", None) is not None:
            args.compress_ratios = [1 if r == 0 else r for r in args.compress_ratios]

        # Detect n_mtp_layers
        n_mtp = config_data.get('n_mtp_layers', 0)
        if n_mtp == 0:
            weight_map_path = os.path.join(str(self.model_path), "model.safetensors.index.json")
            if os.path.exists(weight_map_path):
                weight_map = json_safe_load(weight_map_path)
                if any(k.startswith('mtp.0.') for k in weight_map.get('weight_map', {})):
                    n_mtp = 1
        args.n_mtp_layers = n_mtp

        return args

    @classmethod
    def _get_args_mapping(cls) -> dict:
        return {
            "dim": "hidden_size",
            "moe_inter_dim": "moe_intermediate_size",
            "index_topk": "index_topk",
            "n_routed_experts": "n_routed_experts",
            "n_heads": "num_attention_heads",
            "num_hidden_layers": "num_hidden_layers",
            "o_groups": "o_groups",
            "q_lora_rank": "q_lora_rank",
            "route_scale": "routed_scaling_factor",
            "compress_ratios": "compress_ratios",
        }
