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
from collections import defaultdict
from functools import lru_cache
from typing import List, Any, Generator, Optional, Tuple, Dict, Union, Callable
from unittest.mock import patch

import torch
from safetensors import safe_open
from torch import distributed as dist
from torch import nn
from tqdm import tqdm

from msmodelslim.app.naive_quantization.model_info_interface import ModelInfoInterface
from msmodelslim.core.base.protocol import ProcessRequest
from msmodelslim.core.const import DeviceType
from msmodelslim.core.graph import AdapterConfig, MappingConfig, FusionConfig
from msmodelslim.ir import QuaRotExtraInfoWrapperIR
from msmodelslim.processor.quarot import QuaRotInterface
from msmodelslim.utils.exception import InvalidModelError, UnsupportedError
from msmodelslim.utils.logging import logger_setter, get_logger
from msmodelslim.utils.security import (
    get_valid_read_path,
    json_safe_load,
    json_safe_dump,
    MAX_READ_FILE_SIZE_32G,
)
from msmodelslim.utils.security.model import SafeGenerator
from .model import Transformer, NextPredDecoderLayer
from .quarot import _get_full_expert_range, get_ln_fuse_map, get_rotate_map
from .mtp_quant_module import MTPExtraModule, wrap_mtp_decoder, remove_zero_and_shift
from ..common.layer_wise_forward import TransformersForwardBreak
from ..common.transformers import TransformersModel
from ..glm_5.convert_fp8_to_bf16 import auto_convert_module_fp8_to_bf16
from ..interface_hub import (
    ModelSlimPipelineInterfaceV1,
    FlexSmoothQuantInterface,
    AscendV1SaveInterface,
)


@logger_setter("msmodelslim.model.glm_5_next")
class Glm5NextModelAdapter(  # pylint: disable=too-many-ancestors
    TransformersModel,
    ModelInfoInterface,
    ModelSlimPipelineInterfaceV1,
    FlexSmoothQuantInterface,
    QuaRotInterface,
    AscendV1SaveInterface,
):
    def get_model_pedigree(self) -> str:
        return 'glm_5_next'

    def get_model_type(self) -> str:
        return self.model_type

    def _get_text_config(self):
        """获取 text config：多模态 config（Glm5NextConfig）时取其 text_config。"""
        if hasattr(self.config, 'text_config') and self.config.text_config is not None:
            return self.config.text_config
        return self.config

    def handle_dataset(self, dataset: Any, device: DeviceType = DeviceType.NPU) -> List[Any]:
        return self._get_tokenized_data(dataset, device)

    def init_model(self, device: DeviceType = DeviceType.NPU) -> nn.Module:
        torch.set_default_dtype(torch.bfloat16)
        text_config = self._get_text_config()
        origin_num_hidden_layers = text_config.num_hidden_layers

        # 创建骨架模型：只含 1 个模板层，避免一次性加载所有层的巨大内存开销
        # 始终使用 text_config（vocab_size/hidden_size 等属性在其上）
        text_config.num_hidden_layers = 1
        with torch.device("cpu"):
            model = Transformer(text_config)

        # 恢复原始层数
        text_config.num_hidden_layers = origin_num_hidden_layers

        # 只加载第 0 层 + 非层权重（embed_tokens, norm, lm_head）
        state_dict = self.get_state_dict(model)
        model.load_state_dict(state_dict, strict=False)
        auto_convert_module_fp8_to_bf16("", model, str(self.model_path))
        self._wrap_bare_param_gate(model)
        model.eval()
        # ViT 不参与量化：挂到模型树上，保存时由 saver 作为 FLOAT 权重
        # 写入 quant_model_weights 分片与 index（与其他多模态模型一致）
        self._attach_visual_module(model)
        get_logger().info(
            "Create GLM5Next skeleton model with 1 layer successfully, total %s layers will be loaded on demand",
            origin_num_hidden_layers,
        )
        return model

    def _attach_visual_module(self, model: nn.Module) -> None:
        """将不参与量化的视觉塔挂载到模型树上（保持 bf16）。

        AscendV1 saver 在 post_run 遍历该子树，将权重以 FLOAT 写入常规
        quant_model_weights 分片，与其他多模态适配器（qwen3_vl / glm4_6v）一致。
        权重按嵌套普通模块重建，使 named_parameters 产出与 checkpoint 完全一致的
        key（`model.visual.*`）。
        """
        visual_state = {}
        for file_name in sorted(os.listdir(self.model_path)):
            if not file_name.endswith('.safetensors'):
                continue
            src_path = os.path.join(self.model_path, file_name)
            try:
                with safe_open(src_path, framework='pt', device='cpu') as f:
                    for key in f.keys():
                        if key.startswith('model.visual.'):
                            visual_state[key] = f.get_tensor(key)
            except Exception as err:  # pylint: disable=broad-except
                get_logger().debug("Skip %s when scanning vision weights: %s", file_name, err)
                continue
        if not visual_state:
            return
        visual = nn.Module()
        for key in sorted(visual_state):
            parts = key[len('model.visual.') :].split('.')
            parent = visual
            for part in parts[:-1]:
                if not hasattr(parent, part):
                    parent.add_module(part, nn.Module())
                parent = getattr(parent, part)
            parent.register_parameter(parts[-1], nn.Parameter(visual_state[key], requires_grad=False))
        model.model.visual = visual
        get_logger().info(
            "Attached %s vision tensor(s) onto model.model.visual, they will be exported as FLOAT weights",
            len(visual_state),
        )

    def _has_visual_module(self) -> bool:
        """判断源 checkpoint 是否携带视觉塔（model.visual.* 权重）。

        纯文本 checkpoint 不应注册 merger 旋转路径，否则 _rotate 找不到目标。
        """
        for file_name in sorted(os.listdir(self.model_path)):
            if not file_name.endswith('.safetensors'):
                continue
            src_path = os.path.join(self.model_path, file_name)
            try:
                with safe_open(src_path, framework='pt', device='cpu') as f:
                    for key in f.keys():
                        if key.startswith('model.visual.'):
                            return True
            except Exception as err:  # pylint: disable=broad-except
                get_logger().debug("Skip %s when checking visual weights: %s", file_name, err)
                continue
        return False

    def load_decoder_if_not_exist(self, model: nn.Module, name: str, idx: int):
        """按需创建并加载 decoder layer（懒加载，参考 glm_5_2）。

        骨架模型只含 layer 0，后续层首次访问时创建并加载；
        MTP 层（idx >= num_hidden_layers）使用 NextPredDecoderLayer。
        """
        try:
            decoder = model.get_submodule(name)
        except AttributeError:
            with patch.object(nn.Linear, 'reset_parameters', lambda _self: None):
                get_logger().info("Creating decoder layer %s...", idx)
                module_list = model.model.language_model.layers
                text_config = self._get_text_config()

                # MTP 层使用 NextPredDecoderLayer（不含 HC）
                if idx >= text_config.num_hidden_layers:
                    decoder = NextPredDecoderLayer(text_config, layer_idx=idx)
                else:
                    template_module = module_list[0]
                    decoder = template_module.__class__(text_config, layer_idx=idx)

                state_dict = self.get_state_dict(decoder, prefix=name)
                if state_dict:
                    decoder.load_state_dict(state_dict)
                auto_convert_module_fp8_to_bf16(name, decoder, str(self.model_path))
                self._wrap_bare_param_gate(decoder)

                decoder.eval()
                module_list.append(decoder)
                get_logger().info("Create decoder layer %s successfully", idx)
        return decoder

    def generate_model_visit(self, model: nn.Module) -> Generator[ProcessRequest, Any, None]:
        if dist.is_initialized():
            dist.barrier()

        for name, block in self.generate_decoder_layer(model):
            yield ProcessRequest(name, block, tuple(), {})

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

        # GLM5Next 的 decoder 层位于 model.language_model.layers
        layers = model.model.language_model.layers
        remove_handler = layers[0].register_forward_pre_hook(break_hook, with_kwargs=True, prepend=True)

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

        if dist.is_initialized():
            dist.barrier()

        args, kwargs = first_block_input
        # DecoderLayer.forward(..., prev_topk_indices=None) -> (hidden_states, topk_indices)
        hidden_states = args[0] if args else kwargs.get('hidden_states')

        topk_indices = None
        for name, block in self.generate_decoder_layer(model):
            text_config = self._get_text_config()
            layer_idx = int(name.split('.')[-1])

            # MTP 层走 mtp_preprocess（见其 docstring）
            if layer_idx >= text_config.num_hidden_layers:
                hidden_states = self.mtp_preprocess(model, block, hidden_states, inputs)

            kwargs = dict(kwargs)
            kwargs['prev_topk_indices'] = topk_indices
            outputs = yield ProcessRequest(name, block, (hidden_states,), kwargs)
            if isinstance(outputs, tuple) and len(outputs) >= 2:
                hidden_states = outputs[0]
                topk_indices = outputs[1]
            elif isinstance(outputs, torch.Tensor):
                hidden_states = outputs
            else:
                hidden_states = outputs[0] if isinstance(outputs, tuple) else outputs

    def mtp_preprocess(
        self,
        model: nn.Module,
        mtp_decoder: nn.Module,
        hidden_states: torch.Tensor,
        inputs: Any,
    ) -> torch.Tensor:
        """MTP 预处理：将上层 4D HC 输出转换为 MTP 层 3D 输入（参考 glm5 实现）。

        流程：collapse 4D→3D → norm + lm_head → logits → remove_zero_and_shift
        替换最后一个 token → embed_tokens → enorm → hnorm → cat → eh_proj。
        """
        # 1. collapse 4D HC streams → 3D
        if hidden_states.dim() == 4:
            pre_hidden_states = hidden_states.mean(dim=2)
        else:
            pre_hidden_states = hidden_states

        # 2. norm + lm_head → logits
        # 顶层 norm/lm_head 留在 CPU，vocab=154880 大 matmul 单线程会卡死，临时上 NPU 算完搬回
        mtp_device = mtp_decoder.embed_tokens.weight.device
        was_norm_cpu = model.model.language_model.norm.weight.device
        was_head_cpu = model.lm_head.weight.device
        try:
            model.model.language_model.norm.to(mtp_device)
            model.lm_head.to(mtp_device)
            normed = model.model.language_model.norm(pre_hidden_states.to(mtp_device))
            logits = model.lm_head(normed).float()
        finally:
            model.model.language_model.norm.to(was_norm_cpu)
            model.lm_head.to(was_head_cpu)

        # 3. 获取原始 input_ids
        input_ids = None
        if isinstance(inputs, dict):
            input_ids = inputs.get('input_ids')
        elif isinstance(inputs, (list, tuple)):
            input_ids = inputs[0] if isinstance(inputs[0], torch.Tensor) else None
        if input_ids is None:
            # 无法获取 input_ids，跳过 MTP 预处理
            return pre_hidden_states

        # 4. remove_zero_and_shift + 替换最后一个 token 为预测值
        input_ids_mtp = remove_zero_and_shift(input_ids)
        mtp_device = mtp_decoder.embed_tokens.weight.device
        input_ids_mtp = input_ids_mtp.to(mtp_device)
        input_ids_mtp[:, -1] = logits[:, -1, :].argmax(dim=1)

        # 5. embed_tokens → enorm
        input_embeds_mtp = mtp_decoder.embed_tokens(input_ids_mtp)
        input_embeds_mtp = mtp_decoder.enorm(input_embeds_mtp)

        # 6. hnorm → cat → eh_proj（pre_hidden_states 因 post_offload 仍在 CPU，需对齐到 NPU）
        pre_hidden_states = pre_hidden_states.to(mtp_device)
        hidden_embeds_mtp = mtp_decoder.hnorm(pre_hidden_states)
        hidden_states_mtp = torch.cat([input_embeds_mtp, hidden_embeds_mtp], dim=-1)
        hidden_states_mtp = mtp_decoder.eh_proj(hidden_states_mtp)

        return hidden_states_mtp

    def enable_kv_cache(self, model: nn.Module, need_kv_cache: bool) -> None:
        pass

    def get_adapter_config_for_subgraph(self) -> List[AdapterConfig]:
        adapter_config = []
        text_config = self._get_text_config()
        expert_start, expert_end = _get_full_expert_range(text_config)
        hidden_layers = text_config.num_hidden_layers
        num_nextn = getattr(text_config, 'num_nextn_predict_layers', 0) or 0

        # 层范围：常规层 0..num_hidden_layers-1 + MTP 层 num_hidden_layers..+
        for layer_idx in range(hidden_layers + num_nextn):
            # MTP 层（idx >= num_hidden_layers）为 DSA + MoE，无 HC
            is_mtp = layer_idx >= hidden_layers
            if is_mtp:
                layer_type = "deepseek_sparse_attention"
            else:
                layer_type = text_config.layer_types[layer_idx]

            if layer_type == "deepseek_sparse_attention":
                # DSA attention：MLA 结构，涉及 q_a_proj/q_a_layernorm/q_b_proj/kv_a_proj/kv_b_proj/o_proj
                okv_b_mapping_config = MappingConfig(
                    source=f"model.language_model.layers.{layer_idx}.self_attn.kv_b_proj",
                    targets=[f"model.language_model.layers.{layer_idx}.self_attn.o_proj"],
                )

                input_norm_targets = [
                    f"model.language_model.layers.{layer_idx}.self_attn.q_a_proj",
                    f"model.language_model.layers.{layer_idx}.self_attn.kv_a_proj_with_mqa",
                ]
                qa_norm_targets = [
                    f"model.language_model.layers.{layer_idx}.self_attn.q_b_proj",
                ]

                if self._layer_has_indexer(layer_idx):
                    input_norm_targets.extend(
                        [
                            f"model.language_model.layers.{layer_idx}.self_attn.indexer.wk",
                            f"model.language_model.layers.{layer_idx}.self_attn.indexer.weights_proj",
                        ]
                    )
                    qa_norm_targets.append(f"model.language_model.layers.{layer_idx}.self_attn.indexer.wq_b")
                    # kpool compress_gate 与 wk/weights_proj 共同消费 input_layernorm
                    # 的输出，必须参与 norm-linear 迁移，否则平滑 scale 迁移后
                    # gate 权重得不到补偿。
                    if self._layer_has_kpool_compress():
                        input_norm_targets.append(
                            f"model.language_model.layers.{layer_idx}.self_attn.indexer.index_kpool_compress_gate"
                        )

                input_norm_mapping_config = MappingConfig(
                    source=f"model.language_model.layers.{layer_idx}.input_layernorm",
                    targets=input_norm_targets,
                )

                qa_norm_mapping_config = MappingConfig(
                    source=f"model.language_model.layers.{layer_idx}.self_attn.q_a_layernorm",
                    targets=qa_norm_targets,
                )

                adapter_config.extend(
                    [
                        AdapterConfig(
                            subgraph_type="ov",
                            mapping=okv_b_mapping_config,
                            extra_config={'group_method': 'max'},
                            fusion=FusionConfig(
                                fusion_type="kv",
                                num_attention_heads=text_config.num_attention_heads,
                                num_key_value_heads=text_config.num_key_value_heads,
                                custom_config={
                                    'qk_nope_head_dim': text_config.qk_nope_head_dim,
                                    'v_head_dim': text_config.v_head_dim,
                                },
                            ),
                        ),
                        AdapterConfig(subgraph_type="norm-linear", mapping=input_norm_mapping_config),
                        AdapterConfig(subgraph_type="norm-linear", mapping=qa_norm_mapping_config),
                    ]
                )
            else:
                # KDA (linear attention)：仅配置 norm-linear 子图，无 ov——
                # v→o 路径中 o_norm(RMSNormGated) 对 per-channel scale 不满足线性，
                # 且 ov 使用的全局 num_attention_heads 分组与 KDA linear_num_heads
                # 结构不符（参考 kimi_k3 _kda_subgraph_configs）。
                input_norm_targets = [
                    f"model.language_model.layers.{layer_idx}.self_attn.q_proj",
                    f"model.language_model.layers.{layer_idx}.self_attn.k_proj",
                    f"model.language_model.layers.{layer_idx}.self_attn.v_proj",
                    f"model.language_model.layers.{layer_idx}.self_attn.b_proj",
                    f"model.language_model.layers.{layer_idx}.self_attn.forget_gate.f_a_proj",
                    f"model.language_model.layers.{layer_idx}.self_attn.g_a_proj",
                ]

                input_norm_mapping_config = MappingConfig(
                    source=f"model.language_model.layers.{layer_idx}.input_layernorm",
                    targets=input_norm_targets,
                )

                adapter_config.extend(
                    [
                        AdapterConfig(subgraph_type="norm-linear", mapping=input_norm_mapping_config),
                    ]
                )

            # FFN 配置（两种 attention 类型通用）
            mlp_layer_types = text_config.mlp_layer_types
            if is_mtp:
                is_sparse = True
            else:
                is_sparse = (
                    mlp_layer_types and layer_idx < len(mlp_layer_types) and mlp_layer_types[layer_idx] == "sparse"
                )
            if not is_sparse:
                # 稠密 MLP：up_proj -> down_proj
                up_proj = f'model.language_model.layers.{layer_idx}.mlp.up_proj'
                down_proj = f'model.language_model.layers.{layer_idx}.mlp.down_proj'
                up_down_mapping_config = MappingConfig(
                    source=up_proj,
                    targets=[down_proj],
                )
                adapter_config.extend(
                    [
                        AdapterConfig(subgraph_type="up-down", mapping=up_down_mapping_config),
                    ]
                )
            else:
                # MoE 层：shared_experts + routed experts
                expert_up_proj = f'model.language_model.layers.{layer_idx}.mlp.shared_experts.up_proj'
                expert_down_proj = f'model.language_model.layers.{layer_idx}.mlp.shared_experts.down_proj'
                up_down_mapping_config_shared = MappingConfig(source=expert_up_proj, targets=[expert_down_proj])
                adapter_config.extend([AdapterConfig(subgraph_type="up-down", mapping=up_down_mapping_config_shared)])

                # routed experts：unstack 后的 per-expert nn.Linear
                for expert in range(expert_start, expert_end):
                    up_proj = f'model.language_model.layers.{layer_idx}.mlp.experts.{expert}.up_proj'
                    down_proj = f'model.language_model.layers.{layer_idx}.mlp.experts.{expert}.down_proj'
                    adapter_config.append(
                        AdapterConfig(
                            subgraph_type="up-down",
                            mapping=MappingConfig(source=up_proj, targets=[down_proj]),
                        )
                    )

        return adapter_config

    @lru_cache(maxsize=1)
    def get_weight_map(self):
        model_index_path = os.path.join(self.model_path, "model.safetensors.index.json")
        model_index = json_safe_load(model_index_path)
        return model_index['weight_map']

    def get_state_dict(self, module: nn.Module, prefix: str = ""):
        """从 safetensors 分片加载 state dict，按多种 key 策略匹配 weight_map。

        候选 key：完整名；去掉/替换 'model.language_model.' 前缀；顶层参数补/去
        'model.' 前缀；attn_hc/ffn_hc/forget_gate 嵌套名映射到扁平 checkpoint key。

        Args:
            module: 待加载权重的模块。
            prefix: 模块前缀路径（如 'model.language_model.layers.0'）。
        """
        weight_map = self.get_weight_map()
        strip_prefix = "model.language_model."
        names = map(lambda x: x[0], module.named_parameters())

        groups = defaultdict(list)
        for name in names:
            full_name = f'{prefix}.{name}' if prefix else name
            lookup_name = None

            candidates = [full_name]

            # 策略 2/3：去掉或替换 'model.language_model.' 前缀
            if full_name.startswith(strip_prefix):
                candidates.append(full_name[len(strip_prefix) :])
                # 如 model.layers.X.*
                candidates.append(full_name.replace("language_model.", "", 1))

            # 策略 4：顶层参数（无 model. 前缀）尝试补 'model.' 前缀
            # （如 lm_head.weight -> model.lm_head.weight）
            if not prefix and not full_name.startswith("model."):
                candidates.append(f"model.{full_name}")

            for candidate in candidates:
                if candidate in weight_map:
                    lookup_name = candidate
                    break

            # 策略 5：顶层参数尝试去掉 'model.' 前缀（如 model.lm_head.weight -> lm_head.weight）
            if lookup_name is None and full_name.startswith("model."):
                candidate = full_name[len("model.") :]
                if candidate in weight_map:
                    lookup_name = candidate

            # 策略 6：HyperConnection 参数在 checkpoint 中扁平存放
            # （hc_attn_fn/hc_attn_base/hc_attn_scale），模型中嵌套
            # （attn_hc.fn/attn_hc.base/attn_hc.scale），映射 嵌套 -> 扁平。
            if lookup_name is None and ".attn_hc." in full_name:
                flat_name = full_name.replace(".attn_hc.", ".hc_attn_")
                if flat_name in weight_map:
                    lookup_name = flat_name
            if lookup_name is None and ".ffn_hc." in full_name:
                flat_name = full_name.replace(".ffn_hc.", ".hc_ffn_")
                if flat_name in weight_map:
                    lookup_name = flat_name

            # 策略 7：ForgetGate 参数在 checkpoint 中扁平存放
            # （self_attn.f_a_proj/f_b_proj/dt_bias/A_log），模型中嵌套
            # （self_attn.forget_gate.*），映射 嵌套 -> 扁平。
            if lookup_name is None and ".forget_gate." in full_name:
                flat_name = full_name.replace(".forget_gate.", ".")
                if flat_name in weight_map:
                    lookup_name = flat_name

            if lookup_name is None:
                continue

            file_name = weight_map[lookup_name]
            groups[file_name].append((name, lookup_name))

        state_dict = {}
        for file_name in tqdm(groups, desc=f'Loading {prefix}'):
            file_path = os.path.join(self.model_path, file_name)
            file_path = get_valid_read_path(file_path, extensions='safetensors', size_max=MAX_READ_FILE_SIZE_32G)
            with safe_open(file_path, framework='pt', device='cpu') as f:
                for name, lookup_name in tqdm(groups[file_name], desc=f'Loading {file_path}'):
                    state_dict[name] = f.get_tensor(lookup_name)
        return state_dict

    def load_mtp_if_not_load(self, mtp_decoder: nn.Module):
        """为 MTP 层附加 enorm/hnorm/eh_proj/shared_head/embed_tokens（独立于层主体需单独加载）。

        shared_head.head.weight 与 lm_head、embed_tokens.weight 与主 embed_tokens
        共享，未在 weight_map 中单独出现，需从对应权重读取。
        """
        try:
            mtp_decoder.get_submodule('shared_head')
        except AttributeError:
            get_logger().info('Creating MTP extra modules for layer 45')
            text_config = self._get_text_config()
            mtp_extra = MTPExtraModule(text_config)
            mtp_prefix = f'model.language_model.layers.{text_config.num_hidden_layers}'

            state_dict = self.get_state_dict(mtp_extra, prefix=mtp_prefix)

            # 共享权重：shared_head.head.weight <- lm_head.weight
            #           embed_tokens.weight <- model.language_model.embed_tokens.weight
            head_state_dict = self.get_state_dict(mtp_extra.shared_head.head, prefix='lm_head')
            if 'weight' in head_state_dict:
                state_dict['shared_head.head.weight'] = head_state_dict['weight']
            embed_state_dict = self.get_state_dict(mtp_extra.embed_tokens, prefix='model.language_model.embed_tokens')
            if 'weight' in embed_state_dict:
                state_dict['embed_tokens.weight'] = embed_state_dict['weight']

            if state_dict:
                mtp_extra.load_state_dict(state_dict)
            auto_convert_module_fp8_to_bf16(mtp_prefix, mtp_extra, str(self.model_path))
            wrap_mtp_decoder(mtp_decoder=mtp_decoder, mtp_extra=mtp_extra)
            get_logger().info('Create MTP extra modules successfully')

    def generate_decoder_layer(self, model: nn.Module):
        text_config = self._get_text_config()
        for idx in range(text_config.num_hidden_layers):
            name = f"model.language_model.layers.{idx}"
            decoder = self.load_decoder_if_not_exist(model, name=name, idx=idx)
            yield name, decoder

        # MTP 层（num_nextn_predict_layers）
        num_nextn = getattr(text_config, 'num_nextn_predict_layers', 0) or 0
        for mtp_idx in range(num_nextn):
            idx = text_config.num_hidden_layers + mtp_idx
            name = f"model.language_model.layers.{idx}"
            mtp_decoder = self.load_decoder_if_not_exist(model, name=name, idx=idx)
            self.load_mtp_if_not_load(mtp_decoder)
            yield name, mtp_decoder

    def _layer_has_indexer(self, layer_idx: int) -> bool:
        # 仅 DSA（deepseek_sparse_attention）层有 indexer
        # MTP 层（idx >= num_hidden_layers）使用 DSA，有 indexer
        text_config = self._get_text_config()
        if layer_idx >= text_config.num_hidden_layers:
            # MTP 层总是 DSA + 有 indexer
            return True
        if hasattr(text_config, 'layer_types') and text_config.layer_types is not None:
            if (
                layer_idx >= len(text_config.layer_types)
                or text_config.layer_types[layer_idx] != "deepseek_sparse_attention"
            ):
                return False
        if hasattr(text_config, 'indexer_types') and text_config.indexer_types is not None:
            if layer_idx < len(text_config.indexer_types):
                return text_config.indexer_types[layer_idx] != "shared"
        return False

    def _layer_has_kpool_compress(self) -> bool:
        """是否启用 kpool 压缩（deploy 侧 index_kpool > 1 且 index_kpool_compress）。

        其 compress_gate 直接消费 input_layernorm 输出，需参与 γ-fold 与平滑补偿。
        """
        text_config = self._get_text_config()
        index_kpool = getattr(text_config, 'index_kpool', None)
        if index_kpool is None:
            return False
        try:
            index_kpool = int(index_kpool)
        except (TypeError, ValueError):
            return False
        return index_kpool > 1 and bool(getattr(text_config, 'index_kpool_compress', False))

    @staticmethod
    def _wrap_bare_param_gate(module: nn.Module) -> None:
        """将 Indexer 的裸 Parameter gate 包装为 nn.Linear（量化链路按模块路径解析）。

        裸 Parameter 会在 DTS submit 校验时触发 ``is not an nn.Module``；权重加载
        发生在包装之前（仍按裸参数名对齐 weight_map），包装不改加载语义。保存时由
        ascendv1_save_module_preprocess 拆回裸 Parameter，导出 key 与 checkpoint 一致。
        """
        targets = []
        for name, sub in module.named_modules():
            gate = getattr(sub, 'index_kpool_compress_gate', None)
            if isinstance(gate, nn.Parameter):
                targets.append((name, sub, gate))
        for name, sub, gate in targets:
            # gate 形状 (head_dim, hidden_size) 与 nn.Linear(hidden_size, head_dim).weight 一致
            linear = nn.Linear(gate.shape[1], gate.shape[0], bias=False, dtype=gate.dtype, device=gate.device)
            linear.weight.data.copy_(gate.data)
            linear._ms_bare_param_gate_proxy = True
            # 新版 torch __setattr__ 禁止向已注册为 Parameter 的名字赋 Module
            # （TypeError: cannot assign ... as parameter），先解除注册再赋值
            del sub._parameters['index_kpool_compress_gate']
            sub.index_kpool_compress_gate = linear
            get_logger().info("Wrapped bare parameter gate into nn.Linear: %s.index_kpool_compress_gate", name)

    def ascendv1_save_module_preprocess(  # pylint: disable=unused-argument
        self, prefix: str, module: nn.Module, model: nn.Module
    ) -> Tuple[str, nn.Module]:
        """保存前把包装的 gate 拆回裸 Parameter，使导出 key 与原始 checkpoint 一致。

        nn.Linear 形态会导出为 ``index_kpool_compress_gate.weight``，而 deploy 侧
        期望无 .weight 后缀；分布式 saver 残留的 gate 条目替换为哑模块避免多写出 .weight。
        """
        gate = getattr(module, 'index_kpool_compress_gate', None)
        if isinstance(gate, nn.Linear) and getattr(gate, '_ms_bare_param_gate_proxy', False):
            # nn.Module.__setattr__ 会自动将 gate 从 _modules 移除并注册为 Parameter，
            # 之后 Indexer 的 recurse=False named_parameters 即含该裸参数
            module.index_kpool_compress_gate = nn.Parameter(gate.weight.data, requires_grad=gate.weight.requires_grad)
        elif isinstance(gate, nn.Module) and not isinstance(gate, nn.Linear):
            # gate 被量化 IR 替换：保留替换后形态导出（.weight key）
            get_logger().warning("index_kpool_compress_gate at %s was replaced by quant IR, keep as-is", prefix)
        elif getattr(module, '_ms_bare_param_gate_proxy', False):
            # 分布式 saver 物化列表中的残留 gate 条目：哑模块无参数，不会写出任何 tensor
            return prefix, nn.Module()
        return prefix, module

    def get_ln_fuse_map(self):
        text_config = self._get_text_config()
        num_nextn = getattr(text_config, 'num_nextn_predict_layers', 0) or 0
        num_layers = text_config.num_hidden_layers + num_nextn
        ln_linear_map = get_ln_fuse_map(text_config, num_hidden_layers=num_layers)
        # 路径重映射：'model.layers.{idx}' -> 'model.language_model.layers.{idx}'
        old_prefix = "model.layers."
        new_prefix = "model.language_model.layers."
        remapped_map = {}
        for key, targets in ln_linear_map.items():
            new_key = key.replace(old_prefix, new_prefix, 1) if old_prefix in key else key
            new_targets = [t.replace(old_prefix, new_prefix, 1) if old_prefix in t else t for t in targets]
            remapped_map[new_key] = new_targets
        ln_linear_map = remapped_map

        # 重映射顶层 'model.norm' -> 'model.language_model.norm'
        if "model.norm" in ln_linear_map:
            ln_linear_map["model.language_model.norm"] = ln_linear_map.pop("model.norm")

        for layer_idx in range(num_layers):
            if not self._layer_has_indexer(layer_idx):
                continue
            ln_linear_map[f"model.language_model.layers.{layer_idx}.input_layernorm"].append(
                f"model.language_model.layers.{layer_idx}.self_attn.indexer.wk",
            )
            ln_linear_map[f"model.language_model.layers.{layer_idx}.input_layernorm"].append(
                f"model.language_model.layers.{layer_idx}.self_attn.indexer.weights_proj",
            )
            ln_linear_map[f"model.language_model.layers.{layer_idx}.self_attn.q_a_layernorm"].append(
                f"model.language_model.layers.{layer_idx}.self_attn.indexer.wq_b"
            )
            # kpool compress_gate 与 wk/weights_proj 共同消费 input_layernorm 的输出，
            # 必须一并 fold γ，否则导出后 gate 分数缺失 γ 补偿。
            if self._layer_has_kpool_compress():
                ln_linear_map[f"model.language_model.layers.{layer_idx}.input_layernorm"].append(
                    f"model.language_model.layers.{layer_idx}.self_attn.indexer.index_kpool_compress_gate",
                )

        # MTP 层 LN 融合：enorm + hnorm → eh_proj, shared_head.norm → shared_head.head
        if num_nextn > 0:
            mtp_layer_idx = text_config.num_hidden_layers
            mtp_prefix = f"model.language_model.layers.{mtp_layer_idx}"
            ln_linear_map[(f"{mtp_prefix}.enorm", f"{mtp_prefix}.hnorm")] = [
                f"{mtp_prefix}.eh_proj",
            ]
            ln_linear_map[f"{mtp_prefix}.shared_head.norm"] = [
                f"{mtp_prefix}.shared_head.head",
            ]

        return {}, ln_linear_map

    def get_bake_names(self):
        return [], []

    def get_rotate_map(self, block_size):
        text_config = self._get_text_config()
        num_layers = text_config.num_hidden_layers
        num_nextn = getattr(text_config, 'num_nextn_predict_layers', 0) or 0
        pre_run, rot_pairs, rotate_matrix = get_rotate_map(text_config, block_size, num_hidden_layers=num_layers)
        # 层名重映射：'model.layers.{idx}' -> 'model.language_model.layers.{idx}'
        # pre_run 中同时重映射 'model.embed_tokens' -> 'model.language_model.embed_tokens'
        self._remap_rot_pairs(rot_pairs, rotate_matrix, pre_run)

        # Visual merger 输出左旋对齐旋转域（残差流为 h·R；merger 仅 down_proj 输出
        # 进入残差流，W' = R^T @ W）。visual 塔不参与 preprocess forward，且 target
        # 落到 .weight 裸参数路径（visual 树无 nn.Linear），故放入主阶段 rot_pairs
        # 而非 pre_run：pre_run 首命令须是可解析的模块路径（embed_tokens），
        # merger 路径由 _rotate 的裸 Parameter 回退分支处理。
        if self._has_visual_module():
            rot_pairs['rot'].left_rot["model.visual.merger.down_proj.weight"] = rotate_matrix['rot']

        # 补充 indexer 旋转路径（重映射后使用 model.language_model.layers 前缀）
        for layer_idx in range(num_layers + num_nextn):
            if not self._layer_has_indexer(layer_idx):
                continue
            rot_pairs['rot'].right_rot[f"model.language_model.layers.{layer_idx}.self_attn.indexer.wk"] = rotate_matrix[
                'rot'
            ]
            rot_pairs['rot'].right_rot[f"model.language_model.layers.{layer_idx}.self_attn.indexer.weights_proj"] = (
                rotate_matrix['rot']
            )
            rot_pairs['rot_b_proj'].right_rot[f"model.language_model.layers.{layer_idx}.self_attn.indexer.wq_b"] = (
                rotate_matrix['rot_b_proj']
            )
            # compress_gate 从旋转态 hidden_states 取输入，右旋 G·R 抵消 R
            rot_pairs['rot'].right_rot[
                f"model.language_model.layers.{layer_idx}.self_attn.indexer.index_kpool_compress_gate"
            ] = rotate_matrix['rot']

        # 为 MTP 层（DSA + MoE，无 HC）手动添加旋转路径
        for mtp_idx in range(num_nextn):
            layer_idx = num_layers + mtp_idx
            self._add_mtp_rot_entries(rot_pairs, rotate_matrix, layer_idx)

        return [pre_run], list(rot_pairs.values())

    def _add_mtp_rot_entries(self, rot_pairs, rotate_matrix, layer_idx):
        """为 MTP 层添加 DSA + MoE 旋转路径（与 quarot.py 中 DSA 层逻辑一致）。"""
        prefix = f"model.language_model.layers.{layer_idx}"

        # rot: DSA 右旋 q_a_proj, kv_a_proj_with_mqa; 左旋 o_proj
        rot_pairs['rot'].right_rot[f"{prefix}.self_attn.q_a_proj"] = rotate_matrix['rot']
        rot_pairs['rot'].right_rot[f"{prefix}.self_attn.kv_a_proj_with_mqa"] = rotate_matrix['rot']
        rot_pairs['rot'].left_rot[f"{prefix}.self_attn.o_proj"] = rotate_matrix['rot']

        # rot: MoE 右旋 gate_proj, up_proj; 左旋 down_proj
        text_config = self._get_text_config()
        expert_start, expert_end = _get_full_expert_range(text_config)
        for i in range(expert_start, expert_end):
            rot_pairs['rot'].right_rot[f"{prefix}.mlp.experts.{i}.gate_proj"] = rotate_matrix['rot']
            rot_pairs['rot'].right_rot[f"{prefix}.mlp.experts.{i}.up_proj"] = rotate_matrix['rot']
            rot_pairs['rot'].left_rot[f"{prefix}.mlp.experts.{i}.down_proj"] = rotate_matrix['rot']
        rot_pairs['rot'].right_rot[f"{prefix}.mlp.shared_experts.gate_proj"] = rotate_matrix['rot']
        rot_pairs['rot'].right_rot[f"{prefix}.mlp.shared_experts.up_proj"] = rotate_matrix['rot']
        rot_pairs['rot'].left_rot[f"{prefix}.mlp.shared_experts.down_proj"] = rotate_matrix['rot']
        rot_pairs['rot'].right_rot[f"{prefix}.mlp.gate"] = rotate_matrix['rot']

        # rot_b_proj: DSA q_a_proj 左旋, q_b_proj 右旋
        rot_pairs['rot_b_proj'].left_rot[f"{prefix}.self_attn.q_a_proj"] = rotate_matrix['rot_b_proj']
        rot_pairs['rot_b_proj'].right_rot[f"{prefix}.self_attn.q_b_proj"] = rotate_matrix['rot_b_proj']

        # rot_uv: DSA kv_b_proj 左旋（UV）, o_proj 右旋
        config = text_config
        rot_uv_left = [
            torch.eye(
                config.qk_nope_head_dim, dtype=rotate_matrix['rot_uv'].dtype, device=rotate_matrix['rot_uv'].device
            ),
            rotate_matrix['rot_uv'],
        ]
        rot_pairs['rot_uv'].left_rot[f"{prefix}.self_attn.kv_b_proj"] = rot_uv_left
        rot_pairs['rot_uv'].right_rot[f"{prefix}.self_attn.o_proj"] = rotate_matrix['rot_uv']

        # rot_kv_b_proj: DSA kv_a_proj_with_mqa 左旋（UV）, kv_b_proj 右旋
        rot_kv_left = [
            rotate_matrix['rot_kv_b_proj'],
            torch.eye(
                config.qk_rope_head_dim,
                dtype=rotate_matrix['rot_kv_b_proj'].dtype,
                device=rotate_matrix['rot_kv_b_proj'].device,
            ),
        ]
        rot_pairs['rot_kv_b_proj'].left_rot[f"{prefix}.self_attn.kv_a_proj_with_mqa"] = rot_kv_left
        rot_pairs['rot_kv_b_proj'].right_rot[f"{prefix}.self_attn.kv_b_proj"] = rotate_matrix['rot_kv_b_proj']

        # MTP 额外组件旋转：embed_tokens、eh_proj、shared_head.head
        # embed_tokens 输出需进入旋转域，与 hnorm 输出（旋转域）对齐后再 concat
        rot_pairs['rot'].right_rot[f"{prefix}.embed_tokens"] = rotate_matrix['rot']
        # eh_proj 接收 [enorm(embed_rot), hnorm(h_rot)] 的 concat，输入为双域
        # 右旋 block_diag(R,R) 吸收输入的旋转，左旋 R 使输出回到原始域
        rot_pairs['rot'].right_rot[f"{prefix}.eh_proj"] = torch.block_diag(*[rotate_matrix['rot']] * 2)
        rot_pairs['rot'].left_rot[f"{prefix}.eh_proj"] = rotate_matrix['rot']
        # shared_head.head 输入来自 eh_proj（原始域），需右旋 R 进入旋转域
        rot_pairs['rot'].right_rot[f"{prefix}.shared_head.head"] = rotate_matrix['rot']

    def _remap_rot_pairs(self, rot_pairs, rotate_matrix, pre_run=None):
        """旋转路径重映射：'model.layers.{idx}' -> 'model.language_model.layers.{idx}'。"""
        old_prefix = "model.layers."
        new_prefix = "model.language_model.layers."

        for key in list(rot_pairs.keys()):
            pair = rot_pairs[key]
            # 重映射 left_rot
            new_left_rot = {}
            for k, v in pair.left_rot.items():
                new_k = k.replace(old_prefix, new_prefix, 1) if old_prefix in k else k
                new_left_rot[new_k] = v
            pair.left_rot = new_left_rot
            # 重映射 right_rot
            new_right_rot = {}
            for k, v in pair.right_rot.items():
                new_k = k.replace(old_prefix, new_prefix, 1) if old_prefix in k else k
                new_right_rot[new_k] = v
            pair.right_rot = new_right_rot

        # pre_run 中重映射 'model.embed_tokens' -> 'model.language_model.embed_tokens'
        if pre_run is not None:
            new_right_rot = {}
            for k, v in pre_run.right_rot.items():
                new_k = k.replace("model.embed_tokens", "model.language_model.embed_tokens", 1)
                new_right_rot[new_k] = v
            pre_run.right_rot = new_right_rot

    def get_attention_module_cls(self) -> str:
        return "Glm5NextTextAttention"

    def get_attention_output_extractor(self) -> Callable[[Union[tuple, torch.Tensor]], torch.Tensor]:
        return lambda x: x

    def _save_vlm_assets(self, save_directory: str) -> None:
        """随导出目录分发 Glm5NextImageProcessor 远程代码（旧版 transformers 不识别）。

        通过 `auto_map` 注册，使运行时经 trust_remote_code 加载。
        """
        processor_config_path = os.path.join(save_directory, "processor_config.json")
        if not os.path.exists(processor_config_path):
            return
        processor_config = json_safe_load(processor_config_path, check_user_stat=False)
        image_processor_cfg = processor_config.get("image_processor", processor_config)
        if (
            not isinstance(image_processor_cfg, dict)
            or image_processor_cfg.get("image_processor_type") != "Glm5NextImageProcessor"
        ):
            return
        try:
            from transformers.models.glm5_next import image_processing_glm5_next as ip_module
        except ImportError:
            get_logger().warning(
                "Glm5NextImageProcessor not found in current transformers; "
                "the exported dir needs a runtime transformers with glm5_next support."
            )
            return
        with open(ip_module.__file__, 'r', encoding='utf-8') as f:  # nosec B131
            module_source = f.read().replace("from ...", "from transformers.")
        remote_code_path = os.path.join(save_directory, "image_processing_glm5_next.py")
        with open(remote_code_path, 'w', encoding='utf-8') as f:
            f.write(module_source)
        image_processor_cfg["auto_map"] = {"AutoImageProcessor": "image_processing_glm5_next.Glm5NextImageProcessor"}
        json_safe_dump(processor_config, processor_config_path, indent=2, check_user_stat=False)
        get_logger().info("Shipped Glm5NextImageProcessor remote code with auto_map into %s", save_directory)

    def ascendv1_save_postprocess(self, model: nn.Module, save_directory: str) -> None:
        self._save_vlm_assets(save_directory)
        description_path = os.path.join(save_directory, "quant_model_description.json")
        if os.path.exists(description_path):
            description_data = json_safe_load(description_path, check_user_stat=False)
            changed = False
            for key in list(description_data.keys()):
                if not key.endswith(".quant_type"):
                    continue
                value = description_data[key]
                if not isinstance(value, str):
                    continue
                parts = value.split('_')
                if parts and parts[0] and set(parts[0]) <= set('QKVP'):
                    new_value = '_'.join(parts[1:])
                    if new_value != value:
                        description_data[key] = new_value
                        changed = True
            if changed:
                json_safe_dump(description_data, description_path, indent=2, check_user_stat=False)

        global_rotation = None
        for _, module in model.named_modules():
            if isinstance(module, QuaRotExtraInfoWrapperIR):
                offline_info = module.rotation_info
                global_rotation = offline_info.global_rotation
        if global_rotation is None:
            return

        origin_index_path = os.path.join(self.model_path, "model.safetensors.index.json")
        origin_index_data = json_safe_load(origin_index_path)

        norm_key = None
        for possible_key in ["model.language_model.norm.weight", "model.norm.weight", "norm.weight"]:
            if possible_key in origin_index_data.get("weight_map", {}):
                norm_key = possible_key
                break
        if norm_key is None:
            raise UnsupportedError("model.norm.weight is not found in weight map.")

        weight_path = os.path.join(self.model_path, origin_index_data["weight_map"][norm_key])
        with safe_open(weight_path, framework='pt', device='cpu') as f:
            norm_weight = f.get_tensor(norm_key)
        if norm_weight is None:
            raise UnsupportedError("model.norm.weight is not found.")

        def _apply_rot_transform(w, Q):
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

        original_dtype = norm_weight.dtype
        rot_weight = _apply_rot_transform(norm_weight, global_rotation).to(original_dtype)

        from safetensors.torch import save_file

        save_file({"rot.weight": rot_weight}, os.path.join(save_directory, "rot.safetensors"))

        description_path = os.path.join(save_directory, "quant_model_description.json")
        description_data = json_safe_load(description_path)
        description_data["is_rot_used"] = True
        json_safe_dump(description_data, description_path, indent=2)

        index_path = os.path.join(save_directory, "quant_model_weights.safetensors.index.json")
        index_data = json_safe_load(index_path)
        index_data["weight_map"]["rot.weight"] = "rot.safetensors"
        json_safe_dump(index_data, index_path, indent=2)

    def _load_config(self, trust_remote_code=False) -> object:
        return SafeGenerator.get_config_from_pretrained(
            model_path=str(self.model_path), trust_remote_code=trust_remote_code
        )
