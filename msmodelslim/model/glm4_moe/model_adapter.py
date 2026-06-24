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

import os
from collections import defaultdict
from functools import lru_cache
from typing import List, Any, Generator, Optional, Tuple, Dict, Union
from unittest.mock import patch

import torch
from safetensors import safe_open
from torch import nn
from tqdm import tqdm
from transformers import PreTrainedTokenizerBase
from msmodelslim.core.const import DeviceType
from msmodelslim.core.base.protocol import ProcessRequest
from msmodelslim.core.graph import AdapterConfig, MappingConfig
from msmodelslim.processor.quarot import QuaRotInterface
from msmodelslim.model.interface_hub import (
    ModelSlimPipelineInterfaceV1,
    ModelInfoInterface,
    IterSmoothInterface,
    FlexSmoothQuantInterface,
)
from msmodelslim.model.common.layer_wise_forward import generated_decoder_layer_visit_func, TransformersForwardBreak
from msmodelslim.model.common.transformers import TransformersModel
from msmodelslim.utils.exception import InvalidModelError
from msmodelslim.utils.security import get_valid_read_path, json_safe_load, MAX_READ_FILE_SIZE_32G
from msmodelslim.utils.security.model import SafeGenerator
from msmodelslim.utils.logging import logger_setter, get_logger
from .mtp_quant_module import MTPLayer, wrap_mtp_decoder, remove_zero_and_shift


@logger_setter()
class GLM4MoeModelAdapter(  # pylint: disable=too-many-ancestors
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
        return 'glm4_moe'

    def handle_dataset(self, dataset: Any, device: DeviceType = DeviceType.NPU) -> List[Any]:
        return self._get_tokenized_data(dataset, device)

    def init_model(self, device: DeviceType = DeviceType.NPU) -> nn.Module:
        """
        初始化附带mtp的模型。
        首先通过 TransformersModel._load_model 加载 HF 模型基础层，
        再追加 MTP 层并注入子模块，使模型结构包含完整的基础层 + MTP 层。
        """
        # GLMMoe包含4.5\4.6\4.7三个模型，这里不能进行硬编码
        original_num_layers = self.config.num_hidden_layers
        total_layers = original_num_layers + 1

        get_logger().info(
            "Original model has %d base layers, MTP layer at index %d, total %d",
            original_num_layers,
            original_num_layers,
            total_layers,
        )

        model = self._load_model(device)

        # Required: prevents KeyError when accessing ALL_ATTENTION_FUNCTIONS
        if getattr(self.config, '_attn_implementation', None) is None:
            self.config._attn_implementation = "eager"

        get_logger().info("Adding MTP layer at index %d", original_num_layers)
        self.config.num_hidden_layers = total_layers
        mtp_name = f"model.layers.{original_num_layers}"
        mtp_decoder = self.load_decoder_if_not_exist(model, name=mtp_name, idx=original_num_layers)
        self.load_mtp_if_not_load(mtp_decoder)

        model.eval()
        get_logger().info("Create model with %d layers successfully", self.config.num_hidden_layers)
        return model

    def _get_num_total_layers(self) -> int:
        return self.config.num_hidden_layers

    def generate_model_visit(self, model: nn.Module) -> Generator[ProcessRequest, Any, None]:
        return generated_decoder_layer_visit_func(model, transformer_blocks=self.generate_decoder_layer(model))

    def generate_model_forward(
        self,
        model: nn.Module,
        inputs: Any,
    ) -> Generator[ProcessRequest, Any, None]:
        first_block_input: Optional[Tuple] = None

        def break_hook(module: nn.Module, hook_args: Tuple[Any, ...], hook_kwargs: Dict[str, Any]):
            nonlocal first_block_input
            first_block_input = (
                hook_args,
                hook_kwargs,
            )
            raise TransformersForwardBreak()

        first_layer = model.get_submodule("model.layers.0")
        remove_handler = first_layer.register_forward_pre_hook(break_hook, with_kwargs=True, prepend=True)

        try:
            if isinstance(inputs, (list, tuple)):
                model(*inputs)
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

        current_inputs = first_block_input
        total_layers = self._get_num_total_layers()

        for name, block in self.generate_decoder_layer(model):
            args, kwargs = current_inputs

            if name == f'model.layers.{total_layers - 1}':
                args, kwargs = self.mtp_preprocess(model, mtp_decoder=block, inputs=inputs, args=args, kwargs=kwargs)

            outputs = yield ProcessRequest(name, block, args, kwargs)
            hidden_states = outputs[0] if isinstance(outputs, tuple) else outputs
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

        kwargs['attention_mask'] = attention_mask_mtp
        kwargs['position_ids'] = position_ids

        return (hidden_states_mtp,), kwargs

    def enable_kv_cache(self, model: nn.Module, need_kv_cache: bool) -> None:
        return self._enable_kv_cache(model, need_kv_cache)

    def get_adapter_config_for_subgraph(self) -> List[AdapterConfig]:
        adapter_config = []
        total_layers = self._get_num_total_layers()
        for layer_idx in range(total_layers):
            norm_linear_mapping_config = MappingConfig(
                source=f"model.layers.{layer_idx}.input_layernorm",
                targets=[
                    f"model.layers.{layer_idx}.self_attn.k_proj",
                    f"model.layers.{layer_idx}.self_attn.q_proj",
                    f"model.layers.{layer_idx}.self_attn.v_proj",
                ],
            )

            ov_mapping_config = MappingConfig(
                source=f"model.layers.{layer_idx}.self_attn.v_proj",
                targets=[f"model.layers.{layer_idx}.self_attn.o_proj"],
            )

            adapter_config.extend(
                [
                    AdapterConfig(subgraph_type="norm-linear", mapping=norm_linear_mapping_config),
                    AdapterConfig(subgraph_type="ov", mapping=ov_mapping_config, extra_config={'group_method': 'max'}),
                ]
            )
        return adapter_config

    def get_ln_fuse_map(self):
        return {}, glm4_moe_get_ln_fuse_map(self.config, self._get_num_total_layers())

    def get_bake_names(self):
        return [], []

    def get_rotate_map(self, block_size):
        pre_run, rot_pairs, _, _ = glm4_moe_get_rotate_map(self.config, block_size, self._get_num_total_layers())
        return [pre_run], list(rot_pairs.values())

    def _load_tokenizer(self, trust_remote_code=False) -> PreTrainedTokenizerBase:
        return SafeGenerator.get_tokenizer_from_pretrained(
            model_path=str(self.model_path), legacy=False, trust_remote_code=trust_remote_code
        )

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
            file_name = weight_map[full_name]
            groups[file_name].append(name)

        state_dict = {}
        for file_name in tqdm(groups, desc=f'Loading {prefix}'):
            file_path = os.path.join(self.model_path, file_name)
            file_path = get_valid_read_path(file_path, extensions='safetensors', size_max=MAX_READ_FILE_SIZE_32G)
            with safe_open(file_path, framework='pt', device='cpu') as f:
                for name in tqdm(groups[file_name], desc=f'Loading {file_path}'):
                    state_dict[name] = f.get_tensor(f'{prefix}.{name}' if prefix else name)
        return state_dict

    def get_mtp_layer(self):
        get_logger().debug('Start to load mtp')
        mtp_layer = MTPLayer(self.config)
        mtp_idx = self.config.num_hidden_layers - 1

        embed_tokens = nn.Embedding(self.config.vocab_size, self.config.hidden_size)
        embed_state_dict = self.get_state_dict(embed_tokens, prefix='model.embed_tokens')
        head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
        head_state_dict = self.get_state_dict(head, prefix='lm_head')

        state_dict = self.get_state_dict(mtp_layer, prefix=f'model.layers.{mtp_idx}')
        state_dict['shared_head.head.weight'] = head_state_dict['weight']
        state_dict['embed_tokens.weight'] = embed_state_dict['weight']

        mtp_layer.load_state_dict(state_dict)
        get_logger().debug('Success to load mtp')
        return mtp_layer

    def load_mtp_if_not_load(self, mtp_decoder: nn.Module):
        """
        如果 MTP 子模块尚未挂载，则加载并注入。

        通过检测 mtp_decoder 上是否存在 shared_head 属性来判断 MTP 是否已加载，
        避免重复加载。使用 wrap_mtp_decoder 将 MTPLayer 的子模块注入到目标层。
        """
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
            with patch.object(nn.Linear, 'reset_parameters', lambda _self: None):
                get_logger().info('Creating decoder layer %s', idx)
                module_list: nn.ModuleList = model.model.layers
                template_module = module_list[0]
                decoder = template_module.__class__(config=self.config, layer_idx=idx)

                state_dict = self.get_state_dict(decoder, prefix=name)
                missing, unexpected = decoder.load_state_dict(state_dict, strict=False)
                if missing:
                    get_logger().debug('Missing keys when loading %s: %s', name, missing)
                decoder.eval()
                module_list.append(decoder)
                get_logger().info('Create decoder layer %s successfully', idx)
        return decoder

    def generate_decoder_layer(self, model: nn.Module):
        """
        按索引生成所有 Decoder Layer（包括 MTP 层）。

        遍历基础层 + MTP 层，按需加载每层并确保 MTP 子模块已挂载到最后一层。
        用于 generate_model_visit 和 generate_model_forward 的迭代。
        """
        total_layers = self._get_num_total_layers()
        for idx in range(total_layers):
            name = f"model.layers.{idx}"
            decoder = self.load_decoder_if_not_exist(model, name=name, idx=idx)
            if idx == total_layers - 1:
                self.load_mtp_if_not_load(decoder)
            yield name, decoder


def glm4_moe_get_ln_fuse_map(config, num_total_layers=None):
    if num_total_layers is None:
        num_total_layers = config.num_hidden_layers
    ln_linear_map = {}
    for layer_idx in range(num_total_layers):
        ln_linear_map[f"model.layers.{layer_idx}.input_layernorm"] = [
            f"model.layers.{layer_idx}.self_attn.q_proj",
            f"model.layers.{layer_idx}.self_attn.k_proj",
            f"model.layers.{layer_idx}.self_attn.v_proj",
        ]

        ln_linear_map[f"model.layers.{layer_idx}.post_attention_layernorm"] = [
            f"model.layers.{layer_idx}.mlp.experts.{i}.{proj}"
            for proj in ["gate_proj", "up_proj"]
            for i in range(config.num_experts)
        ]
        ln_linear_map[f"model.layers.{layer_idx}.post_attention_layernorm"] += [f"model.layers.{layer_idx}.mlp.gate"]
    ln_linear_map["model.norm"] = ['lm_head']

    mtp_idx = num_total_layers - 1
    ln_linear_map[(f"model.layers.{mtp_idx}.enorm", f"model.layers.{mtp_idx}.hnorm")] = [
        f"model.layers.{mtp_idx}.eh_proj",
    ]
    ln_linear_map[f"model.layers.{mtp_idx}.shared_head.norm"] = [
        f"model.layers.{mtp_idx}.shared_head.head",
    ]
    return ln_linear_map


def glm4_moe_get_rotate_map(config, block_size, num_total_layers=None):
    if num_total_layers is None:
        num_total_layers = config.num_hidden_layers
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
    left_rot = {}
    right_rot = {}
    right_rot["model.embed_tokens"] = rot
    pre_run = QuaRotInterface.RotatePair(left_rot=left_rot, right_rot=right_rot)
    rot_pairs = {}
    left_rot = {}
    right_rot = {}
    right_rot["lm_head"] = rot
    for layer_idx in range(num_total_layers):
        right_rot[f"model.layers.{layer_idx}.self_attn.q_proj"] = rot
        right_rot[f"model.layers.{layer_idx}.self_attn.k_proj"] = rot
        right_rot[f"model.layers.{layer_idx}.self_attn.v_proj"] = rot
        left_rot[f"model.layers.{layer_idx}.self_attn.o_proj"] = rot
        for i in range(config.num_experts):
            right_rot[f"model.layers.{layer_idx}.mlp.experts.{i}.gate_proj"] = rot
            right_rot[f"model.layers.{layer_idx}.mlp.experts.{i}.up_proj"] = rot
            left_rot[f"model.layers.{layer_idx}.mlp.experts.{i}.down_proj"] = rot
        right_rot[f"model.layers.{layer_idx}.mlp.gate"] = rot
    rot_pairs['rot'] = QuaRotInterface.RotatePair(left_rot=left_rot, right_rot=right_rot)

    left_rot_uv = {}
    right_rot_uv = {}
    for layer_idx in range(num_total_layers):
        left_rot_uv[f"model.layers.{layer_idx}.self_attn.v_proj"] = rot_uv
        right_rot_uv[f"model.layers.{layer_idx}.self_attn.o_proj"] = rot_uv
    rot_pairs["rot_uv"] = QuaRotInterface.RotatePair(left_rot=left_rot_uv, right_rot=right_rot_uv)

    mtp_idx = num_total_layers - 1
    right_rot[f"model.layers.{mtp_idx}.embed_tokens"] = rot
    right_rot[f"model.layers.{mtp_idx}.eh_proj"] = torch.block_diag(*[rot] * 2)
    left_rot[f"model.layers.{mtp_idx}.eh_proj"] = rot
    right_rot[f"model.layers.{mtp_idx}.shared_head.head"] = rot

    return pre_run, rot_pairs, rot, rot_uv
