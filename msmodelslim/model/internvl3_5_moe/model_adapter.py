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

import os
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import List, Any, Generator, Tuple, Dict
from unittest.mock import patch

import torch
from safetensors import safe_open
from torch import nn
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from transformers.masking_utils import create_causal_mask
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeDecoderLayer

from msmodelslim.core.const import DeviceType
from msmodelslim.core.base.protocol import ProcessRequest
from msmodelslim.core.graph import AdapterConfig, MappingConfig
from msmodelslim.model.common.layer_wise_forward import generated_decoder_layer_visit_func
from msmodelslim.model.interface_hub import (
    IterSmoothInterface,
    FlexSmoothQuantInterface,
    ModelSlimPipelineInterfaceV1,
    ModelInfoInterface,
    LayerWiseOffloadOptionalInterface,
)
from msmodelslim.model.common.vlm_base import VLMBaseModelAdapter
from msmodelslim.processor.quarot import QuaRotInterface
from msmodelslim.utils.exception import InvalidModelError
from msmodelslim.utils.logging import logger_setter, get_logger
from msmodelslim.utils.security import get_valid_read_path, json_safe_load, MAX_READ_FILE_SIZE_32G
from msmodelslim.model.internvl3_5_moe.internvl_moe_utils import load_image


@logger_setter()
class InternVL3_5MoeModelAdapter(  # pylint: disable=too-many-ancestors
    VLMBaseModelAdapter,
    ModelInfoInterface,
    ModelSlimPipelineInterfaceV1,
    FlexSmoothQuantInterface,
    IterSmoothInterface,
    QuaRotInterface,
    LayerWiseOffloadOptionalInterface,
):
    def __init__(self, model_type: str, model_path: Path, trust_remote_code: bool = False):
        self._processor = None
        self._tokenizer = None
        self._img_context_token_id = None
        super().__init__(model_type, model_path, trust_remote_code)

    def get_model_pedigree(self) -> str:
        return 'internvl3_5_moe'

    def get_model_type(self) -> str:
        return self.model_type

    def get_layer_wise_offload_device(self):
        return "meta"

    def handle_dataset(self, dataset: Any, device: DeviceType = DeviceType.NPU) -> List[Any]:
        self._tokenizer = AutoTokenizer.from_pretrained(  # nosec B615
            self.model_path,
            trust_remote_code=self.trust_remote_code,
            local_files_only=True,
            fix_mistral_regex=True,
        )
        get_logger().info("Loaded tokenizer for InternVL3.5 MoE")

        IMG_START_TOKEN = '<img>'  # nosec B105
        IMG_END_TOKEN = '</img>'  # nosec B105
        IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'  # nosec B105

        self._img_context_token_id = self._tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        get_logger().info("Set img_context_token_id = %s for vision-text fusion", self._img_context_token_id)

        eos_token_id = self._tokenizer.convert_tokens_to_ids('<|im_end|>')
        get_logger().info("EOS token id: %s", eos_token_id)

        image_size = self.config.force_image_size or self.config.vision_config.image_size

        processed_data = []
        for item in tqdm(dataset, desc="Processing InternVL3.5 MoE calibration dataset"):
            text = item.text
            image_path = getattr(item, 'image', None)
            pixel_values = None

            if image_path:
                image_path = get_valid_read_path(image_path)
                pixel_values = load_image(str(image_path), input_size=image_size, max_num=12)
                num_patches = pixel_values.shape[0]
                query = (
                    '<|im_start|>system\n你是书生·万象，英文名是InternVL，是由上海人工智能实验室、清华大学及多家合作单位联合开发的多模态大语言模型。<|im_end|>'
                    + '<|im_start|>user\n<image>\n'
                    + text
                    + '<|im_end|><|im_start|>assistant\n'
                )
            else:
                num_patches = 1
                query = (
                    '<|im_start|>system\n你是书生·万象，英文名是InternVL，是由上海人工智能实验室、清华大学及多家合作单位联合开发的多模态大语言模型。<|im_end|>'
                    + '<|im_start|>user\n'
                    + text
                    + '<|im_end|><|im_start|>assistant\n'
                )

            if '<image>' in query:
                image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.num_image_token * num_patches + IMG_END_TOKEN
                query = query.replace('<image>', image_tokens, 1)

            self._tokenizer.padding_side = 'left'
            model_inputs = self._tokenizer(query, return_tensors='pt', padding=True)

            processed_item = self._collect_inputs_to_device(
                model_inputs,
                device,
                keys=["input_ids", "attention_mask"],
                defaults={},
            )

            if pixel_values is not None:
                processed_item['pixel_values'] = pixel_values.to(device)
                processed_item['image_flags'] = torch.tensor([[1] * pixel_values.shape[0]], dtype=torch.long).to(device)
            else:
                processed_item['pixel_values'] = None
                processed_item['image_flags'] = torch.tensor([[0]], dtype=torch.long).to(device)

            generation_config = {
                'max_new_tokens': 1024,
                'do_sample': False,
                'eos_token_id': eos_token_id,
            }
            processed_item['generation_config'] = generation_config

            processed_data.append(processed_item)

        get_logger().info("Processed %s samples for InternVL3.5 MoE", len(processed_data))
        return processed_data

    def init_model(self, device: DeviceType = DeviceType.NPU) -> nn.Module:
        get_logger().info("Initializing InternVL3.5 MoE model with layer-wise loading...")

        global_torch_dtype = self.get_global_model_torch_dtype()

        origin_layers = self.config.llm_config.num_hidden_layers
        get_logger().info(
            "Model with %s text layers + %s vision layers",
            origin_layers,
            self.config.vision_config.num_hidden_layers,
        )

        self.config.llm_config.use_cache = False
        self.config.llm_config._attn_implementation = 'eager'

        self.model_path = get_valid_read_path(str(self.model_path), is_dir=True, check_user_stat=True)

        self.config.llm_config.num_hidden_layers = 1

        get_logger().info("Loading InternVL3.5 MoE model...")
        try:
            model = AutoModel.from_pretrained(  # nosec B615
                self.model_path,
                config=self.config,
                trust_remote_code=self.trust_remote_code,
                torch_dtype=global_torch_dtype,
                local_files_only=True,
                device_map="cpu",
            ).eval()
        except Exception as e:
            raise InvalidModelError(
                f"Failed to load InternVL3.5 MoE model: {e}",
                action="Check model directory and trust_remote_code setting",
            ) from e

        self.config.llm_config.num_hidden_layers = origin_layers

        get_logger().info("Loading weights for vision encoder, first decoder layer, and lm_head...")
        state_dict = self._get_state_dict(model)
        state_dict = {k: v.to(global_torch_dtype) for k, v in state_dict.items()}
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            get_logger().warning("Missing keys after load_state_dict: %s", missing)

        if hasattr(model.config.llm_config, "num_attention_heads"):
            model.config.num_attention_heads = model.config.llm_config.num_attention_heads
            get_logger().info("Set model.config.num_attention_heads = %s", model.config.num_attention_heads)
        if hasattr(model.config.llm_config, "num_key_value_heads"):
            model.config.num_key_value_heads = model.config.llm_config.num_key_value_heads
            get_logger().info("Set model.config.num_key_value_heads = %s", model.config.num_key_value_heads)

        get_logger().info("Model initialized with %s layers (1 loaded, others will be loaded on-demand)", origin_layers)

        return model

    def generate_model_visit(self, model: nn.Module) -> Generator[ProcessRequest, Any, None]:
        get_logger().info("Processing vision encoder...")
        yield ProcessRequest(name="vision_model", module=model.vision_model, args=(), kwargs={})

        get_logger().info("Processing text decoder layers...")
        yield from generated_decoder_layer_visit_func(model, transformer_blocks=self.generate_decoder_layer(model))

    def generate_model_forward(self, model: nn.Module, inputs: Any) -> Generator[ProcessRequest, Any, None]:
        if isinstance(inputs, list):
            sample = inputs[0]
        else:
            sample = inputs

        pixel_values = sample.get('pixel_values')
        vit_embeds = None

        if pixel_values is not None:
            try:
                model_dtype = next(model.vision_model.parameters()).dtype
            except StopIteration:
                model_dtype = torch.bfloat16
            pixel_values = pixel_values.to(dtype=model_dtype)
            vit_embeds = yield ProcessRequest(
                name="vision_model",
                module=model.vision_model,
                args=(pixel_values,),
                kwargs={'output_hidden_states': False, 'return_dict': True},
            )
            vit_embeds = (
                vit_embeds['last_hidden_state'] if isinstance(vit_embeds, dict) else vit_embeds.last_hidden_state
            )

        input_ids = sample['input_ids']
        attention_mask = sample['attention_mask']
        position_ids = sample.get('position_ids', None)

        B, N = input_ids.shape
        input_embeds = model.language_model.get_input_embeddings()(input_ids)

        if vit_embeds is not None and self._img_context_token_id is not None:
            C = input_embeds.shape[-1]
            input_embeds = input_embeds.reshape(B * N, C)
            input_ids_flat = input_ids.reshape(B * N)
            selected = input_ids_flat == self._img_context_token_id
            num_img_tokens = selected.sum().item()

            vit_embeds = vit_embeds[:, 1:, :]
            h = w = int(vit_embeds.shape[1] ** 0.5)
            vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)
            vit_embeds = model.pixel_shuffle(vit_embeds, scale_factor=model.downsample_ratio)
            vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1, vit_embeds.shape[-1])
            vit_embeds = model.mlp1(vit_embeds)

            image_flags = sample.get('image_flags')
            if image_flags is not None:
                image_flags = image_flags.view(-1)
                vit_embeds = vit_embeds[image_flags == 1]

            vit_embeds = vit_embeds.reshape(-1, C)
            vit_embeds = vit_embeds.to(device=input_embeds.device, dtype=input_embeds.dtype)

            if vit_embeds.shape[0] != num_img_tokens:
                get_logger().warning(
                    "Vision-text fusion mismatch: vit_embeds count (%s) != image_tokens (%s)",
                    vit_embeds.shape[0],
                    num_img_tokens,
                )
            input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds[:num_img_tokens]

            input_embeds = input_embeds.reshape(B, N, C)

        hidden_states = input_embeds

        cache_position = torch.arange(input_embeds.shape[1], device=input_embeds.device)
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0).expand(input_embeds.shape[0], -1)
        position_embeddings = model.language_model.model.rotary_emb(hidden_states, position_ids)

        causal_mask = create_causal_mask(
            config=model.config.llm_config,
            input_embeds=input_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=None,
            position_ids=position_ids,
        )

        for _, (name, layer) in enumerate(self.generate_decoder_layer(model)):
            layer_outputs = yield ProcessRequest(
                name=name,
                module=layer,
                args=(hidden_states,),
                kwargs={
                    'attention_mask': causal_mask,
                    'position_embeddings': position_embeddings,
                    'position_ids': position_ids,
                    'cache_position': cache_position,
                    'use_cache': False,
                },
            )
            hidden_states = layer_outputs[0] if isinstance(layer_outputs, tuple) else layer_outputs

    def generate_decoder_layer(self, model: nn.Module) -> Generator[Tuple[str, nn.Module], None, None]:
        num_layers = self.config.llm_config.num_hidden_layers

        for layer_idx in range(num_layers):
            name = f"language_model.model.layers.{layer_idx}"
            layer = self._load_decoder_if_not_exist(model, name, layer_idx)
            yield name, layer

    def _load_decoder_if_not_exist(self, model: nn.Module, name: str, idx: int) -> nn.Module:
        try:
            decoder = model.get_submodule(name)
            try:
                _ = decoder.input_layernorm.weight.device
                return decoder
            except RuntimeError:
                get_logger().debug("Decoder layer %s weights not on device, will reload", idx)
        except AttributeError:
            get_logger().debug("Decoder layer %s not found in model, will create", idx)

        get_logger().info("Loading decoder layer %s...", idx)

        with patch.object(nn.Linear, 'reset_parameters', lambda _self: None):
            get_logger().info('Creating decoder layer %s structure...', idx)
            decoder = Qwen3MoeDecoderLayer(self.config.llm_config, layer_idx=idx)

            state_dict = self._get_state_dict(decoder, prefix=name)
            model_dtype = getattr(self, "_model_torch_dtype", None) or next(model.parameters()).dtype
            state_dict = {k: v.to(model_dtype) for k, v in state_dict.items()}
            decoder.load_state_dict(state_dict)
            decoder.eval()

            module_list: nn.ModuleList = model.language_model.model.layers
            if len(module_list) <= idx:
                module_list.append(decoder)
            else:
                module_list[idx] = decoder

            get_logger().info('Decoder layer %s loaded successfully', idx)

        return decoder

    def enable_kv_cache(self, model: nn.Module, need_kv_cache: bool) -> None:
        model.config.llm_config.use_cache = need_kv_cache
        get_logger().info("KV cache %s", 'enabled' if need_kv_cache else 'disabled')

    @property
    def num_image_token(self) -> int:
        image_size = self.config.force_image_size or self.config.vision_config.image_size
        patch_size = self.config.vision_config.patch_size
        downsample_ratio = self.config.downsample_ratio
        return int((image_size // patch_size) ** 2 * (downsample_ratio**2))

    @lru_cache(maxsize=1)
    def _get_weight_map(self) -> Dict[str, str]:
        index_path = os.path.join(self.model_path, "model.safetensors.index.json")
        index_data = json_safe_load(index_path)
        return index_data['weight_map']

    def _get_state_dict(self, module: nn.Module, prefix: str = "") -> Dict[str, torch.Tensor]:
        weight_map = self._get_weight_map()

        param_names = [name for name, _ in module.named_parameters()]

        file_groups = defaultdict(list)
        for param_name in param_names:
            full_name = f"{prefix}.{param_name}" if prefix else param_name
            if full_name in weight_map:
                file_name = weight_map[full_name]
                file_groups[file_name].append(param_name)

        state_dict = {}
        for file_name, names in tqdm(file_groups.items(), desc=f"Loading {prefix}", leave=False):
            file_path = os.path.join(self.model_path, file_name)
            file_path = get_valid_read_path(file_path, extensions='safetensors', size_max=MAX_READ_FILE_SIZE_32G)

            with safe_open(file_path, framework='pt', device='cpu') as f:
                for param_name in names:
                    full_name = f"{prefix}.{param_name}" if prefix else param_name
                    state_dict[param_name] = f.get_tensor(full_name)

        return state_dict

    def _is_moe_layer(self, layer_idx: int) -> bool:
        if layer_idx in self.config.llm_config.mlp_only_layers:
            return False
        if (layer_idx + 1) % self.config.llm_config.decoder_sparse_step == 0:
            return True
        return False

    def get_adapter_config_for_subgraph(self) -> List[AdapterConfig]:
        adapter_config = []
        num_layers = self.config.llm_config.num_hidden_layers

        for layer_idx in range(num_layers):
            input_layernorm_mapping = MappingConfig(
                source=f"language_model.model.layers.{layer_idx}.input_layernorm",
                targets=[
                    f"language_model.model.layers.{layer_idx}.self_attn.q_proj",
                    f"language_model.model.layers.{layer_idx}.self_attn.k_proj",
                    f"language_model.model.layers.{layer_idx}.self_attn.v_proj",
                ],
            )

            ov_mapping = MappingConfig(
                source=f"language_model.model.layers.{layer_idx}.self_attn.v_proj",
                targets=[f"language_model.model.layers.{layer_idx}.self_attn.o_proj"],
            )

            adapter_config.extend(
                [
                    AdapterConfig(subgraph_type="norm-linear", mapping=input_layernorm_mapping),
                    AdapterConfig(subgraph_type="ov", mapping=ov_mapping, extra_config={"group_method": "max"}),
                ]
            )

        return adapter_config

    def get_ln_fuse_map(self):
        ln_linear_map = {}
        num_layers = self.config.llm_config.num_hidden_layers

        for layer_idx in range(num_layers):
            ln_linear_map[f"language_model.model.layers.{layer_idx}.input_layernorm"] = [
                f"language_model.model.layers.{layer_idx}.self_attn.q_proj",
                f"language_model.model.layers.{layer_idx}.self_attn.k_proj",
                f"language_model.model.layers.{layer_idx}.self_attn.v_proj",
            ]

            ln_linear_map[f"language_model.model.layers.{layer_idx}.post_attention_layernorm"] = [
                f"language_model.model.layers.{layer_idx}.mlp.experts.{i}.{proj}"
                for proj in ["gate_proj", "up_proj"]
                for i in range(self.config.llm_config.num_experts)
            ]
            ln_linear_map[f"language_model.model.layers.{layer_idx}.post_attention_layernorm"] += [
                f"language_model.model.layers.{layer_idx}.mlp.gate"
            ]

        ln_linear_map["language_model.model.norm"] = ["language_model.lm_head"]

        return {}, ln_linear_map

    def get_bake_names(self):
        return [], []

    def get_rotate_map(self, block_size):
        config = self.config.llm_config

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

        left_rot_pre = {}
        right_rot_pre = {}

        right_rot_pre["language_model.model.embed_tokens"] = rot
        left_rot_pre["mlp1.3"] = rot

        pre_run = QuaRotInterface.RotatePair(left_rot=left_rot_pre, right_rot=right_rot_pre)

        left_rot = {}
        right_rot = {}
        right_rot["language_model.lm_head"] = rot

        for layer_idx in range(config.num_hidden_layers):
            right_rot[f"language_model.model.layers.{layer_idx}.self_attn.q_proj"] = rot
            right_rot[f"language_model.model.layers.{layer_idx}.self_attn.k_proj"] = rot
            right_rot[f"language_model.model.layers.{layer_idx}.self_attn.v_proj"] = rot
            left_rot[f"language_model.model.layers.{layer_idx}.self_attn.o_proj"] = rot

            num_experts = config.num_experts
            for i in range(num_experts):
                right_rot[f"language_model.model.layers.{layer_idx}.mlp.experts.{i}.gate_proj"] = rot
                right_rot[f"language_model.model.layers.{layer_idx}.mlp.experts.{i}.up_proj"] = rot
                left_rot[f"language_model.model.layers.{layer_idx}.mlp.experts.{i}.down_proj"] = rot
            right_rot[f"language_model.model.layers.{layer_idx}.mlp.gate"] = rot

        rot_pairs = {'rot': QuaRotInterface.RotatePair(left_rot=left_rot, right_rot=right_rot)}

        left_rot_uv = {}
        right_rot_uv = {}
        for layer_idx in range(config.num_hidden_layers):
            left_rot_uv[f"language_model.model.layers.{layer_idx}.self_attn.v_proj"] = rot_uv
            right_rot_uv[f"language_model.model.layers.{layer_idx}.self_attn.o_proj"] = rot_uv
        rot_pairs['rot_uv'] = QuaRotInterface.RotatePair(left_rot=left_rot_uv, right_rot=right_rot_uv)

        return [pre_run], list(rot_pairs.values())
