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
# pylint: disable=no-name-in-module

import os
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import List, Any, Generator, Tuple, Dict
from unittest.mock import patch

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from torch import nn
from tqdm import tqdm
from transformers import AutoProcessor

from msmodelslim.core.const import DeviceType
from msmodelslim.app.naive_quantization.model_info_interface import ModelInfoInterface
from msmodelslim.core.base.protocol import ProcessRequest
from msmodelslim.model.common.layer_wise_forward import generated_decoder_layer_visit_func
from msmodelslim.model.interface_hub import (
    AscendV1SaveInterface,
    ModelSlimPipelineInterfaceV1,
)
from msmodelslim.model.common.vlm_base import VLMBaseModelAdapter
from msmodelslim.utils.exception import InvalidModelError, UnsupportedError
from msmodelslim.utils.logging import logger_setter, get_logger
from msmodelslim.utils.security import (
    get_valid_read_path,
    get_valid_write_path,
    json_safe_dump,
    json_safe_load,
    MAX_READ_FILE_SIZE_512G,
)


_FLOAT_BUFFER_ATTRS_TO_PROMOTE = ("std_bias", "std_scale", "layer_scalar")


def _cast_floating_state_dict(state_dict: Dict[str, torch.Tensor], dtype: torch.dtype) -> Dict[str, torch.Tensor]:
    """Cast floating checkpoint tensors while preserving integer and boolean buffers."""
    return {name: tensor.to(dtype) if tensor.is_floating_point() else tensor for name, tensor in state_dict.items()}


def _promote_float_buffers_to_parameters(root: nn.Module) -> None:
    """Make Gemma4 float buffers visible to the parameters-only AscendV1 saver."""
    for _, module in root.named_modules():
        for attr_name in _FLOAT_BUFFER_ATTRS_TO_PROMOTE:
            if not hasattr(module, attr_name):
                continue
            existing = getattr(module, attr_name)
            if existing is None or isinstance(existing, nn.Parameter):
                continue
            if not isinstance(existing, torch.Tensor):
                continue
            if attr_name in module._buffers:
                del module._buffers[attr_name]
            module.register_parameter(
                attr_name,
                nn.Parameter(existing.detach().to(torch.float32), requires_grad=False),
            )


def _unwrap_forward_output(output: Any, field: str = "last_hidden_state") -> Any:
    """Normalize module forward outputs passed back through the generated runner."""
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, dict):
        if field in output:
            value = output[field]
            return value if isinstance(value, torch.Tensor) else _unwrap_forward_output(value, field)
        return output
    if hasattr(output, field):
        return getattr(output, field)
    if isinstance(output, (list, tuple)) and output:
        return _unwrap_forward_output(output[0], field)
    return output


@logger_setter()
class Gemma4ModelAdapter(  # pylint: disable=too-many-ancestors
    VLMBaseModelAdapter,
    ModelInfoInterface,
    ModelSlimPipelineInterfaceV1,
    AscendV1SaveInterface,
):
    """
    Gemma4 VLM dense Model Adapter for msModelSlim.

    Supports Gemma4ForConditionalGeneration with enable_moe_block=False:
    - Vision encoder + dense text decoder (GELU MLP per layer)
    - Mixed sliding/full attention layers
    - Bidirectional attention for vision tokens

    W8A8 dynamic quantization (per_token act + per_channel weight) is data-free and
    uses ``generate_model_visit`` only; no calibration forward pass is required.

    Layer analysis and static/mixed W8A8 require ``handle_dataset`` + ``generate_model_forward``.

    """

    def __init__(self, model_type: str, model_path: Path, trust_remote_code: bool = False):
        self._processor = None
        super().__init__(model_type, model_path, trust_remote_code)

    def get_model_pedigree(self) -> str:
        """Return model pedigree for best practice matching."""
        return 'gemma4'

    def get_model_type(self) -> str:
        """Return model type."""
        return self.model_type

    def handle_dataset(self, dataset: Any, device: DeviceType = DeviceType.NPU) -> List[Any]:
        """
        Prepare multimodal calibration samples for forward-based tasks (layer analysis, static W8A8).

        Supported sample structure: ``VlmCalibSample(text: str, image: Optional[str])``.

        Data-free dynamic W8A8 (``gemma4_w8a8.yaml``) does not run forward; an empty dataset is
        acceptable and ``linear_quant`` uses ``generate_model_visit`` only.
        """
        if not dataset:
            get_logger().info(
                "No calibration dataset provided; forward-based tasks will not run.",
            )
            return []

        # The validated local path and local_files_only prevent Hub downloads.
        self._processor = AutoProcessor.from_pretrained(  # nosec B615
            self.model_path,
            trust_remote_code=self.trust_remote_code,
            local_files_only=True,
        )

        processed_data = []
        for item in tqdm(dataset, desc="Processing calibration dataset"):
            image_path = get_valid_read_path(item.image)
            text = item.text

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image_path},
                        {"type": "text", "text": text},
                    ],
                }
            ]
            inputs = self._processor.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
            )
            # Gemma4ForConditionalGeneration
            processed_item = self._collect_inputs_to_device(
                inputs,
                device,
                keys=[
                    'input_ids',
                    'pixel_values',
                    'pixel_values_videos',
                    'input_features',
                    'attention_mask',
                    'input_features_mask',
                    'position_ids',
                    'image_position_ids',
                    'video_position_ids',
                    'past_key_values',
                    'mm_token_type_ids',
                    'inputs_embeds',
                    'labels',
                    'use_cache',
                    'logits_to_keep',
                ],
                defaults={'logits_to_keep': 0, 'use_cache': False},
            )
            processed_data.append(processed_item)

        get_logger().info("Processed %s multimodal Gemma4 dense samples", len(processed_data))
        return processed_data

    def init_model(self, device: DeviceType = DeviceType.NPU) -> nn.Module:
        """
        Initialize model with vision encoder on CPU and text decoder with only 1 layer.

        Other decoder layers are loaded on-demand in ``generate_decoder_layer``.
        """
        try:
            from transformers import Gemma4ForConditionalGeneration
        except ImportError as e:
            raise InvalidModelError(
                "Failed to import Gemma4ForConditionalGeneration. "
                "Please install transformers with Gemma4 support (==5.5.3).",
                action="pip install transformers==5.5.3",
            ) from e

        text_config = self.config.text_config
        if getattr(text_config, 'enable_moe_block', False):
            raise UnsupportedError(
                "Gemma4ModelAdapter is for dense models (enable_moe_block=False). "
                "For MoE checkpoints, use --model_type gemma-4-26B-A4B-it.",
                action="Use --model_type gemma-4-26B-A4B-it for MoE variants.",
            )

        get_logger().info("Initializing Gemma4 dense model with v1 framework (layer-wise loading)...")

        global_torch_dtype = self.get_global_model_torch_dtype()
        origin_layers = text_config.num_hidden_layers
        get_logger().info("Model with %s dense text layers + vision encoder", origin_layers)

        text_config.num_hidden_layers = 1
        self.config.use_cache = False

        self.model_path = get_valid_read_path(str(self.model_path), is_dir=True, check_user_stat=True)

        get_logger().info("Loading vision encoder and first text decoder layer...")
        # The validated local path and local_files_only prevent Hub downloads.
        model = Gemma4ForConditionalGeneration.from_pretrained(  # nosec B615
            self.model_path,
            config=self.config,
            trust_remote_code=self.trust_remote_code,
            torch_dtype=global_torch_dtype,
            local_files_only=True,
            device_map="cpu",
            attn_implementation='eager',
        ).eval()

        text_config.num_hidden_layers = origin_layers

        text_config._attn_implementation = 'eager'

        get_logger().info("Loading weights for vision encoder, first decoder layer, and lm_head...")
        state_dict = self._get_state_dict(model)
        state_dict = _cast_floating_state_dict(state_dict, global_torch_dtype)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            get_logger().warning(
                "load_state_dict with strict=False: missing key(s) in state_dict: %s",
                missing,
            )
        if unexpected:
            get_logger().warning(
                "load_state_dict with strict=False: unexpected key(s) in state_dict: %s",
                unexpected,
            )

        if hasattr(model.config.text_config, 'num_attention_heads'):
            model.config.num_attention_heads = model.config.text_config.num_attention_heads
            get_logger().info("Set model.config.num_attention_heads = %s", model.config.num_attention_heads)
        if hasattr(model.config.text_config, 'num_key_value_heads'):
            model.config.num_key_value_heads = model.config.text_config.num_key_value_heads
            get_logger().info("Set model.config.num_key_value_heads = %s", model.config.num_key_value_heads)

        get_logger().info("Model initialized with %s layers (1 loaded, others will be loaded on-demand)", origin_layers)

        _promote_float_buffers_to_parameters(model)

        return model

    def generate_model_visit(self, model: nn.Module) -> Generator[ProcessRequest, Any, None]:
        """
        Generate model visit pipeline for layer-wise processing.

        Processing order:
            1. Vision encoder (model.vision_tower)
            2. Vision projection (model.embed_vision)
            3. Text decoder layers (model.language_model.layers[0..N])
        """
        get_logger().info("Processing vision encoder...")
        if model.model.vision_tower is not None:
            yield ProcessRequest(
                name="model.vision_tower",
                module=model.model.vision_tower,
                args=(),
                kwargs={},
            )

        get_logger().info("Processing vision projection...")
        if model.model.embed_vision is not None:
            yield ProcessRequest(
                name="model.embed_vision",
                module=model.model.embed_vision,
                args=(),
                kwargs={},
            )

        get_logger().info("Processing text decoder layers...")
        yield from generated_decoder_layer_visit_func(
            model,
            transformer_blocks=self.generate_decoder_layer(model),
        )

    def generate_model_forward(self, model: nn.Module, inputs: Any) -> Generator[ProcessRequest, Any, None]:
        """
        Generate model forward pipeline for layer analysis and static/mixed W8A8 calibration.

        Processing order mirrors ``Gemma4Model.forward``:
            1. Vision encoder (model.vision_tower)
            2. Vision projection (model.embed_vision)
            3. Merge image features into text embeddings
            4. Text decoder layers (model.language_model.layers[0..N])
        """
        from transformers import Gemma4ForConditionalGeneration
        from transformers.models.gemma4.modeling_gemma4 import create_causal_mask_mapping

        if isinstance(inputs, list):
            sample = inputs[0]
        else:
            sample = inputs

        text_config = self.config.text_config

        # --- Stage 1: unpack calibration sample tensors ---
        # GeneratedRunner recursively moves calibration samples to the model device.
        pixel_values = sample.get('pixel_values')
        pixel_position_ids = sample.get('pixel_position_ids')
        if pixel_position_ids is None:
            pixel_position_ids = sample.get('image_position_ids')
        input_ids = sample['input_ids']
        attention_mask = sample.get('attention_mask')
        mm_token_type_ids = sample.get('mm_token_type_ids')
        past_key_values = sample.get('past_key_values')

        if pixel_values is not None and pixel_position_ids is None:
            raise UnsupportedError(
                "Calibration sample is missing image position ids required by Gemma4 vision encoder.",
                action=(
                    "Ensure handle_dataset collects image_position_ids from AutoProcessor.apply_chat_template output."
                ),
            )

        # --- Stage 2: vision encoder + projection (per-tensor activation calibration) ---
        image_features = None
        if pixel_values is not None and model.model.vision_tower is not None:
            vision_outputs = yield ProcessRequest(
                name="model.vision_tower",
                module=model.model.vision_tower,
                args=(pixel_values,),
                kwargs={'pixel_position_ids': pixel_position_ids},
            )
            if model.model.embed_vision is not None:
                vision_hidden = _unwrap_forward_output(vision_outputs, "last_hidden_state")
                image_features = yield ProcessRequest(
                    name="model.embed_vision",
                    module=model.model.embed_vision,
                    args=(vision_hidden,),
                    kwargs={},
                )
                image_features = _unwrap_forward_output(image_features)

        # --- Stage 3: build text inputs_embeds ---
        image_mask, video_mask, audio_mask = model.model.get_placeholder_mask(input_ids=input_ids)
        multimodal_mask = image_mask | video_mask | audio_mask

        llm_input_ids = input_ids.clone()
        llm_input_ids[multimodal_mask] = text_config.pad_token_id
        inputs_embeds = model.model.language_model.embed_tokens(llm_input_ids)

        if image_features is not None:
            image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
            image_mask_expanded = image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
            inputs_embeds = inputs_embeds.masked_scatter(
                image_mask_expanded,
                image_features.to(inputs_embeds.device),
            )

        # --- Stage 4: causal masks, RoPE, and decoder layer forward ---
        position_ids = sample.get('position_ids')
        if position_ids is not None:
            if position_ids.ndim == 3:
                position_ids = position_ids[0]
        else:
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device).unsqueeze(0)

        if getattr(text_config, 'use_bidirectional_attention', None) == 'vision':
            causal_mask_mapping = create_causal_mask_mapping(
                self.config,
                inputs_embeds,
                attention_mask,
                past_key_values,
                position_ids,
                mm_token_type_ids,
                pixel_values,
                is_training=False,
            )
        else:
            causal_mask_mapping = Gemma4ForConditionalGeneration.create_masks_for_generate(
                self.config,
                inputs_embeds,
                attention_mask,
                past_key_values,
                position_ids,
                mm_token_type_ids,
            )

        language_model = model.model.language_model
        position_embeddings = {
            layer_type: language_model.rotary_emb(inputs_embeds, position_ids, layer_type)
            for layer_type in language_model.unique_layer_types
        }

        hidden_states = inputs_embeds
        shared_kv_states: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        for layer_idx, (name, layer) in enumerate(self.generate_decoder_layer(model)):
            layer_type = text_config.layer_types[layer_idx]
            hidden_states = yield ProcessRequest(
                name=name,
                module=layer,
                args=(hidden_states,),
                kwargs={
                    'shared_kv_states': shared_kv_states,
                    'attention_mask': causal_mask_mapping[layer_type],
                    'position_ids': position_ids,
                    'position_embeddings': position_embeddings[layer_type],
                    'past_key_values': past_key_values,
                },
            )
            hidden_states = _unwrap_forward_output(hidden_states)

    def generate_decoder_layer(self, model: nn.Module) -> Generator[Tuple[str, nn.Module], None, None]:
        """Generate decoder layers, loading them on-demand."""
        num_layers = self.config.text_config.num_hidden_layers

        for layer_idx in range(num_layers):
            name = f"model.language_model.layers.{layer_idx}"
            layer = self._load_decoder_if_not_exist(model, name, layer_idx)
            yield name, layer

    def enable_kv_cache(self, model: nn.Module, need_kv_cache: bool) -> None:
        """Enable/disable KV cache."""
        model.config.use_cache = need_kv_cache
        get_logger().info("KV cache %s", "enabled" if need_kv_cache else "disabled")

    def ascendv1_save_postprocess(self, model: nn.Module, save_directory: str) -> None:
        """Materialize the tied lm_head required by vLLM Ascend without rewriting all shards."""
        text_config = getattr(self.config, "text_config", None)
        tie_word_embeddings = bool(
            getattr(self.config, "tie_word_embeddings", False) or getattr(text_config, "tie_word_embeddings", False)
        )
        if not tie_word_embeddings:
            return

        index_path = os.path.join(save_directory, "quant_model_weights.safetensors.index.json")
        index_data = json_safe_load(index_path)
        weight_map = index_data["weight_map"]
        lm_head_key = "lm_head.weight"
        embed_key = "model.language_model.embed_tokens.weight"

        if lm_head_key not in weight_map:
            embed_file = weight_map.get(embed_key)
            if embed_file is None:
                raise InvalidModelError(
                    "Cannot export tied lm_head.weight because embed_tokens.weight is missing.",
                    action="Check the generated quant_model_weights.safetensors.index.json.",
                )
            embed_path = get_valid_read_path(
                os.path.join(save_directory, embed_file),
                extensions="safetensors",
                size_max=MAX_READ_FILE_SIZE_512G,
            )
            with safe_open(embed_path, framework="pt", device="cpu") as f:
                lm_head_weight = f.get_tensor(embed_key).clone()

            lm_head_file = "quant_model_weights-tied-lm-head.safetensors"
            lm_head_path = get_valid_write_path(
                os.path.join(save_directory, lm_head_file),
                extensions="safetensors",
                warn_exists=False,
            )
            save_file({lm_head_key: lm_head_weight}, lm_head_path)
            weight_map[lm_head_key] = lm_head_file
            metadata = index_data.get("metadata")
            if isinstance(metadata, dict) and isinstance(metadata.get("total_size"), int):
                metadata["total_size"] += lm_head_weight.numel() * lm_head_weight.element_size()
            json_safe_dump(index_data, index_path, indent=2)

        desc_path = os.path.join(save_directory, "quant_model_description.json")
        desc = json_safe_load(desc_path)
        desc.setdefault(lm_head_key, "FLOAT")
        json_safe_dump(desc, desc_path, indent=2)
        get_logger().info("Gemma4 dense export: materialized tied lm_head.weight")

    @lru_cache(maxsize=1)
    def _get_weight_map(self) -> Dict[str, str]:
        """Get weight map from model.safetensors.index.json."""
        index_path = os.path.join(self.model_path, "model.safetensors.index.json")
        index_data = json_safe_load(index_path)
        return index_data['weight_map']

    def _get_state_dict(self, module: nn.Module, prefix: str = "") -> Dict[str, torch.Tensor]:
        """Load parameters and persistent buffers for a module from safetensors files."""
        weight_map = self._get_weight_map()

        tensor_names = [name for name, _ in module.named_parameters()]
        tensor_names.extend(name for name, _ in module.named_buffers())

        file_groups = defaultdict(list)
        for tensor_name in tensor_names:
            full_name = f"{prefix}.{tensor_name}" if prefix else tensor_name
            if full_name in weight_map:
                file_name = weight_map[full_name]
                file_groups[file_name].append(tensor_name)

        state_dict = {}
        for file_name, names in tqdm(file_groups.items(), desc=f"Loading {prefix}", leave=False):
            file_path = os.path.join(self.model_path, file_name)
            file_path = get_valid_read_path(file_path, extensions='safetensors', size_max=MAX_READ_FILE_SIZE_512G)

            with safe_open(file_path, framework='pt', device='cpu') as f:
                for tensor_name in names:
                    full_name = f"{prefix}.{tensor_name}" if prefix else tensor_name
                    state_dict[tensor_name] = f.get_tensor(full_name)

        return state_dict

    def _load_decoder_if_not_exist(self, model: nn.Module, name: str, idx: int) -> nn.Module:
        """Load a specific decoder layer from safetensors if not already loaded."""
        try:
            decoder = model.get_submodule(name)
            try:
                _ = decoder.input_layernorm.weight.device
                get_logger().debug("Layer %s already loaded", idx)
                _promote_float_buffers_to_parameters(decoder)
                return decoder
            except RuntimeError:
                pass
        except AttributeError:
            pass

        get_logger().info("Loading decoder layer %s...", idx)

        from transformers.models.gemma4.modeling_gemma4 import Gemma4TextDecoderLayer

        text_config = self.config.text_config
        with patch.object(nn.Linear, 'reset_parameters', lambda _self: None):
            get_logger().info('Creating decoder layer %s structure...', idx)
            decoder = Gemma4TextDecoderLayer(
                text_config,
                layer_idx=idx,
            )

            state_dict = self._get_state_dict(decoder, prefix=name)
            state_dict = _cast_floating_state_dict(state_dict, self.get_global_model_torch_dtype())
            decoder.load_state_dict(state_dict, strict=False)
            decoder.eval()
            _promote_float_buffers_to_parameters(decoder)

            module_list: nn.ModuleList = model.model.language_model.layers
            if len(module_list) <= idx:
                module_list.append(decoder)
            else:
                module_list[idx] = decoder

            get_logger().info('Decoder layer %s loaded successfully', idx)

        return decoder
