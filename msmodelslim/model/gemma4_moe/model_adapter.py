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

# pylint: disable=logging-fstring-interpolation,no-name-in-module,too-many-ancestors,ungrouped-imports

import gc
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
from transformers import AutoProcessor

from msmodelslim.core.const import DeviceType
from msmodelslim.app.naive_quantization.model_info_interface import ModelInfoInterface
from msmodelslim.core.base.protocol import ProcessRequest
from msmodelslim.model.common.layer_wise_forward import generated_decoder_layer_visit_func
from msmodelslim.model.interface_hub import (
    ModelSlimPipelineInterfaceV1,
)
from msmodelslim.infra.dataset_loader.vlm_dataset_loader import VlmCalibSample
from msmodelslim.model.common.vlm_base import VLMBaseModelAdapter
from msmodelslim.utils.exception import InvalidModelError, UnsupportedError
from msmodelslim.utils.logging import logger_setter, get_logger
from msmodelslim.utils.security import (
    get_valid_read_path,
    json_safe_load,
    MAX_READ_FILE_SIZE_512G,
)

from .moe_utils import UnstackedGemma4TextExperts

try:
    from transformers import Gemma4ForConditionalGeneration
    from transformers.models.gemma4.modeling_gemma4 import (
        Gemma4TextDecoderLayer,
        Gemma4TextExperts,
        create_causal_mask_mapping,
    )
except ImportError:
    Gemma4ForConditionalGeneration = None
    Gemma4TextDecoderLayer = None
    Gemma4TextExperts = None
    create_causal_mask_mapping = None

# AscendV1 saver 只落盘 named_parameters；这些 HF buffer 需提升为 Parameter。
_FLOAT_BUFFER_ATTRS_TO_PROMOTE = ("std_bias", "std_scale", "layer_scalar")


def _promote_float_buffers_to_parameters(root: nn.Module) -> None:
    """Promote vision ``std_*`` and per-layer ``layer_scalar`` buffers to Parameters."""
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
            # Keep fp32 to match floating checkpoint / HF _keep_in_fp32_modules_strict.
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
class Gemma4MoeModelAdapter(
    VLMBaseModelAdapter,
    ModelInfoInterface,
    ModelSlimPipelineInterfaceV1,
):
    """
    Gemma4 VLM + MoE Model Adapter for msModelSlim.

    Supports Gemma4ForConditionalGeneration with enable_moe_block=True:
    - Vision encoder + MoE text decoder (dense MLP + routed experts per layer)
    - Mixed sliding/full attention layers
    - Bidirectional attention for vision tokens

    W8A8 dynamic quantization (per_token act + per_channel weight) is data-free and
    uses ``generate_model_visit`` only. Mixed/static W8A8 (per_tensor activation) requires
    ``handle_dataset`` + ``generate_model_forward`` for activation calibration.

    MoE 3D expert weights (``experts.gate_up_proj`` / ``experts.down_proj`` in float)
    are split into per-expert ``nn.Linear`` for W8A8 and saved as
    ``experts.{e}.gate_proj`` / ``up_proj`` / ``down_proj`` (no export refusion).
    """

    def __init__(self, model_type: str, model_path: Path, trust_remote_code: bool = False):
        self._processor = None
        super().__init__(model_type, model_path, trust_remote_code)

    def get_model_pedigree(self) -> str:
        """Return model pedigree for best practice matching."""
        return 'gemma4_moe'

    def get_model_type(self) -> str:
        """Return model type."""
        return self.model_type

    def handle_dataset(self, dataset: Any, device: DeviceType = DeviceType.NPU) -> List[Any]:
        """
        Prepare multimodal calibration samples for static/mixed W8A8 (per_tensor activation).

        Supported sample structure: ``VlmCalibSample(text: str, image: Optional[str])``.

        Data-free dynamic W8A8 (``gemma4_moe_w8a8.yaml``) does not run forward; an empty
        dataset is acceptable. Mixed configs (``gemma4_moe_w8a8_mix.yaml``) require image+text.
        """
        if not dataset:
            get_logger().info(
                "No calibration dataset provided; forward-based quantization will not run.",
            )
            return []

        self._processor = AutoProcessor.from_pretrained(  # nosec B615
            self.model_path,
            trust_remote_code=self.trust_remote_code,
            local_files_only=True,
        )

        for item in dataset:
            is_dataclass = isinstance(item, VlmCalibSample)
            image_path = item.image if is_dataclass else item.get('image')
            text = item.text if is_dataclass else item.get('text')
            if image_path is None or text is None:
                raise UnsupportedError(
                    ("Gemma4 MoE adapter requires both image and text for calibration forward."),
                    action=(
                        "Please use multimodal (image+text) calibration data with "
                        "gemma4_moe_w8a8_mix.yaml or static W8A8; pure-text or "
                        "missing image is not supported yet."
                    ),
                )

        processed_data = []
        for item in tqdm(dataset, desc="Processing calibration dataset"):
            image_path = get_valid_read_path(item.image if isinstance(item, VlmCalibSample) else item.get('image'))
            text = item.text if isinstance(item, VlmCalibSample) else item.get('text')

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
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
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

        get_logger().info("Processed %d multimodal Gemma4 MoE samples", len(processed_data))
        return processed_data

    def init_model(self, device: DeviceType = DeviceType.NPU) -> nn.Module:
        """
        Initialize model with vision encoder on CPU and text decoder with only 1 layer.

        Strategy:
            - Save original layer count
            - Temporarily set num_hidden_layers to 1
            - Load model with vision encoder + 1 text decoder layer
            - Restore original layer count
            - Other layers will be loaded on-demand in generate_decoder_layer
        """
        if Gemma4ForConditionalGeneration is None:
            raise InvalidModelError(
                "Failed to import Gemma4ForConditionalGeneration. "
                "Please install transformers with Gemma4 support (==5.5.3).",
                action="pip install transformers==5.5.3",
            )

        text_config = self.config.text_config
        if not getattr(text_config, 'enable_moe_block', False):
            raise UnsupportedError(
                "Gemma4MoeModelAdapter is for MoE models (enable_moe_block=True). "
                "For dense checkpoints (e.g. gemma-4-31B-it), use --model_type gemma4.",
                action="Use gemma4 adapter in config.ini for dense variants.",
            )

        get_logger().info("Initializing Gemma4 MoE model with v1 framework (layer-wise loading)...")

        origin_layers = text_config.num_hidden_layers
        get_logger().info("Model with %d MoE text layers + vision encoder", origin_layers)

        text_config.num_hidden_layers = 1
        self.config.use_cache = False

        self.model_path = get_valid_read_path(str(self.model_path), is_dir=True, check_user_stat=True)

        get_logger().info("Loading vision encoder and first text decoder layer...")
        model = Gemma4ForConditionalGeneration.from_pretrained(  # nosec B615
            self.model_path,
            config=self.config,
            trust_remote_code=self.trust_remote_code,
            torch_dtype="auto",
            local_files_only=True,
            device_map="cpu",
            attn_implementation='eager',
        ).eval()

        text_config.num_hidden_layers = origin_layers

        text_config._attn_implementation = 'eager'

        get_logger().info("Loading weights for vision encoder, first decoder layer, and lm_head...")
        state_dict = self._get_state_dict(model)
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
            get_logger().debug(
                "Set model.config.num_attention_heads = %s",
                model.config.num_attention_heads,
            )
        if hasattr(model.config.text_config, 'num_key_value_heads'):
            model.config.num_key_value_heads = model.config.text_config.num_key_value_heads
            get_logger().debug(
                "Set model.config.num_key_value_heads = %s",
                model.config.num_key_value_heads,
            )

        get_logger().info(
            "Model initialized with %d layers (1 loaded, others will be loaded on-demand)",
            origin_layers,
        )

        # Promote float buffers before save path (parameters-only) drops them.
        _promote_float_buffers_to_parameters(model)

        if text_config.enable_moe_block and len(model.model.language_model.layers) > 0:
            get_logger().info("Layer 0: converting MoE experts to nn.Linear...")
            self._convert_moe_experts(model.model.language_model.layers[0], 0)
            get_logger().info("Layer 0: MoE experts conversion completed")

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
        Generate model forward pipeline for static/mixed W8A8 activation calibration.

        Processing order mirrors ``Gemma4Model.forward``:
            1. Vision encoder (model.vision_tower)
            2. Vision projection (model.embed_vision)
            3. Merge image features into text embeddings
            4. Text decoder layers (model.language_model.layers[0..N])
        """
        if Gemma4ForConditionalGeneration is None or create_causal_mask_mapping is None:
            raise InvalidModelError(
                "Failed to import Gemma4 modules required for forward calibration.",
                action="pip install transformers==5.5.3",
            )

        if isinstance(inputs, list):
            sample = inputs[0]
        else:
            sample = inputs

        text_config = self.config.text_config
        pixel_values = sample.get('pixel_values')
        pixel_position_ids = sample.get('pixel_position_ids')
        if pixel_position_ids is None:
            pixel_position_ids = sample.get('image_position_ids')
        input_ids = sample['input_ids']
        attention_mask = sample.get('attention_mask')
        mm_token_type_ids = sample.get('mm_token_type_ids')

        if pixel_values is not None and pixel_position_ids is None:
            raise UnsupportedError(
                "Calibration sample is missing image position ids required by Gemma4 vision encoder.",
                action="Ensure handle_dataset collects image_position_ids from Gemma4Processor output.",
            )

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

        position_ids = sample.get('position_ids')
        if position_ids is not None:
            if position_ids.ndim == 3:
                position_ids = position_ids[0]
        else:
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device).unsqueeze(0)

        past_key_values = None
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
            per_layer_input = None
            hidden_states = yield ProcessRequest(
                name=name,
                module=layer,
                args=(hidden_states,),
                kwargs={
                    'per_layer_input': per_layer_input,
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
        get_logger().info("KV cache %s", 'enabled' if need_kv_cache else 'disabled')

    @lru_cache(maxsize=1)
    def _get_weight_map(self) -> Dict[str, str]:
        """Get weight map from model.safetensors.index.json."""
        index_path = os.path.join(self.model_path, "model.safetensors.index.json")
        index_data = json_safe_load(index_path)
        return index_data['weight_map']

    def _get_state_dict(self, module: nn.Module, prefix: str = "") -> Dict[str, torch.Tensor]:
        """Load state dict for a specific module from safetensors files.

        Includes both Parameters and persistent Buffers (e.g. ``layer_scalar``,
        vision ``std_bias`` / ``std_scale``). Layer-wise loading creates fresh
        decoder layers whose buffers default to ones; without loading buffers
        from the checkpoint those values stay wrong and get promoted/saved.
        """
        weight_map = self._get_weight_map()

        # Parameters + buffers: layer_scalar etc. are registered as buffers in HF.
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
        """Load a specific MoE decoder layer from safetensors if not already loaded."""
        text_config = self.config.text_config
        try:
            decoder = model.get_submodule(name)
            try:
                _ = decoder.input_layernorm.weight.device
                get_logger().debug("Layer %d already loaded", idx)
                _promote_float_buffers_to_parameters(decoder)
                if text_config.enable_moe_block:
                    self._convert_moe_experts(decoder, idx)
                return decoder
            except RuntimeError:
                pass
        except AttributeError:
            pass

        get_logger().info("Loading decoder layer %d...", idx)

        if Gemma4TextDecoderLayer is None:
            raise InvalidModelError(
                "Failed to import Gemma4TextDecoderLayer.",
                action="pip install transformers==5.5.3",
            )

        with patch.object(nn.Linear, 'reset_parameters', lambda _self: None):
            get_logger().info("Creating decoder layer %d structure...", idx)
            decoder = Gemma4TextDecoderLayer(
                text_config,
                layer_idx=idx,
            )

            state_dict = self._get_state_dict(decoder, prefix=name)
            decoder.load_state_dict(state_dict, strict=False)
            decoder.eval()
            _promote_float_buffers_to_parameters(decoder)

            module_list: nn.ModuleList = model.model.language_model.layers
            if len(module_list) <= idx:
                module_list.append(decoder)
            else:
                module_list[idx] = decoder

            get_logger().info("Decoder layer %d loaded successfully", idx)

        if text_config.enable_moe_block:
            self._convert_moe_experts(decoder, idx)

        return decoder

    def _convert_moe_experts(self, layer: nn.Module, layer_idx: int) -> None:
        """
        Split float 3D ``Gemma4TextExperts`` into per-expert ``nn.Linear`` for W8A8.

        Export keys (same as visit):
            ...experts.{e}.gate_proj.weight / .weight_scale
            ...experts.{e}.up_proj.weight / .weight_scale
            ...experts.{e}.down_proj.weight / .weight_scale
        """
        if Gemma4TextExperts is None:
            get_logger().warning("Gemma4TextExperts not available, skipping MoE conversion")
            return

        original_experts = layer.experts
        if isinstance(original_experts, UnstackedGemma4TextExperts):
            return
        if not isinstance(original_experts, Gemma4TextExperts):
            get_logger().warning(
                "Layer %d experts is not Gemma4TextExperts, skipping conversion. Got: %s",
                layer_idx,
                type(original_experts),
            )
            return

        get_logger().info("Layer %d: converting MoE experts to nn.Linear...", layer_idx)
        unstacked_experts = UnstackedGemma4TextExperts(
            self.config.text_config,
            original_experts,
            copy_weights=False,
        )
        unstacked_experts._transform_weights_from_original(original_experts, in_place=True)
        unstacked_experts.eval()
        layer.experts = unstacked_experts
        del original_experts
        gc.collect()
        get_logger().info("Layer %d: MoE experts conversion completed", layer_idx)
