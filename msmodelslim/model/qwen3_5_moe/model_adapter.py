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

import os
from collections import defaultdict
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from typing import List, Any, Generator, Tuple, Dict
from unittest.mock import patch

import torch
from safetensors import safe_open
from torch import nn
from tqdm import tqdm
from transformers import AutoProcessor
from transformers.masking_utils import create_causal_mask

from transformers import Qwen3_5MoeForConditionalGeneration, Qwen3_5ForConditionalGeneration
from msmodelslim.core.const import DeviceType
from msmodelslim.app.naive_quantization.model_info_interface import ModelInfoInterface
from msmodelslim.core.base.protocol import ProcessRequest
from msmodelslim.core.graph import AdapterConfig, MappingConfig
from msmodelslim.model.common.layer_wise_forward import generated_decoder_layer_visit_func
from msmodelslim.model.interface_hub import IterSmoothInterface, FlexSmoothQuantInterface, ModelSlimPipelineInterfaceV1
from msmodelslim.model.common.vlm_base import VLMBaseModelAdapter
from msmodelslim.infra.dataset_loader.vlm_dataset_loader import VlmCalibSample
from msmodelslim.utils.exception import InvalidModelError, UnsupportedError
from msmodelslim.utils.logging import logger_setter, get_logger
from msmodelslim.utils.security import get_valid_read_path, json_safe_load, MAX_READ_FILE_SIZE_512G

from .moe_utils import Qwen3_5MoeSparseMoeBlockWithMLP, convert_experts_to_mlp
from .modeling_qwen3_5_mtp import Qwen3_5MultiTokenPredictor


def remove_zero_and_shift(matrix):
    """Remove first padding-zero per row and shift left, appending 0 at end.

    Used in MTP to shift input_ids for next-token prediction.
    Reference: deepseek_v3/mtp_quant_module.py
    """
    n, m = matrix.shape
    zero_pos = (matrix == 0).int().argmax(dim=1)
    col_indices = torch.arange(m, device=matrix.device).expand(n, -1)
    mask = col_indices != zero_pos.unsqueeze(1)
    filtered = matrix[mask].view(n, m - 1)
    result = torch.cat([filtered, torch.zeros(n, 1, device=matrix.device)], dim=1)
    return result.to(matrix)


@contextmanager
def default_dtype(dtype):
    """自定义默认 dtype 上下文管理器"""
    original_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(original_dtype)


@logger_setter()
class Qwen3_5ModelAdapter(  # pylint: disable=too-many-ancestors
    VLMBaseModelAdapter, ModelInfoInterface, ModelSlimPipelineInterfaceV1, IterSmoothInterface, FlexSmoothQuantInterface
):
    """
    V1 Framework adapter for Qwen3-VL-MoE models.

    Key features:
    - Layer-wise loading for text decoder
    - Vision encoder processed as a whole
    - Automatic MoE fusion layer conversion via MoeConverterProcessor
    - Multimodal calibration dataset support

    Architecture:
        model.visual (VisionEncoder) - Loaded once, processed first
        model.language_model.layers[i] (TextDecoder) - Loaded layer-by-layer
    """

    def __init__(self, model_type: str, model_path: Path, trust_remote_code: bool = False):
        # Cache for processor (used in dataset handling)
        self._processor = None
        self._tokenizer = None
        super().__init__(model_type, model_path, trust_remote_code)

    def _create_model_instance(self, model_cls) -> nn.Module:
        """创建模型实例的基础方法，子类可重写以修改参数（如torch_dtype）"""
        return model_cls.from_pretrained(  # nosec B615
            self.model_path,
            config=self.config,
            trust_remote_code=self.trust_remote_code,
            torch_dtype="auto",
            local_files_only=True,
            device_map="cpu",  # All on CPU for now
            attn_implementation='eager',  # Required: prevents KeyError when accessing ALL_ATTENTION_FUNCTIONS
        ).eval()

    def get_model_pedigree(self) -> str:
        """Return model pedigree for best practice matching"""
        return 'qwen3_5_moe'

    def get_model_type(self) -> str:
        """Return model type"""
        return self.model_type

    def handle_dataset(self, dataset: Any, device: DeviceType = DeviceType.NPU) -> List[Any]:
        """
        Handle multimodal VLM calibration dataset.

        Supported sample structure (preferred):
            VlmCalibSample(text: str, image: Optional[str])

        For image+text samples:
            [{"role": "user", "content": [
                {"type": "image", "image": "<path>"},
                {"type": "text", "text": text}
            ]}]

        Returns a list of processor-ready dicts for LayerWiseRunner.
        """
        self._processor = AutoProcessor.from_pretrained(  # nosec B615
            self.model_path, trust_remote_code=self.trust_remote_code, local_files_only=True
        )

        # Validate dataset modality: Qwen3-VL-MoE adapter expects image+text only (no pure-text or mixed-without-image)
        for item in dataset:
            is_dataclass = isinstance(item, VlmCalibSample)
            image_path = item.image if is_dataclass else item.get('image')
            text = item.text if is_dataclass else item.get('text')
            if image_path is None or text is None:
                raise UnsupportedError(
                    ("Qwen3-VL-MoE adapter currently requires both image and text for calibration."),
                    action=(
                        "Please use multimodal (image+text) calibration data; pure-text or "
                        "missing image is not supported yet."
                    ),
                )

        # Preprocess each sample
        processed_data = []
        for item in tqdm(dataset, desc="Processing calibration dataset"):
            # Support dataclass
            image_path = item.image
            text = item.text

            # Build messages based on presence of image
            # Validate image path
            image_path = get_valid_read_path(image_path)
            content = [
                {"type": "image", "image": str(image_path)},
                {"type": "text", "text": text},
            ]

            messages = [{"role": "user", "content": content}]

            inputs = self._processor.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
            )

            processed_item = self._collect_inputs_to_device(
                inputs,
                device,
                keys=[
                    'input_ids',
                    'attention_mask',
                    'position_ids',
                    'past_key_values',
                    'inputs_embeds',
                    'labels',
                    'pixel_values',
                    'pixel_values_videos',
                    'image_grid_thw',
                    'video_grid_thw',
                    'cache_position',
                    'logits_to_keep',
                ],
                defaults={'logits_to_keep': 0},
            )

            processed_data.append(processed_item)

        get_logger().info("Processed %d multimodal vlm samples", len(processed_data))
        return processed_data

    def init_model(self, device: DeviceType = DeviceType.NPU) -> nn.Module:
        """
        Initialize model with vision encoder on CPU and text decoder with only 1 layer.

        Strategy (similar to DeepSeek-V3):
            - Save original layer count
            - Temporarily set num_hidden_layers to 1
            - Load model with vision encoder + 1 text decoder layer
            - Restore original layer count
            - Other layers will be loaded on-demand in generate_decoder_layer

        Returns:
            Model with vision encoder + 1 decoder layer loaded, others on meta
        """

        get_logger().info("Initializing Qwen3-VL-MoE model with v1 framework (layer-wise loading)...")

        # Save original layer count
        origin_layers = self.config.text_config.num_hidden_layers
        get_logger().info(
            "Model with %d text layers + %d vision layers", origin_layers, self.config.vision_config.depth
        )

        # Temporarily set to 1 layer for initialization
        self.config.text_config.num_hidden_layers = 1
        self.config.use_cache = False  # Disable cache to save memory

        # Validate model path
        self.model_path = get_valid_read_path(str(self.model_path), is_dir=True, check_user_stat=True)

        # Load model with only 1 text decoder layer
        # Vision encoder is fully loaded, text decoder has only 1 layer
        get_logger().info("Loading vision encoder and first text decoder layer...")
        if self.config.architectures[0] == "Qwen3_5MoeForConditionalGeneration":
            model = self._create_model_instance(Qwen3_5MoeForConditionalGeneration)
        elif self.config.architectures[0] == "Qwen3_5ForConditionalGeneration":
            model = self._create_model_instance(Qwen3_5ForConditionalGeneration)
        else:
            raise InvalidModelError(
                f"Invalid model architecture: {self.config.architectures[0]}",
                action="Please verify the model architecture is correct.",
            )

        # Restore original layer count
        self.config.text_config.num_hidden_layers = origin_layers

        # Ensure _attn_implementation is set for dynamically loaded layers
        # This prevents KeyError when layers access ALL_ATTENTION_FUNCTIONS[config._attn_implementation]
        self.config.text_config._attn_implementation = 'eager'

        # CRITICAL: Copy text_config attention heads to model.config for OV smoothing
        # BaseSmoothProcessor._apply_standard_ov_smooth() reads from model.config, not model.config.text_config
        # This must be done AFTER model is loaded
        if hasattr(model.config.text_config, 'num_attention_heads'):
            model.config.num_attention_heads = model.config.text_config.num_attention_heads
            get_logger().info("Set model.config.num_attention_heads = %d", model.config.num_attention_heads)
        if hasattr(model.config.text_config, 'num_key_value_heads'):
            model.config.num_key_value_heads = model.config.text_config.num_key_value_heads
            get_logger().info("Set model.config.num_key_value_heads = %d", model.config.num_key_value_heads)

        get_logger().info("Model initialized with %d layers (1 loaded, others will be loaded on-demand)", origin_layers)

        return model

    def generate_model_visit(self, model: nn.Module) -> Generator[ProcessRequest, Any, None]:
        """
        Generate model visit pipeline for layer-wise processing.

        Uses the common layer-wise visit function for consistent behavior.

        Processing order:
            1. Vision encoder (model.visual) - processed as a whole
            2. Text decoder layers (model.language_model.layers[0..N]) - loaded on-demand

        Yields:
            ProcessRequest(name, module, args, kwargs)
        """
        # 1. Process vision encoder first
        get_logger().info("Processing vision encoder...")
        yield ProcessRequest(name="model.visual", module=model.model.visual, args=(), kwargs={})

        # 2. Process text decoder layers one by one using standard visit function
        get_logger().info("Processing text decoder layers...")
        yield from generated_decoder_layer_visit_func(model, transformer_blocks=self.generate_decoder_layer(model))

    def generate_model_forward(self, model: nn.Module, inputs: Any) -> Generator[ProcessRequest, Any, None]:
        """
        Generate model forward pipeline for calibration.

        This is more complex as we need to:
            1. Run vision encoder to get image features
            2. Merge image features into text embeddings
            3. Run each text decoder layer with proper inputs

        Args:
            model: The model
            inputs: Preprocessed data from handle_dataset

        Yields:
            ProcessRequest with forward results
        """
        # For multimodal models, forward is more complex
        # We need to handle the vision-language fusion

        # 1. Extract first sample for calibration
        if isinstance(inputs, list):
            sample = inputs[0]
        else:
            sample = inputs

        # 2. Vision encoder forward
        pixel_values = sample['pixel_values']
        image_grid_thw = sample['image_grid_thw']

        # Yield vision encoder result
        vision_outputs = yield ProcessRequest(
            name="model.visual", module=model.model.visual, args=(pixel_values, image_grid_thw), kwargs={}
        )

        image_embeds = vision_outputs['pooler_output']

        # 3. Prepare inputs for text decoder
        input_ids = sample['input_ids']
        attention_mask = sample['attention_mask']

        # Get input embeddings
        inputs_embeds = model.model.language_model.embed_tokens(input_ids)

        # CRITICAL: Merge visual features into text embeddings
        # This mimics Qwen3VLMoeModel.forward (lines 1320-1358)
        if isinstance(image_embeds, (list, tuple)):
            image_embeds_cat = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
        else:
            image_embeds_cat = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)

        # Get image token mask for fusion
        image_mask = (input_ids == model.config.image_token_id).unsqueeze(-1).expand_as(inputs_embeds)
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds_cat)

        # Get cache_position for attention mask creation
        cache_position = torch.arange(0, inputs_embeds.shape[1], device=inputs_embeds.device)

        # Get position ids
        position_ids, rope_deltas = model.model.get_rope_index(
            input_ids=input_ids, image_grid_thw=image_grid_thw, attention_mask=attention_mask
        )

        # Expand position_ids if needed (3D format for mROPE)
        if position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)

        if position_ids.ndim == 3 and position_ids.shape[0] == 4:
            text_position_ids = position_ids[0]
            position_ids = position_ids[1:]
        else:
            text_position_ids = position_ids[0]

        # CRITICAL: Convert 2D attention_mask to 4D causal mask
        # This is what Qwen3VLMoeTextModel.forward does internally
        causal_mask = create_causal_mask(
            config=model.config.text_config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=None,
            position_ids=text_position_ids,
        )

        # Align with text model behavior for linear-attention layers.
        linear_attn_mask = attention_mask
        if cache_position[0] > 0 or (attention_mask is not None and torch.all(attention_mask == 1)):
            linear_attn_mask = None

        # Create position embeddings (shared across layers)
        position_embeddings = model.model.language_model.rotary_emb(inputs_embeds, position_ids)

        # 4. Process each decoder layer
        # num_layers = self.config.text_config.num_hidden_layers
        has_mtp = self._has_mtp()
        hidden_states = inputs_embeds
        for name, layer in self.generate_decoder_layer(model):
            is_mtp_layer = has_mtp and name.startswith("mtp")

            if is_mtp_layer:
                # MTP preprocessing: norm -> lm_head -> shift ids -> embed -> norm -> project
                hidden_states, mtp_kwargs = self.mtp_preprocess(model, inputs=sample, hidden_states=hidden_states)
                hidden_states = yield ProcessRequest(name=name, module=layer, args=(hidden_states,), kwargs=mtp_kwargs)
            else:
                layer_mask = (
                    linear_attn_mask if getattr(layer, "layer_type", None) == "linear_attention" else causal_mask
                )
                # Yield layer result
                hidden_states = yield ProcessRequest(
                    name=name,
                    module=layer,
                    args=(hidden_states,),
                    kwargs={
                        'attention_mask': layer_mask,
                        'position_ids': position_ids,
                        'cache_position': cache_position,
                        'position_embeddings': position_embeddings,
                        'past_key_values': None,
                    },
                )

    def generate_decoder_layer(self, model: nn.Module) -> Generator[Tuple[str, nn.Module], None, None]:
        """
        Generate decoder layers, loading them on-demand.

        Similar to DeepSeekV3's approach but for Qwen3-VL-MoE.
        Each layer is loaded from safetensors file, and MoE layers are converted immediately.
        After all base layers, yields the MTP decoder layer if present.

        Yields:
            (layer_name, layer_module) tuples
        """
        num_layers = self.config.text_config.num_hidden_layers

        for layer_idx in range(num_layers):
            name = f"model.language_model.layers.{layer_idx}"

            # Load layer if not exists (includes MoE conversion for MoE layers)
            layer = self._load_decoder_if_not_exist(model, name, layer_idx)

            yield name, layer

        # Yield MTP decoder layer after all base layers
        if self._has_mtp():
            mtp_layer = self._load_mtp_if_not_loaded(model)
            mtp_name = "mtp"
            yield mtp_name, mtp_layer

    def enable_kv_cache(self, model: nn.Module, need_kv_cache: bool) -> None:
        """
        Enable/disable KV cache.

        For calibration, we typically don't need KV cache.
        """
        model.config.use_cache = need_kv_cache
        get_logger().info("KV cache %s", 'enabled' if need_kv_cache else 'disabled')

    def get_adapter_config_for_subgraph(self) -> List[AdapterConfig]:
        adapter_config = []
        for layer_idx in range(
            self.config.text_config.full_attention_interval - 1,
            self.config.text_config.num_hidden_layers,
            self.config.text_config.full_attention_interval,
        ):
            # Norm-Linear融合的映射配置：输入层归一化到QKV投影
            norm_linear_mapping_config = MappingConfig(
                source=f"model.language_model.layers.{layer_idx}.input_layernorm",  # 第一个LayerNorm
                targets=[
                    f"model.language_model.layers.{layer_idx}.self_attn.k_proj",
                    f"model.language_model.layers.{layer_idx}.self_attn.q_proj",
                    f"model.language_model.layers.{layer_idx}.self_attn.v_proj",
                ],  # 注意力层的QKV投影
            )

            # 为当前layer添加配置
            adapter_config.extend(
                [
                    AdapterConfig(subgraph_type="norm-linear", mapping=norm_linear_mapping_config),
                ]
            )
        return adapter_config

    @lru_cache(maxsize=1)
    def _get_weight_map(self) -> Dict[str, str]:
        """Get weight map from model.safetensors.index.json"""
        index_path = os.path.join(self.model_path, "model.safetensors.index.json")
        index_data = json_safe_load(index_path)
        return index_data['weight_map']

    def _get_state_dict(self, module: nn.Module, prefix: str = "") -> Dict[str, torch.Tensor]:
        """
        Load state dict for a specific module from safetensors files.

        Args:
            module: The module to load weights for
            prefix: Name prefix for the module in the full model

        Returns:
            State dict for the module
        """
        weight_map = self._get_weight_map()

        # Get all parameter names for this module
        param_names = [name for name, _ in module.named_parameters()]

        # Group by safetensors file
        file_groups = defaultdict(list)
        for param_name in param_names:
            full_name = f"{prefix}.{param_name}" if prefix else param_name
            if full_name in weight_map:
                file_name = weight_map[full_name]
                file_groups[file_name].append(param_name)

        # Load weights file by file
        state_dict = {}
        for file_name, names in tqdm(file_groups.items(), desc=f"Loading {prefix}", leave=False):
            file_path = os.path.join(self.model_path, file_name)
            file_path = get_valid_read_path(file_path, extensions='safetensors', size_max=MAX_READ_FILE_SIZE_512G)

            with safe_open(file_path, framework='pt', device='cpu') as f:
                for param_name in names:
                    full_name = f"{prefix}.{param_name}" if prefix else param_name
                    state_dict[param_name] = f.get_tensor(full_name)

        return state_dict

    def _load_decoder_if_not_exist(self, model: nn.Module, name: str, idx: int) -> nn.Module:
        """
        Load a specific decoder layer from safetensors if not already loaded.

        This method:
        1. Checks if layer already exists and is loaded
        2. If not, creates layer structure (without initializing weights)
        3. Loads weights from safetensors files
        4. If it's a MoE layer, converts 3D fused weights to standard nn.Linear
        5. Returns the loaded (and potentially converted) layer

        Args:
            model: The model
            name: Full layer name (e.g., "model.language_model.layers.0")
            idx: Layer index

        Returns:
            Loaded decoder layer module
        """
        with patch.object(nn.Linear, 'reset_parameters', lambda _self: None), default_dtype(torch.bfloat16):
            try:
                # Try to access the layer
                decoder = model.get_submodule(name)
                # Check if it's actually loaded (not on meta device)
                try:
                    _ = decoder.input_layernorm.weight.device
                    # If we can access the device, layer is loaded
                    get_logger().debug("Layer %d already loaded", idx)
                    if (
                        hasattr(decoder, "mlp")
                        and hasattr(decoder.mlp, "experts")
                        and not isinstance(decoder.mlp, Qwen3_5MoeSparseMoeBlockWithMLP)
                    ):
                        decoder.mlp = convert_experts_to_mlp(decoder.mlp, self.config.text_config)
                    return decoder
                except RuntimeError:
                    # Weight is on meta device, need to load
                    pass
            except AttributeError:
                # Layer doesn't exist in the module list yet
                pass

            get_logger().info("Loading decoder layer %d...", idx)

            # Disable reset_parameters to avoid slow and unnecessary initialization
            # We will load weights from safetensors immediately after

            get_logger().info('Creating decoder layer %d structure...', idx)

            # Create layer structure (weights will be on meta or uninitialized)
            module_list: nn.ModuleList = model.model.language_model.layers
            template_module = module_list[0]
            decoder = template_module.__class__(config=self.config.text_config, layer_idx=idx)

            # Load weights from safetensors
            state_dict = self._get_state_dict(decoder, prefix=name)
            decoder.load_state_dict(state_dict)
            decoder.eval()
            if (
                hasattr(decoder, "mlp")
                and hasattr(decoder.mlp, "experts")
                and not isinstance(decoder.mlp, Qwen3_5MoeSparseMoeBlockWithMLP)
            ):
                decoder.mlp = convert_experts_to_mlp(decoder.mlp, self.config.text_config)
            # Add layer to model's layer list
            module_list: nn.ModuleList = model.model.language_model.layers
            if len(module_list) <= idx:
                module_list.append(decoder)
            else:
                module_list[idx] = decoder

            get_logger().info('Decoder layer %d loaded successfully', idx)

        return decoder

    # ===== MTP (Multi-Token Prediction) support =====

    def _has_mtp(self) -> bool:
        """Check if the model checkpoint contains MTP weights."""
        weight_map = self._get_weight_map()
        return any('mtp.' in k for k in weight_map)

    def _load_mtp_predictor(self, model: nn.Module) -> 'Qwen3_5MultiTokenPredictor':
        """
        Create Qwen3_5MultiTokenPredictor and load weights from safetensors.

        Load strategy:
        - Keep original checkpoint key names (e.g. ``mtp.layers.0.xxx``)
        - Match directly against predictor parameter names under ``mtp.*``
        - No parameter renaming/remapping

        Args:
            model: The main model

        Returns:
            Loaded Qwen3_5MultiTokenPredictor instance
        """
        _ = model
        config = self.config.text_config

        with patch.object(nn.Linear, 'reset_parameters', lambda _self: None), default_dtype(torch.bfloat16):
            get_logger().info("Creating MTP predictor...")
            predictor = Qwen3_5MultiTokenPredictor(config)

        weight_map = self._get_weight_map()
        mtp_keys = [key for key in weight_map if key.startswith('mtp.')]

        # Group by safetensors file
        file_groups = defaultdict(list)
        for ckpt_key in mtp_keys:
            file_groups[weight_map[ckpt_key]].append(ckpt_key)

        # Build direct-name view with "mtp." prefix so checkpoint and model paths match.
        mtp_container = nn.Module()
        mtp_container.add_module("mtp", predictor)
        params_dict = dict(mtp_container.named_parameters())
        loaded = set()

        for file_name, keys in tqdm(file_groups.items(), desc="Loading MTP weights", leave=False):
            file_path = os.path.join(self.model_path, file_name)
            file_path = get_valid_read_path(file_path, extensions='safetensors', size_max=MAX_READ_FILE_SIZE_512G)

            with safe_open(file_path, framework='pt', device='cpu') as f:
                for ckpt_key in keys:
                    tensor = f.get_tensor(ckpt_key)
                    if ckpt_key not in params_dict:
                        # Try to unpack experts if they are packed
                        if ".experts.gate_up_proj" in ckpt_key:
                            base_key, suffix = ckpt_key.split(".experts.gate_up_proj", 1)
                            num_experts = tensor.shape[0]
                            for expert_idx in range(num_experts):
                                gate_up_weight = tensor[expert_idx]
                                gate_weight, up_weight = gate_up_weight.chunk(2, dim=0)

                                gate_key = f"{base_key}.experts.{expert_idx}.gate_proj{suffix}.weight"
                                up_key = f"{base_key}.experts.{expert_idx}.up_proj{suffix}.weight"

                                if gate_key in params_dict:
                                    params_dict[gate_key].data.copy_(gate_weight)
                                    loaded.add(gate_key)
                                if up_key in params_dict:
                                    params_dict[up_key].data.copy_(up_weight)
                                    loaded.add(up_key)
                            continue

                        if ".experts.down_proj" in ckpt_key:
                            base_key, suffix = ckpt_key.split(".experts.down_proj", 1)
                            num_experts = tensor.shape[0]
                            for expert_idx in range(num_experts):
                                down_weight = tensor[expert_idx]
                                down_key = f"{base_key}.experts.{expert_idx}.down_proj{suffix}.weight"

                                if down_key in params_dict:
                                    params_dict[down_key].data.copy_(down_weight)
                                    loaded.add(down_key)
                            continue

                        raise InvalidModelError(
                            f"Unexpected MTP checkpoint key: {ckpt_key}",
                            action="Please verify MTP model definition matches checkpoint parameter names.",
                        )
                    if params_dict[ckpt_key].shape != tensor.shape:
                        raise InvalidModelError(
                            (
                                f"Shape mismatch for MTP param {ckpt_key}: "
                                f"expected {params_dict[ckpt_key].shape}, got {tensor.shape}"
                            ),
                            action="Please verify MTP config and checkpoint are from the same model variant.",
                        )
                    params_dict[ckpt_key].data.copy_(tensor)
                    loaded.add(ckpt_key)

        # Ensure every MTP parameter in model is loaded.
        missing_params = set(params_dict.keys()) - loaded
        if missing_params:
            preview = ", ".join(sorted(missing_params)[:10])
            remain = len(missing_params) - min(10, len(missing_params))
            if remain > 0:
                preview = f"{preview}, ... (+{remain} more)"
            raise InvalidModelError(
                f"Found {len(missing_params)} unloaded MTP parameters: {preview}",
                action="Please ensure checkpoint contains complete mtp.* parameter set.",
            )

        predictor.eval()
        get_logger().info("MTP predictor loaded with %d parameters", len(loaded))
        return predictor

    def _load_mtp_if_not_loaded(self, model: nn.Module) -> nn.Module:
        """
        Load MTP predictor and return its decoder layer.

        The predictor is mounted at ``model.mtp`` so mtp_preprocess can read
        MTP-specific modules via ``model.mtp.*`` without polluting decoder modules.

        Args:
            model: The main model

        Returns:
            MTP decoder layer
        """
        # Check if already loaded (cached)
        if hasattr(model, 'mtp') and model.mtp is not None:
            return model.mtp

        get_logger().info("Loading MTP layer...")
        predictor = self._load_mtp_predictor(model)

        # Mount full predictor on model for mtp_preprocess access.
        model.set_submodule('mtp', predictor)
        get_logger().info("MTP layer loaded successfully")
        return model.mtp

    def mtp_preprocess(
        self,
        model: nn.Module,
        inputs: Dict[str, Any],
        hidden_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Preprocess inputs for MTP: only the extra work not done inside
        Qwen3_5MultiTokenPredictor.forward.

        Forward already does: pre_fc_norm_embedding, pre_fc_norm_hidden,
        cat, fc, rotary_emb, layers, norm. So here we only:
        1. Backbone norm + lm_head -> logits (for last-token prediction)
        2. Shift input_ids and fill last position with predicted token
        3. embed_tokens -> inputs_embeds for MTP
        4. Build position_ids, cache_position, causal mask for MTP

        Returns (base_hidden_states, kwargs) so that MTP.forward receives
        hidden_states and inputs_embeds and performs fusion + layers itself.
        """

        def wrap_device(module: nn.Module):
            def auto_module(arg):
                module.to('npu')
                result = module(arg.to('npu'))
                module.to('cpu')
                return result

            return auto_module

        pre_hidden_states = hidden_states.to('npu')

        # 1. Norm + LM head only for next-token prediction (last position)
        normed_hidden = wrap_device(model.model.language_model.norm)(pre_hidden_states)
        logits = wrap_device(model.lm_head)(normed_hidden).float()

        # 2. Shift input_ids and set last position to predicted token
        input_ids = inputs['input_ids'].to('npu')
        input_ids_mtp = remove_zero_and_shift(input_ids)
        input_ids_mtp[:, -1] = logits[:, -1, :].argmax(dim=1)

        # 3. Embed; MTP.forward will do norm/cat/fc/rotary_emb itself
        input_embeds_mtp = wrap_device(model.model.language_model.embed_tokens)(input_ids_mtp)
        seq_len = input_embeds_mtp.shape[1]

        # 4. MTP position_ids (1..seq_len), cache_position, causal mask
        mtp_position_ids = torch.arange(1, seq_len + 1, dtype=torch.long, device=input_embeds_mtp.device)
        mtp_position_ids = mtp_position_ids.unsqueeze(0).unsqueeze(0).expand(3, input_embeds_mtp.shape[0], -1)
        mtp_cache_position = torch.arange(0, seq_len, device=input_embeds_mtp.device)
        text_position_ids = mtp_position_ids[0]
        attention_mask = inputs.get('attention_mask').to('npu')
        causal_mask = create_causal_mask(
            config=self.config.text_config,
            input_embeds=input_embeds_mtp,
            attention_mask=attention_mask,
            cache_position=mtp_cache_position,
            past_key_values=None,
            position_ids=text_position_ids,
        )

        kwargs = {
            'inputs_embeds': input_embeds_mtp,
            'attention_mask': causal_mask,
            'position_ids': mtp_position_ids,
            'cache_position': mtp_cache_position,
            'past_key_values': None,
        }
        return pre_hidden_states, kwargs
