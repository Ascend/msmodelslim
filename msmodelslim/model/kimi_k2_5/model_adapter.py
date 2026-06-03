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

from msmodelslim.core.const import DeviceType
from msmodelslim.app.naive_quantization.model_info_interface import ModelInfoInterface
from msmodelslim.core.base.protocol import ProcessRequest
from msmodelslim.core.graph import AdapterConfig, MappingConfig, FusionConfig
from msmodelslim.model.common.layer_wise_forward import (
    generated_decoder_layer_visit_func,
)
from msmodelslim.model.interface_hub import (
    IterSmoothInterface,
    FlexSmoothQuantInterface,
    ModelSlimPipelineInterfaceV1,
    LayerWiseOffloadOptionalInterface,
    AscendV1SaveInterface,
)
from msmodelslim.model.common.vlm_base import SafeGenerator, VLMBaseModelAdapter
from msmodelslim.utils.exception import UnsupportedError
from msmodelslim.utils.logging import logger_setter, get_logger
from msmodelslim.utils.security import (
    get_valid_read_path,
    json_safe_load,
    MAX_READ_FILE_SIZE_32G,
    safe_copy_file,
)

from transformers import AutoProcessor, AutoTokenizer
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
from transformers.modeling_utils import PreTrainedModel

from contextlib import contextmanager

from .convert_int4_to_bf16 import (
    auto_convert_module_int4_to_bf16,
    replace_compressed_linear_with_bf16,
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


@logger_setter()
class KimiK25ModelAdapter(  # pylint: disable=too-many-ancestors
    VLMBaseModelAdapter,
    ModelInfoInterface,
    ModelSlimPipelineInterfaceV1,
    IterSmoothInterface,
    FlexSmoothQuantInterface,
    LayerWiseOffloadOptionalInterface,
    AscendV1SaveInterface,
):
    def __init__(self, model_type: str, model_path: Path, trust_remote_code: bool = False):
        # Cache for processor (used in dataset handling)
        self._processor = None
        self._tokenizer = None
        super().__init__(model_type, model_path, trust_remote_code)

    def get_model_pedigree(self) -> str:
        """Return model pedigree for best practice matching"""
        return "kimi_k2_5"

    def get_model_type(self) -> str:
        """Return model type"""
        return self.model_type

    def get_layer_wise_offload_device(self):
        """Return preferred offload device for layer-wise runner."""
        return "meta"

    def handle_dataset(self, dataset: Any, device: DeviceType = DeviceType.NPU) -> List[Any]:
        """
        Handle multimodal Kimi2.5 calibration dataset.

        Supported sample structure:
            VlmCalibSample(text: str, image: str)  # image must be non-None

        For image+text samples, messages are constructed as:
            [{"role": "user", "content": [
                {"type": "image_url", "image_url": "<path>"},
                {"type": "text", "text": text}
            ]}]

        Returns a list of processor-ready dicts for LayerWiseRunner.
        """

        # 从模型目录动态加载 processor（权重目录自带源码时可直接使用）
        try:
            self._processor = AutoProcessor.from_pretrained(  # nosec B615
                self.model_path,
                trust_remote_code=self.trust_remote_code,
                local_files_only=True,
            )
        except Exception as e:
            get_logger().warning(
                "AutoProcessor load failed from %s, try tokenizer fallback: %s", self.model_path, str(e)
            )
            try:
                self._tokenizer = AutoTokenizer.from_pretrained(  # nosec B615
                    self.model_path,
                    trust_remote_code=self.trust_remote_code,
                    local_files_only=True,
                )
            except Exception as e_tokenizer:
                raise UnsupportedError(
                    f"Failed to load processor/tokenizer from model_path={self.model_path}.",
                    action=(
                        "Ensure model directory contains processor/tokenizer files and remote code "
                        "definitions (e.g., config.json with auto_map, processor/tokenizer py/json files). "
                        f"processor_error={e}; tokenizer_error={e_tokenizer}"
                    ),
                )

            raise UnsupportedError(
                "Kimi2.5 multimodal calibration requires AutoProcessor, but only tokenizer is available.",
                action=(
                    "Provide processor assets in model_path (processor_config.json and corresponding "
                    "remote code), or keep a compatible processor implementation in runtime environment."
                ),
            )

        # Validate modality: Kimi2.5 requires both image and text
        for item in dataset:
            if item.image is None or item.text is None:
                raise UnsupportedError(
                    "Kimi2.5 adapter requires both image and text for calibration.",
                    action="Use multimodal (image+text) data only.",
                )

        processed_data = []
        for item in tqdm(dataset, desc="Processing Kimi2.5 calibration dataset"):
            # Construct messages in Kimi2.5 expected format
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": get_valid_read_path(item.image)},
                        {"type": "text", "text": item.text},
                    ],
                }
            ]

            # End-to-end processing: tokenize + vision encoding
            inputs = self._processor(messages=messages, return_tensors="pt")  # Returns BatchFeature

            # Collect required tensors and move to device
            processed_item = self._collect_inputs_to_device(
                inputs,
                device,
                keys=[
                    "input_ids",
                    "pixel_values",
                    "grid_thws",  # Kimi2.5 uses 'grid_thws' (plural)
                    "attention_mask",
                ],
                defaults={},
            )
            processed_data.append(processed_item)

        get_logger().info("Kimi2.5 dataset preprocessing finished, samples=%d", len(processed_data))
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

        # Save original layer count
        with default_dtype(torch.bfloat16):
            origin_layers = self.config.text_config.num_hidden_layers
            get_logger().info(
                "Model with %d text layers + %d vision layers",
                origin_layers,
                self.config.vision_config.vt_num_hidden_layers,
            )
            # Temporarily set to 1 layer for initialization
            self.config.text_config.num_hidden_layers = 1
            self.config.use_cache = False  # Disable cache to save memory

            # Validate model path
            self.model_path = get_valid_read_path(str(self.model_path), is_dir=True, check_user_stat=True)

            # Load model with only 1 text decoder layer
            # Vision encoder is fully loaded, text decoder has only 1 layer
            get_logger().info("Loading vision encoder and first text decoder layer...")

            self.config._attn_implementation = "eager"
            self.config.text_config._attn_implementation = "eager"
            self.config.vision_config._attn_implementation = "eager"

            _original_initialize_missing_keys = PreTrainedModel._initialize_missing_keys
            try:
                # 避免 compressed linear 在初始化缺失键时访问 module.weight 导致 AttributeError
                PreTrainedModel._initialize_missing_keys = lambda *args, **kwargs: None
                model = SafeGenerator.get_model_from_pretrained(
                    model_path=str(self.model_path),
                    config=self.config,
                    trust_remote_code=self.trust_remote_code,
                    device_map="cpu",
                    torch_dtype="auto",
                    low_cpu_mem_usage=False,
                    attn_implementation='eager',
                )
            finally:
                PreTrainedModel._initialize_missing_keys = _original_initialize_missing_keys

            # Restore original layer count
            self.config.text_config.num_hidden_layers = origin_layers

            # Ensure _attn_implementation is set for dynamically loaded layers
            # This prevents KeyError when layers access ALL_ATTENTION_FUNCTIONS[config._attn_implementation]
            self.config.text_config._attn_implementation = "eager"
            self.config.vision_config._attn_implementation = "eager"

            # Load full state_dict for the first layer + vision encoder + lm_head
            get_logger().info("Loading weights for vision encoder, first decoder layer, and lm_head...")
            state_dict = self._get_state_dict(model)
            model.load_state_dict(state_dict, strict=False)
            auto_convert_module_int4_to_bf16("", model, str(self.model_path))

            model.vision_tower = replace_compressed_linear_with_bf16(
                model.vision_tower,
                "vision_tower",
                str(self.model_path),
            )
            if model.mm_projector is not None:
                model.mm_projector = replace_compressed_linear_with_bf16(
                    model.mm_projector,
                    "mm_projector",
                    str(self.model_path),
                )
            model.language_model.lm_head = replace_compressed_linear_with_bf16(
                model.language_model.lm_head,
                "language_model.lm_head",
                str(self.model_path),
            )

            model.eval()

            # CRITICAL: Copy text_config attention heads to model.config for OV smoothing
            # BaseSmoothProcessor._apply_standard_ov_smooth() reads from model.config, not model.config.text_config
            # This must be done AFTER model is loaded
            if hasattr(model.config.text_config, "num_attention_heads"):
                model.config.num_attention_heads = model.config.text_config.num_attention_heads
                get_logger().info("Set model.config.num_attention_heads = %d", model.config.num_attention_heads)
            if hasattr(model.config.text_config, "num_key_value_heads"):
                model.config.num_key_value_heads = model.config.text_config.num_key_value_heads
                get_logger().info("Set model.config.num_key_value_heads = %d", model.config.num_key_value_heads)

            get_logger().info(
                "Model initialized with %d layers (1 loaded, others will be loaded on-demand)", origin_layers
            )

            return model

    def generate_model_visit(self, model: nn.Module) -> Generator[ProcessRequest, Any, None]:
        """
        Generate model visit pipeline for layer-wise processing.

        Uses the common layer-wise visit function for consistent behavior.

        Processing order:
            1. Vision encoder (model.vision_tower) - processed as a whole
            2. Multimodal projector (model.mm_projector) - processed as a whole (if present)
            3. Text decoder layers (model.language_model.layers[0..N]) - loaded on-demand

        Yields:
            ProcessRequest(name, module, args, kwargs)
        """
        # 1. Process vision encoder first
        get_logger().info("Processing vision encoder...")
        yield ProcessRequest(name="vision_tower", module=model.vision_tower, args=(), kwargs={})

        # 2. Process multimodal projector (if present)
        get_logger().info("Processing mm_projector encoder...")
        yield ProcessRequest(name="mm_projector", module=model.mm_projector, args=(), kwargs={})

        # 3. Process text decoder layers one by one using standard visit function
        get_logger().info("Processing text decoder layers...")
        yield from generated_decoder_layer_visit_func(model, transformer_blocks=self.generate_decoder_layer(model))

    def generate_model_forward(self, model: nn.Module, inputs: Any) -> Generator[ProcessRequest, Any, None]:
        """
        Generate forward pipeline for Kimi2.5 (Llava-style) multimodal model calibration.

        This function replicates the exact fusion logic in Kimi2.5's forward:
        - Extract image features via vision_tower
        - Project via mm_projector (if exists)
        - Merge into text embeddings using _merge_input_ids_with_image_features
        - Run language model layers with correct attention_mask & position_ids

        Assumptions for calibration:
        - No labels (labels=None)
        - No past_key_values (prefill-only)
        - Input is a single sample or first of batch

        Args:
            model: Full Kimi2.5 model (with vision_tower, mm_projector, language_model)
            inputs: Dict containing "input_ids", "pixel_values", "grid_thws", "attention_mask"

        Yields:
            ProcessRequest for vision_tower, mm_projector, and each decoder layer.
        """
        get_logger().debug("Start Kimi2.5 multimodal forward pipeline")
        # 提取首个样本作为校准输入（与现有计算逻辑保持一致）
        if isinstance(inputs, list):
            sample = inputs[0]
        else:
            sample = inputs

        # 阶段1：准备视觉与文本输入
        input_ids = sample["input_ids"]
        attention_mask = sample.get("attention_mask", None)
        pixel_values = sample.get("pixel_values", None)
        grid_thws = sample.get("grid_thws", None)

        inputs_embeds = model.language_model.model.embed_tokens(input_ids)

        # 保持视觉输入与视觉塔权重 dtype 一致，避免前向中出现 dtype 不匹配
        if pixel_values is not None:
            pixel_values = pixel_values.to(model.vision_tower.patch_embed.proj.weight.dtype)

        get_logger().debug(
            "Prepared forward inputs: input_ids=%s, pixel_values=%s, grid_thws=%s, attention_mask=%s",
            tuple(input_ids.shape) if input_ids is not None else None,
            tuple(pixel_values.shape) if pixel_values is not None else None,
            tuple(grid_thws.shape) if isinstance(grid_thws, torch.Tensor) else None,
            tuple(attention_mask.shape) if attention_mask is not None else None,
        )

        if pixel_values is None or len(pixel_values) == 0 or input_ids.shape[1] == 1:
            position_ids = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0).expand_as(input_ids)
            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids)

        else:
            image_features = yield ProcessRequest(
                name="vision_tower",
                module=model.vision_tower,
                args=(pixel_values,),
                kwargs={"grid_thws": grid_thws} if grid_thws is not None else {},
            )

            # --- 3. Optional multimodal projector ---
            if model.mm_projector is not None:
                image_features = yield ProcessRequest(
                    name="mm_projector",
                    module=model.mm_projector,
                    args=(image_features,),
                    kwargs={},
                )

            get_logger().debug("Image features projected by mm_projector")
            # 对齐 dtype，保持后续融合与解码计算一致
            inputs_embeds = inputs_embeds.to(image_features[0].dtype)

            merged_result = model._merge_input_ids_with_image_features(
                image_features=image_features,
                inputs_embeds=inputs_embeds,
                input_ids=input_ids,
                attention_mask=attention_mask if attention_mask is not None else torch.ones_like(input_ids),
                labels=None,
            )
            inputs_embeds, attention_mask, _, position_ids = merged_result

        get_logger().debug("Building 4D causal attention mask")
        attention_mask_4d = _prepare_4d_causal_attention_mask(
            attention_mask,
            (inputs_embeds.shape[0], inputs_embeds.shape[1]),
            inputs_embeds,
            past_key_values_length=0,
        )
        get_logger().debug("4D causal attention mask ready")

        hidden_states = inputs_embeds
        get_logger().debug("Start layer-wise decoding")
        for name, layer in self.generate_decoder_layer(model):
            get_logger().debug("Processing decoder layer: %s", name)
            hidden_states = yield ProcessRequest(
                name=name,
                module=layer,
                args=(hidden_states,),
                kwargs={
                    "attention_mask": attention_mask_4d,
                    "position_ids": position_ids,
                    "past_key_value": None,
                    "output_attentions": False,
                    "use_cache": False,
                },
            )
            if isinstance(hidden_states, tuple):
                hidden_states = hidden_states[0]

    def generate_decoder_layer(self, model: nn.Module) -> Generator[Tuple[str, nn.Module], None, None]:
        """
        Generate Kimi2.5 text decoder layers, loading them on-demand.

        This follows the layer-wise loading strategy used in this adapter:
        each `language_model.model.layers.{idx}` block is materialized only when visited.

        Yields:
            (layer_name, layer_module) tuples
        """
        num_layers = self.config.text_config.num_hidden_layers

        for layer_idx in range(num_layers):
            name = f"language_model.model.layers.{layer_idx}"
            # Load layer if not exists
            layer = self._load_decoder_if_not_exist(model, name, layer_idx)

            yield name, layer

    def enable_kv_cache(self, model: nn.Module, need_kv_cache: bool) -> None:
        """
        Enable/disable KV cache.

        For calibration, we typically don't need KV cache.
        """
        model.config.use_cache = need_kv_cache
        get_logger().info("KV cache %s", "enabled" if need_kv_cache else "disabled")

    def get_adapter_config_for_subgraph(self) -> List[AdapterConfig]:
        adapter_config = []
        # mtp layer does not apply smooth due to the compatible with pre-refactor
        for layer_idx in range(self.config.text_config.num_hidden_layers - 1):
            # OKV_b融合的映射配置：o_proj -> kv_b_proj
            okv_b_mapping_config = MappingConfig(
                # KV_b投影层
                source=f"language_model.model.layers.{layer_idx}.self_attn.kv_b_proj",
                # 输出投影层
                targets=[f"language_model.model.layers.{layer_idx}.self_attn.o_proj"],
            )

            # Norm-Linear融合的映射配置1：q_a_proj, kv_a_proj_with_mqa -> input_layernorm
            norm_linear_mapping_config1 = MappingConfig(
                # 第一个LayerNorm
                source=f"language_model.model.layers.{layer_idx}.input_layernorm",
                targets=[
                    f"language_model.model.layers.{layer_idx}.self_attn.q_a_proj",
                    f"language_model.model.layers.{layer_idx}.self_attn.kv_a_proj_with_mqa",
                ],  # 注意力层的Q_a,KV_a投影
            )

            # Norm-Linear融合的映射配置2：q_b_proj -> q_a_layernorm
            norm_linear_mapping_config2 = MappingConfig(
                # q_a_layernorm
                source=f"language_model.model.layers.{layer_idx}.self_attn.q_a_layernorm",
                # q_b投影
                targets=[f"language_model.model.layers.{layer_idx}.self_attn.q_b_proj"],
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
                            num_attention_heads=self.config.text_config.num_attention_heads,
                            num_key_value_heads=self.config.text_config.num_key_value_heads,
                            custom_config={
                                'qk_nope_head_dim': self.config.text_config.qk_nope_head_dim,
                                'v_head_dim': self.config.text_config.v_head_dim,
                            },
                        ),
                    ),
                    AdapterConfig(subgraph_type="norm-linear", mapping=norm_linear_mapping_config1),
                    AdapterConfig(subgraph_type="norm-linear", mapping=norm_linear_mapping_config2),
                ]
            )

        return adapter_config

    def ascendv1_save_postprocess(self, model: nn.Module, save_directory: str) -> None:
        """
        导出件后处理：复制 tiktoken.model 文件到保存路径
        @param model: 量化模型（未使用）
        @param save_directory: 量化模型导出目录
        """
        tiktoken_file = os.path.join(self.model_path, "tiktoken.model")
        if os.path.isfile(tiktoken_file):
            dest_file = os.path.join(save_directory, "tiktoken.model")
            safe_copy_file(src_path=tiktoken_file, dest_path=dest_file)
            os.chmod(dest_file, int("600", 8))

    @lru_cache(maxsize=1)
    def _get_weight_map(self) -> Dict[str, str]:
        """Get weight map from model.safetensors.index.json"""
        index_path = os.path.join(self.model_path, "model.safetensors.index.json")
        index_data = json_safe_load(index_path)
        return index_data["weight_map"]

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
            file_path = get_valid_read_path(file_path, extensions="safetensors", size_max=MAX_READ_FILE_SIZE_32G)

            with safe_open(file_path, framework="pt", device="cpu") as f:
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
        4. Returns the loaded (and potentially converted) layer

        Args:
            model: The model
            name: Full layer name (e.g., "model.language_model.layers.0")
            idx: Layer index

        Returns:
            Loaded decoder layer module
        """
        try:
            # Try to access the layer
            decoder = model.get_submodule(name)
            # Check if it's actually loaded (not on meta device)
            try:
                _ = decoder.input_layernorm.weight.device
                # If we can access the device, layer is loaded
                get_logger().debug("Layer %d already loaded", idx)
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
        with patch.object(nn.Linear, "reset_parameters", lambda _self: None), default_dtype(torch.bfloat16):
            get_logger().info("Creating decoder layer %d structure...", idx)

            # Create layer structure (weights will be on meta or uninitialized)
            layer_cls = None
            module_list: nn.ModuleList = model.language_model.model.layers
            if len(module_list) > 0 and module_list[0] is not None:
                layer_cls = module_list[0].__class__
            if layer_cls is None:
                raise UnsupportedError(
                    "Failed to infer decoder layer class from loaded model.",
                    action=(
                        "Ensure model_path contains complete remote model code and "
                        "the first decoder layer can be initialized via from_pretrained."
                    ),
                )
            decoder = layer_cls(self.config.text_config, layer_idx=idx)

            # Load weights from safetensors
            state_dict = self._get_state_dict(decoder, prefix=name)
            decoder.load_state_dict(state_dict, strict=False)
            auto_convert_module_int4_to_bf16(name, decoder, str(self.model_path))
            decoder.eval()

            # Add layer to model's layer list
            module_list: nn.ModuleList = model.language_model.model.layers
            if len(module_list) <= idx:
                module_list.append(decoder)
            else:
                module_list[idx] = decoder

            get_logger().info("Decoder layer %d loaded successfully", idx)

        # Perform architecture adaptation if needed
        # Similar to DeepSeek-V3's MTP layer wrapping in load_mtp_if_not_load
        return decoder
