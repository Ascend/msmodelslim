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
from transformers import AutoProcessor
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLDecoderLayer
from qwen_vl_utils import process_vision_info

from msmodelslim.core.const import DeviceType
from msmodelslim.app.naive_quantization.model_info_interface import ModelInfoInterface
from msmodelslim.core.base.protocol import ProcessRequest
from msmodelslim.core.graph import AdapterConfig, MappingConfig
from msmodelslim.model.common.layer_wise_forward import generated_decoder_layer_visit_func
from msmodelslim.model.interface_hub import (
    IterSmoothInterface,
    FlexSmoothQuantInterface,
    ModelSlimPipelineInterfaceV1,
    QuaRotInterface,
)
from msmodelslim.model.common.vlm_base import VLMBaseModelAdapter
from msmodelslim.utils.exception import InvalidModelError, UnsupportedError
from msmodelslim.utils.logging import logger_setter, get_logger
from msmodelslim.utils.security import get_valid_read_path, json_safe_load, MAX_READ_FILE_SIZE_32G


@logger_setter("msmodelslim.model.qwen2_5_vl")
class Qwen25VLModelAdapter(  # pylint: disable=too-many-ancestors
    VLMBaseModelAdapter,
    ModelInfoInterface,
    ModelSlimPipelineInterfaceV1,
    FlexSmoothQuantInterface,
    IterSmoothInterface,
    QuaRotInterface,
):
    def __init__(self, model_type: str, model_path: Path, trust_remote_code: bool = False):
        # Cache for processor (used in dataset handling)
        self._processor = None
        self._tokenizer = None
        super().__init__(model_type, model_path, trust_remote_code)

    def get_model_pedigree(self) -> str:
        """Return model pedigree for best practice matching"""
        return 'qwen2_5_vl'

    def get_model_type(self) -> str:
        """Return model type"""
        return self.model_type

    def handle_dataset(self, dataset: Any, device: DeviceType = DeviceType.NPU) -> List[Any]:
        """
        Handle multimodal VLM calibration dataset.

        Supported sample structure (preferred):
            VlmCalibSample(text: str, image: Optional[str])

        For text-only samples, messages contain only text:
            [{"role": "user", "content": [{"type": "text", "text": text}]}]
        For image+text samples:
            [{"role": "user", "content": [
                {"type": "image", "image": "<path>"},
                {"type": "text", "text": text}
            ]}]

        Returns a list of processor-ready dicts for LayerWiseRunner.
        """
        self._processor = AutoProcessor.from_pretrained(self.model_path)  # nosec B615

        # Validate dataset modality: Qwen2.5-VL adapter expects image+text only (no pure-text or mixed-without-image)
        for item in dataset:
            image_path = item.image
            text = item.text
            if image_path is None or text is None:
                raise UnsupportedError(
                    ("Qwen2.5-VL adapter currently requires both image and text for calibration."),
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
            image_path = get_valid_read_path(image_path)
            content = [
                {"type": "image", "image": str(image_path)},
                {"type": "text", "text": text},
            ]

            messages = [{"role": "user", "content": content}]

            text = self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

            image_inputs, video_inputs = process_vision_info(messages)

            inputs = self._processor(
                text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt"
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

        try:
            from transformers import Qwen2_5_VLForConditionalGeneration
        except ImportError as e:
            raise InvalidModelError(
                "Failed to import Qwen2_5_VLForConditionalGeneration. "
                "Please install transformers with Qwen2.5-VL support.",
                action="pip install transformers==4.49.0",
            ) from e

        get_logger().info("Initializing Qwen2.5-VL model with v1 framework (layer-wise loading)...")
        # Save original layer count
        origin_layers = self.config.num_hidden_layers
        get_logger().info(
            "Model with %s text layers + %s vision layers", origin_layers, self.config.vision_config.depth
        )

        # Temporarily set to 1 layer for initialization
        self.config.num_hidden_layers = 1
        self.config.use_cache = False  # Disable cache to save memory

        # Validate model path
        self.model_path = get_valid_read_path(str(self.model_path), is_dir=True, check_user_stat=True)

        # Load model with only 1 text decoder layer
        # Vision encoder is fully loaded, text decoder has only 1 layer
        get_logger().info("Loading vision encoder and first text decoder layer...")
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(  # nosec B615
            self.model_path,
            config=self.config,
            trust_remote_code=self.trust_remote_code,
            torch_dtype="auto",
            local_files_only=True,
            device_map="cpu",  # All on CPU for now
            attn_implementation='eager',  # Required: prevents KeyError when accessing ALL_ATTENTION_FUNCTIONS
        ).eval()

        # Restore original layer count
        self.config.num_hidden_layers = origin_layers

        # Ensure _attn_implementation is set for dynamically loaded layers
        # This prevents KeyError when layers access ALL_ATTENTION_FUNCTIONS[config._attn_implementation]
        self.config._attn_implementation = 'eager'

        # Load full state_dict for the first layer + vision encoder + lm_head
        get_logger().info("Loading weights for vision encoder, first decoder layer, and lm_head...")
        state_dict = self._get_state_dict(model)
        model.load_state_dict(state_dict)

        get_logger().info("Model initialized with %s layers (1 loaded, others will be loaded on-demand)", origin_layers)

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
        yield ProcessRequest(name="visual", module=model.visual, args=(), kwargs={})

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
        # # For multimodal models, forward is more complex
        # # We need to handle the vision-language fusion

        # 1. Extract first sample for calibration
        if isinstance(inputs, list):
            sample = inputs[0]
        else:
            sample = inputs

        # 2. Vision encoder forward
        pixel_values = sample['pixel_values']
        image_grid_thw = sample['image_grid_thw']

        # Yield vision encoder result
        image_embeds = yield ProcessRequest(
            name="visual", module=model.visual, args=(pixel_values, image_grid_thw), kwargs={}
        )

        # 3. Prepare inputs for text decoder
        input_ids = sample['input_ids']
        attention_mask = sample['attention_mask']

        # Get input embeddings
        inputs_embeds = model.model.embed_tokens(input_ids)

        # Get image token mask for fusion
        image_mask = (input_ids == model.config.image_token_id).unsqueeze(-1).expand_as(inputs_embeds)
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        # Prepare deepstack visual injection
        # visual_pos_masks: 1D mask indicating which positions have visual tokens
        # deepstack_visual_embeds: list of visual features for each deepstack layer
        _ = image_mask[..., 0]  # Shape: (batch, seq_len)
        # deepstack_image_embeds is already a list of tensors, one per layer

        # Get cache_position for attention mask creation
        cache_position = torch.arange(0, inputs_embeds.shape[1], device=inputs_embeds.device)

        # Get position ids
        position_ids, rope_deltas = model.get_rope_index(
            input_ids=input_ids, image_grid_thw=image_grid_thw, attention_mask=attention_mask
        )

        # Expand position_ids if needed (3D format for mROPE)
        if position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)
        from transformers.cache_utils import DynamicCache

        past_key_values = DynamicCache()
        output_attentions = self.config.output_attentions

        causal_mask = model.model._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )

        # create position embeddings to be shared across the decoder layers
        position_embeddings = model.model.rotary_emb(inputs_embeds, position_ids)

        # 4. Process each decoder layer
        hidden_states = inputs_embeds
        for layer_idx, (name, layer) in enumerate(self.generate_decoder_layer(model)):
            # Yield layer result
            layer_outputs = yield ProcessRequest(
                name=name,
                module=layer,
                args=(hidden_states,),
                kwargs={
                    'attention_mask': causal_mask,
                    'position_ids': position_ids,
                    'cache_position': cache_position,
                    'position_embeddings': position_embeddings,
                    'past_key_value': past_key_values,
                    'use_cache': False,
                },
            )
            hidden_states = (
                layer_outputs[0] if isinstance(layer_outputs, tuple) and len(layer_outputs) >= 1 else layer_outputs
            )

    def generate_decoder_layer(self, model: nn.Module) -> Generator[Tuple[str, nn.Module], None, None]:
        """
        Generate decoder layers, loading them on-demand.

        Yields:
            (layer_name, layer_module) tuples
        """
        num_layers = self.config.num_hidden_layers

        for layer_idx in range(num_layers):
            name = f"model.layers.{layer_idx}"

            # Load layer if not exists
            layer = self._load_decoder_if_not_exist(model, name, layer_idx)

            yield name, layer

    def enable_kv_cache(self, model: nn.Module, need_kv_cache: bool) -> None:
        """
        Enable/disable KV cache.

        For calibration, we typically don't need KV cache.
        """
        model.config.use_cache = need_kv_cache
        get_logger().info("KV cache %s", 'enabled' if need_kv_cache else 'disabled')

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
            file_path = get_valid_read_path(file_path, extensions='safetensors', size_max=MAX_READ_FILE_SIZE_32G)

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
        try:
            # Try to access the layer
            decoder = model.get_submodule(name)
            # Check if it's actually loaded (not on meta device)
            try:
                _ = decoder.input_layernorm.weight.device
                # If we can access the device, layer is loaded
                get_logger().debug("Layer %s already loaded", idx)
                return decoder
            except RuntimeError:
                # Weight is on meta device, need to load
                pass
        except AttributeError:
            # Layer doesn't exist in the module list yet
            pass

        get_logger().info("Loading decoder layer %s...", idx)

        # Disable reset_parameters to avoid slow and unnecessary initialization
        # We will load weights from safetensors immediately after
        with patch.object(nn.Linear, 'reset_parameters', lambda _self: None):
            get_logger().info('Creating decoder layer %s structure...', idx)
            # Create layer structure (weights will be on meta or uninitialized)
            decoder = Qwen2_5_VLDecoderLayer(self.config, layer_idx=idx)

            # Load weights from safetensors
            state_dict = self._get_state_dict(decoder, prefix=name)
            decoder.load_state_dict(state_dict)
            decoder.eval()

            # Add layer to model's layer list
            module_list: nn.ModuleList = model.model.layers
            if len(module_list) <= idx:
                module_list.append(decoder)
            else:
                module_list[idx] = decoder

            get_logger().info('Decoder layer %s loaded successfully', idx)

        return decoder

    def get_adapter_config_for_subgraph(self) -> List[AdapterConfig]:
        adapter_config = []
        for layer_idx in range(self.config.num_hidden_layers):
            # Norm-Linear的映射配置1：输入层归一化到QKV投影
            input_layernorm_linear_mapping_config = MappingConfig(
                source=f"model.layers.{layer_idx}.input_layernorm",  # 第一个LayerNorm
                targets=[
                    f"model.layers.{layer_idx}.self_attn.k_proj",
                    f"model.layers.{layer_idx}.self_attn.q_proj",
                    f"model.layers.{layer_idx}.self_attn.v_proj",
                ],  # 注意力层的QKV投影
            )

            # Norm-Linear的映射配置2：后注意力层归一化到MLP投影
            post_layernorm_linear_mapping_config = MappingConfig(
                source=f"model.layers.{layer_idx}.post_attention_layernorm",  # 第二个LayerNorm
                targets=[
                    f"model.layers.{layer_idx}.mlp.gate_proj",
                    f"model.layers.{layer_idx}.mlp.up_proj",
                ],  # MLP层的门控和上投影
            )

            # Up-Down的映射配置
            up_down_mapping_config = MappingConfig(
                source=f"model.layers.{layer_idx}.mlp.up_proj",  # 上投影层
                targets=[f"model.layers.{layer_idx}.mlp.down_proj"],  # 下投影层
            )

            # 为当前layer添加3个配置
            adapter_config.extend(
                [
                    AdapterConfig(subgraph_type="norm-linear", mapping=input_layernorm_linear_mapping_config),
                    AdapterConfig(subgraph_type="norm-linear", mapping=post_layernorm_linear_mapping_config),
                    AdapterConfig(subgraph_type="up-down", mapping=up_down_mapping_config),
                ]
            )
        return adapter_config

    def get_ln_fuse_map(self):
        """
        Get LayerNorm-Linear fusion mapping for QuaRot.

        For Qwen2.5-VL, LayerNorm (RMSNorm) fusion is applied to text decoder layers:
        - input_layernorm → QKV projections
        - post_attention_layernorm → MLP gate_proj, up_proj, and router
        - Final norm → lm_head

        Visual components (merger, deepstack_merger) do NOT have LayerNorm fusion
        because they use LayerNorm within the vision encoder (before the output projection).
        The visual output projections are directly rotated (left rotation with R^T).

        Returns:
            Tuple of (pre_run_fused_ln, fused_map)
            - pre_run_fused_ln: Empty dict (no pre-run fusion)
            - fused_map: Dict mapping RMSNorm names to Linear layer names
        """
        return {}, _qwen2_5_vl_get_ln_fuse_map(self.config)

    def get_bake_names(self):
        """
        Get bake mean names for QuaRot.

        Qwen2.5-VL uses RMSNorm, so no bake mean is needed.

        Returns:
            Tuple of (pre_run_bake_names, bake_names)
        """
        return [], []

    def get_rotate_map(self, block_size):
        """
        Get rotation mapping for QuaRot on Qwen2.5-VL.

        Strategy:
        ---------
        We rotate the entire residual stream (text + fused visual features) using a global
        rotation matrix R. This requires careful handling of:

        1. Text embeddings: Right rotation (W_new = W_old @ R)
           - Output will be in rotated space

        2. Visual outputs: Left rotation (W_new = R^T @ W_old)
           - Ensures visual features enter the same rotated space as text
           - Includes: merger.linear_fc2 and deepstack_merger_list[*].linear_fc2

        3. Text decoder layers: Mixed rotations
           - Attention: QKV inputs rotated, O output rotated
           - MLP: Inputs rotated, outputs rotated
           - Maintains mathematical equivalence despite rotation

        Note on Visual Rotation:
        ------------------------
        Visual features are fused into the residual stream via masked_scatter.
        To maintain equivalence after fusion, visual outputs MUST be pre-rotated
        using left rotation (R^T @ W) before entering the stream.

        Args:
            block_size: Block size for Hadamard rotation

        Returns:
            Tuple of (pre_run_pairs, rotate_pairs):
            - pre_run_pairs: List containing embedding + visual rotations
            - rotate_pairs: List of RotatePairs for decoder layers
        """
        # Pass full config (includes vision_config) for visual projections
        pre_run, rot_pairs = _qwen2_5_vl_get_rotate_map(self.config, block_size)

        return [pre_run], list(rot_pairs.values())


def _qwen2_5_vl_get_ln_fuse_map(config):
    """
    Get LayerNorm-Linear fusion mapping for Qwen2.5-VL text decoder.

    Args:
        config: Text config (config.text_config)

    Returns:
        Dict mapping LayerNorm names to Linear layer names
    """
    ln_linear_map = {}

    for layer_idx in range(config.num_hidden_layers):
        # Attention: input_layernorm → QKV projections
        ln_linear_map[f"model.layers.{layer_idx}.input_layernorm"] = [
            f"model.layers.{layer_idx}.self_attn.q_proj",
            f"model.layers.{layer_idx}.self_attn.k_proj",
            f"model.layers.{layer_idx}.self_attn.v_proj",
        ]

        ln_linear_map[f"model.layers.{layer_idx}.post_attention_layernorm"] = [
            f"model.layers.{layer_idx}.mlp.{proj}" for proj in ["gate_proj", "up_proj"]
        ]

    # Final norm → lm_head
    ln_linear_map["model.norm"] = ["lm_head"]

    return ln_linear_map


def _qwen2_5_vl_get_rotate_map(config, block_size):
    """
    Get rotation mapping for Qwen2.5-VL model (text decoder + visual projections).

    Mathematical Framework:
    ----------------------
    We apply a global rotation R to the residual stream (hidden states).
    - Text embeddings: W_new = W_old @ R (right rotation)
    - Visual outputs: W_new = R^T @ W_old (left rotation)
      Reason: For nn.Linear with weight [out, in] and forward y = x @ W.T,
              to achieve y' = y @ R, we need W'^T = W.T @ R, i.e., W' = R^T @ W
    - Text layers: Various combinations of left/right rotations to maintain equivalence

    Components:
    -----------
    - Text embedding rotation (model.embed_tokens): right_rot
    - Visual output projection rotation (model.visual.merger.linear_fc2): left_rot
    - Deepstack visual projections (deepstack_merger_list[*].linear_fc2): left_rot
    - Text decoder layer rotations (QKV, O, etc.): mixed

    Args:
        config: Full model config (Qwen2_5_VLConfig, contains text_config and vision_config)
        block_size: Block size for rotation

    Returns:
        Tuple of (pre_run, rot_pairs) where:
        - pre_run: RotatePair for embedding and visual outputs (applied in pre-run phase)
        - rot_pairs: Dict of RotatePairs for decoder layers (applied in main phase)
    """
    # Create rotation matrices
    rot = QuaRotInterface.get_rotate_command(
        size=config.hidden_size,
        mode=QuaRotInterface.QuaRotMode.HADAMARD,
        block_size=block_size,
    )
    head_dim = config.hidden_size // config.num_attention_heads
    rot_uv = QuaRotInterface.get_rotate_command(
        size=head_dim,
        mode=QuaRotInterface.QuaRotMode.BLOCK_HADAMARD_SHIFTED,
        block_size=block_size,
    )

    # Pre-run phase: Rotate embedding layer AND visual output projections
    left_rot = {}
    right_rot = {}

    # 1. Rotate text embedding layer (output will be rotated)
    right_rot["model.embed_tokens"] = rot

    # 2. Rotate visual merger output projection (to match rotated text embeddings)
    left_rot["visual.merger.mlp.2"] = rot

    # 3. Rotate deepstack visual merger output projections
    pre_run = QuaRotInterface.RotatePair(left_rot=left_rot, right_rot=right_rot)
    rot_pairs = {}
    left_rot = {}
    right_rot = {}
    right_rot["lm_head"] = rot

    for layer_idx in range(config.num_hidden_layers):
        right_rot[f"model.layers.{layer_idx}.self_attn.q_proj"] = rot
        right_rot[f"model.layers.{layer_idx}.self_attn.k_proj"] = rot
        right_rot[f"model.layers.{layer_idx}.self_attn.v_proj"] = rot
        left_rot[f"model.layers.{layer_idx}.self_attn.o_proj"] = rot
        right_rot[f"model.layers.{layer_idx}.mlp.gate_proj"] = rot
        right_rot[f"model.layers.{layer_idx}.mlp.up_proj"] = rot
        left_rot[f"model.layers.{layer_idx}.mlp.down_proj"] = rot

    rot_pairs["rot"] = QuaRotInterface.RotatePair(left_rot=left_rot, right_rot=right_rot)

    # OV special rotation
    left_rot_uv = {}
    right_rot_uv = {}
    for layer_idx in range(config.num_hidden_layers):
        left_rot_uv[f"model.layers.{layer_idx}.self_attn.v_proj"] = rot_uv
        right_rot_uv[f"model.layers.{layer_idx}.self_attn.o_proj"] = rot_uv

    rot_pairs["rot_uv"] = QuaRotInterface.RotatePair(left_rot=left_rot_uv, right_rot=right_rot_uv)

    return pre_run, rot_pairs
