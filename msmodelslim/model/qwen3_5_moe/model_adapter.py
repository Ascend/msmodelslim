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

# pylint: disable=too-many-lines

import json
import os
from collections import defaultdict
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from typing import List, Any, Generator, Tuple, Dict, Optional, Callable
from unittest.mock import patch
import types

import torch
from safetensors import safe_open
from torch import nn
from tqdm import tqdm
from transformers import AutoProcessor  # pylint: disable=no-name-in-module
from transformers.masking_utils import create_causal_mask  # pylint: disable=no-name-in-module

from transformers import (  # pylint: disable=no-name-in-module
    Qwen3_5MoeForConditionalGeneration,
    Qwen3_5ForConditionalGeneration,
)
from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm  # pylint: disable=no-name-in-module
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5RMSNorm  # pylint: disable=no-name-in-module
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (  # pylint: disable=no-name-in-module
    Qwen3_5MoeRMSNorm,
)
from msmodelslim.core.const import DeviceType
from msmodelslim.app.naive_quantization.model_info_interface import ModelInfoInterface
from msmodelslim.core.base.protocol import ProcessRequest
from msmodelslim.core.graph import AdapterConfig, MappingConfig
from msmodelslim.model.common.layer_wise_forward import generated_decoder_layer_visit_func
from msmodelslim.processor.quarot import QuaRotInterface
from msmodelslim.model.interface_hub import (
    IterSmoothInterface,
    FlexSmoothQuantInterface,
    ModelSlimPipelineInterfaceV0,
    ModelSlimPipelineInterfaceV1,
    LayerWiseOffloadOptionalInterface,
    AscendV1SaveInterface,
    FA3QuantAdapterInterface,
    FA3QuantPlaceHolder,
)
from msmodelslim.model.common.vlm_base import VLMBaseModelAdapter
from msmodelslim.infra.dataset_loader.vlm_dataset_loader import VlmCalibSample
from msmodelslim.core.infer_engine.interface import FakeQuantInferenceInterface
from msmodelslim.utils.exception import InvalidModelError
from msmodelslim.utils.logging import logger_setter, get_logger
from msmodelslim.utils.security import get_valid_read_path, json_safe_load, MAX_READ_FILE_SIZE_512G

from .moe_utils import Qwen3_5MoeSparseMoeBlockWithMLP, convert_experts_to_mlp
from .modeling_qwen3_5_mtp import Qwen3_5MultiTokenPredictor
from .conv1d_dtype_patch import ensure_conv1d_forward_patched


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
    VLMBaseModelAdapter,
    ModelInfoInterface,
    ModelSlimPipelineInterfaceV0,
    ModelSlimPipelineInterfaceV1,
    IterSmoothInterface,
    FlexSmoothQuantInterface,
    FA3QuantAdapterInterface,
    QuaRotInterface,
    LayerWiseOffloadOptionalInterface,
    AscendV1SaveInterface,
    FakeQuantInferenceInterface,
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
        ensure_conv1d_forward_patched()
        super().__init__(model_type, model_path, trust_remote_code)
        self._processor = AutoProcessor.from_pretrained(  # nosec B615
            self.model_path, trust_remote_code=self.trust_remote_code, local_files_only=True
        )
        self._tokenizer = getattr(self._processor, "tokenizer", None)
        get_logger().info(
            "Loaded VLM processor=%s tokenizer=%s",
            type(self._processor).__name__,
            type(self._tokenizer).__name__,
        )

    @property
    def tokenizer(self):
        """Prefill / infer_loop decode via getattr(adapter, 'tokenizer')."""
        return self._tokenizer

    @staticmethod
    def _convert_qwen3_5_norm_to_standard(model: nn.Module) -> None:
        """Convert Qwen3.5 RMSNorm(1+weight) to standard RMSNorm(weight) for exact LN fusion."""
        for name, module in list(model.named_modules()):
            if module.__class__.__name__ in {"Qwen3_5RMSNorm", "Qwen3_5MoeRMSNorm"}:
                eps = getattr(module, "eps", None)
                if eps is None:
                    continue
                new_module = Qwen3RMSNorm(module.weight.shape[0], eps)
                new_module.weight.data = module.weight.data + 1
                model.set_submodule(name, new_module)

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

    @staticmethod
    def _non_empty_path(path: Any) -> bool:
        return path is not None and bool(str(path).strip())

    def get_layer_wise_offload_device(self):
        """Return preferred offload device for layer-wise runner."""
        return "meta"

    def handle_dataset(self, dataset: Any, device: DeviceType = DeviceType.NPU) -> List[Any]:
        """
        Handle multimodal VLM calibration dataset.

        Supported sample structure (preferred):
            VlmCalibSample(text: str, image: Optional[str])

        Pure-text and image+text samples are supported within one homogeneous task.

        Returns a list of processor-ready dicts for LayerWiseRunner.
        """
        self._processor = AutoProcessor.from_pretrained(  # nosec B615
            self.model_path, trust_remote_code=self.trust_remote_code, local_files_only=True
        )

        has_image = False
        processed_data = []
        for item in tqdm(dataset, desc="Processing calibration dataset"):
            is_dataclass = isinstance(item, VlmCalibSample)
            text = item.text if is_dataclass else item.get('text')
            image_path = item.image if is_dataclass else item.get('image')

            if self._non_empty_path(image_path):
                has_image = True
                image_path = get_valid_read_path(image_path)
                content = [
                    {"type": "image", "image": str(image_path)},
                    {"type": "text", "text": text},
                ]
            else:
                # transformers>=5.2 apply_chat_template expects list-of-parts, not a bare string
                content = [{"type": "text", "text": text}]

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
                    # Required by Qwen3_5Model.get_rope_index (text=0 / image=1 / video=2 grouping).
                    # Returned by the processor next to input_ids.
                    'mm_token_type_ids',
                    'cache_position',
                    'logits_to_keep',
                ],
                defaults={'logits_to_keep': 1},
            )

            processed_data.append(processed_item)

        get_logger().info(
            "Processed %d calibration samples (image=%s)",
            len(processed_data),
            has_image,
        )
        return processed_data

    def handle_dataset_by_batch(self, dataset: Any, batch_size: int, device: DeviceType = DeviceType.NPU) -> List[Any]:
        """V0 anti-outlier path; batching is not required for image calib."""
        _ = batch_size
        return self.handle_dataset(dataset=dataset, device=device)

    def uses_image_calib(self) -> bool:
        return True

    def modelslim_v0_save_postprocess(self, save_directory: str) -> None:
        """Patch saved config torch_dtype for Atlas inference W8A8SC follow-up."""
        torch_dtype = "float16"
        dest = Path(save_directory)
        config_path = dest / "config.json"
        if not config_path.is_file():
            get_logger().warning("post_save_w8a8s: missing %s, skip", config_path)
            return

        with open(config_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # config may store "torch.float16" or "float16"
        dtype_name = torch_dtype.rsplit(".", maxsplit=1)[-1] if "." in torch_dtype else torch_dtype
        data["torch_dtype"] = dtype_name

        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
            f.write("\n")
        try:
            os.chmod(config_path, 0o600)
        except OSError:
            pass
        get_logger().info("post_save_w8a8s: set torch_dtype=%s in %s", dtype_name, config_path)

    def load_model(self, device: DeviceType = DeviceType.NPU) -> nn.Module:
        """
        Full-model load for modelslim_v0 W8A8S (Calibrator + is_lowbit).

        Prefers FP16 for Atlas inference products W8A8SC deployment.
        """
        get_logger().info("Loading full Qwen3.5 model for V0 W8A8S quantization...")
        self.model_path = get_valid_read_path(str(self.model_path), is_dir=True, check_user_stat=True)

        arch = self.config.architectures[0] if self.config.architectures else ""
        if arch == "Qwen3_5MoeForConditionalGeneration":
            model_cls = Qwen3_5MoeForConditionalGeneration
        elif arch == "Qwen3_5ForConditionalGeneration":
            model_cls = Qwen3_5ForConditionalGeneration
        else:
            raise InvalidModelError(
                f"Invalid model architecture: {arch}",
                action="Please verify the model architecture is correct.",
            )

        device_map = "cpu" if device == DeviceType.CPU else "auto"
        model = model_cls.from_pretrained(  # nosec B615
            self.model_path,
            config=self.config,
            trust_remote_code=self.trust_remote_code,
            torch_dtype=torch.float16,
            local_files_only=True,
            device_map=device_map,
            attn_implementation="eager",
        ).eval()

        # Keep OV / smooth helpers that read flat num_*_heads on model.config
        if hasattr(model.config, "text_config"):
            text_cfg = model.config.text_config
            if hasattr(text_cfg, "num_attention_heads"):
                model.config.num_attention_heads = text_cfg.num_attention_heads
            if hasattr(text_cfg, "num_key_value_heads"):
                model.config.num_key_value_heads = text_cfg.num_key_value_heads

        get_logger().info("Full model loaded for V0 (dtype=float16, device_map=%s)", device_map)
        return model

    def inject_fa3_placeholders(
        self, root_name: str, root_module: nn.Module, should_inject: Callable[[str], bool]
    ) -> None:
        """为 Qwen3.5 full attention 层注入 FA3 占位（fa_q/fa_k/fa_v），排除 linear_attention 层。

        注意：Qwen3.5 每个 decoder 层会同时实例化 self_attn(Qwen3_5MoeAttention) 与
        linear_attn(Qwen3_5MoeGatedDeltaNet)，实际执行哪个由 config.layer_types[layer_idx]
        决定。仅对 layer_type 为 full_attention 的层注入，否则 linear 层会空挂占位/观测器，
        既浪费又会导致“每层都跑 FA 量化”的假象。

        forward 对齐 config.ini 中 qwen3_5_moe 约束的 transformers==5.2.0：
        Qwen3_5Attention / Qwen3_5MoeAttention（含 cache_position + cache_kwargs）。
        """
        from importlib import import_module

        def _is_full_attention(module: nn.Module) -> bool:
            cls_name = module.__class__.__name__
            # Qwen3_5Attention / Qwen3_5MoeAttention；排除 LinearAttention / GatedDeltaNet 等
            if "Linear" in cls_name:
                return False
            if not cls_name.endswith("Attention"):
                return False
            # 每层都实例化了 self_attn，但 linear_attention 层的 self_attn 不会被执行：
            # 按 config.layer_types 过滤，只量化真正 full_attention 的层
            config = getattr(module, "config", None)
            layer_types = getattr(config, "layer_types", None)
            if layer_types is None:
                return True
            idx = getattr(module, "layer_idx", None)
            if idx is None or idx >= len(layer_types):
                return True
            return layer_types[idx] != "linear_attention"

        def _wrap_attention_forward(attn_mod: nn.Module):
            attn_module = import_module(attn_mod.forward.__module__)
            apply_rotary_pos_emb = attn_module.apply_rotary_pos_emb
            eager_attention_forward = attn_module.eager_attention_forward
            all_attention_functions = attn_module.ALL_ATTENTION_FUNCTIONS

            def new_forward(
                self,
                hidden_states: torch.Tensor,
                position_embeddings: tuple,
                attention_mask: Optional[torch.Tensor] = None,
                past_key_values: Optional[Any] = None,
                cache_position: Optional[torch.LongTensor] = None,
                **kwargs,
            ):
                input_shape = hidden_states.shape[:-1]
                hidden_shape = (*input_shape, -1, self.head_dim)

                query_states, gate = torch.chunk(
                    self.q_proj(hidden_states).view(*input_shape, -1, self.head_dim * 2), 2, dim=-1
                )
                gate = gate.reshape(*input_shape, -1)

                query_states = self.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
                key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
                value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

                cos, sin = position_embeddings
                query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

                if past_key_values is not None:
                    # transformers==5.2.0: sin/cos + cache_position for static cache
                    cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
                    key_states, value_states = past_key_values.update(
                        key_states, value_states, self.layer_idx, cache_kwargs
                    )

                # ===== FA3 placeholders (after RoPE / cache, before attention kernel) =====
                if hasattr(self, "fa_q"):
                    query_states = self.fa_q(query_states)
                if hasattr(self, "fa_k"):
                    key_states = self.fa_k(key_states)
                if hasattr(self, "fa_v"):
                    value_states = self.fa_v(value_states)
                # ========================================================================

                attention_interface: Callable = all_attention_functions.get_interface(
                    self.config._attn_implementation, eager_attention_forward
                )

                attn_output, attn_weights = attention_interface(
                    self,
                    query_states,
                    key_states,
                    value_states,
                    attention_mask,
                    dropout=0.0 if not self.training else self.attention_dropout,
                    scaling=self.scaling,
                    **kwargs,
                )

                attn_output = attn_output.reshape(*input_shape, -1).contiguous()
                attn_output = attn_output * torch.sigmoid(gate)
                attn_output = self.o_proj(attn_output)
                return attn_output, attn_weights

            attn_mod.forward = types.MethodType(new_forward, attn_mod)

        for name, module in root_module.named_modules():
            if not _is_full_attention(module):
                continue
            full_name = f"{root_name}.{name}" if root_name else name
            if not should_inject(full_name):
                continue
            prefix = f"{name}." if name else ""
            root_module.set_submodule(f"{prefix}fa_q", FA3QuantPlaceHolder(ratio=0.9999))
            root_module.set_submodule(f"{prefix}fa_k", FA3QuantPlaceHolder(ratio=0.9999))
            root_module.set_submodule(f"{prefix}fa_v", FA3QuantPlaceHolder(ratio=0.9999))
            _wrap_attention_forward(module)
            get_logger().info("Injected FA3 placeholders into %s", full_name)

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
        ensure_conv1d_forward_patched()

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

        # Qwen3.5 RMSNorm uses (1 + weight); convert to standard RMSNorm to keep LN fusion mathematically correct.
        self._convert_qwen3_5_norm_to_standard(model)

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

    @staticmethod
    def _resolve_mm_token_type_ids(model: nn.Module, sample: Dict[str, Any], input_ids: torch.Tensor) -> torch.Tensor:
        """Return ``mm_token_type_ids`` (text=0 / image=1 / video=2) for ``get_rope_index``.

        transformers>=5.5 needs it to group the modality spans; the pinned transformers==5.2.0 derives the
        modality from ``input_ids`` itself and ignores the kwarg. The processor emits the tensor together
        with ``input_ids``; if it is absent it is derived from the placeholder tokens instead.
        """
        mm_token_type_ids = sample.get('mm_token_type_ids')
        if mm_token_type_ids is not None:
            return mm_token_type_ids

        get_logger().warning("mm_token_type_ids missing from processor outputs; deriving it from input_ids")
        mm_token_type_ids = torch.zeros_like(input_ids)
        mm_token_type_ids[input_ids == model.config.image_token_id] = 1
        mm_token_type_ids[input_ids == model.config.video_token_id] = 2
        return mm_token_type_ids

    def _decoder_layer_type(self, layer: nn.Module, name: str) -> Optional[str]:
        """Return the token-mixer type of a decoder layer (``linear_attention`` / ``full_attention``).

        transformers<=5.5.x exposes it as ``Qwen3_5DecoderLayer.layer_type``; newer versions use the
        same attribute under the name ``block_type``. Falls back to ``config.layer_types[idx]``
        parsed from the layer name (``model.language_model.layers.<idx>``).
        """
        for attr in ('layer_type', 'block_type'):
            layer_type = getattr(layer, attr, None)
            if layer_type:
                return layer_type

        index = name.rsplit('.', 1)[-1]
        layer_types = getattr(self.config.text_config, 'layer_types', None)
        if index.isdigit() and layer_types and int(index) < len(layer_types):
            return layer_types[int(index)]
        return None

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

        # 2. Vision encoder forward (optional for text-only calibration)
        pixel_values = sample.get('pixel_values')
        image_grid_thw = sample.get('image_grid_thw')
        input_ids = sample['input_ids']
        attention_mask = sample['attention_mask']
        inputs_embeds = model.model.language_model.embed_tokens(input_ids)

        if pixel_values is not None:
            vision_outputs = yield ProcessRequest(
                name="model.visual", module=model.model.visual, args=(pixel_values, image_grid_thw), kwargs={}
            )

            image_embeds = vision_outputs['pooler_output']

            if isinstance(image_embeds, (list, tuple)):
                image_embeds_cat = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            else:
                image_embeds_cat = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)

            image_mask = (input_ids == model.config.image_token_id).unsqueeze(-1).expand_as(inputs_embeds)
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds_cat)
        else:
            get_logger().info("Text-only calibration path: skipping vision encoder forward.")

        # 3. Prepare inputs for text decoder
        cache_position = torch.arange(0, inputs_embeds.shape[1], device=inputs_embeds.device)

        # Get position ids (mRoPE)
        # NOTE: transformers>=5.5 requires `mm_token_type_ids` (groups text/image/video spans); the pinned
        # transformers==5.2.0 derives the modality from input_ids and absorbs the kwarg through **kwargs.
        mm_token_type_ids = self._resolve_mm_token_type_ids(model, sample, input_ids)
        position_ids, rope_deltas = model.model.get_rope_index(
            input_ids=input_ids,
            mm_token_type_ids=mm_token_type_ids,
            image_grid_thw=image_grid_thw,
            attention_mask=attention_mask,
        )

        # Expand position_ids if needed (3D format for mROPE)
        if position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)

        # Mirrors Qwen3_5TextModel.forward
        if position_ids.ndim == 3 and position_ids.shape[0] == 4:
            text_position_ids = position_ids[0]
            position_ids = position_ids[1:]
        else:
            text_position_ids = position_ids[0]

        # CRITICAL: Convert 2D attention_mask to 4D causal mask
        # This is what Qwen3_5TextModel.forward does internally
        # NOTE: keep the kwargs `inputs_embeds` (5.2.0 deprecates `input_embeds`, removed in 5.6) and
        # `cache_position` (required positional argument of create_causal_mask in the pinned 5.2.0).
        causal_mask = create_causal_mask(
            config=model.config.text_config,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=None,
            position_ids=text_position_ids,
        )

        # Align with Qwen3_5TextModel._update_linear_attn_mask(attention_mask, cache_position): the
        # linear-attention layers only need the raw 2D padding mask, and no mask at all when every token
        # is attended to (or when continuing from a cache).
        linear_attn_mask = attention_mask
        if cache_position[0] > 0 or (attention_mask is not None and torch.all(attention_mask == 1)):
            linear_attn_mask = None

        # Create position embeddings (shared across layers)
        position_embeddings = model.model.language_model.rotary_emb(inputs_embeds, position_ids)

        # 4. Process each decoder layer
        # num_layers = self.config.text_config.num_hidden_layers
        has_mtp = self._has_mtp()
        hidden_states = inputs_embeds
        # 确保与模型权重的 dtype 一致（量化器 fake_quant 可能把激活值提升为 float32）
        model_dtype = next(model.parameters(), torch.empty(0)).dtype
        for name, layer in self.generate_decoder_layer(model):
            hidden_states = hidden_states.to(model_dtype)
            is_mtp_layer = has_mtp and name.startswith("mtp")

            if is_mtp_layer:
                # MTP preprocessing: norm -> lm_head -> shift ids -> embed -> norm -> project
                hidden_states, mtp_kwargs = self.mtp_preprocess(model, inputs=sample, hidden_states=hidden_states)
                hidden_states = yield ProcessRequest(name=name, module=layer, args=(hidden_states,), kwargs=mtp_kwargs)
            else:
                layer_mask = (
                    linear_attn_mask if self._decoder_layer_type(layer, name) == "linear_attention" else causal_mask
                )
                # Yield layer result
                # kwargs mirror Qwen3_5TextModel.forward (transformers==5.2.0 passes both the 3D `position_ids`
                # and `cache_position` down to the decoder layer; both are top-level params there).
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

    def _fake_quant_model_cls(self):
        arch = self.config.architectures[0]
        if arch == "Qwen3_5MoeForConditionalGeneration":
            return Qwen3_5MoeForConditionalGeneration
        if arch == "Qwen3_5ForConditionalGeneration":
            return Qwen3_5ForConditionalGeneration
        raise InvalidModelError(
            f"Invalid model architecture for fake-quant: {arch}",
            action="Please verify the AscendV1 export config.architectures.",
        )

    def build_meta_model(self) -> nn.Module:
        ensure_conv1d_forward_patched()
        model_cls = self._fake_quant_model_cls()

        self.config.use_cache = False
        self.config.text_config._attn_implementation = "eager"

        get_logger().info(
            "Building Qwen3.5 fake-quant shell: %d text layers, vision depth %d",
            self.config.text_config.num_hidden_layers,
            self.config.vision_config.depth,
        )

        ctor_kwargs = {
            "torch_dtype": self.get_global_model_torch_dtype(),
            "attn_implementation": "eager",
        }
        from accelerate import init_empty_weights

        with init_empty_weights(include_buffers=False):
            model = model_cls._from_config(self.config, **ctor_kwargs)
        model = model.eval()
        return model

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
            self._convert_qwen3_5_norm_to_standard(decoder)
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
        self._convert_qwen3_5_norm_to_standard(predictor)
        get_logger().info("MTP predictor loaded with %d parameters", len(loaded))
        return predictor

    def ascendv1_save_module_preprocess(
        self, prefix: str, module: nn.Module, model: nn.Module
    ) -> Tuple[str, nn.Module]:
        """
        Restore standard RMSNorm back to Qwen3.5 RMSNorm format before saving.
        """
        if module.__class__.__name__ != "Qwen3RMSNorm":
            return prefix, module

        use_moe_norm = self.config.architectures[0] == "Qwen3_5MoeForConditionalGeneration"
        if prefix.startswith("mtp."):
            restored = Qwen3_5RMSNorm(module.weight.shape[0], module.variance_epsilon)
        elif use_moe_norm:
            restored = Qwen3_5MoeRMSNorm(module.weight.shape[0], module.variance_epsilon)
        else:
            restored = Qwen3_5RMSNorm(module.weight.shape[0], module.variance_epsilon)

        restored.weight.data = module.weight.data - 1
        model.set_submodule(prefix, restored)
        return prefix, restored

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
        # `cache_position` is a required positional argument of create_causal_mask in transformers==5.2.0
        causal_mask = create_causal_mask(
            config=self.config.text_config,
            inputs_embeds=input_embeds_mtp,
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

    def get_ln_fuse_map(self):
        """
        Get LayerNorm-Linear fusion mapping for QuaRot.

        For Qwen3.5-MoE, LayerNorm (RMSNorm) fusion is applied to text decoder layers:
        - input_layernorm → QKV projections
        - post_attention_layernorm → MLP/MoE gate_proj, up_proj, and router
        - Final norm → lm_head

        Returns:
            Tuple of (pre_run_fused_ln, fused_map)
        """
        return {}, _qwen3_5_moe_get_ln_fuse_map(self.config.text_config)

    def get_bake_names(self):
        """
        Get bake mean names for QuaRot.

        Qwen3.5-MoE uses RMSNorm, so no bake mean is needed.

        Returns:
            Tuple of (pre_run_bake_names, bake_names)
        """
        return [], []

    def get_rotate_map(self, block_size):
        """
        Get rotation mapping for QuaRot on Qwen3.5-MoE.

        Args:
            block_size: Block size for Hadamard rotation

        Returns:
            Tuple of (pre_run_pairs, rotate_pairs)
        """
        pre_run, rot_pairs = _qwen3_5_moe_get_rotate_map(self.config, block_size)
        return [pre_run], list(rot_pairs.values())


def _qwen3_5_moe_get_ln_fuse_map(config):
    """
    Get LayerNorm-Linear fusion mapping for Qwen3.5 text decoder.

    Handles mixed attention types (linear_attn + full attention), MoE experts (for MoE models),
    standard MLP (for dense models), and MTP.

    Args:
        config: Text config (config.text_config)

    Returns:
        Dict mapping LayerNorm names to Linear layer names
    """
    ln_linear_map = {}

    full_attn_interval = getattr(config, 'full_attention_interval', None)
    is_moe = hasattr(config, 'num_experts') and getattr(config, 'num_experts', 0) > 0

    for layer_idx in range(config.num_hidden_layers):
        is_full_attn = full_attn_interval and (layer_idx % full_attn_interval == full_attn_interval - 1)

        if is_full_attn:
            ln_linear_map[f"model.language_model.layers.{layer_idx}.input_layernorm"] = [
                f"model.language_model.layers.{layer_idx}.self_attn.q_proj",
                f"model.language_model.layers.{layer_idx}.self_attn.k_proj",
                f"model.language_model.layers.{layer_idx}.self_attn.v_proj",
            ]
        else:
            ln_linear_map[f"model.language_model.layers.{layer_idx}.input_layernorm"] = [
                f"model.language_model.layers.{layer_idx}.linear_attn.in_proj_a",
                f"model.language_model.layers.{layer_idx}.linear_attn.in_proj_b",
                f"model.language_model.layers.{layer_idx}.linear_attn.in_proj_qkv",
                f"model.language_model.layers.{layer_idx}.linear_attn.in_proj_z",
            ]

        if is_moe:
            routed_experts = [
                f"model.language_model.layers.{layer_idx}.mlp.experts.{i}.{proj}"
                for proj in ["gate_proj", "up_proj"]
                for i in range(config.num_experts)
            ]
            shared_experts = [
                f"model.language_model.layers.{layer_idx}.mlp.shared_expert.{proj}" for proj in ["gate_proj", "up_proj"]
            ]
            expert_gate = [f"model.language_model.layers.{layer_idx}.mlp.gate"]
            shared_expert_gate = [f"model.language_model.layers.{layer_idx}.mlp.shared_expert_gate"]

            ln_linear_map[f"model.language_model.layers.{layer_idx}.post_attention_layernorm"] = (
                routed_experts + shared_experts + expert_gate + shared_expert_gate
            )
        else:
            ln_linear_map[f"model.language_model.layers.{layer_idx}.post_attention_layernorm"] = [
                f"model.language_model.layers.{layer_idx}.mlp.gate_proj",
                f"model.language_model.layers.{layer_idx}.mlp.up_proj",
            ]

    ln_linear_map["model.language_model.norm"] = ['lm_head']

    mtp_num_layers = getattr(config, 'mtp_num_hidden_layers', 0)
    if mtp_num_layers > 0:
        ln_linear_map[("mtp.pre_fc_norm_embedding", "mtp.pre_fc_norm_hidden")] = ["mtp.fc"]

        for mtp_layer_idx in range(mtp_num_layers):
            ln_linear_map[f"mtp.layers.{mtp_layer_idx}.input_layernorm"] = [
                f"mtp.layers.{mtp_layer_idx}.self_attn.q_proj",
                f"mtp.layers.{mtp_layer_idx}.self_attn.k_proj",
                f"mtp.layers.{mtp_layer_idx}.self_attn.v_proj",
            ]

            if is_moe:
                mtp_routed_experts = [
                    f"mtp.layers.{mtp_layer_idx}.mlp.experts.{i}.{proj}"
                    for proj in ["gate_proj", "up_proj"]
                    for i in range(config.num_experts)
                ]
                mtp_shared_experts = [
                    f"mtp.layers.{mtp_layer_idx}.mlp.shared_expert.{proj}" for proj in ["gate_proj", "up_proj"]
                ]
                mtp_expert_gate = [f"mtp.layers.{mtp_layer_idx}.mlp.gate"]

                ln_linear_map[f"mtp.layers.{mtp_layer_idx}.post_attention_layernorm"] = (
                    mtp_routed_experts + mtp_shared_experts + mtp_expert_gate
                )
            else:
                ln_linear_map[f"mtp.layers.{mtp_layer_idx}.post_attention_layernorm"] = [
                    f"mtp.layers.{mtp_layer_idx}.mlp.gate_proj",
                    f"mtp.layers.{mtp_layer_idx}.mlp.up_proj",
                ]

    return ln_linear_map


def _qwen3_5_moe_get_rotate_map(config, block_size):
    """
    Get rotation mapping for Qwen3.5 model (text decoder + visual projections + MTP).

    Supports both MoE and Dense model variants.

    Mathematical Framework:
    ----------------------
    We apply a global rotation R to the residual stream (hidden states).
    - Text embeddings: W_new = W_old @ R (right rotation)
    - Visual outputs: W_new = R^T @ W_old (left rotation)
    - Text layers: Mixed attention types require careful handling
    - MTP: Fusion layer uses block_diag for concat inputs

    Special Considerations:
    -----------------------
    1. Mixed Attention Types:
       - Full attention layers (every 4th layer): Standard QKV/O rotation
       - Linear attention layers: Rotate inputs (in_proj_*) and output (out_proj)
       - Linear attention internal ops (conv1d, norm) run in unrotated space

    2. Model Variants:
       - MoE: Routed experts (256) + shared expert + router gate
       - Dense: Standard MLP with gate_proj, up_proj, down_proj

    3. Vision Integration:
       - Visual merger output must be rotated to match text embedding space

    4. MTP (Multi-Token Prediction):
       - FC layer receives concat([norm_embedding, norm_hidden])
       - Use block_diag rotation for FC layer input
       - MTP layers use full attention + MLP/MoE (similar to main model)

    Args:
        config: Full model config (Qwen3_5MoeConfig or Qwen3_5Config)
        block_size: Block size for rotation

    Returns:
        Tuple of (pre_run, rot_pairs)
    """
    text_config = config.text_config

    rot = QuaRotInterface.get_rotate_command(
        size=text_config.hidden_size,
        mode=QuaRotInterface.QuaRotMode.HADAMARD,
        block_size=block_size,
    )

    pre_run_left = {}
    pre_run_right = {}

    pre_run_right["model.language_model.embed_tokens"] = rot

    if hasattr(config, 'vision_config'):
        pre_run_left["model.visual.merger.linear_fc2"] = rot

    pre_run = QuaRotInterface.RotatePair(left_rot=pre_run_left, right_rot=pre_run_right)

    rot_pairs = {}

    left_rot = {}
    right_rot = {}
    right_rot["lm_head"] = rot

    full_attn_interval = getattr(text_config, 'full_attention_interval', None)
    is_moe = hasattr(text_config, 'num_experts') and getattr(text_config, 'num_experts', 0) > 0

    for layer_idx in range(text_config.num_hidden_layers):
        is_full_attn = full_attn_interval and (layer_idx % full_attn_interval == full_attn_interval - 1)

        if is_full_attn:
            right_rot[f"model.language_model.layers.{layer_idx}.self_attn.q_proj"] = rot
            right_rot[f"model.language_model.layers.{layer_idx}.self_attn.k_proj"] = rot
            right_rot[f"model.language_model.layers.{layer_idx}.self_attn.v_proj"] = rot
            left_rot[f"model.language_model.layers.{layer_idx}.self_attn.o_proj"] = rot
        else:
            right_rot[f"model.language_model.layers.{layer_idx}.linear_attn.in_proj_qkv"] = rot
            right_rot[f"model.language_model.layers.{layer_idx}.linear_attn.in_proj_z"] = rot
            right_rot[f"model.language_model.layers.{layer_idx}.linear_attn.in_proj_b"] = rot
            right_rot[f"model.language_model.layers.{layer_idx}.linear_attn.in_proj_a"] = rot
            left_rot[f"model.language_model.layers.{layer_idx}.linear_attn.out_proj"] = rot

        if is_moe:
            for i in range(text_config.num_experts):
                right_rot[f"model.language_model.layers.{layer_idx}.mlp.experts.{i}.gate_proj"] = rot
                right_rot[f"model.language_model.layers.{layer_idx}.mlp.experts.{i}.up_proj"] = rot
                left_rot[f"model.language_model.layers.{layer_idx}.mlp.experts.{i}.down_proj"] = rot

            right_rot[f"model.language_model.layers.{layer_idx}.mlp.shared_expert.gate_proj"] = rot
            right_rot[f"model.language_model.layers.{layer_idx}.mlp.shared_expert.up_proj"] = rot
            right_rot[f"model.language_model.layers.{layer_idx}.mlp.shared_expert_gate"] = rot
            left_rot[f"model.language_model.layers.{layer_idx}.mlp.shared_expert.down_proj"] = rot

            right_rot[f"model.language_model.layers.{layer_idx}.mlp.gate"] = rot
        else:
            right_rot[f"model.language_model.layers.{layer_idx}.mlp.gate_proj"] = rot
            right_rot[f"model.language_model.layers.{layer_idx}.mlp.up_proj"] = rot
            left_rot[f"model.language_model.layers.{layer_idx}.mlp.down_proj"] = rot

    mtp_num_layers = getattr(text_config, 'mtp_num_hidden_layers', 0)
    if mtp_num_layers > 0:
        right_rot["mtp.fc"] = torch.block_diag(*[rot, rot])
        left_rot["mtp.fc"] = rot

        for mtp_layer_idx in range(mtp_num_layers):
            right_rot[f"mtp.layers.{mtp_layer_idx}.self_attn.q_proj"] = rot
            right_rot[f"mtp.layers.{mtp_layer_idx}.self_attn.k_proj"] = rot
            right_rot[f"mtp.layers.{mtp_layer_idx}.self_attn.v_proj"] = rot
            left_rot[f"mtp.layers.{mtp_layer_idx}.self_attn.o_proj"] = rot

            if is_moe:
                for i in range(text_config.num_experts):
                    right_rot[f"mtp.layers.{mtp_layer_idx}.mlp.experts.{i}.gate_proj"] = rot
                    right_rot[f"mtp.layers.{mtp_layer_idx}.mlp.experts.{i}.up_proj"] = rot
                    left_rot[f"mtp.layers.{mtp_layer_idx}.mlp.experts.{i}.down_proj"] = rot

                right_rot[f"mtp.layers.{mtp_layer_idx}.mlp.shared_expert.gate_proj"] = rot
                right_rot[f"mtp.layers.{mtp_layer_idx}.mlp.shared_expert.up_proj"] = rot
                left_rot[f"mtp.layers.{mtp_layer_idx}.mlp.shared_expert.down_proj"] = rot

                right_rot[f"mtp.layers.{mtp_layer_idx}.mlp.gate"] = rot
            else:
                right_rot[f"mtp.layers.{mtp_layer_idx}.mlp.gate_proj"] = rot
                right_rot[f"mtp.layers.{mtp_layer_idx}.mlp.up_proj"] = rot
                left_rot[f"mtp.layers.{mtp_layer_idx}.mlp.down_proj"] = rot

    rot_pairs['rot'] = QuaRotInterface.RotatePair(left_rot=left_rot, right_rot=right_rot)

    return pre_run, rot_pairs
