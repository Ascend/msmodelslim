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
import types
from collections import defaultdict
from contextlib import contextmanager
from importlib import import_module
from pathlib import Path
from typing import List, Any, Generator, Tuple, Dict, Optional, Callable, Union
from unittest.mock import patch

import torch
import torch.distributed as dist
from safetensors import safe_open
from torch import nn
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer
from transformers.masking_utils import create_causal_mask

from msmodelslim.core.const import DeviceType
from msmodelslim.app.naive_quantization.model_info_interface import ModelInfoInterface
from msmodelslim.core.base.protocol import ProcessRequest
from msmodelslim.core.graph import AdapterConfig, MappingConfig, FusionConfig
from msmodelslim.model.common.layer_wise_forward import generated_decoder_layer_visit_func
from msmodelslim.model.common.vlm_base import VLMBaseModelAdapter
from msmodelslim.model.interface_hub import (
    ModelSlimPipelineInterfaceV1,
    LayerWiseOffloadOptionalInterface,
    AscendV1SaveInterface,
    IterSmoothInterface,
    FlexSmoothQuantInterface,
    QuaRotInterface,
    FA3QuantAdapterInterface,
    FA3QuantPlaceHolder,
    AttentionAnalysisInterface,
)
from msmodelslim.utils.exception import UnsupportedError
from msmodelslim.utils.logging import logger_setter, get_logger
from msmodelslim.utils.security import (
    get_valid_read_path,
    MAX_READ_FILE_SIZE_32G,
    safe_copy_file,
)

from .convert_mxfp4_to_bf16 import dequant_subtree_mxfp4_to_bf16, get_full_weight_map
from msmodelslim.model.common.utils import _get_expert_range

from .ep_patches import apply_kimi_k3_ep_patches
from .quarot import (
    attn_prefix,
    get_ln_fuse_map as build_ln_fuse_map,
    get_rotate_map as build_rotate_map,
    input_layernorm_targets,
    is_dense_mlp,
    is_kda_layer,
    kda_use_full_rank_gate,
    layer_prefix,
    moe_prefix,
    post_attention_layernorm_targets,
    q_a_layernorm_targets,
    routed_expert_norm_targets,
)
from .runtime_patches import apply_kimi_k3_runtime_patches


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
class KimiK3ModelAdapter(  # pylint: disable=too-many-ancestors
    VLMBaseModelAdapter,
    ModelInfoInterface,
    ModelSlimPipelineInterfaceV1,
    LayerWiseOffloadOptionalInterface,
    AscendV1SaveInterface,
    IterSmoothInterface,
    FlexSmoothQuantInterface,
    QuaRotInterface,
    FA3QuantAdapterInterface,
    AttentionAnalysisInterface,
):
    """Kimi-K3 VLM MoE adapter with EP monkey-patch + offline QuaRot.

    EP uses ``ep_patches`` on weight-dir ``KimiSparseMoeBlock`` (from_config).
    QuaRot maps (incl. ``rot_latent`` + ``_get_expert_range``) come from the
    verified msmodelslim rotation path; mm_projector gets Identity ``rot_proj``.
    FA3 injects absorb-MLA placeholders on ``KimiMLAAttention`` only (not KDA).
    Attention MSE hooks the same MLA modules for float vs FA3 sensitivity analysis.
    """

    # Per-matrix switches for get_rotate_map (disable to skip that RotatePair).
    # Note: rot_uv dropped — mla_use_output_gate breaks offline V/o_proj Hadamard cancel.
    enable_rot = True
    enable_rot_b_proj = True
    enable_rot_kv_b_proj = True
    enable_rot_latent = True

    def __init__(self, model_type: str, model_path: Path, trust_remote_code: bool = False):
        self._processor = None
        self._tokenizer = None
        super().__init__(model_type, model_path, trust_remote_code)

    def get_model_pedigree(self) -> str:
        return "kimi_k3"

    def get_model_type(self) -> str:
        return self.model_type

    def get_layer_wise_offload_device(self):
        # Prefer meta so layer-wise runner can drop finished layers from host memory.
        return "meta"

    def handle_dataset(self, dataset: Any, device: DeviceType = DeviceType.NPU) -> List[Any]:
        """Handle multimodal Kimi-K3 calibration dataset (image + text)."""
        try:
            self._processor = AutoProcessor.from_pretrained(  # nosec B615
                self.model_path,
                trust_remote_code=self.trust_remote_code,
                local_files_only=True,
            )
        except Exception as e:
            get_logger().warning(
                "AutoProcessor load failed from %s, try tokenizer fallback: %s",
                self.model_path,
                str(e),
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
                        "Ensure model directory contains processor/tokenizer files "
                        f"(preprocessor_config.json, tiktoken.model). "
                        f"processor_error={e}; tokenizer_error={e_tokenizer}"
                    ),
                ) from e_tokenizer

            raise UnsupportedError(
                "Kimi-K3 multimodal calibration requires AutoProcessor, but only tokenizer is available.",
                action=(
                    "Provide processor assets in model_path (preprocessor_config.json and corresponding remote code)."
                ),
            ) from e

        for item in dataset:
            if item.image is None or item.text is None:
                raise UnsupportedError(
                    "Kimi-K3 adapter requires both image and text for calibration.",
                    action="Use multimodal (image+text) data only.",
                )

        processed_data = []
        for item in tqdm(dataset, desc="Processing Kimi-K3 calibration dataset"):
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": get_valid_read_path(item.image)},
                        {"type": "text", "text": item.text},
                    ],
                }
            ]
            inputs = self._processor(messages=messages, return_tensors="pt")
            processed_item = self._collect_inputs_to_device(
                inputs,
                device,
                keys=["input_ids", "pixel_values", "grid_thws", "attention_mask"],
                defaults={},
            )
            processed_data.append(processed_item)

        get_logger().info("Kimi-K3 dataset preprocessing finished, samples=%d", len(processed_data))
        return processed_data

    def _strip_quantization_config(self) -> None:
        """Drop compressed-tensors metadata so HF builds plain Linear modules.

        Weights are loaded manually via ``_get_state_dict`` / mxfp4 dequant.
        Keeping quantization_config would force ``from_pretrained`` to scan the
        full ~500k-key index (O(n^2) unexpected-key filtering) and run
        ``compress_model`` / ``cast_to_fp4``.
        """
        for cfg in (self.config, getattr(self.config, "text_config", None)):
            if cfg is None or not hasattr(cfg, "quantization_config"):
                continue
            try:
                delattr(cfg, "quantization_config")
            except AttributeError:
                cfg.quantization_config = None

    def init_model(self, device: DeviceType = DeviceType.NPU) -> nn.Module:
        """Load vision + 1 decoder layer; remaining layers load on demand."""
        with default_dtype(torch.bfloat16):
            origin_layers = self.config.text_config.num_hidden_layers
            get_logger().info(
                "Model with %d text layers + %d vision layers",
                origin_layers,
                self.config.vision_config.vt_num_hidden_layers,
            )
            self.config.text_config.num_hidden_layers = 1
            self.config.use_cache = False
            self._strip_quantization_config()

            self.model_path = get_valid_read_path(str(self.model_path), is_dir=True, check_user_stat=True)

            get_logger().info(
                "Building vision encoder and first text decoder layer from config (skip HF weight index remap)..."
            )
            self.config._attn_implementation = "eager"
            self.config.text_config._attn_implementation = "eager"
            self.config.vision_config._attn_implementation = "eager"

            # EP twice only: (1) before MoE construct, (2) after from_config imports.
            apply_kimi_k3_ep_patches(model_path=str(self.model_path))

            # from_config + manual shard load avoids transformers matching the
            # full 93-layer mxfp4 index against a 1-layer skeleton.
            with patch.object(nn.Linear, "reset_parameters", lambda _self: None):
                model = AutoModelForCausalLM.from_config(
                    self.config,
                    trust_remote_code=self.trust_remote_code,
                    attn_implementation="eager",
                )
            apply_kimi_k3_ep_patches(model_path=str(self.model_path))

            self.config.text_config.num_hidden_layers = origin_layers
            self.config.text_config._attn_implementation = "eager"
            self.config.vision_config._attn_implementation = "eager"

            get_logger().info("Loading weights for vision encoder, first decoder layer, and lm_head...")
            state_dict = self._get_state_dict(model)
            self._load_state_dict_compatible(model, state_dict)
            dequant_subtree_mxfp4_to_bf16(model, "", str(self.model_path))

            # NPU-safe KDA (do not edit weight-dir modeling_*.py). EP already ensured above.
            apply_kimi_k3_runtime_patches(model)
            # QuaRot: Identity rot_proj so VLM path shares residual rotation with embed.
            self._patch_mm_projector_rot_proj(model)

            model.eval()

            if hasattr(model.config.text_config, "num_attention_heads"):
                model.config.num_attention_heads = model.config.text_config.num_attention_heads
            if hasattr(model.config.text_config, "num_key_value_heads"):
                model.config.num_key_value_heads = model.config.text_config.num_key_value_heads

            get_logger().info("Model initialized with %d layers (1 loaded, others on-demand)", origin_layers)
            return model

    def generate_model_visit(self, model: nn.Module) -> Generator[ProcessRequest, Any, None]:
        get_logger().info("Processing vision encoder...")
        yield ProcessRequest(name="vision_tower", module=model.vision_tower, args=(), kwargs={})

        get_logger().info("Processing mm_projector...")
        yield ProcessRequest(name="mm_projector", module=model.mm_projector, args=(), kwargs={})

        get_logger().info("Processing text decoder layers...")
        yield from generated_decoder_layer_visit_func(model, transformer_blocks=self.generate_decoder_layer(model))

    def generate_model_forward(self, model: nn.Module, inputs: Any) -> Generator[ProcessRequest, Any, None]:
        """Layer-wise forward matching KimiK3ForConditionalGeneration + KimiLinearModel."""
        sample = inputs[0] if isinstance(inputs, list) else inputs

        input_ids = sample["input_ids"]
        attention_mask = sample.get("attention_mask", None)
        pixel_values = sample.get("pixel_values", None)
        grid_thws = sample.get("grid_thws", None)

        language_model = model.language_model
        backbone = language_model.model
        inputs_embeds = backbone.embed_tokens(input_ids)

        if pixel_values is not None:
            pixel_values = pixel_values.to(model.vision_tower.patch_embed.proj.weight.dtype)

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
            if model.mm_projector is not None:
                image_features = yield ProcessRequest(
                    name="mm_projector",
                    module=model.mm_projector,
                    args=(image_features,),
                    kwargs={},
                )
            inputs_embeds = inputs_embeds.to(image_features[0].dtype)
            inputs_embeds, attention_mask, _, position_ids = model._merge_input_ids_with_image_features(
                image_features=image_features,
                inputs_embeds=inputs_embeds,
                input_ids=input_ids,
                attention_mask=attention_mask if attention_mask is not None else torch.ones_like(input_ids),
                labels=None,
            )

        cache_position = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device)
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = create_causal_mask(
            config=backbone.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=None,
            position_ids=position_ids,
        )
        linear_attn_mask = backbone._update_linear_attn_mask(attention_mask, cache_position)

        use_attn_residuals = getattr(backbone, "use_attn_residuals", False)
        hidden_states = inputs_embeds
        block_residual = None
        if use_attn_residuals:
            block_residual = hidden_states.new_zeros(
                hidden_states.shape[0] * hidden_states.shape[1],
                0,
                hidden_states.shape[2],
            )

        for name, layer in self.generate_decoder_layer(model):
            if dist.is_initialized():
                dist.barrier()

            layer_mask = linear_attn_mask if getattr(layer, "is_linear_attn", False) else causal_mask
            kwargs = {
                "attention_mask": layer_mask,
                "position_ids": position_ids,
                "past_key_values": None,
                "output_attentions": False,
                "use_cache": False,
                "cache_position": cache_position,
            }
            if use_attn_residuals:
                kwargs["block_residual"] = block_residual

            layer_out = yield ProcessRequest(name=name, module=layer, args=(hidden_states,), kwargs=kwargs)
            if use_attn_residuals:
                hidden_states, block_residual = layer_out
            else:
                hidden_states = layer_out[0] if isinstance(layer_out, tuple) else layer_out

    def generate_decoder_layer(self, model: nn.Module) -> Generator[Tuple[str, nn.Module], None, None]:
        num_layers = self.config.text_config.num_hidden_layers
        for layer_idx in range(num_layers):
            name = f"language_model.model.layers.{layer_idx}"
            layer = self._load_decoder_if_not_exist(model, name, layer_idx)
            yield name, layer

    def enable_kv_cache(self, model: nn.Module, need_kv_cache: bool) -> None:
        model.config.use_cache = need_kv_cache
        if hasattr(model, "language_model") and hasattr(model.language_model, "config"):
            model.language_model.config.use_cache = need_kv_cache
        get_logger().info("KV cache %s", "enabled" if need_kv_cache else "disabled")

    def _patch_mm_projector_rot_proj(self, model: nn.Module) -> None:
        """Append Identity Linear rot_proj after mm_projector so QuaRot can left_rot it."""
        mm = getattr(model, "mm_projector", None)
        if mm is None or hasattr(mm, "rot_proj"):
            return

        hidden = self.config.text_config.hidden_size
        rot_proj = nn.Linear(hidden, hidden, bias=False)
        with torch.no_grad():
            rot_proj.weight.copy_(torch.eye(hidden, dtype=rot_proj.weight.dtype, device=rot_proj.weight.device))
        mm.add_module("rot_proj", rot_proj)

        orig_forward = mm.forward

        def forward_with_rot(self, x, *args, **kwargs):
            out = orig_forward(x, *args, **kwargs)
            if isinstance(out, (list, tuple)):
                return type(out)(self.rot_proj(t) for t in out)
            return self.rot_proj(out)

        mm.forward = types.MethodType(forward_with_rot, mm)
        get_logger().info("Patched mm_projector.rot_proj (Linear %d->%d, eye init)", hidden, hidden)

    def get_ln_fuse_map(self):
        return build_ln_fuse_map(self.config, num_hidden_layers=self.config.text_config.num_hidden_layers)

    def get_bake_names(self):
        return [], []

    def get_rotate_map(self, block_size):
        return build_rotate_map(
            self.config,
            block_size,
            num_hidden_layers=self.config.text_config.num_hidden_layers,
            enable_rot=self.enable_rot,
            enable_rot_b_proj=self.enable_rot_b_proj,
            enable_rot_kv_b_proj=self.enable_rot_kv_b_proj,
            enable_rot_latent=self.enable_rot_latent,
        )

    def _mla_subgraph_configs(self, layer_idx: int) -> List[AdapterConfig]:
        """IterSmooth mappings for MLA (full-attn) layers."""
        text_cfg = self.config.text_config
        prefix = layer_prefix(layer_idx)
        attn = attn_prefix(layer_idx)
        return [
            AdapterConfig(
                subgraph_type="ov",
                mapping=MappingConfig(
                    source=f"{attn}.kv_b_proj",
                    targets=[f"{attn}.o_proj"],
                ),
                extra_config={"group_method": "max"},
                fusion=FusionConfig(
                    fusion_type="kv",
                    num_attention_heads=text_cfg.num_attention_heads,
                    num_key_value_heads=text_cfg.num_key_value_heads,
                    custom_config={
                        "qk_nope_head_dim": text_cfg.qk_nope_head_dim,
                        "v_head_dim": text_cfg.v_head_dim,
                    },
                ),
            ),
            AdapterConfig(
                subgraph_type="norm-linear",
                mapping=MappingConfig(
                    source=f"{prefix}.input_layernorm",
                    targets=input_layernorm_targets(text_cfg, layer_idx),
                ),
            ),
            AdapterConfig(
                subgraph_type="norm-linear",
                mapping=MappingConfig(
                    source=f"{attn}.q_a_layernorm",
                    targets=q_a_layernorm_targets(text_cfg, layer_idx),
                ),
            ),
            # kv_a_layernorm→kv_b is in get_ln_fuse_map for QuaRot; IterSmooth keeps
            # ov(kv_b→o) instead (DeepSeek / Kimi-K2 convention) to avoid dual scales on kv_b.
        ]

    def _kda_subgraph_configs(self, layer_idx: int) -> List[AdapterConfig]:
        """IterSmooth mappings for KDA (linear-attn) layers."""
        text_cfg = self.config.text_config
        prefix = layer_prefix(layer_idx)
        attn = attn_prefix(layer_idx)
        configs: List[AdapterConfig] = [
            AdapterConfig(
                subgraph_type="norm-linear",
                mapping=MappingConfig(
                    source=f"{prefix}.input_layernorm",
                    targets=input_layernorm_targets(text_cfg, layer_idx),
                ),
            ),
            AdapterConfig(
                subgraph_type="linear-linear",
                mapping=MappingConfig(
                    source=f"{attn}.f_a_proj",
                    targets=[f"{attn}.f_b_proj"],
                ),
            ),
        ]
        # Low-rank output gate: g_a→g_b mirrors f_a→f_b.
        if not kda_use_full_rank_gate(text_cfg):
            configs.append(
                AdapterConfig(
                    subgraph_type="linear-linear",
                    mapping=MappingConfig(
                        source=f"{attn}.g_a_proj",
                        targets=[f"{attn}.g_b_proj"],
                    ),
                )
            )
        return configs

    def _ffn_subgraph_configs(self, layer_idx: int) -> List[AdapterConfig]:
        """IterSmooth FFN/MoE mappings (post_attention norm-linear + up-down + latent MoE).

        ``post_attention_layernorm`` targets mirror ``get_ln_fuse_map`` so every
        Linear/router that consumes that Norm is listed (completeness rule).
        """
        text_cfg = self.config.text_config
        prefix = layer_prefix(layer_idx)
        configs: List[AdapterConfig] = [
            AdapterConfig(
                subgraph_type="norm-linear",
                mapping=MappingConfig(
                    source=f"{prefix}.post_attention_layernorm",
                    targets=post_attention_layernorm_targets(text_cfg, layer_idx),
                ),
            )
        ]

        if is_dense_mlp(text_cfg, layer_idx):
            configs.append(
                AdapterConfig(
                    subgraph_type="up-down",
                    mapping=MappingConfig(
                        source=f"{prefix}.mlp.up_proj",
                        targets=[f"{prefix}.mlp.down_proj"],
                    ),
                )
            )
            return configs

        moe = moe_prefix(layer_idx)
        configs.append(
            AdapterConfig(
                subgraph_type="up-down",
                mapping=MappingConfig(
                    source=f"{moe}.shared_experts.up_proj",
                    targets=[f"{moe}.shared_experts.down_proj"],
                ),
            )
        )

        routed_norm = routed_expert_norm_targets(text_cfg, layer_idx, require_latent_hidden=True)
        if routed_norm is not None:
            configs.append(
                AdapterConfig(
                    subgraph_type="norm-linear",
                    mapping=MappingConfig(
                        source=f"{moe}.routed_expert_norm",
                        targets=routed_norm,
                    ),
                )
            )

        expert_start, expert_end = _get_expert_range(text_cfg)
        for expert in range(expert_start, expert_end):
            configs.append(
                AdapterConfig(
                    subgraph_type="up-down",
                    mapping=MappingConfig(
                        source=f"{moe}.experts.{expert}.w3",
                        targets=[f"{moe}.experts.{expert}.w2"],
                    ),
                )
            )
        return configs

    def get_adapter_config_for_subgraph(self) -> List[AdapterConfig]:
        """Subgraph mappings for IterSmooth (also usable by FlexSmooth).

        Norm-linear targets for a shared Norm must list every Linear that consumes
        that Norm output (same completeness rule as ``get_ln_fuse_map``), otherwise
        absorbing scales into the Norm breaks unlisted branches.

        AttnRes norms remain QuaRot-only (``get_ln_fuse_map``).
        """
        adapter_config: List[AdapterConfig] = []
        num_layers = self.config.text_config.num_hidden_layers
        for layer_idx in range(num_layers):
            if is_kda_layer(self.config.text_config, layer_idx):
                adapter_config.extend(self._kda_subgraph_configs(layer_idx))
            else:
                adapter_config.extend(self._mla_subgraph_configs(layer_idx))
            adapter_config.extend(self._ffn_subgraph_configs(layer_idx))
        get_logger().info(
            "Built %d IterSmooth subgraph configs for %d layers",
            len(adapter_config),
            num_layers,
        )
        return adapter_config

    # ===== FA3QuantAdapterInterface =====
    def inject_fa3_placeholders(
        self, root_name: str, root_module: nn.Module, should_inject: Callable[[str], bool]
    ) -> None:
        """Install FA3 placeholders on KimiMLAAttention (absorb MLA path).

        Mirrors DeepSeekV3 / KimiK2.5 FA3 injection for Ascend MLA FA kernels:
        - Submodules: ``fa_q``, ``fa_k``, ``fa_v``
        - After q_absorb / on compressed_kv::
            q_nope = self.fa_q(q_nope)
            compressed_kv = self.fa_k(compressed_kv.unsqueeze(1)).squeeze(1)
            _ = self.fa_v(compressed_kv.unsqueeze(1)).squeeze(1)

        Only ``KimiMLAAttention`` is wrapped; ``KimiDeltaAttention`` (KDA) is skipped.
        Kimi-K3 uses ``mla_use_nope`` (no RoPE); absorb forward skips rotary when
        ``rotary_emb`` is missing.
        """

        def _wrap_mla_forward(attn_mod: nn.Module):
            apply_rotary_pos_emb = None
            try:
                modeling = import_module(attn_mod.forward.__module__)
                apply_rotary_pos_emb = getattr(modeling, "apply_rotary_pos_emb", None)
            except Exception as exc:  # pylint: disable=broad-exception-caught
                get_logger().warning(
                    "FA3 wrap: cannot import apply_rotary_pos_emb from %s: %s",
                    getattr(attn_mod.forward, "__module__", None),
                    exc,
                )

            def new_forward(
                self,
                hidden_states: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.LongTensor] = None,
                past_key_value: Optional[Any] = None,
                past_key_values: Optional[Any] = None,
                output_attentions: bool = False,
                use_cache: bool = False,
                **kwargs,
            ):
                del output_attentions, use_cache  # absorb calib path returns attn only
                past = past_key_values if past_key_values is not None else past_key_value
                bsz, q_len, _ = hidden_states.size()

                if getattr(self, "q_lora_rank", None) is None:
                    q = self.q_proj(hidden_states)
                else:
                    q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
                q = q.view(bsz, q_len, self.num_heads, self.q_head_dim).transpose(1, 2)
                q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

                compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
                compressed_kv, k_pe = torch.split(compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
                compressed_kv = self.kv_a_layernorm(compressed_kv)
                k_pe = k_pe.view(bsz, q_len, 1, self.qk_rope_head_dim).transpose(1, 2)
                kv_seq_len = k_pe.shape[-2]

                if past is not None:
                    if getattr(self, "layer_idx", None) is None:
                        raise ValueError(
                            f"The cache structure has changed since version v4.36. "
                            f"If you are using {self.__class__.__name__} "
                            f"for auto-regressive decoding with k/v caching, "
                            f"please make sure to initialize the attention class "
                            "with a layer index."
                        )
                    if hasattr(past, "get_usable_length"):
                        kv_seq_len += past.get_usable_length(kv_seq_len, self.layer_idx)

                # Kimi-K3 MLA runs with mla_use_nope (rotary_emb is typically None).
                cos = sin = None
                rotary_emb = getattr(self, "rotary_emb", None)
                if rotary_emb is not None and apply_rotary_pos_emb is not None:
                    cos, sin = rotary_emb(q_pe, seq_len=kv_seq_len)
                    q_pe, k_pe = apply_rotary_pos_emb(q_pe, k_pe, cos, sin, position_ids)

                if past is not None and hasattr(past, "update"):
                    cache_kwargs = {"sin": sin, "cos": cos}
                    compressed_kv = compressed_kv.unsqueeze(1)
                    updated = past.update(k_pe, compressed_kv, self.layer_idx, cache_kwargs)
                    if isinstance(updated, tuple) and len(updated) == 2:
                        k_pe, compressed_kv = updated
                    compressed_kv = compressed_kv.squeeze(1)

                kv_b_proj = self.kv_b_proj.weight.view(self.num_heads, -1, self.kv_lora_rank)
                q_absorb = kv_b_proj[:, : self.qk_nope_head_dim, :]
                out_absorb = kv_b_proj[:, self.qk_nope_head_dim :, :]
                q_nope = torch.matmul(q_nope, q_absorb)

                # ===== FA3 placeholders (Ascend MLA absorb tensors) =====
                if hasattr(self, "fa_q"):
                    q_nope = self.fa_q(q_nope)
                if hasattr(self, "fa_k"):
                    compressed_kv = self.fa_k(compressed_kv.unsqueeze(1)).squeeze(1)
                if hasattr(self, "fa_v"):
                    _ = self.fa_v(compressed_kv.unsqueeze(1)).squeeze(1)
                # ========================================================

                softmax_scale = getattr(self, "softmax_scale", None)
                if softmax_scale is None:
                    softmax_scale = getattr(self, "scaling", self.q_head_dim ** (-0.5))

                attn_weights = torch.matmul(q_pe, k_pe.mT) + torch.matmul(q_nope, compressed_kv.unsqueeze(-3).mT)
                attn_weights = attn_weights * softmax_scale

                if attention_mask is not None:
                    attn_weights = attn_weights + attention_mask

                attn_dropout = getattr(self, "attention_dropout", 0.0)
                attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q_pe.dtype)
                attn_weights = torch.nn.functional.dropout(attn_weights, p=attn_dropout, training=self.training)
                attn_output = torch.einsum("bhql,blc->bhqc", attn_weights, compressed_kv)
                attn_output = torch.matmul(attn_output, out_absorb.mT)
                attn_output = attn_output.transpose(1, 2).contiguous()
                attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.v_head_dim)

                if getattr(self, "use_output_gate", False) and hasattr(self, "g_proj"):
                    attn_output = attn_output * self.g_proj(hidden_states).sigmoid()

                return self.o_proj(attn_output)

            attn_mod.forward = types.MethodType(new_forward, attn_mod)

        for name, module in root_module.named_modules():
            if module.__class__.__name__ != "KimiMLAAttention":
                continue
            full_name = f"{root_name}.{name}" if root_name else name
            if not should_inject(full_name):
                continue
            root_module.set_submodule(f"{name}.fa_q", FA3QuantPlaceHolder(ratio=0.9999))
            root_module.set_submodule(f"{name}.fa_k", FA3QuantPlaceHolder(ratio=0.9999))
            root_module.set_submodule(f"{name}.fa_v", FA3QuantPlaceHolder(ratio=1.0))
            _wrap_mla_forward(module)
            get_logger().info("Injected FA3 placeholders for %s", full_name)

    # ===== AttentionAnalysisInterface (attn --metrics mse) =====
    def get_attention_module_cls(self) -> str:
        """Hook ``KimiMLAAttention`` only; KDA ``KimiDeltaAttention`` has no FA3 path."""
        return "KimiMLAAttention"

    def get_attention_output_extractor(
        self,
    ) -> Callable[[Union[tuple, torch.Tensor]], torch.Tensor]:
        """MLA forward returns ``o_proj`` output tensor; tolerate tuple returns."""

        def _extract(attention_forward_output: Union[tuple, torch.Tensor]) -> torch.Tensor:
            if isinstance(attention_forward_output, tuple):
                return attention_forward_output[0]
            return attention_forward_output

        return _extract

    def ascendv1_save_module_preprocess(
        self, prefix: str, module: nn.Module, model: nn.Module
    ) -> Tuple[str, nn.Module]:
        """Pad ``A_log`` back to float-ckpt layout ``[head_dim]`` before write.

        Runtime uses ``[num_heads]`` (FLA); released float weights store ``[head_dim]``
        with a zero tail. Quantized / converted dumps must match the float shape.
        """
        a_log = getattr(module, "A_log", None)
        if not isinstance(a_log, nn.Parameter) or a_log.ndim != 1:
            return prefix, module

        head_dim = getattr(module, "head_dim", None)
        if head_dim is None:
            text_cfg = getattr(self.config, "text_config", self.config)
            lac = getattr(text_cfg, "linear_attn_config", None)
            if isinstance(lac, dict):
                head_dim = lac.get("head_dim")
        if head_dim is None:
            return prefix, module

        target = int(head_dim)
        if a_log.numel() >= target:
            return prefix, module

        padded = a_log.detach().new_zeros(target)
        padded[: a_log.numel()] = a_log.detach()
        module.A_log = nn.Parameter(padded, requires_grad=False)
        get_logger().info(
            "Padded %s.A_log from %s to %s to match float checkpoint layout.",
            prefix,
            tuple(a_log.shape),
            (target,),
        )
        return prefix, module

    def ascendv1_save_postprocess(self, model: nn.Module, save_directory: str) -> None:
        tiktoken_file = os.path.join(self.model_path, "tiktoken.model")
        if os.path.isfile(tiktoken_file):
            dest_file = os.path.join(save_directory, "tiktoken.model")
            safe_copy_file(src_path=tiktoken_file, dest_path=dest_file)
            os.chmod(dest_file, int("600", 8))

    def _get_state_dict(self, module: nn.Module, prefix: str = "") -> Dict[str, torch.Tensor]:
        weight_map = get_full_weight_map(str(self.model_path))
        param_names = [name for name, _ in module.named_parameters()]

        file_groups = defaultdict(list)
        for param_name in param_names:
            full_name = f"{prefix}.{param_name}" if prefix else param_name
            if full_name in weight_map:
                file_groups[weight_map[full_name]].append(param_name)

        state_dict = {}
        for file_name, names in tqdm(file_groups.items(), desc=f"Loading {prefix}", leave=False):
            file_path = get_valid_read_path(
                os.path.join(self.model_path, file_name),
                extensions="safetensors",
                size_max=MAX_READ_FILE_SIZE_32G,
            )
            with safe_open(file_path, framework="pt", device="cpu") as f:
                for param_name in names:
                    full_name = f"{prefix}.{param_name}" if prefix else param_name
                    state_dict[param_name] = f.get_tensor(full_name)
        return state_dict

    def _load_state_dict_compatible(self, module: nn.Module, state_dict: Dict[str, torch.Tensor]) -> None:
        """Load weights while tolerating known ckpt/code shape skew.

        Released Kimi-K3 checkpoints store ``self_attn.A_log`` as ``[head_dim]`` (128),
        while ``KimiDeltaAttention`` (modeling_kimi / modeling_kimi_linear) / FLA
        expect ``[num_heads]`` (96). ``strict=False`` does not ignore size mismatches,
        so we adapt or skip.
        """
        model_sd = module.state_dict()
        adapted: Dict[str, torch.Tensor] = {}
        for name, tensor in state_dict.items():
            if name not in model_sd:
                adapted[name] = tensor
                continue
            target = model_sd[name]
            if tuple(tensor.shape) == tuple(target.shape):
                adapted[name] = tensor
                continue

            # A_log: ckpt [head_dim] vs model [num_heads] — keep FLA-compatible prefix.
            if name.endswith("A_log") and tensor.ndim == 1 and target.ndim == 1:
                if tensor.numel() >= target.numel():
                    adapted[name] = tensor[: target.numel()].contiguous()
                    get_logger().warning(
                        "Adapted %s from ckpt %s to model %s (slice prefix for FLA num_heads).",
                        name,
                        tuple(tensor.shape),
                        tuple(target.shape),
                    )
                    continue

            get_logger().warning(
                "Skip mismatched weight %s: ckpt %s vs model %s",
                name,
                tuple(tensor.shape),
                tuple(target.shape),
            )

        incompatible = module.load_state_dict(adapted, strict=False)
        if incompatible.missing_keys:
            get_logger().debug("Missing keys after compatible load: %d", len(incompatible.missing_keys))
        if incompatible.unexpected_keys:
            get_logger().debug("Unexpected keys after compatible load: %d", len(incompatible.unexpected_keys))

    def _load_decoder_if_not_exist(self, model: nn.Module, name: str, idx: int) -> nn.Module:
        try:
            decoder = model.get_submodule(name)
            try:
                _ = decoder.input_layernorm.weight.device
                get_logger().debug("Layer %d already loaded", idx)
                return decoder
            except RuntimeError:
                pass
        except AttributeError:
            pass

        get_logger().info("Loading decoder layer %d...", idx)
        with patch.object(nn.Linear, "reset_parameters", lambda _self: None), default_dtype(torch.bfloat16):
            module_list: nn.ModuleList = model.language_model.model.layers
            layer_cls = None
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
            state_dict = self._get_state_dict(decoder, prefix=name)
            self._load_state_dict_compatible(decoder, state_dict)
            dequant_subtree_mxfp4_to_bf16(decoder, name, str(self.model_path))
            decoder.eval()

            if len(module_list) <= idx:
                module_list.append(decoder)
            else:
                module_list[idx] = decoder

            get_logger().info("Decoder layer %d loaded successfully", idx)
        return decoder
