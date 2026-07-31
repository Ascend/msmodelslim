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

import gc
import os
import re
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Generator, List, Tuple
from unittest.mock import patch

import torch
from safetensors import safe_open
from torch import nn
from tqdm import tqdm
from transformers import AutoTokenizer
from transformers.models.minimax_m3_vl.modeling_minimax_m3_vl import (
    MiniMaxM3VLDecoderLayer,
    MiniMaxM3VLDenseMLP,
    MiniMaxM3VLRMSNorm,
    MiniMaxM3VLSparseMoeBlock,
)


from msmodelslim.core.base.protocol import ProcessRequest
from msmodelslim.core.const import DeviceType
from msmodelslim.core.graph import AdapterConfig, MappingConfig
from msmodelslim.model.common.layer_wise_forward import generated_decoder_layer_visit_func
from msmodelslim.model.common.vlm_base import VLMBaseModelAdapter
from msmodelslim.model.interface_hub import (
    AscendV1GlobalModelDtypeInterface,
    AscendV1SaveInterface,
    FlexSmoothQuantInterface,
    IterSmoothInterface,
    LayerWiseOffloadOptionalInterface,
    ModelInfoInterface,
    ModelSlimPipelineInterfaceV1,
)
from msmodelslim.processor.quarot import QuaRotInterface
from msmodelslim.utils.exception import InvalidModelError
from msmodelslim.utils.logging import get_logger, logger_setter
from msmodelslim.utils.security import (
    MAX_READ_FILE_SIZE_32G,
    get_valid_read_path,
    json_safe_load,
)

from .moe_utils import UnstackedM3MoeBlock, UnstackedM3DenseMLP


class _StandardRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


def _device_name(device: DeviceType) -> str:
    return "npu" if device == DeviceType.NPU else "cpu"


@logger_setter()
class MiniMaxM3ModelAdapter(  # pylint: disable=too-many-ancestors
    VLMBaseModelAdapter,
    ModelInfoInterface,
    ModelSlimPipelineInterfaceV1,
    IterSmoothInterface,
    FlexSmoothQuantInterface,
    QuaRotInterface,
    LayerWiseOffloadOptionalInterface,
    AscendV1SaveInterface,
    AscendV1GlobalModelDtypeInterface,
):
    def __init__(self, model_type: str, model_path: Path, trust_remote_code: bool = False):
        self._tokenizer = None
        super().__init__(model_type, model_path, trust_remote_code)
        if getattr(self.config, "model_type", None) != "minimax_m3_vl":
            raise InvalidModelError(
                f"MiniMaxM3ModelAdapter expects model_type minimax_m3_vl, got {getattr(self.config, 'model_type', None)}",
                action="Please check the model path or use another adapter.",
            )

    def get_model_pedigree(self) -> str:
        return "minimax_m3"

    def get_model_type(self) -> str:
        return self.model_type

    def get_layer_wise_offload_device(self):
        return "meta"

    @lru_cache(maxsize=1)
    def _get_weight_map(self) -> Dict[str, str]:
        index_path = os.path.join(self.model_path, "model.safetensors.index.json")
        index_data = json_safe_load(index_path)
        return index_data["weight_map"]

    def _load_raw_weights_for_layer(self, idx: int) -> Dict[str, torch.Tensor]:
        weight_map = self._get_weight_map()
        ckpt_prefix = f"language_model.model.layers.{idx}"
        file_groups = defaultdict(list)
        for ckpt_name, shard_file in weight_map.items():
            if ckpt_name.startswith(ckpt_prefix + "."):
                file_groups[shard_file].append(ckpt_name)
        raw_weights = {}
        for file_name, ckpt_names in tqdm(file_groups.items(), desc=f"Loading layer {idx}", leave=False):
            file_path = os.path.join(self.model_path, file_name)
            file_path = get_valid_read_path(file_path, extensions="safetensors", size_max=MAX_READ_FILE_SIZE_32G)
            with safe_open(file_path, framework="pt", device="cpu") as f:
                for ckpt_name in ckpt_names:
                    raw_weights[ckpt_name] = f.get_tensor(ckpt_name)
        return raw_weights

    @staticmethod
    def _remap_layer_weights(raw_weights: Dict[str, torch.Tensor], idx: int) -> Dict[str, torch.Tensor]:
        ckpt_prefix = f"language_model.model.layers.{idx}"
        state_dict = {}
        gate_proj_parts = {}
        up_proj_parts = {}
        expert_gate_parts = {}
        expert_up_parts = {}
        expert_down_parts = {}

        for ckpt_name, tensor in raw_weights.items():
            rel = ckpt_name[len(ckpt_prefix) + 1 :]

            if rel.startswith("block_sparse_moe."):
                mlp_rel = "mlp." + rel[len("block_sparse_moe.") :]

                if mlp_rel.startswith("mlp.shared_experts.gate_proj.weight"):
                    gate_proj_parts[mlp_rel] = tensor
                    continue
                if mlp_rel.startswith("mlp.shared_experts.up_proj.weight"):
                    up_proj_parts[mlp_rel] = tensor
                    continue
                if mlp_rel.startswith("mlp.experts."):
                    parts = mlp_rel.split(".")
                    expert_idx = int(parts[2])
                    if parts[3] == "w1" and parts[4] == "weight":
                        expert_gate_parts[expert_idx] = tensor
                        continue
                    if parts[3] == "w3" and parts[4] == "weight":
                        expert_up_parts[expert_idx] = tensor
                        continue
                    if parts[3] == "w2" and parts[4] == "weight":
                        expert_down_parts[expert_idx] = tensor
                        continue

                if mlp_rel == "mlp.e_score_correction_bias":
                    state_dict["mlp.gate.e_score_correction_bias"] = tensor
                    continue

                state_dict[mlp_rel] = tensor
                continue

            if rel.startswith("mlp.gate_proj.weight"):
                gate_proj_parts[rel] = tensor
                continue
            if rel.startswith("mlp.up_proj.weight"):
                up_proj_parts[rel] = tensor
                continue

            # Indexer key remapping: checkpoint 用 index_q_proj, 模型用 indexer.q_proj
            if rel.startswith("self_attn.index_q_proj."):
                rel = "self_attn.indexer.q_proj." + rel[len("self_attn.index_q_proj.") :]
            elif rel.startswith("self_attn.index_k_proj."):
                rel = "self_attn.indexer.k_proj." + rel[len("self_attn.index_k_proj.") :]
            elif rel.startswith("self_attn.index_q_norm."):
                rel = "self_attn.indexer.q_norm." + rel[len("self_attn.index_q_norm.") :]
            elif rel.startswith("self_attn.index_k_norm."):
                rel = "self_attn.indexer.k_norm." + rel[len("self_attn.index_k_norm.") :]

            state_dict[rel] = tensor

        for key in sorted(gate_proj_parts.keys()):
            base = key.replace("gate_proj.weight", "gate_up_proj.weight")
            gate_w = gate_proj_parts[key]
            up_key = key.replace("gate_proj.weight", "up_proj.weight")
            up_w = up_proj_parts.get(up_key)
            if up_w is not None:
                state_dict[base] = torch.cat([gate_w, up_w], dim=0)
            else:
                state_dict[base] = gate_w

        if expert_gate_parts and expert_up_parts:
            num_experts = max(expert_gate_parts.keys()) + 1
            inter_size = expert_gate_parts[0].shape[0]
            hidden_size = expert_gate_parts[0].shape[1]
            gate_up_3d = torch.empty(num_experts, 2 * inter_size, hidden_size, dtype=expert_gate_parts[0].dtype)
            for i in range(num_experts):
                gate_up_3d[i, :inter_size, :] = expert_gate_parts[i]
                gate_up_3d[i, inter_size:, :] = expert_up_parts[i]
            state_dict["mlp.experts.gate_up_proj"] = gate_up_3d

            down_3d = torch.empty(num_experts, hidden_size, inter_size, dtype=expert_down_parts[0].dtype)
            for i in range(num_experts):
                down_3d[i] = expert_down_parts[i]
            state_dict["mlp.experts.down_proj"] = down_3d

        return state_dict

    def _load_decoder_if_not_exist(self, model: nn.Module, name: str, idx: int) -> nn.Module:
        try:
            decoder = model.get_submodule(name)
            try:
                _ = decoder.input_layernorm.weight.device
                get_logger().debug("Layer %s already loaded", idx)
                return decoder
            except RuntimeError:
                pass
        except AttributeError:
            pass

        get_logger().debug("Loading MiniMax-M3 decoder layer %s...", idx)

        with patch.object(nn.Linear, "reset_parameters", lambda _self: None):
            decoder = MiniMaxM3VLDecoderLayer(self.config.text_config, layer_idx=idx)

            raw_weights = self._load_raw_weights_for_layer(idx)
            state_dict = self._remap_layer_weights(raw_weights, idx)

            missing, unexpected = decoder.load_state_dict(state_dict, strict=False)
            if unexpected:
                get_logger().warning("Unexpected keys when loading layer %s: %s", idx, unexpected)
            if missing:
                get_logger().debug("Missing keys when loading layer %s: %s", idx, missing)

            decoder.eval()

            module_list: nn.ModuleList = model.model.language_model.layers
            if len(module_list) <= idx:
                module_list.append(decoder)
            else:
                module_list[idx] = decoder

            get_logger().debug("Decoder layer %s loaded successfully", idx)

        self._postprocess_decoder_layer(decoder, idx)
        return decoder

    def _convert_single_moe_layer(self, layer: nn.Module, layer_idx: int):
        original_moe_block = layer.mlp
        if not isinstance(original_moe_block, MiniMaxM3VLSparseMoeBlock):
            get_logger().warning(
                "Layer %s MLP is not a MiniMaxM3VLSparseMoeBlock, skipping conversion. Got: %s",
                layer_idx,
                type(original_moe_block),
            )
            return

        unstacked_moe_block = UnstackedM3MoeBlock(self.config.text_config, original_moe_block, copy_weights=False)
        unstacked_moe_block._transform_weights_from_original(original_moe_block, in_place=True)
        unstacked_moe_block.eval()
        layer.mlp = unstacked_moe_block

        del original_moe_block
        gc.collect()

    def _postprocess_decoder_layer(self, decoder: nn.Module, layer_idx: int) -> None:
        """统一处理 decoder layer 的后处理：RMSNorm 替换、MoE/dense 转换、e_score_correction_bias 转 parameter"""
        # 1. RMSNorm 替换（MiniMaxM3VLRMSNorm → _StandardRMSNorm +1）
        for sub_name, sub_module in decoder.named_modules():
            if isinstance(sub_module, MiniMaxM3VLRMSNorm):
                new_module = _StandardRMSNorm(sub_module.weight.shape[0], sub_module.eps)
                new_module.weight.data = sub_module.weight.data.float() + 1
                decoder.set_submodule(sub_name, new_module)

        # 2. MoE / dense 转换
        if self._is_moe_layer(layer_idx):
            get_logger().debug("Layer %s is a MoE layer, performing architecture adaptation...", layer_idx)
            self._convert_single_moe_layer(decoder, layer_idx)
            get_logger().debug("Layer %s architecture adaptation completed", layer_idx)
        else:
            get_logger().debug("Layer %s is a dense layer, performing architecture adaptation...", layer_idx)
            self._convert_single_dense_layer(decoder, layer_idx)
            get_logger().debug("Layer %s architecture adaptation completed", layer_idx)

        # 3. e_score_correction_bias: UnstackedM3MoeBlock 中为 buffer，转为 parameter 才能导出
        if hasattr(decoder, 'mlp') and hasattr(decoder.mlp, 'e_score_correction_bias'):
            buf = decoder.mlp.e_score_correction_bias
            if isinstance(buf, torch.Tensor) and not isinstance(buf, nn.Parameter):
                del decoder.mlp.e_score_correction_bias
                decoder.mlp.register_parameter('e_score_correction_bias', nn.Parameter(buf.detach()))

    def _convert_single_dense_layer(self, layer: nn.Module, layer_idx: int):
        original_mlp = layer.mlp
        if not isinstance(original_mlp, MiniMaxM3VLDenseMLP):
            get_logger().warning(
                "Layer %s MLP is not a MiniMaxM3VLDenseMLP, skipping conversion. Got: %s",
                layer_idx,
                type(original_mlp),
            )
            return

        text_config = self.config.text_config
        dense_inter = getattr(text_config, "dense_intermediate_size", text_config.intermediate_size)
        unstacked = UnstackedM3DenseMLP(
            text_config.hidden_size,
            dense_inter,
            getattr(text_config, "swiglu_alpha", 1.702),
            getattr(text_config, "swiglu_limit", 7.0),
        )
        with torch.no_grad():
            gate_up_weight = original_mlp.gate_up_proj.weight.data.cpu()
            unstacked.gate_proj.weight = nn.Parameter(gate_up_weight[:dense_inter, :].contiguous(), requires_grad=False)
            unstacked.up_proj.weight = nn.Parameter(gate_up_weight[dense_inter:, :].contiguous(), requires_grad=False)
            unstacked.down_proj.weight = nn.Parameter(
                original_mlp.down_proj.weight.data.cpu().contiguous(), requires_grad=False
            )
        unstacked.eval()
        layer.mlp = unstacked

        del original_mlp
        gc.collect()

    def _is_moe_layer(self, layer_idx: int) -> bool:
        text_config = self.config.text_config
        mlp_layer_types = getattr(text_config, "mlp_layer_types", None)
        if isinstance(mlp_layer_types, list) and layer_idx < len(mlp_layer_types):
            return mlp_layer_types[layer_idx] == "sparse"
        moe_freq = getattr(text_config, "moe_layer_freq", None)
        if isinstance(moe_freq, list) and layer_idx < len(moe_freq):
            return bool(moe_freq[layer_idx])
        return layer_idx >= getattr(text_config, "first_k_dense_replace", 0)

    def _get_num_experts(self) -> int:
        text_config = self.config.text_config
        return getattr(text_config, "num_local_experts", getattr(text_config, "n_routed_experts", 128))

    def init_model(self, device: DeviceType = DeviceType.NPU) -> nn.Module:
        try:
            from transformers import MiniMaxM3SparseForConditionalGeneration
            from transformers.models.minimax_m3_vl.configuration_minimax_m3_vl import (
                MiniMaxM3VLVisionConfig,
                MiniMaxM3VLTextConfig,
            )
        except ImportError as e:
            raise InvalidModelError(
                "Failed to import MiniMaxM3SparseForConditionalGeneration. "
                "Please install transformers with MiniMax-M3 support.",
                action="pip install transformers with minimax_m3_vl support",
            ) from e

        get_logger().info("Initializing MiniMax-M3 model with v1 framework (layer-wise loading)...")

        if not isinstance(self.config.vision_config, MiniMaxM3VLVisionConfig):
            get_logger().info(
                "Converting vision_config from %s to MiniMaxM3VLVisionConfig", type(self.config.vision_config)
            )
            if isinstance(self.config.vision_config, dict):
                self.config.vision_config = MiniMaxM3VLVisionConfig(**self.config.vision_config)
            else:
                self.config.vision_config = MiniMaxM3VLVisionConfig()

        if not isinstance(self.config.text_config, MiniMaxM3VLTextConfig):
            get_logger().info("Converting text_config from %s to MiniMaxM3VLTextConfig", type(self.config.text_config))
            if isinstance(self.config.text_config, dict):
                self.config.text_config = MiniMaxM3VLTextConfig(**self.config.text_config)
            elif hasattr(self.config.text_config, "to_dict"):
                self.config.text_config = MiniMaxM3VLTextConfig(**self.config.text_config.to_dict())
            else:
                self.config.text_config = MiniMaxM3VLTextConfig()

        mlp_layer_types = getattr(self.config.text_config, "mlp_layer_types", None)
        if mlp_layer_types is not None and "sparse" in mlp_layer_types:
            config_path = os.path.join(self.model_path, "config.json")
            raw = json_safe_load(config_path)
            raw_text = raw.get("text_config", {})
            moe_freq = raw_text.get("moe_layer_freq")
            if moe_freq and len(moe_freq) == self.config.text_config.num_hidden_layers:
                self.config.text_config.mlp_layer_types = ["sparse" if f else "dense" for f in moe_freq]

        if not hasattr(self.config, "merged_hidden_size"):
            self.config.merged_hidden_size = self.config.text_config.hidden_size * (
                self.config.vision_config.spatial_merge_size**2
            )
            get_logger().info("Set config.merged_hidden_size = %s", self.config.merged_hidden_size)

        origin_layers = self.config.text_config.num_hidden_layers
        get_logger().info("Model with %s text layers", origin_layers)

        self.config.text_config.num_hidden_layers = 1
        self.config.use_cache = False

        self.model_path = get_valid_read_path(str(self.model_path), is_dir=True, check_user_stat=True)

        get_logger().info("Loading vision encoder and first text decoder layer...")
        model = MiniMaxM3SparseForConditionalGeneration.from_pretrained(  # nosec
            self.model_path,
            config=self.config,
            trust_remote_code=self.trust_remote_code,
            torch_dtype="auto",
            local_files_only=True,
            device_map="cpu",
            attn_implementation="eager",
        ).eval()

        self.config.text_config.num_hidden_layers = origin_layers
        self.config.text_config._attn_implementation = "eager"

        if hasattr(model.config.text_config, "num_attention_heads"):
            model.config.num_attention_heads = model.config.text_config.num_attention_heads
            get_logger().info("Set model.config.num_attention_heads = %s", model.config.num_attention_heads)
        if hasattr(model.config.text_config, "num_key_value_heads"):
            model.config.num_key_value_heads = model.config.text_config.num_key_value_heads
            get_logger().info("Set model.config.num_key_value_heads = %s", model.config.num_key_value_heads)

        get_logger().info("Model initialized with %s layers (1 loaded, others will be loaded on-demand)", origin_layers)

        decoder_layer_0 = model.model.language_model.layers[0]
        self._postprocess_decoder_layer(decoder_layer_0, 0)

        return model

    def generate_decoder_layer(self, model: nn.Module) -> Generator[Tuple[str, nn.Module], None, None]:
        num_layers = self.config.text_config.num_hidden_layers
        for layer_idx in range(num_layers):
            name = f"model.language_model.layers.{layer_idx}"
            layer = self._load_decoder_if_not_exist(model, name, layer_idx)
            yield name, layer

    def generate_model_visit(self, model: nn.Module) -> Generator[ProcessRequest, Any, None]:
        get_logger().info("Processing vision encoder...")
        yield ProcessRequest(
            name="vision_tower",
            module=model.model.vision_tower,
            args=(),
            kwargs={},
        )

        get_logger().info("Processing text decoder layers...")
        yield from generated_decoder_layer_visit_func(
            model,
            transformer_blocks=self.generate_decoder_layer(model),
        )

    def generate_model_forward(self, model: nn.Module, inputs: Any) -> Generator[ProcessRequest, Any, None]:
        sample = inputs[0] if isinstance(inputs, list) else inputs
        input_ids = sample["input_ids"]
        attention_mask = sample.get("attention_mask")

        pixel_values = sample.get("pixel_values")
        image_grid_thw = sample.get("image_grid_thw")

        # Step 1: Always yield vision_tower first（参考qwen3_vl_moe模式）
        has_images = pixel_values is not None and image_grid_thw is not None
        image_embeds = yield ProcessRequest(
            name="vision_tower",
            module=model.model.vision_tower,
            args=(pixel_values, image_grid_thw) if has_images else (),
            kwargs={},
        )

        # Step 2: embed_tokens 直接内联调用（不 yield），匹配 visit 无 embed_tokens
        inputs_embeds = model.model.language_model.embed_tokens(input_ids)

        # Step 3: 有图片时融合视觉嵌入
        if has_images:
            if isinstance(image_embeds, (list, tuple)):
                image_embeds_cat = torch.cat(image_embeds, dim=0)
            else:
                image_embeds_cat = image_embeds
            image_mask = (input_ids == model.config.image_token_id).unsqueeze(-1).expand_as(inputs_embeds)
            inputs_embeds = inputs_embeds.masked_scatter(
                image_mask, image_embeds_cat.to(inputs_embeds.device, inputs_embeds.dtype)
            )

        from transformers.masking_utils import create_causal_mask

        cache_position = torch.arange(0, inputs_embeds.shape[1], device=inputs_embeds.device)
        position_ids = cache_position.unsqueeze(0)

        causal_mask = create_causal_mask(
            config=model.config.text_config,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=None,
            position_ids=position_ids,
        )

        position_embeddings = model.model.language_model.rotary_emb(inputs_embeds, position_ids)

        hidden_states = inputs_embeds
        for name, layer in self.generate_decoder_layer(model):
            hidden_states = yield ProcessRequest(
                name=name,
                module=layer,
                args=(hidden_states,),
                kwargs={
                    "attention_mask": causal_mask,
                    "position_ids": position_ids,
                    "cache_position": cache_position,
                    "position_embeddings": position_embeddings,
                    "past_key_values": None,
                    "use_cache": False,
                },
            )

        hidden_states = model.model.language_model.norm(hidden_states)

    def enable_kv_cache(self, model: nn.Module, need_kv_cache: bool) -> None:
        model.config.use_cache = need_kv_cache
        get_logger().info("KV cache %s", "enabled" if need_kv_cache else "disabled")

    def handle_dataset(self, dataset: Any, device: DeviceType = DeviceType.NPU) -> List[Any]:
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(  # nosec
                self.model_path,
                trust_remote_code=self.trust_remote_code,
                local_files_only=True,
                use_fast=False,
                legacy=False,
            )
        processed = []
        for text in dataset:
            if isinstance(text, dict):
                text = text.get("text") or text.get("prompt") or text.get("inputs_pretokenized") or str(text)
            inputs = self._tokenizer(str(text), return_tensors="pt").to(_device_name(device))
            processed.append(
                {
                    "input_ids": inputs["input_ids"],
                    "attention_mask": inputs["attention_mask"],
                }
            )
        return processed

    @staticmethod
    def _load_output_weight_map(save_directory: str) -> Tuple[Dict[str, Any], str]:
        index_path = os.path.join(save_directory, "quant_model_weights.safetensors.index.json")
        if os.path.exists(index_path):
            return json_safe_load(index_path), index_path

        single_file = os.path.join(save_directory, "quant_model_weights.safetensors")
        if not os.path.exists(single_file):
            return {"metadata": {"total_size": 0}, "weight_map": {}}, index_path

        with safe_open(single_file, framework="pt", device="cpu") as handle:
            weight_map = {key: os.path.basename(single_file) for key in handle.keys()}
        total_size = os.path.getsize(single_file)
        return {"metadata": {"total_size": total_size}, "weight_map": weight_map}, index_path

    def ascendv1_save_module_preprocess(
        self, prefix: str, module: nn.Module, model: nn.Module
    ) -> Tuple[str, nn.Module]:
        # RMSNorm / RMSNormBias（iter_smooth 替换后）→ _StandardRMSNorm，再走下方 -1 统一保存
        if type(module).__name__ in ('RMSNorm', 'RMSNormBias'):
            get_logger().debug(
                "[SavePreprocess] %s → _StandardRMSNorm: %s  w.mean=%.4f",
                type(module).__name__,
                prefix,
                module.weight.data.float().mean().item(),
            )
            orig_w = module.weight.data.float()
            module = _StandardRMSNorm(orig_w.shape[0], getattr(module, 'variance_epsilon', 1e-6))
            module.weight.data = orig_w
        if isinstance(module, _StandardRMSNorm):
            w_before = module.weight.data.float().mean().item()
            new_module = MiniMaxM3VLRMSNorm(module.weight.shape[0], module.variance_epsilon)
            new_module.weight.data = module.weight.data.float().sub(1).to(module.weight.dtype)
            w_after = new_module.weight.data.float().mean().item()
            get_logger().debug(
                "[SavePreprocess] _StandardRMSNorm → MiniMaxM3VLRMSNorm -1: %s  w.mean=%.4f → %.4f",
                prefix,
                w_before,
                w_after,
            )
            module = new_module

        # 再应用命名规则（仅影响保存路径，不影响模型结构）

        # 1. indexer 展平：self_attn.indexer.q_proj → self_attn.index_q_proj
        prefix = re.sub(
            r'\.indexer\.(q_proj|k_proj|q_norm|k_norm)([\.\s]|$)',
            r'.index_\1\2',
            prefix,
        )

        # 2. lm_head 补前缀：lm_head → language_model.lm_head
        if prefix == 'lm_head':
            prefix = 'language_model.lm_head'

        # 3. model.language_model.* → language_model.model.*
        prefix = re.sub(r'^model\.language_model\.', 'language_model.model.', prefix)

        # 4. vision_tower.* → 加上 vision_model 嵌套层（prefix 可能含或不含后续后缀）
        if prefix.startswith('vision_tower.layers'):
            prefix = 'vision_tower.vision_model.encoder.layers' + prefix[len('vision_tower.layers') :]
        elif prefix.startswith('vision_tower.embeddings.proj'):
            prefix = (
                'vision_tower.vision_model.embeddings.patch_embedding' + prefix[len('vision_tower.embeddings.proj') :]
            )
        elif prefix.startswith('vision_tower.pre_layrnorm'):
            prefix = 'vision_tower.vision_model.pre_layrnorm' + prefix[len('vision_tower.pre_layrnorm') :]

        # 5. model.vision_tower.* → vision_tower.*（兜底，处理带有 model. 前缀的情况）
        if prefix.startswith('model.vision_tower.'):
            prefix = 'vision_tower.' + prefix[len('model.vision_tower.') :]

        # 6. multi_modal_projector 下的 merge_linear → patch_merge_mlp.linear
        #    prefix 可能带 model. 也可能不带，分别处理
        if prefix.startswith('multi_modal_projector.merge_linear_1') or prefix.startswith(
            'model.multi_modal_projector.merge_linear_1'
        ):
            get_logger().debug("Renaming projector merge_linear_1: %s", prefix)
            if prefix.startswith('model.'):
                prefix = 'patch_merge_mlp.linear_1' + prefix[len('model.multi_modal_projector.merge_linear_1') :]
            else:
                prefix = 'patch_merge_mlp.linear_1' + prefix[len('multi_modal_projector.merge_linear_1') :]
        elif prefix.startswith('multi_modal_projector.merge_linear_2') or prefix.startswith(
            'model.multi_modal_projector.merge_linear_2'
        ):
            if prefix.startswith('model.'):
                prefix = 'patch_merge_mlp.linear_2' + prefix[len('model.multi_modal_projector.merge_linear_2') :]
            else:
                prefix = 'patch_merge_mlp.linear_2' + prefix[len('multi_modal_projector.merge_linear_2') :]
        elif prefix.startswith('model.multi_modal_projector.'):
            prefix = 'multi_modal_projector.' + prefix[len('model.multi_modal_projector.') :]

        # 7. MoE layer 重命名：mlp → block_sparse_moe，专家 gate_proj/up_proj/down_proj → w1/w3/w2
        moe_match = re.match(r'^language_model\.model\.layers\.(\d+)\.mlp(?:\.(.+))?$', prefix)
        if moe_match:
            layer_idx = int(moe_match.group(1))
            if self._is_moe_layer(layer_idx):
                rest = moe_match.group(2)
                if rest is not None:
                    # 单个 expert 权重重命名：gate_proj/up_proj/down_proj → w1/w3/w2
                    rest = re.sub(r'\bexperts\.\d+\.gate_proj\b', lambda m: m.group(0).replace('gate_proj', 'w1'), rest)
                    rest = re.sub(r'\bexperts\.\d+\.up_proj\b', lambda m: m.group(0).replace('up_proj', 'w3'), rest)
                    rest = re.sub(r'\bexperts\.\d+\.down_proj\b', lambda m: m.group(0).replace('down_proj', 'w2'), rest)
                # mlp → block_sparse_moe
                prefix = f'language_model.model.layers.{layer_idx}.block_sparse_moe'
                prefix = f'{prefix}.{rest}' if rest else prefix

        return prefix, module

    def ascendv1_save_postprocess(self, model: nn.Module, save_directory: str) -> None:
        del model

    def get_adapter_config_for_subgraph(self) -> List[AdapterConfig]:
        adapter_config: List[AdapterConfig] = []
        text_config = self.config.text_config
        num_experts = self._get_num_experts()
        for layer_idx in range(text_config.num_hidden_layers):
            # 判断当前层是否有 sparse attention（indexer 仅存于 sparse 层）
            is_sparse = self._is_sparse_layer(layer_idx)
            attn_targets = [
                f"model.language_model.layers.{layer_idx}.self_attn.q_proj",
                f"model.language_model.layers.{layer_idx}.self_attn.k_proj",
                f"model.language_model.layers.{layer_idx}.self_attn.v_proj",
            ]
            if is_sparse:
                indexer_prefix = f"model.language_model.layers.{layer_idx}.self_attn.indexer"
                attn_targets.extend(
                    [
                        f"{indexer_prefix}.q_proj",
                        f"{indexer_prefix}.k_proj",
                    ]
                )
            norm_linear_attn = AdapterConfig(
                subgraph_type="norm-linear",
                mapping=MappingConfig(
                    source=f"model.language_model.layers.{layer_idx}.input_layernorm",
                    targets=attn_targets,
                ),
            )
            ov = AdapterConfig(
                subgraph_type="ov",
                mapping=MappingConfig(
                    source=f"model.language_model.layers.{layer_idx}.self_attn.v_proj",
                    targets=[f"model.language_model.layers.{layer_idx}.self_attn.o_proj"],
                ),
                extra_config={"group_method": "max"},
            )
            adapter_config.extend([norm_linear_attn, ov])

            if self._is_moe_layer(layer_idx):
                mlp_targets = []
                for i in range(num_experts):
                    mlp_targets.append(f"model.language_model.layers.{layer_idx}.mlp.experts.{i}.gate_proj")
                    mlp_targets.append(f"model.language_model.layers.{layer_idx}.mlp.experts.{i}.up_proj")
                mlp_targets.append(f"model.language_model.layers.{layer_idx}.mlp.shared_experts.gate_proj")
                mlp_targets.append(f"model.language_model.layers.{layer_idx}.mlp.shared_experts.up_proj")
                mlp_targets.append(f"model.language_model.layers.{layer_idx}.mlp.gate")
                norm_linear_mlp = AdapterConfig(
                    subgraph_type="norm-linear",
                    mapping=MappingConfig(
                        source=f"model.language_model.layers.{layer_idx}.post_attention_layernorm",
                        targets=mlp_targets,
                    ),
                )
                adapter_config.append(norm_linear_mlp)
            else:
                norm_linear_mlp = AdapterConfig(
                    subgraph_type="norm-linear",
                    mapping=MappingConfig(
                        source=f"model.language_model.layers.{layer_idx}.post_attention_layernorm",
                        targets=[
                            f"model.language_model.layers.{layer_idx}.mlp.gate_proj",
                            f"model.language_model.layers.{layer_idx}.mlp.up_proj",
                        ],
                    ),
                )
                up_down = AdapterConfig(
                    subgraph_type="up-down",
                    mapping=MappingConfig(
                        source=f"model.language_model.layers.{layer_idx}.mlp.up_proj",
                        targets=[f"model.language_model.layers.{layer_idx}.mlp.down_proj"],
                    ),
                )
                adapter_config.extend([norm_linear_mlp, up_down])

        return adapter_config

    def _is_sparse_layer(self, layer_idx):
        """从原始 JSON 读取 sparse_attention_freq（MiniMaxM3VLTextConfig 会丢弃该字段）。"""
        try:
            raw = json_safe_load(os.path.join(self.model_path, "config.json"))
            sac = raw.get("text_config", {}).get("sparse_attention_config", {})
            freq = sac.get("sparse_attention_freq", [])
            return bool(freq and layer_idx < len(freq) and freq[layer_idx])
        except Exception:
            return False
