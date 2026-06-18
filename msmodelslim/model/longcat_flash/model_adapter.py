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

from copy import deepcopy
import os.path
from collections import defaultdict
from typing import Any, List, Dict, Generator, Optional, Tuple
from functools import lru_cache
from contextlib import contextmanager
from unittest.mock import patch

import torch
from safetensors import safe_open
from torch import nn, distributed as dist
from tqdm import tqdm

from msmodelslim.app.naive_quantization.model_info_interface import ModelInfoInterface
from msmodelslim.core.base.protocol import ProcessRequest
from msmodelslim.core.const import DeviceType
from msmodelslim.core.graph import AdapterConfig, MappingConfig, FusionConfig
from msmodelslim.utils.exception import InvalidModelError
from msmodelslim.utils.security.model import SafeGenerator
from msmodelslim.utils.security import (
    json_safe_load,
    get_valid_read_path,
    MAX_READ_FILE_SIZE_32G,
)
from msmodelslim.model.common.layer_wise_forward import (
    generated_decoder_layer_visit_func,
    TransformersForwardBreak,
)

from ..common.transformers import TransformersModel
from ..interface_hub import (
    ModelSlimPipelineInterfaceV1,
    FlexSmoothQuantInterface,
    IterSmoothInterface,
)
from msmodelslim.core.quant_service.modelslim_v1.save.interface import (
    AscendV1SaveInterface,
)
from msmodelslim.utils.logging import logger_setter, get_logger
from .longcat_flash_mtp import LongCatMultiTokenPredictor

_logger = get_logger()


@contextmanager
def default_dtype(dtype):
    """
    Context manager for setting default dtype temporarily.
    """
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(old_dtype)


@logger_setter("msmodelslim.model.longcat_flash")
# pylint: disable=too-many-ancestors
class LongCatFlashModelAdapter(
    TransformersModel,
    ModelInfoInterface,
    ModelSlimPipelineInterfaceV1,
    FlexSmoothQuantInterface,
    IterSmoothInterface,
    AscendV1SaveInterface,
):
    """
    Model Adapter for LongCat-Flash quantization.
    LongCat-Flash architecture:
    - 560B MoE with 512 experts (256 are zero/identity experts)
    - Dual sub-layer: each layer has 2 Input LN + 2 attention + 2 mlps + 1 MoE shortcut + 2 Post LN
    - MLA (Multi-head Latent Attention)
    - num_layers=28 physical layers, num_hidden_layers=56 (2x for cache)
    - 1 MTP (Multi-Token Predictor) module at the end for next token prediction

    Uses pipeline layer-by-layer loading for memory efficiency.
    """

    def get_model_pedigree(self) -> str:
        return "longcat_flash"

    def get_model_type(self) -> str:
        return self.model_type

    def handle_dataset(self, dataset: Any, device: DeviceType = DeviceType.NPU) -> List[Any]:
        return self._get_tokenized_data(dataset, device=device)

    def _is_longcat_router(self, module: nn.Module) -> bool:
        return module.__class__.__name__ in {
            "LongcatFlashTopkRouter",
            "LongcatFlashTokenTopkRouter",
        }

    def _preserve_router_fp32(self, module: nn.Module) -> None:
        for _, router in module.named_modules():
            if not self._is_longcat_router(router):
                continue

            router.to(dtype=torch.float32)

    def init_model(self, device: DeviceType = DeviceType.NPU) -> nn.Module:
        with default_dtype(torch.bfloat16):
            # Validate config has num_layers
            if not hasattr(self.config, "num_layers"):
                raise InvalidModelError(
                    "Config missing 'num_layers' attribute",
                    action="Please ensure the model config is valid for LongCat-Flash",
                )

            # Store original layer count
            origin_layers = self.config.num_layers
            if _logger:
                _logger.info("Model with %s layers totally", origin_layers)

            # Temporarily set to 1 layer for initial loading
            self.config.num_layers = 1

            self.config.use_cache = False  # Disable cache for initial loading to save memory
            if getattr(self.config, "_attn_implementation", None) in (None, "eager"):
                self.config._attn_implementation = "sdpa"

            # Load model with only one layer (template)
            model = SafeGenerator.get_model_from_pretrained(
                model_path=str(self.model_path),
                config=self.config,
                trust_remote_code=self.trust_remote_code,
                device_map="cpu",
                torch_dtype="auto",
            )

            self._preserve_router_fp32(model)

            # Restore original layer count
            self.config.num_layers = origin_layers + 1  # +1 for MTP module

            # Load weights for the first layer
            state_dict = self._get_state_dict(model)
            model.load_state_dict(state_dict)

            model.eval()
            if _logger:
                _logger.info(
                    "Created model template with 1/%s layers. Additional layers will be loaded on-demand.",
                    self.config.num_layers,
                )
            return model

    def generate_model_visit(self, model: nn.Module) -> Generator[ProcessRequest, Any, None]:
        """Generate layer-wise visits for quantization pipeline."""
        return generated_decoder_layer_visit_func(
            model,
            transformer_blocks=self._generate_decoder_and_mtp_layer(model),
        )

    def _generate_decoder_and_mtp_layer(self, model: nn.Module) -> Generator[Tuple[str, nn.Module], None, None]:
        """
        Generator that yields each decoder layer one by one.

        Loads layers on-demand to save memory for 560B model.
        Each LongCat-Flash layer contains:
        - self_attn[0], self_attn[1] (dual attention)
        - mlps[0], mlps[1] (dual dense MLP)
        - mlp (MoE with shortcuts)
        - input_layernorm[0], input_layernorm[1]
        - post_attention_layernorm[0], post_attention_layernorm[1]
        """
        for idx in range(self.config.num_layers):
            if idx < self.config.num_layers - 1:
                name = f"model.layers.{idx}"
                decoder = self._load_decoder_if_not_exist(model, name=name, idx=idx)
                yield name, decoder
            else:
                # Load MTP module at the end
                name = "model.mtp"
                mtp = self._load_mtp_if_not_exist(model)
                yield name, mtp

    def _load_decoder_if_not_exist(self, model: nn.Module, name: str, idx: int):
        """
        Load a decoder layer if not already loaded.

        Uses lazy loading pattern:
        1. Check if layer already exists
        2. If not, create layer structure from template
        3. Load weights from safetensors
        4. Append to model's layer list
        """
        try:
            decoder = model.get_submodule(name)
        except AttributeError:
            # Disable reset_parameters for faster loading
            # Weight initialization is unnecessary since we load from state_dict
            with (
                patch.object(nn.Linear, "reset_parameters", lambda self: None),
                default_dtype(torch.bfloat16),
            ):
                if _logger:
                    _logger.info("Creating decoder layer %s", idx)

                module_list: nn.ModuleList = model.model.layers
                template_module = module_list[0]

                # Create new layer using template's class
                decoder = template_module.__class__(config=self.config, layer_idx=idx)

                self._preserve_router_fp32(decoder)

                # Load weights for this specific layer
                state_dict = self._get_state_dict(decoder, prefix=name)
                decoder.load_state_dict(state_dict)

                decoder.eval()
                module_list.append(decoder)

                if _logger:
                    _logger.info("Created decoder layer %s successfully", idx)

        return decoder

    def _load_mtp_if_not_exist(self, model: nn.Module):
        """
        The MTP module is not loaded by transformers' from_pretrained,
        so we need load it manually.
        """

        try:
            mtp = model.get_submodule("model.mtp")
        except AttributeError:
            # Disable reset_parameters for faster loading
            with (
                patch.object(nn.Linear, "reset_parameters", lambda self: None),
                default_dtype(torch.bfloat16),
            ):
                if _logger:
                    _logger.info("Creating MTP module")

                mtp = LongCatMultiTokenPredictor(self.config)
                self._preserve_router_fp32(mtp)

                # Load weights for MTP module
                state_dict = self._get_state_dict(mtp, prefix="model.mtp")
                mtp.load_state_dict(state_dict)
                mtp.eval()

                model.model.add_module("mtp", mtp)

                if _logger:
                    _logger.info("Created MTP module successfully")
        return mtp

    def generate_model_forward(self, model: nn.Module, inputs: Any) -> Generator[ProcessRequest, Any, None]:
        """
        Generate forward pass through model for calibration.

        Captures activations at each layer for computing quantization scales.
        Process layers one-by-one to minimize memory usage.
        """
        # Store first transformer block's input
        first_block_input: Optional[Tuple] = None

        def break_hook(model: nn.Module, hook_args: Tuple[Any, ...], hook_kwargs: Dict[str, Any]):
            nonlocal first_block_input
            first_block_input = (hook_args, hook_kwargs)
            raise TransformersForwardBreak()

        layer_0: nn.Module = model.model.layers[0]
        remove_handler = layer_0.register_forward_pre_hook(break_hook, with_kwargs=True, prepend=True)

        # Execute one forward pass to capture first layer's input
        try:
            if isinstance(inputs, (list, tuple)):
                model(inputs[0])
            elif isinstance(inputs, dict):
                model(**inputs)
            else:
                model(inputs)
        except TransformersForwardBreak:
            pass
        except Exception as e:
            if _logger:
                _logger.error("Error during forward pass: %s", e)
            raise e
        finally:
            remove_handler.remove()

        if first_block_input is None:
            raise InvalidModelError(
                "Cannot get first block input.",
                action="Please check the model and input",
            )

        # Synchronize in distributed setting
        if dist.is_initialized():
            dist.barrier()

        # Process each layer sequentially
        current_inputs = first_block_input
        if isinstance(inputs, dict):
            input_ids = inputs["input_ids"]
        else:
            input_ids = inputs[0]
        attention_mask = current_inputs[1].get("attention_mask", None)
        position_embeddings = current_inputs[1].get("position_embeddings", None)
        prev_hidden_states = None

        for name, block in self._generate_decoder_and_mtp_layer(model):
            args, kwargs = current_inputs

            if name == "model.mtp":
                args = (
                    input_ids,
                    prev_hidden_states,
                    position_embeddings,
                    attention_mask,
                )
                kwargs = {}
            # Yield to quantization pipeline
            hidden_states = yield ProcessRequest(name, block, args, kwargs)

            # Update inputs for next layer
            prev_hidden_states = hidden_states
            current_inputs = ((hidden_states,), current_inputs[1])

    def enable_kv_cache(self, model: nn.Module, need_kv_cache: bool) -> None:
        """Enable or disable KV cache for inference."""

        def pre_forward_hook(module, args, kwargs):
            kwargs["need_kv_cache"] = need_kv_cache
            return args, kwargs

        model.model.register_forward_pre_hook(pre_forward_hook, with_kwargs=True)

    def get_adapter_config_for_subgraph(self) -> List[AdapterConfig]:
        """
        Define subgraph relationships for LongCat-Flash quantization.

        Key architecture
        - Dua sublayers: self_attn[0/1], mlps[0/1], input_layernorm[0/1], post_attention_layernorm[0/1]
        - MoE uses fused expert weights (LongCatFlashExperts)
        - Zero experts are identity operation (no quantization needed)
        """
        adapter_config = []

        num_layers = getattr(self.config, "num_layers", 28)
        num_attention_heads = getattr(self.config, "num_attention_heads", 128)
        num_key_value_heads = getattr(self.config, "num_key_value_heads", 128)
        qk_nope_head_dim = getattr(self.config, "qk_nope_head_dim", 128)
        v_head_dim = getattr(self.config, "v_head_dim", 128)
        num_routed_experts = getattr(self.config, "n_routed_experts", 512)

        for layer_idx in range(num_layers):
            # Dual Attention Sublayers (0 and 1)
            for sub_index in [0, 1]:
                # MLA: kv_b_proj -> o_proj fusion
                okv_b_mapping = MappingConfig(
                    source=f"model.layers.{layer_idx}.self_attn.{sub_index}.kv_b_proj",
                    targets=[f"model.layers.{layer_idx}.self_attn.{sub_index}.o_proj"],
                )
                adapter_config.append(
                    AdapterConfig(
                        subgraph_type="ov",
                        mapping=okv_b_mapping,
                        extra_config={"group_method": "max"},
                        fusion=FusionConfig(
                            fusion_type="kv",
                            num_attention_heads=num_attention_heads,
                            num_key_value_heads=num_key_value_heads,
                            custom_config={
                                "qk_nope_head_dim": qk_nope_head_dim,
                                "v_head_dim": v_head_dim,
                            },
                        ),
                    )
                )

                # input_layernorm[i] -> q_a_proj, kv_a_proj
                input_norm_mapping = MappingConfig(
                    source=f"model.layers.{layer_idx}.input_layernorm.{sub_index}",
                    targets=[
                        f"model.layers.{layer_idx}.self_attn.{sub_index}.q_a_proj",
                        f"model.layers.{layer_idx}.self_attn.{sub_index}.kv_a_proj_with_mqa",
                    ],
                )
                adapter_config.append(AdapterConfig(subgraph_type="norm-linear", mapping=input_norm_mapping))

                # q_a_layernorm -> q_b_proj
                qa_norm_mapping = MappingConfig(
                    source=f"model.layers.{layer_idx}.self_attn.{sub_index}.q_a_layernorm",
                    targets=[f"model.layers.{layer_idx}.self_attn.{sub_index}.q_b_proj"],
                )
                adapter_config.append(AdapterConfig(subgraph_type="norm-linear", mapping=qa_norm_mapping))

            # Dual Dense MLP Sublayers (0 and 1)
            for sub_index in [0, 1]:
                # post_attention_layernorm[i] -> mlps[i]: gate_proj, up_proj
                post_norm_mapping = MappingConfig(
                    source=f"model.layers.{layer_idx}.post_attention_layernorm.{sub_index}",
                    targets=[
                        f"model.layers.{layer_idx}.mlps.{sub_index}.gate_proj",
                        f"model.layers.{layer_idx}.mlps.{sub_index}.up_proj",
                    ],
                )
                adapter_config.append(AdapterConfig(subgraph_type="norm-linear", mapping=post_norm_mapping))

                # mlps[i]: up_proj -> down_proj
                up_down_mapping = MappingConfig(
                    source=f"model.layers.{layer_idx}.mlps.{sub_index}.up_proj",
                    targets=[f"model.layers.{layer_idx}.mlps.{sub_index}.down_proj"],
                )
                adapter_config.append(AdapterConfig(subgraph_type="up-down", mapping=up_down_mapping))

            for expert_idx in range(num_routed_experts):
                moe_up_down_mapping = MappingConfig(
                    source=f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.up_proj",
                    targets=[f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.down_proj"],
                )
                adapter_config.append(AdapterConfig(subgraph_type="up-down", mapping=moe_up_down_mapping))

        return adapter_config

    @lru_cache(maxsize=1)
    def _get_weight_map(self) -> Dict[str, str]:
        """Load weight map from model.safetensors.index.json."""
        model_index_path = os.path.join(self.model_path, "model.safetensors.index.json")
        model_index = json_safe_load(model_index_path)
        return model_index.get("weight_map", {})

    def _get_state_dict(
        self,
        module: nn.Module,
        prefix: str = "",
    ) -> Dict[str, torch.Tensor]:
        """
        Load state dict for a specific module.

        Only loads weights needed for the given module (memory efficient).
        Groups weights by file to minimize file I/O operations.
        """
        weight_map = self._get_weight_map()
        names = [name for name, _ in module.named_parameters()]
        names += [name for name, _ in module.named_buffers()]

        # Group weights by their source file
        groups = defaultdict(list)
        for name in names:
            full_name = f"{prefix}.{name}" if prefix else name
            if full_name in weight_map:
                file_name = weight_map[full_name]
                groups[file_name].append(name)

        # Load weights from each file
        state_dict = {}
        for file_name in tqdm(groups, desc=f"Loading {prefix or 'model'}"):
            file_path = os.path.join(self.model_path, file_name)
            file_path = get_valid_read_path(file_path, extensions="safetensors", size_max=MAX_READ_FILE_SIZE_32G)
            with safe_open(file_path, framework="pt", device="cpu") as f:
                for name in groups[file_name]:
                    full_name = f"{prefix}.{name}" if prefix else name
                    state_dict[name] = f.get_tensor(full_name)

        return state_dict

    @staticmethod
    def _register_router_bias_parameter(router: nn.Module) -> None:
        bias = getattr(router, "e_score_correction_bias", None)
        if not isinstance(bias, torch.Tensor):
            return

        if "e_score_correction_bias" in dict(router.named_parameters(recurse=False)):
            return

        delattr(router, "e_score_correction_bias")
        router.register_parameter(
            "e_score_correction_bias",
            nn.Parameter(
                bias.detach().to(device=bias.device, dtype=torch.float32),
                requires_grad=bias.requires_grad,
            ),
        )

    def ascendv1_save_module_preprocess(
        self, prefix: str, module: nn.Module, model: nn.Module
    ) -> Tuple[str, nn.Module]:
        if not self._is_longcat_router(module):
            return prefix, module

        bias = getattr(module, "e_score_correction_bias", None)
        bias_needs_register = isinstance(bias, torch.Tensor) and "e_score_correction_bias" not in dict(
            module.named_parameters(recurse=False)
        )

        router_state = module.state_dict()
        router_needs_cast = any(
            isinstance(tensor, torch.Tensor) and tensor.dtype != torch.float32 for tensor in router_state.values()
        )
        if not bias_needs_register and not router_needs_cast:
            return prefix, module

        new_module = deepcopy(module)
        new_module.to(dtype=torch.float32)
        if bias_needs_register:
            self._register_router_bias_parameter(new_module)

        model.set_submodule(prefix, new_module)
        return prefix, new_module
