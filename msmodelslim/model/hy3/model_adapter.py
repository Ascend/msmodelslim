#!/usr/bin/env python
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
import time
from collections import defaultdict
from functools import lru_cache
from typing import Any, Dict, Generator, List, Optional, Tuple
from unittest.mock import patch

import torch
from safetensors import safe_open
from torch import distributed as dist
from torch import nn
from tqdm import tqdm
from transformers.models.hy_v3.modeling_hy_v3 import HYV3MoE

from msmodelslim.core.base.protocol import ProcessRequest
from msmodelslim.core.const import DeviceType
from msmodelslim.utils.exception import InvalidModelError
from msmodelslim.utils.logging import get_logger, logger_setter
from msmodelslim.utils.security import json_safe_load, get_valid_read_path, MAX_READ_FILE_SIZE_32G
from msmodelslim.utils.security.model import SafeGenerator
from ..common.layer_wise_forward import (
    generated_decoder_layer_visit_func,
    TransformersForwardBreak,
)
from ..default.model_adapter import DefaultModelAdapter
from ..interface_hub import (
    ModelInfoInterface,
    ModelSlimPipelineInterfaceV1,
)
from .moe_utils import UnstackedHy3MoE, convert_hy3_moe_to_unstacked


def _preserve_expert_bias_fp32(root: nn.Module) -> None:
    """Cast ``e_score_correction_bias`` to fp32 before ``load_state_dict``.

    Under ``default_dtype(bf16)``, HF creates this buffer as bf16. Loading a fp32
    checkpoint into that slot via ``copy_`` truncates irreversibly. Cast the
    destination first (buffer stays buffer) so load upcasts or assign replaces
    losslessly. Re-apply after load in case ``assign=True`` swapped dtype.
    """
    for _, module in root.named_modules():
        bias = getattr(module, "e_score_correction_bias", None)
        if isinstance(bias, torch.Tensor) and bias.dtype != torch.float32:
            # Re-assignment keeps registration kind (buffer/parameter).
            module.e_score_correction_bias = bias.detach().to(dtype=torch.float32)


def _promote_expert_bias_to_parameters(root: nn.Module) -> None:
    """Promote HF ``e_score_correction_bias`` buffer to Parameter before MoE unstack.

    Dtype is ensured by ``_preserve_expert_bias_fp32`` around load; this only
    changes registration so the saver / unstack path sees a Parameter.
    """
    for _, module in root.named_modules():
        if not hasattr(module, "e_score_correction_bias"):
            continue
        existing = getattr(module, "e_score_correction_bias")
        if existing is None or isinstance(existing, nn.Parameter):
            continue
        if not isinstance(existing, torch.Tensor):
            continue
        if "e_score_correction_bias" in module._buffers:
            del module._buffers["e_score_correction_bias"]
        module.register_parameter(
            "e_score_correction_bias",
            nn.Parameter(existing.detach().clone(), requires_grad=False),
        )


_MTP_MODULE_NAMES = ("enorm", "hnorm", "eh_proj", "final_layernorm")


def _attach_mtp_modules_to_decoder(mtp_decoder: nn.Module, config, layer_idx: int) -> None:
    hidden_size = config.hidden_size
    rms_norm_eps = config.rms_norm_eps

    mtp_decoder.enorm = nn.RMSNorm(hidden_size, eps=rms_norm_eps)
    mtp_decoder.hnorm = nn.RMSNorm(hidden_size, eps=rms_norm_eps)
    mtp_decoder.eh_proj = nn.Linear(hidden_size * 2, hidden_size, bias=False)
    mtp_decoder.final_layernorm = nn.RMSNorm(hidden_size, eps=rms_norm_eps)

    get_logger().info(
        "Attached MTP modules (enorm, hnorm, eh_proj, final_layernorm) to layer %d",
        layer_idx,
    )


def _load_mtp_weights_to_decoder(
    mtp_decoder: nn.Module,
    layer_idx: int,
    model_path: str,
    weight_map: Dict[str, str],
) -> None:
    prefix = f"model.layers.{layer_idx}"

    files_to_keys = defaultdict(list)
    loaded_count = 0
    for module_name in _MTP_MODULE_NAMES:
        key = f"{prefix}.{module_name}.weight"
        if key in weight_map:
            files_to_keys[weight_map[key]].append((module_name, key))
            loaded_count += 1
        else:
            get_logger().debug("MTP weight key not found in weight map: %s", key)

    for file_name, key_pairs in files_to_keys.items():
        file_path = os.path.join(model_path, file_name)
        file_path = get_valid_read_path(
            file_path,
            extensions=".safetensors",
            size_max=MAX_READ_FILE_SIZE_32G,
        )

        with safe_open(file_path, framework="pt", device="cpu") as f:
            for module_name, key in key_pairs:
                tensor = f.get_tensor(key)
                module = getattr(mtp_decoder, module_name)
                module.weight = nn.Parameter(tensor, requires_grad=False)
                get_logger().debug("Loaded %s from %s", key, file_name)

    get_logger().info(
        "Loaded %d MTP modules (%s) for layer %d",
        loaded_count,
        ", ".join(_MTP_MODULE_NAMES),
        layer_idx,
    )


@logger_setter()
class Hy3ModelAdapter(  # pylint: disable=too-many-ancestors
    DefaultModelAdapter,
    ModelInfoInterface,
    ModelSlimPipelineInterfaceV1,
):
    """
    Hy3 (HF Transformers) 模型适配器。

    支持逐层懒加载（1 层模板 + safetensors 按需物化）、
    checkpoint 中 MTP 模块加载、MoE unstack、标准 Transformers 链式 forward 与 KV 开关。
    """

    def get_model_type(self) -> str:
        return self.model_type

    def get_model_pedigree(self) -> str:
        return "hy3"

    def get_hidden_dim(self):
        return self.config.hidden_size

    def _prepare_mtp_config(self) -> None:
        original_layers = self.config.num_hidden_layers
        self.config.num_hidden_layers += 1
        target_len = self.config.num_hidden_layers
        if getattr(self.config, "mlp_layer_types", None) is None:
            first_k = getattr(self.config, "first_k_dense_replace", 1)
            self.config.mlp_layer_types = ["dense"] * first_k + ["sparse"] * (target_len - first_k)
        elif len(self.config.mlp_layer_types) < target_len:
            extra = target_len - len(self.config.mlp_layer_types)
            self.config.mlp_layer_types.extend(["sparse"] * extra)

        get_logger().info(
            "MTP enabled: expanded num_hidden_layers from %d to %d (MTP layer at index %d)",
            original_layers,
            self.config.num_hidden_layers,
            original_layers,
        )

    def _convert_single_moe_layer(self, layer: nn.Module, layer_idx: int) -> None:
        """Unstack one MoE layer right before layer-wise quant."""
        if isinstance(layer.mlp, UnstackedHy3MoE):
            return
        if not isinstance(layer.mlp, HYV3MoE):
            return

        t0 = time.time()
        layer.mlp = convert_hy3_moe_to_unstacked(layer.mlp, self.config)
        # Reload only once, immediately after convert (still nn.Linear). Re-running
        # after FakeQuantLinear deploy would cast bf16 ckpt into int8 -> all zeros.
        self._reload_unstacked_moe_from_checkpoint(layer, f"model.layers.{layer_idx}")
        get_logger().info(
            "MoE unstack layer %d done in %.1fs",
            layer_idx,
            time.time() - t0,
        )

    def _checkpoint_has_unstacked_moe(self, layer_name: str) -> bool:
        """True when floating checkpoint uses shared_mlp / experts.{i} layout."""
        weight_map = self.get_weight_map()
        return f"{layer_name}.mlp.shared_mlp.gate_proj.weight" in weight_map

    def _reload_unstacked_moe_from_checkpoint(self, layer: nn.Module, layer_name: str) -> None:
        """Reload MoE weights after unstack so names match checkpoint (shared_mlp/experts.i).

        HF HYV3MoE uses shared_experts/gate_up_proj which do not exist in Hy3 floating
        checkpoints. Lazy load then leaves zeros under reset_parameters patch; after
        convert_hy3_moe_to_unstacked the module names match the checkpoint and can be
        reloaded from safetensors.
        """
        if not isinstance(layer.mlp, UnstackedHy3MoE):
            return
        # Already quantized: loading bf16 into int8 FakeQuantLinear.weight truncates to 0.
        if not isinstance(layer.mlp.experts[0].gate_proj, nn.Linear):
            return
        if not self._checkpoint_has_unstacked_moe(layer_name):
            return

        state_dict = self.get_state_dict(layer.mlp, prefix=f"{layer_name}.mlp")
        if not state_dict:
            get_logger().warning(
                "Unstacked MoE checkpoint detected for %s but no tensors were loaded",
                layer_name,
            )
            return

        incompatible = layer.mlp.load_state_dict(state_dict, strict=False)
        get_logger().info(
            "Reloaded %d unstacked MoE tensors for %s (missing=%d, unexpected=%d)",
            len(state_dict),
            layer_name,
            len(incompatible.missing_keys),
            len(incompatible.unexpected_keys),
        )

    @lru_cache(maxsize=1)
    def get_weight_map(self) -> Dict[str, str]:
        model_index_path = os.path.join(self.model_path, "model.safetensors.index.json")
        model_index = json_safe_load(model_index_path)
        return model_index["weight_map"]

    def get_state_dict(self, module: nn.Module, prefix: str = "") -> Dict[str, torch.Tensor]:
        """Load parameters and buffers for ``module`` from safetensors (for expert bias buffers)."""
        weight_map = self.get_weight_map()
        names = [name for name, _ in module.named_parameters()]
        names += [name for name, _ in module.named_buffers()]

        groups = defaultdict(list)
        for name in names:
            full_name = f"{prefix}.{name}" if prefix else name
            if full_name not in weight_map:
                continue
            groups[weight_map[full_name]].append(name)

        state_dict = {}
        for file_name in tqdm(groups, desc=f"Loading {prefix or 'model'}"):
            file_path = os.path.join(self.model_path, file_name)
            file_path = get_valid_read_path(
                file_path,
                extensions="safetensors",
                size_max=MAX_READ_FILE_SIZE_32G,
            )
            with safe_open(file_path, framework="pt", device="cpu") as f:
                for name in tqdm(groups[file_name], desc=f"Loading {file_path}", leave=False):
                    full_name = f"{prefix}.{name}" if prefix else name
                    state_dict[name] = f.get_tensor(full_name)
        return state_dict

    def handle_dataset(self, dataset: Any, device: DeviceType = DeviceType.NPU) -> List[Any]:
        return self._get_tokenized_data(dataset, device)

    def handle_dataset_by_batch(
        self,
        dataset: Any,
        batch_size: int,
        device: DeviceType = DeviceType.NPU,
    ) -> List[Any]:
        return self._get_batch_tokenized_data(
            calib_list=dataset,
            batch_size=batch_size,
            device=device,
        )

    def init_model(self, device: DeviceType = DeviceType.NPU) -> nn.Module:
        torch.set_default_dtype(torch.bfloat16)
        self._prepare_mtp_config()
        origin_layers = self.config.num_hidden_layers
        get_logger().info("Model with %s layers totally", origin_layers)

        self.config.num_hidden_layers = 1
        model = SafeGenerator.get_model_from_pretrained(
            model_path=str(self.model_path),
            config=self.config,
            trust_remote_code=self.trust_remote_code,
            device_map="cpu",
            torch_dtype="auto",
        )
        self.config.num_hidden_layers = origin_layers

        _preserve_expert_bias_fp32(model)
        state_dict = self.get_state_dict(model)
        model.load_state_dict(state_dict, strict=False, assign=True)
        _preserve_expert_bias_fp32(model)
        _promote_expert_bias_to_parameters(model)

        model.eval()
        return model

    def generate_model_visit(self, model: nn.Module) -> Generator[ProcessRequest, Any, None]:
        return generated_decoder_layer_visit_func(
            model,
            transformer_blocks=self.generate_decoder_layer(model),
        )

    def generate_model_forward(
        self,
        model: nn.Module,
        inputs: Any,
    ) -> Generator[ProcessRequest, Any, None]:
        first_block_input: Optional[Tuple] = None

        def break_hook(module: nn.Module, hook_args: Tuple[Any, ...], hook_kwargs: Dict[str, Any]):
            nonlocal first_block_input
            first_block_input = (hook_args, hook_kwargs)
            raise TransformersForwardBreak()

        remove_handler = model.model.layers[0].register_forward_pre_hook(break_hook, with_kwargs=True, prepend=True)

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
            raise e
        finally:
            remove_handler.remove()

        if first_block_input is None:
            raise InvalidModelError("Can't get first block input.", action="Please check the model and input")

        current_inputs = first_block_input

        if dist.is_initialized():
            dist.barrier()

        for name, block in self.generate_decoder_layer(model):
            args, kwargs = current_inputs
            outputs = yield ProcessRequest(name, block, args, kwargs)
            hidden_states = outputs[0] if isinstance(outputs, tuple) else outputs
            current_inputs = ((hidden_states,), current_inputs[1])

    def enable_kv_cache(self, model: nn.Module, need_kv_cache: bool) -> None:
        return self._enable_kv_cache(model, need_kv_cache)

    def load_decoder_if_not_exist(self, model: nn.Module, name: str, idx: int) -> nn.Module:
        try:
            decoder = model.get_submodule(name)
        except AttributeError:
            with patch.object(nn.Linear, "reset_parameters", lambda _self: None):
                get_logger().info("Creating decoder layer %s", idx)
                module_list: nn.ModuleList = model.model.layers
                template_module = module_list[0]
                decoder = template_module.__class__(config=self.config, layer_idx=idx)

                _preserve_expert_bias_fp32(decoder)
                state_dict = self.get_state_dict(decoder, prefix=name)
                decoder.load_state_dict(state_dict, strict=False, assign=True)
                _preserve_expert_bias_fp32(decoder)
                decoder.eval()
                module_list.append(decoder)
                get_logger().info("Create decoder layer %s successfully", idx)

        _promote_expert_bias_to_parameters(decoder)
        self._convert_single_moe_layer(decoder, idx)
        return decoder

    def load_mtp_if_not_exist(self, mtp_decoder: nn.Module) -> None:
        try:
            mtp_decoder.get_submodule("enorm")
            return
        except AttributeError:
            pass

        layer_idx = self.config.num_hidden_layers - 1
        get_logger().info("Creating MTP modules on layer %d", layer_idx)
        _attach_mtp_modules_to_decoder(mtp_decoder, self.config, layer_idx)
        _load_mtp_weights_to_decoder(
            mtp_decoder,
            layer_idx,
            str(self.model_path),
            self.get_weight_map(),
        )
        get_logger().info("Create MTP successfully")

    def generate_decoder_layer(self, model: nn.Module) -> Generator[Tuple[str, nn.Module], None, None]:
        for idx in range(self.config.num_hidden_layers):
            name = f"model.layers.{idx}"
            decoder = self.load_decoder_if_not_exist(model, name=name, idx=idx)
            if idx == self.config.num_hidden_layers - 1:
                self.load_mtp_if_not_exist(decoder)
            yield name, decoder
