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

import os
from collections import defaultdict
from contextlib import contextmanager
from functools import lru_cache
from typing import List, Any, Generator, Dict, Tuple
from unittest.mock import patch

import torch
from safetensors import safe_open
from torch import nn
from tqdm import tqdm
from transformers import AutoModelForCausalLM
from msmodelslim.core.base.protocol import ProcessRequest
from msmodelslim.core.const import DeviceType
from msmodelslim.model.common.layer_wise_forward import (
    generated_decoder_layer_visit_func,
    transformers_generated_forward_func,
)
from msmodelslim.model.common.transformers import TransformersModel
from msmodelslim.model.interface_hub import ModelInfoInterface, ModelSlimPipelineInterfaceV1
from msmodelslim.utils.exception import InvalidModelError
from msmodelslim.utils.logging import logger_setter, get_logger
from msmodelslim.utils.security import get_valid_read_path, json_safe_load


@contextmanager
def default_dtype(dtype: torch.dtype):
    original = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(original)


@logger_setter()
class MiMoV2FlashAdapter(TransformersModel, ModelInfoInterface, ModelSlimPipelineInterfaceV1):  # pylint: disable=too-many-ancestors
    # ==================== ModelInfoInterface ====================
    def get_model_pedigree(self) -> str:
        # Must match the best-practice directory lab_practice/mimo_v2.
        return "mimo_v2"

    def get_model_type(self) -> str:
        return self.model_type

    # ==================== ModelSlimPipelineInterfaceV1 ====================
    def handle_dataset(self, dataset: Any, device: DeviceType = DeviceType.NPU) -> List[Any]:
        return self._get_tokenized_data(dataset, device)

    def init_model(self, device: DeviceType = DeviceType.NPU) -> nn.Module:
        """
        逐层加载初始化（MiMo-V2-Flash）：
        1) 临时将层数改为 1
        2) 初始化模型骨架 + 首层
        3) 恢复原始层数，其余层在 generate_decoder_layer 按需加载
        """
        _ = device
        if not hasattr(self.config, "num_hidden_layers"):
            raise InvalidModelError(
                "MiMo-V2-Flash config 缺少 num_hidden_layers。",
                action="请按目标模型实际 config 字段改写 init_model。",
            )

        origin_layers = self.config.num_hidden_layers
        self.config.num_hidden_layers = 1
        self.config.use_cache = False
        self.model_path = get_valid_read_path(str(self.model_path), is_dir=True, check_user_stat=True)

        model = self._create_model_instance().eval()

        self.config.num_hidden_layers = origin_layers
        state_dict = self._get_state_dict(model)
        model.load_state_dict(state_dict, strict=False)
        get_logger().info("MiMo-V2-Flash 初始化完成：首层已加载，其余层按需加载。")
        return model

    def generate_model_visit(self, model: nn.Module) -> Generator[ProcessRequest, Any, None]:
        yield from generated_decoder_layer_visit_func(
            model,
            transformer_blocks=self.generate_decoder_layer(model),
        )

    def generate_model_forward(self, model: nn.Module, inputs: Any) -> Generator[ProcessRequest, Any, None]:
        yield from transformers_generated_forward_func(model, inputs)

    def enable_kv_cache(self, model: nn.Module, need_kv_cache: bool) -> None:
        self._enable_kv_cache(model, need_kv_cache)

    # ==================== Layer-wise Helpers ====================
    def generate_decoder_layer(self, model: nn.Module) -> Generator[Tuple[str, nn.Module], None, None]:
        num_layers = self.config.num_hidden_layers
        for idx in range(num_layers):
            name = f"{self._decoder_layer_prefix()}.{idx}"
            layer = self._load_decoder_if_not_exist(model, name, idx)
            yield name, layer

    def _decoder_layer_prefix(self) -> str:
        """
        MiMo-V2-Flash decoder 层路径
        """
        return "model.layers"

    def _get_decoder_module_list(self, model: nn.Module) -> nn.ModuleList:
        path = self._decoder_layer_prefix()
        module_list = model.get_submodule(path)
        if not isinstance(module_list, nn.ModuleList):
            raise InvalidModelError(
                f"decoder 路径不是 ModuleList: {path}",
                action="请检查 _decoder_layer_prefix() 返回值是否正确。",
            )
        return module_list

    @lru_cache(maxsize=1)
    def _get_weight_map(self) -> Dict[str, str]:
        index_path = os.path.join(self.model_path, "model.safetensors.index.json")
        index_data = json_safe_load(index_path)
        return index_data["weight_map"]

    def _get_state_dict(self, module: nn.Module, prefix: str = "") -> Dict[str, torch.Tensor]:
        weight_map = self._get_weight_map()
        file_groups = defaultdict(list)
        for param_name, _ in module.named_parameters():
            full_name = f"{prefix}.{param_name}" if prefix else param_name
            if full_name in weight_map:
                file_groups[weight_map[full_name]].append(param_name)

        state_dict = {}
        for file_name, names in tqdm(file_groups.items(), desc=f"Loading {prefix}", leave=False):
            file_path = os.path.join(self.model_path, file_name)
            with safe_open(file_path, framework="pt", device="cpu") as f:
                for param_name in names:
                    full_name = f"{prefix}.{param_name}" if prefix else param_name
                    state_dict[param_name] = f.get_tensor(full_name)
        return state_dict

    def _is_layer_loaded(self, layer: nn.Module) -> bool:
        probe = next(layer.parameters(), None)
        return probe is not None and probe.device.type != "meta"

    def _post_process_loaded_decoder(self, layer: nn.Module) -> nn.Module:
        """
        MiMo-V2-Flash 后处理：记录 attention 类型和 MoE/MLP 类型
        """
        return layer

    def _load_decoder_if_not_exist(self, model: nn.Module, name: str, idx: int) -> nn.Module:
        """
        按需加载 decoder 层
        MiMo-V2-Flash 特殊处理：
        - Hybrid Attention: 根据 hybrid_layer_pattern 判断 attention 类型
        - MoE/MLP 混合: 根据 moe_layer_freq 判断使用 MoE 还是 MLP
        """
        with patch.object(nn.Linear, "reset_parameters", lambda _self: None), default_dtype(torch.bfloat16):
            try:
                decoder = model.get_submodule(name)
                if self._is_layer_loaded(decoder):
                    return self._post_process_loaded_decoder(decoder)
            except (AttributeError, RuntimeError):
                pass

            get_logger().info(
                "Loading MiMo-V2-Flash decoder layer %s: %s",
                idx,
                name,
            )

            # 获取层配置信息
            is_swa = hasattr(self.config, 'hybrid_layer_pattern') and self.config.hybrid_layer_pattern[idx] == 1
            is_moe = hasattr(self.config, 'moe_layer_freq') and self.config.moe_layer_freq[idx] == 1

            get_logger().info(
                "  Layer %s: attention=%s, mlp=%s",
                idx,
                "SWA" if is_swa else "Full",
                "MoE" if is_moe else "MLP",
            )

            module_list = self._get_decoder_module_list(model)
            template_module = module_list[0]

            # MiMo-V2-Flash 的 decoder layer 需要 layer_idx 参数
            decoder = template_module.__class__(config=self.config, layer_idx=idx)

            state_dict = self._get_state_dict(decoder, prefix=name)
            decoder.load_state_dict(state_dict, strict=False)
            decoder.eval()
            decoder = self._post_process_loaded_decoder(decoder)

            if len(module_list) <= idx:
                module_list.append(decoder)
            else:
                module_list[idx] = decoder
            return decoder

    def _create_model_instance(self) -> nn.Module:
        """
        创建 MiMo-V2-Flash 模型实例
        """
        return AutoModelForCausalLM.from_pretrained(  # nosec B615
            pretrained_model_name_or_path=str(self.model_path),
            config=self.config,
            trust_remote_code=False,
            device_map="cpu",
            torch_dtype="auto",
            local_files_only=True,
        )
