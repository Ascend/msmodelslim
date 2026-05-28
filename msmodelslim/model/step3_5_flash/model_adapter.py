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

from typing import List, Any, Generator, Dict
import os
from collections import defaultdict
from pathlib import Path
from functools import lru_cache

from torch import nn
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM
from safetensors import safe_open

from msmodelslim.core.base.protocol import ProcessRequest
from msmodelslim.core.const import DeviceType
from msmodelslim.utils.logging import logger_setter, get_logger
from msmodelslim.utils.security import get_valid_read_path, json_safe_load, MAX_READ_FILE_SIZE_512G
from .moe_utils import convert_step35_moe_to_unpacked
from ..common.layer_wise_forward import generated_decoder_layer_visit_func
from ..default.model_adapter import DefaultModelAdapter
from ..interface_hub import ModelInfoInterface, ModelSlimPipelineInterfaceV1
from .step3p5_mtp import Step3p5MTPModule


# pylint: disable=too-many-ancestors
@logger_setter()
class Step3_5FlashModelAdapter(
    DefaultModelAdapter,
    ModelInfoInterface,
    ModelSlimPipelineInterfaceV1,
):
    def __init__(self, model_type: str, model_path: Path, trust_remote_code: bool = False):
        self._processor = None
        self._tokenizer = None
        super().__init__(model_type, model_path, trust_remote_code)
        self.mtp_start_layer = self.config.num_hidden_layers + 1
        self.mtp_layer_num = 3

    def get_model_type(self) -> str:
        return self.model_type

    def get_model_pedigree(self) -> str:
        return 'step3_5_flash'

    def load_model(self, device: DeviceType = DeviceType.NPU) -> nn.Module:
        return self._load_model(device)

    def handle_dataset(self, dataset: Any, device: DeviceType = DeviceType.NPU) -> List[Any]:
        return self._get_tokenized_data(dataset, device)

    def init_model(self, device: DeviceType = DeviceType.NPU) -> nn.Module:
        get_logger().info("Initializing Step3_5Flash model with msmodelslim v1 framework!")

        self.config.use_cache = False  # Disable cache to save device memory

        self.model_path = get_valid_read_path(str(self.model_path), is_dir=True, check_user_stat=True)

        model = AutoModelForCausalLM.from_pretrained(  # nosec B615: model_path is validated and local_files_only blocks Hub downloads.
            self.model_path,
            config=self.config,
            trust_remote_code=self.trust_remote_code,
            local_files_only=True,
            torch_dtype="auto",
            device_map="cpu",
            attn_implementation='eager',
        ).eval()

        # Ensure _attn_implementation is set for dynamically loaded layers
        # This prevents KeyError when layers access ALL_ATTENTION_FUNCTIONS[config._attn_implementation]
        # self.config._attn_implementation = 'eager'
        # TODO @yejiajun 通过后处理直接保存MTP层权重更加优雅，而不是在加载时添加MTP层 # pylint: disable=fixme
        for idx in range(self.mtp_start_layer, self.mtp_start_layer + self.mtp_layer_num):
            mtp = Step3p5MTPModule(model.config, layer_idx=idx)
            model.model.layers.append(mtp)
        state_dict = self._get_state_dict(model)
        model.load_state_dict(state_dict)

        if hasattr(model.config, 'num_attention_heads'):
            model.config.num_attention_heads = model.config.num_attention_heads
            get_logger().info("Set model.config.num_attention_heads = %s", model.config.num_attention_heads)
        if hasattr(model.config, 'num_key_value_heads'):
            model.config.num_key_value_heads = model.config.num_key_value_heads
            get_logger().info("Set model.config.num_key_value_heads = %s", model.config.num_key_value_heads)

        # Convert MoE layers to unpacked format for linear-quant
        self._convert_moe_layers_to_unpacked(model)

        return model

    def _convert_moe_layers_to_unpacked(self, model: nn.Module) -> None:
        """
        Convert all MoE layers in the model from Step3p5MoEMLP to Step3p5MoEMLPWithUnpackExperts.
        """
        get_logger().info("Starting MoE layer conversion to unpacked format...")

        converted_count = 0
        moe_layers_enum = getattr(self.config, "moe_layers_enum", None)
        if moe_layers_enum is not None:
            moe_layers_idx = [int(i) for i in moe_layers_enum.strip().split(',')]
        else:
            moe_layers_idx = list(range(1, self.config.num_hidden_layers))

        # Traverse all decoder layers
        for layer_idx in range(self.config.num_hidden_layers):
            if layer_idx not in moe_layers_idx:
                continue

            layer = model.model.layers[layer_idx]
            # Check if this layer has a moe attribute
            if hasattr(layer, 'moe'):
                try:
                    # Convert the MoE layer
                    new_moe = convert_step35_moe_to_unpacked(layer.moe, self.config)
                    layer.moe = new_moe
                    converted_count += 1
                    get_logger().info("Successfully converted MoE layer at index %s", layer_idx)
                except Exception as e:
                    get_logger().error("Failed to convert MoE layer at index %s: %s", layer_idx, e)
                    raise

        if converted_count > 0:
            get_logger().info("Successfully converted %s MoE layers to unpacked format", converted_count)
        else:
            get_logger().info("No MoE layers found to convert")

    def generate_model_visit(self, model: nn.Module) -> Generator[ProcessRequest, Any, None]:
        yield from generated_decoder_layer_visit_func(model)

    def generate_model_forward(
        self,
        model: nn.Module,
        inputs: Any,
    ) -> Generator[ProcessRequest, Any, None]:
        raise NotImplementedError(
            "Step3_5FlashModelAdapter: only supports dynamic quantization for this model; model forward is not implemented yet."
        )

    def enable_kv_cache(self, model: nn.Module, need_kv_cache: bool) -> None:
        return self._enable_kv_cache(model, need_kv_cache)

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

        state_dict = {}
        for file_name, names in tqdm(file_groups.items(), desc=f"Loading {prefix}", leave=False):
            file_path = os.path.join(self.model_path, file_name)
            file_path = get_valid_read_path(file_path, extensions='safetensors', size_max=MAX_READ_FILE_SIZE_512G)

            with safe_open(file_path, framework='pt', device='cpu') as f:
                for param_name in names:
                    full_name = f"{prefix}.{param_name}" if prefix else param_name
                    state_dict[param_name] = f.get_tensor(full_name)

        return state_dict
