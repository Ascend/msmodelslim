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
# pylint: disable=no-name-in-module,ungrouped-imports

import re
from pathlib import Path

import torch
from regex import F  # noqa: F401
from torch import nn
from transformers import PretrainedConfig
from msmodelslim.core.const import DeviceType  # noqa: F401
from transformers import PreTrainedTokenizerBase, PreTrainedModel  # noqa: F401

from msmodelslim.utils.exception import SchemaValidateError
from msmodelslim.model.interface_hub import AscendV1GlobalModelDtypeInterface
from msmodelslim.utils.security.model import SafeGenerator
from ..base import BaseModelAdapter


class VLMBaseModelAdapter(BaseModelAdapter, AscendV1GlobalModelDtypeInterface):
    """
    VLM base model adapter providing basic attributes and methods for VLM models.
    To use, subclass and implement the required methods for your specific model.
    """

    def __init__(self, model_type: str, model_path: Path, trust_remote_code: bool = False):
        super().__init__(model_type, model_path, trust_remote_code)
        self.config = self._load_config(trust_remote_code=trust_remote_code)
        self.model_pedigree = self._get_model_pedigree(self.model_type)
        self.model_type = self._get_model_type(self.model_type)

    def get_global_model_torch_dtype(self) -> torch.dtype:
        """AscendV1GlobalModelDtypeInterface: return global model torch dtype (delegate to get_global_torch_dtype)."""
        dt = (
            getattr(self.config, "torch_dtype", None)
            or getattr(getattr(self.config, "text_config", None), "torch_dtype", None)
            or getattr(getattr(self.config, "vision_config", None), "torch_dtype", None)
        )
        if dt is None:
            return torch.float32
        if isinstance(dt, torch.dtype):
            return dt
        if isinstance(dt, str):
            m = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
            return m.get(dt, torch.float32)
        return torch.float32

    @staticmethod
    def _maybe_to_device(value, device):
        """Safely move a tensor-like object to target device; return original if not applicable."""
        if value is None:
            return None
        try:
            return value.to(device)
        except Exception:
            return value

    def _enable_kv_cache(self, model: nn.Module, enable: bool):
        model.model.config.use_cache = enable

    def _load_config(self, trust_remote_code=False) -> PretrainedConfig:
        return SafeGenerator.get_config_from_pretrained(
            model_path=str(self.model_path), trust_remote_code=trust_remote_code
        )

    def _get_model_type(self, model_type: str) -> str:
        if model_type is None:
            return self.config.model_type
        return model_type

    def _get_model_pedigree(self, model_type: str) -> str:
        if model_type is None:
            return self.config.model_type

        model_type = re.match(r'^[a-zA-Z]+', model_type)
        if model_type is None:
            raise SchemaValidateError(f"Invalid model_name: {model_type}.", action='Please check the model type')
        return model_type.group().lower()

    def _collect_inputs_to_device(self, inputs, device, keys, defaults=None):
        """
        Collect optional attributes from `inputs`, move to device when present, or fill defaults/None.

        Args:
            inputs: object (e.g., processor outputs) supporting attribute access
            device: torch device or DeviceType enum
            keys: iterable of keys to collect
            defaults: optional dict of default values when attribute missing

        Returns:
            dict with collected values
        """
        device_val = device.value if hasattr(device, "value") else device
        defaults = defaults or {}
        collected = {}
        for key in keys:
            val = getattr(inputs, key, None)
            if val is None:
                collected[key] = defaults.get(key)
            else:
                collected[key] = self._maybe_to_device(val, device_val)
        return collected
