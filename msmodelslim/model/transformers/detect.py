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

from enum import Enum
from pathlib import Path
from typing import Any, Dict

from msmodelslim.utils.exception import UnsupportedError, SecurityError
from msmodelslim.utils.security import json_safe_load


class ModelClass(str, Enum):
    LLM = "llm"
    VLM = "vlm"
    DIT = "dit"


LLM_ADAPTER_CLASS_PATH = "msmodelslim.model.transformers.model_adapter:LLMTransformersModel"

_VLM_CONFIG_KEYS = ("vision_config",)

_DIT_CONFIG_KEYS = ("_diffusers_version",)


def _is_vlm_config(config: Dict[str, Any]) -> bool:
    if any(key in config for key in _VLM_CONFIG_KEYS):
        return True
    return False


def _is_dit_config(config: Dict[str, Any]) -> bool:
    if any(key in config for key in _DIT_CONFIG_KEYS):
        return True
    return False


def detect_transformers_kind(model_path: Path) -> ModelClass:
    """Classify a HuggingFace / diffusers directory as llm, vlm, or dit."""
    config_path = model_path / "config.json"
    try:
        config = json_safe_load(str(config_path))
    except SecurityError:
        raise UnsupportedError(
            f"The path {config_path} doesn't exist or isn't a file. "
            "Only standard transformers-based LLM model is supported currently "
            "when using model_type transformers.",
            action=(f"Please check (1) files in {model_path} are complete. (2) the model is a LLM model."),
        )

    if _is_vlm_config(config):
        return ModelClass.VLM

    if _is_dit_config(config):
        return ModelClass.DIT

    return ModelClass.LLM


def resolve_adapter_class_path(model_path: Path) -> str:
    """Return adapter class path for ``--model_type transformers``.

    VLMTransformersModel is not registered yet; VLM/DiT raise until a dedicated
    generic adapter exists.
    """

    kind = detect_transformers_kind(model_path)

    if kind == ModelClass.VLM:
        raise UnsupportedError(
            "VLM quantization is not supported for model_type transformers",
            action="Please use a dedicated model adapter for multimodal models (VLM/DiT)",
        )
    if kind == ModelClass.DIT:
        raise UnsupportedError(
            "DiT quantization is not supported for model_type transformers",
            action="Please use a dedicated model adapter for multimodal models (VLM/DiT)",
        )
    return LLM_ADAPTER_CLASS_PATH
