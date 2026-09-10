#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
-------------------------------------------------------------------------
This file is part of the MindStudio project.
Copyright (c) 2026 Huawei Technologies Co.,Ltd.

MindStudio is licensed under Mulan PSL v2.
You may use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:

         http://license.coscl.org.cn/MulanPSL2

THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details.
-------------------------------------------------------------------------

Lazy reader for AscendV1 safetensors (single file or sharded with index).
"""

import os
from typing import Dict, Optional, Tuple

import torch
from safetensors import safe_open

from msmodelslim.utils.exception import InvalidModelError, SchemaValidateError
from msmodelslim.utils.security import json_safe_load
from msmodelslim.utils.security.path import MAX_READ_FILE_SIZE_32G, get_valid_read_path

ASCENDV1_SAFETENSORS_NAME = "quant_model_weights.safetensors"
ASCENDV1_INDEX_NAME = "quant_model_weights.safetensors.index.json"


class AscendV1TensorStore:
    """Lazy reader for AscendV1 safetensors (single file or sharded with index)."""

    def __init__(self, model_path: str):
        self.model_path = get_valid_read_path(model_path, is_dir=True)
        self._weight_map: Dict[str, str] = {}
        self._single_file: Optional[str] = None
        self._init_map()

    def _init_map(self) -> None:
        index_path = os.path.join(self.model_path, ASCENDV1_INDEX_NAME)
        single_path = os.path.join(self.model_path, ASCENDV1_SAFETENSORS_NAME)

        if os.path.isfile(index_path):
            index = json_safe_load(index_path, check_user_stat=False)
            weight_map = index.get("weight_map")
            if not isinstance(weight_map, dict):
                raise InvalidModelError(
                    f"Invalid AscendV1 index at {index_path}: missing weight_map",
                    action="Please ensure quant_model_weights.safetensors.index.json is a valid AscendV1 export.",
                )
            self._weight_map = {str(k): str(v) for k, v in weight_map.items()}
            return

        if os.path.isfile(single_path):
            self._single_file = single_path
            return

        raise InvalidModelError(
            f"AscendV1 weights not found under {self.model_path}",
            action="Expect quant_model_weights.safetensors or quant_model_weights.safetensors.index.json.",
        )

    def _resolve_file(self, key: str) -> str:
        if self._single_file is not None:
            return self._single_file
        file_name = self._weight_map.get(key)
        if file_name is None:
            raise SchemaValidateError(
                f"Tensor key '{key}' not found in AscendV1 weight map",
                action="Please check quant_model_description.json matches the safetensors export.",
            )
        return os.path.join(self.model_path, file_name)

    def get(self, key: str) -> torch.Tensor:
        return self._read_from_file(self._resolve_file(key), key)

    def get_shape(self, key: str) -> Tuple[int, ...]:
        """Return tensor shape from safetensors header without loading values."""
        file_path = get_valid_read_path(
            self._resolve_file(key), extensions="safetensors", size_max=MAX_READ_FILE_SIZE_32G
        )
        with safe_open(file_path, framework="pt", device="cpu") as handler:
            if key not in handler.keys():
                raise SchemaValidateError(
                    f"Tensor key '{key}' not found in {file_path}",
                    action="Please check AscendV1 export integrity.",
                )
            shape = handler.get_slice(key).get_shape()
            return tuple(int(d) for d in shape)

    def has(self, key: str) -> bool:
        if self._single_file is not None:
            file_path = get_valid_read_path(
                self._single_file, extensions="safetensors", size_max=MAX_READ_FILE_SIZE_32G
            )
            with safe_open(file_path, framework="pt", device="cpu") as handler:
                return key in handler.keys()
        return key in self._weight_map

    @staticmethod
    def _read_from_file(file_path: str, key: str) -> torch.Tensor:
        file_path = get_valid_read_path(file_path, extensions="safetensors", size_max=MAX_READ_FILE_SIZE_32G)
        with safe_open(file_path, framework="pt", device="cpu") as handler:
            if key not in handler.keys():
                raise SchemaValidateError(
                    f"Tensor key '{key}' not found in {file_path}",
                    action="Please check AscendV1 export integrity.",
                )
            return handler.get_tensor(key)


__all__ = [
    "AscendV1TensorStore",
    "ASCENDV1_SAFETENSORS_NAME",
    "ASCENDV1_INDEX_NAME",
]
