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
# pylint: disable=logging-fstring-interpolation

import os
from pathlib import Path
from typing import Optional, Dict, Any

import hashlib  # noqa: F401
import torch

from msmodelslim.core.quant_service import KeyInfoPersistenceInfra
from msmodelslim.core.context import IContext, get_current_context
from msmodelslim.core.quant_service.modelslim_v1.save.utils.json import JsonWriter
from msmodelslim.core.quant_service.modelslim_v1.save.utils.safetensors import (
    SafetensorsWriter,
)
from msmodelslim.utils.exception import SchemaValidateError
from msmodelslim.utils.logging import get_logger, logger_setter
from msmodelslim.utils.security import get_write_directory

DEBUG_INFO_JSON_NAME = "debug_info.json"
DEBUG_INFO_SAFETENSORS_NAME = "debug_info.safetensors"


@logger_setter("msmodelslim.infra.context_persistence")
class DebugInfoPersistence(KeyInfoPersistenceInfra):
    """
    Context persistence for saving debug information from quantization context.

    This class saves context data into two files:
        - debug_info.json: Contains non-tensor data and tensor metadata (shape, dtype)
        - debug_info.safetensors: Contains all PyTorch tensors in SafeTensors format

    File format follows the specification in ctx_save.md:
        - Tensors are stored with flattened keys: {namespace_key}.{dict_type}.{item_key}
        - Tensor metadata (shape, dtype) is stored in debug_info.json under namespaces
        - Non-tensor data is stored directly in debug_info.json under namespaces

    Typical usage:
        >>> from msmodelslim.infra.context_persistence import DebugInfoPersistence
        >>> from msmodelslim.core.context import get_current_context
        >>>
        >>> ctx = get_current_context()
        >>> persistence = DebugInfoPersistence(save_dir="./debug_output")
        >>> persistence.save_from_context(ctx)
    """

    def __init__(self, save_dir: Optional[str] = None):
        """
        Initialize DebugInfoPersistence.

        Args:
            save_dir: Directory to save context files. If None, uses current directory.
        """
        if save_dir is None:
            save_dir = "."
        self.save_dir = Path(save_dir) / "debug_info"

        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Validate write permissions
        get_write_directory(str(self.save_dir))

        # Initialize writers
        self.json_writer = JsonWriter(str(self.save_dir), DEBUG_INFO_JSON_NAME)
        self.safetensors_writer = SafetensorsWriter(
            logger=get_logger(),
            file_path=os.path.join(str(self.save_dir), DEBUG_INFO_SAFETENSORS_NAME),
        )

    def save_from_context(self, ctx: Optional[IContext] = None) -> None:
        """
        Save context data to debug_info.json and debug_info.safetensors.

        Args:
            ctx: Context to save. If None, uses current context from get_current_context().

        Raises:
            SchemaValidateError: If context is None or invalid
        """
        if ctx is None:
            ctx = get_current_context()

        if ctx is None:
            raise SchemaValidateError(
                "No context available to save",
                action="Ensure context is created and set before calling save_from_context",
            )

        get_logger().info(f"Saving context to {self.save_dir}")

        # Process each namespace - only save debug information
        for namespace_key in ctx.keys():
            namespace = ctx[namespace_key]

            # Directly save debug information without the "debug" wrapper
            namespace_data = self._process_dict(namespace.debug, f"{namespace_key}")

            if namespace_data:
                # Write each namespace directly to JSON root (no "namespaces" wrapper)
                self.json_writer.write(namespace_key, namespace_data)

        # Close writers
        self.json_writer.close()
        self.safetensors_writer.close()

        get_logger().info(f"Context saved successfully to {self.save_dir}")

    def _process_dict(
        self,
        data_dict: Dict[str, Any],
        prefix: str,
    ) -> Dict[str, Any]:
        """
        Process a dictionary, separating tensors and non-tensors.

        For tensors: save to safetensors and return reference info (file and key)
        For non-tensors: return the serialized value directly

        Args:
            data_dict: Dictionary to process
            prefix: Key prefix for flattened storage in safetensors

        Returns:
            Dictionary with non-tensor values and tensor reference info
        """
        result = {}
        ref_cache = {}
        ref_key = [0]
        for key, value in data_dict.items():
            full_key = f"{prefix}.{key}"
            result[key] = self._serialize_value(value, full_key, ref_cache, ref_key)

        return result

    def _serialize_value(self, value: Any, key_path: str, ref_cache, ref_key) -> Any:
        """
        Serialize any value, handling tensors and non-tensors recursively.

        For tensors: save to safetensors and return reference info
        For non-tensors: return the serialized value directly

        Args:
            value: Value to serialize
            key_path: Full key path for tensor storage

        Returns:
            Serialized value or tensor reference
        """
        if isinstance(value, torch.Tensor):
            hash_hex = value.data_ptr()
            if hash_hex not in ref_cache:
                ref_cache[hash_hex] = {
                    "_type": "tensor",
                    "_file": DEBUG_INFO_SAFETENSORS_NAME,
                    "_key": f"tensor_{ref_key[0]}",
                }
                ref_key[0] += 1
                self.safetensors_writer.write(f"tensor_{ref_key[0]}", value)

            return ref_cache[hash_hex]
        elif isinstance(value, dict):
            return {k: self._serialize_value(v, f"{key_path}.{k}", ref_cache, ref_key) for k, v in value.items()}
        elif isinstance(value, (list, tuple)):
            return [self._serialize_value(v, f"{key_path}[{idx}]", ref_cache, ref_key) for idx, v in enumerate(value)]
        elif isinstance(value, (int, float, str, bool, type(None))):
            return value
        else:
            # For other types, try to convert to string
            try:
                return str(value)
            except Exception as e:
                get_logger().warning(f"Failed to serialize value of type {type(value)}: {e}")
                return f"<unserializable: {type(value).__name__}>"
