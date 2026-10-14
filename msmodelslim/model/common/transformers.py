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
# pylint: disable=no-name-in-module

import re
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from transformers import PreTrainedTokenizerBase, PreTrainedModel, PretrainedConfig

from msmodelslim.core.const import DeviceType
from msmodelslim.utils.exception import SchemaValidateError
from msmodelslim.model.interface_hub import AscendV1GlobalModelDtypeInterface
from msmodelslim.utils.security.model import SafeGenerator
from ..base import BaseModelAdapter


class TransformersModel(BaseModelAdapter, AscendV1GlobalModelDtypeInterface):
    """
    Transformers model which implements some basic attrs and methods for transformers model.
    You can reuse these basic attrs and methods to implement interface for your own model adapter.
    HOWEVER, it may be not functional.
    """

    def __init__(self, model_type: str, model_path: Path, trust_remote_code: bool = False):
        super().__init__(model_type, model_path, trust_remote_code)
        self.config = self._load_config(trust_remote_code=trust_remote_code)
        self.tokenizer = self._load_tokenizer(trust_remote_code=trust_remote_code)

        self.model_pedigree = self._get_model_pedigree(self.model_type)
        self.model_type = self._get_model_type(self.model_type)

    def get_global_model_torch_dtype(self) -> torch.dtype:
        """AscendV1GlobalModelDtypeInterface: return global model torch dtype (delegate to get_global_torch_dtype)."""
        dt = getattr(self.config, "dtype", None) or getattr(self.config, "torch_dtype", None)
        if dt is None:
            return torch.float32
        if isinstance(dt, torch.dtype):
            return dt
        if isinstance(dt, str):
            m = {
                "float32": torch.float32,
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
                "bf16": torch.bfloat16,
            }
            return m.get(dt, torch.float32)
        return torch.float32

    def _enable_kv_cache(self, model: nn.Module, enable: bool):
        model.model.config.use_cache = enable

    def _load_config(self, trust_remote_code=False) -> PretrainedConfig:
        return SafeGenerator.get_config_from_pretrained(
            model_path=str(self.model_path), trust_remote_code=trust_remote_code
        )

    def _load_tokenizer(self, trust_remote_code=False) -> PreTrainedTokenizerBase:
        return SafeGenerator.get_tokenizer_from_pretrained(
            model_path=str(self.model_path),
            use_fast=False,
            legacy=False,
            trust_remote_code=trust_remote_code,
        )

    def _load_model(self, device: DeviceType) -> PreTrainedModel:
        device_map = "auto" if device == DeviceType.NPU else "cpu"
        self.config.num_hidden_layers = self.config.num_hidden_layers

        return SafeGenerator.get_model_from_pretrained(
            model_path=str(self.model_path),
            device_map=device_map,
            torch_dtype="auto",
            low_cpu_mem_usage=True,
            config=self.config,
            trust_remote_code=self.trust_remote_code,
        )

    def _get_model_type(self, model_type: str) -> str:
        if model_type is None:
            return self.config.model_type
        return model_type

    def _get_model_pedigree(self, model_type: str) -> str:
        if model_type is None:
            return self.config.model_type

        model_type = re.match(r"^[a-zA-Z]+", model_type)
        if model_type is None:
            raise SchemaValidateError(
                f"Invalid model_name: {model_type}.",
                action="Please check the model type",
            )
        return model_type.group().lower()

    def _get_padding_data(self, calib_list, device: DeviceType = DeviceType.NPU):
        """
        Get the padding data for the calibration.
        """
        calib_dataset = []
        max_len = 0
        for calib_data in calib_list:
            inputs = self.tokenizer(calib_data, return_tensors="pt", add_special_tokens=False)
            calib_dataset.append(inputs.data["input_ids"].to("npu" if device is DeviceType.NPU else "cpu"))
            max_len = max(max_len, inputs.data["input_ids"].size(1))
        new_calib_dataset = []
        for inputs in calib_dataset:
            new_inputs = F.pad(inputs, (0, max_len - inputs.size(1)), value=0)
            new_calib_dataset.append(new_inputs)
        return [torch.cat(new_calib_dataset)]

    def _get_batch_tokenized_data(self, calib_list, batch_size, device: DeviceType = DeviceType.NPU):
        """
        Get the batch tokenized data for the calibration.
        """
        if not isinstance(calib_list, list):
            raise SchemaValidateError(f"calib_list must be a list, but got {type(calib_list)}")

        calib_dataset = []
        calib_list = [calib_list[i : i + batch_size] for i in range(0, len(calib_list), batch_size)]
        for calib_data in calib_list:
            tmp = self._get_padding_data(calib_data, device)
            calib_dataset.append(tmp)
        return calib_dataset

    def _get_tokenized_data(
        self,
        calib_list,
        device: DeviceType,
        input_ids_name="input_ids",
        attention_mask_name="attention_mask",
        padding=True,
    ):
        if not isinstance(calib_list, list):
            raise SchemaValidateError(f"calib_list must be a list, but got {type(calib_list)}")

        tokenized_data = []
        for input_text in calib_list:
            inputs = self.tokenizer(input_text, return_tensors="pt", padding=padding).to(
                "npu" if device is DeviceType.NPU else "cpu"
            )
            tokenized_data.append([inputs.data[input_ids_name], inputs.data[attention_mask_name]])
        return tokenized_data
