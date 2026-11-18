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
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from transformers import PreTrainedTokenizerBase, PreTrainedModel, PretrainedConfig

from msmodelslim.core.const import DeviceType
from msmodelslim.utils.exception import SchemaValidateError
from msmodelslim.model.interface_hub import AscendV1GlobalModelDtypeInterface
from msmodelslim.utils.security.model import SafeGenerator
from ..base import BaseModelAdapter


def _is_fsdp_nonmain_rank() -> bool:
    """True for non-zero ranks when Accelerate FSDP CPU-RAM-efficient loading is on.

    In that mode only rank 0 loads real weights; other ranks must hold an
    (empty) meta model so that ``accelerator.prepare`` can broadcast rank 0's
    shards. Building a full CPU model on every rank would blow host RAM
    (16 x 62 GB for a 32B model).
    """
    if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
        return False
    try:
        from transformers.modeling_utils import is_fsdp_enabled

        if not is_fsdp_enabled():
            return False
    except Exception:  # pragma: no cover - diagnostic only
        return False
    return int(os.environ.get("LOCAL_RANK", "0")) != 0


class _QATLoadMode(Enum):
    META = "meta"
    ASCENDV1 = "ascendv1"
    PRETRAINED = "pretrained"


@dataclass(frozen=True)
class _QATLoadPlan:
    mode: _QATLoadMode
    torch_dtype: Any


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

    def _load_tokenizer_for_qat(
        self,
        *,
        model_max_length: int = 8192,
    ) -> PreTrainedTokenizerBase:
        """QAT path (ModelOpt-aligned): load tokenizer with ``model_max_length``.

        Sets max sequence length used for train truncate / pad (default 8192).
        Also fills ``pad_token`` from ``eos_token`` when missing.
        """
        tokenizer = SafeGenerator.get_tokenizer_from_pretrained(
            model_path=str(self.model_path),
            model_max_length=model_max_length,
        )
        if getattr(tokenizer, "pad_token", None) is None and getattr(tokenizer, "eos_token", None) is not None:
            tokenizer.pad_token = tokenizer.eos_token
        if getattr(tokenizer, "pad_token_id", None) is None and getattr(tokenizer, "eos_token_id", None) is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        return tokenizer

    def _load_model(self, device: DeviceType) -> PreTrainedModel:
        """PTQ / Runner path: may set device_map=auto and place on device immediately."""
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

    def _resolve_qat_torch_dtype(self, torch_dtype: Any) -> Any:
        if torch_dtype is None or torch_dtype == "auto":
            return self.get_global_model_torch_dtype()
        return torch_dtype

    def _load_model_for_qat_from_ascendv1(
        self,
        *,
        torch_dtype: Any = "auto",
    ) -> PreTrainedModel:
        """Build the HF shell, apply the AscendV1 fake-quant IR loader, then hydrate
        weights on rank 0 only.

        The loaded IR is consumed by the QAT algorithm's source dispatch table;
        unsupported quant IR is rejected by the algorithm itself.
        """
        from msmodelslim.core.quantization_format.format_handler import build_default_format_chain
        from msmodelslim.utils.logging import get_logger

        resolved = self._resolve_qat_torch_dtype(torch_dtype)
        loader = build_default_format_chain().handle(str(self.model_path), device="cpu")
        loader.bind_adapter(self)

        model = self._build_qat_empty_model(torch_dtype=resolved)
        loader.apply_ir(model)
        if _is_fsdp_nonmain_rank():
            get_logger().info(
                "QAT AscendV1 load: applied IR structure on non-main rank (hydrate skipped) path=%s dtype=%s",
                self.model_path,
                resolved,
            )
        else:
            loader.hydrate(model)
            get_logger().info(
                "QAT AscendV1 load: applied IR and hydrated weights path=%s dtype=%s",
                self.model_path,
                resolved,
            )
        return model

    def _load_model_for_qat(
        self,
        *,
        torch_dtype: Any = "auto",
    ) -> PreTrainedModel:
        """QAT / FSDP path (ModelOpt-aligned): do **not** set device_map.

        Rank placement is handled by Accelerate env
        (``FSDP_CPU_RAM_EFFICIENT_LOADING``) + Transformers ``from_pretrained``
        after the process group is initialized. Trainer/FSDP wraps later.

        When ``--model_path`` points to a recognized fake-quant export, build the
        model via ``from_config`` and let the format loader apply FakeQuant IR.
        """
        plan = self._resolve_qat_load_plan(torch_dtype=torch_dtype)
        if plan.mode is _QATLoadMode.META:
            model = self._build_qat_empty_model(torch_dtype=plan.torch_dtype)
        elif plan.mode is _QATLoadMode.ASCENDV1:
            model = self._load_model_for_qat_from_ascendv1(torch_dtype=plan.torch_dtype)
        else:
            model = self._load_qat_pretrained_model(torch_dtype=plan.torch_dtype)
        return model

    def _resolve_qat_load_plan(self, *, torch_dtype: Any) -> _QATLoadPlan:
        from msmodelslim.core.quantization_format.format_handler import build_default_format_chain

        if build_default_format_chain().can_handle(str(self.model_path)):
            return _QATLoadPlan(mode=_QATLoadMode.ASCENDV1, torch_dtype=torch_dtype)
        if _is_fsdp_nonmain_rank():
            return _QATLoadPlan(
                mode=_QATLoadMode.META,
                torch_dtype=self._resolve_qat_torch_dtype(torch_dtype),
            )
        return _QATLoadPlan(mode=_QATLoadMode.PRETRAINED, torch_dtype=torch_dtype)

    def _build_qat_empty_model(self, *, torch_dtype: Any) -> PreTrainedModel:
        """Build a meta/empty HF shell for QAT (FSDP non-main rank or AscendV1 apply_ir).

        Only rank 0 materializes full weights; other ranks keep this shell so
        ``accelerator.prepare`` can broadcast rank0 state without blowing host RAM.
        """
        from accelerate import init_empty_weights
        from transformers import AutoModelForCausalLM

        if hasattr(self.config, "use_cache"):
            self.config.use_cache = False
        with init_empty_weights(include_buffers=False):
            model = AutoModelForCausalLM.from_config(  # nosec B615: local config only
                self.config,
                torch_dtype=torch_dtype,
                trust_remote_code=self.trust_remote_code,
            )
        # from_config under init_empty_weights leaves lm_head / embed_tokens as
        # separate meta params even when tie_word_embeddings=True. Rank0's
        # from_pretrained keeps them tied, so FSDP2 param remapping then fails
        # with: missing ['lm_head.weight']. Re-tie to match rank0.
        if getattr(self.config, "tie_word_embeddings", False) and hasattr(model, "tie_weights"):
            model.tie_weights()
        return model

    def _load_qat_pretrained_model(self, *, torch_dtype: Any) -> PreTrainedModel:
        # Do not force flash_attention_2: Ascend/NPU envs typically lack flash_attn
        # and Transformers will raise ImportError during from_pretrained.
        return SafeGenerator.get_model_from_pretrained(
            model_path=str(self.model_path),
            torch_dtype=torch_dtype,
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
