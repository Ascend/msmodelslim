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

import time
from typing import Any, List, Optional

from msmodelslim.core.const import DeviceType
from msmodelslim.core.context import ContextManager, IContextFactory
from msmodelslim.core.infer_engine import (
    DPFakeQuantInferenceEngine,
    FakeQuantInferenceEngine,
    InferenceConfig,
)
from msmodelslim.core.infer_engine.interface import FakeQuantInferenceInterface
from msmodelslim.core.quant_service import DatasetLoaderInfra
from msmodelslim.core.quant_service.multimodal_sd_v1.legacy_pipeline_interface import (
    LegacyMultimodalPipelineInterface,
)
from msmodelslim.core.quant_service.multimodal_sd_v1.pipeline_interface import (
    MultimodalPipelineInterface,
)
from msmodelslim.format.format_handler import build_default_format_chain
from msmodelslim.infra.file_dataset_loader import FileDatasetLoader
from msmodelslim.model import IModelFactory
from msmodelslim.model.common.vlm_base import VLMBaseModelAdapter
from msmodelslim.utils.exception import InvalidModelError, SchemaValidateError, UnsupportedError
from msmodelslim.utils.exception_decorator import exception_catcher
from msmodelslim.utils.logging import get_logger, logger_setter
from msmodelslim.utils.validation.conversion import convert_to_readable_dir
from msmodelslim.utils.validation.value import validate_str_length

from .result_displayer_infra import InferenceResultDisplayerInfra


@logger_setter()
class InferenceApplication:
    """CLI application for model inference verification (e.g. AscendV1 FakeQuant)."""

    def __init__(
        self,
        model_factory: IModelFactory,
        dataset_loader: FileDatasetLoader,
        context_factory: IContextFactory,
        result_displayer: InferenceResultDisplayerInfra,
        vlm_dataset_loader: Optional[DatasetLoaderInfra] = None,
    ):
        self.model_factory = model_factory
        self.dataset_loader = dataset_loader
        self.context_factory = context_factory
        self.result_displayer = result_displayer
        self.vlm_dataset_loader = vlm_dataset_loader

    @exception_catcher
    def run(
        self,
        model_type: str,
        model_path: str,
        prompt_file: str,
        device: DeviceType = DeviceType.NPU,
        device_indices: Optional[List[int]] = None,
        max_new_tokens: int = 1,
    ) -> dict:
        if not isinstance(model_type, str):
            raise SchemaValidateError(f"model_type must be a string, but got {type(model_type)}")
        validate_str_length(input_str=model_type, str_name="model_type", max_len=256)
        model_path_obj = convert_to_readable_dir(model_path)
        model_path = str(model_path_obj)
        if max_new_tokens <= 0:
            raise SchemaValidateError(
                f"max_new_tokens must be positive, got {max_new_tokens}",
                action="Please set --max_new_tokens to a positive integer.",
            )
        get_logger().info(
            "Building adapter and FakeQuant model from export: %s",
            model_path,
        )
        t_start = time.perf_counter()
        try:
            adapter = self.model_factory.create(
                model_type=model_type,
                model_path=model_path_obj,
                trust_remote_code=True,
            )
        except Exception as exc:
            raise InvalidModelError(
                f"Failed to build adapter from model_path={model_path}: {exc}",
                action="Please ensure the AscendV1 export contains complete config.json and "
                "tokenizer files (e.g. tokenizer.json / vocab.json / merges.txt).",
            ) from exc
        if isinstance(adapter, (MultimodalPipelineInterface, LegacyMultimodalPipelineInterface)):
            raise UnsupportedError(
                f"Model adapter for {model_type!r} is a DiT (diffusion-based image/video "
                "generation) model, which fake-quant inference does not support.",
                action="Fake-quant inference supports LLM and VLM text-generation models only. "
                "Use a supported LLM/VLM model_type, or use the dedicated multimodal quant "
                "pipeline for DiT models.",
            )
        if not isinstance(adapter, FakeQuantInferenceInterface):
            raise UnsupportedError(
                f"Model adapter for {model_type!r} does not implement FakeQuantInferenceInterface",
                action="Fake-quant requires build_meta_model and handle_dataset "
                "(the engine drives native model.forward with per-layer weight hooks).",
            )

        format_loader = build_default_format_chain().handle(model_path, device="cpu")

        distributed = device_indices is not None and len(device_indices) > 1
        samples = self._load_samples(adapter, prompt_file)

        infer_config = InferenceConfig(
            max_new_tokens=max_new_tokens,
        )
        engine = DPFakeQuantInferenceEngine() if distributed else FakeQuantInferenceEngine()
        devices = device_indices if device_indices is not None else [0]
        get_logger().info(
            "Fake-quant inference start: model_type=%s, device=%s, devices=%s, mode=%s, samples=%d, max_new_tokens=%d",
            model_type,
            device,
            devices,
            "multi-card-DP" if distributed else "single-card",
            len(samples),
            infer_config.max_new_tokens,
        )
        # Pass raw (untokenized) samples to the engine; tokenization happens in each worker
        # via adapter.handle_dataset, avoiding the HF tokenizers parallelism-before-fork
        # warning and pickling large VLM pixel_values across processes. The DP engine shares
        # one context across spawned workers so partial results merge back to global order.
        with ContextManager(self.context_factory.create(is_distributed=distributed)):
            infer_result = engine.run(
                adapter=adapter,
                format_loader=format_loader,
                inputs=samples,
                inference_config=infer_config,
                device=device,
                device_indices=device_indices,
            )

        self.result_displayer.display_result(infer_result, device_indices=device_indices)

        result = {
            "samples_ok": len(samples),
            "model_path": model_path,
            "mode": "layer_wise_distributed" if distributed else "layer_wise",
            "device_indices": device_indices,
            "max_new_tokens": max_new_tokens,
            "generated_token_ids": infer_result.generated_token_ids,
            "generated_texts": infer_result.generated_texts,
        }
        get_logger().info(
            "Fake-quant layer-wise verification finished: samples_ok=%d, max_new_tokens=%d, distributed=%s, total=%.1fs",
            result["samples_ok"],
            max_new_tokens,
            distributed,
            time.perf_counter() - t_start,
        )
        get_logger().info("===========SUCCESS===========")
        return result

    def _load_samples(self, adapter: FakeQuantInferenceInterface, prompt_file: str) -> List[Any]:
        loader = self._resolve_dataset_loader(adapter)
        data = loader.get_dataset_by_name(prompt_file)
        if not data:
            raise SchemaValidateError(
                f"Prompt file '{prompt_file}' is empty",
                action="Please provide a non-empty prompt file.",
            )
        return list(data)

    def _resolve_dataset_loader(self, adapter: FakeQuantInferenceInterface) -> DatasetLoaderInfra:
        if isinstance(adapter, VLMBaseModelAdapter):
            if self.vlm_dataset_loader is None:
                raise UnsupportedError(
                    f"Model adapter {type(adapter).__name__} requires a VLM dataset loader",
                    action="Please configure VLMDatasetLoader for multimodal fake-quant verification.",
                )
            return self.vlm_dataset_loader
        return self.dataset_loader
