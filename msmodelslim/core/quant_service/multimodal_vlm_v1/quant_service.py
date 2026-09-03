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

from pathlib import Path
from typing import Optional, Literal, Any, List

import torch

from msmodelslim.core.const import RunnerType, DeviceType
from msmodelslim.core.quant_service import DatasetLoaderInfra
from msmodelslim.core.quant_service import KeyInfoPersistenceInfra
from msmodelslim.core.runner.layer_wise_runner import LayerWiseRunner
from msmodelslim.core.runner.pipeline_interface import PipelineInterface
from msmodelslim.core.runner.optional_interface import LayerWiseOffloadOptionalInterface
from msmodelslim.utils.exception import InvalidDatasetError, SchemaValidateError, UnsupportedError
from msmodelslim.utils.logging import get_logger, logger_setter
from msmodelslim.utils.seed import seed_all
from msmodelslim.core.context import ContextManager, IContextFactory
from .quant_config import MultimodalVLMModelslimV1QuantConfig
from ..interface import BaseQuantConfig, IQuantService, QuantServiceConfig


class MultimodalVLMModelslimV1QuantServiceConfig(QuantServiceConfig):
    """multimodal_vlm_modelslim_v1 量化服务配置，用于插件选择与 QuantService 初始化。"""

    apiversion: Literal["multimodal_vlm_modelslim_v1"] = "multimodal_vlm_modelslim_v1"


@logger_setter(
    prefix='msmodelslim.core.quant_service.multimodal_vlm_modelslim_v1'
)  # 4-level: msmodelslim.core.quant_service.multimodal_vlm_modelslim_v1
class MultimodalVLMModelslimV1QuantService(IQuantService):
    """
    Quantization service for multimodal vision-language models (V1 framework).

    Features:
    - Layer-wise loading and processing (memory efficient)
    - Automatic MoE fusion layer conversion
    - Multi-modal calibration dataset support
    - Data-parallel layer-wise quantization (DP) when multiple devices are specified
    - Compatible with msmodelslim quant command

    Supported models:
    - Qwen3-VL-MoE
    - Other multimodal VLM models (extensible)
    """

    backend_name: str = "multimodal_vlm_modelslim_v1"

    def __init__(
        self,
        quant_service_config: MultimodalVLMModelslimV1QuantServiceConfig,
        dataset_loader: DatasetLoaderInfra,
        context_factory: Optional[IContextFactory] = None,
        debug_info_persistence: Optional[KeyInfoPersistenceInfra] = None,
        **kwargs,
    ):
        """
        Initialize multimodal VLM quantization service.

        Args:
            quant_service_config: MultimodalVLMModelslimV1QuantServiceConfig.
            dataset_loader: DatasetLoaderInfra（用于加载数据集）.
            context_factory: Optional context factory for dependency injection.
            debug_info_persistence: Optional context persistence for saving debug info.
        """
        self.quant_service_config = quant_service_config
        self.dataset_loader = dataset_loader
        self.context_factory = context_factory
        self.debug_info_persistence = debug_info_persistence

    @staticmethod
    def _choose_runner_type(
        quant_config: MultimodalVLMModelslimV1QuantConfig,
        model_adapter: PipelineInterface,
        device_indices: Optional[List[int]] = None,
    ) -> Literal[RunnerType.LAYER_WISE, RunnerType.DP_LAYER_WISE]:
        """
        Choose runner type based on config and device list.

        VLM stays on layer-wise pipelines (single-card or DP). model_wise is not
        supported and falls back to layer_wise.
        """
        if quant_config.spec.runner == RunnerType.MODEL_WISE:
            get_logger().warning(
                "Model-wise runner is not supported for %s; falling back to layer-wise.",
                MultimodalVLMModelslimV1QuantService.backend_name,
            )
            return RunnerType.LAYER_WISE

        if quant_config.spec.runner == RunnerType.LAYER_WISE:
            get_logger().info("Layer-wise runner detected, using layer-wise pipeline.")
            return RunnerType.LAYER_WISE

        if quant_config.spec.runner == RunnerType.DP_LAYER_WISE:
            get_logger().info("Distributed layer-wise runner detected, using distributed layer-wise pipeline.")
            return RunnerType.DP_LAYER_WISE

        if quant_config.spec.runner == RunnerType.AUTO and device_indices is not None and len(device_indices) > 1:
            get_logger().info("Multi-device configuration detected, using distributed layer-wise pipeline.")
            return RunnerType.DP_LAYER_WISE

        get_logger().info("Runner type not detected, defaulting to layer-wise pipeline (recommended for VLM).")
        return RunnerType.LAYER_WISE

    @staticmethod
    def _resolve_offload_device(model_adapter: PipelineInterface) -> str:
        offload_device = "cpu"
        if isinstance(model_adapter, LayerWiseOffloadOptionalInterface):
            preferred_offload = model_adapter.get_layer_wise_offload_device()
            if isinstance(preferred_offload, str) and preferred_offload:
                if preferred_offload in ("cpu", "meta"):
                    offload_device = preferred_offload
                else:
                    get_logger().warning(
                        "Invalid offload device %s from model adapter, fallback to 'cpu'. Supported: ['cpu', 'meta'].",
                        preferred_offload,
                    )
        return offload_device

    @staticmethod
    def _validate_loaded_samples_text(dataset: List[Any]) -> None:
        """Validate that every loaded calibration sample has non-empty text."""
        from msmodelslim.infra.dataset_loader.vlm_dataset_loader import VlmCalibSample

        for index, sample in enumerate(dataset, start=1):
            if isinstance(sample, VlmCalibSample):
                text = sample.text
            elif isinstance(sample, dict):
                text = sample.get("text")
            else:
                text = getattr(sample, "text", None)
            if text is None or (isinstance(text, str) and not text.strip()):
                raise InvalidDatasetError(
                    "text data is missing",
                    action="Provide non-empty text in calibration samples.",
                )

    def quantize(
        self,
        quant_config: BaseQuantConfig,
        model_adapter: Any,
        save_path: Optional[Path] = None,
        device: DeviceType = DeviceType.NPU,
        device_indices: Optional[List[int]] = None,
    ):
        """
        Main quantization entry point.

        Args:
            quant_config: Base quantization config (will be converted to MultimodalVLMV1QuantConfig)
            model_adapter: Model adapter implementing PipelineInterface
            save_path: Path to save quantized model
            device: Device for quantization (NPU or CPU)
            device_indices: Physical device indices for DP (e.g. [0, 1, 2, 3])
        """
        # Validate inputs
        if not isinstance(quant_config, BaseQuantConfig):
            raise SchemaValidateError("task is not a BaseTask", action="Please make sure the task is a BaseTask")
        if not isinstance(model_adapter, PipelineInterface):
            raise SchemaValidateError(
                "model_adapter must be a PipelineInterface",
                action="Please make sure the model_adapter is a PipelineInterface",
            )
        if save_path is not None and not isinstance(save_path, Path):
            raise SchemaValidateError(
                "save_path must be a Path or None", action="Please make sure the save_path is a Path or None"
            )
        if not isinstance(device, DeviceType):
            raise SchemaValidateError(
                "device must be a DeviceType", action="Please make sure the device is a DeviceType"
            )

        return self.quant_process(
            MultimodalVLMModelslimV1QuantConfig.from_base(quant_config),
            model_adapter,
            save_path,
            device,
            device_indices,
        )

    def quant_process(
        self,
        quant_config: MultimodalVLMModelslimV1QuantConfig,
        model_adapter: PipelineInterface,
        save_path: Optional[Path],
        device: DeviceType = DeviceType.NPU,
        device_indices: Optional[List[int]] = None,
    ):
        """
        Core quantization process.

        Steps:
        1. Set random seed
        2. Load dataset (multimodal images)
        3. Choose layer-wise or DP layer-wise runner
        4. Add processors (anti-outlier, quant, save, etc.)
        5. Run quantization

        Args:
            quant_config: Multimodal VLM quantization config
            model_adapter: Model adapter
            save_path: Save path
            device: Device
            device_indices: Physical device indices for DP
        """
        common_seed = 42
        seed_all(seed=common_seed, mode=True)

        if device == DeviceType.NPU:
            # Enable binary compilation for NPU
            torch.npu.set_compile_mode(jit_compile=False)

        get_logger().info("==========QUANTIZATION: Prepare Dataset==========")

        dataset_path = quant_config.spec.dataset
        # Set default_text to dataset_loader
        self.dataset_loader.default_text = quant_config.spec.default_text
        dataset = self.dataset_loader.get_dataset_by_name(dataset_path)
        self._validate_loaded_samples_text(dataset)
        get_logger().info("Prepared dataset from %s successfully", dataset_path)

        final_process_cfg = quant_config.spec.process.copy()

        # Note: MoE conversion is now handled automatically in model_adapter during layer loading
        # No need for separate MoeConverterProcessor

        if save_path is not None:
            get_logger().info("==========QUANTIZATION: Prepare Persistence==========")
            for save_cfg in quant_config.spec.save:
                save_cfg.set_save_directory(save_path)

            # Register save processors
            final_process_cfg += quant_config.spec.save
            get_logger().info("Prepared persistence to %s successfully", save_path)

        get_logger().info("==========QUANTIZATION: Run Quantization==========")

        runner_type = self._choose_runner_type(quant_config, model_adapter, device_indices)
        offload_device = self._resolve_offload_device(model_adapter)

        if runner_type == RunnerType.DP_LAYER_WISE:
            from msmodelslim.core.runner.dp_layer_wise_runner import DPLayerWiseRunner

            runner = DPLayerWiseRunner(adapter=model_adapter, offload_device=offload_device)
        elif runner_type == RunnerType.LAYER_WISE:
            runner = LayerWiseRunner(adapter=model_adapter, offload_device=offload_device)
        else:
            raise UnsupportedError(
                f"Invalid runner type for {self.backend_name}: {runner_type}",
                action="Please use RunnerType.LAYER_WISE or RunnerType.DP_LAYER_WISE",
            )

        ctx = self.context_factory.create(is_distributed=(runner_type == RunnerType.DP_LAYER_WISE))
        get_logger().info("Created runner %s successfully", type(runner).__name__)
        with ContextManager(ctx=ctx):
            # Add all processors
            for process_cfg in final_process_cfg:
                runner.add_processor(processor_cfg=process_cfg)

            # Run quantization (device_indices enables DP spawn when runner is DP)
            runner.run(calib_data=dataset, device=device, device_indices=device_indices)
            get_logger().info("==========QUANTIZATION: END==========")

        # Save context if persistence is provided
        if self.debug_info_persistence is not None:
            get_logger().info("==========SAVE CONTEXT DEBUG INFO==========")
            try:
                self.debug_info_persistence.save_from_context(ctx=ctx)
            except Exception as e:
                get_logger().warning("Failed to save debug info: %s", e)


def get_plugin():
    """
    获取 multimodal_vlm_modelslim_v1 量化服务插件（返回配置类与组件类，由框架完成注册）。
    Returns:
        (MultimodalVLMModelslimV1QuantServiceConfig, MultimodalVLMModelslimV1QuantService) 元组
    """
    return MultimodalVLMModelslimV1QuantServiceConfig, MultimodalVLMModelslimV1QuantService
