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

from pathlib import Path
from typing import Dict, Optional, Literal, List, Type

import torch
from msmodelslim.processor.save.processor import QuantSaveProcessorConfig

from msmodelslim.processor.base import AutoProcessorConfig
from msmodelslim.core.const import RunnerType, DeviceType
from msmodelslim.core.context import ContextManager, IContextFactory
from msmodelslim.core.quant_service import KeyInfoPersistenceInfra
from msmodelslim.core.quant_service.dataset_loader_infra import DatasetLoaderInfra
from msmodelslim.core.runner.layer_wise_runner import LayerWiseRunner
from msmodelslim.core.runner.pipeline_interface import PipelineInterface
from msmodelslim.core.runner.pipeline_parallel_runner import PPRunner
from msmodelslim.model import IModel
from msmodelslim.utils.exception import SchemaValidateError, UnsupportedError
from msmodelslim.utils.logging import get_logger, logger_setter
from msmodelslim.utils.seed import seed_all
from .quant_config import ModelslimV1QuantConfig
from msmodelslim.format.base import QuantFormatConfig
from .save import AscendV1Config, MindIEFormatConfig
from ..interface import BaseQuantConfig, QuantServiceConfig, IQuantService


class ModelslimV1QuantServiceConfig(QuantServiceConfig):
    """modelslim_v1 量化服务配置，用于插件选择与 QuantService 初始化。"""

    apiversion: Literal["modelslim_v1"] = "modelslim_v1"


@logger_setter(
    prefix='msmodelslim.core.quant_service.modelslim_v1'
)  # 4-level: msmodelslim.core.quant_service.modelslim_v1
class ModelslimV1QuantService(IQuantService):
    backend_name: str = "modelslim_v1"
    #: 旧 format type → 旧 processor config（字段与对应 QuantFormatConfig 子类对齐）
    SAVE_FORMAT_PROCESSOR_ROUTING: Dict[str, Type[AutoProcessorConfig]] = {
        "ascendv1_saver": AscendV1Config,
        "mindie_format_saver": MindIEFormatConfig,
    }

    def _build_save_processor_config(
        self, format_cfg: QuantFormatConfig, save_path: Optional[Path] = None
    ) -> AutoProcessorConfig:
        processor_cls = self.SAVE_FORMAT_PROCESSOR_ROUTING.get(format_cfg.type)
        if processor_cls is None:
            proc = QuantSaveProcessorConfig(type="saver", format=format_cfg)
            if save_path is not None:
                proc.set_save_directory(str(save_path))
            return proc

        proc = processor_cls.model_validate(format_cfg.model_dump(mode="python"))
        if save_path is not None:
            proc.set_save_directory(str(save_path))
        return proc

    def __init__(
        self,
        quant_service_config: ModelslimV1QuantServiceConfig,
        dataset_loader: DatasetLoaderInfra,
        context_factory: IContextFactory,
        debug_info_persistence: Optional[KeyInfoPersistenceInfra] = None,
        **kwargs,
    ):
        self.quant_service_config = quant_service_config
        self.dataset_loader = dataset_loader
        self.context_factory = context_factory
        self.debug_info_persistence = debug_info_persistence

    @staticmethod
    def _choose_runner_type(
        quant_config: ModelslimV1QuantConfig,
        model_adapter: PipelineInterface,
        device_indices: Optional[List[int]] = None,
    ) -> Literal[RunnerType.MODEL_WISE, RunnerType.LAYER_WISE]:
        """根据模型和配置确定使用的pipeline类型。

        Args:
            quant_config: 量化配置
            model_adapter: 模型适配器

        Returns:
            Literal['model_wise', 'layer_wise']: 确定的pipeline类型
        """
        if quant_config.spec.runner == RunnerType.MODEL_WISE:
            get_logger().info("Model-wise runner detected, using model-wise pipeline.")
            return RunnerType.MODEL_WISE

        if quant_config.spec.runner == RunnerType.LAYER_WISE:
            get_logger().info("Layer-wise runner detected, using layer-wise pipeline.")
            return RunnerType.LAYER_WISE

        if quant_config.spec.runner == RunnerType.DP_LAYER_WISE:
            get_logger().info("Distributed layer-wise runner detected, using distributed layer-wise pipeline.")
            return RunnerType.DP_LAYER_WISE

        if quant_config.spec.runner == RunnerType.AUTO and device_indices is not None and len(device_indices) > 1:
            get_logger().info("multi device configuration detected, using distributed layer-wise pipeline.")
            return RunnerType.DP_LAYER_WISE

        get_logger().info("Runner type not detected, using layer-wise pipeline.")
        return RunnerType.LAYER_WISE

    def quantize(
        self,
        quant_config: BaseQuantConfig,
        model_adapter: IModel,
        save_path: Optional[Path] = None,
        device: DeviceType = DeviceType.NPU,
        device_indices: Optional[List[int]] = None,
    ):
        if not isinstance(quant_config, BaseQuantConfig):
            raise SchemaValidateError(
                "task is NOT BaseQuantConfig", action="Please make sure the task is a BaseQuantConfig"
            )
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
            ModelslimV1QuantConfig.from_base(quant_config), model_adapter, save_path, device, device_indices
        )

    def quant_process(
        self,
        quant_config: ModelslimV1QuantConfig,
        model_adapter: PipelineInterface,
        save_path: Optional[Path],
        device: DeviceType = DeviceType.NPU,
        device_indices: Optional[List[int]] = None,
    ):
        # clear quant_model_path before quantization
        if save_path and save_path.exists():
            # 仅清理 safetensors 文件，保留其他文件与目录
            for item in save_path.iterdir():
                if item.is_file() and item.suffix == ".safetensors":
                    item.unlink()
            get_logger().info("Cleared safetensors under save_path: %s", save_path)

        common_seed = 42
        seed_all(seed=common_seed, mode=True)

        if device == DeviceType.NPU:
            # 如果使用npu进行量化需开启二进制编译，避免在线编译算子
            torch.npu.set_compile_mode(jit_compile=False)

        if save_path is not None:
            get_logger().info("==========QUANTIZATION: Prepare Persistence==========")
            for save_cfg in quant_config.spec.save:
                save_cfg.set_save_directory(save_path)
            get_logger().info("prepare Persistence to %s success", save_path)

        runner_type = self._choose_runner_type(quant_config, model_adapter, device_indices)
        ctx = self.context_factory.create(is_distributed=(runner_type == RunnerType.DP_LAYER_WISE))

        def _create_runner():
            if runner_type == RunnerType.MODEL_WISE:
                return PPRunner(adapter=model_adapter)
            if runner_type == RunnerType.LAYER_WISE:
                return LayerWiseRunner(adapter=model_adapter)
            if runner_type == RunnerType.DP_LAYER_WISE:
                from msmodelslim.core.runner.dp_layer_wise_runner import DPLayerWiseRunner

                return DPLayerWiseRunner(adapter=model_adapter)
            raise UnsupportedError(
                "Invalid runner type", action="Please use RunnerType.MODEL_WISE or RunnerType.LAYER_WISE"
            )

        with ContextManager(ctx):
            # 前置阶段：每阶段独立 process + dataset，结果通过 get_current_context() 传递到主阶段
            for idx, prior_stage in enumerate(quant_config.spec.prior):
                get_logger().info(
                    "==========QUANTIZATION: Prior Stage %s/%s==========", idx + 1, len(quant_config.spec.prior)
                )
                prior_dataset_name = prior_stage.dataset if prior_stage.dataset else quant_config.spec.dataset
                if prior_stage.dataset:
                    get_logger().info("Prior stage dataset specified, using dataset: %s", prior_dataset_name)
                else:
                    get_logger().info(
                        "Prior stage dataset not provided, fallback to spec.dataset: %s", prior_dataset_name
                    )
                prior_dataset = self.dataset_loader.get_dataset_by_name(prior_dataset_name)
                get_logger().info("prepare dataset from %s success", prior_dataset_name)
                runner = _create_runner()
                for process_cfg in prior_stage.process:
                    runner.add_processor(processor_cfg=process_cfg)
                runner.run(calib_data=prior_dataset, device=device, device_indices=device_indices)

            # 主阶段：process + save，使用 spec.dataset（可读取 prior 写入的 context）
            get_logger().info("==========QUANTIZATION: Run Quantization==========")
            dataset = self.dataset_loader.get_dataset_by_name(quant_config.spec.dataset)
            get_logger().info("prepare dataset from %s success", quant_config.spec.dataset)
            get_logger().info("Create runner %s success", runner_type)
            runner = _create_runner()
            for process_cfg in quant_config.spec.process:
                runner.add_processor(processor_cfg=process_cfg)
            if save_path is not None:
                for save_cfg in quant_config.spec.save:
                    runner.add_processor(processor_cfg=self._build_save_processor_config(save_cfg, save_path))
            runner.run(calib_data=dataset, device=device, device_indices=device_indices)
        get_logger().info("==========QUANTIZATION: END==========")

        # Save context if persistence is provided
        if self.debug_info_persistence is not None:
            get_logger().info("==========SAVE DEBUG INFO==========")
            try:
                self.debug_info_persistence.save_from_context(ctx=ctx)
            except Exception as e:
                get_logger().warning("Failed to save debug info: %s", e)


def get_plugin():
    """
    获取 modelslim_v1 量化服务插件（返回配置类与组件类，由框架完成注册）。
    Returns:
        (ModelslimV1QuantServiceConfig, ModelslimV1QuantService) 元组
    """
    return ModelslimV1QuantServiceConfig, ModelslimV1QuantService
