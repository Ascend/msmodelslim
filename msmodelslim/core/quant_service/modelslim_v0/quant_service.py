#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# pylint: disable=duplicate-code

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

import os
from pathlib import Path
from typing import Optional, List, Literal

import torch

from msmodelslim.core.const import DeviceType
from msmodelslim.core.quant_service import KeyInfoPersistenceInfra
from msmodelslim.model import IModel
from msmodelslim.pytorch.llm_ptq.anti_outlier import AntiOutlier, AntiOutlierConfig
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools import Calibrator, QuantConfig
from msmodelslim.utils.exception import SchemaValidateError
from msmodelslim.utils.logging import logger_setter, get_logger
from msmodelslim.utils.security import safe_copy_file
from msmodelslim.core.context.interface import IContextFactory
from .pipeline_interface import PipelineInterface
from .quant_config import ModelslimV0QuantConfig
from ..dataset_loader_infra import DatasetLoaderInfra
from ..interface import BaseQuantConfig, QuantServiceConfig, IQuantService


def copy_files(input_path, output_path):
    """
    复制模型配置文件
    @param input_path: 源目录
    @param output_path: 目标目录
    """
    for file in os.listdir(input_path):
        if not any((file.endswith(subfix) for subfix in ['.json', '.py'])):
            continue

        if any((file.endswith(subfix) for subfix in ['index.json'])):
            continue

        ori_file = os.path.join(input_path, file)
        dest_file = os.path.join(output_path, file)
        safe_copy_file(src_path=ori_file, dest_path=dest_file)
        os.chmod(dest_file, int("600", 8))


class ModelslimV0QuantServiceConfig(QuantServiceConfig):
    """modelslim_v0 量化服务配置，用于插件选择与 QuantService 初始化。"""

    apiversion: Literal["modelslim_v0"] = "modelslim_v0"


@logger_setter('msmodelslim.core.quant_service.modelslim_v0')  # 4-level: msmodelslim.core.quant_service.modelslim_v0
class ModelslimV0QuantService(IQuantService):
    backend_name: str = "modelslim_v0"

    def __init__(
        self,
        quant_service_config: ModelslimV0QuantServiceConfig,
        dataset_loader: DatasetLoaderInfra,
        context_factory: IContextFactory,
        debug_info_persistence: Optional[KeyInfoPersistenceInfra] = None,
        **kwargs,
    ):
        self.quant_service_config = quant_service_config
        self.dataset_loader = dataset_loader
        self.context_factory = context_factory
        self.debug_info_persistence = debug_info_persistence

    def quantize(
        self,
        quant_config: BaseQuantConfig,
        model_adapter: IModel,
        save_path: Optional[Path] = None,
        device: DeviceType = DeviceType.NPU,
        device_indices: Optional[List[int]] = None,
    ) -> None:
        if not isinstance(quant_config, BaseQuantConfig):
            raise SchemaValidateError("task must be a BaseTask", action='Please make sure the task is a BaseTask')
        if not isinstance(model_adapter, PipelineInterface):
            raise SchemaValidateError(
                "model must be a PipelineInterface", action='Please make sure the model is a PipelineInterface'
            )
        if save_path is not None and not isinstance(save_path, Path):
            raise SchemaValidateError(
                "save_path must be a Path or None", action='Please make sure the save_path is a Path or None'
            )
        if not isinstance(device, DeviceType):
            raise SchemaValidateError(
                "device must be a DeviceType", action='Please make sure the device is a DeviceType'
            )

        if device_indices is not None:
            get_logger().warning(
                "Specifying device indices is not supported in %s quant_service. Device indices will be ignored.",
                self.backend_name,
            )

        return self.quant_process(ModelslimV0QuantConfig.from_base(quant_config), model_adapter, save_path, device)

    def quant_process(
        self,
        quant_config: ModelslimV0QuantConfig,
        model_adapter: PipelineInterface,
        save_path: Optional[Path],
        device: DeviceType = DeviceType.NPU,
    ):
        # init
        if device == DeviceType.NPU:
            # 如果使用npu进行量化需开启二进制编译，避免在线编译算子
            torch.npu.set_compile_mode(jit_compile=False)

        # handle dataset
        get_logger().info("==========QUANTIZATION: PREPARE DATASET==========")
        anti_dataset = quant_config.spec.anti_dataset
        calib_dataset = quant_config.spec.calib_dataset
        batch_size = quant_config.spec.batch_size

        calib_data = None

        if calib_dataset is not None:
            get_logger().info("prepare calib_data from %s", calib_dataset)
            dataset = self.dataset_loader.get_dataset_by_name(calib_dataset)
            calib_data = model_adapter.handle_dataset(dataset=dataset, device=device)
            get_logger().info("prepare calib_data from %s success", calib_dataset)

        anti_data = calib_data

        if anti_dataset is not None:
            get_logger().info("prepare anti_data from %s", anti_dataset)
            dataset = self.dataset_loader.get_dataset_by_name(anti_dataset)
            anti_data = model_adapter.handle_dataset_by_batch(dataset=dataset, batch_size=batch_size, device=device)
            get_logger().info("prepare anti_data from %s success", anti_dataset)

        # load model
        get_logger().info("==========QUANTIZATION: LOAD MODEL==========")
        model = model_adapter.load_model(device=device)
        get_logger().info("load model from %s success", model_adapter.model_path)

        # anti outlier
        if quant_config.spec.anti_cfg is not None:
            get_logger().info("==========QUANTIZATION: ANTI OUTLIER==========")
            get_logger().debug("anti outlier config: %s", quant_config.spec.anti_cfg)
            anti_cfg = AntiOutlierConfig(dev_type=device.value, **quant_config.spec.anti_cfg)
            anti_outlier = AntiOutlier(model=model, calib_data=anti_data, cfg=anti_cfg, **quant_config.spec.anti_params)
            anti_outlier.process()
            get_logger().info("anti outlier success")

        # quantization
        get_logger().info("==========QUANTIZATION: CALIBRATION==========")
        get_logger().debug("calibration config: %s", quant_config.spec.calib_cfg)

        use_fa_quant = bool(quant_config.spec.calib_cfg.pop('use_fa_quant', False))
        fa_amp = quant_config.spec.calib_cfg.pop('fa_amp', 0)
        calib_cfg = QuantConfig(dev_type=device.value, **quant_config.spec.calib_cfg)
        if use_fa_quant:
            calib_cfg = calib_cfg.fa_quant(fa_amp=fa_amp)

        calibrator = Calibrator(model=model, cfg=calib_cfg, calib_data=calib_data, **quant_config.spec.calib_params)
        calibrator.run()
        get_logger().info("calibration success")

        # persist stage
        if not save_path:
            get_logger().warning("save_path is not provided, skip persist")
            get_logger().info("==========QUANTIZATION: END==========")
            return

        # persist
        get_logger().info("==========QUANTIZATION: PERSIST==========")
        calibrator.save(
            output_path=str(save_path),
            json_name='quant_model_description.json',
            save_type=['ascendV1'],
            **quant_config.spec.calib_save_params,
        )

        # copy .json and .py files
        copy_files(str(model_adapter.model_path), str(save_path))

        get_logger().info("==========QUANTIZATION: END==========")


def get_plugin():
    """
    获取 modelslim_v0 量化服务插件（返回配置类与组件类，由框架完成注册）。
    Returns:
        (ModelslimV0QuantServiceConfig, ModelslimV0QuantService) 元组
    """
    return ModelslimV0QuantServiceConfig, ModelslimV0QuantService
