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

from typing import List, Optional, Any

import torch
from torch import nn

from msmodelslim.core.base.protocol import DataUnit
from msmodelslim.core.const import DeviceType
from msmodelslim.core.runner.generated_runner import GeneratedRunner, get_input_datas
from msmodelslim.core.runner.pipeline_interface import PipelineInterface
from msmodelslim.processor import LoadProcessorConfig, AutoProcessorConfig
from msmodelslim.utils.logging import logger_setter, get_logger


@logger_setter()
class LayerWiseRunner(GeneratedRunner):
    """
    "Module-Processor-Dataset" Mode Runner
    WARNING: Module will be REMOVED after handing, do NOT yield module more than once

    Calibrate quant perception:
    1、Only perceive the processors in front
    2、Perceive the calibration results of all processors in front
    """

    def __init__(self, adapter: PipelineInterface, offload_device: str = "meta"):
        super().__init__(adapter)
        self.offload_device = offload_device

    def preprocess_processor(
        self, processor_list: List[AutoProcessorConfig], model: nn.Module, device: DeviceType = DeviceType.NPU
    ):
        if device == DeviceType.NPU:
            # 从当前设备获取实际设备索引
            current_device_idx = torch.npu.current_device()
            device_str = f"npu:{current_device_idx}"
            get_logger().debug(
                "Using device index %s from torch.npu.current_device()",
                current_device_idx,
            )
        else:
            device_str = device

        # 逐层上传推理设备
        processor_list.insert(0, LoadProcessorConfig(device=device_str, mode="load", post_offload=True))
        # 逐层释放推理设备内存
        processor_list.append(LoadProcessorConfig(device=self.offload_device, mode="offload", cleanup=True))

    def run(
        self,
        model: nn.Module = None,
        calib_data: Optional[List[Any]] = None,
        device: DeviceType = DeviceType.NPU,
        device_indices: Optional[List[int]] = None,
    ):
        _ = get_input_datas(self.adapter, calib_data, DeviceType.CPU)

        if model is None:
            get_logger().info('Start to init model')
            model = self.adapter.init_model(device=DeviceType.CPU)
            get_logger().info('Init model success')

        if device == DeviceType.NPU and device_indices:
            torch.npu.set_device(f"npu:{device_indices[0]}")

        processor_list = self.process_config_list.copy()
        self.preprocess_processor(processor_list, model, device=device)

        data_recorder = DataUnit(None, None)
        process_units = self.build_process_unit(
            processor_list, model=model, adapter=self.adapter, calib_data=calib_data, data_recorder=data_recorder
        )

        self.generated_schedule(process_units, model, calib_data)
