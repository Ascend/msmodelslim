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
from msmodelslim.onnx.squant_ptq import OnnxCalibrator, QuantConfig  # 导入squant_ptq量化接口
from msmodelslim import set_logger_level  # 可选，导入日志配置接口

set_logger_level("info")
config = QuantConfig(is_dynamic_shape=True, input_shape=[[1, 3, 640, 640]])

input_model_path = (
    f"{os.environ['PROJECT_PATH']}/resource/onnx_squant/yolov5m.onnx"  # 配置待量化模型的输入路径，请根据实际路径配置
)
output_model_path = f"{os.environ['PROJECT_PATH']}/output/onnx_squant/yolov5m_quant.onnx"  # 配置量化后模型的名称及输出路径，请根据实际路径配置

calib = OnnxCalibrator(input_model_path, config)  # 使用OnnxCalibrator接口，输入待量化模型路径，量化
calib.run()  # 执行量化
calib.export_quant_onnx(output_model_path)  # 导出量化后模
