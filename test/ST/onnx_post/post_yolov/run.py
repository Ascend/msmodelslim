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
from msmodelslim.onnx.post_training_quant import QuantConfig, run_quantize
from msmodelslim import set_logger_level

set_logger_level("info")


quant_config = QuantConfig(is_dynamic_shape=True, input_shape=[[1, 3, 640, 640]])
input_model_path = f"{os.environ['PROJECT_PATH']}/resource/onnx_post/yolov5m.onnx"
output_model_path = f"{os.environ['PROJECT_PATH']}/output/onnx_post/yolov5m_quant.onnx"
run_quantize(input_model_path, output_model_path, quant_config)
