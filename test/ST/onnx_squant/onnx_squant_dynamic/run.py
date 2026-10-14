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
from msmodelslim.onnx.squant_ptq import OnnxCalibrator, QuantConfig

input_model = f"{os.environ['PROJECT_PATH']}/resource/onnx_squant/swin_tiny_patch4_window7_224_dynamic.onnx"
output_path = f"{os.environ['PROJECT_PATH']}/output/onnx_squant/swin_tiny_patch4_window7_224_dynamic_quant.onnx"


disable_names = []
config = QuantConfig(
    disable_names=disable_names,
    quant_mode=1,
    amp_num=0,
    disable_first_layer=False,
    disable_last_layer=False,
    is_dynamic_shape=True,
    input_shape=[[1, 3, 224, 224]],
)


calib = OnnxCalibrator(input_model, config)
calib.run()
calib.export_quant_onnx(output_path)
del calib
