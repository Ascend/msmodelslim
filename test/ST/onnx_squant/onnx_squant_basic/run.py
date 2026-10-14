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

import cv2
import numpy as np
import os
from msmodelslim.onnx.squant_ptq import OnnxCalibrator, QuantConfig


def get_calib_data():
    img = cv2.imread(f"{os.environ['PROJECT_PATH']}/resource/onnx_squant/000000000139.jpg")
    img_data = cv2.resize(img, (224, 224))
    img_data = img_data[:, :, ::-1].transpose(2, 0, 1).astype(np.float32)
    img_data /= 255.0
    img_data = np.expand_dims(img_data, axis=0)
    return [[img_data]]


input_model = f"{os.environ['PROJECT_PATH']}/resource/onnx_squant/resnet50_official_1batch.onnx"
output_path = f"{os.environ['PROJECT_PATH']}/output/onnx_squant/resnet50_quant.onnx"


disable_names = []
config = QuantConfig(
    disable_names=disable_names, quant_mode=1, amp_num=0, disable_first_layer=False, disable_last_layer=False
)

calib_data = get_calib_data()
calib = OnnxCalibrator(input_model, config)
calib.run()
calib.export_quant_onnx(output_path)
del calib
