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
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import torch
from PIL import Image
import requests  # noqa: F401
from msmodelslim.pytorch.llm_ptq.anti_outlier import AntiOutlier, AntiOutlierConfig
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools import Calibrator, QuantConfig


LOAD_PATH = f"{os.environ['PROJECT_PATH']}/resource/llm_ptq/llava_weight/"
processor = LlavaNextProcessor.from_pretrained(LOAD_PATH, local_files_only=True)
model = LlavaNextForConditionalGeneration.from_pretrained(
    LOAD_PATH, torch_dtype=torch.float32, low_cpu_mem_usage=True, local_files_only=True
).cpu()


images_list = os.listdir(f"{os.environ['PROJECT_PATH']}/resource/llm_ptq/coco2/")
image = Image.open(os.path.join(f"{os.environ['PROJECT_PATH']}/resource/llm_ptq/coco2/", images_list[0]))
prompt = "USER: <image>\nDescribe this image in detail. ASSISTANT:"
calib_data = []
for i in images_list[:1]:
    image = Image.open(os.path.join(f"{os.environ['PROJECT_PATH']}/resource/llm_ptq/coco2/", i))
    data_1 = processor(images=image, text=prompt, return_tensors="pt").to('cpu')
    calib_data.append(
        [
            data_1.data['input_ids'],
            data_1.data['pixel_values'],
            torch.tensor([[480, 640]], dtype=torch.int64),
            data_1.data['attention_mask'],
        ]
    )

disable_names = []

anti_config = AntiOutlierConfig(a_bit=16, w_bit=8, anti_method="m3", dev_type="cpu", w_sym=True)
anti_outlier = AntiOutlier(model, calib_data=calib_data, cfg=anti_config, norm_class_name='LlamaRMSNorm')
anti_outlier.process()

quant_conf = QuantConfig(
    w_bit=8,
    a_bit=16,
    disable_names=disable_names,
    dev_type='cpu',
    act_method=1,
    pr=1.0,
    nonuniform=False,
    w_sym=True,
    mm_tensor=False,
)

calibrator = Calibrator(model, quant_conf, calib_data=calib_data, disable_level='L0')
calibrator.run()
calibrator.save(f"{os.environ['PROJECT_PATH']}/output/llm_ptq_multi_model")
image.close()
