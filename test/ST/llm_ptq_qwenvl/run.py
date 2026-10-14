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
# pylint: disable=no-name-in-module,pointless-string-statement

"""
导入相关依赖
"""
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
import requests  # noqa: E402, F401
import os  # noqa: E402

from msmodelslim.pytorch.llm_ptq.llm_ptq_tools import Calibrator, QuantConfig  # noqa: E402
from msmodelslim.pytorch.llm_ptq.anti_outlier import AntiOutlier, AntiOutlierConfig  # noqa: E402


torch.npu.set_compile_mode(jit_compile=False)
option = {}
option["NPU_FUZZY_COMPILE_BLACKLIST"] = "ReduceProd"
torch.npu.set_option(option)

"""
导入相关模型
"""
LOAD_PATH = f"{os.environ['PROJECT_PATH']}/resource/llm_ptq/qwen-vl/"
tokenizer = AutoTokenizer.from_pretrained(LOAD_PATH, trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained(
    LOAD_PATH, device_map='auto', trust_remote_code=True, torch_dtype=torch.float16
).eval()


images_list = os.listdir(f"{os.environ['PROJECT_PATH']}/resource/llm_ptq/coco2/")
image = Image.open(os.path.join(f"{os.environ['PROJECT_PATH']}/resource/llm_ptq/coco2/", images_list[0]))
prompt = "USER: <image>\nDescribe this image in detail. ASSISTANT:"
calib_data = []
for i in images_list[:1]:
    imagePath = os.path.join(f"{os.environ['PROJECT_PATH']}/resource/llm_ptq/coco2/", i)
    query = tokenizer.from_list_format(
        [
            {'image': imagePath},  # Either a local path or an url
            {'text': prompt},
        ]
    )
    inputs = tokenizer(query, return_tensors='pt')
    inputs = inputs.to(model.device)
    calib_data.append([inputs['input_ids'], None, inputs['attention_mask']])


disable_names = []
disable_names.append("transformer.visual.attn_pool.attn.out_proj")

"""
对于linear算子中的激活值如果有表示范围过大，或者"尖刺"的异常值过多，
需要使用anti outlier功能，使用方法如下
"""
anti_config = AntiOutlierConfig(
    a_bit=8,
    w_bit=8,
    anti_method="m2",
    dev_type="npu",
    dev_id=model.device.index,
)
anti_outlier = AntiOutlier(model, calib_data=calib_data, cfg=anti_config)
anti_outlier.process()


quant_config = QuantConfig(
    w_bit=8,
    a_bit=8,
    disable_names=disable_names,
    dev_type='npu',
    dev_id=model.device.index,
    act_method=2,
    mm_tensor=False,
)

calibrator = Calibrator(model, quant_config, calib_data=calib_data, disable_level='L0')
calibrator.run()
calibrator.save(f"{os.environ['PROJECT_PATH']}/output/llm_ptq_qwenvl")
