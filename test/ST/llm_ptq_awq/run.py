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
# pylint: disable=no-name-in-module

import json
import os
import torch.utils.data
from transformers import AutoTokenizer, AutoModelForCausalLM
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools import Calibrator, QuantConfig
from msmodelslim.pytorch.llm_ptq.anti_outlier import AntiOutlier, AntiOutlierConfig


# for local path
fp16_path = f"{os.environ['PROJECT_PATH']}/resource/llm_ptq/llama2_7b/"  # 原始模型路径，其中的内容如下图
tokenizer = AutoTokenizer.from_pretrained(
    pretrained_model_name_or_path=fp16_path, trust_remote_code=True, local_files_only=True
)
model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path=fp16_path, torch_dtype=torch.float32, trust_remote_code=True, local_files_only=True
).cpu()


# 获取校准数据函数定义
def get_calib_dataset(fp16_tokenizer, calib_list):
    calib_dataset = []
    for calib_data in calib_list:
        text = calib_data['inputs_pretokenized']
        inputs = fp16_tokenizer([text], return_tensors='pt').to('cpu')
        print(inputs)
        calib_dataset.append([inputs.data['input_ids'], inputs.data['token_type_ids'], inputs.data['attention_mask']])
    return calib_dataset


entry = f"{os.environ['PROJECT_PATH']}/resource/CEval/val/Social_Science/teacher_qualification.jsonl"
dataset = []
with open(entry, encoding='utf-8') as file:
    for line in file:
        dataset.append(json.loads(line))

dataset_calib = get_calib_dataset(tokenizer, dataset[:1])

disable_names = []
disable_names.append('lm_head')

model.eval()
w_sym = False
anti_config = AntiOutlierConfig(a_bit=16, w_bit=8, anti_method="m3", dev_type="cpu", w_sym=w_sym)
anti_outlier = AntiOutlier(model, calib_data=dataset_calib, cfg=anti_config)
anti_outlier.process()

# w_sym=True：对称量化，w_sym=False：非对称量化
quant_config = QuantConfig(
    a_bit=16, w_bit=8, disable_names=disable_names, dev_type='cpu', w_sym=w_sym, mm_tensor=False, w_method='MinMax'
)
calibrator = Calibrator(model, quant_config, calib_data=[], disable_level='L0')
calibrator.run()  # 执行PTQ量化校准

calibrator.save(f"{os.environ['PROJECT_PATH']}/output/llm_ptq_awq", save_type=["numpy", "safe_tensor"])
