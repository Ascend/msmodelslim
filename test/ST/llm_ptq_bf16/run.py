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
# pylint: disable=no-name-in-module,ungrouped-imports,unnecessary-comprehension

import os
import stat  # noqa: F401
import shutil  # noqa: F401
import json  # noqa: F401
from msmodelslim.pytorch.llm_ptq.anti_outlier import AntiOutlier, AntiOutlierConfig  # noqa: F401
import torch
import torch.utils.data
from transformers import AutoTokenizer, AutoModelForCausalLM
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools import Calibrator, QuantConfig

IN_MODEL_PATH = f"{os.environ['PROJECT_PATH']}/resource/llm_ptq/llama2_7b/"
OUT_MODEL_PATH = f"{os.environ['PROJECT_PATH']}/output/llm_ptq_bf16"
NUM_LAYERS = 2  #
ANTI_METHOD = "m1"


def get_calib_dataset(_tokenizer, _calib_list):
    calib_dataset = []
    for calib_data in _calib_list:
        inputs = _tokenizer([calib_data], return_tensors='pt')
        print(inputs)
        calib_dataset.append([inputs.data['input_ids'].cpu(), None, inputs.data['attention_mask'].cpu()])
    return calib_dataset


tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=IN_MODEL_PATH, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path=IN_MODEL_PATH, torch_dtype=torch.bfloat16, use_safetensors=True, local_files_only=True
).cpu()
print("loading success!")

calib_list = [
    "Where is the capital of China?",
]

dataset_calib = get_calib_dataset(tokenizer, calib_list)

print("quantization start...")
disabled_names = []
disabled_layers = [i for i in range(0, NUM_LAYERS)]
for i in disabled_layers:
    disabled_names.append(f"model.layers.{i}.mlp.down_proj")

quant_config = QuantConfig(
    a_bit=8, w_bit=8, disable_names=disabled_names, dev_type='cpu', act_method=3, mm_tensor=False
)

calibrator = Calibrator(model, quant_config, calib_data=dataset_calib, disable_level='L0')

calibrator.run()
print("quantization success!")

calibrator.save(OUT_MODEL_PATH, save_type=["numpy", "safe_tensor"])

print("saved successfully")
