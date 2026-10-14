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
# pylint: disable=no-name-in-module,pointless-string-statement,redefined-outer-name,ungrouped-imports

"""
导入相关依赖
"""
import os  # noqa: E402
import json  # noqa: E402, F401
import torch  # noqa: E402
import torch_npu  # noqa: E402, F401
from torch_npu.contrib import transfer_to_npu  # noqa: E402, F401
from msmodelslim.pytorch.llm_ptq.anti_outlier import AntiOutlierConfig, AntiOutlier  # noqa: E402, F401
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig  # noqa: E402
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools import Calibrator, QuantConfig  # noqa: E402


torch.npu.set_compile_mode(jit_compile=False)
option = {}
option["NPU_FUZZY_COMPILE_BLACKLIST"] = "ReduceProd"
torch.npu.set_option(option)

"""
导入相关模型
"""
fp16_path = f"{os.environ['PROJECT_PATH']}/resource/llm_ptq/llama3-8b/"

config = AutoConfig.from_pretrained(pretrained_model_name_or_path=fp16_path, trust_remote_code=True)
config.num_hidden_layers = 2
config.use_cache = False
tokenizer = AutoTokenizer.from_pretrained(
    pretrained_model_name_or_path=fp16_path, config=config, trust_remote_code=True
)
model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path=fp16_path,
    config=config,
    torch_dtype='auto',
    device_map='auto',
    trust_remote_code=True,
).eval()


def get_calib_dataset(tokenizer, calib_list, device="npu"):
    calib_dataset = []
    for calib_data in calib_list:
        inputs = tokenizer(calib_data, return_tensors='pt')
        calib_dataset.append([inputs.data['input_ids'].to(device), inputs.data['attention_mask'].to(device)])
    return calib_dataset


calib_set = [
    "Where is the capital of China?",
    "Please make a poem:",
    "I want to learn python, how should I learn it?",
    "Please help me write a job report on large model inference optimization:",
    "What are the most worth visiting scenic spots in China?",
]
dataset_calib = get_calib_dataset(tokenizer, calib_set, device=model.device)

disable_names = []
quant_config = QuantConfig(
    w_bit=4,
    a_bit=4,
    disable_names=[],
    dev_type='npu',  # dev_type="npu", dev_id=0  如果需要使用npu进行量化
    dev_id=model.device.index,
    is_dynamic=True,
)
calibrator = Calibrator(model, quant_config, calib_data=dataset_calib, disable_level='L0')
calibrator.run()
calibrator.save(f"{os.environ['PROJECT_PATH']}/output/llm_ptq_w4a4", save_type=["numpy", "safe_tensor"])

print('Save quant weight success!')
