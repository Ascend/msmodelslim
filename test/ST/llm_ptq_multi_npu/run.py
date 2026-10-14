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

import os
import torch
import torch_npu  # noqa: F401
import torch.utils.data
from transformers import AutoTokenizer, AutoModelForCausalLM

SEQ_LEN_OUT = 32

torch.npu.set_compile_mode(jit_compile=False)
option = {}
option["NPU_FUZZY_COMPILE_BLACKLIST"] = "ReduceProd"
torch.npu.set_option(option)

LOAD_PATH = f"{os.environ['PROJECT_PATH']}/resource/llm_ptq/llama2_7b/"
tokenizer = AutoTokenizer.from_pretrained(
    pretrained_model_name_or_path=LOAD_PATH, trust_remote_code=True, local_files_only=True
)
model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path=LOAD_PATH,
    torch_dtype=torch.float16,
    trust_remote_code=True,
    device_map="auto",
    local_files_only=True,
).eval()

'''
# for 中文模型
calib_list = ["中国的首都在哪里？",
              "请做一首诗歌：",
              "我想要学习python，该怎么学习？",
              "请帮我写一篇关于大模型推理优化的任职报告：",
              "中国最值得去的几个景点"]
'''

calib_list = [
    "Where is the capital of China?",
    "Please make a poem:",
    "I want to learn python, how should I learn it?",
    "Please help me write a job report on large model inference optimization:",
    "What are the most worth visiting scenic spots in China?",
]


def get_calib_dataset(tokenizer, calib_list):
    calib_dataset = []
    for calib_data in calib_list:
        inputs = tokenizer([calib_data], return_tensors='pt')
        print(inputs)
        calib_dataset.append(
            [inputs.data['input_ids'].to(model.device), inputs.data['attention_mask'].to(model.device)]
        )
    return calib_dataset


dataset_calib = get_calib_dataset(tokenizer, calib_list)

from msmodelslim.pytorch.llm_ptq.llm_ptq_tools import Calibrator, QuantConfig  # noqa: E402
from msmodelslim.pytorch.llm_ptq.anti_outlier import AntiOutlier, AntiOutlierConfig  # noqa: E402

anti_config = AntiOutlierConfig(anti_method="m1", dev_type='npu', dev_id=model.device.index)
anti_outlier = AntiOutlier(model, calib_data=dataset_calib, cfg=anti_config)
anti_outlier.process()

quant_config = QuantConfig(w_bit=8, dev_type='npu', dev_id=model.device.index, act_method=3, pr=0.5, mm_tensor=False)
calibrator = Calibrator(model, quant_config, calib_data=dataset_calib, disable_level='L0')
calibrator.run()

'''
print("testing quantized weights...")
test_prompt = "Common sense questions and answers\n\nQuestion: How to learn a new language\nFactual answer:"
test_input = tokenizer(test_prompt, return_tensors="pt")
print("model is inferring...")
model = model.cpu()
model.eval()
generate_ids = model.generate(test_input.input_ids.cpu(), attention_mask=test_input.attention_mask.cpu(), max_new_tokens=SEQ_LEN_OUT)
res = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
print(res)
for idx, item in enumerate(res):
    print(item)
'''

calibrator.save(f"{os.environ['PROJECT_PATH']}/output/llm_ptq_multi_npu", save_type=["numpy", "safe_tensor"])
print("Save quant weight success!")
