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
# pylint: disable=no-name-in-module,pointless-string-statement,redefined-outer-name

"""
导入相关依赖
"""
import os  # noqa: E402
import torch  # noqa: E402
import torch.utils.data  # noqa: E402
from transformers import AutoTokenizer, AutoModelForCausalLM  # noqa: E402


torch.npu.set_compile_mode(jit_compile=False)
option = {}
option["NPU_FUZZY_COMPILE_BLACKLIST"] = "ReduceProd"
torch.npu.set_option(option)

SEQ_LEN_OUT = 32

"""
导入相关模型
"""
LOAD_PATH = f"{os.environ['PROJECT_PATH']}/resource/llm_ptq/llama2_7b/"
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=LOAD_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path=LOAD_PATH, torch_dtype=torch.float32, trust_remote_code=True
).cpu()

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
        inputs = tokenizer([calib_data], return_tensors='pt').to('cpu')
        print(inputs)
        calib_dataset.append([inputs.data['input_ids'], inputs.data['attention_mask']])
    return calib_dataset


dataset_calib = get_calib_dataset(tokenizer, calib_list)

from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.layer_select import LayerSelector  # noqa: E402

ls = LayerSelector(model, range_method='quantile')
ls.run(dataset_calib)
layers = ls.select_layers_by_threshold(1)
print("layer select by threshold:", layers)
layers = ls.select_layers_by_disable_level(2)
print("layer select by disable level:", layers)
