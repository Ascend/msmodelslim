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
import torch  # noqa: E402
import torch_npu  # noqa: E402, F401
import torch.utils.data  # noqa: E402
from transformers import AutoTokenizer, AutoModelForCausalLM  # noqa: E402

from msmodelslim.pytorch.llm_ptq.llm_ptq_tools import Calibrator, QuantConfig  # noqa: E402
from msmodelslim.pytorch.llm_ptq.anti_outlier import AntiOutlier, AntiOutlierConfig  # noqa: E402, F401

SEQ_LEN_OUT = 32

torch.npu.set_compile_mode(jit_compile=False)
option = {}
option["NPU_FUZZY_COMPILE_BLACKLIST"] = "ReduceProd"
torch.npu.set_option(option)

"""
导入相关模型
"""
LOAD_PATH = f"{os.environ['PROJECT_PATH']}/resource/llm_ptq/llama3.1_8b/"
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=LOAD_PATH, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path=LOAD_PATH,
    torch_dtype='auto',
    trust_remote_code=True,
).npu()

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

disable_names = []
llama_layers = 1
disable_idx_lst = list(range(llama_layers))
for layer_index in disable_idx_lst:
    down_proj_name = "model.layers.{}.mlp.down_proj".format(layer_index)
    disable_names.append(down_proj_name)
disable_names.append('lm_head')

"""
对于linear算子中的激活值如果有表示范围过大，或者"尖刺"的异常值过多，
需要使用anti outlier功能，使用方法如下
"""
quant_config = QuantConfig(
    a_bit=8,
    w_bit=8,
    disable_names=disable_names,
    dev_type='npu',
    dev_id=0,
    act_method=3,
    pr=1.0,
    w_sym=True,
    mm_tensor=False,
).fa_quant(fa_amp=0)
calibrator = Calibrator(model, quant_config, calib_data=dataset_calib, disable_level='L0')
calibrator.run()

print("testing quantized weights...")
test_prompt = "Common sense questions and answers\n\nQuestion: How to learn a new language\nFactual answer:"
test_input = tokenizer(test_prompt, return_tensors="pt")
print("model is inferring...")
model = model.cpu()
model.eval()
generate_ids = model.generate(
    test_input.input_ids.cpu(), attention_mask=test_input.attention_mask.cpu(), max_new_tokens=SEQ_LEN_OUT
)
res = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
print(res)
for idx, item in enumerate(res):
    print(item)

calibrator.save(f"{os.environ['PROJECT_PATH']}/output/llm_ptq_fa3/", save_type=["safe_tensor"])
print('Save quant weight success!')
