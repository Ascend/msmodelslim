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
# pylint: disable=no-name-in-module,ungrouped-imports

import os
import torch
import torch_npu  # noqa: F401
import torch.utils.data
from transformers import AutoTokenizer, AutoModelForCausalLM

from msmodelslim import logger as msmodelslim_logger

SEQ_LEN_OUT = 32

default_device = torch.npu.current_device()
torch.npu.get_device_name(default_device)

torch.npu.set_compile_mode(jit_compile=False)
option = {}
option["NPU_FUZZY_COMPILE_BLACKLIST"] = "ReduceProd"
torch.npu.set_option(option)

LOAD_PATH = f"{os.environ['PROJECT_PATH']}/resource/llm_ptq/llama2_7b/"
tokenizer = AutoTokenizer.from_pretrained(
    pretrained_model_name_or_path=LOAD_PATH, trust_remote_code=True, local_files_only=True
)
model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path=LOAD_PATH, torch_dtype=torch.float16, trust_remote_code=True, local_files_only=True
).npu()

calib_list_all = [
    "Where is the capital of China?",
    "Please make a poem:",
    "I want to learn python, how should I learn it?",
    "Please help me write a job report on large model inference optimization:",
    "What are the most worth visiting scenic spots in China?",
]


def get_calib_dataset(tokenizer_instance, calib_list):
    calib_dataset = []
    for calib_data in calib_list:
        inputs = tokenizer_instance([calib_data], return_tensors='pt')
        msmodelslim_logger.info(inputs)
        calib_dataset.append([inputs.data['input_ids'].npu(), inputs.data['attention_mask'].npu()])
    return calib_dataset


dataset_calib = get_calib_dataset(tokenizer, calib_list_all)

from msmodelslim.pytorch.llm_ptq.llm_ptq_tools import Calibrator, QuantConfig  # noqa: E402

quant_config = QuantConfig(
    a_bit=8,
    w_bit=8,
    disable_names=[
        'transformer.encoder.layers.0.self_attention.query_key_value',
        'transformer.encoder.layers.0.self_attention.dense',
        'transformer.encoder.layers.0.mlp.dense_h_to_4h',
    ],
    dev_type='npu',  # 在cpu进行量化时，需要配置参数dev_type='cpu'，并取消参数dev_id=model.device.index的配置
    dev_id=model.device.index,
    act_method=3,
    pr=1.0,
    mm_tensor=False,
    use_kvcache_quant=True,
)

calibrator = Calibrator(model, quant_config, calib_data=dataset_calib, disable_level='L0')
calibrator.run()

msmodelslim_logger.info("testing quantized weights...")
test_prompt = "Common sense questions and answers\n\nQuestion: How to learn a new language\nFactual answer:"
test_input = tokenizer(test_prompt, return_tensors="pt")
msmodelslim_logger.info("model is inferring...")
model = model.npu()
model.eval()
generate_ids = model.generate(
    test_input.input_ids.npu(), attention_mask=test_input.attention_mask.npu(), max_new_tokens=SEQ_LEN_OUT
)
res = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
msmodelslim_logger.info(res)
for _, item in enumerate(res):
    msmodelslim_logger.info(item)

calibrator.save(f"{os.environ['PROJECT_PATH']}/output/llm_ptq_kvcache", save_type=["safe_tensor"])
msmodelslim_logger.info('Save quant weight success!')
