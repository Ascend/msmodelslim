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
import os  # noqa: E402
import torch  # noqa: E402

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig  # noqa: E402
from msmodelslim.pytorch.llm_ptq.anti_outlier import AntiOutlierConfig, AntiOutlier  # noqa: E402
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools import Calibrator, QuantConfig  # noqa: E402


torch.npu.set_compile_mode(jit_compile=False)
option = {}
option["NPU_FUZZY_COMPILE_BLACKLIST"] = "ReduceProd"
torch.npu.set_option(option)

"""
导入相关模型
"""
LOAD_PATH = f"{os.environ['PROJECT_PATH']}/resource/llm_ptq/Hunyuan-A52B-Instruct-sorted/"
SAVE_PATH = f"{os.environ['PROJECT_PATH']}/output/llm_ptq_hooks_hunyuan/"

config = AutoConfig.from_pretrained(pretrained_model_name_or_path=LOAD_PATH, trust_remote_code=True)
config.num_hidden_layers = 2

tokenizer = AutoTokenizer.from_pretrained(
    pretrained_model_name_or_path=LOAD_PATH,
    config=config,
    use_fast=True,
    add_eos_token=True,
    trust_remote_code=True,
    local_files_only=True,
)

model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path=LOAD_PATH,
    config=config,
    low_cpu_mem_usage=True,
    torch_dtype='auto',
    device_map={
        "model.embed_tokens": 0,
        "model.layers": "cpu",
        "model.norm": "cpu",
        "lm_head": 0,
    },
    trust_remote_code=True,
    local_files_only=True,
)


def get_calib_dataset(tokenizer_instance, calib_list, device='cpu'):
    calib_dataset = []
    for calib_data in calib_list:
        inputs = tokenizer_instance(calib_data, return_tensors='pt')
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


"""
对于linear算子中的激活值如果有表示范围过大，或者"尖刺"的异常值过多，
需要使用anti outlier功能，使用方法如下
"""
anti_config = AntiOutlierConfig(
    w_bit=8,
    a_bit=8,
    anti_method="m4",
    dev_type='npu',
    dev_id=model.device.index,
)
anti_outlier = AntiOutlier(model, calib_data=dataset_calib, cfg=anti_config)
anti_outlier.process()

disable_names = []
for layer_index in range(config.num_hidden_layers):
    gate_name = f"model.layers.{layer_index}.mlp.gate.wg"
    disable_names.append(gate_name)


quant_config = QuantConfig(
    a_bit=8,
    w_bit=8,
    disable_names=disable_names,
    dev_type='npu',
    dev_id=model.device.index,
    act_method=1,
    pr=1.0,
    w_sym=True,
    mm_tensor=False,
)

mix_cfg = {"*.experts.*": "w8a8_dynamic", "*": "w8a8"}

calibrator = Calibrator(model, quant_config, calib_data=dataset_calib, disable_level='L0', mix_cfg=mix_cfg)
calibrator.run()
calibrator.save(SAVE_PATH, save_type=["numpy", "safe_tensor"], part_file_size=4)
