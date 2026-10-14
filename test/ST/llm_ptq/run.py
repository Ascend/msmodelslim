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

LOAD_PATH = f"{os.environ['PROJECT_PATH']}/resource/llm_ptq/llama2_7b/"
tokenizer = AutoTokenizer.from_pretrained(
    pretrained_model_name_or_path=LOAD_PATH, trust_remote_code=True, local_files_only=True
)
model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path=LOAD_PATH, torch_dtype=torch.float32, trust_remote_code=True, local_files_only=True
).cpu()

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

from msmodelslim.pytorch.llm_ptq.anti_outlier import AntiOutlierConfig, AntiOutlier  # noqa: E402

anti_config = AntiOutlierConfig(anti_method="m1")
anti_outlier = AntiOutlier(model, calib_data=dataset_calib, cfg=anti_config)
msmodelslim_logger.info(anti_outlier.norm_linear_subgraph)
anti_outlier.process()
msmodelslim_logger.info('m1 antioutlier is finished!')

quant_config = QuantConfig(w_bit=8, dev_type='cpu', act_method=3, pr=0.5, mm_tensor=False)
calibrator = Calibrator(model, quant_config, calib_data=dataset_calib, disable_level='L0')
calibrator.run()

calibrator.save(f"{os.environ['PROJECT_PATH']}/output/llm_ptq", save_type=["numpy", "safe_tensor"], part_file_size=1)
msmodelslim_logger.info('Save quant weight success!')
