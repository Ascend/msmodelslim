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

import os

from msmodelslim.pytorch.ra_compression import RACompressConfig, RACompressor
from transformers import AutoTokenizer, AutoModelForCausalLM

config = RACompressConfig(theta=0.00001, alpha=100)
input_model_path = f"{os.environ['PROJECT_PATH']}/resource/llm_ptq/baichuan2-13b/"
output_model_path = f"{os.environ['PROJECT_PATH']}/output/ra_compression_baichuan/win.pt"
tokenizer = AutoTokenizer.from_pretrained(
    pretrained_model_name_or_path=input_model_path, trust_remote_code=True, local_files_only=True
)
model = (
    AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=input_model_path, trust_remote_code=True, local_files_only=True
    )
    .float()
    .cpu()
)
ra = RACompressor(model, config)
ra.get_alibi_windows(output_model_path)
print('success!')
