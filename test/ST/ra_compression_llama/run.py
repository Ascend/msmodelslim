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
import torch
from msmodelslim.pytorch.ra_compression import RARopeCompressConfig, RARopeCompressor
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch_npu  # noqa: F401

torch.npu.set_compile_mode(jit_compile=False)

config = RARopeCompressConfig(induction_head_ratio=0.14, echo_head_ratio=0.01)

model_path = f"{os.environ['PROJECT_PATH']}/resource/llm_ptq/llama2_7b/"
output_model_path = f"{os.environ['PROJECT_PATH']}/output/ra_compression_llama/win.pt"

model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path=model_path,
    torch_dtype=torch.float16,
    trust_remote_code=True,
    device_map="auto",
    local_files_only=True,
).eval()

tokenizer = AutoTokenizer.from_pretrained(
    pretrained_model_name_or_path=model_path,
    pad_token='<|extra_0|>',
    eos_token='<|endoftext|>',
    padding_side='left',
    trust_remote_code=True,
    local_files_only=True,
)

ra = RARopeCompressor(model, tokenizer, config)
ra.get_compress_heads(output_model_path)

print('success!')
