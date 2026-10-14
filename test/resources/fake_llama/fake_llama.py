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

from transformers import AutoTokenizer
from transformers.models.llama import LlamaConfig, LlamaForCausalLM


def get_fake_llama_model_and_tokenizer():
    """
    获取一个随机的、非常小的Llama模型以及其所对应的tokenizer，用于验证工具中某些数值算法的正确性
    """
    # 使用绝对路径确保无论从哪个目录运行测试都能正确找到配置文件
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir, "config.json")
    config = LlamaConfig.from_json_file(config_path)
    tokenizer = AutoTokenizer.from_pretrained(current_dir)
    return LlamaForCausalLM(config), tokenizer
