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

from transformers.models.qwen2_vl.configuration_qwen2_vl import Qwen2VLConfig
from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLVisionBlock, Qwen2VLDecoderLayer


class FakeQwenCreator:
    """
    用于生成一个随机的、非常小的Qwen2VLVisionBlock，用于验证工具中某些流程的正确性
    """

    config = Qwen2VLConfig.from_pretrained(os.path.join(os.path.dirname(__file__), "config.json"))
    vision_config = config.vision_config

    @classmethod
    def get_block(cls):
        """
        获取一个随机的、非常小的Qwen2VLVisionBlock，用于验证工具中某些流程的正确性
        """
        block = Qwen2VLVisionBlock(config=cls.vision_config, attn_implementation="eager")
        return block

    @classmethod
    def get_decoder_layer(cls):
        """
        获取一个随机的、非常小的Qwen2VLDecoderLayer，用于验证工具中某些流程的正确性
        """
        layer = Qwen2VLDecoderLayer(config=cls.config, layer_idx=0)
        return layer
