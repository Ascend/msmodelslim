#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
-------------------------------------------------------------------------
This file is part of the MindStudio project.
Copyright (c) 2025 Huawei Technologies Co.,Ltd.

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


import torch


class CuttingMethodRegistry:
    def __init__(self):
        # 初始化 cutting_methods 字典
        self.cutting_methods = {}
        # 在类内部注册切割方法
        self.register_cutting_methods()

    @staticmethod
    def default_cut(comming_max, comming_min, in_hidden_size):
        """
        默认切割方法，可修改。
        """
        res_dim = comming_max.shape[-1] - in_hidden_size
        _, kv_max = torch.split(comming_max, [in_hidden_size, res_dim])
        _, kv_min = torch.split(comming_min, [in_hidden_size, res_dim])
        key_max, value_max = torch.chunk(kv_max, 2, dim=0)
        key_min, value_min = torch.chunk(kv_min, 2, dim=0)
        return key_max, value_max, key_min, value_min

    @staticmethod
    def internlm2_cut(comming_max, comming_min, m):
        """
        internlm2 模型的切割方法。
        """
        if m.config.num_key_value_heads == 0:
            raise ValueError('Num_key_value_heads must not be zero in model config.')
        if m.config.num_attention_heads == 0:
            raise ValueError('Num_attention_heads must not be zero in model config.')
        if m.config.hidden_size == 0:
            raise ValueError('Hidden_size must not be zero in model config.')

        gs = m.config.num_attention_heads // m.config.num_key_value_heads + 2
        d = m.config.hidden_size // m.config.num_attention_heads
        h = comming_max.shape[0] // (gs * d)
        qkv_states_max = comming_max.view(h, gs, d)
        qkv_states_min = comming_min.view(h, gs, d)
        key_max = qkv_states_max[:, -2, :].reshape(-1)
        value_max = qkv_states_max[:, -1, :].reshape(-1)
        key_min = qkv_states_min[:, -2, :].reshape(-1)
        value_min = qkv_states_min[:, -1, :].reshape(-1)
        return key_max, value_max, key_min, value_min

    def register_cutting_methods(self):
        """
        注册切割方法，将方法添加到 cutting_methods 字典中。
        """
        self.cutting_methods["internlm2"] = self.internlm2_cut

    def get_cutting_method(self, model_type):
        """
        根据模型类型获取对应的切割方法。如果模型类型不存在于字典中，则返回 None。

        参数:
        model_type (str): 模型类型的名称。

        返回:
        function: 对应的切割方法或 None。
        """
        return self.cutting_methods.get(model_type, None)


cutting_method_registry = CuttingMethodRegistry()
