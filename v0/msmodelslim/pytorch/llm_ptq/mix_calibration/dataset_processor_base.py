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

from abc import ABC, abstractmethod


class DatasetProcessorBase(ABC):
    def __init__(self, dataset_path, tokenizer=None, model=None):
        self.dataset_path = dataset_path
        self.ori_prompts = []
        self.ori_answers = []
        self.tokenizer = tokenizer
        self.model = model

    @abstractmethod
    def process_data(self, indexs):
        """解析一组样本的数据格式"""
        prpt_ans = {}
        return prpt_ans

    @abstractmethod
    def verify_positive_prompt(self, prompts, labels):
        """校验一组样本是否为正样本"""
        prpt_ans = []
        return prpt_ans

    def get_dataset_size(self):
        return len(self.ori_prompts)