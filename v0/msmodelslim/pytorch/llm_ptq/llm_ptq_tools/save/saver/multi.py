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

from typing import List

from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.save.saver.base import BaseSaver


class MultiSaver(BaseSaver):
    def __init__(self):
        super().__init__()

        self.saver_list: List[BaseSaver] = []

    def register(self, saver: BaseSaver):
        if not isinstance(saver, BaseSaver):
            raise TypeError(f'Saver must be a subclass of BaseSaver, not {type(saver).__name__}')
        self.saver_list.append(saver)

    def pre_process(self) -> None:
        for saver in self.saver_list:
            saver.pre_process()

    def save(self, name, meta, data) -> None:
        for saver in self.saver_list:
            saver.save(name, meta, data)

    def post_process(self) -> None:
        for saver in self.saver_list:
            saver.post_process()
