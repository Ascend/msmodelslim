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

from abc import ABC, abstractmethod
from typing import Dict


class RaCompressAnalysisInterface(ABC):
    """RA Compress 分析需要在模型适配器中实现的接口。

    提供 Q、K、QKV 投影层的名称模式，用于定位目标层。
    """

    @abstractmethod
    def get_ra_compress_proj_patterns(self) -> Dict[str, str]:
        """返回 Q/K/QKV 投影层名称模式字典。

        返回值格式::

            {
                "q": "q_proj",    # Q 投影层名称模式
                "k": "k_proj",    # K 投影层名称模式
                "qkv": "qkv_proj",  # QKV 融合投影层名称模式
            }

        未使用的模式可以留空字符串（如无 QKV 融合时 qkv=""）。
        """
        raise NotImplementedError
