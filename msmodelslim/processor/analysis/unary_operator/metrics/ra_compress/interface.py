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
from typing import Any, Dict


class RaCompressAnalysisInterface(ABC):
    """RA Compress 分析的模型适配接口：告诉分析方法怎么找 Q / K 投影层，以及用哪个 tokenizer。

    适配只需两步：

    1. 让模型适配器同时继承本接口，例如 ``class MyModelAdapter(DefaultModelAdapter,
       RaCompressAnalysisInterface)``；
    2. 实现下面两个方法，``get_proj_names`` 给出 Q / K / QKV 投影层的名称片段，
       ``get_tokenizer`` 返回模型实际使用的 tokenizer。

    例如::

        class MyModelAdapter(DefaultModelAdapter, RaCompressAnalysisInterface):
            def get_proj_names(self):
                return {"q": "q_proj", "k": "k_proj", "qkv": "qkv_proj"}

            def get_tokenizer(self):
                return AutoTokenizer.from_pretrained(model_path)

    这两个方法都必须实现，缺一个适配器就无法实例化。完整参考见
    ``msmodelslim/model/qwen2/model_adapter.py``。
    """

    @abstractmethod
    def get_proj_names(self) -> Dict[str, str]:
        """告诉分析方法 Q / K / QKV 投影层叫什么名字。

        分析方法是拿这里返回的字符串去匹配模型里的 ``nn.Linear`` 层名（子串匹配），所以只要给
        名字里有代表性的一段，不用给完整层名，也不是正则。

        Returns:
            例如 ``{"q": "q_proj", "k": "k_proj", "qkv": "qkv_proj"}``。没有 QKV 融合层时，
            ``qkv`` 可以省略或留空字符串，两者等价。
        """
        ...

    @abstractmethod
    def get_tokenizer(self) -> Any:
        """告诉分析方法这个模型用的是哪个 tokenizer。

        分析方法要用它取校准输入的首 token，直接返回加载模型时用的那个 tokenizer 即可，通常是
        ``AutoTokenizer.from_pretrained(model_path)``。不要固定返回 BOS 或 0，否则首 token 会与
        模型真实推理不一致。返回 ``None`` 或抛异常时，分析方法会告警并改用兜底取值，流程不中断。
        """
        ...
