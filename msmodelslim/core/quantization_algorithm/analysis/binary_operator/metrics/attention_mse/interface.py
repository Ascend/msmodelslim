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
from typing import Callable, Union

import torch


class AttentionMSEAnalysisInterface(ABC):
    """Attention结构的敏感分析需要在模型适配器中实现的接口方法"""

    @abstractmethod
    def get_attention_module_cls(self) -> str:
        """返回用于匹配注意力层的模块类（即需要挂 hook 的 attention 子模块）的字符串表示。"""
        ...

    @abstractmethod
    def get_attention_output_extractor(self) -> Callable[[Union[tuple, torch.Tensor]], torch.Tensor]:
        """
        返回一个提取函数，用于从 attention 模块的 forward 输出中取出用于敏感层分析的张量部分。

        参数（由调用方传入，通常为 attention 的 forward 返回值）：
            attention_forward_output: 注意力模块 forward 的完整输出。
                类型依模型而定，多为 tuple 或单个 Tensor。
                例如 DeepSeek-V3 为 (attn_output, ...) 的 tuple，取第 0 项即为注意力输出。

        返回值：
            用于敏感度分析的注意力输出张量。
        """
        ...
