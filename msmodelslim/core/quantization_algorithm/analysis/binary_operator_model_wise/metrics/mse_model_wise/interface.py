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
from typing import Any

import torch


class MSEModelWiseAnalysisInterface(ABC):
    """mse_model_wise 敏感层分析的可选模型适配器接口。

    与 ``AttentionMSEAnalysisInterface``（attn_mse 必选）不同，本接口为可选：
    未实现时 Processor 使用 ``DefaultMSEModelWiseBlockData`` 处理常见 LLM block I/O。
    VLM 等结构若默认逻辑不足，可在 adapter 中继承并实现 ``extract_hidden_states``。
    """

    @abstractmethod
    def extract_hidden_states(self, value: Any) -> torch.Tensor:
        """从 block 相关数据中提取用于 MSE 比较 / 层间链式传播的主 tensor。

        参数 ``value`` 可能是：
        - block ``forward`` 返回值（Tensor、tuple、ModelOutput、dict 等）
        - forward 输入行 ``(args, kwargs)``（链式一致性校验时）

        层间链式传播时，Processor 将返回值作为下一层 ``forward`` 的首个 positional arg：
        ``(extract_hidden_states(block_output),)``。
        """
