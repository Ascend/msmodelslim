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

import torch
from torch import nn

from msmodelslim.utils.logging import get_logger
from .model import RMSNorm


def remove_zero_and_shift(matrix):
    n, m = matrix.shape

    # Step 1: 找到每行第一个 0 的位置（即要删除的位置）
    zero_pos = (matrix == 0).int().argmax(dim=1)  # [n,]

    # Step 2: 构造掩码，标记要保留的元素（排除每行的第一个 0）
    col_indices = torch.arange(m, device=matrix.device).expand(n, -1)  # [n, m]
    mask = col_indices != zero_pos.unsqueeze(1)  # [n, m]

    # Step 3: 用掩码筛选元素（自动展平，需要重新调整形状）
    filtered = matrix[mask].view(n, m - 1)  # [n, m-1]

    # Step 4: 在最后一列补 0
    result = torch.cat([filtered, torch.zeros(n, 1, device=matrix.device)], dim=1)  # [n, m]

    return result.to(matrix)


class SharedHead(nn.Module):
    """MTP 输出头：norm + lm_head（head.weight 与主模型 lm_head 共享）。"""

    def __init__(self, config):
        super().__init__()
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, hidden_states):
        normalized_states = self.norm(hidden_states)
        logits = self.head(normalized_states)
        return logits


class MTPExtraModule(nn.Module):
    """MTP 额外组件，用于附加到 layer 45（nextn_predict 层）上。

    包含：
    - eh_proj: 将 [embed_norm, hidden_norm] 投影回 hidden_size
    - enorm: 输入 token embedding 的 RMSNorm
    - hnorm: 隐层状态的 RMSNorm
    - shared_head: 输出头（norm + head，head.weight 与 lm_head 共享）
    - embed_tokens: token embedding（权重与主模型共享）
    """

    def __init__(self, config):
        super().__init__()
        self.enorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hnorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.shared_head = SharedHead(config)
        self.eh_proj = nn.Linear(config.hidden_size * 2, config.hidden_size, bias=False)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)


def wrap_mtp_decoder(mtp_decoder: nn.Module, mtp_extra: nn.Module):
    """将 MTPExtraModule 的组件挂载到 decoder layer 上。

    Args:
        mtp_decoder: 目标 DecoderLayer（layer 45）
        mtp_extra: MTPExtraModule 实例
    """
    get_logger().debug('Start to wrap mtp for GLM-5-Next')
    mtp_decoder.enorm = mtp_extra.enorm
    mtp_decoder.hnorm = mtp_extra.hnorm
    mtp_decoder.shared_head = mtp_extra.shared_head
    mtp_decoder.eh_proj = mtp_extra.eh_proj
    mtp_decoder.embed_tokens = mtp_extra.embed_tokens
    get_logger().debug('Success to wrap mtp for GLM-5-Next')
