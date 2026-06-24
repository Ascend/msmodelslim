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


def remove_zero_and_shift(matrix):
    n, m = matrix.shape

    # Step 1: 找到每行第一个 0 的位置（即要删除的位置）
    # 如果某行没有 0，则默认保留所有元素（但根据题意，应该每行都有一个 0）
    zero_pos = (matrix == 0).int().argmax(dim=1)  # [n,]

    # Step 2: 构造掩码，标记要保留的元素（排除每行的第一个 0）
    # 生成一个 [n, m] 的坐标矩阵，标记每列是否等于 zero_pos
    col_indices = torch.arange(m, device=matrix.device).expand(n, -1)  # [n, m]
    mask = col_indices != zero_pos.unsqueeze(1)  # [n, m]

    # Step 3: 用掩码筛选元素（自动展平，需要重新调整形状）
    filtered = matrix[mask].view(n, m - 1)  # [n, m-1]

    # Step 4: 在最后一列补 0
    result = torch.cat([filtered, torch.zeros(n, 1, device=matrix.device)], dim=1)  # [n, m]

    return result.to(matrix)


class GLM4MoeRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class SharedHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm = GLM4MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, hidden_states):
        normalized_states = self.norm(hidden_states)
        logits = self.head(normalized_states)
        return logits


class MTPLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.enorm = GLM4MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hnorm = GLM4MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.shared_head = SharedHead(config)

        self.eh_proj = nn.Linear(config.hidden_size * 2, config.hidden_size, bias=False)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)


def wrap_mtp_decoder(mtp_decoder: nn.Module, mtp_layer: nn.Module):
    get_logger().debug('Start to wrap mtp')
    mtp_decoder.enorm = mtp_layer.enorm
    mtp_decoder.hnorm = mtp_layer.hnorm
    mtp_decoder.shared_head = mtp_layer.shared_head
    mtp_decoder.eh_proj = mtp_layer.eh_proj
    mtp_decoder.embed_tokens = mtp_layer.embed_tokens
    get_logger().debug('Success to wrap mtp')
