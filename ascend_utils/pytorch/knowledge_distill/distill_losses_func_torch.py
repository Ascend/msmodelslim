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
# pylint: disable=consider-using-from-import,super-with-arguments,useless-parent-delegation

import torch
import torch.nn as nn
import torch.nn.functional as func


def update_logits_by_temperature_pt(logits_s, logits_t, temperature):
    if isinstance(temperature, torch.Tensor) and temperature.dim() > 0:
        temperature = temperature.unsqueeze(-1)

    return torch.div(logits_s, temperature), torch.div(logits_t, temperature)


class KDMse(nn.Module):
    def __init__(self):
        super(KDMse, self).__init__()

    def forward(self, logits_s, logits_t, temperature=1):
        beta_logits_s, beta_logits_t = update_logits_by_temperature_pt(logits_s, logits_t, temperature)
        loss = func.mse_loss(beta_logits_s, beta_logits_t)
        return loss


class KDCrossEntropy(nn.Module):
    def __int__(self):
        super(KDCrossEntropy, self).__init__()

    def forward(self, logits_s, logits_t, temperature=1):
        logits_s = logits_s.float()
        logits_t = logits_t.float()
        beta_logits_s, beta_logits_t = update_logits_by_temperature_pt(logits_s, logits_t, temperature)

        p_t = func.softmax(beta_logits_t, dim=-1)
        loss = -(p_t * func.log_softmax(beta_logits_s, dim=-1)).sum(dim=-1).mean()
        return loss


class HardKDCrossEntropy(nn.Module):
    def __int__(self):
        super(HardKDCrossEntropy, self).__init__()

    def forward(self, logits_s, logits_t):
        loss = func.cross_entropy(logits_s, logits_t.argmax(dim=1))
        return loss


class HiddenMse(nn.Module):
    def __int__(self):
        super(HiddenMse, self).__init__()

    def forward(self, state_s, state_t, mask=None):
        state_s = state_s.float()
        state_t = state_t.float()
        if mask is None:
            return func.mse_loss(state_s, state_t)

        mask = mask.to(state_s)
        valid_count = mask.sum() * state_s.size(-1)
        loss = (func.mse_loss(state_s, state_t, reduction='none') * mask.unsqueeze(-1)).sum()
        loss = torch.div(loss, valid_count)
        return loss


class MMD(nn.Module):
    def __init__(self, batch_size):
        super(MMD, self).__init__()
        self.batch_size = batch_size

    def forward(self, state_s, state_t, mask=None):
        """
        state_s: student feature tensor
        state_t: teahcer feature tensor
        """
        if state_s.dim() == 2:  # for npu bert
            hidden_dim_s = state_s.shape[-1]
            state_s = state_s.view(self.batch_size, -1, hidden_dim_s)  # (batch_size , length, hidden_dim_s)
        if state_t.dim() == 2:  # for npu bert
            hidden_dim_t = state_t.shape[-1]
            state_t = state_t.view(self.batch_size, -1, hidden_dim_t)  # (batch_size , length, hidden_dim_t)
        state_s = [state_s, state_s]
        state_t = [state_t, state_t]
        state_s_0 = state_s[0]
        state_s_1 = state_s[1]
        state_t_0 = state_t[0]
        state_t_1 = state_t[1]
        if mask is None:
            gram_s = torch.div(torch.bmm(state_s_0, state_s_1.transpose(1, 2)), state_s_1.size(2))
            gram_t = torch.div(torch.bmm(state_t_0, state_t_1.transpose(1, 2)), state_t_1.size(2))

            gram_s = gram_s.float()
            gram_t = gram_t.float()
            loss = func.mse_loss(gram_s, gram_t)
        else:
            mask = mask.to(state_s[0])
            valid_count = torch.pow(mask.sum(dim=1), 2).sum()
            gram_s = torch.div(torch.bmm(state_s_0, state_s_1.transpose(1, 2)), state_s_1.size(1))
            gram_t = torch.div(torch.bmm(state_t_0, state_t_1.transpose(1, 2)), state_t_1.size(1))
            loss = func.mse_loss(gram_s, gram_t, reduction='none') * mask.unsqueeze(-1) * mask.unsqueeze(1)
            loss = loss.sum()
            loss = torch.div(loss, valid_count)

        return loss


DISTILL_LOSS_FUNC_TORCH = {"KDCrossEntropy": KDCrossEntropy}
