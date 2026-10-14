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
# pylint: disable=self-assigning-variable,super-with-arguments,useless-parent-delegation

import mindspore
from mindspore import nn, ops
from mindspore.ops import functional as func
from mindspore.nn import Cell

REDUCE_NONE = 0
REDUCE_MEAN = 1
REDUCE_SUM = 2


def mse_loss_ms(state_s, state_t, reduction=REDUCE_MEAN):
    output = func.square(state_s - state_t)
    if reduction == REDUCE_MEAN:
        output = ops.reduce_sum(output.sum())
        output = ops.div(output, ops.size(state_s))
    elif reduction == REDUCE_SUM:
        output = ops.reduce_sum(output.sum())
    else:
        output = output
    return output


def update_logits_by_temperature_ms(logits_s, logits_t, temperature):
    if isinstance(temperature, mindspore.Tensor) and temperature.ndim > 0:
        ops_expand_dims = ops.ExpandDims()
        temperature = ops_expand_dims(temperature, -1)

    return ops.div(logits_s, temperature), ops.div(logits_t, temperature)


class KDMse(Cell):
    def __init__(self):
        super(KDMse, self).__init__()

    def construct(self, logits_s, logits_t, temperature=1):
        beta_logits_s, beta_logits_t = update_logits_by_temperature_ms(logits_s, logits_t, temperature)
        loss = mse_loss_ms(beta_logits_s, beta_logits_t)
        return loss


class KLDivLoss(Cell):
    def __init__(self):
        super(KLDivLoss, self).__init__()
        self.cast = ops.Cast()
        self.soft_max = ops.Softmax()
        self.log = ops.Log()

    def construct(self, logits_s, logits_t, temperature=1):
        logits_s = self.cast(logits_s, mindspore.float32)
        logits_t = self.cast(logits_t, mindspore.float32)
        beta_logits_s, beta_logits_t = update_logits_by_temperature_ms(logits_s, logits_t, temperature)
        p_t = self.soft_max(beta_logits_t)
        p_s = self.soft_max(beta_logits_s)
        loss = (p_t * (self.log(p_t) - self.log(p_s))).sum(axis=-1).mean()
        return loss


class Mse(Cell):
    """
    Do not use now.
    """

    def __init__(self):
        super(Mse, self).__init__()
        self.mse = nn.MSE()

    def construct(self, t_pred, s_pred):
        self.mse.clear()
        self.mse.update(s_pred, t_pred)
        return self.mse.eval()


class KDCrossEntropy(Cell):
    def __int__(self):
        super(KDCrossEntropy, self).__init__()

    def construct(self, logits_s, logits_t, temperature=1):
        cast = ops.Cast()
        logits_s = cast(logits_s, mindspore.float32)
        logits_t = cast(logits_t, mindspore.float32)
        beta_logits_s, beta_logits_t = update_logits_by_temperature_ms(logits_s, logits_t, temperature)

        soft_max = ops.Softmax()
        log_softmax = nn.LogSoftmax()
        p_t = soft_max(beta_logits_t)
        loss = -(p_t * log_softmax(beta_logits_s)).sum(axis=-1).mean()
        return loss


class HardKDCrossEntropy(Cell):
    """
    Do not use now.
    """

    def __int__(self):
        super(HardKDCrossEntropy, self).__init__()

    def construct(self, logits_s, logits_t):
        ops_binary_cross_entropy = ops.BinaryCrossEntropy()
        loss = ops_binary_cross_entropy(logits_s, logits_t.argmax(axis=1))
        return loss


class HiddenMse(Cell):
    def __int__(self):
        super(HiddenMse, self).__init__()

    def construct(self, state_s, state_t, mask=None):
        cast = ops.Cast()
        state_s = cast(state_s, mindspore.float32)
        state_t = cast(state_t, mindspore.float32)
        if mask is None:
            return mse_loss_ms(state_s, state_t)

        mask = mask.to(state_s)
        valid_count = mask.sum() * state_s.shape()
        expand_dims = ops.ExpandDims()
        mask = expand_dims(mask, -1)
        loss = (mse_loss_ms(state_s, state_t) * mask).sum()
        loss = ops.div(loss, valid_count)
        return loss


class MMD(Cell):
    """
    Do not use now.
    """

    def __init__(self, batch_size):
        super(MMD, self).__init__()
        self.batch_size = batch_size

    def construct(self, state_s, state_t, mask=None):
        """
        state_s: student feature tensor
        state_t: teahcer feature tensor
        """
        if state_s.ndim == 2:  # for npu bert
            hidden_dim_s = state_s.shape[-1]
            state_s = state_s.view(self.batch_size, -1, hidden_dim_s)  # (batch_size , length, hidden_dim_s)
        if state_t.ndim == 2:  # for npu bert
            hidden_dim_t = state_t.shape[-1]
            state_t = state_t.view(self.batch_size, -1, hidden_dim_t)  # (batch_size , length, hidden_dim_t)
        state_s = [state_s, state_s]
        state_t = [state_t, state_t]
        state_s_0 = state_s[0]
        state_s_1 = state_s[1]
        state_t_0 = state_t[0]
        state_t_1 = state_t[1]
        bmm = ops.BatchMatMul()
        transpose = ops.Transpose()
        if mask is None:
            cast = ops.Cast()
            gram_s = ops.div(bmm(state_s_0, transpose(state_s_1, (0, 2, 1))), state_s_1.shape[1])
            gram_t = ops.div(bmm(state_t_0, transpose(state_t_1, (0, 2, 1))), state_t_1.shape[1])
            gram_s = cast(gram_s, mindspore.dtype.float32)
            gram_t = cast(gram_t, mindspore.dtype.float32)
            loss = mse_loss_ms(gram_s, gram_t)
        else:
            ms_pow = ops.Pow()
            expand_dims = ops.ExpandDims()
            mask = mask.to(state_s[0])
            valid_count = ms_pow(mask.sum(dim=1), 2).sum()
            gram_s = ops.div(bmm(state_s_0, transpose(state_s_1, (0, 2, 1))), state_s_1.shape[1])
            gram_t = ops.div(bmm(state_t_0, transpose(state_t_1, (0, 2, 1))), state_t_1.shape[1])
            loss = mse_loss_ms(gram_s, gram_t, reduction=REDUCE_NONE) * expand_dims(mask, -1) * expand_dims(mask, 1)
            loss = ops.div(loss.sum(), valid_count)

        return loss


DISTILL_LOSS_FUNC_MS = {"KDCrossEntropy": KDCrossEntropy}
