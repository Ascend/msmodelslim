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

from msmodelslim.utils.logging import logger_setter

from ..api import calculate_qparam, fake_quantize
from ..auto import AutoFakeQuantActivation
from ..const import hif4_per_block_sym
from ..qal import QABCRegistry, QDType, QParam, QStorage
from ..utils import pad_and_reshape, undo_pad_and_reshape
from .per_block import FakeQuantActivationPerBlock


@QABCRegistry.multi_register(
    dispatch_key=[
        hif4_per_block_sym,
    ],
    abc_type=AutoFakeQuantActivation,
)
@logger_setter()
class HIF4FakeQuantActivationPerBlock(FakeQuantActivationPerBlock):
    """HiF4 per-block 动态伪量化/反量化。

    输入形状: (batch_size, num_head, seq_len, head_dim)，沿最后一维按 HiF4 分块结构
    （block_size=64）动态计算三级缩放因子（scale_lv3/scale_lv2/scale_factor）。
    复用 w4a4_hif4_dynamic.py 中的 HiF4 三级缩放算法。
    """

    def __init__(self, x_q_param: QParam):
        super().__init__(x_q_param)
        self.hifx_finfo = x_q_param.scheme.dtype.hifx_finfo
        self.block_size = self.hifx_finfo.block_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        axes = self.x_axes
        ori_shape, ori_dtype = x.shape, x.dtype
        x, axes, padc, dim0 = pad_and_reshape(x.to(torch.float32), ori_shape, axes, self.hifx_finfo)
        x_level = torch.abs(x)
        max_levels = {}
        quant_level = 3
        for ax in range(axes, axes - 3, -1):
            self.x_minmax_block_observer.update(x_level, shared_exp_axes=[ax])
            _, max_lv = self.x_minmax_block_observer.get_min_max()
            max_levels[quant_level] = max_lv
            quant_level -= 1
            x_level = max_lv
        x_q_param = calculate_qparam(
            min_val=None,
            max_val=None,
            q_dtype=self.x_q_scheme.dtype,
            q_scope=self.x_q_scheme.scope,
            symmetric=self.x_q_scheme.symmetric,
            max_levels=max_levels,
            axes=axes,
        )
        x_q_dq = fake_quantize(QStorage(QDType.FLOAT, x), x_q_param)
        kwargs = {"padc": padc, "padded_shape": x.shape, "orig_shape": ori_shape, "dim0": dim0}
        x_q_dq.value = undo_pad_and_reshape(x_q_dq.value.to(ori_dtype), axes, **kwargs)
        return x_q_dq.value
