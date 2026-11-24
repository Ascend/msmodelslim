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

from ..api import fake_quantize
from ..auto import AutoFakeQuantActivation
from ..qal import QDType, QParam, QStorage


class FakeQuantActivationPerTensor(AutoFakeQuantActivation):
    """Static activation fake quantization with one shared scale."""

    def __init__(self, x_q_param: QParam):
        super().__init__()
        self.x_q_scheme = x_q_param.scheme

        scale = x_q_param.ext.get("scale")
        if scale is None:
            class_name = self.__class__.__name__
            raise ValueError(f"`scale` is needed in ext but is missing for {class_name}")

        self.input_scale = nn.Parameter(scale, requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ext = {"scale": self.input_scale.data}
        x_q_param = QParam(scheme=self.x_q_scheme, ext=ext)
        x_q_dq = fake_quantize(QStorage(QDType.FLOAT, x), x_q_param)
        return x_q_dq.value.to(dtype=x.dtype)
