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

import unittest
from unittest.mock import MagicMock

import torch
from torch import nn

from msmodelslim.core.quantizer.base import QConfig
from msmodelslim.core.quantizer.linear import LinearQConfig
from msmodelslim.ir.qal import QDType, QScope
from msmodelslim.processor.trainable_linear_quant.pipeline.runtime import BlockTLQContext


def _int8_qconfig() -> LinearQConfig:
    return LinearQConfig(
        act=QConfig(dtype=QDType.FLOAT, scope=QScope.PER_TENSOR, symmetric=True, method="none"),
        weight=QConfig(dtype=QDType.INT8, scope=QScope.PER_CHANNEL, symmetric=True, method="minmax"),
    )


class TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.block = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 2))


class TestBlockTLQContextRelease(unittest.TestCase):
    def test_release_clears_training_state(self):
        op = MagicMock()
        op.target_modules = {"block0.fc": MagicMock()}
        ctx = BlockTLQContext(block_name="block0")
        ctx.ops = [op]
        ctx.wrappers_by_path = {"block0.fc": MagicMock()}
        ctx.teacher_outputs = [torch.randn(2, 4)]

        ctx.release()

        op.release_cached_params.assert_called_once()
        self.assertEqual(op.target_modules, {})
        self.assertEqual(ctx.ops, [])
        self.assertEqual(ctx.wrappers_by_path, {})
        self.assertEqual(ctx.teacher_outputs, [])
