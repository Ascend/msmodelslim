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
from unittest.mock import patch

import torch
from torch import nn

from msmodelslim.model.glm_5_next.mtp_quant_module import (
    MTPExtraModule,
    SharedHead,
    remove_zero_and_shift,
    wrap_mtp_decoder,
)
from msmodelslim.model.glm_5_next.model import RMSNorm


class DummyConfig:
    """模拟 text_config，仅供 MTP 模块构建使用。"""

    def __init__(self):
        self.hidden_size = 32
        self.vocab_size = 64
        self.rms_norm_eps = 1e-6


class TestRemoveZeroAndShift(unittest.TestCase):
    def test_shouldRemoveFirstZeroPerRow_when_zerosExist(self):
        matrix = torch.tensor([[1, 2, 0, 4], [5, 0, 7, 8]])
        result = remove_zero_and_shift(matrix)

        # 每行删去第一个 0，末尾补 0
        self.assertEqual(result.tolist(), [[1, 2, 4, 0], [5, 7, 8, 0]])

    def test_shouldRemoveFirstElement_when_noZeroInRow(self):
        matrix = torch.tensor([[1, 2, 3]])
        result = remove_zero_and_shift(matrix)

        # 无 0 时 argmax 对全 0 掩码返回首个索引 0，等效删除首元素后末尾补 0
        self.assertEqual(result.tolist(), [[2, 3, 0]])

    def test_shouldKeepShapeAndDtype_when_called(self):
        matrix = torch.ones(3, 5, dtype=torch.float32)
        result = remove_zero_and_shift(matrix)

        self.assertEqual(result.shape, matrix.shape)
        self.assertEqual(result.dtype, matrix.dtype)

    def test_shouldKeepDevice_when_called(self):
        matrix = torch.ones(2, 4)
        result = remove_zero_and_shift(matrix)

        self.assertEqual(result.device, matrix.device)


class TestSharedHead(unittest.TestCase):
    def test_shouldContainNormAndHead_when_initialized(self):
        head = SharedHead(DummyConfig())

        self.assertIsInstance(head.norm, RMSNorm)
        self.assertIsInstance(head.head, nn.Linear)
        self.assertEqual(head.head.in_features, 32)
        self.assertEqual(head.head.out_features, 64)
        self.assertIsNone(head.head.bias)

    def test_forward_shouldReturnLogits_when_called(self):
        head = SharedHead(DummyConfig())
        head.eval()

        hidden_states = torch.randn(2, 32)
        logits = head(hidden_states)

        self.assertEqual(logits.shape, (2, 64))


class TestMTPExtraModule(unittest.TestCase):
    def test_shouldContainAllComponents_when_initialized(self):
        module = MTPExtraModule(DummyConfig())

        self.assertIsInstance(module.enorm, RMSNorm)
        self.assertIsInstance(module.hnorm, RMSNorm)
        self.assertIsInstance(module.shared_head, SharedHead)
        self.assertIsInstance(module.eh_proj, nn.Linear)
        self.assertIsInstance(module.embed_tokens, nn.Embedding)
        # eh_proj: [hidden*2, hidden]
        self.assertEqual(module.eh_proj.in_features, 64)
        self.assertEqual(module.eh_proj.out_features, 32)
        self.assertEqual(module.embed_tokens.num_embeddings, 64)


class TestWrapMtpDecoder(unittest.TestCase):
    def test_shouldAttachComponents_when_wrapping(self):
        config = DummyConfig()
        mtp_extra = MTPExtraModule(config)
        mtp_decoder = nn.Module()

        wrap_mtp_decoder(mtp_decoder=mtp_decoder, mtp_extra=mtp_extra)

        self.assertIs(mtp_decoder.enorm, mtp_extra.enorm)
        self.assertIs(mtp_decoder.hnorm, mtp_extra.hnorm)
        self.assertIs(mtp_decoder.shared_head, mtp_extra.shared_head)
        self.assertIs(mtp_decoder.eh_proj, mtp_extra.eh_proj)
        self.assertIs(mtp_decoder.embed_tokens, mtp_extra.embed_tokens)

    def test_shouldNotAffectOtherAttributes_when_wrapping(self):
        config = DummyConfig()
        mtp_extra = MTPExtraModule(config)
        mtp_decoder = nn.Module()
        original_attr = nn.Linear(8, 8)
        mtp_decoder.self_attn = original_attr

        wrap_mtp_decoder(mtp_decoder=mtp_decoder, mtp_extra=mtp_extra)

        self.assertIs(mtp_decoder.self_attn, original_attr)

    def test_shouldLogDebug_when_wrapping(self):
        config = DummyConfig()
        mtp_extra = MTPExtraModule(config)
        mtp_decoder = nn.Module()

        with patch("msmodelslim.model.glm_5_next.mtp_quant_module.get_logger") as mock_logger:
            wrap_mtp_decoder(mtp_decoder=mtp_decoder, mtp_extra=mtp_extra)

        self.assertEqual(mock_logger.return_value.debug.call_count, 2)


if __name__ == '__main__':
    unittest.main()
