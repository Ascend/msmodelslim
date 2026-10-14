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
# pylint: disable=attribute-defined-outside-init,comparison-with-callable,consider-using-from-import,no-member,no-name-in-module,no-value-for-parameter

from dataclasses import dataclass
import logging
import sys
from types import ModuleType
from unittest.mock import patch
from contextlib import contextmanager

import pytest
import torch
import torch.nn as nn
import torch.distributed as dist
from transformers import PretrainedConfig

from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.fa_quant_adapter.hyvideo import (
    get_hyvideo_config,
    _install_forward_adapter,
    install_for_hyvideo_model,
    hyvideo_mm_double_stream_block_double_forward_adapter,
    hyvideo_mm_single_stream_block_single_forward_adapter,
    ForwardFactory,
)


# ------------------------------ 公共工具函数 ------------------------------
@contextmanager
def mock_hyvideo_module():
    original_module = sys.modules.get("hyvideo_double_module")
    try:
        sys.modules["hyvideo_double_module"] = MockHYVideoModule()
        yield
    finally:
        if original_module is not None:
            sys.modules["hyvideo_double_module"] = original_module
        else:
            sys.modules.pop("hyvideo_double_module", None)


# ------------------------------ Mock 类定义 ------------------------------
class MockHYVideoModule(ModuleType):
    def __init__(self):
        super().__init__("hyvideo_double_module")
        self.modulate = self._mock_modulate
        self.rearrange = self._mock_rearrange
        self.apply_rotary_emb = self._mock_apply_rotary_emb
        self.attention = self._mock_attention
        self.parallel_attention = self._mock_parallel_attention

    def _mock_modulate(self, x, shift, scale):
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

    def _mock_rearrange(self, x, pattern, *args, **kwargs):
        # 从 kwargs 中提取 K 或 k（优先 K，兼容实际调用）
        k = kwargs.get('K', kwargs.get('k', 0))  # 实际调用传 K=3，这里取 K 的值
        h = kwargs.get('H', kwargs.get('h', 0))  # 同理兼容 H/h

        # 原有逻辑不变
        b_rearrange, l_rearrange, total_dim = x.shape
        d_head = total_dim // (k * h)
        return x.view(b_rearrange, l_rearrange, k, h, d_head).permute(2, 0, 1, 3, 4)

    def _mock_apply_rotary_emb(self, q, k, freqs_cis, head_first):
        if hasattr(self, "force_rotary_shape_mismatch") and self.force_rotary_shape_mismatch:
            return q[:, :-1, :, :], k
        return q, k

    def _mock_attention(self, q, k, v, **kwargs):
        return torch.randn(q.shape)

    def _mock_parallel_attention(self, *args, **kwargs):
        return torch.randn(1, 1, 1, 1)


def mock_original_forward(self, *args, **kwargs):
    pass


mock_original_forward.__module__ = "hyvideo_double_module"


class MockMMDoubleStreamBlock(nn.Module):
    def __init__(self, heads_num=8, hidden_dim=256):
        super().__init__()
        self.heads_num = heads_num
        self.hidden_dim = hidden_dim
        self.head_dim = hidden_dim // heads_num
        self.scale = 1.0 / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        self.hybrid_seq_parallel_attn = False

        self.img_norm1 = nn.Identity()
        self.img_attn_qkv = nn.Linear(hidden_dim, hidden_dim * 3)
        self.img_attn_q_norm = nn.Identity()
        self.img_attn_k_norm = nn.Identity()

        self.txt_norm1 = nn.Identity()
        self.txt_attn_qkv = nn.Linear(hidden_dim, hidden_dim * 3)
        self.txt_attn_q_norm = nn.Identity()
        self.txt_attn_k_norm = nn.Identity()

    def double_forward(self, *args, **kwargs):
        pass


class MockMMSingleStreamBlock(nn.Module):
    def __init__(self, heads_num=8, hidden_dim=256):
        super().__init__()
        self.heads_num = heads_num
        self.hidden_dim = hidden_dim
        self.head_dim = hidden_dim // heads_num
        self.scale = 1.0 / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        self.hybrid_seq_parallel_attn = False

        self.q_norm = nn.Identity()
        self.k_norm = nn.Identity()

    def single_forward(self, *args, **kwargs):
        pass


class MMDoubleStreamBlock(MockMMDoubleStreamBlock):
    pass


class MMSingleStreamBlock(MockMMSingleStreamBlock):
    pass


class MockFAQuantizer:
    def __init__(self, config=None, logger=None):
        self.quant_called = 0
        self.quant_args = []

    def quant(self, x, qkv):
        self.quant_called += 1
        self.quant_args.append((x.shape, qkv))
        return x


@dataclass
class HyVideoConfigTestCase:
    is_distributed: bool
    world_size: int
    heads_num: int
    hidden_size: int
    expected_heads: int
    expected_hidden: int


# ------------------------------ 测试用例 ------------------------------
class TestHYVideoFAQuantAdapter:
    @staticmethod
    def test_install_forward_adapter_success(mocker):
        mock_module = MMDoubleStreamBlock()
        mock_logger = mocker.MagicMock(spec=logging.Logger)
        original_method = mock_module.double_forward

        mock_adapted_method = mocker.MagicMock()
        mock_descriptor = mocker.MagicMock()
        mock_descriptor.__get__ = mocker.MagicMock(return_value=mock_adapted_method)
        mock_forward_adapter = mocker.MagicMock(return_value=mock_descriptor)

        mocker.patch.object(ForwardFactory, "get_forward_adapter", return_value=mock_forward_adapter)

        _install_forward_adapter(
            module=mock_module,
            module_type="MMDoubleStreamBlock",
            adapter_type="double_forward",
            module_name="test_block",
            logger=mock_logger,
        )

        ForwardFactory.get_forward_adapter.assert_called_once_with("MMDoubleStreamBlock", "double_forward")
        mock_forward_adapter.assert_called_once_with(original_method)
        mock_descriptor.__get__.assert_called_once_with(mock_module, mock_module.__class__)
        assert getattr(mock_module, "double_forward") == mock_adapted_method
        mock_logger.info.assert_called_once_with("Successfully installed FAQuantizer for module test_block")

    @staticmethod
    def test_install_forward_adapter_failure(mocker):
        mock_module = MockMMDoubleStreamBlock()
        mock_logger = mocker.MagicMock(spec=logging.Logger)
        mock_error = RuntimeError("Adapter not found")

        mocker.patch.object(ForwardFactory, "get_forward_adapter", side_effect=mock_error)

        with pytest.raises(RuntimeError) as excinfo:
            _install_forward_adapter(
                module=mock_module,
                module_type="MMDoubleStreamBlock",
                adapter_type="double_forward",
                module_name="test_block",
                logger=mock_logger,
            )

        assert str(excinfo.value) == "Adapter not found"
        mock_logger.error.assert_called_once_with(
            "Failed to install FAQuantizer for module test_block: Adapter not found"
        )

    @pytest.mark.parametrize(
        "test_case",
        [
            HyVideoConfigTestCase(
                is_distributed=False, world_size=1, heads_num=8, hidden_size=256, expected_heads=8, expected_hidden=256
            ),
            HyVideoConfigTestCase(
                is_distributed=True, world_size=2, heads_num=8, hidden_size=256, expected_heads=4, expected_hidden=128
            ),
            HyVideoConfigTestCase(
                is_distributed=True, world_size=4, heads_num=8, hidden_size=256, expected_heads=2, expected_hidden=64
            ),
        ],
    )
    def test_get_hyvideo_config(self, mocker, test_case):
        mock_logger = mocker.MagicMock(spec=logging.Logger)
        mock_config = PretrainedConfig()
        mock_config.heads_num = test_case.heads_num
        mock_config.hidden_size = test_case.hidden_size

        with patch.object(dist, "is_initialized", return_value=test_case.is_distributed):
            with patch.object(dist, "get_world_size", return_value=test_case.world_size):
                result = get_hyvideo_config(mock_config, mock_logger)

                # 从 test_case 中获取预期值
                assert result.num_attention_heads == test_case.expected_heads
                assert result.hidden_size == test_case.expected_hidden
                assert result.num_key_value_heads == test_case.expected_heads

                if test_case.is_distributed:
                    mock_logger.info.assert_any_call(f"sp_size: {test_case.world_size}")
                else:
                    mock_logger.info.assert_any_call("sp_size = 1 (not in distributed environment)")

    def test_install_for_hyvideo_model_normal(self, mocker):
        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.double_block1 = MMDoubleStreamBlock()
                self.single_block1 = MMSingleStreamBlock()
                self.other_block = nn.Linear(10, 10)

        mock_model = TestModel()
        mock_config = PretrainedConfig()
        mock_config.heads_num = 8
        mock_config.hidden_size = 256
        mock_logger = mocker.MagicMock(spec=logging.Logger)
        skip_layers = ["skip_block"]

        mock_install_adapter = mocker.patch(
            "msmodelslim.pytorch.llm_ptq.llm_ptq_tools.fa_quant_adapter.hyvideo._install_forward_adapter"
        )
        mocker.patch(
            "msmodelslim.pytorch.llm_ptq.llm_ptq_tools.fa_quant_adapter.hyvideo.FAQuantizer",
            return_value=MockFAQuantizer(),
        )

        install_for_hyvideo_model(mock_model, mock_config, mock_logger, skip_layers)

        assert hasattr(mock_model.double_block1, "fa_quantizer")
        assert hasattr(mock_model.single_block1, "fa_quantizer")
        assert not hasattr(mock_model.other_block, "fa_quantizer")
        assert mock_install_adapter.call_count == 2
        mock_logger.info.assert_any_call("Installed quantizer at double_block1")
        mock_logger.info.assert_any_call("Installed quantizer at single_block1")

    def test_install_for_hyvideo_model_skip_cases(self, mocker):
        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.double_block_skip = MMDoubleStreamBlock()
                self.double_block_exist = MMDoubleStreamBlock()

        mock_model = TestModel()
        mock_model.double_block_exist.fa_quantizer = MockFAQuantizer()

        mock_config = PretrainedConfig()
        mock_config.heads_num = 8
        mock_config.hidden_size = 256
        mock_logger = mocker.MagicMock(spec=logging.Logger)
        skip_layers = ["double_block_skip"]

        mock_install_adapter = mocker.patch(
            "msmodelslim.pytorch.llm_ptq.llm_ptq_tools.fa_quant_adapter.hyvideo._install_forward_adapter"
        )

        install_for_hyvideo_model(mock_model, mock_config, mock_logger, skip_layers)

        assert not hasattr(mock_model.double_block_skip, "fa_quantizer")
        mock_logger.warning.assert_called_once_with("Module double_block_exist already has FAQuantizer installed.")
        assert mock_install_adapter.call_count == 0
        mock_logger.info.assert_called_once_with("Skipping FAQuantizer installation for module double_block_skip")

    @pytest.mark.parametrize("hybrid_attn, expect_parallel", [(False, False), (True, True)])
    def test_double_forward_adapter_normal(self, mocker, hybrid_attn, expect_parallel):
        with mock_hyvideo_module():
            mock_hy_mod = MockHYVideoModule()
            mocker.patch("importlib.import_module", return_value=mock_hy_mod)

            b_forward, l_forward, hidden_dim = 2, 10, 256
            img = torch.randn(b_forward, l_forward, hidden_dim)
            txt = torch.randn(b_forward, l_forward, hidden_dim)
            shift_scale = torch.randn(b_forward, hidden_dim)
            freqs_cis = torch.randn(l_forward, hidden_dim // 2, 2)
            cu_seqlens_q = torch.tensor([0, l_forward, 2 * l_forward, 3 * l_forward, 4 * l_forward], dtype=torch.int32)
            cu_seqlens_kv = cu_seqlens_q
            max_seqlen_q = 2 * l_forward
            max_seqlen_kv = 2 * l_forward

            mock_block = MMDoubleStreamBlock(heads_num=8, hidden_dim=hidden_dim)
            mock_block.hybrid_seq_parallel_attn = hybrid_attn
            mock_block.fa_quantizer = MockFAQuantizer()

            adapter = hyvideo_mm_double_stream_block_double_forward_adapter(mock_original_forward)
            new_forward = adapter.__get__(mock_block, MMDoubleStreamBlock)

            output = new_forward(
                img=img,
                txt=txt,
                img_mod1_shift=shift_scale,
                img_mod1_scale=shift_scale,
                txt_mod1_shift=shift_scale,
                txt_mod1_scale=shift_scale,
                freqs_cis=freqs_cis,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_kv=cu_seqlens_kv,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_kv=max_seqlen_kv,
            )

            assert mock_block.fa_quantizer.quant_called == 3
            assert mock_block.fa_quantizer.quant_args[0][1] == "q"
            assert mock_block.fa_quantizer.quant_args[1][1] == "k"
            assert mock_block.fa_quantizer.quant_args[2][1] == "v"

            if expect_parallel:
                assert output.shape == (1, 1, 1, 1)
            else:
                expected_shape = (b_forward, 2 * l_forward, mock_block.heads_num, mock_block.head_dim)
                assert output.shape == expected_shape

    @pytest.mark.parametrize("hybrid_attn, expect_parallel", [(False, False), (True, True)])
    def test_single_forward_adapter_normal(self, mocker, hybrid_attn, expect_parallel):
        with mock_hyvideo_module():
            mock_hy_mod = MockHYVideoModule()
            mocker.patch("importlib.import_module", return_value=mock_hy_mod)

            b_forward, l_forward, hidden_dim, txt_len = 2, 20, 256, 10
            qkv = torch.randn(b_forward, l_forward, hidden_dim * 3)
            freqs_cis = torch.randn(l_forward - txt_len, hidden_dim // 2, 2)
            x = torch.randn(b_forward, l_forward, hidden_dim)
            cu_seqlens_q = torch.tensor([0, l_forward, 2 * l_forward, 3 * l_forward, 4 * l_forward], dtype=torch.int32)
            cu_seqlens_kv = cu_seqlens_q
            max_seqlen_q = l_forward
            max_seqlen_kv = l_forward

            mock_block = MMSingleStreamBlock(heads_num=8, hidden_dim=hidden_dim)
            mock_block.hybrid_seq_parallel_attn = hybrid_attn
            mock_block.fa_quantizer = MockFAQuantizer()

            adapter = hyvideo_mm_single_stream_block_single_forward_adapter(mock_original_forward)
            new_forward = adapter.__get__(mock_block, MMSingleStreamBlock)

            output = new_forward(
                qkv=qkv,
                freqs_cis=freqs_cis,
                txt_len=txt_len,
                x=x,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_kv=cu_seqlens_kv,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_kv=max_seqlen_kv,
            )

            assert mock_block.fa_quantizer.quant_called == 3
            assert mock_block.fa_quantizer.quant_args[0][1] == "q"
            assert mock_block.fa_quantizer.quant_args[1][1] == "k"
            assert mock_block.fa_quantizer.quant_args[2][1] == "v"

            if expect_parallel:
                assert output.shape == (1, 1, 1, 1)
            else:
                expected_shape = (b_forward, l_forward, mock_block.heads_num, mock_block.head_dim)
                assert output.shape == expected_shape


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
