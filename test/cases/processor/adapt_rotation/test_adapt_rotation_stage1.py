#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
-------------------------------------------------------------------------
This file is part of the MindStudio project.
Copyright (c) 2026 Huawei Technologies Co.,Ltd.

MindStudio is licensed under Mulan PSL v2.
You may use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:

         http://license.coscl.org.cn/MulanPSL2

THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details.
-------------------------------------------------------------------------
"""
# msmodelslim.processor.adapt_rotation.adapt_rotation_stage1 模块的单元测试
import unittest
from unittest.mock import MagicMock, patch

import torch
from torch import nn

from msmodelslim.core.base.protocol import BatchProcessRequest
from msmodelslim.processor.adapt_rotation.adapt_rotation_stage1 import (
    AdaptRotationStage1Processor,
    AdaptRotationStage1ProcessorConfig,
    LAYER_TYPE_STR_MAX_LEN,
)
from msmodelslim.utils.exception import SchemaValidateError, UnsupportedError

from .common import MockQuaRotAdapter


class TestAdaptRotationStage1ProcessorConfig(unittest.TestCase):
    """测试 AdaptRotationStage1ProcessorConfig 类"""

    def test_config_default_and_valid_values(self):
        """测试默认配置及合法参数（block_size、layer_type、quant_dtype）"""
        config = AdaptRotationStage1ProcessorConfig(type="_adapt_rotation_stage1")
        self.assertEqual(config.steps, 20)
        self.assertEqual(config.quant_dtype, "int4")
        self.assertEqual(config.layer_type, ["up_proj"])
        self.assertEqual(config.block_size, -1)
        self.assertEqual(config.max_samples, 2048)
        config2 = AdaptRotationStage1ProcessorConfig(
            type="_adapt_rotation_stage1",
            block_size=8,
            layer_type=["up_proj", "gate_proj"],
            quant_dtype="int8",
        )
        self.assertEqual(config2.block_size, 8)
        self.assertEqual(config2.layer_type, ["up_proj", "gate_proj"])
        self.assertEqual(config2.quant_dtype, "int8")

    def test_config_validators_reject_invalid(self):
        """测试非法参数时抛出 SchemaValidateError"""
        invalid_cases = [
            ({"block_size": 0}, None),
            ({"block_size": -2}, None),
            ({"block_size": 3}, None),
            ({"layer_type": []}, None),
            ({"layer_type": ["up_proj", 1]}, "string"),
            ({"layer_type": ["  "]}, "empty string"),
            ({"layer_type": ["x" * (LAYER_TYPE_STR_MAX_LEN + 1)]}, str(LAYER_TYPE_STR_MAX_LEN)),
        ]
        for kwargs, keyword in invalid_cases:
            with self.subTest(**kwargs):
                with self.assertRaises(SchemaValidateError) as ctx:
                    AdaptRotationStage1ProcessorConfig(type="_adapt_rotation_stage1", **kwargs)
                if keyword:
                    self.assertIn(keyword, str(ctx.exception).lower())


class TestAdaptRotationStage1Processor(unittest.TestCase):
    """测试 AdaptRotationStage1Processor 类"""

    def test_init_raise_unsupported_error_when_adapter_not_adapt_rotation_interface(self):
        """测试 adapter 非 AdaptRotationInterface 时抛出 UnsupportedError"""
        mock_model = MagicMock()
        config = AdaptRotationStage1ProcessorConfig(type="_adapt_rotation_stage1")
        invalid_adapter = MagicMock()
        with self.assertRaises(UnsupportedError) as ctx:
            AdaptRotationStage1Processor(mock_model, config, invalid_adapter)
        self.assertIn("AdaptRotationInterface", str(ctx.exception))

    def test_init_succeed_when_adapter_implements_adapt_rotation_interface(self):
        """测试 adapter 实现 AdaptRotationInterface 时初始化成功"""
        mock_model = MagicMock()
        config = AdaptRotationStage1ProcessorConfig(type="_adapt_rotation_stage1")
        adapter = MockQuaRotAdapter()
        proc = AdaptRotationStage1Processor(mock_model, config, adapter)
        self.assertEqual(proc.config, config)
        self.assertEqual(proc.model, mock_model)
        self.assertEqual(len(proc.act_dict), 0)
        self.assertEqual(len(proc._stat_hooks), 0)

    def test_support_distributed_return_true(self):
        """测试 support_distributed 返回 True"""
        mock_model = MagicMock()
        config = AdaptRotationStage1ProcessorConfig(type="_adapt_rotation_stage1")
        proc = AdaptRotationStage1Processor(mock_model, config, MockQuaRotAdapter())
        self.assertTrue(proc.support_distributed())

    def test_is_data_free_return_false(self):
        """测试 is_data_free 返回 False"""
        mock_model = MagicMock()
        config = AdaptRotationStage1ProcessorConfig(type="_adapt_rotation_stage1")
        proc = AdaptRotationStage1Processor(mock_model, config, MockQuaRotAdapter())
        self.assertFalse(proc.is_data_free())

    def test_stat_tensor_append_to_act_dict_when_name_exists(self):
        """测试 stat_tensor 当 name 已存在时追加到 act_dict"""
        mock_model = MagicMock()
        config = AdaptRotationStage1ProcessorConfig(type="_adapt_rotation_stage1")
        proc = AdaptRotationStage1Processor(mock_model, config, MockQuaRotAdapter())
        t1 = torch.randn(2, 4, 8)
        t2 = torch.randn(2, 4, 8)
        proc.stat_tensor("layer1", t1)
        proc.stat_tensor("layer1", t2)
        self.assertIn("layer1", proc.act_dict)
        self.assertEqual(len(proc.act_dict["layer1"]), 2)

    def test_stat_tensor_create_new_entry_when_name_not_exists(self):
        """测试 stat_tensor 当 name 不存在时创建新条目"""
        mock_model = MagicMock()
        config = AdaptRotationStage1ProcessorConfig(type="_adapt_rotation_stage1")
        proc = AdaptRotationStage1Processor(mock_model, config, MockQuaRotAdapter())
        t = torch.randn(2, 4, 8)
        proc.stat_tensor("layer1", t)
        self.assertIn("layer1", proc.act_dict)
        self.assertEqual(len(proc.act_dict["layer1"]), 1)
        self.assertEqual(proc.act_dict["layer1"][0].shape, (2, 8))

    def test_pre_run_sets_rot_matrix_and_calls_fuse_norm(self):
        """测试 pre_run 设置 rot_matrix 并调用 _fuse_norm"""
        mock_model = MagicMock()
        config = AdaptRotationStage1ProcessorConfig(type="_adapt_rotation_stage1")
        adapter = MockQuaRotAdapter(hidden_dim=4)
        proc = AdaptRotationStage1Processor(mock_model, config, adapter)
        proc.pre_run()
        self.assertIsNotNone(proc.rot_matrix)
        self.assertTrue(hasattr(proc.rot_matrix, "__len__"))

    def test_preprocess_registers_hooks_for_matching_linear_layers(self):
        """测试 preprocess 为匹配 layer_type 的 Linear 注册前向钩子"""
        linear = nn.Linear(4, 4)
        container = nn.ModuleDict({"up_proj": linear})
        request = BatchProcessRequest(name="layers.0", module=container)
        config = AdaptRotationStage1ProcessorConfig(type="_adapt_rotation_stage1", layer_type=["up_proj"])
        proc = AdaptRotationStage1Processor(MagicMock(), config, MockQuaRotAdapter())
        proc.preprocess(request)
        self.assertEqual(len(proc._stat_hooks), 1)
        inp = torch.randn(2, 1, 4)
        container.up_proj(inp)
        self.assertIn("layers.0.up_proj", proc.act_dict)
        self.assertEqual(len(proc.act_dict["layers.0.up_proj"]), 1)

    def test_preprocess_handles_prefix_empty_string(self):
        """测试 preprocess 当 request.name 为空时 prefix 正确"""
        linear = nn.Linear(4, 4)
        container = nn.ModuleDict({"up_proj": linear})
        request = BatchProcessRequest(name="", module=container)
        config = AdaptRotationStage1ProcessorConfig(type="_adapt_rotation_stage1", layer_type=["up_proj"])
        proc = AdaptRotationStage1Processor(MagicMock(), config, MockQuaRotAdapter())
        proc.preprocess(request)
        container.up_proj(torch.randn(2, 1, 4))
        self.assertIn("up_proj", proc.act_dict)

    def test_preprocess_handles_2d_input_unsqueeze(self):
        """测试 preprocess 钩子对 2D 输入做 unsqueeze"""
        linear = nn.Linear(4, 4)
        container = nn.ModuleDict({"up_proj": linear})
        request = BatchProcessRequest(name="", module=container)
        config = AdaptRotationStage1ProcessorConfig(type="_adapt_rotation_stage1", layer_type=["up_proj"])
        proc = AdaptRotationStage1Processor(MagicMock(), config, MockQuaRotAdapter())
        proc.preprocess(request)
        container.up_proj(torch.randn(2, 4))
        self.assertIn("up_proj", proc.act_dict)

    def test_postprocess_removes_hooks(self):
        """测试 postprocess 移除已注册的钩子"""
        linear = nn.Linear(4, 4)
        container = nn.ModuleDict({"up_proj": linear})
        request = BatchProcessRequest(name="", module=container)
        config = AdaptRotationStage1ProcessorConfig(type="_adapt_rotation_stage1", layer_type=["up_proj"])
        proc = AdaptRotationStage1Processor(MagicMock(), config, MockQuaRotAdapter())
        proc.preprocess(request)
        self.assertEqual(len(proc._stat_hooks), 1)
        proc.postprocess(request)
        self.assertEqual(len(proc._stat_hooks), 0)

    def test_postprocess_when_no_hooks_is_noop(self):
        """测试 postprocess 无钩子时不报错"""
        request = BatchProcessRequest(name="", module=nn.Linear(4, 4))
        config = AdaptRotationStage1ProcessorConfig(type="_adapt_rotation_stage1")
        proc = AdaptRotationStage1Processor(MagicMock(), config, MockQuaRotAdapter())
        proc.postprocess(request)
        self.assertEqual(len(proc._stat_hooks), 0)

    def test_post_run_success_stores_adapted_matrix_in_context(self):
        """测试 post_run 成功时将 adapted_matrix 写入 context"""
        mock_model = MagicMock()
        config = AdaptRotationStage1ProcessorConfig(type="_adapt_rotation_stage1", max_samples=10, steps=2)
        adapter = MockQuaRotAdapter(hidden_dim=4)
        proc = AdaptRotationStage1Processor(mock_model, config, adapter)
        proc.pre_run()
        proc.act_dict["layer1"] = [torch.randn(4, 4)]
        ns = type("NS", (), {"state": {}})()
        ctx = {"adapt_rotation": ns}
        with patch("msmodelslim.processor.adapt_rotation.adapt_rotation_stage1.get_current_context", return_value=ctx):
            proc.post_run()
        self.assertIn("adapted_matrix", ns.state)
        self.assertEqual(ns.state["adapted_matrix"].shape, (4, 4))

    def test_post_run_when_total_exceeds_max_samples_uses_linspace_subsample(self):
        """测试 post_run 当总样本数超过 max_samples 时使用 linspace 下采样"""
        mock_model = MagicMock()
        config = AdaptRotationStage1ProcessorConfig(type="_adapt_rotation_stage1", max_samples=5, steps=2)
        adapter = MockQuaRotAdapter(hidden_dim=4)
        proc = AdaptRotationStage1Processor(mock_model, config, adapter)
        proc.pre_run()
        proc.act_dict["layer1"] = [torch.randn(20, 4) for _ in range(2)]
        ns = type("NS", (), {"state": {}})()
        with patch("msmodelslim.processor.adapt_rotation.adapt_rotation_stage1.get_current_context", return_value={"adapt_rotation": ns}):
            proc.post_run()
        self.assertIn("adapted_matrix", ns.state)

    def test_post_run_raise_when_act_dict_empty(self):
        """测试 post_run 当 act_dict 为空时抛出 UnsupportedError"""
        mock_model = MagicMock()
        config = AdaptRotationStage1ProcessorConfig(type="_adapt_rotation_stage1")
        proc = AdaptRotationStage1Processor(mock_model, config, MockQuaRotAdapter())
        with self.assertRaises(UnsupportedError) as ctx:
            proc.post_run()
        self.assertIn("no activations", str(ctx.exception).lower())

    def test_post_run_raise_when_context_none(self):
        """测试 post_run 当 get_current_context 返回 None 时抛出 UnsupportedError"""
        mock_model = MagicMock()
        config = AdaptRotationStage1ProcessorConfig(type="_adapt_rotation_stage1", max_samples=10, steps=2)
        adapter = MockQuaRotAdapter(hidden_dim=4)
        proc = AdaptRotationStage1Processor(mock_model, config, adapter)
        proc.pre_run()
        proc.act_dict["layer1"] = [torch.randn(4, 4)]
        with patch("msmodelslim.processor.adapt_rotation.adapt_rotation_stage1.get_current_context", return_value=None):
            with self.assertRaises(UnsupportedError) as ctx:
                proc.post_run()
            self.assertIn("context", str(ctx.exception).lower())

    def test_fuse_norm_with_single_key_value(self):
        """测试 _fuse_norm 单 key/value 时调用 get_submodule 并 fuse_ln_linear"""
        ln = nn.LayerNorm(4)
        linear = nn.Linear(4, 4)
        model = nn.ModuleDict({"ln": ln, "linear": linear})
        config = AdaptRotationStage1ProcessorConfig(type="_adapt_rotation_stage1")
        proc = AdaptRotationStage1Processor(model, config, MockQuaRotAdapter())
        with patch("msmodelslim.processor.adapt_rotation.adapt_rotation_stage1.fuse_ln_linear") as mock_fuse:
            proc._fuse_norm({"ln": "linear"})
        mock_fuse.assert_called_once()
        self.assertEqual(len(mock_fuse.call_args[0][0]), 1)
        self.assertEqual(len(mock_fuse.call_args[0][1]), 1)

    def test_fuse_norm_with_list_key_value(self):
        """测试 _fuse_norm key/value 为 list 时收集多个 module"""
        ln1 = nn.LayerNorm(4)
        ln2 = nn.LayerNorm(4)
        linear = nn.Linear(4, 4)
        model = nn.ModuleDict({"ln1": ln1, "ln2": ln2, "linear": linear})
        config = AdaptRotationStage1ProcessorConfig(type="_adapt_rotation_stage1")
        proc = AdaptRotationStage1Processor(model, config, MockQuaRotAdapter())
        with patch("msmodelslim.processor.adapt_rotation.adapt_rotation_stage1.fuse_ln_linear") as mock_fuse:
            proc._fuse_norm({("ln1", "ln2"): "linear"})
        mock_fuse.assert_called_once()
        self.assertEqual(len(mock_fuse.call_args[0][0]), 2)
        self.assertEqual(len(mock_fuse.call_args[0][1]), 1)

    def test_fuse_norm_reraises_unsupported_error_from_fuse_ln_linear(self):
        """测试 _fuse_norm 当 fuse_ln_linear 抛出 UnsupportedError 时包装后重新抛出"""
        model = nn.ModuleDict({"ln": nn.LayerNorm(4), "linear": nn.Linear(4, 4)})
        config = AdaptRotationStage1ProcessorConfig(type="_adapt_rotation_stage1")
        proc = AdaptRotationStage1Processor(model, config, MockQuaRotAdapter())
        with patch("msmodelslim.processor.adapt_rotation.adapt_rotation_stage1.fuse_ln_linear") as mock_fuse:
            mock_fuse.side_effect = UnsupportedError("size mismatch", action="check")
            with self.assertRaises(UnsupportedError) as ctx:
                proc._fuse_norm({"ln": "linear"})
            self.assertIn("fuse layer norm and linear error", str(ctx.exception))
            self.assertIsNotNone(ctx.exception.__cause__)
