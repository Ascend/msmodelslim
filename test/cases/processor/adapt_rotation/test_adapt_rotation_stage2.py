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
# pylint: disable=pointless-string-statement

"""
msmodelslim.processor.adapt_rotation.adapt_rotation_stage2 模块的单元测试
"""
import unittest  # noqa: E402
from unittest.mock import MagicMock, patch  # noqa: E402

import torch  # noqa: E402

from msmodelslim.processor.adapt_rotation.adapt_rotation_stage2 import (  # noqa: E402
    AdaptRotationStage2ProcessorConfig,
    AdaptRotationStage2Processor,
)
from msmodelslim.processor.quarot.offline_quarot.quarot_interface import RotatePair  # noqa: E402
from msmodelslim.utils.exception import SchemaValidateError  # noqa: E402

from .common import MockQuaRotAdapter  # noqa: E402


class TestAdaptRotationStage2ProcessorConfig(unittest.TestCase):
    """测试 AdaptRotationStage2ProcessorConfig 类"""

    def test_config_default_and_valid_values(self):
        """测试默认配置及合法参数（type、block_size、max_tp_size、down_proj_online_layers）"""
        config = AdaptRotationStage2ProcessorConfig(type="_adapt_rotation_stage2")
        self.assertEqual(config.type, "_adapt_rotation_stage2")
        self.assertEqual(config.block_size, -1)
        self.assertEqual(config.max_tp_size, 4)
        self.assertEqual(config.down_proj_online_layers, [])
        config2 = AdaptRotationStage2ProcessorConfig(
            type="_adapt_rotation_stage2",
            block_size=8,
            max_tp_size=2,
            down_proj_online_layers=[0, 1],
        )
        self.assertEqual(config2.block_size, 8)
        self.assertEqual(config2.max_tp_size, 2)
        self.assertEqual(config2.down_proj_online_layers, [0, 1])

    def test_config_validators_reject_invalid(self):
        """测试非法参数时抛出 SchemaValidateError"""
        invalid_cases = [
            ({"down_proj_online_layers": [0, "x"]}, "int"),
            ({"down_proj_online_layers": [-1]}, ">= 0"),
            ({"max_tp_size": 0}, None),  # Field(ge=1) 报错
            ({"max_tp_size": 3}, "power"),
            ({"block_size": 0}, None),
            ({"block_size": -2}, None),
            ({"block_size": 3}, "power"),
        ]
        for kwargs, keyword in invalid_cases:
            with self.subTest(**kwargs):
                with self.assertRaises(SchemaValidateError) as ctx:
                    AdaptRotationStage2ProcessorConfig(type="_adapt_rotation_stage2", **kwargs)
                if keyword:
                    self.assertIn(keyword, str(ctx.exception).lower())


class TestAdaptRotationStage2ProcessorOverlayAdaptedMatrix(unittest.TestCase):
    """测试 AdaptRotationStage2Processor._overlay_adapted_matrix 方法"""

    def test_overlay_adapted_matrix(self):
        """测试形状匹配时替换、形状不匹配/list/tuple 时跳过"""
        D = 4
        adapted = torch.eye(D)
        config = AdaptRotationStage2ProcessorConfig(type="_adapt_rotation_stage2")
        processor = AdaptRotationStage2Processor(model=MagicMock(), config=config, adapter=MockQuaRotAdapter())

        with self.subTest(match="形状匹配时替换"):
            pair = RotatePair(
                left_rot={"layer1": torch.randn(D, D)},
                right_rot={"layer2": torch.randn(D, D)},
            )
            processor._overlay_adapted_matrix([pair], adapted)
            self.assertTrue(torch.equal(pair.left_rot["layer1"], adapted))
            self.assertTrue(torch.equal(pair.right_rot["layer2"], adapted))

        with self.subTest(match="形状不匹配时跳过"):
            original = torch.randn(8, 8)
            pair2 = RotatePair(
                left_rot={"layer1": torch.randn(D, D).clone()},
                right_rot={"layer2": original},
            )
            processor._overlay_adapted_matrix([pair2], adapted)
            self.assertTrue(torch.equal(pair2.left_rot["layer1"], adapted))
            self.assertTrue(torch.equal(pair2.right_rot["layer2"], original))

        with self.subTest(match="rot 为 list/tuple 时跳过"):
            pair3 = RotatePair(
                left_rot={"layer1": [torch.eye(D), torch.randn(2, 2)]},
                right_rot={"layer2": (torch.eye(D),)},
            )
            processor._overlay_adapted_matrix([pair3], adapted)
            self.assertIsInstance(pair3.left_rot["layer1"], list)
            self.assertIsInstance(pair3.right_rot["layer2"], tuple)


class TestAdaptRotationStage2ProcessorPreRun(unittest.TestCase):
    """测试 AdaptRotationStage2Processor.pre_run 方法"""

    def _make_proc_with_mocked_fuse_bake_rotate(self):
        model = MagicMock()
        config = AdaptRotationStage2ProcessorConfig(type="_adapt_rotation_stage2")
        proc = AdaptRotationStage2Processor(model, config, MockQuaRotAdapter())
        proc._fuse_norm = MagicMock()
        proc._bake_mean = MagicMock()
        proc._rotate = MagicMock()
        return proc

    def test_pre_run_without_adapted_matrix_in_context(self):
        """测试 context 为 None 或缺少 adapt_rotation/adapted_matrix 时走默认旋转路径"""
        for ctx_value in (None, {}):
            with self.subTest(ctx=ctx_value):
                proc = self._make_proc_with_mocked_fuse_bake_rotate()
                with patch(
                    "msmodelslim.processor.adapt_rotation.adapt_rotation_stage2.get_current_context",
                    return_value=ctx_value,
                ):
                    proc.pre_run()
                proc._fuse_norm.assert_called_once()
                proc._bake_mean.assert_called_once()
                self.assertEqual(proc.rotate_commands, [])

    def test_pre_run_when_context_has_adapted_matrix_overlays_onto_rotations(self):
        """测试 context 含 adapted_matrix 时覆盖到匹配维度的旋转上"""
        D = 4
        adapted = torch.eye(D)
        pair = RotatePair(
            left_rot={"l": torch.randn(D, D)},
            right_rot={"r": torch.randn(D, D)},
        )

        class AdapterWithPairs(MockQuaRotAdapter):
            def get_rotate_map(self, block_size: int):
                return [pair], [pair]

        proc = AdaptRotationStage2Processor(
            MagicMock(), AdaptRotationStage2ProcessorConfig(type="_adapt_rotation_stage2"), AdapterWithPairs()
        )
        proc._fuse_norm = MagicMock()
        proc._bake_mean = MagicMock()
        proc._rotate = MagicMock()
        ns = type("NS", (), {"state": {"adapted_matrix": adapted}})()
        with patch(
            "msmodelslim.processor.adapt_rotation.adapt_rotation_stage2.get_current_context",
            return_value={"adapt_rotation": ns},
        ):
            proc.pre_run()
        self.assertTrue(torch.equal(pair.left_rot["l"], adapted))
        self.assertTrue(torch.equal(pair.right_rot["r"], adapted))
        proc._fuse_norm.assert_called_once()
        proc._rotate.assert_called()

    def test_pre_run_when_online_true_calls_online_processor_pre_run(self):
        """测试 config.online=True 时调用 online_processor.pre_run"""
        model = MagicMock()
        config = AdaptRotationStage2ProcessorConfig(type="_adapt_rotation_stage2", online=True)
        adapter = MockQuaRotAdapter()
        with patch.object(AdaptRotationStage2Processor, "__init__", lambda s, m, c, a, **kw: None):
            proc = AdaptRotationStage2Processor.__new__(AdaptRotationStage2Processor)
            proc.model = model
            proc.config = config
            proc.adapter = adapter
            proc.fused_map = {}
            proc.bake_names = []
            proc.rotate_commands = []
            proc.rotate_pairs = []
            proc.online_processor = MagicMock()
            proc._fuse_norm = MagicMock()
            proc._bake_mean = MagicMock()
            proc._rotate = MagicMock()
        with patch("msmodelslim.processor.adapt_rotation.adapt_rotation_stage2.get_current_context", return_value=None):
            with patch(
                "msmodelslim.processor.adapt_rotation.adapt_rotation_stage2.get_rotate_command", return_value=[]
            ):
                proc.pre_run()
        proc.online_processor.pre_run.assert_called_once()
