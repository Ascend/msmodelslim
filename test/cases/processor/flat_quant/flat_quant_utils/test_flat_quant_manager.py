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

import pytest
from torch import nn

from msmodelslim.processor.flat_quant.flat_quant_utils.flat_fake_quant_linear import ForwardMode
from msmodelslim.processor.flat_quant.flat_quant_utils.flat_quant_manager import FlatQuantLayerManager
from msmodelslim.processor.flat_quant.flat_quant_utils.structure_pair import StructurePair
from msmodelslim.utils.exception import SchemaValidateError


class _FakeStructurePair(StructurePair):
    """最简 StructurePair 子类，跳过真实 wrap/rollback 逻辑。"""

    pair_type = "fake"

    def __init__(self, name="fake_pair"):  # pylint: disable=super-init-not-called
        # 跳过父类 __init__ 避免 torch 参数注册要求
        self.name = name
        self.wrap_called = False
        self.rollback_called = False
        self.mode_changed = None

    def wrap_linear(self):
        self.wrap_called = True

    def rollback_trans(self, pair_name=""):
        self.rollback_called = True

    def change_mode(self, mod):
        self.mode_changed = mod


class TestFlatQuantLayerManager:
    """Test suite for FlatQuantLayerManager — 逐层管理结构对。"""

    def _build_manager(self):
        module = nn.Linear(4, 4)
        return FlatQuantLayerManager(module)

    # ---------- 正常情形 ----------

    def test_register_structure_pair_adds_pair_to_map_when_called_with_valid_pair(self):
        """主路径：注册合法 pair 应加入 _structure_pair_map。"""
        manager = self._build_manager()
        pair = _FakeStructurePair("p1")

        manager.register_structure_pair(pair)

        assert "_FakeStructurePair" in manager._structure_pair_map
        assert pair in manager._structure_pair_map["_FakeStructurePair"]

    def test_register_structure_pair_dedups_when_same_pair_registered_twice(self):
        """主路径：相同 pair 不应重复注册。"""
        manager = self._build_manager()
        pair = _FakeStructurePair("p1")

        manager.register_structure_pair(pair)
        manager.register_structure_pair(pair)

        assert len(manager._structure_pair_map["_FakeStructurePair"]) == 1

    def test_wrap_linear_invokes_method_on_each_pair_when_called(self):
        """主路径：wrap_linear 应调用每个 pair 的 wrap_linear。"""
        manager = self._build_manager()
        p1, p2 = _FakeStructurePair("p1"), _FakeStructurePair("p2")
        manager.register_structure_pair(p1)
        manager.register_structure_pair(p2)
        # 强制加入 _layer_pairs_list
        manager._layer_pairs_list = [p1, p2]

        manager.wrap_linear(device="cpu")

        assert p1.wrap_called is True
        assert p2.wrap_called is True

    def test_rollback_trans_invokes_method_on_each_pair_when_called(self):
        """主路径：rollback_trans 应调用每个 pair 的 rollback。"""
        manager = self._build_manager()
        p1 = _FakeStructurePair()
        manager._layer_pairs_list = [p1]

        manager.rollback_trans()

        assert p1.rollback_called is True

    def test_change_mode_passes_mode_to_each_pair_when_called(self):
        """主路径：change_mode 应将 mod 传给每个 pair。"""
        manager = self._build_manager()
        p1 = _FakeStructurePair()
        manager._layer_pairs_list = [p1]
        mod = ForwardMode.CALIB

        manager.change_mode(mod)

        assert p1.mode_changed is mod

    def test_match_pair_is_a_no_op_when_called(self):
        """边界：match_pair 当前是预留 no-op。"""
        manager = self._build_manager()

        # 不应抛错
        manager.match_pair("any_name")

    # ---------- 异常情形 ----------

    def test_register_structure_pair_raises_schema_validate_error_when_not_pair_instance(self):
        """异常：非 StructurePair 实例应抛 SchemaValidateError。"""
        manager = self._build_manager()

        with pytest.raises(SchemaValidateError):
            manager.register_structure_pair("not a pair")  # type: ignore[arg-type]
