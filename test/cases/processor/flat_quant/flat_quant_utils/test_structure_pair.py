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

from msmodelslim.processor.flat_quant.flat_quant_utils.structure_pair import StructurePair


class _FakeStructurePair(StructurePair, register=False):
    """跳过注册到 support_structure_pairs，避免污染全局列表。"""

    def __init__(self, target_modules):  # pylint: disable=super-init-not-called
        # 跳过父类 __init__：直接设置属性以测试 contain 等方法
        self.target_modules = target_modules

    def wrap_linear(self):
        pass

    def rollback_trans(self, pair_name=""):
        pass

    def change_mode(self, mod):
        pass


class TestStructurePairContain:
    """Test suite for StructurePair.contain — 判断模块是否属于该结构对目标。"""

    # ---------- 正常情形 ----------

    def test_contain_returns_true_when_name_matches_target(self):
        """主路径：name 命中 target_modules 应返回 True。"""
        pair = _FakeStructurePair(target_modules=["layer.weight", "layer.bias"])

        assert pair.contain("layer.weight") is True
        assert pair.contain("layer.bias") is True

    def test_contain_returns_false_when_name_does_not_match_any_target(self):
        """主路径：name 不在 target_modules 应返回 False。"""
        pair = _FakeStructurePair(target_modules=["layer.weight"])

        assert pair.contain("other.weight") is False

    # ---------- 边界情形 ----------

    def test_contain_returns_false_when_target_modules_is_empty(self):
        """边界：target_modules 为空时任何 name 都返回 False。"""
        pair = _FakeStructurePair(target_modules=[])

        assert pair.contain("any.name") is False

    def test_contain_returns_false_when_name_is_empty_string(self):
        """边界：name 为空串时不应匹配（除非 target_modules 含空串）。"""
        pair = _FakeStructurePair(target_modules=["layer.weight"])

        assert pair.contain("") is False

    def test_contain_handles_unicode_name_when_target_is_unicode(self):
        """边界：unicode name 也能匹配 unicode target。"""
        pair = _FakeStructurePair(target_modules=["层.权重"])

        assert pair.contain("层.权重") is True


class TestStructurePairRegistration:
    """Test suite for StructurePair 子类注册 — 验证 __init_subclass__ 行为。"""

    def test_subclass_is_registered_in_support_structure_pairs_by_default(self):
        """主路径：默认情况下子类应被加入 support_structure_pairs。"""
        before = list(StructurePair.support_structure_pairs)

        class _Registered(StructurePair, register=True):
            pass

        try:
            assert _Registered in StructurePair.support_structure_pairs
        finally:
            # 清理类级共享状态，避免影响其他用例
            StructurePair.support_structure_pairs[:] = before

    def test_subclass_with__register_false_is_not_registered(self):
        """主路径：_register=False 类属性时子类不应被加入。"""

        class _NotRegistered(StructurePair):
            _register = False

        # 该类不应出现在注册列表中
        assert _NotRegistered not in StructurePair.support_structure_pairs
