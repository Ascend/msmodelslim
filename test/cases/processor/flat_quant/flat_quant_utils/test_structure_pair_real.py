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

from torch import nn

from msmodelslim.processor.flat_quant.flat_quant import FlatQuantProcessorConfig
from msmodelslim.processor.flat_quant.flat_quant_utils.flat_fake_quant_linear import (
    FlatFakeQuantLinear,
    FlatNormWrapper,
    ForwardMode,
)
from msmodelslim.processor.flat_quant.flat_quant_utils.structure_pair import (
    AttnLinearLinearPair,
    AttnNormLinearPair,
    MLPLinearLinearPair,
    StructurePair,
)
from msmodelslim.processor.flat_quant.flat_quant_utils.trans_matrix import GeneralMatrixTrans


def _make_config():
    """构造最小 FlatQuantProcessorConfig（init=True 字段）。"""
    return FlatQuantProcessorConfig(type="flatquant", include=["*"], exclude=[])


class _AttnModel(nn.Module):
    """带 Norm + QKV Linear 的最小模型（适配 AttnNormLinearPair）。"""

    def __init__(self, dim=8):
        super().__init__()
        self.attn_norm = nn.LayerNorm(dim)
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)

    def tie_weights(self):
        pass


class _QModel(nn.Module):
    """带前缀属性 Q.* 的模型：prefix="Q" 切到 Q.q_proj 等子模块。"""

    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(8, 8)
        self.up_proj = nn.Linear(8, 8)

    def tie_weights(self):
        pass


class _QMLPModel(nn.Module):
    """带 Q.gate_proj / Q.up_proj 的模型。"""

    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(8, 8)
        self.up_proj = nn.Linear(8, 8)

    def tie_weights(self):
        pass


class TestAttnNormLinearPair:
    """Test suite for AttnNormLinearPair — 注意力层 Norm+Q/K/V Linear PAI。"""

    def _make_pair(self, inner):
        config = _make_config()
        return AttnNormLinearPair(
            config=config,
            attn_norm_name="Q.attn_norm",
            linear_name=["Q.q_proj", "Q.k_proj", "Q.v_proj"],
            prefix_name="Q",
            model=inner,
        )

    def test_init_stores_config_and_sources(self):
        """主路径：构造时应记录 config、source、target。"""
        pair = self._make_pair(_AttnModel())

        assert pair.config is not None
        assert "Q.attn_norm" in pair.source_modules
        assert "Q.q_proj" in pair.target_modules
        assert "Q.k_proj" in pair.target_modules
        assert "Q.v_proj" in pair.target_modules

    def test_init_sets_register_to_true_and_in_global_list(self):
        """主路径：AttnNormLinearPair 应被注册到 support_structure_pairs。"""
        assert AttnNormLinearPair._register is True
        assert AttnNormLinearPair in StructurePair.support_structure_pairs

    def test_wrap_linear_replaces_all_qkv_with_flat_fake_quant_linear(self):
        """主路径：wrap_linear 应将所有 Q/K/V Linear 替换为 FlatFakeQuantLinear。"""
        inner = _AttnModel()
        pair = self._make_pair(inner)

        pair.wrap_linear()

        assert isinstance(inner.q_proj, FlatFakeQuantLinear)
        assert isinstance(inner.k_proj, FlatFakeQuantLinear)
        assert isinstance(inner.v_proj, FlatFakeQuantLinear)

    def test_wrap_linear_creates_trans_matrix(self):
        """主路径：wrap_linear 应创建 linear_trans（GeneralMatrixTrans）。"""
        pair = self._make_pair(_AttnModel())

        pair.wrap_linear()

        assert isinstance(pair.linear_trans, GeneralMatrixTrans)

    def test_wrap_linear_replaces_norm_with_flat_norm_wrapper(self):
        """主路径：wrap_linear 应将 attn_norm 替换为 FlatNormWrapper。"""
        inner = _AttnModel()
        pair = self._make_pair(inner)

        pair.wrap_linear()

        assert isinstance(inner.attn_norm, FlatNormWrapper)

    def test_change_mode_to_eval_reparameterizes_trans(self):
        """主路径：change_mode(EVAL) 应触发重参数化（_eval_mode=True）。"""
        pair = self._make_pair(_AttnModel())
        pair.wrap_linear()

        pair.change_mode(ForwardMode.EVAL)

        assert pair.linear_trans.left_trans._eval_mode is True
        assert pair.linear_trans.right_trans._eval_mode is True

    def test_change_mode_to_calib_does_not_reparameterize(self):
        """边界：change_mode(CALIB) 不应触发重参数化。"""
        pair = self._make_pair(_AttnModel())
        pair.wrap_linear()

        pair.change_mode(ForwardMode.CALIB)

        assert pair.linear_trans.left_trans._eval_mode is False
        assert pair.linear_trans.right_trans._eval_mode is False

    def test_change_mode_to_org_does_not_reparameterize(self):
        """边界：change_mode(ORG) 不应触发重参数化。"""
        pair = self._make_pair(_AttnModel())
        pair.wrap_linear()

        pair.change_mode(ForwardMode.ORG)

        assert pair.linear_trans.left_trans._eval_mode is False

    def test_rollback_trans_does_not_raise_when_called(self):
        """主路径：rollback_trans 是预留 no-op（不抛错）。"""
        pair = self._make_pair(_AttnModel())
        pair.wrap_linear()

        pair.rollback_trans("attn_norm")


class TestAttnLinearLinearPair:
    """Test suite for AttnLinearLinearPair — 注意力层 Linear+Linear PAI。"""

    def _make_pair(self):
        config = _make_config()
        wrapper = _QModel()
        # 预替换 source Linear 为 FlatFakeQuantLinear（生产时由前一个 PAI 完成）
        original_q = wrapper.q_proj
        wrapper.q_proj = FlatFakeQuantLinear(config=config, linear=original_q)
        # 同样预替换 post（如果 wrap_linear 不替换了）
        return AttnLinearLinearPair(
            config=config,
            pre_linear_name="Q.q_proj",  # sources = str
            post_linear_name=["Q.up_proj"],  # targets = list
            prefix_name="Q",
            model=wrapper,
            head_dim=4,
            num_attention_heads=2,
        )

    def test_init_stores_config_and_source_target(self):
        """主路径：构造时应记录 source/target。"""
        pair = self._make_pair()

        assert "Q.q_proj" in pair.source_modules
        assert "Q.up_proj" in pair.target_modules

    def test_wrap_linear_creates_flat_fake_quant_linear(self):
        """主路径：wrap_linear 应将 post Linear 替换为 FlatFakeQuantLinear。"""
        pair = self._make_pair()

        pair.wrap_linear()

        assert isinstance(pair.model.up_proj, FlatFakeQuantLinear)

    def test_rollback_trans_does_not_raise_when_called(self):
        """主路径：rollback_trans 应能处理 pre_linear_name 在目标列表中的情况。"""
        pair = self._make_pair()
        pair.wrap_linear()

        # rollback_trans 检查 pair_name 是否在 target_modules 中
        pair.rollback_trans("Q.up_proj")  # 不抛错


class TestMLPLinearLinearPair:
    """Test suite for MLPLinearLinearPair — MLP 层 Linear+Linear PAI。"""

    def _make_pair(self):
        config = _make_config()
        wrapper = _QMLPModel()
        # 预替换 source Linear 为 FlatFakeQuantLinear
        original_gate = wrapper.gate_proj
        wrapper.gate_proj = FlatFakeQuantLinear(config=config, linear=original_gate)
        return MLPLinearLinearPair(
            config=config,
            pre_linear_name="Q.gate_proj",
            post_linear_name=["Q.up_proj"],
            prefix_name="Q",
            model=wrapper,
        )

    def test_init_stores_config(self):
        """主路径：构造时应记录 config。"""
        pair = self._make_pair()

        assert pair.config is not None
        assert "Q.gate_proj" in pair.source_modules
        assert "Q.up_proj" in pair.target_modules

    def test_wrap_linear_creates_flat_fake_quant_linear(self):
        """主路径：wrap_linear 应将 post Linear 替换为 FlatFakeQuantLinear。"""
        pair = self._make_pair()

        pair.wrap_linear()

        assert isinstance(pair.model.up_proj, FlatFakeQuantLinear)

    def test_rollback_trans_does_not_raise_when_called(self):
        """主路径：rollback_trans 应能处理 pre_linear_name 在目标列表中的情况。"""
        pair = self._make_pair()
        pair.wrap_linear()

        pair.rollback_trans("Q.up_proj")
