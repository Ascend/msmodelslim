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
# pylint: disable=arguments-out-of-order

import pytest

from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.quant_config import QuantConfig
from .old_quant_config import OldQuantConfig
from .compare_old_and_new_config import compare_config_parameters


class TestQuantConfig:
    def test_default_quant_config_param_should_equal_old_quant_config_given_defalt_param(self):
        new_config = QuantConfig()
        old_config = OldQuantConfig()
        assert compare_config_parameters(new_config, old_config) is True

    def test_default_quant_config_param_should_not_equal_old_quant_config_given_not_equal_param(self):
        new_config = QuantConfig(a_bit=16)
        old_config = OldQuantConfig()
        assert compare_config_parameters(new_config, old_config) is False

    def test_old_w8a8_quant_config_should_compatible_when_given_equal_w8a8_param(self):
        new_config = QuantConfig(
            a_bit=8,
            w_bit=8,
            disable_names=['layer.0.a'],
            dev_type='cpu',
            act_method=3,
            pr=1.0,
            w_sym=True,
            mm_tensor=False,
        )
        old_config = OldQuantConfig(
            a_bit=8,
            w_bit=8,
            disable_names=['layer.0.a'],
            dev_type='cpu',
            act_method=3,
            pr=1.0,
            w_sym=True,
            mm_tensor=False,
        )
        assert compare_config_parameters(new_config, old_config) is True

    def test_old_w8a8_quant_config_should_not_compatible_when_given_not_equal_w8a8_param(self):
        new_config = QuantConfig(
            a_bit=8,
            w_bit=8,
            disable_names=['layer.0.linear1'],
            dev_type='cpu',
            act_method=3,
            pr=1.0,
            w_sym=True,
            mm_tensor=False,
        )
        old_config = OldQuantConfig(
            a_bit=8,
            w_bit=8,
            disable_names=['layer.0.linear2'],
            dev_type='cpu',
            act_method=3,
            pr=1.0,
            w_sym=True,
            mm_tensor=False,
        )
        assert compare_config_parameters(new_config, old_config) is False

    def test_old_w8a16_quant_config_should_compatible_when_given_equal_w8a16_param(self):
        new_config = QuantConfig(a_bit=16, w_bit=8, disable_names=[], dev_type='cpu', w_sym=True, mm_tensor=False)
        old_config = OldQuantConfig(a_bit=16, w_bit=8, disable_names=[], dev_type='cpu', w_sym=True, mm_tensor=False)
        assert compare_config_parameters(new_config, old_config) is True

    def test_old_w8a16_quant_config_should_not_compatible_when_given_not_equal_w8a16_param(self):
        new_config = QuantConfig(a_bit=16, w_bit=8, disable_names=[], dev_type='cpu', w_sym=True, mm_tensor=False)
        old_config = OldQuantConfig(a_bit=16, w_bit=8, disable_names=[], dev_type='cpu', w_sym=True, mm_tensor=True)
        assert compare_config_parameters(new_config, old_config) is False

    def test_old_sparse_quant_config_should_compatible_when_given_equal_sparse_param(self):
        new_config = QuantConfig(
            act_method=3, pr=2.0, fraction=0.011, nonuniform=False, mm_tensor=False, co_sparse=True
        )
        old_config = OldQuantConfig(
            act_method=3, pr=2.0, fraction=0.011, nonuniform=False, mm_tensor=False, co_sparse=True
        )
        assert compare_config_parameters(new_config, old_config) is True

    def test_old_sparse_quant_config_should_not_compatible_when_given_not_equal_sparse_param(self):
        new_config = QuantConfig(
            act_method=3, pr=2.0, fraction=0.011, nonuniform=False, mm_tensor=False, co_sparse=True
        )
        old_config = OldQuantConfig(
            act_method=3, pr=2.0, fraction=0.012, nonuniform=False, mm_tensor=False, co_sparse=True
        )
        assert compare_config_parameters(new_config, old_config) is False

    def test_old_sparse_quant_config_should_compatible_when_given_equal_low_bit_param(self):
        new_config = QuantConfig(
            disable_names=[
                'lm_head',
                'model.layers.0.self_attn.q_proj',
                'model.layers.0.self_attn.k_proj',
                'model.layers.0.self_attn.v_proj',
                'model.layers.0.self_attn.o_proj',
                'model.layers.0.mlp.gate_proj',
                'model.layers.0.mlp.up_proj',
                'model.layers.0.mlp.down_proj',
            ],
            do_smooth=False,
            is_lowbit=True,
            use_sigma=False,
        )
        old_config = OldQuantConfig(
            disable_names=[
                'lm_head',
                'model.layers.0.self_attn.q_proj',
                'model.layers.0.self_attn.k_proj',
                'model.layers.0.self_attn.v_proj',
                'model.layers.0.self_attn.o_proj',
                'model.layers.0.mlp.gate_proj',
                'model.layers.0.mlp.up_proj',
                'model.layers.0.mlp.down_proj',
            ],
            do_smooth=False,
            is_lowbit=True,
            use_sigma=False,
        )
        assert compare_config_parameters(new_config, old_config) is True

    def test_old_sparse_quant_config_should_not_compatible_when_given_not_equal_low_bit_param(self):
        new_config = QuantConfig(
            disable_names=[
                'lm_head',
                'model.layers.0.self_attn.q_proj',
                'model.layers.0.self_attn.k_proj',
                'model.layers.0.self_attn.v_proj',
                'model.layers.0.self_attn.o_proj',
                'model.layers.0.mlp.gate_proj',
                'model.layers.0.mlp.up_proj',
                'model.layers.0.mlp.down_proj',
            ],
            do_smooth=False,
            is_lowbit=True,
            use_sigma=False,
        )
        old_config = OldQuantConfig(
            disable_names=[
                'lm_head',
                'model.layers.0.self_attn.q_proj',
                'model.layers.0.self_attn.k_proj',
                'model.layers.0.self_attn.v_proj',
                'model.layers.0.self_attn.o_proj',
                'model.layers.0.mlp.gate_proj',
                'model.layers.0.mlp.up_proj',
                'model.layers.0.mlp.down_proj',
            ],
            do_smooth=True,
            is_lowbit=True,
            use_sigma=False,
        )
        assert compare_config_parameters(new_config, old_config) is False

    def test_old_kv_quant_config_should_compatible_when_given_equal_kv_param(self):
        new_config = QuantConfig(
            a_bit=8,
            w_bit=8,
            disable_names=[],
            dev_type='cpu',
            dev_id=0,
            act_method=3,
            pr=1.0,
            w_sym=True,
            mm_tensor=False,
            use_kvcache_quant=True,
        )
        old_config = OldQuantConfig(
            a_bit=8,
            w_bit=8,
            disable_names=[],
            dev_type='cpu',
            dev_id=0,
            act_method=3,
            pr=1.0,
            w_sym=True,
            mm_tensor=False,
            use_kvcache_quant=True,
        )
        assert compare_config_parameters(new_config, old_config) is True

    def test_old_kv_quant_config_should_not_compatible_when_given_not_equal_kv_param(self):
        new_config = QuantConfig(
            a_bit=8,
            w_bit=8,
            disable_names=[],
            dev_type='cpu',
            dev_id=0,
            act_method=3,
            pr=1.0,
            w_sym=True,
            mm_tensor=False,
            use_kvcache_quant=True,
        )
        old_config = OldQuantConfig(
            a_bit=16,
            w_bit=8,
            disable_names=[],
            dev_type='cpu',
            dev_id=0,
            act_method=3,
            pr=1.0,
            w_sym=True,
            mm_tensor=False,
            use_kvcache_quant=True,
        )
        assert compare_config_parameters(new_config, old_config) is False

    def test_new_kv_quant_config_should_equal_to_old_kv_quant_config_when_given_right_kv_param(self):
        old_usage = QuantConfig(use_kvcache_quant=True)
        new_usage = QuantConfig().kv_quant()
        assert compare_config_parameters(old_usage, new_usage) is True

    def test_new_kv_quant_config_should_not_equal_to_old_kv_quant_config_when_given_fault_kv_param(self):
        old_usage = QuantConfig(use_kvcache_quant=False)
        new_usage = QuantConfig().kv_quant()
        assert compare_config_parameters(old_usage, new_usage) is False

    def test_new_weight_quant_config_should_equal_to_old_weight_quant_config_when_given_right_param(self):
        old_usage = QuantConfig(a_bit=16, w_bit=8, mm_tensor=False, w_method='MinMax')
        new_usage = QuantConfig(a_bit=16, w_bit=8).weight_quant(w_method='MinMax', mm_tensor=False, w_sym=True)
        assert compare_config_parameters(old_usage, new_usage) is True

    def test_new_weight_quant_config_should_equal_to_old_weight_quant_config_when_given_fault_param(self):
        old_usage = QuantConfig(a_bit=16, w_bit=8, mm_tensor=False, w_method='HQQ')
        new_usage = QuantConfig(a_bit=16, w_bit=8).weight_quant(w_method='MinMax', mm_tensor=False, w_sym=True)
        assert compare_config_parameters(old_usage, new_usage) is False

    def test_new_weight_quant_config_should_equal_to_old_weight_quant_config_when_given_right_w4a16_param(self):
        old_usage = QuantConfig(a_bit=16, w_bit=4, mm_tensor=False)
        new_usage = QuantConfig(a_bit=16, w_bit=4).weight_quant(mm_tensor=False, w_sym=True)
        assert compare_config_parameters(old_usage, new_usage) is True

    def test_new_weight_quant_config_should_equal_to_old_weight_quant_config_when_given_right_group_size_param(self):
        old_usage = QuantConfig(a_bit=16, w_bit=4, mm_tensor=False, group_size=128)
        new_usage = QuantConfig(a_bit=16, w_bit=4).weight_quant(mm_tensor=False, w_sym=True, group_size=128)
        assert compare_config_parameters(old_usage, new_usage) is True

    @pytest.mark.skip()
    def test_new_sparse_quant_config_should_equal_to_old_weight_quant_config_when_given_sparse_param(self):
        old_usage = QuantConfig(
            a_bit=8,
            w_bit=4,
            disable_names=[],
            dev_type='cpu',
            act_method=3,
            fraction=0.011,
            nonuniform=False,
            mm_tensor=False,
            co_sparse=True,
        )
        new_usage = QuantConfig(a_bit=8, w_bit=4, disable_names=[], dev_type='cpu', mm_tensor=False).sparse_quant(
            act_method=3, fraction=0.011, nonuniform=False
        )
        assert compare_config_parameters(old_usage, new_usage) is True

    @pytest.mark.skip()
    def test_new_sparse_quant_config_should_not_equal_to_old_weight_quant_config_when_given_fault_sparse_param(self):
        old_usage = QuantConfig(
            a_bit=8,
            w_bit=4,
            disable_names=[],
            dev_type='cpu',
            act_method=3,
            fraction=0.011,
            nonuniform=False,
            mm_tensor=False,
            co_sparse=True,
        )
        new_usage = QuantConfig(a_bit=8, w_bit=4, disable_names=[], dev_type='cpu', mm_tensor=False).sparse_quant(
            act_method=1, fraction=0.012, nonuniform=False
        )
        assert compare_config_parameters(old_usage, new_usage) is False

    @pytest.mark.skip()
    def test_new_sparse_quant_config_should_equal_to_old_weight_quant_config_when_given_lowbit_param(self):
        old_usage = QuantConfig(
            a_bit=8,
            w_bit=4,
            disable_names=[],
            dev_type='cpu',
            act_method=2,
            do_smooth=True,
            use_sigma=True,
            sigma_factor=3.0,
            is_lowbit=True,
            mm_tensor=False,
        )
        new_usage = QuantConfig(a_bit=8, w_bit=4, disable_names=[], dev_type='cpu', mm_tensor=False).sparse_quant(
            is_lowbit=True, act_method=2, use_sigma=True, sigma_factor=3.0, do_smooth=True
        )
        assert compare_config_parameters(old_usage, new_usage) is True

    @pytest.mark.skip()
    def test_new_sparse_quant_config_should_not_equal_to_old_weight_quant_config_when_given_fault_lowbit_param(self):
        old_usage = QuantConfig(
            a_bit=8,
            w_bit=4,
            disable_names=[],
            dev_type='cpu',
            act_method=2,
            do_smooth=True,
            use_sigma=True,
            sigma_factor=3.0,
            is_lowbit=True,
            mm_tensor=False,
        )
        new_usage = QuantConfig(a_bit=8, w_bit=4, disable_names=[], dev_type='cpu', mm_tensor=False).sparse_quant(
            is_lowbit=False, act_method=2, use_sigma=True, sigma_factor=3.0, do_smooth=True
        )
        assert compare_config_parameters(old_usage, new_usage) is False

    def test_new_weight_activation_quant_config_should_equal_to_old_config_when_given_w8a8_param(self):
        old_usage = QuantConfig(
            a_bit=8, w_bit=8, disable_names=["test_linear"], dev_type='cpu', act_method=3, mm_tensor=False
        )
        new_usage = QuantConfig(
            a_bit=8, w_bit=8, disable_names=["test_linear"], dev_type='cpu', mm_tensor=False
        ).weight_activation_quant(act_method=3)
        assert compare_config_parameters(old_usage, new_usage) is True

    def test_new_weight_activation_quant_config_should_equal_to_old_config_when_given_per_token_w8a8_param(self):
        old_usage = QuantConfig(
            a_bit=8,
            w_bit=8,
            disable_names=["test_linear"],
            dev_type='cpu',
            act_method=3,
            mm_tensor=False,
            is_dynamic=True,
        )
        new_usage = QuantConfig(
            a_bit=8, w_bit=8, disable_names=["test_linear"], dev_type='cpu', mm_tensor=False
        ).weight_activation_quant(act_method=3, is_dynamic=True)
        assert compare_config_parameters(old_usage, new_usage) is True

    def test_new_weight_activation_quant_config_should_not_equal_to_old_config_when_given_fault_w8a8_param(self):
        old_usage = QuantConfig(
            a_bit=8, w_bit=8, disable_names=["test_linear"], dev_type='cpu', act_method=3, mm_tensor=False
        )
        new_usage = QuantConfig(
            a_bit=8, w_bit=8, disable_names=["test_linear"], dev_type='cpu', mm_tensor=False
        ).weight_activation_quant(act_method=1)
        assert compare_config_parameters(old_usage, new_usage) is False

    def test_new_weight_activation_quant_config_should_not_equal_to_old_config_when_given_fault_per_token_w8a8_param(
        self,
    ):
        old_usage = QuantConfig(
            a_bit=8,
            w_bit=8,
            disable_names=["test_linear"],
            dev_type='cpu',
            act_method=3,
            mm_tensor=False,
            is_dynamic=True,
        )
        new_usage = QuantConfig(
            a_bit=8, w_bit=8, disable_names=[""], dev_type='cpu', mm_tensor=False
        ).weight_activation_quant(act_method=3, is_dynamic=True)
        assert compare_config_parameters(old_usage, new_usage) is False

    def test_pdmix_flag_should_raise_error_when_given_non_w8a8_param(self):
        with pytest.raises(ValueError):
            QuantConfig(a_bit=8, w_bit=4, is_lowbit=True, pdmix=True)
