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
# pylint: disable=attribute-defined-outside-init,consider-using-from-import,useless-parent-delegation

import pytest
import torch
import torch.nn as nn
from unittest.mock import Mock, patch

from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.flat_quant.processors.flat_quant import (
    FakeQuantizerVisitor,
    FlatQuantQuantizerConfig,
    FlatQuantQuantizerMapVisitor,
    stat_input_hook,
    stat_tensor,
    get_n_set_parameters_byname,
    get_trainable_parameters,
    convert_config,
    quantize_model,
)
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.flat_quant.components.flat_linear import FakeQuantizedLinearConfig
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.llm_ptq_utils import QuantType


class TestStatFunctions:
    def test_stat_tensor(self):
        act_stats = {'test_layer': {}}
        x = torch.randn(2, 10)

        stat_tensor(act_stats, 'test_layer', x)

        assert 'input_max' in act_stats['test_layer']
        assert act_stats['test_layer']['input_max'].shape == (10,)

    def test_stat_tensor_update_max(self):
        act_stats = {'test_layer': {'input_max': torch.ones(10)}}
        x = torch.randn(2, 10) * 2

        stat_tensor(act_stats, 'test_layer', x)

        assert torch.any(act_stats['test_layer']['input_max'] > 1.0)

    def test_stat_input_hook_tuple_input(self):
        act_stats = {'test_layer': {}}
        x = (torch.randn(2, 10), torch.randn(2, 5))

        stat_input_hook(None, x, None, 'test_layer', act_stats)

        assert 'input_max' in act_stats['test_layer']
        assert act_stats['test_layer']['input_max'].shape == (10,)

    def test_stat_input_hook_non_tuple(self):
        act_stats = {'test_layer': {}}
        x = torch.randn(2, 10)
        stat_input_hook(None, x, None, 'test_layer', act_stats)
        assert 'input_max' in act_stats['test_layer']

    def test_stat_tensor_update_min(self):
        act_stats = {'test_layer': {'input_max': torch.ones(10) * 10}}
        x = torch.ones(2, 10)
        stat_tensor(act_stats, 'test_layer', x)
        assert torch.all(act_stats['test_layer']['input_max'] <= 10)


class TestFakeQuantizerVisitor:
    def setup_method(self):
        self.model = nn.Sequential(nn.Linear(128, 64), nn.Linear(64, 32))
        self.config = FakeQuantizedLinearConfig(w_bits=8, a_bits=8)
        self.visitor = FakeQuantizerVisitor(self.model, self.config)

    def test_initialization(self):
        assert self.visitor.model == self.model
        assert self.visitor.config == self.config
        assert len(self.visitor.quantizer_dict) == 0

    def test_register_forward_hook(self):
        linear = nn.Linear(128, 64)

        self.visitor.register_forward_hook(linear, 'test_linear')

        assert 'test_linear' in self.visitor.act_stats
        assert 'test_linear' in self.visitor.hooks

    def test_remove_forward_hook_with_prefix(self):
        linear1 = nn.Linear(128, 64)
        linear2 = nn.Linear(64, 32)
        self.visitor.register_forward_hook(linear1, 'layer1.linear')
        self.visitor.register_forward_hook(linear2, 'layer2.linear')

        self.visitor.remove_forward_hook(prefix="layer1")

        assert 'layer1.linear' in self.visitor.hooks
        assert 'layer2.linear' in self.visitor.hooks

    def test_mode_transitions(self):
        mock_quantizer = Mock()
        self.visitor.quantizer_dict['test_layer'] = mock_quantizer

        self.visitor.to_org_mode()
        mock_quantizer.to_org_mode.assert_called_once()

        self.visitor.to_calib_mode()
        mock_quantizer.to_calib_mode.assert_called_once()

        self.visitor.to_eval_mode(quant_weight=False)
        mock_quantizer.to_eval_mode.assert_called_once()
        mock_quantizer.fake_quant_weight.assert_not_called()

    def test_remove_forward_hook_no_prefix(self):
        model = nn.Sequential(nn.Linear(10, 10))
        config = FakeQuantizedLinearConfig(w_bits=8, a_bits=8)
        visitor = FakeQuantizerVisitor(model, config)
        linear = nn.Linear(10, 10)
        visitor.register_forward_hook(linear, 'foo')
        visitor.remove_forward_hook()  # no prefix, should remove all
        # hooks dict still has key, but hook is removed
        assert 'foo' in visitor.hooks

    def test_mode_transitions_with_prefix(self):
        model = nn.Sequential(nn.Linear(10, 10))
        config = FakeQuantizedLinearConfig(w_bits=8, a_bits=8)
        visitor = FakeQuantizerVisitor(model, config)
        mock_quantizer = Mock()
        visitor.quantizer_dict['foo'] = mock_quantizer
        visitor.quantizer_dict['bar'] = mock_quantizer
        visitor.to_org_mode(prefix='foo')
        visitor.to_calib_mode(prefix='foo')
        visitor.to_eval_mode(prefix='foo', quant_weight=True)
        visitor.fake_quant_weight(prefix='foo')
        assert mock_quantizer.to_org_mode.called
        assert mock_quantizer.to_calib_mode.called
        assert mock_quantizer.to_eval_mode.called
        assert mock_quantizer.fake_quant_weight.called

    def test_visit_linear_pair_basic(self):
        class DummyLinear(nn.Linear):
            def __init__(self):
                super().__init__(10, 10)

            def set_trans(self, **kwargs):
                self.trans_set = True

            def forward(self, x):
                return super().forward(x)

        model = nn.Sequential()
        model.foo = DummyLinear()
        config = FakeQuantizedLinearConfig(w_bits=8, a_bits=8)
        visitor = FakeQuantizerVisitor(model, config)
        pair = Mock()
        pair.target_modules = ['foo']
        visitor._visit_linear_pair(pair)
        assert 'foo' in visitor.quantizer_dict


class TestFlatQuantQuantizerMapVisitor:
    def setup_method(self):
        self.model = nn.Sequential(nn.LayerNorm(128), nn.Linear(128, 64))
        self.config = FlatQuantQuantizerConfig(w_bits=8, a_bits=8)
        self.visitor = FlatQuantQuantizerMapVisitor(self.model, self.config)

    def test_initialization(self):
        assert self.visitor.add_diag == self.config.add_diag
        assert self.visitor.diag_alpha == self.config.diag_alpha
        assert len(self.visitor.decompose_trans_dict) == 0
        assert len(self.visitor.norm_dict) == 0

    def test_mode_transitions_with_norm_and_trans(self):
        mock_norm = Mock()
        mock_trans = Mock()
        mock_pair = Mock()
        mock_pair.__str__ = Mock(return_value="test_pair")

        self.visitor.norm_dict['test_norm'] = mock_norm
        self.visitor.decompose_trans_dict[mock_pair] = mock_trans

        # Test org mode
        self.visitor.to_org_mode()
        mock_norm.to_org_mode.assert_called_once()

        # Test calib mode
        with patch.object(self.visitor, '_init_diag_scale'):
            self.visitor.to_calib_mode()
        mock_norm.to_calib_mode.assert_called_once()

        # Test eval mode
        with patch.object(self.visitor, '_reparameterize_act_diag_scale'):
            self.visitor.to_eval_mode()
        mock_norm.to_eval_mode.assert_called_once()
        mock_trans.to_eval_mode.assert_called_once()

    def test_flat_quant_quantizer_map_visitor_private_methods(self):
        # _init_diag_scale, _reparameterize_act_diag_scale, _visit_norm_linear_pair
        class DummyNorm(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.ones(4))

            def set_trans(self, **kwargs):
                self.trans_set = True

        class DummyLinear(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.ones(4, 4))

            def set_trans(self, **kwargs):
                self.trans_set = True

        model = nn.Sequential()
        model.norm = DummyNorm()
        model.linear = DummyLinear()
        config = FlatQuantQuantizerConfig(w_bits=8, a_bits=8)
        visitor = FlatQuantQuantizerMapVisitor(model, config)
        # _visit_norm_linear_pair
        pair = Mock()
        pair.source_modules = 'norm'
        pair.target_modules = ['linear']
        visitor._visit_norm_linear_pair(pair)
        assert 'norm' in visitor.norm_dict

        # _init_diag_scale
        class DummyDiagTrans:
            def __init__(self):
                self.diag_scale = nn.Parameter(torch.ones(4))

        class DummyTrans:
            def __init__(self):
                self.diag_trans = DummyDiagTrans()

        visitor.decompose_trans_dict = {pair: DummyTrans()}
        visitor.act_stats = {'linear': {'input_max': torch.ones(4)}}
        visitor._init_diag_scale()
        # _reparameterize_act_diag_scale
        trans = DummyTrans()
        trans.diag_trans.diag_scale = nn.Parameter(torch.ones(4))
        model.linear.weight = nn.Parameter(torch.ones(4, 4))
        pair.source_modules = 'linear'
        visitor.model = model
        visitor._reparameterize_act_diag_scale(trans, pair)

    def test_flat_quant_quantizer_map_visitor_norm_dict_decompose_trans_dict(self):
        model = nn.Sequential()
        model.add_module('foo', nn.Linear(10, 10))
        model.add_module('bar', nn.Linear(10, 10))
        config = FlatQuantQuantizerConfig(w_bits=8, a_bits=8)
        visitor = FlatQuantQuantizerMapVisitor(model, config)
        mock_norm = Mock()
        visitor.norm_dict['foo'] = mock_norm
        mock_pair = Mock()
        mock_pair.source_modules = 'foo'
        mock_pair.target_modules = ['bar']
        mock_pair.__str__ = lambda self=mock_pair: 'foo'

        # 用 DummyTrans 替换 mock_trans，保证 diag_trans.diag_scale 是 nn.Parameter
        class DummyDiagTrans:
            def __init__(self, size):
                self.diag_scale = nn.Parameter(torch.ones(size))

        class DummyTrans:
            def __init__(self, size):
                self.diag_trans = DummyDiagTrans(size)
                self.to_eval_mode = Mock()

        size = 10
        visitor.decompose_trans_dict[mock_pair] = DummyTrans(size)
        visitor.act_stats['bar'] = {'input_max': torch.ones(size)}

        visitor.to_org_mode()
        visitor.to_calib_mode()
        visitor.to_eval_mode()
        assert mock_norm.to_org_mode.called
        assert mock_norm.to_calib_mode.called
        assert mock_norm.to_eval_mode.called
        assert visitor.decompose_trans_dict[mock_pair].to_eval_mode.called


class TestUtilityFunctions:
    def test_get_n_set_parameters_byname(self):
        model = nn.Sequential(nn.Linear(128, 64), nn.BatchNorm1d(64))

        params = get_n_set_parameters_byname(model, ['weight'])

        assert len(params) == 2
        for param in params:
            assert param.requires_grad is True

    def test_convert_config_w4a4_dynamic(self):
        model = nn.Linear(128, 64)
        quant_config = Mock()
        quant_config.model_quant_type = QuantType.W4A4_FLATQUANT_DYNAMIC
        flat_quant_config = FlatQuantQuantizerConfig()

        config, visitor = convert_config(quant_config, flat_quant_config, model)

        assert config == flat_quant_config
        assert isinstance(visitor, FlatQuantQuantizerMapVisitor)

    def test_convert_config_w8a8_dynamic(self):
        model = nn.Linear(128, 64)
        quant_config = Mock()
        quant_config.model_quant_type = QuantType.W8A8_DYNAMIC
        quant_config.w_bit = 8
        quant_config.a_bit = 8
        quant_config.w_sym = True
        quant_config.a_sym = True
        quant_config.is_dynamic = True
        flat_quant_config = FlatQuantQuantizerConfig()

        config, visitor = convert_config(quant_config, flat_quant_config, model)

        assert isinstance(config, FlatQuantQuantizerConfig)
        assert isinstance(visitor, FakeQuantizerVisitor)

    def test_convert_config_float(self):
        model = nn.Linear(128, 64)
        quant_config = Mock()
        quant_config.model_quant_type = QuantType.FLOAT
        flat_quant_config = FlatQuantQuantizerConfig()

        config, visitor = convert_config(quant_config, flat_quant_config, model)

        assert config is None
        assert visitor is None

    def test_convert_config_invalid_type(self):
        model = nn.Linear(128, 64)
        quant_config = Mock()
        quant_config.model_quant_type = "INVALID_TYPE"
        flat_quant_config = FlatQuantQuantizerConfig()

        with pytest.raises(ValueError):
            convert_config(quant_config, flat_quant_config, model)

    def test_quantize_model(self):
        model = nn.Sequential(nn.Linear(128, 64))
        model_bridge = Mock()

        mock_pair = Mock()
        mock_pair.contain.return_value = True
        mock_pair.name = "test_pair"
        mock_pair.accept = Mock()

        pairs_dict = {
            'AttnNormLinearPair': [mock_pair],
            'AttnLinearLinearPair': [],
            'MLPNormLinearPair': [],
            'MLPLinearLinearPair': [],
        }
        model_bridge.get_structure_pairs.return_value = pairs_dict
        model_bridge.model = model

        layer_map = {'test_layer': Mock()}
        layer_map['test_layer'].model_quant_type = QuantType.W4A4_FLATQUANT_DYNAMIC
        flat_quant_config = FlatQuantQuantizerConfig()

        with patch(
            'msmodelslim.pytorch.llm_ptq.llm_ptq_tools.flat_quant.processors.flat_quant.QuantizerMapper'
        ) as mock_mapper_class:
            mock_mapper = Mock()
            mock_mapper_class.return_value = mock_mapper

            result = quantize_model(model_bridge, layer_map, flat_quant_config)

            mock_mapper.register_pattern.assert_called()
            mock_mapper.apply_quantizer.assert_called_once_with(pairs_dict)
            assert result == mock_mapper

    def test_quantize_model_mixed_quant_type_error(self):
        model = nn.Sequential(nn.Linear(128, 64))  # noqa: F841
        model_bridge = Mock()

        mock_pair = Mock()
        mock_pair.contain.return_value = True
        mock_pair.name = "test_pair"

        pairs_dict = {
            'AttnNormLinearPair': [mock_pair],
            'AttnLinearLinearPair': [],
            'MLPNormLinearPair': [],
            'MLPLinearLinearPair': [],
        }
        model_bridge.get_structure_pairs.return_value = pairs_dict

        quant_config1 = Mock()
        quant_config1.model_quant_type = QuantType.W4A4_FLATQUANT_DYNAMIC
        quant_config2 = Mock()
        quant_config2.model_quant_type = QuantType.W8A8_DYNAMIC

        layer_map = {'layer1': quant_config1, 'layer2': quant_config2}

        flat_quant_config = FlatQuantQuantizerConfig()

        with pytest.raises(ValueError, match="Find different quant type"):
            quantize_model(model_bridge, layer_map, flat_quant_config)

    def test_get_trainable_parameters_all_empty(self):
        class Dummy(nn.Module):
            def named_parameters(self):
                return []

        model = Dummy()
        params, trainable_params, need_train = get_trainable_parameters(model)
        assert need_train is False
        assert isinstance(params, dict)
        assert isinstance(trainable_params, list)

    def test_convert_config_w4a4_flatquant_dynamic(self):
        model = nn.Linear(10, 10)
        quant_config = Mock()
        quant_config.model_quant_type = QuantType.W4A4_FLATQUANT_DYNAMIC
        flat_quant_config = FlatQuantQuantizerConfig()
        config, visitor = convert_config(quant_config, flat_quant_config, model)
        assert config == flat_quant_config
        assert isinstance(visitor, FlatQuantQuantizerMapVisitor)

    def test_convert_config_w8a8_per_tensor(self):
        model = nn.Linear(10, 10)
        quant_config = Mock()
        quant_config.model_quant_type = QuantType.W8A8
        quant_config.w_bit = 8
        quant_config.a_bit = 8
        quant_config.w_sym = True
        quant_config.a_sym = True
        quant_config.is_dynamic = False
        flat_quant_config = FlatQuantQuantizerConfig()
        config, visitor = convert_config(quant_config, flat_quant_config, model)
        assert isinstance(config, FlatQuantQuantizerConfig)
        assert isinstance(visitor, FakeQuantizerVisitor)
        assert config.a_groupsize == -1
        assert config.a_per_tensor is True

    def test_quantize_model_empty_pairs(self):
        model_bridge = Mock()
        model_bridge.get_structure_pairs.return_value = {
            'AttnNormLinearPair': [],
            'AttnLinearLinearPair': [],
            'MLPNormLinearPair': [],
            'MLPLinearLinearPair': [],
        }
        layer_map = {}
        flat_quant_config = FlatQuantQuantizerConfig()
        with patch(
            'msmodelslim.pytorch.llm_ptq.llm_ptq_tools.flat_quant.processors.flat_quant.QuantizerMapper'
        ) as mock_mapper_class:
            mock_mapper = Mock()
            mock_mapper_class.return_value = mock_mapper
            result = quantize_model(model_bridge, layer_map, flat_quant_config)
            mock_mapper.apply_quantizer.assert_called_once()
            assert result == mock_mapper

    def test_quantize_model_quant_visitor_none(self):
        model_bridge = Mock()
        mock_pair = Mock()
        mock_pair.contain.return_value = True
        mock_pair.name = "test_pair"
        mock_pair.accept = Mock()
        pairs_dict = {
            'AttnNormLinearPair': [mock_pair],
            'AttnLinearLinearPair': [],
            'MLPNormLinearPair': [],
            'MLPLinearLinearPair': [],
        }
        model_bridge.get_structure_pairs.return_value = pairs_dict
        model_bridge.model = Mock()
        layer_map = {'test_layer': Mock()}
        layer_map['test_layer'].model_quant_type = QuantType.FLOAT
        flat_quant_config = FlatQuantQuantizerConfig()
        with patch(
            'msmodelslim.pytorch.llm_ptq.llm_ptq_tools.flat_quant.processors.flat_quant.QuantizerMapper'
        ) as mock_mapper_class:
            mock_mapper = Mock()
            mock_mapper_class.return_value = mock_mapper
            result = quantize_model(model_bridge, layer_map, flat_quant_config)
            mock_mapper.apply_quantizer.assert_called_once()
            assert result == mock_mapper
