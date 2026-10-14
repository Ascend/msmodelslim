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
# pylint: disable=attribute-defined-outside-init,consider-using-from-import

import sys
import types
from collections import OrderedDict
import unittest
from unittest.mock import Mock, MagicMock, patch
from unittest.mock import PropertyMock  # noqa: F401
import torch
import torch.nn as nn


class ColumnParallelLinear(nn.Module):
    """Mock ColumnParallelLinear for testing"""

    pass


class RowParallelLinear(nn.Module):
    """Mock RowParallelLinear for testing"""

    pass


class MixedFusedLayerNorm(nn.Module):
    """Mock MixedFusedLayerNorm for testing"""

    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))


# Create mock modules with proper class types
def reduce_from_tensor_model_parallel_region(x):
    """Mock function for reduce_from_tensor_model_parallel_region"""
    return x


def mock_tqdm_side_effect(x):
    """Mock function for tqdm side_effect - return the iterable as-is"""
    return x


# Mock modules must be set up BEFORE importing msmodelslim modules
# that depend on megatron and mindspeed
for module_name in [
    'megatron',
    'megatron.inference',
    'megatron.inference.text_generation',
    'megatron.core',
    'megatron.core.tensor_parallel',
    'megatron.legacy',
    'megatron.legacy.model',
    'megatron.legacy.model.fused_layer_norm',
    'megatron.core.tensor_parallel.mappings',
    'megatron.training',
]:
    sys.modules[module_name] = MagicMock()

sys.modules['mindspeed'] = MagicMock()
sys.modules['mindspeed.patch_utils'] = MagicMock()

# Mock torch_npu to avoid ImportError when torch_npu is not available
torch_npu_spec = types.ModuleType(name='torch_npu')
mock_torch_npu = Mock(__spec__=torch_npu_spec, __version__='2.1.0', npu_init=Mock(return_value=None))
sys.modules['torch_npu'] = mock_torch_npu

# Mock torch.npu to avoid AttributeError when torch_npu is not available
# torch_npu adds npu attribute to torch module when imported
mock_torch_npu_module = Mock()
mock_torch_npu_module.is_available = Mock(return_value=False)
mock_torch_npu_module.current_device = Mock(return_value=0)
mock_torch_npu_module.get_device_name = Mock(return_value='Ascend910')
mock_torch_npu_module.set_compile_mode = Mock(return_value=None)
mock_torch_npu_module.set_option = Mock(return_value=None)
torch.npu = mock_torch_npu_module

# Mock torch.device to handle npu device strings
# PyTorch doesn't recognize "npu" as a valid device type without torch_npu
original_device = torch.device


def mock_device(device):
    """Mock device function to handle npu device strings"""
    if isinstance(device, str) and device.startswith('npu'):
        # Return a mock device object for npu
        mock_dev = Mock()
        mock_dev.type = 'npu'
        if ':' in device:
            mock_dev.index = int(device.split(':')[1])
        else:
            mock_dev.index = None
        mock_dev.__str__ = Mock(return_value=device)
        mock_dev.__repr__ = Mock(return_value=f"device(type='npu', index={mock_dev.index})")
        return mock_dev
    # For other devices, use the original function
    return original_device(device)


torch.device = mock_device

# Configure mock modules with proper class types BEFORE importing
mock_megatron_core_tp = MagicMock()
mock_megatron_core_tp.ColumnParallelLinear = ColumnParallelLinear
mock_megatron_core_tp.RowParallelLinear = RowParallelLinear
mock_megatron_core_tp.mappings = MagicMock()
mock_megatron_core_tp.mappings.reduce_from_tensor_model_parallel_region = reduce_from_tensor_model_parallel_region

mock_megatron_fused_ln = MagicMock()
mock_megatron_fused_ln.MixedFusedLayerNorm = MixedFusedLayerNorm

sys.modules['megatron.inference.text_generation'].generate = MagicMock()
sys.modules['megatron.core.tensor_parallel'] = mock_megatron_core_tp
sys.modules['megatron.core.tensor_parallel.mappings'] = mock_megatron_core_tp.mappings
sys.modules['megatron.legacy.model.fused_layer_norm'] = mock_megatron_fused_ln
sys.modules['megatron.training'].get_args = MagicMock(return_value=MagicMock(max_new_tokens=1))

# Now import msmodelslim modules after all mocks are set up
from msmodelslim.pytorch.llm_ptq.anti_outlier import AntiOutlierConfig  # noqa: E402
from msmodelslim.pytorch.mindspeed_adapter.modelslim_adapter import (  # noqa: E402
    ModelAdapter,
    CalibratorAdapter,
    MegatronLinearAdapter,
    Linear,
    GenerateForward,
    ModelGenerateForward,
    set_module,
    get_norm_linear_subgraph,
)
from msmodelslim.pytorch.mindspeed_adapter.anti_outlier_adapter import (  # noqa: E402
    AntiOutlierAdapter,
)


class MockConfig:
    def __init__(self, torch_dtype=torch.float16, params_dtype=torch.float16, num_attention_heads=None):
        self.torch_dtype = torch_dtype
        self.params_dtype = params_dtype
        self.num_attention_heads = num_attention_heads


class MockModelWithArgs(nn.Module):
    def __init__(self):
        super().__init__()
        self.args = MockConfig()
        self.linear = nn.Linear(10, 20)

    def forward(self, x):
        return self.linear(x)


class MockModelWithConfig(nn.Module):
    def __init__(self, num_attention_heads=None):
        super().__init__()
        self.config = MockConfig(num_attention_heads=num_attention_heads)
        self.norm = nn.LayerNorm(10)  # Add norm module for init_dag
        self.linear = nn.Linear(10, 20)

    def forward(self, x):
        x = self.norm(x)
        return self.linear(x)

    def generate(self, x, max_new_tokens=1):
        return torch.randn(1, 10)


class MockModelNoConfig(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 20)


class TestModelAdapter(unittest.TestCase):
    def test_state_dict(self):
        model = MockModelWithArgs()
        adapter = ModelAdapter(model, dev_type='npu', prefix='model.')
        state = adapter.state_dict()
        self.assertIsInstance(state, OrderedDict)

    def test_state_dict_with_prefix(self):
        model = MockModelWithArgs()
        adapter = ModelAdapter(model, dev_type='npu', prefix='model.')
        state = adapter.state_dict(prefix='custom.')
        self.assertIsInstance(state, OrderedDict)

    @patch('msmodelslim.pytorch.mindspeed_adapter.modelslim_adapter.generate')
    def test_forward_with_generate_forward(self, mock_generate):
        model = MockModelWithArgs()
        adapter = ModelAdapter(model, dev_type='npu', prefix='model.')
        mock_generate.return_value = torch.randn(1, 10)
        x = [torch.randn(1, 10)]
        result = adapter.forward(x)  # noqa: F841
        mock_generate.assert_called_once()

    def test_forward_with_model_generate_forward(self):
        model = MockModelWithConfig()
        adapter = ModelAdapter(model, dev_type='gpu', prefix='model.')
        x = torch.randn(1, 10)
        with patch('msmodelslim.pytorch.mindspeed_adapter.modelslim_adapter.get_args') as mock_get_args:
            mock_args = Mock()
            mock_args.max_new_tokens = 10
            mock_get_args.return_value = mock_args
            result = adapter.forward(x)
            self.assertIsNotNone(result)


class TestGenerateForward(unittest.TestCase):
    @patch('msmodelslim.pytorch.mindspeed_adapter.modelslim_adapter.generate')
    def test_call_with_list_input(self, mock_generate):
        forward = GenerateForward()
        model = Mock()
        x = [[torch.randn(1, 10)]]
        mock_generate.return_value = torch.randn(1, 10)
        result = forward(model, x)  # noqa: F841
        mock_generate.assert_called_once_with(model, x[0], tokens_to_generate=1)

    @patch('msmodelslim.pytorch.mindspeed_adapter.modelslim_adapter.generate')
    def test_call_with_tensor_input(self, mock_generate):
        forward = GenerateForward()
        model = Mock()
        x = torch.randn(1, 10)
        mock_generate.return_value = torch.randn(1, 10)
        result = forward(model, x)  # noqa: F841
        mock_generate.assert_called_once_with(model, x, tokens_to_generate=1)


class TestModelGenerateForward(unittest.TestCase):
    def test_call(self):
        forward = ModelGenerateForward()
        model = MockModelWithConfig()
        x = torch.randn(1, 10)
        with patch('msmodelslim.pytorch.mindspeed_adapter.modelslim_adapter.get_args') as mock_get_args:
            mock_args = Mock()
            mock_args.max_new_tokens = 10
            mock_get_args.return_value = mock_args
            result = forward(model, x)
            self.assertIsNotNone(result)
            self.assertEqual(mock_args.max_new_tokens, 10)  # Should be restored


class TestSetModule(unittest.TestCase):
    def test_set_module(self):
        model = nn.Sequential(nn.Linear(10, 20), nn.Linear(20, 30))
        new_module = nn.Linear(20, 40)
        set_module(model, '1', new_module)
        self.assertIsInstance(model[1], nn.Linear)
        self.assertEqual(model[1].out_features, 40)


class TestLinear(unittest.TestCase):
    def test_init(self):
        mock_linear = Mock()
        mock_linear.input_size = 10
        mock_linear.output_size = 20
        mock_linear.weight = torch.randn(20, 10)
        mock_linear.bias = None
        linear = Linear(mock_linear)
        self.assertEqual(linear.in_features, 10)
        self.assertEqual(linear.out_features, 20)

    def test_weight_property(self):
        mock_linear = Mock()
        mock_linear.input_size = 10
        mock_linear.output_size = 20
        mock_linear.weight = torch.randn(20, 10)
        mock_linear.bias = None
        linear = Linear(mock_linear)
        self.assertIsNotNone(linear.weight)

    def test_bias_property(self):
        mock_linear = Mock()
        mock_linear.input_size = 10
        mock_linear.output_size = 20
        mock_linear.weight = torch.randn(20, 10)
        mock_linear.bias = torch.randn(20)
        linear = Linear(mock_linear)
        self.assertIsNotNone(linear.bias)

    def test_forward(self):
        mock_linear = Mock()
        mock_linear.input_size = 10
        mock_linear.output_size = 20
        mock_linear.weight = torch.randn(20, 10)
        mock_linear.bias = None
        mock_linear.return_value = (torch.randn(1, 20), None)
        linear = Linear(mock_linear)
        x = torch.randn(1, 10)
        result = linear(x)  # noqa: F841
        mock_linear.assert_called_once_with(x)

    def test_state_dict(self):
        mock_linear = Mock()
        mock_linear.input_size = 10
        mock_linear.output_size = 20
        mock_linear.weight = torch.randn(20, 10)
        mock_linear.bias = None
        linear = Linear(mock_linear)
        state = linear.state_dict()
        self.assertIn('weight', state)

    def test_state_dict_with_bias(self):
        mock_linear = Mock()
        mock_linear.input_size = 10
        mock_linear.output_size = 20
        mock_linear.weight = torch.randn(20, 10)
        mock_linear.bias = torch.randn(20)
        linear = Linear(mock_linear)
        state = linear.state_dict()
        self.assertIn('weight', state)
        self.assertIn('bias', state)

    def test_load_state_dict(self):
        mock_linear = Mock()
        mock_linear.input_size = 10
        mock_linear.output_size = 20
        mock_linear.weight = torch.randn(20, 10)
        mock_linear.bias = None
        linear = Linear(mock_linear)
        state_dict = {'weight': torch.randn(20, 10)}
        linear.load_state_dict(state_dict)
        mock_linear.load_state_dict.assert_called_once()

    def test_get_linear(self):
        mock_linear = Mock()
        mock_linear.input_size = 10
        mock_linear.output_size = 20
        mock_linear.weight = torch.randn(20, 10)
        mock_linear.bias = None
        linear = Linear(mock_linear)
        result = linear.get_linear()
        self.assertEqual(result, mock_linear)


class TestMegatronLinearAdapter(unittest.TestCase):
    def test_init(self):
        mock_linear = Mock()
        mock_linear.input_size = 10
        mock_linear.output_size = 20
        mock_linear.weight = torch.randn(20, 10)
        mock_linear.bias = None
        mock_linear.skip_bias_add = True
        adapter = MegatronLinearAdapter(mock_linear)
        self.assertTrue(adapter.skip_bias_add)

    def test_init_with_quant_params(self):
        mock_linear = Mock()
        mock_linear.input_size = 10
        mock_linear.output_size = 20
        mock_linear.weight = torch.randn(20, 10)
        mock_linear.bias = None
        mock_linear.skip_bias_add = True
        mock_linear.quant_params = {'scale': 1.0}
        adapter = MegatronLinearAdapter(mock_linear)
        self.assertIsNotNone(adapter.quant_params)

    def test_weight_property(self):
        mock_linear = Mock()
        mock_linear.input_size = 10
        mock_linear.output_size = 20
        mock_linear.weight = torch.randn(20, 10)
        mock_linear.bias = None
        mock_linear.skip_bias_add = True
        adapter = MegatronLinearAdapter(mock_linear)
        self.assertIsNotNone(adapter.weight)

    def test_bias_property(self):
        mock_linear = Mock()
        mock_linear.input_size = 10
        mock_linear.output_size = 20
        mock_linear.weight = torch.randn(20, 10)
        mock_linear.bias = torch.randn(20)
        mock_linear.skip_bias_add = True
        adapter = MegatronLinearAdapter(mock_linear)
        self.assertIsNotNone(adapter.bias)

    def test_get_linear(self):
        mock_linear = Mock()
        mock_linear.input_size = 10
        mock_linear.output_size = 20
        mock_linear.weight = torch.randn(20, 10)
        mock_linear.bias = None
        mock_linear.skip_bias_add = True
        adapter = MegatronLinearAdapter(mock_linear)
        result = adapter.get_linear()
        self.assertIsNotNone(result)

    @patch('msmodelslim.pytorch.mindspeed_adapter.modelslim_adapter.reduce_from_tensor_model_parallel_region')
    def test_forward_row_parallel(self, mock_reduce):
        # Mock RowParallelLinear without importing
        mock_linear = Mock()
        mock_linear.input_size = 10
        mock_linear.output_size = 20
        mock_linear.weight = torch.randn(20, 10)
        mock_linear.bias = None
        mock_linear.skip_bias_add = True
        mock_linear.return_value = (torch.randn(1, 20), None)
        mock_reduce.return_value = torch.randn(1, 20)
        adapter = MegatronLinearAdapter(mock_linear)
        adapter.is_row = True  # Set directly to test the row parallel path
        # Mock get_linear to return non-RowParallelLinear
        adapter.get_linear = Mock(return_value=Mock())
        x = torch.randn(1, 10)
        result = adapter.forward(x)
        self.assertEqual(len(result), 2)
        mock_reduce.assert_called_once()


class TestCalibratorAdapter(unittest.TestCase):
    @patch('msmodelslim.pytorch.mindspeed_adapter.modelslim_adapter.TorchDAGAdapter')
    def test_extract_dag_with_calib_data(self, mock_dag):
        model = MockModelWithConfig()
        calib_data = [torch.randint(0, 100, (1, 128)).type(torch.int64)]
        # Use actual Calibrator class - __new__() bypasses __init__
        adapter = CalibratorAdapter.__new__(CalibratorAdapter)
        adapter.calib_data = calib_data
        adapter.norm_class_name = 'LayerNorm'
        adapter.logger = Mock()
        with patch.object(adapter, 'init_dag', return_value=True):
            with patch.object(adapter, 'get_norm_class', return_value=[nn.LayerNorm]):
                mock_dag_instance = Mock()
                mock_dag.return_value = mock_dag_instance
                result = adapter.extract_dag(model)
                self.assertIsNotNone(result)

    @patch('msmodelslim.pytorch.mindspeed_adapter.modelslim_adapter.TorchDAGAdapter')
    def test_extract_dag_without_calib_data(self, mock_dag):
        model = MockModelWithConfig()
        # Use actual Calibrator class - __new__() bypasses __init__
        adapter = CalibratorAdapter.__new__(CalibratorAdapter)
        adapter.calib_data = None
        adapter.norm_class_name = 'LayerNorm'
        adapter.logger = Mock()
        with patch.object(adapter, 'init_dag', return_value=True):
            with patch.object(adapter, 'get_norm_class', return_value=[nn.LayerNorm]):
                mock_dag_instance = Mock()
                mock_dag.return_value = mock_dag_instance
                result = adapter.extract_dag(model)
                self.assertIsNotNone(result)

    def test_get_ori_model_weight(self):
        model = nn.Sequential(nn.Linear(10, 20), nn.LayerNorm(20))
        # Use actual Calibrator class - __new__() bypasses __init__
        adapter = CalibratorAdapter.__new__(CalibratorAdapter)
        adapter.logger = Mock()
        result = adapter.get_ori_model_weight(model, Mock())
        self.assertIsInstance(result, dict)
        self.assertGreater(len(result), 0)


class TestGetNormLinearSubgraph(unittest.TestCase):
    def test_get_norm_linear_subgraph(self):
        mock_self = Mock()
        mock_node1 = Mock()
        mock_node1.name_in_network = 'norm1'
        mock_node1.op_type = 'LayerNorm'
        mock_node2 = Mock()
        mock_node2.name_in_network = 'linear1'
        mock_node2.op_type = 'Linear'
        mock_node3 = Mock()
        mock_node3.name_in_network = 'norm2'
        mock_node3.op_type = 'LayerNorm'
        mock_self.node_list = [mock_node1, mock_node2, mock_node3]
        mock_self.norm_nodes = ['layernorm']
        result = get_norm_linear_subgraph(mock_self)
        self.assertIsInstance(result, dict)


class TestAntiOutlierAdapter(unittest.TestCase):
    def setUp(self):
        self.model = MockModelWithConfig()
        self.calib_data = [[torch.randn(1, 10)]]
        self.cfg = AntiOutlierConfig(anti_method='m3', dev_type='npu')

    def test_init_invalid_anti_method(self):
        cfg = AntiOutlierConfig(anti_method='m1', dev_type='npu')
        with self.assertRaises(ValueError) as context:
            AntiOutlierAdapter(self.model, self.calib_data, cfg)
        self.assertIn("m3 and m5", str(context.exception))

    def test_init_cpu_device(self):
        cfg = AntiOutlierConfig(anti_method='m3', dev_type='cpu')
        with self.assertRaises(ValueError) as context:
            AntiOutlierAdapter(self.model, self.calib_data, cfg)
        self.assertIn("CPU", str(context.exception))

    def test_init_invalid_norm_class_name(self):
        with self.assertRaises(TypeError) as context:
            AntiOutlierAdapter(self.model, self.calib_data, self.cfg, norm_class_name=123)
        self.assertIn("str", str(context.exception))

    def test_init_invalid_model(self):
        with self.assertRaises(TypeError) as context:
            AntiOutlierAdapter("not_a_model", self.calib_data, self.cfg)
        self.assertIn("nn.Module", str(context.exception))

    def test_init_empty_calib_data(self):
        with self.assertRaises(ValueError) as context:
            AntiOutlierAdapter(self.model, [], self.cfg)
        self.assertIn("empty", str(context.exception))

    @patch('msmodelslim.pytorch.mindspeed_adapter.anti_outlier_adapter.extract_dag')
    def test_init_with_context_embedder(self, mock_extract_dag):
        model = MockModelWithConfig()
        model.state_dict = Mock(return_value={'context_embedder.weight': torch.randn(10, 20)})
        mock_dag = Mock()
        mock_dag.get_norm_linear_subgraph.return_value = {'norm1': ['linear1']}
        mock_extract_dag.return_value = mock_dag
        adapter = AntiOutlierAdapter(model, self.calib_data, self.cfg)
        self.assertTrue(adapter.is_context_embedder_model)

    def test_trans_to_dict(self):
        with patch('msmodelslim.pytorch.mindspeed_adapter.anti_outlier_adapter.extract_dag') as mock_extract_dag:
            mock_dag = Mock()
            mock_dag.get_norm_linear_subgraph.return_value = {'norm1': ['linear1']}
            mock_extract_dag.return_value = mock_dag
            adapter = AntiOutlierAdapter(self.model, self.calib_data, self.cfg)
            data = [torch.randn(1, 10)]
            result = adapter.trans_to_dict(data)
            self.assertIn('x', result)
            self.assertTrue(torch.equal(result['x'], data[0]))

    @patch('msmodelslim.pytorch.mindspeed_adapter.anti_outlier_adapter.extract_dag')
    def test_stat_tensor(self, mock_extract_dag):
        mock_dag = Mock()
        mock_dag.get_norm_linear_subgraph.return_value = {'norm1': ['linear1']}
        mock_extract_dag.return_value = mock_dag
        adapter = AntiOutlierAdapter(self.model, self.calib_data, self.cfg)
        act_stats = {}
        tensor = torch.randn(2, 10)
        adapter.stat_tensor(act_stats, 'test', tensor)
        self.assertIn('test', act_stats)
        test_stats = act_stats.get('test', {})
        self.assertIn('max', test_stats)
        self.assertIn('min', test_stats)

    @patch('msmodelslim.pytorch.mindspeed_adapter.anti_outlier_adapter.extract_dag')
    def test_stat_tensor_with_ch_align(self, mock_extract_dag):
        mock_dag = Mock()
        mock_dag.get_norm_linear_subgraph.return_value = {'norm1': ['linear1']}
        mock_extract_dag.return_value = mock_dag
        cfg = AntiOutlierConfig(anti_method='m3', dev_type='npu')
        cfg.ch_align = True
        adapter = AntiOutlierAdapter(self.model, self.calib_data, cfg)
        act_stats = {}
        tensor = torch.randn(2, 10)
        adapter.stat_tensor(act_stats, 'test', tensor)
        self.assertIn('test', act_stats)
        test_stats = act_stats.get('test', {})
        self.assertIn('shift', test_stats)

    @patch('msmodelslim.pytorch.mindspeed_adapter.anti_outlier_adapter.extract_dag')
    @patch('msmodelslim.pytorch.mindspeed_adapter.anti_outlier_adapter.tqdm')
    def test_os_stats(self, mock_tqdm, mock_extract_dag):
        mock_dag = Mock()
        mock_dag.get_norm_linear_subgraph.return_value = {'norm1': ['linear1']}
        mock_extract_dag.return_value = mock_dag
        mock_tqdm.return_value = range(len(self.calib_data))
        adapter = AntiOutlierAdapter(self.model, self.calib_data, self.cfg)
        act_stats = adapter.os_stats()
        self.assertIsInstance(act_stats, dict)

    @patch('msmodelslim.pytorch.mindspeed_adapter.anti_outlier_adapter.extract_dag')
    def test_get_num_attention_heads(self, mock_extract_dag):
        mock_dag = Mock()
        mock_dag.get_norm_linear_subgraph.return_value = {'norm1': ['linear1']}
        mock_extract_dag.return_value = mock_dag
        model = MockModelWithConfig(num_attention_heads=8)
        adapter = AntiOutlierAdapter(model, self.calib_data, self.cfg)
        num_heads = adapter.get_num_attention_heads()
        self.assertEqual(num_heads, 8)

    @patch('msmodelslim.pytorch.mindspeed_adapter.anti_outlier_adapter.extract_dag')
    def test_check_calib_data(self, mock_extract_dag):
        mock_dag = Mock()
        mock_dag.get_norm_linear_subgraph.return_value = {'norm1': ['linear1']}
        mock_extract_dag.return_value = mock_dag
        adapter = AntiOutlierAdapter(self.model, self.calib_data, self.cfg)
        # Test with valid data
        valid_data = [[torch.randn(1, 10)]]
        result = adapter.check_calib_data(valid_data)
        self.assertEqual(result, valid_data)

    @patch('msmodelslim.pytorch.mindspeed_adapter.anti_outlier_adapter.extract_dag')
    def test_check_multimodel(self, mock_extract_dag):
        mock_dag = Mock()
        mock_dag.get_norm_linear_subgraph.return_value = {'norm1': ['linear1']}
        mock_extract_dag.return_value = mock_dag
        adapter = AntiOutlierAdapter(self.model, self.calib_data, self.cfg)
        result = adapter.check_multimodel(self.model)
        self.assertFalse(result)


if __name__ == '__main__':
    unittest.main()
