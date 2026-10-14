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
# pylint: disable=attribute-defined-outside-init,consider-using-from-import,no-member,redefined-outer-name,reimported,unnecessary-lambda

import pytest
import torch
import torch.nn as nn
from unittest.mock import Mock, patch, MagicMock
from dataclasses import dataclass  # noqa: F401
from typing import List, Dict, Any  # noqa: F401
import types
from contextlib import nullcontext

from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.flat_quant.trainer import (
    TrainingConfig,
    ModelAdapter,
    DataPreparer,
    LayerTrainer,
    LayerTrainingData,
    FlatQuantTrainer,
    empty_cache,
    get_device_str,
    convert_outputs_to_inputs,
    flat_quant_train,
)
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.flat_quant.config import FlatQuantConfig  # noqa: F401


class MockArgs:
    """Mock arguments class for testing."""

    def __init__(self):
        self.seed = 0
        self.epochs = 2
        self.flat_lr = 5e-3
        self.warmup = False
        self.deactive_amp = True
        self.amp_dtype = "bfloat16"
        self.quant_by_quant = False
        self.w_bits = 4
        self.a_bits = 4
        self.w_asym = False
        self.a_asym = False
        self.lwc = True
        self.lac = True
        self.a_groupsize = -1
        self.add_diag = True
        self.diag_alpha = 0.3
        self.diag_relu = True
        self.direct_inv = False


class MockModel(nn.Module):
    """Mock model for testing."""

    def __init__(self):
        super().__init__()
        self.device = torch.device("cpu")
        self.hf_device_map = None
        self.layers = nn.ModuleList([nn.Linear(10, 10), nn.Linear(10, 10)])

    def forward(self, x):
        return x

    def tie_weights(self):
        pass


class MockModelBridge:
    """Mock model bridge for testing."""

    def __init__(self):
        self.layers_name = "layers"

    def analyze_structure(self):
        pass

    def get_layers(self):
        return self.layers_name

    def get_layer_by_index(self, index):
        return f"layers.{index}"


class MockQuantizer:
    """Mock quantizer for testing."""

    def to_eval_mode(self):
        pass

    def to_org_mode(self):
        pass

    def to_calib_mode(self, prefix):
        pass


class TestTrainingConfig:
    def test_init_with_deactive_amp_true(self):
        args = MockArgs()
        args.deactive_amp = True
        config = TrainingConfig(args)

        assert config.seed == 0
        assert config.epochs == 2
        assert config.flat_lr == 5e-3
        assert config.warmup is False
        assert config.deactive_amp is True
        assert config.dtype == torch.float32
        assert config.traincast.__name__ == "nullcontext"

    def test_init_with_deactive_amp_false_bfloat16(self):
        args = MockArgs()
        args.deactive_amp = False
        args.amp_dtype = "bfloat16"
        config = TrainingConfig(args)

        assert config.deactive_amp is False
        assert config.dtype == torch.bfloat16

    def test_init_with_deactive_amp_false_float16(self):
        args = MockArgs()
        args.deactive_amp = False
        args.amp_dtype = "float16"
        config = TrainingConfig(args)

        assert config.deactive_amp is False
        assert config.dtype == torch.float16

    def test_init_with_invalid_amp_dtype(self):
        args = MockArgs()
        args.deactive_amp = False
        args.amp_dtype = "invalid_dtype"

        with pytest.raises(ValueError, match="Invalid AMP dtype: invalid_dtype"):
            TrainingConfig(args)


class TestModelAdapter:
    @pytest.fixture
    def mock_model(self):
        return MockModel()

    @pytest.fixture
    def mock_args(self):
        return MockArgs()

    @patch('msmodelslim.pytorch.llm_ptq.llm_ptq_tools.flat_quant.trainer.get_model_bridge')
    @patch('msmodelslim.pytorch.llm_ptq.llm_ptq_tools.flat_quant.trainer.get_module_by_name')
    def test_init_should_setup_model_adapter_correctly(self, mock_get_module, mock_get_bridge, mock_model, mock_args):
        mock_bridge = MockModelBridge()
        mock_get_bridge.return_value = mock_bridge
        mock_get_module.return_value = [nn.Linear(10, 10), nn.Linear(10, 10)]

        adapter = ModelAdapter(mock_model, mock_args)

        assert adapter.model == mock_model
        assert adapter.args == mock_args
        assert adapter.default_device == mock_model.device
        assert adapter.hf_device_map is None
        mock_get_bridge.assert_called_once_with(mock_model)
        mock_bridge.analyze_structure()

    @patch('msmodelslim.pytorch.llm_ptq.llm_ptq_tools.flat_quant.trainer.get_model_bridge')
    @patch('msmodelslim.pytorch.llm_ptq.llm_ptq_tools.flat_quant.trainer.get_module_by_name')
    def test_get_model_bridge_should_return_bridge(self, mock_get_module, mock_get_bridge, mock_model, mock_args):
        mock_bridge = MockModelBridge()
        mock_get_bridge.return_value = mock_bridge
        mock_get_module.return_value = []

        adapter = ModelAdapter(mock_model, mock_args)
        result = adapter.get_model_bridge()

        assert result == mock_bridge

    @patch('msmodelslim.pytorch.llm_ptq.llm_ptq_tools.flat_quant.trainer.get_model_bridge')
    @patch('msmodelslim.pytorch.llm_ptq.llm_ptq_tools.flat_quant.trainer.get_module_by_name')
    @patch('msmodelslim.pytorch.llm_ptq.llm_ptq_tools.flat_quant.trainer.quantize_model')
    def test_create_quantizer_should_create_quantizer_with_config(
        self, mock_quantize, mock_get_module, mock_get_bridge, mock_model, mock_args
    ):
        mock_bridge = MockModelBridge()
        mock_get_bridge.return_value = mock_bridge
        mock_get_module.return_value = []
        mock_quantizer = MockQuantizer()
        mock_quantize.return_value = mock_quantizer

        adapter = ModelAdapter(mock_model, mock_args)
        layer_map = {}
        result = adapter.create_quantizer(mock_bridge, layer_map)

        assert result == mock_quantizer
        mock_quantize.assert_called_once()

    @patch('msmodelslim.pytorch.llm_ptq.llm_ptq_tools.flat_quant.trainer.get_model_bridge')
    @patch('msmodelslim.pytorch.llm_ptq.llm_ptq_tools.flat_quant.trainer.get_module_by_name')
    def test_get_layer_device_should_return_default_device_when_no_hf_device_map(
        self, mock_get_module, mock_get_bridge, mock_model, mock_args
    ):
        mock_bridge = MockModelBridge()
        mock_get_bridge.return_value = mock_bridge
        mock_get_module.return_value = []

        adapter = ModelAdapter(mock_model, mock_args)
        device = adapter.get_layer_device(0)

        assert device == mock_model.device

    @patch('msmodelslim.pytorch.llm_ptq.llm_ptq_tools.flat_quant.trainer.get_model_bridge')
    @patch('msmodelslim.pytorch.llm_ptq.llm_ptq_tools.flat_quant.trainer.get_module_by_name')
    def test_get_layer_device_should_return_mapped_device_when_hf_device_map_exists(
        self, mock_get_module, mock_get_bridge, mock_model, mock_args
    ):
        mock_bridge = MockModelBridge()
        mock_get_bridge.return_value = mock_bridge
        mock_get_module.return_value = []
        mock_model.hf_device_map = {"model.layers.0": "cuda:0"}

        adapter = ModelAdapter(mock_model, mock_args)
        device = adapter.get_layer_device(0)

        assert device == "cuda:0"


class TestDataPreparer:
    @pytest.fixture
    def mock_model_adapter(self):
        adapter = Mock()
        adapter.prepare_first_layer_input.return_value = {
            'inputs': [([torch.randn(2, 10)],), ([torch.randn(2, 10)],)],
            'kwargs_list': [{}, {}],
        }
        return adapter

    @pytest.fixture
    def mock_training_config(self):
        return Mock()

    def test_prepare_calibration_data_should_return_data_info(self, mock_model_adapter, mock_training_config):
        preparer = DataPreparer(mock_model_adapter, mock_training_config)
        calib_data = [torch.randn(2, 10), torch.randn(2, 10)]

        result = preparer.prepare_calibration_data(calib_data)

        assert 'layer_inputs' in result
        assert 'layer_kwargs_list' in result
        assert 'nsamples' in result
        assert result['nsamples'] == 2
        mock_model_adapter.prepare_first_layer_input.assert_called_once_with(calib_data)


class TestLayerTrainer:
    @pytest.fixture
    def mock_training_config(self):
        config = Mock()
        config.epochs = 2
        config.flat_lr = 5e-3
        config.warmup = False
        config.quant_by_quant = False
        config.traincast.return_value.__enter__ = Mock(return_value=None)
        config.traincast.return_value.__exit__ = Mock(return_value=None)
        return config

    def test_init_should_setup_layer_trainer(self, mock_training_config):
        trainer = LayerTrainer(mock_training_config)

        assert trainer.config == mock_training_config
        assert isinstance(trainer.loss_fn, torch.nn.MSELoss)

    @patch('msmodelslim.pytorch.llm_ptq.llm_ptq_tools.flat_quant.trainer.get_trainable_parameters')
    def test_train_layer_should_return_outputs_when_no_training_needed(self, mock_get_params, mock_training_config):
        mock_get_params.return_value = ([], [], False)

        trainer = LayerTrainer(mock_training_config)
        training_data = LayerTrainingData(
            layer=nn.Linear(10, 10),
            layer_inputs=[],
            fp_outs=[torch.randn(2, 10)],
            data_info={'nsamples': 1, 'layer_kwargs_list': [{}]},
            device=torch.device("cpu"),
            layer_idx=0,
        )
        logger = Mock()

        result = trainer.train_layer(training_data, logger)

        assert len(result) == 1
        assert len(result[0]) == 1


class TestLayerTrainingData:
    def test_layer_training_data_creation(self):
        layer = nn.Linear(10, 10)
        layer_inputs = [([torch.randn(2, 10)],)]
        fp_outs = [torch.randn(2, 10)]
        data_info = {'nsamples': 1}
        device = torch.device("cpu")
        layer_idx = 0

        training_data = LayerTrainingData(
            layer=layer,
            layer_inputs=layer_inputs,
            fp_outs=fp_outs,
            data_info=data_info,
            device=device,
            layer_idx=layer_idx,
        )

        assert training_data.layer == layer
        assert training_data.layer_inputs == layer_inputs
        assert training_data.fp_outs == fp_outs
        assert training_data.data_info == data_info
        assert training_data.device == device
        assert training_data.layer_idx == layer_idx


class TestFlatQuantTrainer:
    @pytest.fixture
    def mock_model(self):
        return MockModel()

    @pytest.fixture
    def mock_args(self):
        return MockArgs()

    @pytest.fixture
    def mock_logger(self):
        return Mock()

    @patch('msmodelslim.pytorch.llm_ptq.llm_ptq_tools.flat_quant.trainer.ModelAdapter')
    @patch('msmodelslim.pytorch.llm_ptq.llm_ptq_tools.flat_quant.trainer.TrainingConfig')
    @patch('msmodelslim.pytorch.llm_ptq.llm_ptq_tools.flat_quant.trainer.LayerTrainer')
    @patch('msmodelslim.pytorch.llm_ptq.llm_ptq_tools.flat_quant.trainer.DataPreparer')
    def test_init_should_setup_trainer_components(
        self,
        mock_data_preparer,
        mock_layer_trainer,
        mock_training_config,
        mock_model_adapter,
        mock_model,
        mock_args,
        mock_logger,
    ):
        trainer = FlatQuantTrainer(mock_model, mock_args, mock_logger)

        assert trainer.logger == mock_logger
        assert trainer.args == mock_args
        mock_model_adapter.assert_called_once_with(mock_model, mock_args)
        mock_training_config.assert_called_once_with(mock_args)

    @patch('msmodelslim.pytorch.llm_ptq.llm_ptq_tools.flat_quant.trainer.empty_cache')
    @patch('msmodelslim.pytorch.llm_ptq.llm_ptq_tools.flat_quant.trainer.tqdm')
    def test_train_should_execute_training_process(
        self, mock_tqdm, mock_empty_cache, mock_model, mock_args, mock_logger
    ):
        # Setup mocks
        mock_model_adapter = Mock()
        mock_model_bridge = Mock()
        mock_model_adapter.get_model_bridge.return_value = mock_model_bridge
        mock_model_adapter.num_layers = 1
        mock_model_adapter.layers = [nn.Linear(10, 10)]
        mock_model_adapter.get_layer_device.return_value = torch.device("cpu")
        mock_model_adapter.get_layer_by_index.return_value = "layers.0"
        mock_model_adapter.create_quantizer.return_value = Mock()
        mock_model_adapter.finalize_model = Mock()

        mock_data_preparer = Mock()
        mock_data_info = {'layer_inputs': [([torch.randn(2, 10)],)], 'layer_kwargs_list': [{}], 'nsamples': 1}
        mock_data_preparer.prepare_calibration_data.return_value = mock_data_info

        mock_layer_trainer = Mock()
        mock_layer_trainer.extract_fp_outputs.return_value = [torch.randn(2, 10)]
        mock_layer_trainer.train_layer.return_value = [[torch.randn(2, 10)]]

        mock_training_config = Mock()
        mock_training_config.dtype = torch.float32
        mock_training_config.quant_by_quant = False

        with (
            patch(
                'msmodelslim.pytorch.llm_ptq.llm_ptq_tools.flat_quant.trainer.ModelAdapter',
                return_value=mock_model_adapter,
            ),
            patch(
                'msmodelslim.pytorch.llm_ptq.llm_ptq_tools.flat_quant.trainer.DataPreparer',
                return_value=mock_data_preparer,
            ),
            patch(
                'msmodelslim.pytorch.llm_ptq.llm_ptq_tools.flat_quant.trainer.LayerTrainer',
                return_value=mock_layer_trainer,
            ),
            patch(
                'msmodelslim.pytorch.llm_ptq.llm_ptq_tools.flat_quant.trainer.TrainingConfig',
                return_value=mock_training_config,
            ),
        ):
            trainer = FlatQuantTrainer(mock_model, mock_args, mock_logger)
            trainer.train([], {})

            mock_data_preparer.prepare_calibration_data.assert_called_once()
            mock_model_adapter.create_quantizer.assert_called_once()
            mock_model_adapter.finalize_model.assert_called_once()


class TestFlatQuantTrainFunction:
    @patch('msmodelslim.pytorch.llm_ptq.llm_ptq_tools.flat_quant.trainer.FlatQuantTrainer')
    def test_flat_quant_train_should_create_trainer_and_call_train(self, mock_trainer_class):
        mock_trainer = Mock()
        mock_trainer_class.return_value = mock_trainer

        model = MockModel()
        calib_data = []
        layer_map = {}
        args = MockArgs()
        logger = Mock()

        flat_quant_train(model, calib_data, layer_map, args, logger)

        mock_trainer_class.assert_called_once_with(model, args, logger)
        mock_trainer.train.assert_called_once_with(calib_data, layer_map)


@patch('msmodelslim.pytorch.llm_ptq.llm_ptq_tools.flat_quant.trainer.npu_available', True)
@patch('msmodelslim.pytorch.llm_ptq.llm_ptq_tools.flat_quant.trainer.torch')
def test_empty_cache_npu(mock_torch):
    empty_cache()
    mock_torch.npu.empty_cache.assert_called_once()


@patch('msmodelslim.pytorch.llm_ptq.llm_ptq_tools.flat_quant.trainer.npu_available', False)
@patch('msmodelslim.pytorch.llm_ptq.llm_ptq_tools.flat_quant.trainer.torch')
def test_empty_cache_cuda(mock_torch):
    empty_cache()
    mock_torch.cuda.empty_cache.assert_called_once()


@patch('msmodelslim.pytorch.llm_ptq.llm_ptq_tools.flat_quant.trainer.npu_available', False)
def test_get_device_str_cuda():
    assert get_device_str(0) == 'cuda:0'
    assert get_device_str('cuda:1') == 'cuda:1'
    assert get_device_str('cpu') == 'cuda:cpu'


@patch('msmodelslim.pytorch.llm_ptq.llm_ptq_tools.flat_quant.trainer.npu_available', True)
def test_get_device_str_npu():
    assert get_device_str(0) == 'npu:0'
    assert get_device_str('npu:1') == 'npu:1'
    assert get_device_str('cpu') == 'npu:cpu'


def test_convert_outputs_to_inputs_basic():
    outs = [torch.tensor([1, 2]), torch.tensor([3, 4])]
    result = convert_outputs_to_inputs(outs)
    assert result == [[outs[0]], [outs[1]]]


@patch('msmodelslim.pytorch.llm_ptq.llm_ptq_tools.flat_quant.trainer.get_model_bridge')
@patch('msmodelslim.pytorch.llm_ptq.llm_ptq_tools.flat_quant.trainer.get_module_by_name')
def test_run_calibration_tuple_list_dict(mock_get_module, mock_get_bridge):
    from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.flat_quant.trainer import ModelAdapter

    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.device = torch.device('cpu')
            self.layers = torch.nn.ModuleList([torch.nn.Linear(2, 2)])

        def forward(self, *args, **kwargs):
            return torch.ones(2, 2)

        def tie_weights(self):
            pass

    mock_get_bridge.return_value.get_layers.return_value = 'layers'
    mock_get_module.return_value = [torch.nn.Linear(2, 2)]
    args = MockArgs()
    adapter = ModelAdapter(DummyModel(), args)
    # tuple
    adapter._run_calibration((torch.ones(2, 2),))
    # list
    adapter._run_calibration([torch.ones(2, 2)])
    # dict
    adapter._run_calibration({'x': torch.ones(2, 2)})


@patch('msmodelslim.pytorch.llm_ptq.llm_ptq_tools.flat_quant.trainer.dispatch_model')
def test_finalize_model_dispatch_and_eval_mode(mock_dispatch):
    from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.flat_quant.trainer import ModelAdapter

    class DummyQuant:
        def to_eval_mode(self):
            self.eval_called = True

    class DummyModel:
        def __init__(self):
            self.tie_called = False
            self.hf_device_map = {'a': 'b'}
            self.device = 'cpu'

        def tie_weights(self):
            self.tie_called = True

    args = MockArgs()
    adapter = ModelAdapter.__new__(ModelAdapter)
    adapter.model = DummyModel()
    adapter.args = args
    adapter.model_bridge = MagicMock()
    adapter.layers = []
    adapter.num_layers = 0
    adapter.default_device = 'cpu'
    quant = DummyQuant()
    adapter.finalize_model(quant)
    assert hasattr(quant, 'eval_called')
    mock_dispatch.assert_called_once()


@patch('torch.optim.AdamW')
@patch('torch.optim.lr_scheduler.CosineAnnealingLR')
@patch('torch.optim.lr_scheduler.LinearLR')
@patch('torch.optim.lr_scheduler.ChainedScheduler')
def test_setup_optimizer_with_and_without_warmup(mock_chain, mock_linear, mock_cosine, mock_adam):
    config = Mock()
    config.epochs = 2
    config.flat_lr = 1e-3
    config.warmup = True
    trainer = LayerTrainer(config)
    trainable_params = [torch.nn.Parameter(torch.randn(2, 2))]
    data_info = {'nsamples': 2}
    trainer.setup_optimizer(trainable_params, data_info)
    config.warmup = False
    trainer.setup_optimizer(trainable_params, data_info)
    assert mock_chain.called and mock_linear.called and mock_cosine.called and mock_adam.called


@patch('msmodelslim.pytorch.llm_ptq.llm_ptq_tools.flat_quant.trainer.get_trainable_parameters')
def test_train_layer_with_training_and_quant_by_quant(mock_get_params):
    config = Mock()
    config.epochs = 1
    config.flat_lr = 1e-3
    config.warmup = False
    config.quant_by_quant = True
    config.traincast = lambda: nullcontext()
    trainer = LayerTrainer(config)
    param = torch.nn.Parameter(torch.randn(2, 2, requires_grad=True))
    layer = torch.nn.Linear(2, 2)
    mock_get_params.return_value = ([param], [param], True)
    training_data = LayerTrainingData(
        layer=layer,
        layer_inputs=[(torch.ones(2, 2),)],
        fp_outs=[torch.ones(2, 2)],
        data_info={'nsamples': 1, 'layer_kwargs_list': [{}]},
        device=torch.device('cpu'),
        layer_idx=0,
    )
    logger = Mock()
    result = trainer.train_layer(training_data, logger)
    assert isinstance(result, list)


def test_extract_fp_outputs_tensor_and_kwargs():
    layer = torch.nn.Linear(2, 2)
    layer_inputs = [(torch.ones(2, 2),)]
    data_info = {'nsamples': 1, 'layer_kwargs_list': [{}]}
    device = torch.device('cpu')
    outs = LayerTrainer.extract_fp_outputs(layer, layer_inputs, data_info, device)
    assert isinstance(outs, list)
    assert outs[0].shape[0] == 2

    # test with kwargs, avoid multiple values for 'x'
    def forward(self, *args, **kwargs):
        # Accept both positional and keyword arguments
        if args:
            return args[0], 0
        elif 'x' in kwargs:
            return kwargs['x'], 0
        else:
            raise ValueError("No input provided")

    layer.forward = types.MethodType(forward, layer)
    layer_inputs = [(torch.ones(2, 2),)]
    data_info = {'nsamples': 1, 'layer_kwargs_list': [{'x': torch.ones(2, 2)}]}
    outs = LayerTrainer.extract_fp_outputs(layer, layer_inputs, data_info, device)
    assert isinstance(outs, list)
    assert outs[0].shape[0] == 2
