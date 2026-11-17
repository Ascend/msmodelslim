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

import argparse
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch

import pytest
import torch
from torch import nn

from msmodelslim.core.base.protocol import ProcessRequest
from msmodelslim.model.flux1.model_adapter import FLUX1ModelAdapter, InvalidModelError, SchemaValidateError

test_model_path = "."


class FluxTransformerBlock(nn.Module):
    def forward(self, hidden_states, encoder_hidden_states=None):
        return encoder_hidden_states, hidden_states


class FluxModel(nn.Module):
    def __init__(self, num_blocks=3):
        super().__init__()
        for i in range(num_blocks):
            setattr(self, f"block{i}", FluxTransformerBlock())

    def forward(self, hidden_states=None, encoder_hidden_states=None, **kwargs):
        h = hidden_states
        for i in range(self._num_blocks()):
            block = getattr(self, f"block{i}")
            enc, h = block(h, encoder_hidden_states=encoder_hidden_states)
        return h

    def _num_blocks(self):
        return sum(1 for name in dir(self) if name.startswith("block"))


class FluxAttention(nn.Module):
    pass


test_model_path = "."


# ------------------------------ 适配器基础功能测试 ------------------------------
class TestFLUX1ModelAdapter:
    @staticmethod
    def test_initialization():
        with patch("msmodelslim.model.flux1.model_adapter.FLUX1ModelAdapter._get_default_model_args"):
            adapter = FLUX1ModelAdapter("flux1", Path(test_model_path))
            assert adapter.model_type == "flux1"
            assert adapter.model_path == Path(test_model_path)
            assert adapter.transformer_blocks_layers == 19
            assert adapter.single_transformer_blocks_layers == 38

    @staticmethod
    def test_get_model_info():
        with patch("msmodelslim.model.flux1.model_adapter.FLUX1ModelAdapter._get_default_model_args"):
            adapter = FLUX1ModelAdapter("flux1", Path(test_model_path))
            assert adapter.get_model_type() == "flux1"
            assert adapter.get_model_pedigree() == "flux1"

    @staticmethod
    def test_handle_dataset():
        with patch("msmodelslim.model.flux1.model_adapter.FLUX1ModelAdapter._get_default_model_args"):
            adapter = FLUX1ModelAdapter("flux1", Path(test_model_path))
            mock_dataset = [Mock(), Mock()]
            result = adapter.handle_dataset(mock_dataset)
            assert list(result) == mock_dataset

    @staticmethod
    def test_enable_kv_cache():
        with patch("msmodelslim.model.flux1.model_adapter.FLUX1ModelAdapter._get_default_model_args"):
            adapter = FLUX1ModelAdapter("flux1", Path(test_model_path))
            # 方法应无异常执行
            adapter.enable_kv_cache(Mock(), True)

    @staticmethod
    def test_init_model():
        with patch("msmodelslim.model.flux1.model_adapter.FLUX1ModelAdapter._get_default_model_args"):
            adapter = FLUX1ModelAdapter("flux1", Path(test_model_path))
            adapter.transformer = Mock()
            result = adapter.init_model()
            assert isinstance(result, dict)
            assert '' in result
            assert result[''] == adapter.transformer


# ------------------------------ apply_quantization方法测试 ------------------------------
class TestApplyQuantization:
    @staticmethod
    def test_apply_quantization_with_no_sync(mock_self, process_func):
        mock_self.no_sync = MagicMock()
        mock_context = MagicMock()
        mock_self.no_sync.return_value = mock_context

        mock_self.model_args.param_dtype = torch.float16
        with patch('torch.cuda.amp.autocast'):
            FLUX1ModelAdapter.apply_quantization(mock_self, process_func)

        mock_self.no_sync.assert_called_once()
        process_func.assert_called_once()

    @pytest.fixture
    def mock_self(self):
        mock = Mock()
        transformer = Mock()
        module_embedding = Mock()
        module_block = Mock()
        module_norm = Mock()
        transformer.named_modules.return_value = [
            ('embedding', module_embedding),
            ('blocks.0', module_block),
            ('norm', module_norm),
        ]
        mock.transformers = transformer
        return mock

    @pytest.fixture
    def process_func(self):
        return Mock()


# ------------------------------ generate_model_forward方法测试 ------------------------------
class TestGenerateModelForward:
    @staticmethod
    def test_generate_model_forward_basic():
        with patch("msmodelslim.model.flux1.model_adapter.FLUX1ModelAdapter._get_default_model_args"):
            adapter = FLUX1ModelAdapter("flux1", Path(test_model_path))
            # 模拟模型和transformer blocks
            mock_block1 = Mock()
            mock_block2 = Mock()
            mock_block3 = Mock()
            mock_model = Mock()
            mock_model.named_modules.return_value = [
                ('transformer.block1', mock_block1),
                ('transformer.block2', mock_block2),
                ('transformer.block3', mock_block3),
                ('other.module', Mock()),
            ]
            adapter.transformer_blocks_layers = 2
            # 模拟hook移除
            mock_hook = Mock()
            mock_block1.register_forward_pre_hook.return_value = mock_hook
            # 模拟输入
            mock_inputs = {'hidden_states': torch.randn(1, 10, 768)}
            with patch.object(adapter, 'generate_model_forward'):
                generator = adapter.generate_model_forward(mock_model, mock_inputs)
                try:
                    next(generator)
                except StopIteration:
                    pass

    # 以下为 generate_model_forward 真实逻辑补充用例（不 mock 方法本身）
    def test_generate_model_forward_call_barrier_when_dist_initialized(self, flux1_adapter):
        """分布式已初始化时应调用 dist.barrier"""
        model = FluxModel(num_blocks=1)
        flux1_adapter.transformer_blocks_layers = 1
        with patch("torch.distributed.is_initialized", return_value=True):
            with patch("torch.distributed.barrier") as mock_barrier:
                gen = flux1_adapter.generate_model_forward(model, {"hidden_states": torch.randn(1, 4, 8)})
                next(gen)
                mock_barrier.assert_called_once()
        with pytest.raises(StopIteration):
            gen.send((torch.randn(1, 2, 8), torch.randn(1, 2, 8)))

    def test_generate_model_forward_yield_requests_when_dict_inputs(self, flux1_adapter):
        """dict 输入时应逐块 yield ProcessRequest 并更新 hidden_states"""
        model = FluxModel()
        flux1_adapter.transformer_blocks_layers = 2
        inputs = {
            "hidden_states": torch.randn(1, 4, 8),
            "encoder_hidden_states": torch.randn(1, 4, 8),
        }

        gen = flux1_adapter.generate_model_forward(model, inputs)
        req0 = next(gen)
        assert isinstance(req0, ProcessRequest)
        assert req0.name == "block0"

        req1 = gen.send((torch.randn(1, 4, 8), torch.randn(1, 4, 8)))
        assert req1.name == "block1"

        req2 = gen.send((torch.randn(1, 2, 8), torch.randn(1, 2, 8)))
        assert req2.name == "block2"

        with pytest.raises(StopIteration):
            gen.send(torch.randn(1, 6, 8))

    def test_generate_model_forward_yield_requests_when_list_inputs(self, flux1_adapter):
        """list/tuple 输入时应走 model(*inputs) 分支"""
        model = FluxModel(num_blocks=1)
        flux1_adapter.transformer_blocks_layers = 1
        gen = flux1_adapter.generate_model_forward(model, [torch.randn(1, 4, 8)])
        assert next(gen).name == "block0"
        with pytest.raises(StopIteration):
            gen.send((torch.randn(1, 2, 8), torch.randn(1, 2, 8)))

    def test_generate_model_forward_yield_requests_when_tensor_inputs(self, flux1_adapter):
        """单 tensor 输入时应走 model(inputs) 分支"""
        model = FluxModel(num_blocks=1)
        flux1_adapter.transformer_blocks_layers = 1
        gen = flux1_adapter.generate_model_forward(model, torch.randn(1, 4, 8))
        assert next(gen).name == "block0"

    def test_generate_model_forward_raise_invalid_model_when_no_hook_input(self, flux1_adapter):
        """forward 未触发首块 hook 时应抛 InvalidModelError"""

        class SkipBlockModel(nn.Module):
            def named_modules(self):
                yield "block0", FluxTransformerBlock()

            def forward(self, **_kwargs):
                return None

        with pytest.raises(InvalidModelError, match="Can't get first block input"):
            list(flux1_adapter.generate_model_forward(SkipBlockModel(), {"hidden_states": torch.randn(1, 4, 8)}))

    def test_generate_model_forward_reraise_when_non_break_exception(self, flux1_adapter):
        """非 TransformersForwardBreak 异常应原样抛出"""
        model = FluxModel(num_blocks=1)
        model.forward = lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("forward failed"))
        with pytest.raises(RuntimeError, match="forward failed"):
            list(flux1_adapter.generate_model_forward(model, {"hidden_states": torch.randn(1, 4, 8)}))


# ------------------------------ generate_model_visit方法测试 ------------------------------
class TestGenerateModelVisit:
    @staticmethod
    def test_generate_model_visit():
        with patch("msmodelslim.model.flux1.model_adapter.FLUX1ModelAdapter._get_default_model_args"):
            adapter = FLUX1ModelAdapter("flux1", Path(test_model_path))
            mock_model = Mock()
            # 测试使用默认的generated_decoder_layer_visit_func_with_keyword
            with patch(
                'msmodelslim.model.flux1.model_adapter.generated_decoder_layer_visit_func_with_keyword'
            ) as mock_func:
                mock_func.return_value = iter([Mock()])
                result = adapter.generate_model_visit(mock_model)
                mock_func.assert_called_once_with(mock_model, keyword="transformerblock")
                assert result is not None


# ------------------------------ load_pipeline方法测试 ------------------------------
class TestLoadPipeline:
    @staticmethod
    def test_load_pipeline_success():
        with patch("msmodelslim.model.flux1.model_adapter.FLUX1ModelAdapter._get_default_model_args"):
            adapter = FLUX1ModelAdapter("flux1", Path(test_model_path))
            # 模拟FluxPipeline
            mock_flux_pipeline = MagicMock()
            mock_transformer = MagicMock()
            mock_transformer.config = MagicMock()
            mock_transformer.config.num_layers = 19
            mock_transformer.config.num_single_layers = 38
            mock_flux_pipeline.transformer = mock_transformer

            with patch('diffusers.FluxPipeline.from_pretrained', return_value=mock_flux_pipeline):
                with patch('msmodelslim.model.flux1.model_adapter.get_valid_read_path', return_value="/valid/path"):
                    adapter._load_pipeline()
                    assert adapter.model is not None
                    assert adapter.transformer is not None
                    assert adapter.transformer_blocks_layers == 19
                    assert adapter.single_transformer_blocks_layers == 38

    @staticmethod
    def test_load_pipeline_import_error():
        with patch("msmodelslim.model.flux1.model_adapter.FLUX1ModelAdapter._get_default_model_args"):
            adapter = FLUX1ModelAdapter("flux1", Path(test_model_path))
            # 模拟ImportError
            with patch.dict('sys.modules', {'diffusers': None}):
                with pytest.raises(InvalidModelError) as exc_info:
                    adapter._load_pipeline()

                assert "Failed to import FluxPipeline" in str(exc_info.value)

    @staticmethod
    def test_init_model_returns_transformer_dict():
        with patch("msmodelslim.model.flux1.model_adapter.FLUX1ModelAdapter._get_default_model_args"):
            adapter = FLUX1ModelAdapter("flux1", Path(test_model_path))
            mock_transformer = Mock()
            adapter.transformer = mock_transformer
            assert adapter.init_model() == {'': mock_transformer}

    @staticmethod
    def test_load_pipeline_calls_internal():
        with patch("msmodelslim.model.flux1.model_adapter.FLUX1ModelAdapter._get_default_model_args"):
            adapter = FLUX1ModelAdapter("flux1", Path(test_model_path))
            with patch.object(adapter, '_load_pipeline') as mock_internal:
                adapter.load_pipeline()
                mock_internal.assert_called_once()


# ------------------------------ set_model_args方法测试 ------------------------------
class TestSetModelArgs:
    @staticmethod
    def test_set_model_args_valid():
        with patch("msmodelslim.model.flux1.model_adapter.FLUX1ModelAdapter._get_default_model_args"):
            adapter = FLUX1ModelAdapter("flux1", Path(test_model_path))
            # 设置初始参数
            adapter.model_path = Path(test_model_path)

            # 创建模型参数对象
            class MockModelArgs:
                def __init__(self):
                    self.model_path = ""
                    self.prompt = None
                    self.num_inference_steps = 50
                    self.batch_size = 1
                    self.seed = 42
                    self.height = 1024
                    self.width = 1024
                    self.guidance_scale = 3.5
                    self.num_images_per_prompt = 1
                    self.save_path = "./results"
                    self.save_path_suffix = ""

            adapter.model_args = MockModelArgs()
            # 模拟参数解析器
            mock_parser = MagicMock()
            mock_parser.parse_args.return_value = adapter.model_args
            with patch.object(adapter, '_get_parser', return_value=mock_parser):
                with patch.object(adapter, '_validate_args'):
                    # 测试更新配置
                    override_config = {'prompt': 'test prompt', 'num_inference_steps': 30, 'height': 512, 'width': 512}
                    adapter.set_model_args(override_config)
                    assert adapter.model_args.model_path == adapter.model_path
                    mock_parser.parse_args.assert_called_once()

    @staticmethod
    def test_set_model_args_illegal_attributes():
        with patch("msmodelslim.model.flux1.model_adapter.FLUX1ModelAdapter._get_default_model_args"):
            adapter = FLUX1ModelAdapter("flux1", Path(test_model_path))

            # 设置初始参数
            class MockModelArgs:
                def __init__(self):
                    self.model_path = ""
                    self.prompt = None

            adapter.model_args = MockModelArgs()
            # 模拟有非法属性的配置
            override_config = {'prompt': 'test prompt', 'illegal_attr': 'invalid'}
            with pytest.raises(SchemaValidateError) as exc_info:
                adapter.set_model_args(override_config)
            assert "illegal config attributes" in str(exc_info.value)

    @staticmethod
    def test_set_model_args_skip_none():
        with patch("msmodelslim.model.flux1.model_adapter.FLUX1ModelAdapter._get_default_model_args"):
            adapter = FLUX1ModelAdapter("flux1", Path(test_model_path))

            # 设置初始参数
            class MockModelArgs:
                def __init__(self):
                    self.model_path = ""
                    self.prompt = None
                    self.num_inference_steps = 50

            adapter.model_args = MockModelArgs()
            mock_parser = MagicMock()
            mock_parser.parse_args.return_value = adapter.model_args
            with patch.object(adapter, '_get_parser', return_value=mock_parser):
                with patch.object(adapter, '_validate_args'):
                    # 测试None值被跳过
                    override_config = {
                        'prompt': 'test prompt',
                        'num_inference_steps': None,  # 应该被跳过
                    }
                    adapter.set_model_args(override_config)
                    # 检查传递给parse_args的参数
                    call_args = mock_parser.parse_args.call_args[0][0]
                    assert '--num_inference_steps' not in call_args
                    assert '--prompt' in call_args


# ------------------------------ _validate_args方法测试 ------------------------------
class TestValidateArgs:
    @staticmethod
    def test_validate_args_valid():
        with patch("msmodelslim.model.flux1.model_adapter.FLUX1ModelAdapter._get_default_model_args"):
            adapter = FLUX1ModelAdapter("flux1", Path(test_model_path))

            # 创建有效的参数对象
            class Args:
                def __init__(self):
                    self.prompt = "test prompt"
                    self.num_inference_steps = 50
                    self.batch_size = 1
                    self.seed = 42
                    self.height = 1024
                    self.width = 1024
                    self.save_path = "./results"
                    self.save_path_suffix = ""
                    self.task_config = None

            args = Args()
            with patch('os.makedirs'):
                adapter._validate_args(args)
                assert args.task_config == 'FLUX.1-dev'
                assert args.num_inference_steps == 50
                assert args.batch_size == 1
                assert args.seed == 42

    @staticmethod
    def test_validate_args_missing_prompt():
        with patch("msmodelslim.model.flux1.model_adapter.FLUX1ModelAdapter._get_default_model_args"):
            adapter = FLUX1ModelAdapter("flux1", Path(test_model_path))

            # 创建缺少prompt的参数对象
            class Args:
                def __init__(self):
                    self.prompt = None
                    self.num_inference_steps = 50
                    self.save_path = "./results"
                    self.save_path_suffix = ""
                    self.batch_size = 1
                    self.seed = 42

            args = Args()
            with patch('os.makedirs'):
                with pytest.raises(SchemaValidateError) as exc_info:
                    adapter._validate_args(args)
                assert "Missing required parameter: prompt" in str(exc_info.value)

    @staticmethod
    def test_validate_args_invalid_prompt_type():
        with patch("msmodelslim.model.flux1.model_adapter.FLUX1ModelAdapter._get_default_model_args"):
            adapter = FLUX1ModelAdapter("flux1", Path(test_model_path))

            # 创建类型错误的prompt参数
            class Args:
                def __init__(self):
                    self.prompt = 123  # 非字符串
                    self.num_inference_steps = 50
                    self.save_path = "./results"
                    self.save_path_suffix = ""
                    self.batch_size = 1
                    self.seed = 42

            args = Args()
            with patch('os.makedirs'):
                with pytest.raises(SchemaValidateError) as exc_info:
                    adapter._validate_args(args)

                assert "prompt must be a string" in str(exc_info.value)

    @staticmethod
    def test_validate_args_empty_prompt():
        with patch("msmodelslim.model.flux1.model_adapter.FLUX1ModelAdapter._get_default_model_args"):
            adapter = FLUX1ModelAdapter("flux1", Path(test_model_path))

            # 创建空字符串prompt
            class Args:
                def __init__(self):
                    self.prompt = "   "  # 空白字符串
                    self.num_inference_steps = 50
                    self.save_path = "./results"
                    self.save_path_suffix = ""
                    self.batch_size = 1
                    self.seed = 42

            args = Args()
            with patch('os.makedirs'):
                with pytest.raises(SchemaValidateError) as exc_info:
                    adapter._validate_args(args)

                assert "prompt cannot be an empty string" in str(exc_info.value)

    @staticmethod
    def test_validate_args_invalid_num_inference_steps():
        with patch("msmodelslim.model.flux1.model_adapter.FLUX1ModelAdapter._get_default_model_args"):
            adapter = FLUX1ModelAdapter("flux1", Path(test_model_path))

            # 创建无效的num_inference_steps参数
            class Args:
                def __init__(self):
                    self.prompt = "test prompt"
                    self.num_inference_steps = 0  # 无效值
                    self.save_path = "./results"
                    self.save_path_suffix = ""
                    self.batch_size = 1
                    self.seed = 42

            args = Args()
            with patch('os.makedirs'):
                with pytest.raises(SchemaValidateError) as exc_info:
                    adapter._validate_args(args)
                assert "num_inference_steps must be greater than 0" in str(exc_info.value)

    @staticmethod
    def test_validate_args_save_path_creation():
        with patch("msmodelslim.model.flux1.model_adapter.FLUX1ModelAdapter._get_default_model_args"):
            adapter = FLUX1ModelAdapter("flux1", Path(test_model_path))

            # 测试保存路径创建
            class Args:
                def __init__(self):
                    self.prompt = "test prompt"
                    self.num_inference_steps = 50
                    self.save_path = "./test_results"
                    self.save_path_suffix = ""
                    self.batch_size = 1
                    self.seed = 42

            args = Args()
            mock_makedirs = Mock()
            with patch('os.makedirs', mock_makedirs):
                adapter._validate_args(args)
                mock_makedirs.assert_called_once_with("./test_results", exist_ok=True)

    @staticmethod
    def test_validate_args_save_path_with_suffix():
        with patch("msmodelslim.model.flux1.model_adapter.FLUX1ModelAdapter._get_default_model_args"):
            adapter = FLUX1ModelAdapter("flux1", Path(test_model_path))

            # 测试带后缀的保存路径
            class Args:
                def __init__(self):
                    self.prompt = "test prompt"
                    self.num_inference_steps = 50
                    self.save_path = "./results"
                    self.save_path_suffix = "test_suffix"
                    self.batch_size = 1
                    self.seed = 42

            args = Args()
            mock_makedirs = Mock()
            with patch('os.makedirs', mock_makedirs):
                adapter._validate_args(args)
                mock_makedirs.assert_called_once_with('./results_test_suffix', exist_ok=True)


# ------------------------------ run_calib_inference方法测试 ------------------------------
class TestRunCalibInference:
    @staticmethod
    def test_run_calib_inference():
        """测试校准推理"""
        with patch("msmodelslim.model.flux1.model_adapter.FLUX1ModelAdapter._get_default_model_args"):
            adapter = FLUX1ModelAdapter("flux1", Path(test_model_path))
            # 1. 创建模拟的模型
            mock_model = Mock()
            # 2. 模拟模型返回的对象需要具有 images 属性
            mock_return_value = Mock()
            mock_return_value.images = [Mock()]  # 模拟一个包含图像的列表
            mock_model.return_value = mock_return_value
            # 3. 模拟模型的 device 属性
            mock_model.device = "npu:0"  # 或者使用 torch.device('npu:0')
            # 4. 将模型设置到适配器
            adapter.model = mock_model

            # 5. 模拟模型参数
            class MockModelArgs:
                def __init__(self):
                    self.prompt = "test prompt"
                    self.height = 1024
                    self.width = 1024
                    self.num_inference_steps = 50
                    self.num_images_per_prompt = 1
                    self.seed = 42

            adapter.model_args = MockModelArgs()
            # 6. 模拟所有依赖项
            mock_torch_npu = Mock()
            mock_torch_npu.is_available.return_value = True
            mock_torch_npu.manual_seed = Mock()
            mock_torch_npu.manual_seed_all = Mock()
            mock_npu = Mock()

            class MockStream:
                def __init__(self):
                    self.synchronize_called = False

                def synchronize(self):
                    self.synchronize_called = True
                    return "synchronized"

            mock_npu.Stream = MockStream
            torch.npu = mock_npu
            stream = torch.npu.Stream()
            stream.synchronize()
            assert stream.synchronize_called

            with patch.dict('sys.modules', {"torch.npu": mock_torch_npu}):
                with patch('torch.Generator') as mock_generator_class:
                    # 创建模拟的生成器
                    mock_generator = Mock()
                    mock_generator.manual_seed = Mock(return_value=mock_generator)
                    mock_generator_class.return_value = mock_generator

                    with patch('time.time', side_effect=[1000.0, 1002.0]):
                        with patch('msmodelslim.model.flux1.model_adapter.tqdm') as mock_tqdm:
                            # 模拟 tqdm 迭代器
                            mock_tqdm.return_value.__iter__ = Mock(return_value=iter([1]))

                            with patch('msmodelslim.model.flux1.model_adapter.get_logger') as mock_logger:
                                mock_logger_instance = Mock()
                                mock_logger_instance.info = Mock()
                                mock_logger.return_value = mock_logger_instance
                                # 执行测试
                                adapter.run_calib_inference()
                                # 验证模型是否被正确调用
                                mock_model.assert_called_once_with(
                                    prompt="test prompt",
                                    height=1024,
                                    width=1024,
                                    num_inference_steps=50,
                                    num_images_per_prompt=1,
                                    generator=mock_generator,
                                )
                                # 验证日志记录
                                mock_logger_instance.info.assert_called_once()


# ------------------------------ _get_parser方法测试 ------------------------------
class TestGetParser:
    @staticmethod
    def test_get_parser():
        with patch("msmodelslim.model.flux1.model_adapter.FLUX1ModelAdapter._get_default_model_args"):
            adapter = FLUX1ModelAdapter("flux1", Path(test_model_path))
            parser = adapter._get_parser()
            assert isinstance(parser, argparse.ArgumentParser)
            assert parser.description == "Flux.1-dev inference script"
            # 检查是否添加了必要的参数组
            action_names = [action.dest for action in parser._actions]
            expected_args = [
                'model_path',
                'batch_size',
                'num_inference_steps',
                'save_path',
                'save_path_suffix',
                'prompt',
                'guidance_scale',
                'num_images_per_prompt',
                'height',
                'width',
                'seed',
            ]
            for arg in expected_args:
                assert arg in action_names

    @staticmethod
    def test_add_inference_args():
        with patch("msmodelslim.model.flux1.model_adapter.FLUX1ModelAdapter._get_default_model_args"):
            adapter = FLUX1ModelAdapter("flux1", Path(test_model_path))
            parser = argparse.ArgumentParser()
            result = adapter._FLUX1ModelAdapter__add_inference_args(parser)
            assert result == parser
            # 检查参数是否被添加
            action_names = [action.dest for action in parser._actions]
            expected_args = [
                'model_path',
                'batch_size',
                'num_inference_steps',
                'save_path',
                'save_path_suffix',
                'prompt',
                'guidance_scale',
                'num_images_per_prompt',
                'height',
                'width',
                'seed',
            ]
            for arg in expected_args:
                assert arg in action_names


# ------------------------------ inject_fa3_placeholders方法测试 ------------------------------
class TestInjectFA3Placeholders:
    """测试 inject_fa3_placeholders 方法"""

    @staticmethod
    def test_inject_fa3_placeholders_with_attention_modules():
        """为 Attention 模块注入 fa3 占位符"""
        with patch("msmodelslim.model.flux1.model_adapter.FLUX1ModelAdapter._get_default_model_args"):
            adapter = FLUX1ModelAdapter("flux1", Path(test_model_path))
            attn1 = FluxAttention()
            attn2 = FluxAttention()
            mock_root = Mock()
            mock_root.named_modules.return_value = [
                ("attention1", attn1),
                ("sub.attention2", attn2),
                ("linear", nn.Linear(4, 4)),
            ]
            calls = []
            mock_root.set_submodule = lambda p, _ph: calls.append(p)
            adapter.inject_fa3_placeholders("root", mock_root, lambda n: "attention" in n.lower())
            assert len([c for c in calls if "fa3" in c]) == 6
            assert not any("linear" in c for c in calls)


# ------------------------------ get_online_rotation_configs方法测试 ------------------------------
class TestGetOnlineRotationConfigs:
    """测试 get_online_rotation_configs 方法"""

    @staticmethod
    def test_get_online_rotation_configs_with_model():
        """测试传入模型时的情况"""
        with patch("msmodelslim.model.flux1.model_adapter.FLUX1ModelAdapter._get_default_model_args"):
            adapter = FLUX1ModelAdapter("flux1", Path(test_model_path))
            # 创建模拟模块
            mock_flux_module = MagicMock()
            mock_flux_module.__class__.__name__ = "FluxAttention"
            mock_flux_module.attention_head_dim = 64

            # 模拟 register_module 方法
            def mock_register_module(name, module):
                setattr(mock_flux_module, name, module)

            mock_flux_module.register_module = mock_register_module
            # 创建模拟模型
            mock_model = MagicMock()
            mock_model.named_modules.return_value = [("test.attention", mock_flux_module)]
            from msmodelslim.core.quantization_algorithm.quarot import OnlineQuaRotInterface

            online_quarot_base = OnlineQuaRotInterface

            class MockRotationConfig:
                def __init__(self, rotation_type, rotation_size, rotation_mode, block_size, seed, dtype):
                    self.rotation_type = rotation_type
                    self.rotation_size = rotation_size
                    self.rotation_mode = rotation_mode
                    self.block_size = block_size
                    self.seed = seed
                    self.dtype = dtype

            # 模拟 RotationConfig 和 QuaRotMode
            mock_rotation_config_class = Mock(
                return_value=MockRotationConfig("replace", 64, "hadamard", -1, 1234, torch.bfloat16)
            )
            mock_quarot_mode = Mock()
            mock_quarot_mode.HADAMARD = "hadamard"
            with patch.object(online_quarot_base, 'RotationConfig', mock_rotation_config_class):
                with patch.object(online_quarot_base, 'QuaRotMode', mock_quarot_mode):
                    # 执行方法
                    configs = adapter.get_online_rotation_configs(mock_model)
            # 验证结果
            assert isinstance(configs, dict)
            # 验证 q_rot 和 k_rot 已注册
            assert hasattr(mock_flux_module, 'q_rot')
            assert hasattr(mock_flux_module, 'k_rot')
            # 验证配置数量
            assert len(configs) == 2
            assert "test.attention.q_rot" in configs
            assert "test.attention.k_rot" in configs


# ------------------------------ 集成测试 ------------------------------
class TestIntegration:
    @staticmethod
    def test_full_initialization_flow():
        """测试完整的初始化流程"""
        with patch("msmodelslim.model.flux1.model_adapter.FLUX1ModelAdapter._get_default_model_args"):
            adapter = FLUX1ModelAdapter("flux1", Path(test_model_path))
            adapter.model_args = Mock()
            # 验证基本属性
            assert adapter.model_type == "flux1"
            assert adapter.get_model_pedigree() == "flux1"
            # 验证模型参数已初始化
            assert adapter.model_args is not None
            # 验证transformer blocks层数
            assert adapter.transformer_blocks_layers == 19
            assert adapter.single_transformer_blocks_layers == 38

    @staticmethod
    def test_configuration_update_flow():
        """测试配置更新流程"""
        with patch("msmodelslim.model.flux1.model_adapter.FLUX1ModelAdapter._get_default_model_args"):
            adapter = FLUX1ModelAdapter("flux1", Path(test_model_path))
            adapter.model_args = Mock()
            # 更新配置
            new_config = {
                'prompt': 'updated prompt',
                'num_inference_steps': 30,
                'height': 512,
                'width': 512,
                'save_path': './new_results',
            }
            with patch.object(adapter, '_get_parser') as mock_parser:
                mock_parser.return_value.parse_args.return_value = MagicMock(**new_config)
                with patch.object(adapter, '_validate_args'):
                    adapter.set_model_args(new_config)
            # 验证配置已更新
            assert adapter.model_args is not None

    @staticmethod
    def test_pipeline_loading_flow():
        """测试pipeline加载流程"""
        with patch("msmodelslim.model.flux1.model_adapter.FLUX1ModelAdapter._get_default_model_args"):
            adapter = FLUX1ModelAdapter("flux1", Path(test_model_path))
            # 模拟成功的pipeline加载
            mock_flux_pipeline = MagicMock()
            mock_transformer = MagicMock()
            mock_transformer.config = MagicMock()
            mock_transformer.config.num_layers = 19
            mock_transformer.config.num_single_layers = 38
            mock_flux_pipeline.transformer = mock_transformer
            with patch('diffusers.FluxPipeline.from_pretrained', return_value=mock_flux_pipeline):
                with patch('msmodelslim.model.flux1.model_adapter.get_valid_read_path', return_value="/valid/path"):
                    adapter._load_pipeline()
                    # 验证pipeline已加载
                    assert adapter.model is not None
                    assert adapter.transformer is not None
                    # 验证层数配置已更新
                    assert adapter.transformer_blocks_layers == 19
                    assert adapter.single_transformer_blocks_layers == 38


# ------------------------------ 错误处理测试 ------------------------------
class TestErrorHandling:
    @staticmethod
    def test_invalid_model_path():
        """测试无效模型路径处理"""
        with patch("msmodelslim.model.flux1.model_adapter.FLUX1ModelAdapter._get_default_model_args"):
            adapter = FLUX1ModelAdapter("flux1", Path("/invalid/path"))
            with patch('msmodelslim.model.flux1.model_adapter.get_valid_read_path') as mock_validator:
                mock_validator.side_effect = Exception("Invalid path")
                with pytest.raises(Exception):
                    adapter._load_pipeline()

    @staticmethod
    def test_missing_required_args():
        """测试缺少必需参数"""
        with patch("msmodelslim.model.flux1.model_adapter.FLUX1ModelAdapter._get_default_model_args"):
            adapter = FLUX1ModelAdapter("flux1", Path(test_model_path))
            # 创建一个 Mock 对象作为 model_args
            mock_model_args = Mock()
            mock_model_args.model_path = ""
            adapter.model_args = mock_model_args
            # 创建缺少必需参数的配置
            invalid_config = {
                'num_inference_steps': 30
                # 缺少 prompt 参数
            }
            # 模拟参数解析器
            mock_parser = Mock()
            # 创建解析后的参数对象（模拟缺少 prompt 的情况）
            parsed_args = argparse.Namespace(
                model_path=adapter.model_path,
                batch_size=1,
                num_inference_steps=30,
                save_path="./results",
                save_path_suffix="",
                prompt=None,  # 这是 None，会在 _validate_args 中触发错误
                guidance_scale=3.5,
                num_images_per_prompt=1,
                height=1024,
                width=1024,
                seed=None,
            )
            mock_parser.parse_args.return_value = parsed_args
            with patch.object(adapter, '_get_parser', return_value=mock_parser):
                with pytest.raises(SchemaValidateError) as exc_info:
                    adapter.set_model_args(invalid_config)
                assert "prompt" in str(exc_info.value).lower()

    @staticmethod
    def test_invalid_parameter_types():
        """测试无效参数类型"""
        with patch("msmodelslim.model.flux1.model_adapter.FLUX1ModelAdapter._get_default_model_args"):
            adapter = FLUX1ModelAdapter("flux1", Path(test_model_path))
            # 创建类型错误的配置
            invalid_config = {
                'prompt': 123,  # 应该是字符串
                'num_inference_steps': 'invalid',  # 应该是整数
            }
            with pytest.raises(Exception):
                # 注意：实际的类型检查在_validate_args中
                adapter.set_model_args(invalid_config)


class TestSupplementary:
    def test_set_model_args_bool_flag(self, flux1_adapter):
        class M:
            def __init__(self):
                self.model_path = ""
                self.prompt = "hello"
                self.use_cache = True
                self.skip_flag = False

        flux1_adapter.model_args = M()
        mock_parser = MagicMock()
        mock_parser.parse_args.return_value = flux1_adapter.model_args
        with (
            patch.object(flux1_adapter, "_get_parser", return_value=mock_parser),
            patch.object(flux1_adapter, "_validate_args"),
        ):
            flux1_adapter.set_model_args({"prompt": "hello", "use_cache": True, "skip_flag": False})
        assert "--use_cache" in mock_parser.parse_args.call_args[0][0]
        assert "--skip_flag" not in mock_parser.parse_args.call_args[0][0]

    def test_apply_quantization_no_sync_fallback(self, flux1_adapter):
        pf = Mock()
        with patch("torch.cuda.amp.autocast"):
            FLUX1ModelAdapter.apply_quantization(flux1_adapter, pf)
        pf.assert_called_once()

    def test_get_online_rotation_skip_non_attention(self, flux1_adapter):
        m = MagicMock()
        m.named_modules.return_value = [("linear", nn.Linear(4, 4))]
        m.attention_head_dim, m.dtype = 64, torch.float32
        assert flux1_adapter.get_online_rotation_configs(m) == {}

    def test_get_online_rotation_register_warning(self, flux1_adapter):
        attn = FluxAttention()
        m = MagicMock()
        m.named_modules.return_value = [("attn", attn)]
        m.attention_head_dim, m.dtype = 32, torch.float16
        with patch.object(attn, "register_module", side_effect=RuntimeError("register failed")):
            with patch("msmodelslim.model.flux1.model_adapter.get_logger") as lg:
                lg.return_value.warning = Mock()
                flux1_adapter.get_online_rotation_configs(m)
        lg.return_value.warning.assert_called()

    def test_inject_fa3_empty_name(self, flux1_adapter):
        root = MagicMock()
        root.named_modules.return_value = [("", FluxAttention())]
        calls = []
        root.set_submodule = lambda p, _: calls.append(p)
        flux1_adapter.inject_fa3_placeholders("", root, lambda _: True)
        assert set(calls) == {"fa3_q", "fa3_k", "fa3_v"}

    def test_validate_args_fill_defaults(self, flux1_adapter):
        class A:
            prompt = "test"
            num_inference_steps = batch_size = seed = None
            save_path = "./results"
            save_path_suffix = task_config = ""

        args = A()
        with patch("os.makedirs"):
            flux1_adapter._validate_args(args)
        assert args.num_inference_steps == 50 and args.batch_size == 1 and args.seed == 42

    @staticmethod
    def test_get_default_model_args():
        with patch("msmodelslim.model.flux1.model_adapter.FLUX1ModelAdapter._get_default_model_args"):
            inst = FLUX1ModelAdapter("flux1", Path(test_model_path))
        inst._get_default_model_args()
        assert isinstance(inst.model_args, argparse.Namespace)
