#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# pylint: disable=too-many-lines

"""
-------------------------------------------------------------------------
This file is part of the MindStudio project.
Copyright (c) 2025 Huawei Technologies Co.,Ltd.

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

# pylint: disable=no-name-in-module,redefined-outer-name,no-member

import sys
import argparse
import tempfile
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch

import torch
import pytest

from msmodelslim.infra.dataset_loader.vlm_dataset_loader import VlmCalibSample
from msmodelslim.model.hunyuan_video.model_adapter import (
    HunyuanVideoModelAdapter,
    SchemaValidateError,
    UnsupportedError,
)
from msmodelslim.model.hunyuan_video.constants import PLACEHOLDER_PROMPT, TASK_TYPE


@pytest.fixture(autouse=True)
def mock_hyvideo_modules(monkeypatch):
    """统一模拟所有hunyuan video相关模块"""
    mock_modules = [
        'hyvideo',
        'hyvideo.config',
        'hyvideo.constants',
        'hyvideo.modules.models',
        'hyvideo.inference',
        'hyvideo.utils.file_utils',
    ]

    original_modules = {mod: sys.modules.get(mod) for mod in mock_modules}
    for module_path in mock_modules:
        sys.modules[module_path] = MagicMock()

    # 配置关键模块
    hyvideo_constants = sys.modules['hyvideo.constants']
    hyvideo_constants.PRECISIONS = ['fp32', 'fp16', 'bf16']
    hyvideo_constants.VAE_PATH = {'884-16c-hy': '/fake/vae/path'}
    hyvideo_constants.TEXT_ENCODER_PATH = {'llm': '/fake/text/encoder'}
    hyvideo_constants.TOKENIZER_PATH = {'llm': '/fake/tokenizer'}
    hyvideo_constants.PROMPT_TEMPLATE = ['dit-llm-encode', 'dit-llm-encode-video']

    hyvideo_models = sys.modules['hyvideo.modules.models']
    hyvideo_models.HUNYUAN_VIDEO_CONFIG = {'HYVideo-T/2-cfgdistill': Mock()}

    # 模拟HunyuanVideoSampler
    mock_sampler = MagicMock()
    mock_sampler.pipeline = MagicMock()
    mock_sampler.pipeline.transformer = MagicMock()
    mock_sampler.predict = MagicMock(return_value=MagicMock())

    hyvideo_inference = sys.modules['hyvideo.inference']
    hyvideo_inference.HunyuanVideoSampler = MagicMock()
    hyvideo_inference.HunyuanVideoSampler.from_pretrained = MagicMock(return_value=mock_sampler)

    def _default_parse_args(namespace=None):
        ns = argparse.Namespace(
            model_base="ckpts",
            prompt=PLACEHOLDER_PROMPT,
            save_path="./results",
            save_path_suffix="",
            infer_steps=50,
            batch_size=1,
            seed=42,
            video_size=(720, 1280),
            video_length=129,
            neg_prompt=None,
            cfg_scale=1.0,
            num_videos=1,
            flow_shift=7.0,
            embedded_cfg_scale=6.0,
            ulysses_degree=1,
            ring_degree=1,
            vae_parallel=False,
            use_cache=False,
            use_cache_double=False,
            use_attentioncache=False,
            vae="884-16c-hy",
            latent_channels=16,
        )
        return ns

    sys.modules['hyvideo.config'].parse_args = _default_parse_args

    yield

    # 恢复原始模块
    for mod, original in original_modules.items():
        if original is not None:
            sys.modules[mod] = original
        else:
            if mod in sys.modules:
                del sys.modules[mod]


@pytest.fixture
def mock_env(monkeypatch):
    """环境变量模拟"""
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("LOCAL_RANK", "0")


@pytest.fixture
def temp_model_dir():
    """创建临时模型目录"""
    with tempfile.TemporaryDirectory() as temp_dir:
        model_base = Path(temp_dir)
        # 创建必要的子目录结构
        (model_base / "hunyuan-video-t2v-720p" / "transformers").mkdir(parents=True)
        (model_base / "vae").mkdir(parents=True)
        (model_base / "text_encoder").mkdir(parents=True)
        (model_base / "clip-vit-large-patch14").mkdir(parents=True)

        # 创建必要的权重文件
        weight_file = model_base / "hunyuan-video-t2v-720p" / "transformers" / "mp_rank_00_model_states.pt"
        weight_file.touch()

        yield model_base


@pytest.fixture
def hunyuan_adapter_for_config_parse(temp_model_dir):
    """供 validate_inference_config / get_inference_config_class 测试，无需 configure_runtime。"""
    with patch("msmodelslim.model.hunyuan_video.model_adapter.HunyuanVideoModelAdapter._check_import_dependency"):
        return HunyuanVideoModelAdapter("hunyuan_video", temp_model_dir)


@pytest.fixture
def adapter(temp_model_dir):
    """创建适配器实例（model_args 在 configure_runtime 后才可用）。"""
    with patch("msmodelslim.model.hunyuan_video.model_adapter.HunyuanVideoModelAdapter._check_import_dependency"):
        adapter_instance = HunyuanVideoModelAdapter("hunyuan_video", temp_model_dir)
    mock_args = argparse.Namespace(
        model_base=str(temp_model_dir),
        ulysses_degree=1,
        ring_degree=1,
        vae_parallel=False,
        video_size=[720, 1280],
        video_length=129,
        infer_steps=50,
        seed=None,
        neg_prompt=None,
        cfg_scale=1.0,
        num_videos=1,
        flow_shift=7.0,
        batch_size=1,
        embedded_cfg_scale=6.0,
    )
    mock_args.task_config = TASK_TYPE
    adapter_instance.model_args = mock_args
    return adapter_instance


# ------------------------------ 适配器基础功能测试 ------------------------------
class TestHunyuanVideoModelAdapter:
    @staticmethod
    def test_initialization(adapter, temp_model_dir):
        """测试初始化"""
        assert adapter.model_type == "hunyuan_video"
        assert adapter.model_path == temp_model_dir

    @staticmethod
    def test_get_model_info(adapter):
        """测试模型信息获取"""
        assert adapter.get_model_type() == "hunyuan_video"
        assert adapter.get_model_pedigree() == "hunyuan_video"

    @staticmethod
    def test_handle_dataset(adapter):
        """测试数据集处理"""
        samples = [
            VlmCalibSample(text="prompt one"),
            VlmCalibSample(text="prompt two"),
        ]
        result = adapter.handle_dataset(samples)
        assert result == samples

    @staticmethod
    def test_handle_dataset_rejects_image(adapter):
        with pytest.raises(SchemaValidateError):
            adapter.handle_dataset([VlmCalibSample(text="hi", image="/tmp/a.png")])

    @staticmethod
    def test_enable_kv_cache(adapter):
        """测试KV缓存启用"""
        mock_model = Mock()
        adapter.enable_kv_cache(mock_model, True)
        # 方法应无异常执行
        assert True

    @staticmethod
    def test_init_model_returns_transformer(adapter):
        """测试模型初始化返回transformer"""
        mock_transformer = Mock()
        adapter.transformer = mock_transformer
        with patch.object(adapter, "_load_pipeline"), patch.object(adapter, "_setup_cache"):
            result = adapter.init_model()
        assert result == {'': mock_transformer}


# ------------------------------ 模型前向传播测试 ------------------------------
class TestModelForward:
    @staticmethod
    @pytest.mark.parametrize(
        "mock_inputs",  # 参数名
        [
            # 测试用例1：字典输入
            {"input_tensor": torch.randn(1, 10, 512)},
            # 测试用例2：元组输入
            (torch.randn(1, 10, 512), torch.randn(1, 10)),
        ],
        # 可选：为每个用例命名，方便识别测试结果
        ids=["dict_input", "tuple_input"],
    )
    def test_generate_model_forward_with_different_inputs(adapter_with_model, mock_inputs):
        """测试不同类型输入（字典/元组）的前向传播"""
        mock_model = adapter_with_model.transformer

        def mock_to_device_func(x, _):
            return x

        with (
            patch('msmodelslim.model.hunyuan_video.model_adapter.to_device') as mock_to_device,
            patch('msmodelslim.model.hunyuan_video.model_adapter.TransformersForwardBreak', Exception),
            patch('msmodelslim.model.hunyuan_video.model_adapter.dist') as mock_dist,
        ):
            mock_to_device.side_effect = mock_to_device_func
            mock_dist.is_initialized.return_value = False

            generator = adapter_with_model.generate_model_forward(mock_model, mock_inputs)
            assert generator is not None

    @staticmethod
    def test_generate_model_visit(adapter_with_model):
        """测试模型遍历生成器"""
        mock_model = adapter_with_model.transformer

        # 使用patch来模拟generated_decoder_layer_visit_func_with_keyword
        with patch(
            'msmodelslim.model.hunyuan_video.model_adapter.generated_decoder_layer_visit_func_with_keyword'
        ) as mock_visit_func:
            mock_generator = MagicMock()
            mock_visit_func.return_value = mock_generator

            generator = adapter_with_model.generate_model_visit(mock_model)

            # 验证返回的是正确的生成器
            assert generator == mock_generator
            # 验证函数被正确调用
            mock_visit_func.assert_called_once_with(mock_model, keyword="streamblock")

    @pytest.fixture
    def adapter_with_model(self, temp_model_dir):
        """创建带有正确模拟transformer模型的适配器"""
        with patch("msmodelslim.model.hunyuan_video.model_adapter.HunyuanVideoModelAdapter._check_import_dependency"):
            adapter = HunyuanVideoModelAdapter("hunyuan_video", temp_model_dir)

        # 创建真实的transformer模拟结构
        mock_transformer = MagicMock()

        # 创建包含"streamblock"的模拟模块
        mock_streamblock_1 = MagicMock()
        mock_streamblock_2 = MagicMock()

        type(mock_streamblock_1).__name__ = "StreamBlock"
        type(mock_streamblock_2).__name__ = "StreamBlock"

        # 设置模块的named_modules返回值
        mock_modules = [
            ('blocks.0', mock_streamblock_1),
            ('blocks.1', mock_streamblock_2),
            ('embedding', MagicMock()),
            ('norm', MagicMock()),
        ]

        mock_transformer.named_modules.return_value = mock_modules

        # 设置forward方法，确保会调用注册的hook
        def transformer_forward(*args, **kwargs):
            # 模拟调用第一个streamblock的前向传播
            # 这会触发注册的forward_pre_hook
            return mock_streamblock_1.forward(*args, **kwargs)

        mock_transformer.forward = transformer_forward
        mock_transformer.__call__ = transformer_forward

        # 设置适配器的transformer
        adapter.transformer = mock_transformer

        return adapter


# ------------------------------ Pipeline加载测试 ------------------------------
class TestLoadPipeline:
    @staticmethod
    def test_normal_execution(mock_self, mock_env):
        """测试正常执行流程"""
        # 模拟HunyuanVideoSampler.from_pretrained的返回值
        mock_sampler = MagicMock()
        mock_sampler.pipeline = MagicMock()
        mock_sampler.pipeline.transformer = MagicMock()

        from hyvideo.inference import HunyuanVideoSampler

        HunyuanVideoSampler.from_pretrained.return_value = mock_sampler

        HunyuanVideoModelAdapter._load_pipeline(mock_self)

        # 验证HunyuanVideoSampler被正确调用
        HunyuanVideoSampler.from_pretrained.assert_called_once()

        # 验证transformer被设置
        assert mock_self.transformer is not None
        assert mock_self.hunyuan_video_sampler is not None

    @staticmethod
    def test_unsupported_ulysses_degree(mock_self, mock_env):
        """测试不支持的ulysses_degree"""
        mock_self.model_args.ulysses_degree = 2
        with pytest.raises(UnsupportedError):
            HunyuanVideoModelAdapter._load_pipeline(mock_self)

    @staticmethod
    def test_unsupported_ring_degree(mock_self, mock_env):
        """测试不支持的ring_degree"""
        mock_self.model_args.ring_degree = 2
        with pytest.raises(UnsupportedError):
            HunyuanVideoModelAdapter._load_pipeline(mock_self)

    @staticmethod
    def test_unsupported_vae_parallel(mock_self, mock_env):
        """测试不支持的vae_parallel"""
        mock_self.model_args.vae_parallel = True
        with pytest.raises(UnsupportedError):
            HunyuanVideoModelAdapter._load_pipeline(mock_self)

    @staticmethod
    def test_init_model_loads_pipeline_and_cache(mock_self):
        """测试init_model绑定加载pipeline与cache"""
        mock_self._load_pipeline = Mock()
        mock_self._setup_cache = Mock()

        HunyuanVideoModelAdapter.init_model(mock_self)

        mock_self._load_pipeline.assert_called_once()
        mock_self._setup_cache.assert_called_once()

    @pytest.fixture
    def mock_self(self, temp_model_dir):
        """创建模拟的self对象"""
        mock = Mock()
        mock.model_args = Mock()
        mock.model_args.model_base = str(temp_model_dir)
        mock.model_args.ulysses_degree = 1
        mock.model_args.ring_degree = 1
        mock.model_args.vae_parallel = False
        mock._check_import_dependency = Mock()
        return mock


class TestValidateInferenceConfig:
    """get_inference_config_class + quant_service.validate_inference_config"""

    @staticmethod
    def test_validate_inference_config_returns_config_when_valid(hunyuan_adapter_for_config_parse):
        from msmodelslim.core.quant_service.multimodal_sd_v1.quant_config import (
            DumpConfig,
            MultimodalSDConfig,
            validate_inference_config,
        )

        sd_cfg = MultimodalSDConfig(
            dump_config=DumpConfig(),
            inference_config={"infer_steps": 60, "batch_size": 2},
        )
        result = validate_inference_config(hunyuan_adapter_for_config_parse, sd_cfg)
        assert result.infer_steps == 60
        assert result.batch_size == 2

    @staticmethod
    def test_validate_inference_config_raises_schema_error_when_unknown_field(hunyuan_adapter_for_config_parse):
        from msmodelslim.core.quant_service.multimodal_sd_v1.quant_config import (
            DumpConfig,
            MultimodalSDConfig,
            validate_inference_config,
        )

        sd_cfg = MultimodalSDConfig(dump_config=DumpConfig(), inference_config={"illegal_attr": 1})
        with pytest.raises(SchemaValidateError):
            validate_inference_config(hunyuan_adapter_for_config_parse, sd_cfg)

    @staticmethod
    def test_validate_inference_config_raises_schema_error_when_extra_field_task(hunyuan_adapter_for_config_parse):
        from msmodelslim.core.quant_service.multimodal_sd_v1.quant_config import (
            DumpConfig,
            MultimodalSDConfig,
            validate_inference_config,
        )

        sd_cfg = MultimodalSDConfig(dump_config=DumpConfig(), inference_config={"task": "hunyuan_video"})
        with pytest.raises(SchemaValidateError):
            validate_inference_config(hunyuan_adapter_for_config_parse, sd_cfg)

    @staticmethod
    def test_get_inference_config_class_returns_hunyuan_inference_config(hunyuan_adapter_for_config_parse):
        assert (
            hunyuan_adapter_for_config_parse.get_inference_config_class()
            is HunyuanVideoModelAdapter.HunyuanVideoInferenceConfig
        )


class TestConfigureRuntime:
    @staticmethod
    def test_configure_runtime_writes_infer_steps_to_model_args_when_yaml_override(adapter, temp_model_dir):
        """configure_runtime 将 InferenceConfig 落到 model_args，不保留 adapter.inference_config 属性。"""
        _ = temp_model_dir

        class _YamlInferenceConfig:
            """仅含 hyvideo CLI 可映射字段，不含 model_resolution 等 configure_runtime 会拒绝的键。"""

            @staticmethod
            def model_dump(exclude_none=True):
                _ = exclude_none
                return {
                    "infer_steps": 55,
                    "batch_size": 2,
                    "video_size": [720, 1280],
                }

        merged_args = argparse.Namespace(
            model_base=str(adapter.model_path),
            ulysses_degree=1,
            ring_degree=1,
            vae_parallel=False,
            video_size=(720, 1280),
            video_length=129,
            infer_steps=55,
            seed=None,
            neg_prompt=None,
            cfg_scale=1.0,
            num_videos=1,
            flow_shift=7.0,
            batch_size=2,
            embedded_cfg_scale=6.0,
        )
        allowed_keys = frozenset(
            {
                "infer_steps",
                "batch_size",
                "video_size",
                "video_length",
                "seed",
                "neg_prompt",
                "cfg_scale",
                "num_videos",
                "flow_shift",
                "embedded_cfg_scale",
                "model_base",
                "ulysses_degree",
                "ring_degree",
                "vae_parallel",
            },
        )
        with (
            patch.object(adapter, "_allowed_hyvideo_config_keys", return_value=allowed_keys),
            patch.object(adapter, "_parse_args_from_hyvideo", return_value=merged_args) as mock_parse,
        ):
            adapter.configure_runtime(_YamlInferenceConfig())
            assert mock_parse.call_count == 1
            cli_args = mock_parse.call_args[0][0]
            assert "--infer-steps" in cli_args
            assert "55" in cli_args
        assert adapter.model_args.infer_steps == 55
        assert adapter.model_args.batch_size == 2
        assert adapter.model_args.task_config == TASK_TYPE
        assert not hasattr(adapter, "inference_config")


class TestRuntimeValue:
    def test_runtime_value_reads_from_inference_config_model_when_pydantic(self, adapter):
        adapter.model_args.infer_steps = 10
        cfg = HunyuanVideoModelAdapter.HunyuanVideoInferenceConfig(infer_steps=55)
        assert adapter._runtime_value(cfg, "infer_steps") == 55

    def test_runtime_value_reads_from_dict_when_inference_config_is_mapping(self, adapter):
        adapter.model_args.infer_steps = 10
        assert adapter._runtime_value({"infer_steps": 55}, "infer_steps") == 55

    def test_runtime_value_falls_back_to_model_args_when_inference_config_is_none(self, adapter):
        adapter.model_args.infer_steps = 10
        assert adapter._runtime_value(None, "infer_steps") == 10

    def test_runtime_value_falls_back_to_model_args_when_key_missing_in_dict(self, adapter):
        adapter.model_args.infer_steps = 10
        assert adapter._runtime_value({}, "infer_steps") == 10


# ------------------------------ 校准推理测试 ------------------------------
class TestInferenceDumpCalibData:
    @staticmethod
    def test_inference_dump_calib_data_success():
        """测试校准 dump 推理成功执行（使用 __new__ 实例以调用真实的 _runtime_value）。"""
        inference_config = HunyuanVideoModelAdapter.HunyuanVideoInferenceConfig(
            video_size=(720, 1280),
            infer_steps=50,
            seed=42,
        )
        dataset = [VlmCalibSample(text="calib prompt")]
        host = HunyuanVideoModelAdapter.__new__(HunyuanVideoModelAdapter)
        host.model_args = argparse.Namespace(
            video_size=(720, 1280),
            video_length=129,
            seed=42,
            neg_prompt=None,
            infer_steps=50,
            cfg_scale=1.0,
            num_videos=1,
            flow_shift=7.0,
            batch_size=1,
            embedded_cfg_scale=6.0,
        )
        host.hunyuan_video_sampler = Mock()
        host.hunyuan_video_sampler.predict = Mock(return_value=Mock())
        with (
            patch("msmodelslim.model.hunyuan_video.model_adapter.torch") as mock_torch,
            patch("msmodelslim.model.hunyuan_video.model_adapter.tqdm") as mock_tqdm,
            patch("msmodelslim.model.hunyuan_video.model_adapter.time") as mock_time,
            patch("msmodelslim.model.hunyuan_video.model_adapter.logging"),
        ):
            mock_tqdm.return_value.__iter__.return_value = dataset
            mock_time.time.side_effect = [1.0, 3.0]
            mock_stream = Mock()
            mock_torch.npu.Stream.return_value = mock_stream
            HunyuanVideoModelAdapter.inference_dump_calib_data(
                host,
                dataset=dataset,
                inference_config=inference_config,
            )
        host.hunyuan_video_sampler.predict.assert_called_once()
        _, kwargs = host.hunyuan_video_sampler.predict.call_args
        assert kwargs["prompt"] == "calib prompt"

    @staticmethod
    def test_inference_dump_empty_dataset_no_op_when_dataset_empty():
        """空 dataset 时 inference_dump_calib_data 不调用 predict（样本校验在 handle_dataset 阶段）。"""
        inference_config = HunyuanVideoModelAdapter.HunyuanVideoInferenceConfig(
            video_size=(720, 1280),
        )
        host = HunyuanVideoModelAdapter.__new__(HunyuanVideoModelAdapter)
        host.model_args = argparse.Namespace(video_size=(720, 1280), video_length=129)
        host.hunyuan_video_sampler = Mock()
        host.hunyuan_video_sampler.predict = Mock(return_value=Mock())
        with (
            patch("msmodelslim.model.hunyuan_video.model_adapter.torch") as mock_torch,
            patch("msmodelslim.model.hunyuan_video.model_adapter.tqdm") as mock_tqdm,
            patch("msmodelslim.model.hunyuan_video.model_adapter.time"),
            patch("msmodelslim.model.hunyuan_video.model_adapter.logging"),
        ):
            mock_tqdm.return_value.__iter__.return_value = iter([])
            mock_torch.npu.Stream.return_value = Mock()
            host.inference_dump_calib_data(dataset=[], inference_config=inference_config)
        host.hunyuan_video_sampler.predict.assert_not_called()

    @pytest.fixture
    def mock_self(self):
        mock = Mock()
        mock.model_args = argparse.Namespace(
            video_size=(720, 1280),
            video_length=129,
            seed=42,
            neg_prompt=None,
            infer_steps=50,
            cfg_scale=1.0,
            num_videos=1,
            flow_shift=7.0,
            batch_size=1,
            embedded_cfg_scale=6.0,
        )
        mock.hunyuan_video_sampler = Mock()
        mock.hunyuan_video_sampler.predict = Mock(return_value=Mock())
        return mock

    @pytest.fixture(autouse=True)
    def mock_dependencies(self):
        """模拟依赖"""
        with (
            patch('msmodelslim.model.hunyuan_video.model_adapter.torch') as mock_torch,
            patch('msmodelslim.model.hunyuan_video.model_adapter.tqdm') as mock_tqdm,
            patch('msmodelslim.model.hunyuan_video.model_adapter.time') as mock_time,
            patch('msmodelslim.model.hunyuan_video.model_adapter.logging'),
        ):
            mock_tqdm.return_value.__iter__.return_value = [1]
            mock_time.time.side_effect = [1.0, 3.0]  # begin=1.0, end=3.0
            mock_stream = Mock()
            mock_torch.npu.Stream.return_value = mock_stream

            yield


# ------------------------------ 量化应用测试 ------------------------------
class TestQuantizationContext:
    @staticmethod
    def test_quantization_context_success(mock_self, process_func):
        """测试量化上下文成功"""

        class MockNoSync:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

        mock_self.no_sync = Mock(return_value=MockNoSync())
        mock_self.transformer = Mock()
        mock_self.transformer.named_modules.return_value = []

        with patch('torch.cuda.amp.autocast'):
            with HunyuanVideoModelAdapter.quantization_context(mock_self):
                process_func()

        mock_self.transformer.named_modules.assert_called_once()
        process_func.assert_called_once()

    @pytest.fixture
    def mock_self(self):
        """创建包含transformer结构的模拟self对象"""
        mock = Mock()

        # 创建模拟的transformer模块
        module_embedding = Mock()
        module_block = Mock()
        module_norm = Mock()

        transformer = Mock()
        transformer.named_modules.return_value = [
            ('embedding', module_embedding),
            ('blocks.0', module_block),
            ('norm', module_norm),
        ]
        mock.transformer = transformer

        return mock

    @pytest.fixture
    def process_func(self):
        """模拟的处理函数"""
        return Mock()


# ------------------------------ Cache设置测试 ------------------------------
class TestSetupCache:
    @staticmethod
    def test_setup_cache_basic(mock_self):
        """测试基础cache设置"""
        with patch('mindiesd.CacheConfig') as mock_cache_config, patch('mindiesd.CacheAgent') as mock_cache_agent:
            # 确保CacheConfig可以被实例化
            mock_cache_config.return_value = MagicMock()
            mock_cache_agent.return_value = MagicMock()

            HunyuanVideoModelAdapter._setup_cache(mock_self)

            # 验证CacheConfig被调用
            assert mock_cache_config.call_count >= 2

    @staticmethod
    def test_setup_cache_with_attention_cache(mock_self):
        """测试启用attention cache"""
        mock_self.model_args.use_attentioncache = True
        mock_self.model_args.start_step = 9
        mock_self.model_args.attentioncache_interval = 3
        mock_self.model_args.end_step = 47

        with patch('mindiesd.CacheConfig') as mock_cache_config, patch('mindiesd.CacheAgent') as mock_cache_agent:
            # 确保CacheConfig可以被实例化
            mock_cache_config.return_value = MagicMock()
            mock_cache_agent.return_value = MagicMock()

            HunyuanVideoModelAdapter._setup_cache(mock_self)

            # 验证CacheConfig被调用时包含attention cache参数
            call_args_list = mock_cache_config.call_args_list
            attention_cache_calls = [call for call in call_args_list if call[1].get('method') == 'attention_cache']
            assert len(attention_cache_calls) >= 2

    @pytest.fixture
    def mock_self(self):
        """创建模拟的self对象"""
        mock = Mock()
        mock.model_args = Mock()
        mock.model_args.use_cache = False
        mock.model_args.use_cache_double = False
        mock.model_args.use_attentioncache = False
        mock.model_args.infer_steps = 50

        # 模拟transformer结构
        mock.transformer = Mock()
        mock.transformer.single_blocks = [Mock() for _ in range(10)]
        mock.transformer.double_blocks = [Mock() for _ in range(10)]

        # 模拟mindiesd模块
        mock_mindiesd = Mock()
        mock_mindiesd.CacheConfig = Mock()
        mock_mindiesd.CacheAgent = Mock()

        # 使用patch.dict模拟sys.modules
        with patch.dict('sys.modules', {'mindiesd': mock_mindiesd}):
            yield mock


class TestHunyuanVideoPrepareCalibData:
    """HunyuanVideoModelAdapter.prepare_calib_data"""

    @staticmethod
    def test_prepare_calib_data_returns_none_per_expert_when_enable_dump_false_and_user_confirms(tmp_path):
        from msmodelslim.core.quant_service.multimodal_sd_v1.quant_config import DumpConfig

        adapter = HunyuanVideoModelAdapter.__new__(HunyuanVideoModelAdapter)
        adapter.model_args = MagicMock(task_config=TASK_TYPE)
        models = {"transformer": MagicMock()}
        dump_config = DumpConfig(enable_dump=False)

        with patch("builtins.input", return_value="y"):
            with patch(
                "msmodelslim.model.hunyuan_video.model_adapter.load_cached_data_for_models",
            ) as mock_load:
                result = adapter.prepare_calib_data(
                    models=models,
                    dump_config=dump_config,
                    save_path=Path(tmp_path),
                    dataset=[VlmCalibSample(text="hello")],
                    inference_config=None,
                )

        mock_load.assert_not_called()
        assert result == {"transformer": None}

    @staticmethod
    def test_prepare_calib_data_raises_unsupported_error_when_enable_dump_false_and_user_declines(tmp_path):
        from msmodelslim.core.quant_service.multimodal_sd_v1.quant_config import DumpConfig

        adapter = HunyuanVideoModelAdapter.__new__(HunyuanVideoModelAdapter)
        adapter.model_args = MagicMock(task_config=TASK_TYPE)
        models = {"transformer": MagicMock()}
        dump_config = DumpConfig(enable_dump=False)

        with patch("builtins.input", return_value="n"):
            with pytest.raises(UnsupportedError):
                adapter.prepare_calib_data(
                    models=models,
                    dump_config=dump_config,
                    save_path=Path(tmp_path),
                    dataset=[],
                    inference_config=None,
                )


# ------------------------------ generate_model_forward 补充测试 ------------------------------
class TestGenerateModelForwardExtended:
    """generate_model_forward 补充测试"""

    @staticmethod
    def test_raises_when_no_streamblock_found(adapter):
        """没有streamblock时应抛出InvalidModelError"""
        model = MagicMock()
        model.named_modules.return_value = [("embedding", MagicMock())]

        with patch("msmodelslim.model.hunyuan_video.model_adapter.TransformersForwardBreak", Exception):
            with pytest.raises(Exception):
                list(adapter.generate_model_forward(model, {"input": torch.randn(1, 10)}))

    @staticmethod
    def test_handles_single_tensor_input(adapter):
        """单tensor输入应被正确处理"""
        model = MagicMock()
        streamblock = MagicMock()
        type(streamblock).__name__ = "StreamBlock"
        model.named_modules.return_value = [("blocks.0", streamblock)]

        # 模拟hook触发TransformersForwardBreak
        def forward_side_effect(*args, **kwargs):
            return torch.randn(1, 10, 64)

        model.side_effect = forward_side_effect

        with (
            patch("msmodelslim.model.hunyuan_video.model_adapter.to_device", side_effect=lambda x, _: x),
            patch("msmodelslim.model.hunyuan_video.model_adapter.TransformersForwardBreak", Exception),
            patch("msmodelslim.model.hunyuan_video.model_adapter.dist") as mock_dist,
        ):
            mock_dist.is_initialized.return_value = False
            gen = adapter.generate_model_forward(model, torch.randn(1, 10, 64))
            # 应该能创建generator
            assert gen is not None


# ------------------------------ validate_calib_samples 补充测试 ------------------------------
class TestValidateCalibSamples:
    """validate_calib_samples 边界场景"""

    @staticmethod
    def test_validate_calib_samples_passes_valid(adapter):
        """有效样本应原样返回"""
        samples = [VlmCalibSample(text="hello"), VlmCalibSample(text="world")]
        result = adapter.validate_calib_samples(samples)
        assert result == samples

    @staticmethod
    def test_validate_calib_samples_raises_when_text_is_empty(adapter):
        """空text应抛出SchemaValidateError"""
        with pytest.raises(SchemaValidateError, match="non-empty text"):
            adapter.validate_calib_samples([VlmCalibSample(text="")])

    @staticmethod
    def test_validate_calib_samples_raises_when_text_is_whitespace(adapter):
        """纯空格text应抛出SchemaValidateError"""
        with pytest.raises(SchemaValidateError, match="non-empty text"):
            adapter.validate_calib_samples([VlmCalibSample(text="   ")])

    @staticmethod
    def test_validate_calib_samples_raises_when_image_present(adapter):
        """带image应抛出SchemaValidateError"""
        with pytest.raises(SchemaValidateError, match="must not include image"):
            adapter.validate_calib_samples([VlmCalibSample(text="hi", image="/tmp/a.png")])

    @staticmethod
    def test_validate_calib_samples_raises_on_second_sample(adapter):
        """第二个样本有问题应报index=1"""
        samples = [VlmCalibSample(text="ok"), VlmCalibSample(text="", image=None)]
        with pytest.raises(SchemaValidateError, match="sample\\[1\\]"):
            adapter.validate_calib_samples(samples)


# ------------------------------ handle_dataset 补充测试 ------------------------------
class TestHandleDataset:
    """handle_dataset 边界场景"""

    @staticmethod
    def test_handle_dataset_returns_empty_when_none(adapter):
        """None输入返回空列表"""
        assert adapter.handle_dataset(None) == []

    @staticmethod
    def test_handle_dataset_wraps_single_sample(adapter):
        """单个VlmCalibSample应被包装为列表"""
        sample = VlmCalibSample(text="single")
        result = adapter.handle_dataset(sample)
        assert len(result) == 1
        assert result[0].text == "single"


# ------------------------------ get_online_rotation_configs 测试 ------------------------------
class TestGetOnlineRotationConfigs:
    """get_online_rotation_configs 测试"""

    @staticmethod
    def test_returns_empty_when_no_model_and_no_transformer(adapter):
        """无model且无transformer时返回空"""
        adapter.transformer = None
        result = adapter.get_online_rotation_configs(model=None)
        assert result == {}

    @staticmethod
    def test_returns_empty_when_model_missing_hidden_size(adapter):
        """model缺少hidden_size时返回空"""
        model = MagicMock()
        model.hidden_size = 64
        # 缺少heads_num
        del model.heads_num
        model.named_modules.return_value = []
        result = adapter.get_online_rotation_configs(model=model)
        assert result == {}

    @staticmethod
    def test_returns_empty_when_model_missing_heads_num(adapter):
        """model缺少heads_num时返回空"""
        model = MagicMock()
        del model.hidden_size
        model.heads_num = 8
        model.named_modules.return_value = []
        result = adapter.get_online_rotation_configs(model=model)
        assert result == {}

    @staticmethod
    def test_registers_identity_modules_for_double_stream_block(adapter):
        """MMDoubleStreamBlock应被注册q_rot和k_rot"""
        model = MagicMock()
        model.hidden_size = 64
        model.heads_num = 8

        block = MagicMock()
        block.__class__.__name__ = "MMDoubleStreamBlock"
        del block.q_rot
        del block.k_rot
        model.named_modules.return_value = [("blocks.0", block)]

        result = adapter.get_online_rotation_configs(model=model)
        assert "blocks.0.q_rot" in result
        assert "blocks.0.k_rot" in result

    @staticmethod
    def test_registers_identity_modules_for_single_stream_block(adapter):
        """MMSingleStreamBlock应被注册q_rot和k_rot"""
        model = MagicMock()
        model.hidden_size = 64
        model.heads_num = 8

        block = MagicMock()
        block.__class__.__name__ = "MMSingleStreamBlock"
        del block.q_rot
        del block.k_rot
        model.named_modules.return_value = [("blocks.0", block)]

        result = adapter.get_online_rotation_configs(model=model)
        assert "blocks.0.q_rot" in result
        assert "blocks.0.k_rot" in result

    @staticmethod
    def test_skips_non_target_module(adapter):
        """非目标模块类型应被跳过"""
        model = MagicMock()
        model.hidden_size = 64
        model.heads_num = 8

        block = MagicMock()
        block.__class__.__name__ = "SomeOtherBlock"
        model.named_modules.return_value = [("blocks.0", block)]

        result = adapter.get_online_rotation_configs(model=model)
        assert len(result) == 0

    @staticmethod
    def test_handles_empty_root_name(adapter):
        """空name时q_rot/k_rot路径不带前缀点"""
        model = MagicMock()
        model.hidden_size = 64
        model.heads_num = 8

        block = MagicMock()
        block.__class__.__name__ = "MMDoubleStreamBlock"
        del block.q_rot
        del block.k_rot
        model.named_modules.return_value = [("", block)]

        result = adapter.get_online_rotation_configs(model=model)
        assert "q_rot" in result
        assert "k_rot" in result

    @staticmethod
    def test_uses_transformer_when_model_none(adapter):
        """model=None时使用adapter.transformer"""
        mock_transformer = MagicMock()
        mock_transformer.hidden_size = 64
        mock_transformer.heads_num = 8
        block = MagicMock()
        block.__class__.__name__ = "MMDoubleStreamBlock"
        del block.q_rot
        del block.k_rot
        mock_transformer.named_modules.return_value = [("blocks.0", block)]
        adapter.transformer = mock_transformer

        result = adapter.get_online_rotation_configs(model=None)
        assert "blocks.0.q_rot" in result


# ------------------------------ inject_fa3_placeholders 测试 ------------------------------
class TestInjectFA3Placeholders:
    """inject_fa3_placeholders 测试"""

    @staticmethod
    def test_injects_placeholders_for_double_stream_block(adapter):
        """MMDoubleStreamBlock应被注入fa3_q/k/v"""
        root_module = MagicMock()
        block = MagicMock()
        block.__class__.__name__ = "MMDoubleStreamBlock"
        block.forward.__module__ = "hyvideo.modules.models"
        root_module.named_modules.return_value = [("blocks.0", block)]

        # mock import_module返回的模块
        mock_hyvideo_mod = MagicMock()
        with patch("msmodelslim.model.hunyuan_video.model_adapter.import_module", return_value=mock_hyvideo_mod):
            adapter.inject_fa3_placeholders("root", root_module, should_inject=lambda name: True)

        # 验证set_submodule被调用注入fa3占位符
        calls = [str(c) for c in root_module.set_submodule.call_args_list]
        assert any("fa3_q" in c for c in calls)
        assert any("fa3_k" in c for c in calls)
        assert any("fa3_v" in c for c in calls)

    @staticmethod
    def test_injects_placeholders_for_single_stream_block(adapter):
        """MMSingleStreamBlock应被注入fa3_q/k/v"""
        root_module = MagicMock()
        block = MagicMock()
        block.__class__.__name__ = "MMSingleStreamBlock"
        block.forward.__module__ = "hyvideo.modules.models"
        root_module.named_modules.return_value = [("blocks.0", block)]

        mock_hyvideo_mod = MagicMock()
        with patch("msmodelslim.model.hunyuan_video.model_adapter.import_module", return_value=mock_hyvideo_mod):
            adapter.inject_fa3_placeholders("root", root_module, should_inject=lambda name: True)

        calls = [str(c) for c in root_module.set_submodule.call_args_list]
        assert any("fa3_q" in c for c in calls)

    @staticmethod
    def test_skips_non_target_module(adapter):
        """非目标模块类型应被跳过"""
        root_module = MagicMock()
        block = MagicMock()
        block.__class__.__name__ = "SomeOtherBlock"
        root_module.named_modules.return_value = [("blocks.0", block)]

        adapter.inject_fa3_placeholders("root", root_module, should_inject=lambda name: True)
        root_module.set_submodule.assert_not_called()

    @staticmethod
    def test_skips_when_should_inject_returns_false(adapter):
        """should_inject返回False时跳过"""
        root_module = MagicMock()
        block = MagicMock()
        block.__class__.__name__ = "MMDoubleStreamBlock"
        root_module.named_modules.return_value = [("blocks.0", block)]

        adapter.inject_fa3_placeholders("root", root_module, should_inject=lambda name: False)
        root_module.set_submodule.assert_not_called()

    @staticmethod
    def test_handles_empty_name(adapter):
        """空name时prefix为空"""
        root_module = MagicMock()
        block = MagicMock()
        block.__class__.__name__ = "MMDoubleStreamBlock"
        block.forward.__module__ = "hyvideo.modules.models"
        root_module.named_modules.return_value = [("", block)]

        mock_hyvideo_mod = MagicMock()
        with patch("msmodelslim.model.hunyuan_video.model_adapter.import_module", return_value=mock_hyvideo_mod):
            adapter.inject_fa3_placeholders("root", root_module, should_inject=lambda name: True)

        # 验证set_submodule被调用（不带前缀点）
        root_module.set_submodule.assert_called()


# ------------------------------ configure_runtime 补充测试 ------------------------------
class TestConfigureRuntimeExtended:
    """configure_runtime 补充测试"""

    @staticmethod
    def test_raises_when_unknown_attrs(adapter):
        """未知属性应抛出SchemaValidateError"""

        class _Config:
            @staticmethod
            def model_dump(exclude_none=True):
                return {"unknown_key": 123}

        with patch.object(adapter, "_allowed_hyvideo_config_keys", return_value=frozenset({"infer_steps"})):
            with pytest.raises(SchemaValidateError, match="illegal config attributes"):
                adapter.configure_runtime(_Config())


# ------------------------------ prepare_calib_data 补充测试 ------------------------------
class TestPrepareCalibDataExtended:
    """prepare_calib_data enable_dump=True 分支"""

    @staticmethod
    def test_prepare_calib_data_loads_cached_when_enable_dump_true(tmp_path):
        from msmodelslim.core.quant_service.multimodal_sd_v1.quant_config import DumpConfig

        adapter = HunyuanVideoModelAdapter.__new__(HunyuanVideoModelAdapter)
        adapter.model_args = MagicMock(task_config=TASK_TYPE)
        models = {"transformer": MagicMock()}
        dump_config = DumpConfig(enable_dump=True, dump_data_dir=str(tmp_path))

        with patch(
            "msmodelslim.model.hunyuan_video.model_adapter.load_cached_data_for_models",
            return_value={"transformer": "cached_data"},
        ) as mock_load:
            result = adapter.prepare_calib_data(
                models=models,
                dump_config=dump_config,
                save_path=tmp_path,
                dataset=[VlmCalibSample(text="hello")],
                inference_config=None,
            )
        mock_load.assert_called_once()
        assert result == {"transformer": "cached_data"}


# ------------------------------ _allowed_hyvideo_config_keys 补充测试 ------------------------------
class TestAllowedHyvideoConfigKeys:
    """_allowed_hyvideo_config_keys 缓存行为"""

    @staticmethod
    def test_caches_result_on_second_call(adapter):
        """第二次调用应使用缓存"""
        mock_probe = argparse.Namespace(infer_steps=50, batch_size=1)
        # 清除类缓存
        HunyuanVideoModelAdapter._HYVIDEO_CONFIG_KEYS = None
        with patch.object(adapter, "_parse_args_from_hyvideo", return_value=mock_probe) as mock_parse:
            keys1 = adapter._allowed_hyvideo_config_keys()
            keys2 = adapter._allowed_hyvideo_config_keys()
        # parse_args只应被调用一次（缓存）
        mock_parse.assert_called_once()
        assert keys1 == keys2


# ------------------------------ _parse_args_from_hyvideo 补充测试 ------------------------------
class TestParseArgsFromHyvideo:
    """_parse_args_from_hyvideo 测试"""

    @staticmethod
    def test_restores_sys_argv_after_call(adapter):
        """调用后应恢复sys.argv"""
        original_argv = sys.argv.copy()
        mock_parse = MagicMock(return_value=argparse.Namespace())
        with patch.dict(sys.modules, {"hyvideo.config": MagicMock(parse_args=mock_parse)}):
            adapter._parse_args_from_hyvideo(["--prompt", "test"])
        assert sys.argv == original_argv

    @staticmethod
    def test_passes_correct_argv(adapter):
        """应传递正确的argv"""
        mock_parse = MagicMock(return_value=argparse.Namespace())
        with patch.dict(sys.modules, {"hyvideo.config": MagicMock(parse_args=mock_parse)}):
            adapter._parse_args_from_hyvideo(["--prompt", "test", "--seed", "42"])
        mock_parse.assert_called_once()


# ------------------------------ _setup_cache 补充测试 ------------------------------
class TestSetupCacheExtended:
    """_setup_cache 补充分支测试"""

    @pytest.fixture
    def cache_mock_self(self, temp_model_dir):
        """创建用于cache测试的mock对象"""
        mock = Mock()
        mock.model_args = Mock()
        mock.model_args.use_cache = False
        mock.model_args.use_cache_double = False
        mock.model_args.use_attentioncache = False
        mock.model_args.infer_steps = 50
        mock.transformer = Mock()
        mock.transformer.single_blocks = [Mock() for _ in range(10)]
        mock.transformer.double_blocks = [Mock() for _ in range(10)]
        # 模拟mindiesd模块
        mock_mindiesd = Mock()
        mock_mindiesd.CacheConfig = Mock()
        mock_mindiesd.CacheAgent = Mock()
        with patch.dict('sys.modules', {'mindiesd': mock_mindiesd}):
            yield mock

    @staticmethod
    def test_setup_cache_with_use_cache_true(cache_mock_self):
        """use_cache=True时应配置single block cache"""
        cache_mock_self.model_args.use_cache = True
        cache_mock_self.model_args.cache_start_steps = 0
        cache_mock_self.model_args.cache_interval = 5
        cache_mock_self.model_args.single_block_start = 0
        cache_mock_self.model_args.single_block_end = 9

        with patch('mindiesd.CacheConfig') as mock_cfg, patch('mindiesd.CacheAgent') as mock_agent:
            mock_cfg.return_value = MagicMock()
            mock_agent.return_value = MagicMock()
            HunyuanVideoModelAdapter._setup_cache(cache_mock_self)

        calls = [str(c) for c in mock_cfg.call_args_list]
        assert any("dit_block_cache" in c for c in calls)

    @staticmethod
    def test_setup_cache_with_use_cache_double_true(cache_mock_self):
        """use_cache_double=True时应配置double block cache"""
        cache_mock_self.model_args.use_cache_double = True
        cache_mock_self.model_args.cache_start_steps = 0
        cache_mock_self.model_args.cache_interval = 5
        cache_mock_self.model_args.double_block_start = 0
        cache_mock_self.model_args.double_block_end = 9

        with patch('mindiesd.CacheConfig') as mock_cfg, patch('mindiesd.CacheAgent') as mock_agent:
            mock_cfg.return_value = MagicMock()
            mock_agent.return_value = MagicMock()
            HunyuanVideoModelAdapter._setup_cache(cache_mock_self)

        calls = [str(c) for c in mock_cfg.call_args_list]
        assert any("dit_block_cache" in c for c in calls)

    @staticmethod
    def test_setup_cache_with_empty_single_blocks(cache_mock_self):
        """single_blocks为空时跳过single cache配置"""
        cache_mock_self.transformer.single_blocks = []
        cache_mock_self.transformer.double_blocks = [Mock() for _ in range(5)]

        with patch('mindiesd.CacheConfig') as mock_cfg, patch('mindiesd.CacheAgent') as mock_agent:
            mock_cfg.return_value = MagicMock()
            mock_agent.return_value = MagicMock()
            HunyuanVideoModelAdapter._setup_cache(cache_mock_self)

        assert mock_agent.call_count >= 1

    @staticmethod
    def test_setup_cache_with_empty_double_blocks(cache_mock_self):
        """double_blocks为空时跳过double cache配置"""
        cache_mock_self.transformer.single_blocks = [Mock() for _ in range(5)]
        cache_mock_self.transformer.double_blocks = []

        with patch('mindiesd.CacheConfig') as mock_cfg, patch('mindiesd.CacheAgent') as mock_agent:
            mock_cfg.return_value = MagicMock()
            mock_agent.return_value = MagicMock()
            HunyuanVideoModelAdapter._setup_cache(cache_mock_self)

        assert mock_agent.call_count >= 1


# ------------------------------ quantization_context 补充测试 ------------------------------
class TestQuantizationContextExtended:
    """quantization_context 分支测试"""

    @staticmethod
    def test_moves_non_blocks_to_npu_and_blocks_to_cpu(adapter):
        """非blocks模块移到npu，blocks模块移到cpu"""
        module_embedding = Mock()
        module_block = Mock()
        module_norm = Mock()

        mock_transformer = Mock()
        mock_transformer.named_modules.return_value = [
            ('embedding', module_embedding),
            ('blocks.0', module_block),
            ('norm', module_norm),
        ]
        adapter.transformer = mock_transformer

        with patch('torch.cuda.amp.autocast'):
            with adapter.quantization_context():
                pass

        module_embedding.to.assert_called_with('npu')
        module_block.to.assert_called_with('cpu')
        module_norm.to.assert_called_with('npu')


# ------------------------------ _setup_cache ImportError 测试 ------------------------------
class TestSetupCacheImportError:
    """_setup_cache mindiesd导入失败分支"""

    @staticmethod
    def test_raises_import_error_when_mindiesd_missing(adapter):
        """mindiesd不可用时应抛出ImportError"""
        adapter.model_args = Mock()
        adapter.model_args.use_cache = False
        adapter.model_args.use_cache_double = False
        adapter.model_args.use_attentioncache = False
        adapter.transformer = Mock()
        adapter.transformer.single_blocks = []
        adapter.transformer.double_blocks = []

        # 确保mindiesd不在sys.modules中
        with patch.dict('sys.modules', {'mindiesd': None}):
            with pytest.raises(ImportError, match="mindiesd"):
                adapter._setup_cache()


# ------------------------------ get_online_rotation_configs warning分支测试 ------------------------------
class TestGetOnlineRotationConfigsWarning:
    """get_online_rotation_configs warning分支"""

    @staticmethod
    def test_logs_warning_when_register_fails(adapter):
        """注册rotation模块失败时应记录warning"""
        model = MagicMock()
        model.hidden_size = 64
        model.heads_num = 8

        block = MagicMock()
        block.__class__.__name__ = "MMDoubleStreamBlock"
        # 确保hasattr返回False，触发register_module调用
        del block.q_rot
        del block.k_rot
        block.register_module.side_effect = RuntimeError("register failed")
        model.named_modules.return_value = [("blocks.0", block)]

        with patch("msmodelslim.model.hunyuan_video.model_adapter.get_logger") as mock_logger:
            adapter.get_online_rotation_configs(model=model)
        # 应该有warning日志
        mock_logger.return_value.warning.assert_called()


# ------------------------------ _fixed_quant_runtime_overrides 测试 ------------------------------
class TestFixedQuantRuntimeOverrides:
    """_fixed_quant_runtime_overrides 测试"""

    @staticmethod
    def test_returns_expected_keys(adapter):
        """返回的字典包含预期的key"""
        result = adapter._fixed_quant_runtime_overrides()
        assert result["ulysses_degree"] == 1
        assert result["ring_degree"] == 1
        assert result["vae_parallel"] is False
        assert result["use_cache"] is False
        assert result["use_cache_double"] is False
        assert result["use_attentioncache"] is False


# ------------------------------ _calib_data_when_dump_disabled 补充测试 ------------------------------
class TestCalibDataWhenDumpDisabled:
    """_calib_data_when_dump_disabled 补充测试"""

    @staticmethod
    def test_raises_unsupported_error_when_user_declines(adapter):
        """用户输入非y时应抛出UnsupportedError"""
        with patch("builtins.input", return_value="n"):
            with pytest.raises(UnsupportedError):
                adapter._calib_data_when_dump_disabled({"model": MagicMock()})

    @staticmethod
    def test_returns_none_per_expert_when_user_confirms(adapter):
        """用户输入y时返回每个expert的None"""
        with patch("builtins.input", return_value="y"):
            result = adapter._calib_data_when_dump_disabled({"m1": MagicMock(), "m2": MagicMock()})
        assert result == {"m1": None, "m2": None}


# ------------------------------ 依赖检查测试 ------------------------------
class TestCheckImportDependency:
    @staticmethod
    def test_successful_import(mock_self):
        """测试依赖导入成功"""
        # 由于使用了fixture模拟，导入应该成功
        HunyuanVideoModelAdapter._check_import_dependency(mock_self)

    @staticmethod
    def test_import_failure(mock_self, monkeypatch):
        """测试依赖导入失败"""
        # 临时移除hyvideo模块
        original_hyvideo = sys.modules.get('hyvideo')
        if 'hyvideo' in sys.modules:
            monkeypatch.delitem(sys.modules, 'hyvideo')

        try:
            with pytest.raises(ImportError) as exc_info:
                HunyuanVideoModelAdapter._check_import_dependency(mock_self)
            assert "Failed to import required components from hunyuanvideo" in str(exc_info.value)
        finally:
            # 恢复模块
            if original_hyvideo is not None:
                sys.modules['hyvideo'] = original_hyvideo

    @pytest.fixture
    def mock_self(self):
        return Mock()
