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

# pylint: disable=redefined-outer-name,too-many-lines,use-maxsplit-arg

import sys
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
import pytest
import torch

from msmodelslim.model.wan2_2.model_adapter import (
    Wan2Point2Adapter,
    InvalidModelError,
    SchemaValidateError,
    UnsupportedError,
)


@pytest.fixture(autouse=True)
def mock_wan_modules(monkeypatch):
    """统一模拟所有wan相关模块，所有测试类共用"""
    mock_modules = [
        'wan',
        'wan.configs',
        'wan.utils.prompt_extend',
        'wan.utils.utils',
        'wan.distributed.parallel_mgr',
        'wan.distributed.tp_applicator',
        'wan.distributed',
        'wan.utils',
    ]

    original_modules = {mod: sys.modules.get(mod) for mod in mock_modules}
    for module_path in mock_modules:
        sys.modules[module_path] = MagicMock()

    # 配置关键模块
    wan_configs = sys.modules['wan.configs']
    wan_configs.WAN_CONFIGS = MagicMock()
    wan_configs.WAN_CONFIGS["test_task"] = Mock(num_heads=8)
    wan_configs.WAN_CONFIGS["divisible_task"] = Mock(num_heads=12)
    wan_configs.SUPPORTED_SIZES = {
        't2v-A14B': ('720*1280', '1280*720', '480*832', '832*480', '432*768', '768*432'),
        'i2v-A14B': ('720*1280', '1280*720', '480*832', '832*480', '432*768', '768*432'),
        'ti2v-5B': ('704*1280', '1280*704'),
    }
    wan_configs.SIZE_CONFIGS = {
        '720*1280': (720, 1280),
        '1280*720': (1280, 720),
        '480*832': (480, 832),
        '832*480': (832, 480),
        '704*1280': (704, 1280),
        '1280*704': (1280, 704),
        '432*768': (432, 768),
        '768*432': (768, 432),
    }

    wan_main = sys.modules['wan']
    mock_wan_i2v = MagicMock()
    mock_wan_i2v.low_noise_model = MagicMock()
    mock_wan_i2v.high_noise_model = MagicMock()
    mock_wan_t2v = MagicMock()
    mock_wan_i2v.low_noise_model = MagicMock()
    mock_wan_i2v.high_noise_model = MagicMock()
    mock_wan_ti2v = MagicMock()
    mock_wan_ti2v.model = MagicMock()
    wan_main.WanI2V.return_value = mock_wan_i2v
    wan_main.WanT2V.return_value = mock_wan_t2v
    wan_main.WanTI2V.return_value = mock_wan_ti2v

    yield

    # 恢复原始模块
    for mod, original in original_modules.items():
        if original is not None:
            sys.modules[mod] = original
        else:
            del sys.modules[mod]


@pytest.fixture
def mock_env(monkeypatch):
    """环境变量模拟，所有测试共用"""
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("LOCAL_RANK", "0")


# ------------------------------ 适配器基础功能测试 ------------------------------
class TestWan2Point2Adapter:
    @staticmethod
    def test_initialization(adapter):
        assert adapter.model_type == "t2v-A14B"
        assert adapter.model_path == Path("/test/model/path")

    @staticmethod
    def test_get_model_info(adapter):
        assert adapter.get_model_type() == "t2v-A14B"
        assert adapter.get_model_pedigree() == "wan2_2"

    @staticmethod
    def test_handle_dataset(adapter):
        mock_dataset = [Mock(), Mock()]
        result = adapter.handle_dataset(mock_dataset)
        assert list(result) == mock_dataset

    @staticmethod
    def test_enable_kv_cache(adapter):
        adapter.enable_kv_cache(Mock(), True)
        assert True  # 方法无异常执行即通过

    @staticmethod
    def test_init_model_returns_transformer():
        mock_self = Mock()

        # 测试非ti2v任务
        mock_self.model_args.task = "t2v-A14B"
        mock_self.low_noise_model = Mock()
        mock_self.high_noise_model = Mock()

        result = Wan2Point2Adapter.init_model(mock_self)
        assert isinstance(result, dict)
        assert 'low_noise_model' in result
        assert 'high_noise_model' in result

    @staticmethod
    def test_init_model_for_ti2v():
        mock_self = Mock()

        # 测试ti2v任务
        mock_self.model_args.task = "ti2v-5B"
        mock_self.transformer = Mock()

        result = Wan2Point2Adapter.init_model(mock_self)
        assert isinstance(result, dict)

    @pytest.fixture
    def adapter(self):
        """当前类专用的适配器实例fixture"""
        with patch("msmodelslim.model.wan2_2.model_adapter.Wan2Point2Adapter._check_import_dependency"):
            adapter_instance = Wan2Point2Adapter("t2v-A14B", Path("/test/model/path"))

        adapter_instance.model_args = MagicMock()
        adapter_instance.model_args.task = "t2v-A14B"
        adapter_instance.model_args.size = "1280*720"
        return adapter_instance


# ------------------------------ _load_pipeline方法测试 ------------------------------
class TestLoadPipeline:
    @staticmethod
    def test_t5_fsdp_unsupported(mock_self, mock_env):
        mock_self.model_args.t5_fsdp = True
        with pytest.raises(SchemaValidateError):
            Wan2Point2Adapter._load_pipeline(mock_self)

    @staticmethod
    def test_dit_fsdp_unsupported(mock_self, mock_env):
        mock_self.model_args.dit_fsdp = True
        with pytest.raises(SchemaValidateError):
            Wan2Point2Adapter._load_pipeline(mock_self)

    @staticmethod
    def test_ulysses_size_validation(mock_self, mock_env):
        mock_self.model_args.ulysses_size = 2
        with pytest.raises(SchemaValidateError):
            Wan2Point2Adapter._load_pipeline(mock_self)

    @staticmethod
    def test_vae_parallel_unsupported(mock_self, mock_env):
        mock_self.model_args.vae_parallel = True
        with pytest.raises(SchemaValidateError):
            Wan2Point2Adapter._load_pipeline(mock_self)

    @staticmethod
    def test_load_pipeline_calls_internal(mock_self):
        mock_self._load_pipeline = Mock()
        Wan2Point2Adapter.load_pipeline(mock_self)
        mock_self._load_pipeline.assert_called_once()

    @pytest.fixture
    def mock_self(self):
        mock = Mock()
        mock.model_args = Mock()
        mock.model_args.t5_fsdp = None
        mock.model_args.dit_fsdp = None
        mock.model_args.vae_parallel = None
        mock.model_args.cfg_size = 0
        mock.model_args.ulysses_size = 0
        mock.model_args.ring_size = 0
        mock.model_args.tp_size = 0
        mock.model_args.task = "t2v-A14B"
        mock.model_args.use_attentioncache = False
        mock._check_import_dependency = Mock()
        mock._init_logging = Mock()
        return mock

    @pytest.fixture(autouse=True)
    def mock_all_dependencies(self, monkeypatch):
        # 模拟mindiesd模块
        mock_mindiesd = Mock()
        mock_mindiesd.CacheConfig = Mock()
        mock_mindiesd.CacheAgent = Mock()

        # 使用patch.dict模拟sys.modules
        with patch.dict('sys.modules', {'mindiesd': mock_mindiesd}):
            yield


# ------------------------------ _validate_args方法测试 ------------------------------
class TestValidateArgs:
    @staticmethod
    def test_valid_base_case(mock_self, base_args):
        with patch('random.randint', return_value=42):
            Wan2Point2Adapter._validate_args(mock_self, base_args)

    @staticmethod
    def test_missing_ckpt_dir(mock_self, base_args):
        base_args.ckpt_dir = None
        with pytest.raises(InvalidModelError) as exc_info:
            Wan2Point2Adapter._validate_args(mock_self, base_args)
        assert "Please specify the checkpoint directory" in str(exc_info.value)

    @staticmethod
    def test_invalid_task_type(mock_self, base_args):
        base_args.task = 123
        with pytest.raises(SchemaValidateError) as exc_info:
            Wan2Point2Adapter._validate_args(mock_self, base_args)
        assert "task must be a str" in str(exc_info.value)

    @staticmethod
    def test_unsupported_task(mock_self, base_args):
        base_args.task = "invalid-task"
        with pytest.raises(UnsupportedError) as exc_info:
            Wan2Point2Adapter._validate_args(mock_self, base_args)
        assert "Unsupported task" in str(exc_info.value)

    @staticmethod
    def test_prompt_auto_set_from_example(mock_self, base_args):
        base_args.prompt = None
        Wan2Point2Adapter._validate_args(mock_self, base_args)
        assert base_args.prompt == "test prompt for t2v"

    @staticmethod
    def test_image_auto_set_from_example_for_i2v(mock_self, base_args):
        base_args.task = "i2v-A14B"
        base_args.image = None
        Wan2Point2Adapter._validate_args(mock_self, base_args)
        assert base_args.image == "test_image.jpg"

    @staticmethod
    def test_i2v_missing_image(mock_self, base_args):
        base_args.task = "i2v-A14B"
        base_args.image = None
        with patch.dict(
            'msmodelslim.model.wan2_2.model_adapter.EXAMPLE_PROMPT', {"i2v-A14B": {"prompt": "test", "image": None}}
        ):
            with pytest.raises(SchemaValidateError) as exc_info:
                Wan2Point2Adapter._validate_args(mock_self, base_args)
            assert "Please specify the image path for i2v" in str(exc_info.value)

    @staticmethod
    def test_task_config_set(mock_self, base_args):
        Wan2Point2Adapter._validate_args(mock_self, base_args)
        assert base_args.task_config == "t2v"

    @staticmethod
    def test_param_dtype_set(mock_self, base_args):
        Wan2Point2Adapter._validate_args(mock_self, base_args)
        assert base_args.param_dtype == torch.float16

    @staticmethod
    def test_sample_steps_default(mock_self, base_args):
        base_args.sample_steps = None
        Wan2Point2Adapter._validate_args(mock_self, base_args)
        assert base_args.sample_steps == 50

    @staticmethod
    def test_sample_shift_default(mock_self, base_args):
        base_args.sample_shift = None
        Wan2Point2Adapter._validate_args(mock_self, base_args)
        assert base_args.sample_shift == 5.0

    @staticmethod
    def test_sample_guide_scale_default(mock_self, base_args):
        base_args.sample_guide_scale = None
        Wan2Point2Adapter._validate_args(mock_self, base_args)
        assert base_args.sample_guide_scale == 7.5

    @staticmethod
    def test_frame_num_default(mock_self, base_args):
        base_args.frame_num = None
        Wan2Point2Adapter._validate_args(mock_self, base_args)
        assert base_args.frame_num == 81

    @staticmethod
    def test_base_seed_negative(mock_self, base_args):
        base_args.base_seed = -1
        with patch('random.randint', return_value=999) as mock_randint, patch('sys.maxsize', 1000):
            Wan2Point2Adapter._validate_args(mock_self, base_args)
            assert base_args.base_seed == 999
            mock_randint.assert_called_once_with(0, 1000)

    @staticmethod
    def test_base_seed_non_negative(mock_self, base_args):
        original_seed = base_args.base_seed
        Wan2Point2Adapter._validate_args(mock_self, base_args)
        assert base_args.base_seed == original_seed

    @staticmethod
    def test_invalid_sample_steps_type(mock_self, base_args):
        base_args.sample_steps = "invalid"
        with pytest.raises(SchemaValidateError) as exc_info:
            Wan2Point2Adapter._validate_args(mock_self, base_args)
        assert "sample_steps must be an integer" in str(exc_info.value)

    @staticmethod
    def test_invalid_sample_steps_value(mock_self, base_args):
        base_args.sample_steps = 0
        with pytest.raises(SchemaValidateError) as exc_info:
            Wan2Point2Adapter._validate_args(mock_self, base_args)
        assert "sample_steps must be greater than 0" in str(exc_info.value)

    @staticmethod
    def test_invalid_frame_num_type(mock_self, base_args):
        base_args.frame_num = "invalid"
        with pytest.raises(SchemaValidateError) as exc_info:
            Wan2Point2Adapter._validate_args(mock_self, base_args)
        assert "frame_num must be an integer" in str(exc_info.value)

    @staticmethod
    def test_invalid_frame_num_value(mock_self, base_args):
        base_args.frame_num = 0
        with pytest.raises(SchemaValidateError) as exc_info:
            Wan2Point2Adapter._validate_args(mock_self, base_args)
        assert "frame_num must be greater than 0" in str(exc_info.value)

    @staticmethod
    def test_unsupported_size(mock_self, base_args):
        base_args.size = "invalid_size"
        with pytest.raises(UnsupportedError) as exc_info:
            Wan2Point2Adapter._validate_args(mock_self, base_args)
        assert "Unsupported size" in str(exc_info.value)

    @staticmethod
    def test_missing_prompt(mock_self, base_args):
        del base_args.prompt
        with pytest.raises(SchemaValidateError) as exc_info:
            Wan2Point2Adapter._validate_args(mock_self, base_args)
        assert "Missing required parameter: prompt" in str(exc_info.value)

    @staticmethod
    def test_invalid_prompt_type(mock_self, base_args):
        base_args.prompt = 123
        with pytest.raises(SchemaValidateError) as exc_info:
            Wan2Point2Adapter._validate_args(mock_self, base_args)
        assert "prompt must be a string" in str(exc_info.value)

    @staticmethod
    def test_empty_prompt(mock_self, base_args):
        base_args.prompt = ""
        with pytest.raises(SchemaValidateError) as exc_info:
            Wan2Point2Adapter._validate_args(mock_self, base_args)
        assert "prompt cannot be an empty string" in str(exc_info.value)

    @staticmethod
    def test_invalid_offload_model_type(mock_self, base_args):
        base_args.offload_model = "invalid"
        with pytest.raises(SchemaValidateError) as exc_info:
            Wan2Point2Adapter._validate_args(mock_self, base_args)
        assert "offload_model must be a boolean" in str(exc_info.value)

    @staticmethod
    def test_valid_offload_model(mock_self, base_args):
        base_args.offload_model = True
        Wan2Point2Adapter._validate_args(mock_self, base_args)

        base_args.offload_model = False
        Wan2Point2Adapter._validate_args(mock_self, base_args)

    @staticmethod
    def test_different_tasks(mock_self, base_args):
        tasks = ["t2v-A14B", "i2v-A14B", "ti2v-5B"]

        for task in tasks:
            base_args.task = task
            if task == "i2v-A14B":
                base_args.image = "test_image.jpg"

            with patch('random.randint', return_value=42):
                Wan2Point2Adapter._validate_args(mock_self, base_args)

            assert base_args.task_config == task.split('-')[0]
            assert base_args.param_dtype == torch.float16
            assert base_args.sample_steps == 50

    @staticmethod
    def test_all_parameters_none(mock_self, base_args):
        base_args.sample_steps = None
        base_args.sample_shift = None
        base_args.sample_guide_scale = None
        base_args.frame_num = None
        base_args.base_seed = -1

        with patch('random.randint', return_value=999), patch('sys.maxsize', 1000):
            Wan2Point2Adapter._validate_args(mock_self, base_args)

            assert base_args.sample_steps == 50
            assert base_args.sample_shift == 5.0
            assert base_args.sample_guide_scale == 7.5
            assert base_args.frame_num == 81
            assert base_args.base_seed == 999

    @pytest.fixture
    def mock_self(self):
        mock = Mock()
        mock._check_import_dependency = Mock()
        return mock

    @pytest.fixture
    def base_args(self):
        class Args:
            def __init__(self):
                self.ckpt_dir = "/test/ckpt"
                self.task = "t2v-A14B"
                self.sample_steps = 50
                self.sample_shift = 5.0
                self.frame_num = 81
                self.base_seed = 42
                self.size = "1280 * 720"
                self.prompt = "test prompt"
                self.offload_model = False
                self.image = None
                self.sample_guide_scale = 3.5

            def __contains__(self, item):
                return hasattr(self, item)

        return Args()

    @pytest.fixture(autouse=True)
    def mock_all_dependencies(self, monkeypatch):
        """模拟所有必要的依赖"""
        # 模拟SUPPORTED_TASKS
        monkeypatch.setattr(
            "msmodelslim.model.wan2_2.model_adapter.SUPPORTED_TASKS", ["t2v-A14B", "i2v-A14B", "ti2v-5B"]
        )

        # 模拟EXAMPLE_PROMPT
        monkeypatch.setattr(
            "msmodelslim.model.wan2_2.model_adapter.EXAMPLE_PROMPT",
            {
                "t2v-A14B": {"prompt": "test prompt for t2v", "image": None},
                "i2v-A14B": {"prompt": "test prompt for i2v", "image": "test_image.jpg"},
                "ti2v-5B": {"prompt": "test prompt for ti2v", "image": None},
            },
        )

        # 模拟TASK_CONFIGS
        monkeypatch.setattr(
            "msmodelslim.model.wan2_2.model_adapter.TASK_CONFIGS",
            {"t2v-A14B": "t2v", "i2v-A14B": "i2v", "ti2v-5B": "ti2v"},
        )

        # 模拟wan.configs模块
        mock_configs = Mock()
        mock_configs.SUPPORTED_SIZES = {
            "t2v-A14B": ["1280 * 720", "720 * 1280"],
            "i2v-A14B": ["1280 * 720", "832 * 480", "480 * 832"],
            "ti2v-5B": ["1280 * 720", "832 * 480"],
        }

        # 模拟WAN_CONFIGS
        mock_cfg = Mock()
        mock_cfg.param_dtype = torch.float16
        mock_cfg.sample_steps = 50
        mock_cfg.sample_shift = 5.0
        mock_cfg.sample_guide_scale = 7.5
        mock_cfg.frame_num = 81

        mock_configs.WAN_CONFIGS = {"t2v-A14B": mock_cfg, "i2v-A14B": mock_cfg, "ti2v-5B": mock_cfg}

        # 模拟wan模块及其子模块
        mock_wan = Mock()
        mock_wan.configs = mock_configs

        # 模拟wan.distributed.util模块
        mock_distributed_util = Mock()
        mock_distributed_util.init_distributed_group = Mock()

        # 模拟wan.utils模块
        mock_utils = Mock()
        mock_utils.prompt_extend = Mock()
        mock_utils.prompt_extend.DashScopePromptExpander = Mock()
        mock_utils.prompt_extend.QwenPromptExpander = Mock()
        mock_utils.utils = Mock()
        mock_utils.utils.save_video = Mock()
        mock_utils.utils.str2bool = Mock()

        # 模拟wan.distributed模块
        mock_distributed = Mock()
        mock_distributed.util = mock_distributed_util
        mock_distributed.parallel_mgr = Mock()
        mock_distributed.parallel_mgr.ParallelConfig = Mock()
        mock_distributed.parallel_mgr.init_parallel_env = Mock()
        mock_distributed.parallel_mgr.finalize_parallel_env = Mock()
        mock_distributed.tp_applicator = Mock()
        mock_distributed.tp_applicator.TensorParallelApplicator = Mock()

        mock_wan.distributed = mock_distributed
        mock_wan.utils = mock_utils

        # 模拟PIL模块
        mock_pil = Mock()
        mock_pil.Image = Mock()

        # 模拟mindiesd模块
        mock_mindiesd = Mock()
        mock_mindiesd.CacheConfig = Mock()
        mock_mindiesd.CacheAgent = Mock()

        # 使用patch.dict模拟sys.modules
        with patch.dict(
            'sys.modules',
            {
                'wan': mock_wan,
                'wan.configs': mock_configs,
                'wan.distributed': mock_distributed,
                'wan.distributed.util': mock_distributed_util,
                'wan.utils': mock_utils,
                'wan.utils.prompt_extend': mock_utils.prompt_extend,
                'wan.utils.utils': mock_utils.utils,
                'PIL': mock_pil,
                'PIL.Image': mock_pil.Image,
                'mindiesd': mock_mindiesd,
            },
        ):
            yield


# ------------------------------ _init_logging方法测试 ------------------------------
class TestInitLogging:
    @staticmethod
    def test_rank_zero_config(mock_self, mocker):
        import logging

        mock_stream_handler = mocker.patch('msmodelslim.model.wan2_2.model_adapter.logging.StreamHandler')
        mock_basic_config = mocker.patch('msmodelslim.model.wan2_2.model_adapter.logging.basicConfig')
        Wan2Point2Adapter._init_logging(mock_self, rank=0)
        mock_stream_handler.assert_called_once_with(stream=sys.stdout)

        mock_basic_config.assert_called_once_with(
            level=logging.INFO,
            format="[%(asctime)s] %(levelname)s: %(message)s",
            handlers=[mock_stream_handler.return_value],
        )

    @staticmethod
    def test_non_zero_rank_logging_config(mock_self, mocker):
        import logging

        mock_basic_config = mocker.patch('msmodelslim.model.wan2_2.model_adapter.logging.basicConfig')

        for rank in [1, 2, -1]:
            mock_basic_config.reset_mock()
            Wan2Point2Adapter._init_logging(mock_self, rank=rank)

            mock_basic_config.assert_called_once_with(level=logging.ERROR)

    @pytest.fixture
    def mock_self(self):
        return Mock()


class TestCheckImportDependency:
    @staticmethod
    def test_successful_import(mock_self):
        Wan2Point2Adapter._check_import_dependency(mock_self)

    @staticmethod
    @pytest.mark.parametrize("module_to_remove", ['wan'])
    def test_import_failure_simple(mock_self, monkeypatch, module_to_remove):
        original_module = sys.modules.get(module_to_remove)
        try:
            if module_to_remove in sys.modules:
                monkeypatch.delitem(sys.modules, module_to_remove)
            with pytest.raises(ImportError) as exc_info:
                Wan2Point2Adapter._check_import_dependency(mock_self)
            assert "Failed to import required components from wan." in str(exc_info.value)

        finally:
            if original_module is not None:
                sys.modules[module_to_remove] = original_module
            else:
                if module_to_remove in sys.modules:
                    del sys.modules[module_to_remove]

    @pytest.fixture
    def mock_self(self):
        return Mock()

    @pytest.fixture(autouse=True)
    def mock_all_dependencies(self, monkeypatch):
        monkeypatch.setattr(
            "msmodelslim.model.wan2_2.model_adapter.SUPPORTED_TASKS", ["t2v-A14B", "i2v-A14B", "ti2v-5B"]
        )

        monkeypatch.setattr(
            "msmodelslim.model.wan2_2.model_adapter.EXAMPLE_PROMPT",
            {
                "t2v-A14B": {"prompt": "test prompt for t2v", "image": None},
                "i2v-A14B": {"prompt": "test prompt for i2v", "image": "test_image.jpg"},
                "ti2v-5B": {"prompt": "test prompt for ti2v", "image": None},
            },
        )

        monkeypatch.setattr(
            "msmodelslim.model.wan2_2.model_adapter.TASK_CONFIGS",
            {"t2v-A14B": "t2v", "i2v-A14B": "i2v", "ti2v-5B": "ti2v"},
        )

        mock_configs = Mock()
        mock_configs.SUPPORTED_SIZES = {
            "t2v-A14B": ["1280 * 720", "720 * 1280"],
            "i2v-A14B": ["1280 * 720", "832 * 480", "480 * 832"],
            "ti2v-5B": ["1280 * 720", "832 * 480"],
        }

        mock_cfg = Mock()
        mock_cfg.param_dtype = torch.float16
        mock_cfg.sample_steps = 50
        mock_cfg.sample_shift = 5.0
        mock_cfg.sample_guide_scale = 7.5
        mock_cfg.frame_num = 81

        mock_configs.WAN_CONFIGS = {"t2v-A14B": mock_cfg, "i2v-A14B": mock_cfg, "ti2v-5B": mock_cfg}

        mock_wan = Mock()
        mock_wan.configs = mock_configs

        mock_distributed_util = Mock()
        mock_distributed_util.init_distributed_group = Mock()

        mock_utils = Mock()
        mock_utils.prompt_extend = Mock()
        mock_utils.prompt_extend.DashScopePromptExpander = Mock()
        mock_utils.prompt_extend.QwenPromptExpander = Mock()
        mock_utils.utils = Mock()
        mock_utils.utils.save_video = Mock()
        mock_utils.utils.str2bool = Mock()

        mock_distributed = Mock()
        mock_distributed.util = mock_distributed_util
        mock_distributed.parallel_mgr = Mock()
        mock_distributed.parallel_mgr.ParallelConfig = Mock()
        mock_distributed.parallel_mgr.init_parallel_env = Mock()
        mock_distributed.parallel_mgr.finalize_parallel_env = Mock()
        mock_distributed.tp_applicator = Mock()
        mock_distributed.tp_applicator.TensorParallelApplicator = Mock()

        mock_wan.distributed = mock_distributed
        mock_wan.utils = mock_utils

        mock_pil = Mock()
        mock_pil.Image = Mock()

        mock_mindiesd = Mock()
        mock_mindiesd.CacheConfig = Mock()
        mock_mindiesd.CacheAgent = Mock()

        with patch.dict(
            'sys.modules',
            {
                'wan': mock_wan,
                'wan.configs': mock_configs,
                'wan.distributed': mock_distributed,
                'wan.distributed.util': mock_distributed_util,
                'wan.utils': mock_utils,
                'wan.utils.prompt_extend': mock_utils.prompt_extend,
                'wan.utils.utils': mock_utils.utils,
                'PIL': mock_pil,
                'PIL.Image': mock_pil.Image,
                'mindiesd': mock_mindiesd,
            },
        ):
            yield


# ------------------------------ set_model_args方法测试 ------------------------------
class TestSetModelArgs:
    @staticmethod
    def test_valid_update(mock_self, base_override):
        valid_config = base_override({"sample_steps": 60, "offload_model": True, "use_attentioncache": True})
        mock_self.set_model_args(valid_config)

        assert mock_self.model_args.ckpt_dir == mock_self.model_path
        assert mock_self.model_args.sample_steps == 60
        mock_self._validate_args.assert_called_once()

    @staticmethod
    def test_illegal_attr_raise_error(mock_self, base_override):
        invalid_config = base_override({"sample_steps": 60, "illegal_attr": "invalid"})
        with pytest.raises(SchemaValidateError) as exc:
            mock_self.set_model_args(invalid_config)

        assert "illegal config attributes: ['illegal_attr']" in str(exc.value)

    @staticmethod
    def test_skip_none_value(mock_self, base_override):
        none_config = base_override({"sample_steps": 60, "offload_model": None})
        mock_parser = mock_self._get_parser()
        mock_self.set_model_args(none_config)
        argv = mock_parser.parse_args.call_args[0][0]

        assert "--sample_steps" in argv
        assert "offload_model" not in str(argv)

    @staticmethod
    def test_bool_false_handling(mock_self, base_override):
        false_bool_config = base_override({"sample_steps": 60, "offload_model": True, "use_attentioncache": False})
        mock_parser = mock_self._get_parser()
        mock_self.set_model_args(false_bool_config)
        argv = mock_parser.parse_args.call_args[0][0]

        assert "--use_attentioncache" not in argv
        assert "--offload_model" in argv and "true" in argv

    @pytest.fixture
    def mock_self(self):
        mock = Wan2Point2Adapter('t2v-A14B', Path('/test/model/ckpt'))

        class ModelArgs:
            def __init__(self):
                self.ckpt_dir = None
                self.task = "t2v-A14B"
                self.size = "1280*720"
                self.prompt = "test prompt"
                self.sample_steps = 50
                self.offload_model = False
                self.use_attentioncache = False

        mock.model_args = ModelArgs()
        mock._validate_args = Mock()
        mock._get_parser = Mock(return_value=Mock(parse_args=Mock(return_value=mock.model_args)))
        return mock

    @pytest.fixture(autouse=True)
    def mock_all_dependencies(self, monkeypatch):
        monkeypatch.setattr(
            "msmodelslim.model.wan2_2.model_adapter.SUPPORTED_TASKS", ["t2v-A14B", "i2v-A14B", "ti2v-5B"]
        )

        monkeypatch.setattr(
            "msmodelslim.model.wan2_2.model_adapter.EXAMPLE_PROMPT",
            {
                "t2v-A14B": {"prompt": "test prompt for t2v", "image": None},
                "i2v-A14B": {"prompt": "test prompt for i2v", "image": "test_image.jpg"},
                "ti2v-5B": {"prompt": "test prompt for ti2v", "image": None},
            },
        )

        monkeypatch.setattr(
            "msmodelslim.model.wan2_2.model_adapter.TASK_CONFIGS",
            {"t2v-A14B": "t2v", "i2v-A14B": "i2v", "ti2v-5B": "ti2v"},
        )

        mock_configs = Mock()
        mock_configs.SUPPORTED_SIZES = {
            "t2v-A14B": ["1280*720", "720*1280"],
            "i2v-A14B": ["1280*720", "832*480", "480*832"],
            "ti2v-5B": ["1280*720", "832*480"],
        }

        mock_cfg = Mock()
        mock_cfg.param_dtype = torch.float16
        mock_cfg.sample_steps = 50
        mock_cfg.sample_shift = 5.0
        mock_cfg.sample_guide_scale = 7.5
        mock_cfg.frame_num = 81

        mock_configs.WAN_CONFIGS = {"t2v-A14B": mock_cfg, "i2v-A14B": mock_cfg, "ti2v-5B": mock_cfg}

        mock_configs.SIZE_CONFIGS = {
            '720*1280': (720, 1280),
            '1280*720': (1280, 720),
            '480*832': (480, 832),
            '832*480': (832, 480),
            '704*1280': (704, 1280),
            '1280*704': (1280, 704),
            '432*768': (432, 768),
            '768*432': (768, 432),
        }

        mock_wan = Mock()
        mock_wan.configs = mock_configs

        mock_distributed_util = Mock()
        mock_distributed_util.init_distributed_group = Mock()

        mock_utils = Mock()
        mock_utils.prompt_extend = Mock()
        mock_utils.prompt_extend.DashScopePromptExpander = Mock()
        mock_utils.prompt_extend.QwenPromptExpander = Mock()
        mock_utils.utils = Mock()
        mock_utils.utils.save_video = Mock()
        mock_utils.utils.str2bool = Mock()

        mock_distributed = Mock()
        mock_distributed.util = mock_distributed_util
        mock_distributed.parallel_mgr = Mock()
        mock_distributed.parallel_mgr.ParallelConfig = Mock()
        mock_distributed.parallel_mgr.init_parallel_env = Mock()
        mock_distributed.parallel_mgr.finalize_parallel_env = Mock()
        mock_distributed.tp_applicator = Mock()
        mock_distributed.tp_applicator.TensorParallelApplicator = Mock()

        mock_wan.distributed = mock_distributed
        mock_wan.utils = mock_utils

        mock_pil = Mock()
        mock_pil.Image = Mock()

        mock_mindiesd = Mock()
        mock_mindiesd.CacheConfig = Mock()
        mock_mindiesd.CacheAgent = Mock()

        with patch.dict(
            'sys.modules',
            {
                'wan': mock_wan,
                'wan.configs': mock_configs,
                'wan.distributed': mock_distributed,
                'wan.distributed.util': mock_distributed_util,
                'wan.utils': mock_utils,
                'wan.utils.prompt_extend': mock_utils.prompt_extend,
                'wan.utils.utils': mock_utils.utils,
                'PIL': mock_pil,
                'PIL.Image': mock_pil.Image,
                'mindiesd': mock_mindiesd,
            },
        ):
            yield

    @pytest.fixture
    def base_override(self):
        class Override:
            def __init__(self, config):
                self.config = config

            def __getitem__(self, key):
                return self.config[key]

            def keys(self):
                return self.config.keys()

        return Override


class TestApplyQuantization:
    @staticmethod
    def test_apply_quantization_with_no_sync(mock_self, process_func):
        mock_self.low_noise_model.no_sync = MagicMock()
        mock_self.high_noise_model.no_sync = MagicMock()

        mock_context = MagicMock()
        mock_self.low_noise_model.no_sync.return_value = mock_context
        mock_self.high_noise_model.no_sync.return_value = mock_context

        mock_self.model_args.param_dtype = torch.float16
        with patch('torch.cuda.amp.autocast'):
            Wan2Point2Adapter.apply_quantization(mock_self, process_func)

        assert mock_self.low_noise_model.no_sync.call_count >= 1
        assert mock_self.high_noise_model.no_sync.call_count >= 1
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
        mock.transformer = transformer
        mock.low_noise_model = transformer
        mock.high_noise_model = transformer
        mock.model_args = Mock(param_dtype=torch.float16)
        return mock

    @pytest.fixture
    def process_func(self):
        return Mock()


class TestRunCalibInference:
    @staticmethod
    def test_run_calib_inference_success(mock_self):
        with patch('PIL.Image.open') as mock_image_open:
            # 创建真实的图像 mock
            mock_image_instance = Mock()
            mock_image_instance.convert.return_value = mock_image_instance
            mock_image_open.return_value = mock_image_instance

            mock_self.model_args.image = "/path/to/real/image.jpg"
            mock_self.model_args.task = "t2v-A14B"
            mock_self.model_args.prompt = "test prompt"

            mock_self.model_args.base_seed = 42
            mock_self.model_args.size = "1280*720"
            mock_self.model_args.frame_num = 81
            mock_self.model_args.sample_shift = 5.0
            mock_self.model_args.sample_solver = "unipc"
            mock_self.model_args.sample_steps = 50
            mock_self.model_args.sample_guide_scale = 7.5
            mock_self.model_args.offload_model = False

            mock_self.wan_t2v.generate.return_value = Mock()

            with (
                patch('msmodelslim.model.wan2_2.model_adapter.torch'),
                patch('msmodelslim.model.wan2_2.model_adapter.time') as mock_time,
                patch('msmodelslim.model.wan2_2.model_adapter.tqdm') as mock_tqdm,
            ):
                mock_time.time.side_effect = [1.0, 3.0]
                mock_tqdm.return_value.__iter__.return_value = [1]

                Wan2Point2Adapter.run_calib_inference(mock_self)

                mock_image_open.assert_not_called()  # t2v任务不应该调用Image.open
                mock_self.wan_t2v.generate.assert_called_once()

    @pytest.fixture
    def mock_self(self):
        mock = Mock()

        mock.model_args = Mock(
            base_seed=42,
            size="1280*720",
            frame_num=81,
            sample_shift=5.0,
            sample_solver="euler",
            sample_steps=50,
            sample_guide_scale=7.5,
            offload_model=False,
            prompt="test prompt",
        )

        mock.wan_t2v = Mock()
        mock.wan_t2v.low_noise_model = Mock()
        mock.wan_t2v.high_noise_model = Mock()
        mock.wan_t2v.generate = Mock(return_value=Mock())

        return mock

    @pytest.fixture(autouse=True)
    def mock_all_dependencies(self, monkeypatch):
        monkeypatch.setattr(
            "msmodelslim.model.wan2_2.model_adapter.SUPPORTED_TASKS", ["t2v-A14B", "i2v-A14B", "ti2v-5B"]
        )

        monkeypatch.setattr(
            "msmodelslim.model.wan2_2.model_adapter.EXAMPLE_PROMPT",
            {
                "t2v-A14B": {"prompt": "test prompt for t2v", "image": None},
                "i2v-A14B": {"prompt": "test prompt for i2v", "image": "test_image.jpg"},
                "ti2v-5B": {"prompt": "test prompt for ti2v", "image": None},
            },
        )

        monkeypatch.setattr(
            "msmodelslim.model.wan2_2.model_adapter.TASK_CONFIGS",
            {"t2v-A14B": "t2v", "i2v-A14B": "i2v", "ti2v-5B": "ti2v"},
        )

        mock_configs = Mock()
        mock_configs.SUPPORTED_SIZES = {
            "t2v-A14B": ["1280*720", "720*1280"],
            "i2v-A14B": ["1280*720", "832*480", "480*832"],
            "ti2v-5B": ["1280*720", "832*480"],
        }

        mock_cfg = Mock()
        mock_cfg.param_dtype = torch.float16
        mock_cfg.sample_steps = 50
        mock_cfg.sample_shift = 5.0
        mock_cfg.sample_guide_scale = 7.5
        mock_cfg.frame_num = 81

        mock_configs.WAN_CONFIGS = {"t2v-A14B": mock_cfg, "i2v-A14B": mock_cfg, "ti2v-5B": mock_cfg}

        mock_configs.SIZE_CONFIGS = {
            '720*1280': (720, 1280),
            '1280*720': (1280, 720),
            '480*832': (480, 832),
            '832*480': (832, 480),
            '704*1280': (704, 1280),
            '1280*704': (1280, 704),
            '432*768': (432, 768),
            '768*432': (768, 432),
        }

        mock_wan = Mock()
        mock_wan.configs = mock_configs

        mock_distributed_util = Mock()
        mock_distributed_util.init_distributed_group = Mock()

        mock_utils = Mock()
        mock_utils.prompt_extend = Mock()
        mock_utils.prompt_extend.DashScopePromptExpander = Mock()
        mock_utils.prompt_extend.QwenPromptExpander = Mock()
        mock_utils.utils = Mock()
        mock_utils.utils.save_video = Mock()
        mock_utils.utils.str2bool = Mock()

        mock_distributed = Mock()
        mock_distributed.util = mock_distributed_util
        mock_distributed.parallel_mgr = Mock()
        mock_distributed.parallel_mgr.ParallelConfig = Mock()
        mock_distributed.parallel_mgr.init_parallel_env = Mock()
        mock_distributed.parallel_mgr.finalize_parallel_env = Mock()
        mock_distributed.tp_applicator = Mock()
        mock_distributed.tp_applicator.TensorParallelApplicator = Mock()

        mock_wan.distributed = mock_distributed
        mock_wan.utils = mock_utils

        mock_pil = Mock()
        mock_pil.Image = Mock()

        mock_mindiesd = Mock()
        mock_mindiesd.CacheConfig = Mock()
        mock_mindiesd.CacheAgent = Mock()

        with patch.dict(
            'sys.modules',
            {
                'wan': mock_wan,
                'wan.configs': mock_configs,
                'wan.distributed': mock_distributed,
                'wan.distributed.util': mock_distributed_util,
                'wan.utils': mock_utils,
                'wan.utils.prompt_extend': mock_utils.prompt_extend,
                'wan.utils.utils': mock_utils.utils,
                'PIL': mock_pil,
                'PIL.Image': mock_pil.Image,
                'mindiesd': mock_mindiesd,
            },
        ):
            yield
