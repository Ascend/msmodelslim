#!/usr/bin/env python
# -*- coding: UTF-8 -*-

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

# pylint: disable=redefined-outer-name

import sys
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
import pytest

from msmodelslim.model.wan2_1.model_adapter import (
    Wan2Point1Adapter,
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
        "t2v-14B": ["1280*720", "832*480"],
        "t2v-1.3B": ["832*480"],
        "i2v-14B": ["1280*720", "832*480", "480*832"],
        "t2i-14B": ("1280*720", "832*480"),
    }

    wan_main = sys.modules['wan']
    mock_wan_t2v = MagicMock()
    mock_wan_t2v.model = MagicMock()
    wan_main.WanT2V.return_value = mock_wan_t2v

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
class TestWan2Point1Adapter:
    @staticmethod
    def test_initialization(adapter):
        assert adapter.model_type == "t2v-14B"
        assert adapter.model_path == Path("/test/model/path")

    @staticmethod
    def test_get_model_info(adapter):
        assert adapter.get_model_type() == "t2v-14B"
        assert adapter.get_model_pedigree() == "wan2_1"

    @staticmethod
    def test_handle_dataset(adapter):
        mock_dataset = [Mock(), Mock()]
        result = adapter.handle_dataset(mock_dataset)
        assert list(result) == mock_dataset

    @staticmethod
    def test_enable_kv_cache(adapter):
        """测试enable_kv_cache方法的调用和参数传递"""
        adapter.enable_kv_cache(Mock(), True)
        assert True  # 方法无异常执行即通过

    @staticmethod
    def test_init_model_returns_transformer():
        """测试init_model方法是否正确返回transformer"""
        mock_self = Mock()
        mock_transformer = Mock()
        mock_self.transformer = mock_transformer  # 设置预期返回值

        result = Wan2Point1Adapter.init_model(mock_self)
        assert result == {'': mock_transformer}, "init_model应返回self.transformer"

    @pytest.fixture
    def adapter(self):
        """当前类专用的适配器实例fixture"""
        with patch("msmodelslim.model.wan2_1.model_adapter.Wan2Point1Adapter._check_import_dependency"):
            adapter_instance = Wan2Point1Adapter("t2v-14B", Path("/test/model/path"))

        adapter_instance.model_args = MagicMock()
        adapter_instance.model_args.task = "t2v-14B"
        adapter_instance.model_args.size = "1280*720"
        return adapter_instance


# ------------------------------ _load_pipeline方法测试 ------------------------------
class TestLoadPipeline:
    @staticmethod
    def test_normal_execution(mock_self, mock_env):
        Wan2Point1Adapter._load_pipeline(mock_self)
        sys.modules['wan'].WanT2V.assert_called_once()
        assert mock_self.wan_t2v is not None

    @staticmethod
    def test_t5_fsdp_unsupported(mock_self, mock_env):
        mock_self.model_args.t5_fsdp = True
        with pytest.raises(SchemaValidateError):
            Wan2Point1Adapter._load_pipeline(mock_self)

    @staticmethod
    def test_dit_fsdp_unsupported(mock_self, mock_env):
        mock_self.model_args.dit_fsdp = True
        with pytest.raises(SchemaValidateError):
            Wan2Point1Adapter._load_pipeline(mock_self)

    @staticmethod
    def test_ulysses_size_validation(mock_self, mock_env):
        mock_self.model_args.ulysses_size = 3
        with pytest.raises(SchemaValidateError):
            Wan2Point1Adapter._load_pipeline(mock_self)

    @staticmethod
    def test_vae_parallel_unsupported(mock_self, mock_env):
        mock_self.model_args.vae_parallel = True
        with pytest.raises(SchemaValidateError):
            Wan2Point1Adapter._load_pipeline(mock_self)

    @staticmethod
    def test_load_pipeline_calls_internal(mock_self):
        """Legacy 路径：load_pipeline 负责加载 pipeline 与 cache"""
        mock_self._load_pipeline = Mock()
        mock_self._setup_cache = Mock()
        Wan2Point1Adapter.load_pipeline(mock_self)
        mock_self._load_pipeline.assert_called_once()
        mock_self._setup_cache.assert_called_once()

    @pytest.fixture
    def mock_self(self):
        """当前类专用的self对象fixture"""
        mock = Mock()
        mock.model_args = Mock()
        mock.model_args.t5_fsdp = None
        mock.model_args.dit_fsdp = None
        mock.model_args.vae_parallel = None
        mock.model_args.cfg_size = 0
        mock.model_args.ulysses_size = 0
        mock.model_args.ring_size = 0
        mock.model_args.task = "test_task"
        mock._check_import_dependency = Mock()
        mock._init_logging = Mock()
        return mock


# ------------------------------ _validate_args方法测试 ------------------------------
class TestValidateArgs:
    @staticmethod
    def test_valid_base_case(mock_self, base_args):
        Wan2Point1Adapter._validate_args(mock_self, base_args)

    @staticmethod
    def test_missing_prompt(mock_self, base_args):
        del base_args.prompt  # 删除prompt属性
        with pytest.raises(SchemaValidateError) as exc_info:
            Wan2Point1Adapter._validate_args(mock_self, base_args)
        assert "Missing required parameter: prompt" in str(exc_info.value)

    @staticmethod
    def test_invalid_prompt_type(mock_self, base_args):
        base_args.prompt = 123  # 非字符串类型
        with pytest.raises(SchemaValidateError):
            Wan2Point1Adapter._validate_args(mock_self, base_args)

    @staticmethod
    def test_empty_prompt(mock_self, base_args):
        base_args.prompt = ""
        with pytest.raises(SchemaValidateError):
            Wan2Point1Adapter._validate_args(mock_self, base_args)

    @staticmethod
    def test_missing_ckpt_dir(mock_self, base_args):
        base_args.ckpt_dir = None
        with pytest.raises(InvalidModelError):
            Wan2Point1Adapter._validate_args(mock_self, base_args)

    @staticmethod
    def test_unsupported_task(mock_self, base_args):
        base_args.task = True
        with pytest.raises(SchemaValidateError):
            Wan2Point1Adapter._validate_args(mock_self, base_args)

        base_args.task = "invalid-task"
        with pytest.raises(UnsupportedError):
            Wan2Point1Adapter._validate_args(mock_self, base_args)

    @staticmethod
    def test_sample_steps_defaults(mock_self, base_args):
        base_args.sample_steps = None
        base_args.task = "i2v-14B"
        Wan2Point1Adapter._validate_args(mock_self, base_args)
        assert base_args.sample_steps == 40

    @staticmethod
    def test_sample_shift_i2v_832x480(mock_self, base_args):
        """测试i2v任务+832*480尺寸时sample_shift默认值为3.0"""
        base_args.task = "i2v-14B"
        base_args.size = "832*480"
        base_args.sample_shift = None  # 触发默认值逻辑

        Wan2Point1Adapter._validate_args(mock_self, base_args)
        assert base_args.sample_shift == 3.0

    @staticmethod
    def test_sample_shift_other_cases(mock_self, base_args):
        """测试其他场景下sample_shift默认值为5.0"""
        # 场景1: t2v任务+任意尺寸
        base_args.task = "t2v-14B"
        base_args.size = "832*480"
        base_args.sample_shift = None
        Wan2Point1Adapter._validate_args(mock_self, base_args)
        assert base_args.sample_shift == 5.0

        # 场景2: i2v任务+非832*480尺寸
        base_args.task = "i2v-14B"
        base_args.size = "1280*720"
        base_args.sample_shift = None
        Wan2Point1Adapter._validate_args(mock_self, base_args)
        assert base_args.sample_shift == 5.0

    @staticmethod
    def test_base_seed_non_negative(mock_self, base_args):
        """测试base_seed非负时保持原值"""
        base_args.base_seed = 100  # 非负值
        Wan2Point1Adapter._validate_args(mock_self, base_args)
        assert base_args.base_seed == 100

    @staticmethod
    def test_base_seed_negative_random(mock_self, base_args, mocker):
        """测试base_seed为负时随机生成"""
        mocker.patch("msmodelslim.model.wan2_1.model_adapter.random.randint", return_value=999)
        base_args.base_seed = -5  # 负值
        Wan2Point1Adapter._validate_args(mock_self, base_args)
        assert base_args.base_seed == 999

    @staticmethod
    def test_offload_model_absent(mock_self, base_args):
        del base_args.offload_model
        Wan2Point1Adapter._validate_args(mock_self, base_args)

    @staticmethod
    def test_offload_model_valid_boolean(mock_self, base_args):
        base_args.offload_model = True
        Wan2Point1Adapter._validate_args(mock_self, base_args)

        base_args.offload_model = False
        Wan2Point1Adapter._validate_args(mock_self, base_args)

    @staticmethod
    def test_offload_model_invalid_str(mock_self, base_args):
        base_args.offload_model = "invalid"
        with pytest.raises(SchemaValidateError):
            Wan2Point1Adapter._validate_args(mock_self, base_args)

    @staticmethod
    def test_frame_num_t2i_must_be_1(mock_self, base_args):
        """强化t2i任务frame_num必须为1的检查"""
        base_args.task = "t2i-14B"
        base_args.frame_num = 0
        with pytest.raises(SchemaValidateError):
            Wan2Point1Adapter._validate_args(mock_self, base_args)

        base_args.frame_num = 2
        with pytest.raises(UnsupportedError):
            Wan2Point1Adapter._validate_args(mock_self, base_args)

    @staticmethod
    def test_sample_steps_invalid_values(mock_self, base_args):
        base_args.sample_steps = "invalid"
        with pytest.raises(SchemaValidateError):
            Wan2Point1Adapter._validate_args(mock_self, base_args)

    @staticmethod
    def test_sample_steps_valid_values(mock_self, base_args):
        base_args.sample_steps = 0
        with pytest.raises(SchemaValidateError):
            Wan2Point1Adapter._validate_args(mock_self, base_args)

        base_args.sample_steps = 1
        Wan2Point1Adapter._validate_args(mock_self, base_args)  # 不应抛出异常

    @staticmethod
    def test_frame_num_boundary_values(mock_self, base_args):
        base_args.frame_num = 0
        with pytest.raises(SchemaValidateError):
            Wan2Point1Adapter._validate_args(mock_self, base_args)

        base_args.task = "t2v-14B"
        base_args.frame_num = 1
        Wan2Point1Adapter._validate_args(mock_self, base_args)

    @staticmethod
    def test_frame_num_default_t2i(mock_self, base_args):
        base_args.task = "t2i-14B"
        base_args.frame_num = None  # 触发默认值逻辑
        Wan2Point1Adapter._validate_args(mock_self, base_args)
        assert base_args.frame_num == 1

    @staticmethod
    def test_frame_num_default_t2v(mock_self, base_args):
        base_args.task = "t2v-14B"
        base_args.frame_num = None  # 触发默认值逻辑
        Wan2Point1Adapter._validate_args(mock_self, base_args)
        assert base_args.frame_num == 81

    @staticmethod
    def test_frame_num_default_i2v(mock_self, base_args):
        base_args.task = "i2v-14B"
        base_args.frame_num = None  # 触发默认值逻辑
        Wan2Point1Adapter._validate_args(mock_self, base_args)
        assert base_args.frame_num == 81

    @staticmethod
    def test_frame_num_invalid_type_string(mock_self, base_args):
        base_args.frame_num = "81"
        with pytest.raises(SchemaValidateError) as exc_info:
            Wan2Point1Adapter._validate_args(mock_self, base_args)
        assert "frame_num must be an integer, got str" in str(exc_info.value)

    @staticmethod
    def test_size_valid_values(mock_self, base_args):
        base_args.size = "1280*720"
        base_args.task = "t2v-14B"
        Wan2Point1Adapter._validate_args(mock_self, base_args)

        base_args.size = "1280*720"
        base_args.task = "t2v-1.3B"
        with pytest.raises(UnsupportedError) as exc_info:
            Wan2Point1Adapter._validate_args(mock_self, base_args)
        assert "Unsupported size '1280*720' for task 't2v-1.3B'" in str(exc_info.value)

    @pytest.fixture
    def mock_self(self):
        """当前类专用的self对象fixture"""
        mock = Mock()
        mock._check_import_dependency = Mock()
        return mock

    @pytest.fixture
    def base_args(self):
        """当前类专用的基础参数fixture，使用普通类替代Mock以支持属性检查"""

        class Args:
            def __init__(self):
                self.ckpt_dir = "/test/ckpt"
                self.task = "t2v-14B"
                self.sample_steps = 50
                self.sample_shift = 5.0
                self.frame_num = 81
                self.base_seed = 42
                self.size = "1280*720"
                self.prompt = "test prompt"
                self.offload_model = False

            # 支持 "prompt" in args 检查
            def __contains__(self, item):
                return hasattr(self, item)

        return Args()

    @pytest.fixture(autouse=True)
    def mock_supported_tasks(self, monkeypatch):
        """当前类专用的任务配置模拟"""
        monkeypatch.setattr(
            "msmodelslim.model.wan2_1.model_adapter.SUPPORTED_TASKS", ["t2v-14B", "i2v-14B", "t2i-14B", "t2v-1.3B"]
        )


# ------------------------------ _init_logging方法测试 ------------------------------
class TestInitLogging:
    @staticmethod
    def test_rank_zero_config(mock_self, mocker):
        import logging

        mock_stream_handler = mocker.patch('msmodelslim.model.wan2_1.model_adapter.logging.StreamHandler')
        mock_basic_config = mocker.patch('msmodelslim.model.wan2_1.model_adapter.logging.basicConfig')
        Wan2Point1Adapter._init_logging(mock_self, rank=0)  # 执行测试
        mock_stream_handler.assert_called_once_with(stream=sys.stdout)  # 验证StreamHandler创建

        # 验证basicConfig调用
        mock_basic_config.assert_called_once_with(
            level=logging.INFO,
            format="[%(asctime)s] %(levelname)s: %(message)s",
            handlers=[mock_stream_handler.return_value],
        )

    @staticmethod
    def test_non_zero_rank_logging_config(mock_self, mocker):
        import logging

        mock_basic_config = mocker.patch('msmodelslim.model.wan2_1.model_adapter.logging.basicConfig')

        for rank in [1, 2, -1]:
            mock_basic_config.reset_mock()  # 重置mock
            Wan2Point1Adapter._init_logging(mock_self, rank=rank)

            mock_basic_config.assert_called_once_with(level=logging.ERROR)  # 验证配置

    @pytest.fixture
    def mock_self(self):
        """当前类专用的self对象fixture"""
        return Mock()


class TestCheckImportDependency:
    @staticmethod
    def test_successful_import(mock_self):
        """测试所有依赖都能正常导入的场景"""
        # 执行方法，不应抛出异常
        Wan2Point1Adapter._check_import_dependency(mock_self)

    @pytest.mark.parametrize("module_to_remove", ['wan'])
    def test_import_failure_simple(self, mock_self, monkeypatch, module_to_remove):
        """测试导入失败（临时移除指定模块并手动处理恢复）"""
        original_module = sys.modules.get(module_to_remove)
        try:
            if module_to_remove in sys.modules:
                monkeypatch.delitem(sys.modules, module_to_remove)
            with pytest.raises(ImportError) as exc_info:
                Wan2Point1Adapter._check_import_dependency(mock_self)
            assert "Failed to import required components from wan." in str(exc_info.value)

        finally:
            if original_module is not None:
                sys.modules[module_to_remove] = original_module
            else:
                if module_to_remove in sys.modules:
                    del sys.modules[module_to_remove]

    @pytest.fixture
    def mock_self(self):
        """创建模拟的self对象"""
        return Mock()


# ------------------------------ set_model_args方法测试 ------------------------------
class TestSetModelArgs:
    @staticmethod
    def test_valid_update(mock_self, base_override):
        """测试合法配置更新：用base_override创建有效配置"""
        valid_config = base_override({"sample_steps": 60, "offload_model": True, "use_attentioncache": True})
        mock_self.set_model_args(valid_config)

        assert mock_self.model_args.ckpt_dir == mock_self.model_path
        assert mock_self.model_args.sample_steps == 60
        mock_self._validate_args.assert_called_once()

    @staticmethod
    def test_illegal_attr_raise_error(mock_self, base_override):
        """测试非法属性：动态传入含非法键的配置"""
        invalid_config = base_override(
            {
                "sample_steps": 60,
                "illegal_attr": "invalid",  # 非法属性
            }
        )
        with pytest.raises(SchemaValidateError) as exc:
            mock_self.set_model_args(invalid_config)

        assert "illegal config attributes: ['illegal_attr']" in str(exc.value)

    @staticmethod
    def test_skip_none_value(mock_self, base_override):
        """测试跳过None值：动态传入含None的配置"""
        none_config = base_override(
            {
                "sample_steps": 60,
                "offload_model": None,  # None值参数
            }
        )
        mock_parser = mock_self._get_parser()
        mock_self.set_model_args(none_config)
        argv = mock_parser.parse_args.call_args[0][0]

        assert "--sample_steps" in argv
        assert "offload_model" not in str(argv)

    @staticmethod
    def test_bool_false_handling(mock_self, base_override):
        """测试False布尔值：动态传入含False的配置（无需单独类）"""
        false_bool_config = base_override(
            {
                "sample_steps": 60,
                "offload_model": True,  # True保留
                "use_attentioncache": False,  # False忽略
            }
        )
        mock_parser = mock_self._get_parser()
        mock_self.set_model_args(false_bool_config)
        argv = mock_parser.parse_args.call_args[0][0]

        assert "--use_attentioncache" not in argv
        assert "--offload_model" in argv and "true" in argv

    @pytest.fixture
    def mock_self(self):
        """初始化核心属性+必要mock"""
        mock = Wan2Point1Adapter('t2v-14B', Path('/test/model/ckpt'))

        # ModelArgs：仅保留关键属性
        class ModelArgs:
            def __init__(self):
                self.ckpt_dir = None
                self.task = "t2v-14B"
                self.size = "1280*720"
                self.prompt = "test prompt"
                self.sample_steps = 50
                self.offload_model = False
                self.use_attentioncache = False

        mock.model_args = ModelArgs()
        mock._validate_args = Mock()
        mock._get_parser = Mock(return_value=Mock(parse_args=Mock(return_value=mock.model_args)))
        return mock

    @pytest.fixture
    def base_override(self):
        """通用配置基类：避免重复定义，支持动态修改"""

        class Override:
            def __init__(self, config):
                self.config = config  # 接收动态配置字典

            def __getitem__(self, key):
                return self.config[key]  # 按配置返回值

            def keys(self):
                return self.config.keys()  # 跟随配置动态返回键

        return Override


class TestApplyQuantization:
    @pytest.fixture
    def mock_self(self):
        """创建包含transformer结构的模拟self对象"""
        mock = Mock()
        transformer = Mock()  # 模拟transformer及其子模块
        module_embedding = Mock()  # 非blocks模块
        module_block = Mock()  # blocks模块
        module_norm = Mock()  # 非blocks模块
        transformer.named_modules.return_value = [
            ('embedding', module_embedding),
            ('blocks.0', module_block),
            ('norm', module_norm),
        ]
        mock.transformer = transformer

        return mock

    @pytest.fixture
    def process_func(self):
        """创建模拟的回调函数"""
        return Mock()

    def test_apply_quantization_with_no_sync(self, mock_self, process_func):
        """测试当存在no_sync方法时的量化上下文"""

        class MockNoSync:
            @staticmethod
            def __enter__():
                return self

            @staticmethod
            def __exit__(*args):
                return False

        mock_self.no_sync = Mock(return_value=MockNoSync())
        mock_self.transformer = Mock()
        mock_self.transformer.named_modules.return_value = []
        Wan2Point1Adapter.apply_quantization(mock_self, process_func)
        mock_self.no_sync.assert_called_once()
        process_func.assert_called_once()


class TestRunCalibInference:
    @staticmethod
    def test_run_calib_inference_success(mock_self, mock_logging):
        """测试生成流程完整执行"""
        Wan2Point1Adapter.run_calib_inference(mock_self)

        # 验证核心调用
        mock_self.wan_t2v.model.to.assert_called_once_with('npu')
        mock_self.wan_t2v.generate.assert_called_once()

        # 验证日志正确输出
        mock_logging.info.assert_called_once()
        assert "Generating video used time  2.0000s" in str(mock_logging.info.call_args)

    @pytest.fixture
    def mock_self(self):
        """创建完整的模拟对象"""
        mock = Mock()

        # 模拟model_args
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

        # 模拟wan_t2v
        mock.wan_t2v = Mock()
        mock.wan_t2v.model = Mock()
        mock.wan_t2v.generate = Mock(return_value=Mock())

        return mock

    @pytest.fixture(autouse=True)
    def mock_logging(self):
        with patch('msmodelslim.model.wan2_1.model_adapter.logging') as mock_log:
            yield mock_log

    @pytest.fixture(autouse=True)
    def mock_dependencies(self):
        """模拟依赖并设置具体时间值"""
        with (
            patch('wan.configs.SIZE_CONFIGS', {'1280*720': (1280, 720)}),
            patch('msmodelslim.model.wan2_1.model_adapter.torch'),
        ):
            # 关键：让tqdm可迭代
            with patch('msmodelslim.model.wan2_1.model_adapter.tqdm') as mock_tqdm:
                mock_tqdm.return_value.__iter__.return_value = [1]

                # 关键：给time.time()设置具体返回值，避免Mock对象参与格式化
                with patch('msmodelslim.model.wan2_1.model_adapter.time') as mock_time:
                    mock_time.time.side_effect = [1.0, 3.0]  # begin=1.0, end=3.0
                    yield
