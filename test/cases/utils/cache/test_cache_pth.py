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

import pytest
import os
from unittest.mock import Mock, patch

from msmodelslim.utils.cache.pth import load_cached_data, InputCapture, DumperManager, to_device
from msmodelslim.utils.exception import SchemaValidateError


class TestLoadCachedData:
    """测试load_cached_data函数"""

    @pytest.fixture
    def mock_pth_file_path(self):
        """创建模拟的PTH文件路径"""
        return os.path.join("test", "cache", "calib_data.pth")

    @pytest.fixture
    def mock_generate_func(self):
        """创建模拟的生成函数"""
        return Mock()

    @pytest.fixture
    def mock_model(self):
        """创建模拟的模型"""
        return Mock()

    @pytest.fixture
    def mock_dump_config(self):
        """创建模拟的dump配置"""
        config = Mock()
        config.capture_mode = "args"
        return config

    def test_load_cached_data_file_exists(self, mock_pth_file_path, mock_generate_func, mock_model, mock_dump_config):
        """测试缓存文件存在时的加载"""
        # Mock文件存在
        with patch('os.path.exists', return_value=True):
            # Mock safe_torch_load
            with patch('msmodelslim.utils.cache.pth.safe_torch_load') as mock_load:
                mock_data = {"test": "data"}
                mock_load.return_value = mock_data

                # Mock get_valid_read_path
                with patch('msmodelslim.utils.cache.pth.get_valid_read_path') as mock_valid_path:
                    mock_valid_path.return_value = mock_pth_file_path

                    # Mock logger
                    with patch('msmodelslim.utils.cache.pth.get_logger'):
                        result = load_cached_data(mock_pth_file_path, mock_generate_func, mock_model, mock_dump_config)

                        assert result == mock_data
                        mock_load.assert_called_once_with(mock_pth_file_path)

    def test_load_cached_data_file_not_exists(
        self, mock_pth_file_path, mock_generate_func, mock_model, mock_dump_config
    ):
        """测试缓存文件不存在时的处理"""
        # Mock文件不存在
        with patch('os.path.exists', return_value=False):
            # Mock DumperManager
            with patch('msmodelslim.utils.cache.pth.DumperManager') as mock_dumper_class:
                mock_dumper = Mock()
                mock_dumper_class.return_value = mock_dumper

                # Mock safe_torch_load
                with patch('msmodelslim.utils.cache.pth.safe_torch_load') as mock_load:
                    mock_data = {"generated": "data"}
                    mock_load.return_value = mock_data

                    # Mock logger
                    with patch('msmodelslim.utils.cache.pth.get_logger'):
                        result = load_cached_data(mock_pth_file_path, mock_generate_func, mock_model, mock_dump_config)

                        assert result == mock_data
                        # 验证生成函数被调用
                        mock_generate_func.assert_called_once()
                        # 验证dump管理器被创建和保存
                        mock_dumper_class.assert_called_once_with(mock_model, capture_mode="args")
                        mock_dumper.save.assert_called_once_with(mock_pth_file_path)


class TestInputCapture:
    """测试InputCapture类（实例级捕获，非全局 ContextVar）"""

    def test_input_capture_reset(self):
        """测试InputCapture的reset方法"""
        capture = InputCapture()
        capture.add_record({"test": "data"})
        assert len(capture.get_all()) > 0

        capture.reset()
        assert len(capture.get_all()) == 0

    def test_input_capture_add_and_get_record(self):
        """测试InputCapture的add_record和get_all方法"""
        capture = InputCapture()

        test_record = {"test_key": "test_value"}
        capture.add_record(test_record)

        all_records = capture.get_all()
        assert len(all_records) == 1
        assert all_records[0] == test_record

    def test_input_capture_instances_are_isolated(self):
        """不同 InputCapture 实例互不共享状态"""
        a = InputCapture()
        b = InputCapture()
        a.add_record({"from": "a"})
        assert len(a.get_all()) == 1
        assert len(b.get_all()) == 0

    def test_input_capture_capture_forward_inputs_args_mode(self):
        """测试InputCapture的capture_forward_inputs方法（args模式）"""
        capture = InputCapture()

        def test_function(arg1, arg2, kwarg1="default"):
            return arg1 + arg2

        wrapped_func = capture.capture_forward_inputs(test_function, capture_mode="args")
        result = wrapped_func(10, 20, kwarg1="custom")

        assert result == 30
        captured = capture.get_all()
        assert len(captured) == 1
        assert len(captured[0]) >= 2

    def test_input_capture_capture_forward_inputs_method(self):
        """测试InputCapture的capture_forward_inputs方法（方法调用）"""
        capture = InputCapture()

        class TestClass:
            def test_method(self, arg1, arg2):
                return arg1 + arg2

        wrapped_method = capture.capture_forward_inputs(TestClass.test_method, capture_mode="args")
        obj = TestClass()
        result = wrapped_method(obj, 15, 25)

        assert result == 40
        captured = capture.get_all()
        assert len(captured) == 1
        assert len(captured[0]) >= 2

    def test_input_capture_capture_forward_inputs_invalid_mode(self):
        """无效 capture_mode 在调用包装函数时抛出 SchemaValidateError"""

        def test_function():
            pass

        capture = InputCapture()
        wrapped_func = capture.capture_forward_inputs(test_function, capture_mode="invalid_mode")
        with pytest.raises(SchemaValidateError, match="Invalid capture_mode"):
            wrapped_func()


class TestDumperManager:
    """测试DumperManager类"""

    @pytest.fixture
    def mock_module(self):
        """创建模拟的模块"""
        return Mock()

    @pytest.fixture
    def mock_dump_config(self):
        """创建模拟的dump配置"""
        config = Mock()
        config.capture_mode = "args"
        return config

    def test_dumper_manager_initialization(self, mock_module, mock_dump_config):
        """测试DumperManager的初始化"""
        dumper = DumperManager(mock_module, capture_mode="args")

        assert dumper.module is mock_module
        assert dumper.capture_mode == "args"
        assert dumper.old_forward is not None
        assert isinstance(dumper.input_capture, InputCapture)

    def test_dumper_manager_initialization_invalid_capture_mode(self, mock_module):
        """测试DumperManager使用无效capture_mode时的初始化"""
        with pytest.raises(SchemaValidateError, match="Invalid capture_mode: 'invalid_mode'"):
            DumperManager(mock_module, capture_mode="invalid_mode")

    def test_dumper_manager_save(self, mock_module, mock_dump_config):
        """测试DumperManager的save方法"""
        dumper = DumperManager(mock_module, capture_mode="args")
        dumper.input_capture.add_record({"test": "data"})

        with (
            patch('msmodelslim.utils.cache.pth.torch.save') as mock_torch_save,
            patch('msmodelslim.utils.cache.pth.SafeWriteUmask') as mock_umask,
        ):
            mock_umask.return_value.__enter__ = Mock(return_value=None)
            mock_umask.return_value.__exit__ = Mock(return_value=None)
            with patch('msmodelslim.utils.cache.pth.get_logger'):
                dumper.save("/test/output.pth")

            mock_torch_save.assert_called_once()
            mock_umask.assert_called_once()
            assert dumper.old_forward is None

    def test_dumper_manager_save_restores_forward_on_failure(self, mock_module, mock_dump_config):
        """torch.save 失败时也应恢复 module.forward"""
        dumper = DumperManager(mock_module, capture_mode="args")
        hooked_forward = mock_module.forward
        original_forward = dumper.old_forward

        with (
            patch('msmodelslim.utils.cache.pth.torch.save', side_effect=OSError("disk full")),
            patch('msmodelslim.utils.cache.pth.SafeWriteUmask') as mock_umask,
        ):
            mock_umask.return_value.__enter__ = Mock(return_value=None)
            mock_umask.return_value.__exit__ = Mock(return_value=None)
            with pytest.raises(OSError, match="disk full"):
                dumper.save("/test/output.pth")

        assert mock_module.forward is original_forward
        assert dumper.old_forward is None
        assert hooked_forward is not original_forward

    def test_dumper_manager_reset(self, mock_module, mock_dump_config):
        """测试DumperManager的reset方法"""
        dumper = DumperManager(mock_module, capture_mode="args")
        dumper.input_capture.add_record({"test": "data"})
        assert len(dumper.input_capture.get_all()) > 0

        dumper.reset()
        assert len(dumper.input_capture.get_all()) == 0

    def test_dumper_manager_add_hook(self, mock_module, mock_dump_config):
        """测试DumperManager的_add_hook方法"""
        dumper = DumperManager(mock_module, capture_mode="args")

        assert mock_module.forward != dumper.old_forward
        assert hasattr(mock_module.forward, '__wrapped__')


class TestToDevice:
    """测试to_device函数"""

    @pytest.fixture
    def mock_torch(self):
        """Mock torch库"""
        with patch('msmodelslim.utils.cache.pth.torch') as mock_torch:
            mock_torch.Tensor = Mock
            yield mock_torch

    def test_to_device_dict(self, mock_torch):
        """测试to_device处理字典类型数据"""
        test_dict = {"key1": Mock(), "key2": Mock()}

        with patch('msmodelslim.utils.cache.pth.to_device') as mock_to_device:
            mock_to_device.return_value = "device_data"

            to_device(test_dict, "cpu")

            # 验证递归调用
            assert mock_to_device.call_count >= 2

    def test_to_device_list(self, mock_torch):
        """测试to_device处理列表类型数据"""
        test_list = [Mock(), Mock()]

        with patch('msmodelslim.utils.cache.pth.to_device') as mock_to_device:
            mock_to_device.return_value = "device_data"

            to_device(test_list, "cpu")

            # 验证递归调用
            assert mock_to_device.call_count >= 2

    def test_to_device_tuple(self, mock_torch):
        """测试to_device处理元组类型数据"""
        test_tuple = (Mock(), Mock())

        with patch('msmodelslim.utils.cache.pth.to_device') as mock_to_device:
            mock_to_device.return_value = "device_data"

            to_device(test_tuple, "cpu")

            # 验证递归调用
            assert mock_to_device.call_count >= 2

    def test_to_device_tensor(self, mock_torch):
        """测试to_device处理张量类型数据"""
        mock_tensor = Mock()
        mock_tensor.to.return_value = "moved_tensor"

        result = to_device(mock_tensor, "cpu")

        assert result == "moved_tensor"
        mock_tensor.to.assert_called_once_with("cpu")

    def test_to_device_other_types(self, mock_torch):
        """测试to_device处理其他类型数据"""
        test_data = "string_data"
        result = to_device(test_data, "cpu")
        assert result == test_data

    def test_to_device_recursion_depth_limit(self, mock_torch):
        """测试to_device的递归深度限制"""
        # 创建嵌套过深的数据结构
        deep_data = {}
        current = deep_data
        for i in range(25):  # 超过MAX_RECURSION_DEPTH (20)
            current["nested"] = {}
            current = current["nested"]

        with pytest.raises(RecursionError, match="Maximum recursion depth 20 exceeded"):
            to_device(deep_data, "cpu")
