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
# pylint: disable=attribute-defined-outside-init,pointless-string-statement

"""
Unit tests for `msmodelslim.model.common.vlm_base.VLMBaseModelAdapter`.

这些用例主要覆盖：
- 初始化流程（配置加载、model_type 和 model_pedigree 的获取）
- 设备移动辅助方法（_maybe_to_device）
- KV cache 控制（_enable_kv_cache）
- 配置加载（_load_config）
- 模型类型和谱系解析（_get_model_type, _get_model_pedigree）
- 输入收集与设备移动（_collect_inputs_to_device）
"""

from pathlib import Path  # noqa: E402
from unittest.mock import Mock, patch  # noqa: E402

import pytest  # noqa: E402
import torch  # noqa: E402

from torch import nn  # noqa: E402, F401
from msmodelslim.core.const import DeviceType  # noqa: E402
from msmodelslim.model.common.vlm_base import VLMBaseModelAdapter  # noqa: E402
from msmodelslim.utils.exception import SchemaValidateError  # noqa: E402


class DummyConfig:
    """模拟配置对象"""

    def __init__(self, model_type="qwen"):
        self.model_type = model_type
        self.use_cache = False


class TestVLMBaseModelAdapterInit:
    """测试 VLMBaseModelAdapter 的初始化流程"""

    @patch('msmodelslim.model.common.vlm_base.SafeGenerator.get_config_from_pretrained')
    def test_init_loads_config_and_sets_attributes(self, mock_get_config):
        """验证初始化时会加载配置并设置 model_type 和 model_pedigree"""
        mock_config = DummyConfig(model_type="qwen")
        mock_get_config.return_value = mock_config

        adapter = VLMBaseModelAdapter(model_type="qwen-vl", model_path=Path("/tmp/model"), trust_remote_code=True)

        # 验证配置加载
        mock_get_config.assert_called_once_with(model_path=str(Path("/tmp/model")), trust_remote_code=True)
        # 验证属性设置
        assert adapter.config == mock_config
        assert adapter.model_type == "qwen-vl"
        assert adapter.model_pedigree == "qwen"

    @patch('msmodelslim.model.common.vlm_base.SafeGenerator.get_config_from_pretrained')
    def test_init_with_none_model_type_uses_config(self, mock_get_config):
        """验证当 model_type 为 None 时，使用配置中的 model_type"""
        mock_config = DummyConfig(model_type="custom_model")
        mock_get_config.return_value = mock_config

        adapter = VLMBaseModelAdapter(model_type=None, model_path=Path("/tmp/model"), trust_remote_code=False)

        assert adapter.model_type == "custom_model"
        assert adapter.model_pedigree == "custom_model"


class TestMaybeToDevice:
    """测试 _maybe_to_device 静态方法"""

    def test_maybe_to_device_with_none_returns_none(self):
        """验证传入 None 时返回 None"""
        result = VLMBaseModelAdapter._maybe_to_device(None, "cpu")
        assert result is None

    def test_maybe_to_device_with_tensor_moves_to_device(self):
        """验证传入 tensor 时会移动到目标设备"""
        tensor = torch.tensor([1.0, 2.0])
        result = VLMBaseModelAdapter._maybe_to_device(tensor, "cpu")
        assert isinstance(result, torch.Tensor)
        assert result.device.type == "cpu"

    def test_maybe_to_device_with_non_tensor_returns_original(self):
        """验证传入非 tensor 对象时返回原值（即使 to() 失败）"""
        non_tensor = {"key": "value"}
        result = VLMBaseModelAdapter._maybe_to_device(non_tensor, "cpu")
        assert result == non_tensor

    def test_maybe_to_device_with_exception_returns_original(self):
        """验证当 to() 抛出异常时返回原值"""
        mock_obj = Mock()
        mock_obj.to.side_effect = RuntimeError("Device error")

        result = VLMBaseModelAdapter._maybe_to_device(mock_obj, "cpu")
        assert result == mock_obj


class TestEnableKvCache:
    """测试 _enable_kv_cache 方法"""

    @patch('msmodelslim.model.common.vlm_base.SafeGenerator.get_config_from_pretrained')
    def test_enable_kv_cache_sets_use_cache_true(self, mock_get_config):
        """验证启用 KV cache 时设置 use_cache=True"""
        mock_config = DummyConfig()
        mock_get_config.return_value = mock_config

        adapter = VLMBaseModelAdapter("test", Path("/tmp"), False)
        mock_model = Mock()
        mock_model.model = Mock()
        mock_model.model.config = mock_config

        adapter._enable_kv_cache(mock_model, enable=True)

        assert mock_model.model.config.use_cache is True

    @patch('msmodelslim.model.common.vlm_base.SafeGenerator.get_config_from_pretrained')
    def test_enable_kv_cache_sets_use_cache_false(self, mock_get_config):
        """验证禁用 KV cache 时设置 use_cache=False"""
        mock_config = DummyConfig()
        mock_get_config.return_value = mock_config

        adapter = VLMBaseModelAdapter("test", Path("/tmp"), False)
        mock_model = Mock()
        mock_model.model = Mock()
        mock_model.model.config = mock_config

        adapter._enable_kv_cache(mock_model, enable=False)

        assert mock_model.model.config.use_cache is False


class TestLoadConfig:
    """测试 _load_config 方法"""

    @patch('msmodelslim.model.common.vlm_base.SafeGenerator.get_config_from_pretrained')
    def test_load_config_delegates_to_safe_generator(self, mock_get_config):
        """验证 _load_config 委托给 SafeGenerator"""
        mock_config = DummyConfig()
        mock_get_config.return_value = mock_config

        adapter = VLMBaseModelAdapter("test", Path("/tmp/model"), False)
        # __init__ 已调用过一次 SafeGenerator，重置计数以便只验证本次调用
        mock_get_config.reset_mock()

        result = adapter._load_config(trust_remote_code=True)

        assert result == mock_config
        mock_get_config.assert_called_once_with(model_path=str(Path("/tmp/model")), trust_remote_code=True)


class TestGetModelType:
    """测试 _get_model_type 方法"""

    @patch('msmodelslim.model.common.vlm_base.SafeGenerator.get_config_from_pretrained')
    def test_get_model_type_with_none_returns_config_type(self, mock_get_config):
        """验证当 model_type 为 None 时返回配置中的 model_type"""
        mock_config = DummyConfig(model_type="config_model")
        mock_get_config.return_value = mock_config

        adapter = VLMBaseModelAdapter("test", Path("/tmp"), False)
        result = adapter._get_model_type(None)

        assert result == "config_model"

    @patch('msmodelslim.model.common.vlm_base.SafeGenerator.get_config_from_pretrained')
    def test_get_model_type_with_value_returns_value(self, mock_get_config):
        """验证当 model_type 有值时返回该值"""
        mock_config = DummyConfig()
        mock_get_config.return_value = mock_config

        adapter = VLMBaseModelAdapter("test", Path("/tmp"), False)
        result = adapter._get_model_type("custom_type")

        assert result == "custom_type"


class TestGetModelPedigree:
    """测试 _get_model_pedigree 方法"""

    @patch('msmodelslim.model.common.vlm_base.SafeGenerator.get_config_from_pretrained')
    def test_get_model_pedigree_with_none_returns_config_type(self, mock_get_config):
        """验证当 model_type 为 None 时返回配置中的 model_type"""
        mock_config = DummyConfig(model_type="config_pedigree")
        mock_get_config.return_value = mock_config

        adapter = VLMBaseModelAdapter("test", Path("/tmp"), False)
        result = adapter._get_model_pedigree(None)

        assert result == "config_pedigree"

    @patch('msmodelslim.model.common.vlm_base.SafeGenerator.get_config_from_pretrained')
    def test_get_model_pedigree_with_valid_match_returns_lowercase(self, mock_get_config):
        """验证当 model_type 匹配正则时返回小写的匹配部分"""
        mock_config = DummyConfig()
        mock_get_config.return_value = mock_config

        adapter = VLMBaseModelAdapter("test", Path("/tmp"), False)
        result = adapter._get_model_pedigree("Qwen-VL-MoE")

        assert result == "qwen"

    @patch('msmodelslim.model.common.vlm_base.SafeGenerator.get_config_from_pretrained')
    def test_get_model_pedigree_with_no_match_raises_error(self, mock_get_config):
        """验证当 model_type 不匹配正则时抛出 SchemaValidateError"""
        mock_config = DummyConfig()
        mock_get_config.return_value = mock_config

        adapter = VLMBaseModelAdapter("test", Path("/tmp"), False)

        with pytest.raises(SchemaValidateError) as exc_info:
            adapter._get_model_pedigree("123-invalid")

        assert "Invalid model_name" in str(exc_info.value)
        assert exc_info.value.action == "Please check the model type"

    @patch('msmodelslim.model.common.vlm_base.SafeGenerator.get_config_from_pretrained')
    def test_get_model_pedigree_with_empty_string_raises_error(self, mock_get_config):
        """验证当 model_type 为空字符串时抛出 SchemaValidateError"""
        mock_config = DummyConfig()
        mock_get_config.return_value = mock_config

        adapter = VLMBaseModelAdapter("test", Path("/tmp"), False)

        with pytest.raises(SchemaValidateError) as exc_info:
            adapter._get_model_pedigree("")

        assert "Invalid model_name" in str(exc_info.value)


class TestCollectInputsToDevice:
    """测试 _collect_inputs_to_device 方法"""

    @patch('msmodelslim.model.common.vlm_base.SafeGenerator.get_config_from_pretrained')
    def test_collect_inputs_to_device_with_device_type_enum(self, mock_get_config):
        """验证使用 DeviceType 枚举时正确提取 value"""
        mock_config = DummyConfig()
        mock_get_config.return_value = mock_config

        adapter = VLMBaseModelAdapter("test", Path("/tmp"), False)
        mock_inputs = Mock()
        mock_inputs.key1 = torch.tensor([1.0])
        mock_inputs.key2 = None

        result = adapter._collect_inputs_to_device(
            mock_inputs, DeviceType.CPU, ["key1", "key2"], defaults={"key2": "default_value"}
        )

        assert "key1" in result
        assert isinstance(result["key1"], torch.Tensor)
        assert result["key2"] == "default_value"

    @patch('msmodelslim.model.common.vlm_base.SafeGenerator.get_config_from_pretrained')
    def test_collect_inputs_to_device_with_string_device(self, mock_get_config):
        """验证使用字符串设备时直接使用"""
        mock_config = DummyConfig()
        mock_get_config.return_value = mock_config

        adapter = VLMBaseModelAdapter("test", Path("/tmp"), False)
        mock_inputs = Mock()
        mock_inputs.key1 = torch.tensor([1.0])

        result = adapter._collect_inputs_to_device(mock_inputs, "cuda:0", ["key1"])

        assert "key1" in result
        assert isinstance(result["key1"], torch.Tensor)

    @patch('msmodelslim.model.common.vlm_base.SafeGenerator.get_config_from_pretrained')
    def test_collect_inputs_to_device_with_none_and_no_defaults(self, mock_get_config):
        """验证当属性为 None 且无默认值时返回 None"""
        mock_config = DummyConfig()
        mock_get_config.return_value = mock_config

        adapter = VLMBaseModelAdapter("test", Path("/tmp"), False)
        mock_inputs = Mock()
        mock_inputs.key1 = None

        result = adapter._collect_inputs_to_device(mock_inputs, "cpu", ["key1"])

        assert result["key1"] is None

    @patch('msmodelslim.model.common.vlm_base.SafeGenerator.get_config_from_pretrained')
    def test_collect_inputs_to_device_with_multiple_keys(self, mock_get_config):
        """验证收集多个键时都能正确处理"""
        mock_config = DummyConfig()
        mock_get_config.return_value = mock_config

        adapter = VLMBaseModelAdapter("test", Path("/tmp"), False)
        mock_inputs = Mock()
        mock_inputs.key1 = torch.tensor([1.0])
        mock_inputs.key2 = torch.tensor([2.0])
        mock_inputs.key3 = None

        result = adapter._collect_inputs_to_device(
            mock_inputs, "cpu", ["key1", "key2", "key3"], defaults={"key3": "default"}
        )

        assert len(result) == 3
        assert "key1" in result
        assert "key2" in result
        assert result["key3"] == "default"


class TestVLMBaseModelAdapterGetGlobalModelTorchDtype:
    """Tests for VLMBaseModelAdapter.get_global_model_torch_dtype (AscendV1GlobalModelDtypeInterface)."""

    @patch('msmodelslim.model.common.vlm_base.SafeGenerator.get_config_from_pretrained')
    def test_config_torch_dtype_none_returns_float32(self, mock_get_config):
        mock_config = DummyConfig()
        mock_get_config.return_value = mock_config
        adapter = VLMBaseModelAdapter("test", Path("/tmp"), False)
        assert adapter.get_global_model_torch_dtype() == torch.float32

    @patch('msmodelslim.model.common.vlm_base.SafeGenerator.get_config_from_pretrained')
    def test_config_torch_dtype_bfloat16_returns_bfloat16(self, mock_get_config):
        mock_config = DummyConfig()
        mock_config.torch_dtype = torch.bfloat16
        mock_get_config.return_value = mock_config
        adapter = VLMBaseModelAdapter("test", Path("/tmp"), False)
        assert adapter.get_global_model_torch_dtype() == torch.bfloat16

    @patch('msmodelslim.model.common.vlm_base.SafeGenerator.get_config_from_pretrained')
    def test_text_config_torch_dtype_fallback(self, mock_get_config):
        mock_config = DummyConfig()
        mock_config.torch_dtype = None
        mock_config.text_config = Mock()
        mock_config.text_config.torch_dtype = "bfloat16"
        mock_config.vision_config = None
        mock_get_config.return_value = mock_config
        adapter = VLMBaseModelAdapter("test", Path("/tmp"), False)
        assert adapter.get_global_model_torch_dtype() == torch.bfloat16
