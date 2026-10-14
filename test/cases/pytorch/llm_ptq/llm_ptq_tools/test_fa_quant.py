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
# pylint: disable=attribute-defined-outside-init,comparison-with-callable,consider-using-from-import,no-member,no-name-in-module,redefined-outer-name

import pytest
import logging
from unittest.mock import MagicMock, patch
from transformers import PretrainedConfig
import torch.nn as nn
from types import SimpleNamespace

from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.fa_quant import (
    TensorType,
    FAQuantizer,
    is_attn_module_and_then_check_quantizer,
    install_fa_quantizer,
)
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.fa_quant_adapter import ForwardFactory, AttentionType

from msmodelslim import logger


# Fixtures for common test objects
@pytest.fixture
def mock_config():
    config = MagicMock(spec=PretrainedConfig)
    config.model_type = "test_model"
    config.num_attention_heads = 8
    config.hidden_size = 64
    config.num_key_value_heads = 8
    return config


@pytest.fixture
def mock_logger():
    logger = MagicMock(spec=logging.Logger)
    logger.warning = MagicMock()
    return logger


@pytest.fixture
def fa_quantizer(mock_config, mock_logger):
    return FAQuantizer(mock_config, mock_logger)


# Test cases for FAQuantizer initialization
class TestConfig:
    pass


def test_init_given_valid_config_when_mha_then_success(mock_logger):
    config = TestConfig()
    config.num_attention_heads = 8
    config.hidden_size = 64

    quantizer = FAQuantizer(config, mock_logger)

    assert quantizer.num_head == 8
    assert quantizer.num_kv_head == 8
    assert quantizer.head_dim == 8  # 64/8
    assert not quantizer.is_calib
    assert not quantizer.dequant_infer
    mock_logger.warning.assert_called_once()


def test_init_given_valid_config_when_gqa_then_success(mock_logger):
    config = TestConfig()
    config.num_attention_heads = 8
    config.hidden_size = 64
    config.num_key_value_heads = 4

    quantizer = FAQuantizer(config, mock_logger)

    assert quantizer.num_head == 8
    assert quantizer.num_kv_head == 4
    assert quantizer.head_dim == 8  # 64/8
    mock_logger.warning.assert_not_called()


def test_init_given_missing_required_attr_when_heads_then_fail(mock_logger):
    config = TestConfig()
    config.hidden_size = 64

    with pytest.raises(AttributeError) as excinfo:
        FAQuantizer(config, mock_logger)
    assert "num_attention_heads" in str(excinfo.value)


def test_init_given_missing_required_attr_when_hidden_size_then_fail(mock_logger):
    config = TestConfig()
    config.num_attention_heads = 8

    with pytest.raises(AttributeError) as excinfo:
        FAQuantizer(config, mock_logger)
    assert "hidden_size" in str(excinfo.value)


def test_init_given_invalid_type_when_heads_then_fail(mock_logger):
    config = TestConfig()
    config.num_attention_heads = "8"
    config.hidden_size = 64

    with pytest.raises(TypeError) as excinfo:
        FAQuantizer(config, mock_logger)
    assert "num_attention_heads" in str(excinfo.value)


def test_init_given_invalid_type_when_hidden_size_then_fail(mock_logger):
    config = TestConfig()
    config.num_attention_heads = 8
    config.hidden_size = "64"

    with pytest.raises(TypeError) as excinfo:
        FAQuantizer(config, mock_logger)
    assert "hidden_size" in str(excinfo.value)


def test_init_given_invalid_type_when_kv_heads_then_fail(mock_logger):
    config = TestConfig()
    config.num_attention_heads = 8
    config.hidden_size = 64
    config.num_key_value_heads = "4"

    with pytest.raises(TypeError) as excinfo:
        FAQuantizer(config, mock_logger)
    assert "num_key_value_heads" in str(excinfo.value)


# Test cases for install_fa_quantizer
def test_install_fa_quantizer_given_model_when_called_then_installs_quantizers(mock_config, mock_logger):
    # 创建真实的测试模块
    class TestAttention:
        def forward(self, *args, **kwargs):
            return "original_result"

    # 准备测试数据
    model = MagicMock(spec=nn.Module)
    attn_module = TestAttention()
    model.named_modules.return_value = [("attn", attn_module)]
    mock_config.model_type = "test_model"

    # 记录原始forward方法
    original_forward = attn_module.forward

    # 创建真实的适配器函数
    def real_adapter(original_fn):
        def wrapped(*args, **kwargs):
            return f"adapted_{original_fn(*args, **kwargs)}"

        return wrapped

    # 使用真实FAQuantizer类进行测试
    with (
        patch.dict(
            'msmodelslim.pytorch.llm_ptq.llm_ptq_tools.fa_quant_adapter.ForwardFactory._forward_adapters',
            {('test_model', 'mha'): real_adapter},
        ),
        patch('msmodelslim.pytorch.llm_ptq.llm_ptq_tools.fa_quant.FAQuantizer') as mock_quantizer_class,
    ):
        # 配置mock quantizer
        mock_quantizer_instance = MagicMock()
        mock_quantizer_class.return_value = mock_quantizer_instance

        # 执行测试
        install_fa_quantizer(model, mock_config, mock_logger)

        # 验证点1: 检查是否调用了FAQuantizer构造函数
        mock_quantizer_class.assert_called_once_with(mock_config, logger=mock_logger)

        # 验证点2: 检查quantizer实例是否被正确附加
        assert hasattr(attn_module, 'fa_quantizer')
        assert attn_module.fa_quantizer is mock_quantizer_instance

        # 验证点3: 检查forward方法是否被正确包装
        assert attn_module.forward != original_forward
        assert attn_module.forward() == "adapted_original_result"

        # 修正的验证点4: 检查正确的日志记录
        expected_log = "Successfully installed FAQuantizer for module attn"
        mock_logger.info.assert_called_once_with(expected_log)
        mock_logger.warning.assert_not_called()  # 确保没有警告日志


def test_install_fa_quantizer_should_warn_when_already_installed(mock_config, mock_logger):
    model = MagicMock(spec=nn.Module)
    attn_module = MagicMock()
    attn_module.fa_quantizer = MagicMock()
    attn_module.__class__.__name__ = "Attention"
    model.named_modules.return_value = [("attn", attn_module)]

    install_fa_quantizer(model, mock_config, mock_logger)
    mock_logger.warning.assert_called_with("Module attn already has FAQuantizer installed.")


class MockMMDoubleStreamBlock(nn.Module):
    def __init__(self, heads_num, hidden_size, sp_size):
        super().__init__()
        self.heads_num = heads_num
        self.hidden_size = hidden_size
        self.txt_mlp = MagicMock()  # 模拟 MLP
        self.hybrid_seq_parallel_attn = None
        self.cache = None

        # 模拟 FAQuantizer 的初始化
        config_dict = {
            'num_attention_heads': self.heads_num // sp_size,
            'hidden_size': self.hidden_size // sp_size,
            'num_key_value_heads': self.heads_num // sp_size,
        }
        config = SimpleNamespace(**config_dict)
        self.fa_quantizer = FAQuantizer(config, logger=logger)


def _test_install_fa_quantizer_for_class(class_name, mock_config, mock_logger):
    """
    辅助函数：测试指定类名的模块是否正确安装了 fa_quantizer。
    """
    module_classes = {
        "MMDoubleStreamBlock": MockMMDoubleStreamBlock,
    }

    # 创建模拟的模块实例
    if class_name == "MMDoubleStreamBlock":
        attn_module = module_classes[class_name](heads_num=8, hidden_size=512, sp_size=1)
    else:
        attn_module = module_classes[class_name]()

    model = MagicMock(spec=nn.Module)
    model.named_modules.return_value = [("attn", attn_module)]

    with patch(
        'msmodelslim.pytorch.llm_ptq.llm_ptq_tools.fa_quant_adapter.ForwardFactory.get_forward_adapter'
    ) as mock_adapter:
        mock_adapter.return_value = lambda x: x
        install_fa_quantizer(model, mock_config, mock_logger)

        # 检查 fa_quantizer 是否被正确设置
        assert hasattr(attn_module, "fa_quantizer"), f"fa_quantizer not found for class {class_name}"
        assert attn_module.fa_quantizer is not None, f"fa_quantizer is None for class {class_name}"
        assert not isinstance(attn_module.fa_quantizer, MagicMock), (
            f"fa_quantizer is a MagicMock object for class {class_name}"
        )


def test_install_fa_quantizer_should_match_different_module_types(mock_config, mock_logger):
    """
    测试不同模块类型是否正确安装了 fa_quantizer。
    """
    module_classes = [
        "MMDoubleStreamBlock",
    ]

    for class_name in module_classes:
        _test_install_fa_quantizer_for_class(class_name, mock_config, mock_logger)


# Test cases for ForwardFactory
def test_forward_factory_register_given_decorator_when_called_then_registers_adapter():
    ForwardFactory._forward_adapters = {}  # Reset

    @ForwardFactory.register("test_model", "test_attn")
    def test_adapter():
        pass

    assert ("test_model", "test_attn") in ForwardFactory._forward_adapters


def test_forward_factory_get_forward_adapter_given_registered_adapter_when_called_then_returns_adapter():
    ForwardFactory._forward_adapters = {}  # Reset

    def test_adapter():
        pass

    ForwardFactory._forward_adapters[("test_model", "test_attn")] = test_adapter
    assert ForwardFactory.get_forward_adapter("test_model", "test_attn") == test_adapter


# Test cases for enums
def test_attention_type_enum_values_when_accessed_then_returns_correct_values():
    assert AttentionType.MHA.value == "mha"
    assert AttentionType.MQA.value == "mqa"
    assert AttentionType.GQA.value == "gqa"
    assert AttentionType.MLA.value == "mla"


def test_tensor_type_enum_values_when_accessed_then_returns_correct_values():
    assert TensorType.Q.value == "q"
    assert TensorType.K.value == "k"
    assert TensorType.V.value == "v"
    assert set(TensorType.get_values()) == {"q", "k", "v"}


# Test cases for is_attn_module_and_then_check_quantizer
@pytest.mark.parametrize(
    "class_name, add_quantizer, expected, should_raise",
    [
        ("Attention", True, True, False),
        ("MMSingleStreamBlock", True, True, False),
        ("MMDoubleStreamBlock", True, True, False),
        ("Linear", False, False, False),
        ("CustomAttentionLayer", True, False, True),  # 如果有量化器但不是正确类型，应该抛出异常
    ],
)
def test_is_attn_module_check_module_class_name(
    class_name, add_quantizer, expected, should_raise, mock_config, mock_logger
):
    module = MagicMock()
    module.__class__.__name__ = class_name

    if add_quantizer:
        # 对于应该抛出异常的情况，设置错误的量化器类型
        if should_raise:
            module.fa_quantizer = "invalid_type"
        else:
            module.fa_quantizer = FAQuantizer(mock_config, mock_logger)

    if should_raise:
        with pytest.raises(AttributeError):
            is_attn_module_and_then_check_quantizer(module, "test_module")
    else:
        assert is_attn_module_and_then_check_quantizer(module, "test_module") == expected


def test_is_attn_module_with_invalid_quantizer_type_should_raise_error():
    module = MagicMock()
    module.__class__.__name__ = "Attention"
    module.fa_quantizer = "invalid_type"

    with pytest.raises(AttributeError) as excinfo:
        is_attn_module_and_then_check_quantizer(module, "test_module")

    assert "`FAQuantizer` is not detected" in str(excinfo.value)
    assert "check the modeling file" in str(excinfo.value)


def test_is_attn_module_with_valid_quantizer_should_return_true(mock_config, mock_logger):
    module = MagicMock()
    module.__class__.__name__ = "Attention"
    module.fa_quantizer = FAQuantizer(mock_config, mock_logger)

    assert is_attn_module_and_then_check_quantizer(module, "test_module")
