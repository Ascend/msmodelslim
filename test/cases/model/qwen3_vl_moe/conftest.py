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
# pylint: disable=no-name-in-module,pointless-string-statement

"""
Pytest config for qwen3_vl_moe tests.
"""

import sys  # noqa: E402
import types  # noqa: E402
from unittest.mock import MagicMock  # noqa: E402

import pytest  # noqa: E402, F401

# 记录创建的模块和原始模块，用于 pytest_unconfigure 中清理
_created_modules = {}
_original_modules = {}


def _setup_mock_modules():
    """设置 transformers 相关模块的 mock，记录创建的模块以便后续清理"""
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig  # noqa: F401
    from transformers import PreTrainedTokenizerBase, PreTrainedModel, PretrainedConfig  # noqa: F401
    from transformers import models, activations  # noqa: F401

    _original_modules['transformers'] = sys.modules['transformers']
    transformers_module = sys.modules['transformers']

    required_attrs = {
        'AutoProcessor': MagicMock(),
        'Qwen3VLMoeForConditionalGeneration': MagicMock(),
    }

    for attr_name, attr_value in required_attrs.items():
        if not hasattr(transformers_module, attr_name):
            setattr(transformers_module, attr_name, attr_value)

    # transformers.masking_utils（缺失的依赖，必须在 transformers 之后设置）
    if 'transformers.masking_utils' not in sys.modules:
        _original_modules['transformers.masking_utils'] = None
        # 创建真实的模块对象，并添加 create_causal_mask 属性
        masking_utils = types.ModuleType('transformers.masking_utils')
        masking_utils.create_causal_mask = MagicMock()
        sys.modules['transformers.masking_utils'] = masking_utils
        _created_modules['transformers.masking_utils'] = masking_utils
    else:
        _original_modules['transformers.masking_utils'] = sys.modules['transformers.masking_utils']
        masking_utils = sys.modules['transformers.masking_utils']

    # 关键：无论模块是否已存在，都要确保父模块有正确的属性引用
    setattr(transformers_module, 'masking_utils', masking_utils)

    _original_modules['transformers.models'] = sys.modules['transformers.models']
    models_module = sys.modules['transformers.models']

    if 'transformers.models.qwen3_vl_moe' not in sys.modules:
        _original_modules['transformers.models.qwen3_vl_moe'] = None
        qwen3_vl_moe_module = types.ModuleType('transformers.models.qwen3_vl_moe')
        sys.modules['transformers.models.qwen3_vl_moe'] = qwen3_vl_moe_module
        setattr(models_module, 'qwen3_vl_moe', qwen3_vl_moe_module)
        _created_modules['transformers.models.qwen3_vl_moe'] = qwen3_vl_moe_module
    else:
        _original_modules['transformers.models.qwen3_vl_moe'] = sys.modules['transformers.models.qwen3_vl_moe']
        qwen3_vl_moe_module = sys.modules['transformers.models.qwen3_vl_moe']

    # modeling_qwen3_vl_moe 模块并添加必需的类
    if 'transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe' not in sys.modules:

        def _make_modeling():
            m = types.ModuleType('transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe')

            class MockQwen3VLMoeTextDecoderLayer:
                pass

            class MockQwen3VLMoeTextModel:
                pass

            class MockQwen3VLMoeTextSparseMoeBlock:
                pass

            m.Qwen3VLMoeTextDecoderLayer = MockQwen3VLMoeTextDecoderLayer
            m.Qwen3VLMoeTextModel = MockQwen3VLMoeTextModel
            m.Qwen3VLMoeTextSparseMoeBlock = MockQwen3VLMoeTextSparseMoeBlock
            return m

        _original_modules['transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe'] = sys.modules.get(
            'transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe'
        )
        mock_modeling = _make_modeling()
        sys.modules['transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe'] = mock_modeling
        setattr(qwen3_vl_moe_module, 'modeling_qwen3_vl_moe', mock_modeling)
        _created_modules['transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe'] = mock_modeling
    else:
        _original_modules['transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe'] = sys.modules[
            'transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe'
        ]

    # configuration_qwen3_vl_moe 模块
    if 'transformers.models.qwen3_vl_moe.configuration_qwen3_vl_moe' not in sys.modules:
        _original_modules['transformers.models.qwen3_vl_moe.configuration_qwen3_vl_moe'] = sys.modules.get(
            'transformers.models.qwen3_vl_moe.configuration_qwen3_vl_moe'
        )
        mock_config = types.ModuleType('transformers.models.qwen3_vl_moe.configuration_qwen3_vl_moe')
        sys.modules['transformers.models.qwen3_vl_moe.configuration_qwen3_vl_moe'] = mock_config
        setattr(qwen3_vl_moe_module, 'configuration_qwen3_vl_moe', mock_config)
        _created_modules['transformers.models.qwen3_vl_moe.configuration_qwen3_vl_moe'] = mock_config
    else:
        _original_modules['transformers.models.qwen3_vl_moe.configuration_qwen3_vl_moe'] = sys.modules[
            'transformers.models.qwen3_vl_moe.configuration_qwen3_vl_moe'
        ]


_setup_mock_modules()


def pytest_configure(config):
    """
    pytest 配置阶段再次确保 mock 已设置。
    这确保在导入测试模块之前，所有必需的 mock 都已就位。
    """
    _setup_mock_modules()


def pytest_unconfigure(config):
    """
    在测试结束后清理 mock，恢复原始模块（如果存在）。
    仅清理在本 conftest 中新创建的模块，避免影响外部真实依赖。
    """
    for module_name in _created_modules:
        if module_name in sys.modules:
            del sys.modules[module_name]
        # 恢复原始模块（如果存在）
        if _original_modules.get(module_name) is not None:
            sys.modules[module_name] = _original_modules[module_name]
