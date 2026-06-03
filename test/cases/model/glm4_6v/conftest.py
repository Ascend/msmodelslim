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

import sys
import types
from unittest.mock import MagicMock

_created_modules = {}
_original_modules = {}


def _setup_mock_modules():
    """注入 transformers 相关 mock，确保在导入 model_adapter 前生效"""

    _original_modules["transformers"] = sys.modules["transformers"]
    transformers_module = sys.modules["transformers"]

    required_attrs = {
        "Glm4vMoeForConditionalGeneration": MagicMock(),
    }

    for attr_name, attr_value in required_attrs.items():
        if not hasattr(transformers_module, attr_name):
            setattr(transformers_module, attr_name, attr_value)

    _original_modules["transformers.models"] = sys.modules["transformers.models"]
    models_module = sys.modules["transformers.models"]

    if "transformers.models.glm4v_moe" not in sys.modules:
        _original_modules["transformers.models.glm4v_moe"] = None
        glm4v_moe_module = types.ModuleType("transformers.models.glm4v_moe")
        sys.modules["transformers.models.glm4v_moe"] = glm4v_moe_module
        setattr(models_module, "glm4v_moe", glm4v_moe_module)
        _created_modules["transformers.models.glm4v_moe"] = glm4v_moe_module
    else:
        _original_modules["transformers.models.glm4v_moe"] = sys.modules["transformers.models.glm4v_moe"]
        glm4v_moe_module = sys.modules["transformers.models.glm4v_moe"]

    if 'transformers.models.glm4v_moe.modeling_glm4v_moe' not in sys.modules:
        _original_modules['transformers.models.glm4v_moe.modeling_glm4v_moe'] = sys.modules.get(
            'transformers.models.glm4v_moe.modeling_glm4v_moe'
        )
        modeling = types.ModuleType('transformers.models.glm4v_moe.modeling_glm4v_moe')
        # Mock Glm4vMoeTextDecoderLayer
        modeling.Glm4vMoeTextDecoderLayer = MagicMock()
        # Mock Glm4vMoeTextMoE
        modeling.Glm4vMoeTextMoE = MagicMock()
        # Mock Glm4vMoeTextNaiveMoe
        modeling.Glm4vMoeTextNaiveMoe = MagicMock()
        sys.modules['transformers.models.glm4v_moe.modeling_glm4v_moe'] = modeling
        setattr(glm4v_moe_module, 'modeling_glm4v_moe', modeling)
        _created_modules['transformers.models.glm4v_moe.modeling_glm4v_moe'] = modeling
    else:
        modeling = sys.modules['transformers.models.glm4v_moe.modeling_glm4v_moe']
        if not hasattr(modeling, 'Glm4vMoeTextDecoderLayer'):
            modeling.Glm4vMoeTextDecoderLayer = MagicMock()
        if not hasattr(modeling, 'Glm4vMoeTextMoE'):
            modeling.Glm4vMoeTextMoE = MagicMock()
        if not hasattr(modeling, 'Glm4vMoeTextNaiveMoe'):
            modeling.Glm4vMoeTextNaiveMoe = MagicMock()

    if "transformers.masking_utils" not in sys.modules:
        _original_modules["transformers.masking_utils"] = None
        masking_utils_module = types.ModuleType("transformers.masking_utils")
        masking_utils_module.create_causal_mask = MagicMock()
        masking_utils_module.create_sliding_window_causal_mask = MagicMock()
        sys.modules["transformers.masking_utils"] = masking_utils_module
        setattr(transformers_module, "masking_utils", masking_utils_module)
        _created_modules["transformers.masking_utils"] = masking_utils_module
    else:
        _original_modules["transformers.masking_utils"] = sys.modules["transformers.masking_utils"]


_setup_mock_modules()


def pytest_configure(config):
    """确保在收集测试模块前 mock 已就绪"""
    _setup_mock_modules()


def pytest_unconfigure(config):
    """清理本 conftest 注入的 mock，恢复原始模块（若存在）"""
    for module_name in _created_modules:
        if module_name in sys.modules:
            del sys.modules[module_name]
        if _original_modules.get(module_name) is not None:
            sys.modules[module_name] = _original_modules[module_name]
