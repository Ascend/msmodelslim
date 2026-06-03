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

import sys
import types
from unittest.mock import MagicMock


_created_modules = {}
_original_modules = {}


def _setup_mock_modules():
    """注入 transformers.masking_utils 与 transformers.models.qwen3_vl mock，确保在导入 model_adapter 前生效"""
    if 'transformers.masking_utils' not in sys.modules:
        _original_modules['transformers.masking_utils'] = None
        masking_utils = types.ModuleType('transformers.masking_utils')
        masking_utils.create_causal_mask = MagicMock()
        sys.modules['transformers.masking_utils'] = masking_utils
        _created_modules['transformers.masking_utils'] = masking_utils
    else:
        _original_modules['transformers.masking_utils'] = sys.modules['transformers.masking_utils']
        masking_utils = sys.modules['transformers.masking_utils']

    if 'transformers' in sys.modules:
        setattr(sys.modules['transformers'], 'masking_utils', masking_utils)
    else:
        transformers_mod = types.ModuleType('transformers')
        transformers_mod.masking_utils = masking_utils
        sys.modules['transformers'] = transformers_mod

    transformers_mod = sys.modules['transformers']
    if not hasattr(transformers_mod, 'Qwen3VLForConditionalGeneration'):
        transformers_mod.Qwen3VLForConditionalGeneration = MagicMock(name='Qwen3VLForConditionalGeneration')

    # transformers.models.qwen3_vl（缺失时注入，以便 model_adapter 能 import Qwen3VLTextDecoderLayer）
    try:
        import transformers.models  # noqa: F401
    except Exception:
        pass
    models_module = sys.modules.get('transformers.models')
    if models_module is None:
        return

    if 'transformers.models.qwen3_vl' not in sys.modules:
        _original_modules['transformers.models.qwen3_vl'] = None
        qwen3_vl = types.ModuleType('transformers.models.qwen3_vl')
        sys.modules['transformers.models.qwen3_vl'] = qwen3_vl
        setattr(models_module, 'qwen3_vl', qwen3_vl)
        _created_modules['transformers.models.qwen3_vl'] = qwen3_vl
    else:
        _original_modules['transformers.models.qwen3_vl'] = sys.modules['transformers.models.qwen3_vl']
        qwen3_vl = sys.modules['transformers.models.qwen3_vl']

    if 'transformers.models.qwen3_vl.modeling_qwen3_vl' not in sys.modules:
        _original_modules['transformers.models.qwen3_vl.modeling_qwen3_vl'] = sys.modules.get(
            'transformers.models.qwen3_vl.modeling_qwen3_vl'
        )
        modeling = types.ModuleType('transformers.models.qwen3_vl.modeling_qwen3_vl')
        modeling.Qwen3VLTextDecoderLayer = MagicMock()
        sys.modules['transformers.models.qwen3_vl.modeling_qwen3_vl'] = modeling
        setattr(qwen3_vl, 'modeling_qwen3_vl', modeling)
        _created_modules['transformers.models.qwen3_vl.modeling_qwen3_vl'] = modeling


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
