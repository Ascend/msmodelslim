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

Pytest config for qwen3_5_moe tests. Mocks transformers components when missing.
"""

import sys
import types
from unittest.mock import MagicMock

_created_modules = {}
_original_modules = {}


def _make_modeling_qwen3_5():
    """Create mock transformers.models.qwen3_5.modeling_qwen3_5 module."""
    m = types.ModuleType("transformers.models.qwen3_5.modeling_qwen3_5")

    class MockQwen3_5Attention:
        pass

    class MockQwen3_5MLP:
        pass

    class MockQwen3_5RMSNorm:
        pass

    class MockQwen3_5TextRotaryEmbedding:
        pass

    m.Qwen3_5Attention = MockQwen3_5Attention
    m.Qwen3_5MLP = MockQwen3_5MLP
    m.Qwen3_5RMSNorm = MockQwen3_5RMSNorm
    m.Qwen3_5TextRotaryEmbedding = MockQwen3_5TextRotaryEmbedding
    return m


def _setup_mock_modules():
    """Inject transformers mocks so qwen3_5_moe modules import without transformers>=5.2.0."""

    _original_modules["transformers"] = sys.modules["transformers"]
    transformers_module = sys.modules["transformers"]

    required_attrs = {
        "AutoProcessor": MagicMock(),
        "Qwen3_5MoeForConditionalGeneration": MagicMock(),
        "Qwen3_5ForConditionalGeneration": MagicMock(),
    }
    for attr_name, attr_value in required_attrs.items():
        if not hasattr(transformers_module, attr_name):
            setattr(transformers_module, attr_name, attr_value)

    if "transformers.masking_utils" not in sys.modules:
        _original_modules["transformers.masking_utils"] = None
        masking_utils = types.ModuleType("transformers.masking_utils")
        masking_utils.create_causal_mask = MagicMock()
        sys.modules["transformers.masking_utils"] = masking_utils
        _created_modules["transformers.masking_utils"] = masking_utils
    else:
        _original_modules["transformers.masking_utils"] = sys.modules["transformers.masking_utils"]
        masking_utils = sys.modules["transformers.masking_utils"]
        if not hasattr(masking_utils, "create_causal_mask"):
            masking_utils.create_causal_mask = MagicMock()

    setattr(transformers_module, "masking_utils", masking_utils)

    _original_modules["transformers.models"] = sys.modules["transformers.models"]
    models_module = sys.modules["transformers.models"]

    if "transformers.models.qwen3_5" not in sys.modules:
        _original_modules["transformers.models.qwen3_5"] = None
        qwen3_5_module = types.ModuleType("transformers.models.qwen3_5")
        sys.modules["transformers.models.qwen3_5"] = qwen3_5_module
        setattr(models_module, "qwen3_5", qwen3_5_module)
        _created_modules["transformers.models.qwen3_5"] = qwen3_5_module
    else:
        _original_modules["transformers.models.qwen3_5"] = sys.modules["transformers.models.qwen3_5"]
        qwen3_5_module = sys.modules["transformers.models.qwen3_5"]

    modeling_key = "transformers.models.qwen3_5.modeling_qwen3_5"
    if modeling_key not in sys.modules:
        _original_modules[modeling_key] = sys.modules.get(modeling_key)
        mock_modeling = _make_modeling_qwen3_5()
        sys.modules[modeling_key] = mock_modeling
        setattr(qwen3_5_module, "modeling_qwen3_5", mock_modeling)
        _created_modules[modeling_key] = mock_modeling
    else:
        _original_modules[modeling_key] = sys.modules[modeling_key]
        mock_modeling = sys.modules[modeling_key]
        for attr_name, cls in (
            ("Qwen3_5Attention", type("MockQwen3_5Attention", (), {})),
            ("Qwen3_5MLP", type("MockQwen3_5MLP", (), {})),
            ("Qwen3_5RMSNorm", type("MockQwen3_5RMSNorm", (), {})),
            ("Qwen3_5TextRotaryEmbedding", type("MockQwen3_5TextRotaryEmbedding", (), {})),
        ):
            if not hasattr(mock_modeling, attr_name):
                setattr(mock_modeling, attr_name, cls)


_setup_mock_modules()


def pytest_configure(config):
    """Ensure mocks are in place before test modules are collected."""
    _setup_mock_modules()


def pytest_unconfigure(config):
    """Tear down mocks created by this conftest and restore originals if any."""
    for module_name in _created_modules:
        if module_name in sys.modules:
            del sys.modules[module_name]
        if _original_modules.get(module_name) is not None:
            sys.modules[module_name] = _original_modules[module_name]
