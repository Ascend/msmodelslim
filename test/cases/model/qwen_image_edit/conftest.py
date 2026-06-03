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
Pytest fixtures for qwen_image_edit tests.
"""

import sys
from unittest.mock import MagicMock

import pytest


@pytest.fixture(autouse=True)
def mock_qwenimage_edit_modules():
    """Mock qwenimage_edit 依赖，避免单测拉取真实权重。"""
    modules = [
        "qwenimage_edit",
        "qwenimage_edit.transformer_qwenimage",
        "qwenimage_edit.pipeline_qwenimage_edit_plus",
    ]
    saved = {m: sys.modules.get(m) for m in modules}
    pkg = MagicMock()
    transformer_cls = MagicMock()
    pipeline_cls = MagicMock()
    pkg.transformer_qwenimage = MagicMock(QwenImageTransformer2DModel=transformer_cls)
    pkg.pipeline_qwenimage_edit_plus = MagicMock(QwenImageEditPlusPipeline=pipeline_cls)
    sys.modules["qwenimage_edit"] = pkg
    sys.modules["qwenimage_edit.transformer_qwenimage"] = pkg.transformer_qwenimage
    sys.modules["qwenimage_edit.pipeline_qwenimage_edit_plus"] = pkg.pipeline_qwenimage_edit_plus
    yield {"transformer_cls": transformer_cls, "pipeline_cls": pipeline_cls}
    for name, mod in saved.items():
        if mod is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = mod
