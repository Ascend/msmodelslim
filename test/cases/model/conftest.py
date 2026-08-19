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

# Pytest config for model tests.
# 复用全局 mock 工具，避免在导入 msmodelslim 期间真正初始化配置文件和安全路径检查。
# 同时按会话级别 mock 缺失的第三方依赖，避免对其他测试任务造成长期污染。

import pytest
import torch

from testing_utils.mock import mock_security_library, mock_kia_library, mock_init_config


mock_init_config()
mock_kia_library()
mock_security_library()


# fixup: replace torch.npu with a clean stub. Some tests (e.g. flux1)
# overwrite torch.npu with a bare Mock and don't clean up, causing
# subsequent tests to see Mock-tainted current_device(). An autouse
# fixture ensures the stub is restored before every test.
class _NpuStub:
    """Safe NPU stub — not a Mock, so no auto-created child mocks."""

    def is_available(self):
        return False

    def current_device(self):
        return 0


@pytest.fixture(autouse=True)
def _reset_npu_stub():
    """Restore torch.npu stub before each test, preventing cross-test Mock leakage."""
    torch.npu = _NpuStub()  # type: ignore[attr-defined]
    yield
