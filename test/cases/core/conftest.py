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

统一的 pytest 配置文件，包含所有 core 测试目录的通用配置。
"""

import sys

if sys.version_info < (3, 9):
    import typing
    import typing_extensions

    typing.Annotated = typing_extensions.Annotated

import pytest
from unittest.mock import Mock, patch, MagicMock
from testing_utils.mock import mock_kia_library, mock_security_library, mock_init_config


# ========== 基础 Mock 配置 ==========
# 这些必须在模块级别执行，确保在任何导入之前运行
mock_init_config()
mock_kia_library()
mock_security_library()


# ========== 额外的 Mock 配置 ==========
def _mock_check_dirpath_before_read(path):
    """Mock function for check_dirpath_before_read that bypasses validation"""
    return path


# Mock check_dirpath_before_read which is called by get_valid_read_path
# but not included in mock_security_library()
if 'msmodelslim.utils.security.path' not in sys.modules:
    sys.modules['msmodelslim.utils.security.path'] = MagicMock()
sys.modules['msmodelslim.utils.security.path'].check_dirpath_before_read = _mock_check_dirpath_before_read

# Mock optional third-party dependency wcmatch to avoid ModuleNotFoundError in tests
if 'wcmatch' not in sys.modules:
    sys.modules['wcmatch'] = MagicMock()


# ========== Pytest Fixtures ==========
@pytest.fixture
def sample_torch_tensor():
    """标准 float tensor，供 observer/quantizer 测试复用。"""
    import torch

    return torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])


@pytest.fixture
def mock_dataset_loader():
    """返回固定校准数据的 DatasetLoader mock。"""
    loader = Mock()
    loader.get_dataset_by_name.return_value = [{"input_ids": [1, 2, 3]}]
    return loader


@pytest.fixture
def mock_context_factory():
    """返回 LocalDictContext 的 IContextFactory mock。"""
    from msmodelslim.core.context.context_factory import ContextFactory

    factory = ContextFactory()
    mock_factory = Mock()
    mock_factory.create.return_value = factory.create(is_distributed=False)
    return mock_factory


@pytest.fixture
def mock_pipeline_interface():
    """最小 PipelineInterface stub。"""
    adapter = Mock()
    adapter.model_type = "test"
    adapter.init_model.return_value = Mock()
    adapter.handle_dataset.return_value = [[{"input_ids": [1]}]]
    return adapter


@pytest.fixture
def mock_torch():
    """Mock torch库，确保不会误判NPU可用"""
    with patch('torch') as patched_torch:
        patched_torch.device.return_value = Mock()
        patched_torch.manual_seed.return_value = None
        # 创建 npu mock 对象，但确保 is_available() 返回 False
        mock_npu = Mock()
        mock_npu.manual_seed.return_value = None
        mock_npu.manual_seed_all.return_value = None
        mock_npu.Stream.return_value = Mock()
        mock_npu.set_compile_mode.return_value = None
        mock_npu.is_available.return_value = False  # 关键：明确返回 False，避免误判
        patched_torch.npu = mock_npu
        yield patched_torch
