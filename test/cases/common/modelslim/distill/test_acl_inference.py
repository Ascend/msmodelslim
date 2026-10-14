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

import pytest
from unittest.mock import patch, MagicMock

# Import the module to test
import numpy as np  # noqa: F401
from ascend_utils.common.security import check_int, get_valid_read_path  # noqa: F401
import ascend_utils.common.acl_inference as acl_inf

# Constants
ACL_ERROR_NONE = 0
ACL_ERROR_OTHER = 1
ACL_MEM_MALLOC_HUGE_FIRST = 0
ACL_MEMCPY_HOST_TO_DEVICE = 1
ACL_MEMCPY_DEVICE_TO_HOST = 2


@pytest.fixture(autouse=True)
def reset_global_state():
    """Reset global variable before each test"""
    acl_inf.IS_ACL_INITIALIZED_BY_THIS_MODULE = False


@pytest.fixture
def mock_acl_module():
    with patch.dict(
        'sys.modules',
        {
            'acl': MagicMock(),
            'acl.rt': MagicMock(),
            'acl.mdl': MagicMock(),
            'acl.create_data_buffer': MagicMock(),
            'acl.destroy_data_buffer': MagicMock(),
            'acl.mdl.add_dataset_buffer': MagicMock(),
            'acl.mdl.get_dataset_num_buffers': MagicMock(),
            'acl.mdl.get_dataset_buffer': MagicMock(),
            'acl.mdl.destroy_dataset': MagicMock(),
            'acl.mdl.load_from_file': MagicMock(),
            'acl.mdl.create_desc': MagicMock(),
            'acl.mdl.get_desc': MagicMock(),
            'acl.mdl.get_num_inputs': MagicMock(),
            'acl.mdl.get_input_dims': MagicMock(),
            'acl.mdl.get_input_format': MagicMock(),
            'acl.mdl.get_input_size_by_index': MagicMock(),
            'acl.mdl.get_input_data_type': MagicMock(),
            'acl.mdl.get_output_name_by_index': MagicMock(),
            'acl.mdl.get_num_outputs': MagicMock(),
            'acl.mdl.get_output_dims': MagicMock(),
            'acl.mdl.get_output_format': MagicMock(),
            'acl.mdl.get_output_size_by_index': MagicMock(),
            'acl.mdl.get_output_data_type': MagicMock(),
            'acl.mdl.execute': MagicMock(),
            'acl.mdl.unload': MagicMock(),
            'acl.mdl.destroy_desc': MagicMock(),
            'acl.rt.malloc': MagicMock(),
            'acl.rt.free': MagicMock(),
            'acl.rt.memset': MagicMock(),
            'acl.rt.memcpy': MagicMock(),
            'acl.rt.set_context': MagicMock(),
            'acl.rt.create_context': MagicMock(),
            'acl.rt.destroy_context': MagicMock(),
            'acl.rt.get_device': MagicMock(),
            'acl.rt.set_device': MagicMock(),
            'acl.rt.reset_device': MagicMock(),
            'acl.finalize': MagicMock(),
            'acl.init': MagicMock(),
            'acl.create_tensor_desc': MagicMock(),
            'acl.mdl.set_dataset_tensor_desc': MagicMock(),
        },
    ) as mock:
        yield mock


@patch("ascend_utils.common.acl_inference.acl")
def test_check_ret_given_success_when_called_then_no_exception(mock_acl):
    acl_inf._check_ret("Test message", ACL_ERROR_NONE)


@patch("ascend_utils.common.acl_inference.acl")
def test_check_ret_given_failure_when_called_then_raise_exception(mock_acl):
    with pytest.raises(Exception) as exc_info:
        acl_inf._check_ret("Test message", ACL_ERROR_OTHER)
    assert "Test message failed ret = 1" in str(exc_info.value)


@patch("ascend_utils.common.acl_inference.acl")
def test_init_acl_given_no_config_when_not_initialized(mock_acl):
    mock_acl.rt.get_device.return_value = (None, ACL_ERROR_OTHER)  # Not initialized
    mock_acl.init.return_value = ACL_ERROR_NONE
    mock_acl.rt.set_device.return_value = ACL_ERROR_NONE
    acl_inf.init_acl(device_id=0)
    mock_acl.init.assert_called_once()
    mock_acl.rt.set_device.assert_called_with(0)


@patch("ascend_utils.common.acl_inference.acl")
def test_init_acl_given_config_path(mock_acl):
    mock_acl.rt.get_device.return_value = (None, ACL_ERROR_OTHER)
    mock_acl.init.return_value = ACL_ERROR_NONE
    mock_acl.rt.set_device.return_value = ACL_ERROR_NONE
    acl_inf.init_acl(device_id=0, config_path="/path/to/config")
    mock_acl.init.assert_called_with("/path/to/config")


@patch("ascend_utils.common.acl_inference.acl")
def test_release_acl_given_not_initialized_by_module(mock_acl):
    acl_inf.IS_ACL_INITIALIZED_BY_THIS_MODULE = False
    mock_acl.rt.reset_device.return_value = ACL_ERROR_NONE
    mock_acl.finalize.return_value = ACL_ERROR_NONE
    acl_inf.release_acl(device_id=0)
    mock_acl.rt.reset_device.assert_called_with(0)
    mock_acl.finalize.assert_not_called()


@patch("ascend_utils.common.acl_inference.acl")
def test_release_acl_given_initialized_by_module(mock_acl):
    acl_inf.IS_ACL_INITIALIZED_BY_THIS_MODULE = True
    mock_acl.rt.reset_device.return_value = ACL_ERROR_NONE
    mock_acl.finalize.return_value = ACL_ERROR_NONE
    acl_inf.release_acl(device_id=0)
    mock_acl.rt.reset_device.assert_called_with(0)
    mock_acl.finalize.assert_called_once()
