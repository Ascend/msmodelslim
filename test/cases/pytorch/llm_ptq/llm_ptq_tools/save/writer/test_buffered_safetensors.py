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
# pylint: disable=redefined-outer-name,unspecified-encoding

import json
import logging
import os.path
import tempfile
from unittest import mock

import pytest
import torch
from safetensors.torch import load_file

from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.save.writer import BufferedSafetensorsWriter

TMP_PREFIX = 'test'
TMP_KEY = 'test_tensor'


@pytest.fixture(scope="function")
def writer():
    logger = logging.getLogger()
    with tempfile.TemporaryDirectory() as tmp:
        writer = BufferedSafetensorsWriter(logger=logger, save_directory=tmp, save_prefix=TMP_PREFIX)
        writer.max_size = 33  # 文件大小限制：33B
        yield writer


def test_write_multi_files_when_exceed_limit(writer):
    weight_dict = {
        'w0': torch.randn(1, 4, dtype=torch.float32),  # 1*4*4 = 16B
        'w1': torch.randn(1, 4, dtype=torch.float32),
        'w2': torch.randn(1, 4, dtype=torch.float32),
        'w3': torch.randn(1, 4, dtype=torch.float32),
        'w4': torch.randn(1, 4, dtype=torch.float32),
        'w5': torch.randn(1, 4, dtype=torch.float32),
    }

    # write to files
    for k, v in weight_dict.items():
        writer.write(k, v)
    writer.close()

    def get_path_helper(file_name):
        return os.path.join(writer.save_directory, file_name)

    # check files
    assert os.path.exists(get_path_helper(f'{TMP_PREFIX}.safetensors.index.json'))
    for i in range(3):
        assert os.path.exists(get_path_helper(f'{TMP_PREFIX}-{i + 1:05d}-of-00003.safetensors'))

    with open(get_path_helper(f'{TMP_PREFIX}.safetensors.index.json'), 'r') as f:
        weight_map = json.load(f)['weight_map']

    def get_tensor_helper(tensor_name):
        weights = load_file(get_path_helper(weight_map[tensor_name]))
        return weights[tensor_name]

    # check values
    for k, v in weight_dict.items():
        assert torch.allclose(get_tensor_helper(k), v)


def test_write_one_file_when_one_large_tensor(writer):
    tensor = torch.randn(1, 16, dtype=torch.float32)  # 1*16*4 = 64B

    with mock.patch.object(writer.logger, 'warning') as mock_warning:
        writer.write(TMP_KEY, tensor)
        writer.close()
        assert mock_warning.called

    def get_path_helper(file_name):
        return os.path.join(writer.save_directory, file_name)

    # check files
    assert os.path.exists(get_path_helper(f'{TMP_PREFIX}.safetensors.index.json'))
    assert os.path.exists(get_path_helper(f'{TMP_PREFIX}-{1:05d}-of-00001.safetensors'))
    assert not os.path.exists(get_path_helper(f'{TMP_PREFIX}-{1:05d}-of-00002.safetensors'))
    assert not os.path.exists(get_path_helper(f'{TMP_PREFIX}-{2:05d}-of-00001.safetensors'))
    assert not os.path.exists(get_path_helper(f'{TMP_PREFIX}-{2:05d}-of-00002.safetensors'))

    with open(get_path_helper(f'{TMP_PREFIX}.safetensors.index.json'), 'r') as f:
        weight_map = json.load(f)['weight_map']

    def get_tensor_helper(tensor_name):
        weights = load_file(get_path_helper(weight_map[tensor_name]))
        return weights[tensor_name]

    # check values
    assert torch.allclose(get_tensor_helper(TMP_KEY), tensor)
