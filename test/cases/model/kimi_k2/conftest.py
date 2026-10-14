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
# pylint: disable=no-name-in-module

import pytest
from transformers import PretrainedConfig


class _DummyConfig(PretrainedConfig):
    model_type = "dummy"

    def __init__(self, **kwargs):
        super().__init__(pad_token_id=0, **kwargs)
        self.hidden_size = 64
        self.vocab_size = 128
        self.rms_norm_eps = 1e-6
        self.num_hidden_layers = 3


@pytest.fixture
def dummy_config():
    return _DummyConfig()
