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

from unittest.mock import Mock

import pytest

from msmodelslim.model.wan2_2.expert_sub_adapter import Wan2_2LowNoiseSubAdapter
from msmodelslim.model.wan2_2.i2v.model_adapter import Wan2_2I2VModelAdapter
from msmodelslim.model.wan2_2.t2v.model_adapter import Wan2_2T2VModelAdapter
from msmodelslim.model.wan2_2.ti2v.model_adapter import Wan2_2TI2VModelAdapter
from msmodelslim.utils.exception import InvalidModelError


def _adapter_without_init(cls):
    return cls.__new__(cls)


class TestGetExpertAdapterDualExpert:
    @staticmethod
    def test_raises_invalid_model_error_when_low_noise_sub_adapter_missing():
        adapter = _adapter_without_init(Wan2_2T2VModelAdapter)
        adapter._expert_adapters = {}
        with pytest.raises(InvalidModelError, match="low_noise_model"):
            adapter.get_expert_adapter("low_noise_model")

    @staticmethod
    def test_raises_invalid_model_error_when_high_noise_sub_adapter_missing():
        adapter = _adapter_without_init(Wan2_2I2VModelAdapter)
        adapter._expert_adapters = {}
        with pytest.raises(InvalidModelError, match="high_noise_model"):
            adapter.get_expert_adapter("high_noise_model")

    @staticmethod
    def test_returns_bound_sub_adapter_when_dual_expert_registered():
        parent = Mock()
        sub = Wan2_2LowNoiseSubAdapter(parent, "low_noise_model")
        adapter = _adapter_without_init(Wan2_2T2VModelAdapter)
        adapter._expert_adapters = {"low_noise_model": sub}
        assert adapter.get_expert_adapter("low_noise_model") is sub


class TestGetExpertAdapterSingleExpert:
    @staticmethod
    def test_falls_back_to_self_when_ti2v_empty_expert_not_bound():
        adapter = _adapter_without_init(Wan2_2TI2VModelAdapter)
        adapter._expert_adapters = {}
        assert adapter.get_expert_adapter("") is adapter

    @staticmethod
    def test_raises_invalid_model_error_when_ti2v_unknown_expert_name():
        adapter = _adapter_without_init(Wan2_2TI2VModelAdapter)
        adapter._expert_adapters = {}
        with pytest.raises(InvalidModelError, match="low_noise_model"):
            adapter.get_expert_adapter("low_noise_model")
