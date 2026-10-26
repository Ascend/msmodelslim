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

from pathlib import Path


from testing_utils.mock import mock_init_config

mock_init_config()

from msmodelslim.app.naive_quantization.model_info_interface import ModelInfoInterface  # noqa: E402


class _ConcreteModelInfo(ModelInfoInterface):
    @property
    def model_type(self) -> str:
        return "Qwen3-32B"

    @property
    def model_path(self) -> Path:
        return Path("/models/qwen3")

    @property
    def trust_remote_code(self) -> bool:
        return False

    def get_model_pedigree(self) -> str:
        return "qwen3"

    def get_model_type(self) -> str:
        return "Qwen3-32B"


def test_model_info_interface_has_exact_abstract_contract_when_inspected():
    abstract = set(ModelInfoInterface.__abstractmethods__)
    # ModelInfoInterface adds its own two abstract methods on top of the IModel contract
    assert {"get_model_pedigree", "get_model_type"} <= abstract
    assert {"model_type", "model_path", "trust_remote_code"} <= abstract


def test_concrete_model_info_returns_contract_values_when_queried():
    model_info = _ConcreteModelInfo()

    assert model_info.get_model_pedigree() == "qwen3"
    assert model_info.get_model_type() == "Qwen3-32B"
