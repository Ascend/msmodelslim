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

import pytest

from testing_utils.mock import mock_init_config

mock_init_config()

from msmodelslim.app.naive_quantization.practice_manager_infra import (  # noqa: E402
    PracticeManagerInfra,
    QuantConfigExportInfra,
)


class _ConcretePracticeManager(PracticeManagerInfra):
    def __contains__(self, model_pedigree) -> bool:
        return False

    def get_config_by_id(self, model_pedigree, config_id: str):
        return object()

    def iter_config(self, model_pedigree):
        return iter(())


def test_practice_manager_infra_cannot_instantiate_when_contract_is_abstract():
    with pytest.raises(TypeError, match="abstract"):
        PracticeManagerInfra()  # pylint: disable=abstract-class-instantiated


def test_get_config_url_returns_none_when_subclass_does_not_override_default():
    manager = _ConcretePracticeManager()

    assert manager.get_config_url("qwen3", "default") is None


def test_quant_config_export_infra_cannot_instantiate_when_export_is_abstract():
    with pytest.raises(TypeError, match="abstract"):
        QuantConfigExportInfra()  # pylint: disable=abstract-class-instantiated


def test_quant_config_export_infra_accepts_concrete_export_when_subclass_implements_contract(tmp_path):
    class _ConcreteExporter(QuantConfigExportInfra):
        def __init__(self):
            self.calls = []

        def export_quant_config(self, quant_config, model_type: str, save_path: Path) -> None:
            self.calls.append((quant_config, model_type, save_path))

    exporter = _ConcreteExporter()
    config = object()

    exporter.export_quant_config(config, "Qwen3-32B", tmp_path)

    assert exporter.calls == [(config, "Qwen3-32B", tmp_path)]
