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

from msmodelslim.core.quant_service.dataset_loader_infra import DatasetLoaderInfra


class _StubDatasetLoader(DatasetLoaderInfra):
    def get_dataset_by_name(self, dataset_id: str):
        return [dataset_id]


class TestDatasetLoaderInfra:
    """Tests for DatasetLoaderInfra."""

    def test_get_dataset_by_name_returns_id_when_stub_implemented(self):
        """场景：子类实现 get_dataset_by_name。预期：返回数据集标识。"""
        loader = _StubDatasetLoader()
        assert loader.get_dataset_by_name("calib") == ["calib"]

    def test_instantiate_raises_type_error_when_abstract(self):
        """场景：直接实例化抽象类。预期：TypeError。"""
        with pytest.raises(TypeError):
            DatasetLoaderInfra()  # pylint: disable=abstract-class-instantiated
