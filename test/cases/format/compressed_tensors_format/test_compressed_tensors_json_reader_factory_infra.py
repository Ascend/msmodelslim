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

Unit tests for msmodelslim.format.compressed_tensors_format.compressed_tensors_json_reader_factory_infra.
"""

# pylint: disable=abstract-class-instantiated

from __future__ import annotations

import pytest

from msmodelslim.format.compressed_tensors_format.compressed_tensors_json_reader_factory_infra import (
    CompressedTensorJsonReaderFactoryInfra,
    CompressedTensorJsonReaderInfra,
)


class ConcreteJsonReader(CompressedTensorJsonReaderInfra):
    def __init__(self, data: dict) -> None:
        self._data = data

    def load(self) -> dict:
        return self._data


class ConcreteJsonReaderFactory(CompressedTensorJsonReaderFactoryInfra):
    def create_json_reader(self, file_path: str) -> CompressedTensorJsonReaderInfra:
        return ConcreteJsonReader({"path": file_path})


class TestCompressedTensorJsonReaderInfra:
    def test_json_reader_infra_raise_type_error_when_instantiated_directly(self):
        with pytest.raises(TypeError):
            CompressedTensorJsonReaderInfra()

    def test_concrete_json_reader_load_when_implemented(self):
        reader = ConcreteJsonReader({"model_type": "test"})

        assert reader.load() == {"model_type": "test"}


class TestCompressedTensorJsonReaderFactoryInfra:
    def test_json_reader_factory_infra_raise_type_error_when_instantiated_directly(self):
        with pytest.raises(TypeError):
            CompressedTensorJsonReaderFactoryInfra()

    def test_concrete_json_reader_factory_return_reader_when_create_called(self):
        factory = ConcreteJsonReaderFactory()

        reader = factory.create_json_reader("/tmp/config.json")

        assert reader.load() == {"path": "/tmp/config.json"}


class TestCompressedTensorJsonReaderAbcBodies:
    """Invoke ABC stub bodies for full line coverage."""

    def test_json_reader_abc_body_when_super_called(self):
        class Reader(CompressedTensorJsonReaderInfra):
            def load(self) -> dict:
                CompressedTensorJsonReaderInfra.load(self)
                return {}

        assert not Reader().load()

    def test_json_reader_factory_abc_body_when_super_called(self):
        class Factory(CompressedTensorJsonReaderFactoryInfra):
            def create_json_reader(self, file_path: str) -> CompressedTensorJsonReaderInfra:
                CompressedTensorJsonReaderFactoryInfra.create_json_reader(self, file_path)
                return ConcreteJsonReader({})

        assert not Factory().create_json_reader("/tmp/config.json").load()
