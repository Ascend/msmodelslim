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

Unit tests for msmodelslim.format.compressed_tensors_format.compressed_tensors_json_writer_factory_infra.
"""

# pylint: disable=abstract-class-instantiated

from __future__ import annotations

import pytest

from msmodelslim.format.compressed_tensors_format.compressed_tensors_json_writer_factory_infra import (
    CompressedTensorJsonWriterFactoryInfra,
    CompressedTensorJsonWriterInfra,
)


class ConcreteJsonWriter(CompressedTensorJsonWriterInfra):
    def __init__(self) -> None:
        self.data = None

    def dump(self, data: dict, *, indent: int = 2) -> None:
        self.data = data


class ConcreteJsonWriterFactory(CompressedTensorJsonWriterFactoryInfra):
    def create_json_writer(self, save_directory: str, file_name: str) -> CompressedTensorJsonWriterInfra:
        return ConcreteJsonWriter()


class TestCompressedTensorJsonWriterInfra:
    def test_json_writer_infra_raise_type_error_when_instantiated_directly(self):
        with pytest.raises(TypeError):
            CompressedTensorJsonWriterInfra()

    def test_concrete_json_writer_dump_when_implemented(self):
        writer = ConcreteJsonWriter()

        writer.dump({"key": "value"})

        assert writer.data == {"key": "value"}


class TestCompressedTensorJsonWriterFactoryInfra:
    def test_json_writer_factory_infra_raise_type_error_when_instantiated_directly(self):
        with pytest.raises(TypeError):
            CompressedTensorJsonWriterFactoryInfra()

    def test_concrete_json_writer_factory_return_writer_when_create_called(self):
        factory = ConcreteJsonWriterFactory()

        writer = factory.create_json_writer("/tmp", "config.json")

        assert isinstance(writer, ConcreteJsonWriter)


class TestCompressedTensorJsonWriterAbcBody:
    """Invoke ABC stub bodies for full line coverage."""

    def test_json_writer_abc_body_when_super_called(self):
        class Writer(CompressedTensorJsonWriterInfra):
            def dump(self, data: dict, *, indent: int = 2) -> None:
                CompressedTensorJsonWriterInfra.dump(self, data, indent=indent)

        assert Writer().dump({}) is None

    def test_json_writer_factory_abc_body_when_super_called(self):
        class Factory(CompressedTensorJsonWriterFactoryInfra):
            def create_json_writer(self, save_directory: str, file_name: str) -> CompressedTensorJsonWriterInfra:
                CompressedTensorJsonWriterFactoryInfra.create_json_writer(self, save_directory, file_name)
                return ConcreteJsonWriter()

        assert isinstance(Factory().create_json_writer("/tmp", "config.json"), ConcreteJsonWriter)
