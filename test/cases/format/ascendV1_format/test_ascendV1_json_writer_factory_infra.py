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

Unit tests for msmodelslim.format.ascendV1_format.ascendV1_json_writer_factory_infra.
"""

# pylint: disable=abstract-class-instantiated

from __future__ import annotations

import pytest

from msmodelslim.format.ascendV1_format.ascendV1_json_writer_factory_infra import (
    AscendV1JsonWriterFactoryInfra,
    AscendV1JsonWriterInfra,
)


class ConcreteJsonWriter(AscendV1JsonWriterInfra):
    def __init__(self) -> None:
        self.storage = {}

    def write(self, prefix: str, desc: object) -> None:
        self.storage[prefix] = desc

    def close(self) -> None:
        self.storage.clear()


class ConcreteJsonWriterFactory(AscendV1JsonWriterFactoryInfra):
    def create_json_writer(self, save_directory: str, file_name: str) -> AscendV1JsonWriterInfra:
        return ConcreteJsonWriter()


class TestAscendV1JsonWriterInfra:
    def test_json_writer_infra_raise_type_error_when_instantiated_directly(self):
        with pytest.raises(TypeError):
            AscendV1JsonWriterInfra()

    def test_concrete_json_writer_write_and_close_when_implemented(self):
        writer = ConcreteJsonWriter()

        writer.write("layer", {"dtype": "int8"})

        assert writer.storage == {"layer": {"dtype": "int8"}}
        writer.close()
        assert not writer.storage


class TestAscendV1JsonWriterFactoryInfra:
    def test_json_factory_infra_raise_type_error_when_instantiated_directly(self):
        with pytest.raises(TypeError):
            AscendV1JsonWriterFactoryInfra()

    def test_concrete_json_factory_return_writer_when_create_called(self):
        factory = ConcreteJsonWriterFactory()

        writer = factory.create_json_writer("/tmp", "desc.json")

        assert isinstance(writer, ConcreteJsonWriter)


class TestAscendV1JsonWriterAbcBodies:
    """Invoke ABC stub bodies for full line coverage."""

    def test_json_writer_abc_bodies_when_super_called(self):
        class Writer(AscendV1JsonWriterInfra):
            def write(self, prefix: str, desc: object) -> None:
                AscendV1JsonWriterInfra.write(self, prefix, desc)

            def close(self) -> None:
                AscendV1JsonWriterInfra.close(self)

        writer = Writer()

        assert writer.write("layer", {}) is None
        assert writer.close() is None

    def test_json_factory_abc_body_when_super_called(self):
        class Factory(AscendV1JsonWriterFactoryInfra):
            def create_json_writer(self, save_directory: str, file_name: str) -> AscendV1JsonWriterInfra:
                AscendV1JsonWriterFactoryInfra.create_json_writer(self, save_directory, file_name)
                return ConcreteJsonWriter()

        assert isinstance(Factory().create_json_writer("/tmp", "desc.json"), ConcreteJsonWriter)
