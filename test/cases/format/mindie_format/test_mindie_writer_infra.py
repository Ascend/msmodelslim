#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
-------------------------------------------------------------------------
This file is part of the MindStudio project.
Copyright (c) 2025 Huawei Technologies Co.,Ltd.

MindStudio is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:

         http://license.coscl.org.cn/MulanPSL2

THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details.
-------------------------------------------------------------------------

Unit tests for msmodelslim.format.mindie_format.mindie_writer_infra.
"""

# pylint: disable=abstract-class-instantiated

from __future__ import annotations

import pytest

from msmodelslim.format.mindie_format.mindie_json_writer_factory_infra import (
    MindIEJsonWriterFactoryInfra,
    MindIEJsonWriterInfra,
)
from msmodelslim.format.mindie_format.mindie_tensors_writer_factory_infra import (
    MindIESafetensorsWriterFactoryInfra,
    MindIESafetensorsWriterInfra,
)


class ConcreteJsonWriter(MindIEJsonWriterInfra):
    def __init__(self) -> None:
        self.storage = {}

    def write(self, prefix: str, desc: object) -> None:
        self.storage[prefix] = desc

    def close(self) -> None:
        self.storage.clear()


class ConcreteJsonWriterFactory(MindIEJsonWriterFactoryInfra):
    def create_json_writer(self, save_directory: str, file_name: str) -> MindIEJsonWriterInfra:
        return ConcreteJsonWriter()


class ConcreteSafetensorsWriter(MindIESafetensorsWriterInfra):
    def __init__(self) -> None:
        self.storage = {}

    def write(self, key: str, value) -> None:
        self.storage[key] = value

    def close(self) -> None:
        self.storage.clear()


class ConcreteSafetensorsWriterFactory(MindIESafetensorsWriterFactoryInfra):
    def create_safetensors_writer(self, part_file_size: int, save_directory: str, save_prefix: str):
        return ConcreteSafetensorsWriter()


class TestMindIEJsonWriterInfra:
    def test_json_writer_infra_raise_type_error_when_instantiated_directly(self):
        with pytest.raises(TypeError):
            MindIEJsonWriterInfra()

    def test_concrete_json_writer_write_when_implemented(self):
        writer = ConcreteJsonWriter()

        writer.write("layer", {"dtype": "int8"})

        assert writer.storage["layer"] == {"dtype": "int8"}


class TestMindIEJsonWriterFactoryInfra:
    def test_json_factory_infra_raise_type_error_when_instantiated_directly(self):
        with pytest.raises(TypeError):
            MindIEJsonWriterFactoryInfra()

    def test_concrete_json_factory_return_writer_when_create_called(self):
        factory = ConcreteJsonWriterFactory()

        writer = factory.create_json_writer("/tmp", "desc.json")

        assert isinstance(writer, ConcreteJsonWriter)


class TestMindIESafetensorsWriterInfra:
    def test_safetensors_writer_infra_raise_type_error_when_instantiated_directly(self):
        with pytest.raises(TypeError):
            MindIESafetensorsWriterInfra()

    def test_concrete_safetensors_factory_return_writer_when_create_called(self):
        factory = ConcreteSafetensorsWriterFactory()

        writer = factory.create_safetensors_writer(4, "/tmp", "model")

        assert isinstance(writer, ConcreteSafetensorsWriter)


class TestMindIEInfraAbcBodies:
    """Invoke ABC stub bodies for full line coverage."""

    def test_json_writer_abc_bodies_when_super_called(self):
        class Writer(MindIEJsonWriterInfra):
            def write(self, prefix: str, desc: object) -> None:
                MindIEJsonWriterInfra.write(self, prefix, desc)

            def close(self) -> None:
                MindIEJsonWriterInfra.close(self)

        writer = Writer()
        writer.write("layer", {})
        writer.close()

    def test_json_factory_abc_body_when_super_called(self):
        class Factory(MindIEJsonWriterFactoryInfra):
            def create_json_writer(self, save_directory: str, file_name: str) -> MindIEJsonWriterInfra:
                MindIEJsonWriterFactoryInfra.create_json_writer(self, save_directory, file_name)
                return ConcreteJsonWriter()

        Factory().create_json_writer("/tmp", "desc.json")

    def test_safetensors_writer_abc_bodies_when_super_called(self):
        class Writer(MindIESafetensorsWriterInfra):
            def write(self, key: str, value) -> None:
                MindIESafetensorsWriterInfra.write(self, key, value)

            def close(self) -> None:
                MindIESafetensorsWriterInfra.close(self)

        writer = Writer()
        writer.write("key", None)
        writer.close()

    def test_safetensors_factory_abc_body_when_super_called(self):
        class Factory(MindIESafetensorsWriterFactoryInfra):
            def create_safetensors_writer(self, part_file_size: int, save_directory: str, save_prefix: str):
                MindIESafetensorsWriterFactoryInfra.create_safetensors_writer(
                    self, part_file_size, save_directory, save_prefix
                )
                return ConcreteSafetensorsWriter()

        Factory().create_safetensors_writer(4, "/tmp", "model")
