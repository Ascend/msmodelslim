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

Unit tests for msmodelslim.format.ascendV1_format.ascendV1_tensors_writer_factory_infra.
"""

# pylint: disable=abstract-class-instantiated

from __future__ import annotations

import pytest

from msmodelslim.format.ascendV1_format.ascendV1_tensors_writer_factory_infra import (
    AscendV1SafetensorsWriterFactoryInfra,
    AscendV1SafetensorsWriterInfra,
)


class ConcreteSafetensorsWriter(AscendV1SafetensorsWriterInfra):
    def __init__(self) -> None:
        self.storage = {}

    def write(self, key: str, value) -> None:
        self.storage[key] = value

    def close(self) -> None:
        self.storage.clear()


class ConcreteSafetensorsWriterFactory(AscendV1SafetensorsWriterFactoryInfra):
    def create_safetensors_writer(self, part_file_size: int, save_directory: str, save_prefix: str):
        return ConcreteSafetensorsWriter()


class TestAscendV1SafetensorsWriterInfra:
    def test_safetensors_writer_infra_raise_type_error_when_instantiated_directly(self):
        with pytest.raises(TypeError):
            AscendV1SafetensorsWriterInfra()

    def test_safetensors_writer_infra_raise_type_error_for_factory_directly(self):
        with pytest.raises(TypeError):
            AscendV1SafetensorsWriterFactoryInfra()

    def test_concrete_safetensors_writer_write_and_close_when_implemented(self):
        writer = ConcreteSafetensorsWriter()

        writer.write("weight", "tensor")

        assert writer.storage == {"weight": "tensor"}
        writer.close()
        assert not writer.storage


class TestAscendV1SafetensorsWriterFactoryInfra:
    def test_concrete_safetensors_factory_return_writer_when_create_called(self):
        factory = ConcreteSafetensorsWriterFactory()

        writer = factory.create_safetensors_writer(4, "/tmp", "model")

        assert isinstance(writer, ConcreteSafetensorsWriter)


class TestAscendV1SafetensorsAbcBodies:
    """Invoke ABC stub bodies for full line coverage."""

    def test_safetensors_writer_abc_bodies_when_super_called(self):
        class Writer(AscendV1SafetensorsWriterInfra):
            def write(self, key: str, value) -> None:
                AscendV1SafetensorsWriterInfra.write(self, key, value)

            def close(self) -> None:
                AscendV1SafetensorsWriterInfra.close(self)

        writer = Writer()

        assert writer.write("key", None) is None
        assert writer.close() is None

    def test_safetensors_factory_abc_body_when_super_called(self):
        class Factory(AscendV1SafetensorsWriterFactoryInfra):
            def create_safetensors_writer(self, part_file_size: int, save_directory: str, save_prefix: str):
                AscendV1SafetensorsWriterFactoryInfra.create_safetensors_writer(
                    self, part_file_size, save_directory, save_prefix
                )
                return ConcreteSafetensorsWriter()

        assert isinstance(Factory().create_safetensors_writer(4, "/tmp", "model"), ConcreteSafetensorsWriter)
