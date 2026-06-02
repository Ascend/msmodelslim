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

Unit tests for msmodelslim.format.compressed_tensors_format.compressed_tensors_safetensors_writer_factory_infra.
"""

# pylint: disable=abstract-class-instantiated

from __future__ import annotations

import pytest
import torch

from msmodelslim.format.compressed_tensors_format.compressed_tensors_safetensors_writer_factory_infra import (
    CompressedTensorSafetensorsWriterFactoryInfra,
    CompressedTensorSafetensorsWriterInfra,
)


class ConcreteWriter(CompressedTensorSafetensorsWriterInfra):
    def __init__(self) -> None:
        self.storage = {}

    def write(self, key: str, value: torch.Tensor) -> None:
        self.storage[key] = value

    def close(self) -> None:
        self.storage.clear()


class ConcreteWriterFactory(CompressedTensorSafetensorsWriterFactoryInfra):
    def create_safetensors_writer(self, part_file_size: int, save_directory: str, save_prefix: str):
        return ConcreteWriter()


class TestCompressedTensorSafetensorsWriterInfra:
    """Tests for CompressedTensorSafetensorsWriterInfra abstract interface."""

    def test_writer_infra_raise_type_error_when_instantiated_directly(self):
        with pytest.raises(TypeError):
            CompressedTensorSafetensorsWriterInfra()

    def test_concrete_writer_write_tensor_when_implemented(self):
        writer = ConcreteWriter()
        tensor = torch.tensor([1, 2, 3])

        writer.write("key", tensor)

        assert writer.storage["key"] is tensor


class TestCompressedTensorSafetensorsWriterFactoryInfra:
    """Tests for CompressedTensorSafetensorsWriterFactoryInfra abstract interface."""

    def test_factory_infra_raise_type_error_when_instantiated_directly(self):
        with pytest.raises(TypeError):
            CompressedTensorSafetensorsWriterFactoryInfra()

    def test_concrete_factory_return_writer_when_create_called(self):
        factory = ConcreteWriterFactory()

        writer = factory.create_safetensors_writer(4, "/tmp/save", "model")

        assert isinstance(writer, ConcreteWriter)
