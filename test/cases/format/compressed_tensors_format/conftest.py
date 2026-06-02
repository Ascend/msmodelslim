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

Shared fixtures for compressed_tensors_format unit tests.
"""

# pylint: disable=redefined-outer-name

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from testing_utils.mock import mock_init_config, mock_kia_library, mock_security_library

mock_init_config()
mock_kia_library()
mock_security_library()


@pytest.fixture
def temp_dir() -> str:
    return tempfile.mkdtemp()


@pytest.fixture
def export_ctx(temp_dir: str):
    from msmodelslim.format.interface import ExportContext

    return ExportContext(save_directory=Path(temp_dir))


@pytest.fixture
def writer_infra():
    from test.cases.format.compressed_tensors_format.helpers import (
        MockSafetensorsWriterCreatorInfra,
    )

    return MockSafetensorsWriterCreatorInfra()


@pytest.fixture
def quant_format(export_ctx, writer_infra):
    from test.cases.format.compressed_tensors_format.helpers import (
        make_compressed_tensors_quant_format,
    )

    return make_compressed_tensors_quant_format(export_ctx, writer_infra, prepare=True)
