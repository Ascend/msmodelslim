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

Unit tests for msmodelslim.format.interface.
"""

# pylint: disable=abstract-class-instantiated

from __future__ import annotations

from pathlib import Path

import pytest
from torch import nn

from msmodelslim.format.interface import ExportContext, IFormat


class TestExportContext:
    """Tests for ExportContext dataclass."""

    def test_export_context_defaults_when_minimal(self):
        ctx = ExportContext(save_directory=Path("/tmp/save"))

        assert ctx.source_model_path is None
        assert ctx.rank == 0
        assert ctx.world_size == 1

    def test_export_context_store_fields_when_provided(self):
        ctx = ExportContext(
            save_directory=Path("/tmp/save"),
            source_model_path=Path("/tmp/src"),
            rank=1,
            world_size=4,
        )

        assert ctx.source_model_path == Path("/tmp/src")
        assert ctx.rank == 1
        assert ctx.world_size == 4


class TestIFormat:
    """Tests for IFormat abstract protocol."""

    def test_iformat_raise_type_error_when_instantiated_directly(self):
        with pytest.raises(TypeError):
            IFormat()

    def test_iformat_prepare_export_noop_when_default(self):
        class ConcreteFormat(IFormat):
            def process_module_tensors(self, prefix: str, module: nn.Module) -> None:
                IFormat.process_module_tensors(self, prefix, module)

            def finalize_export(self, model: nn.Module) -> None:
                IFormat.finalize_export(self, model)

        fmt = ConcreteFormat()

        fmt.prepare_export()
        fmt.process_module_tensors("", nn.Linear(2, 2))
        fmt.finalize_export(nn.Linear(2, 2))
