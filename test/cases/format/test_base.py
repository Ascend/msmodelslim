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

Unit tests for msmodelslim.format.base.
"""

from __future__ import annotations

# pylint: disable=no-name-in-module

from typing import Callable, Dict, Set, Type

from torch import nn

import msmodelslim.ir as qir
from msmodelslim.format.base import QuantFormatBase, QuantFormatConfig
from msmodelslim.format.interface import ExportContext
from msmodelslim.ir.wrapper import WrapperIR

from test.cases.format.compressed_tensors_format.helpers import make_w8a8_static_module


class AtomicWrapper(WrapperIR):
    @staticmethod
    def is_atomic() -> bool:
        return True


class NonAtomicWrapper(WrapperIR):
    @staticmethod
    def is_atomic() -> bool:
        return False


class DummyFormat(QuantFormatBase):
    """Minimal concrete format for testing base traversal logic."""

    def __init__(self, config: QuantFormatConfig, ctx: ExportContext) -> None:
        super().__init__(config, ctx)
        self.handled: Set[str] = set()
        self.float_calls: list[str] = []

    def build_module_handler_map(self) -> Dict[Type[nn.Module], Callable[[str, nn.Module], None]]:
        return {qir.W8A8StaticFakeQuantLinear: self._on_quant}

    def on_float_module(self, prefix: str, module: nn.Module) -> None:
        self.float_calls.append(prefix)

    def _on_quant(self, prefix: str, module: nn.Module) -> None:
        self.handled.add(prefix)


class TestQuantFormatConfig:
    def test_set_save_directory_update_path_when_called(self):
        config = QuantFormatConfig()

        config.set_save_directory("/tmp/out")

        assert config.save_directory == "/tmp/out"


class TestQuantFormatBase:
    def test_process_module_tensors_call_handler_when_qir_module(self, tmp_path):
        ctx = ExportContext(save_directory=tmp_path)
        fmt = DummyFormat(QuantFormatConfig(), ctx)
        model = nn.Module()
        model.quant = make_w8a8_static_module()

        fmt.process_module_tensors("", model)

        assert "quant" in fmt.handled

    def test_process_module_tensors_call_on_float_when_no_handler(self, tmp_path):
        ctx = ExportContext(save_directory=tmp_path)
        fmt = DummyFormat(QuantFormatConfig(), ctx)
        model = nn.Sequential(nn.Linear(4, 2))

        fmt.process_module_tensors("", model)

        assert fmt.float_calls

    def test_process_module_tensors_skip_duplicate_when_already_processed(self, tmp_path):
        ctx = ExportContext(save_directory=tmp_path)
        fmt = DummyFormat(QuantFormatConfig(), ctx)
        model = nn.Module()
        model.quant = make_w8a8_static_module()
        fmt.process_module_tensors("", model)
        fmt.handled.clear()

        fmt.process_module_tensors("", model)

        assert not fmt.handled

    def test_process_module_tensors_handle_non_atomic_wrapper_and_wrapped(self, tmp_path):
        ctx = ExportContext(save_directory=tmp_path)
        fmt = DummyFormat(QuantFormatConfig(), ctx)
        inner = make_w8a8_static_module()
        wrapper = NonAtomicWrapper(inner)
        model = nn.Module()
        model.wrapper = wrapper

        fmt.process_module_tensors("", model)

        assert "wrapper" in fmt.handled

    def test_process_module_tensors_handle_atomic_wrapper_only(self, tmp_path):
        ctx = ExportContext(save_directory=tmp_path)
        fmt = DummyFormat(QuantFormatConfig(), ctx)
        inner = make_w8a8_static_module()
        wrapper = AtomicWrapper(inner)
        model = nn.Module()
        model.wrapper = wrapper

        fmt.process_module_tensors("", model)

        assert "wrapper" in fmt.float_calls

    def test_finalize_export_and_merge_ranks_noop_when_default(self, tmp_path):
        ctx = ExportContext(save_directory=tmp_path)
        fmt = DummyFormat(QuantFormatConfig(), ctx)
        model = nn.Linear(4, 2)

        fmt.finalize_export(model)
        fmt.merge_ranks()
