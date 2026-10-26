# -*- coding: UTF-8 -*-

# pylint: disable=abstract-class-instantiated

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

Unit tests for msmodelslim.format.format_handler.
"""

from __future__ import annotations

import os

import pytest

from msmodelslim.format.ascendV1_format.format import ASCENDV1_DESC_JSON_NAME
from msmodelslim.format.format_handler import AscendV1FormatHandler, FormatHandler, build_default_format_chain
from msmodelslim.utils.exception import SchemaValidateError


class _StubHandler(FormatHandler):
    """Concrete handler whose recognition stub is fully controlled by the test."""

    def __init__(self, handled: bool, result: object = None, successor: FormatHandler = None):
        super().__init__(successor)
        self._handled = handled
        self._result = result
        self.create_calls = []

    def can_handle(self, model_path: str) -> bool:
        return self._handled

    def create(self, model_path: str, device: str = "cpu") -> object:
        self.create_calls.append((model_path, device))
        return self._result


class TestFormatHandler:
    """Tests for the abstract chain-of-responsibility base."""

    def test_format_handler_raises_type_error_when_instantiated_directly(self):
        with pytest.raises(TypeError):
            FormatHandler()

    def test_handle_returns_created_loader_when_can_handle(self):
        sentinel = object()
        handler = _StubHandler(handled=True, result=sentinel)

        assert handler.handle("/model", "npu") is sentinel
        assert handler.create_calls == [("/model", "npu")]

    def test_handle_delegates_to_successor_when_not_handled(self):
        sentinel = object()
        successor = _StubHandler(handled=True, result=sentinel)
        head = _StubHandler(handled=False)
        assert head.set_next(successor) is successor

        assert head.handle("/model") is sentinel

    def test_handle_raises_schema_validate_error_when_no_successor_recognizes(self):
        head = _StubHandler(handled=False)

        with pytest.raises(SchemaValidateError, match="No fake-quant weight format handler recognized"):
            head.handle("/unknown")


class TestAscendV1FormatHandler:
    """Tests for the default AscendV1 recognizer."""

    def test_can_handle_true_when_desc_json_present(self, tmp_path):
        (tmp_path / ASCENDV1_DESC_JSON_NAME).write_text("{}", encoding="utf-8")

        assert AscendV1FormatHandler().can_handle(str(tmp_path)) is True

    def test_can_handle_false_when_desc_json_missing(self, tmp_path):
        assert AscendV1FormatHandler().can_handle(str(tmp_path)) is False

    def test_can_handle_false_when_path_is_directory_only(self, tmp_path):
        nested = os.path.join(str(tmp_path), "not_a_dir_file")
        assert AscendV1FormatHandler().can_handle(nested) is False


class TestBuildDefaultFormatChain:
    """Tests for build_default_format_chain."""

    def test_returns_ascendv1_handler(self):
        assert isinstance(build_default_format_chain(), AscendV1FormatHandler)
