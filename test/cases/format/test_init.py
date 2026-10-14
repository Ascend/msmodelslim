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

Unit tests for msmodelslim.format.__init__ lazy exports.
"""

from __future__ import annotations

import pytest

import msmodelslim.format as format_pkg
from msmodelslim.format.ascendV1_format.ascendV1 import AscendV1QuantFormatConfig
from msmodelslim.format.compressed_tensors_format.compressed_tensors import (
    CompressedTensorsQuantFormat,
    CompressedTensorsQuantFormatConfig,
)
from msmodelslim.format.mindie_format.mindie import MindIEQuantFormatConfig


class TestFormatLazyExports:
    """Tests for format package lazy attribute loading."""

    def test_lazy_export_compressed_tensors_quant_format_when_accessed(self):
        assert format_pkg.CompressedTensorsQuantFormat is CompressedTensorsQuantFormat

    def test_lazy_export_compressed_tensors_quant_format_config_when_accessed(self):
        assert format_pkg.CompressedTensorsQuantFormatConfig is CompressedTensorsQuantFormatConfig

    def test_lazy_export_ascend_v1_quant_format_config_when_accessed(self):
        assert format_pkg.AscendV1QuantFormatConfig is AscendV1QuantFormatConfig

    def test_lazy_export_mindie_quant_format_config_when_accessed(self):
        assert format_pkg.MindIEQuantFormatConfig is MindIEQuantFormatConfig

    def test_getattr_raise_attribute_error_when_name_unknown(self):
        with pytest.raises(AttributeError, match="has no attribute"):
            _ = format_pkg.NotARealAttribute
