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

Unit tests for msmodelslim.format.subpackage_init.
"""

from __future__ import annotations

from msmodelslim.format.ascendV1_format import AscendV1QuantFormatConfig as Av1FromInit
from msmodelslim.format.ascendV1_format.ascendV1 import AscendV1QuantFormatConfig
from msmodelslim.format.common import deqscale2int64, deqscale2int64_by_dtype
from msmodelslim.format.common.deqscale import (
    deqscale2int64 as deq_fn,
    deqscale2int64_by_dtype as deq_by_dtype_fn,
)
from msmodelslim.format.compressed_tensors_format import (
    CompressedTensorsQuantFormat,
    CompressedTensorsQuantFormatConfig,
)
from msmodelslim.format.mindie_format import MindIEQuantFormatConfig as MieFromInit
from msmodelslim.format.mindie_format.mindie import MindIEQuantFormatConfig


class TestSubpackageInitExports:
    def test_ascendv1_init_export_config_when_imported(self):
        assert Av1FromInit is AscendV1QuantFormatConfig

    def test_mindie_init_export_config_when_imported(self):
        assert MieFromInit is MindIEQuantFormatConfig

    def test_common_init_export_deqscale_when_imported(self):
        assert deqscale2int64 is deq_fn
        assert deqscale2int64_by_dtype is deq_by_dtype_fn

    def test_compressed_tensors_init_export_symbols_when_imported(self):
        from msmodelslim.format.compressed_tensors_format.compressed_tensors import (
            CompressedTensorsQuantFormat as CT,
            CompressedTensorsQuantFormatConfig as CTC,
        )

        assert CompressedTensorsQuantFormat is CT
        assert CompressedTensorsQuantFormatConfig is CTC
