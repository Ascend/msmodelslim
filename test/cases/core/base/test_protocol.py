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
"""

from torch import nn

from msmodelslim.core.base.protocol import BatchProcessRequest, DataUnit, ProcessRequest


class TestProcessRequest:
    """Tests for ProcessRequest dataclass."""

    def test_process_request_stores_fields_when_constructed(self):
        """场景：正常构造。预期：字段可读。"""
        module = nn.Linear(4, 4)
        req = ProcessRequest(name="layer0", module=module, args=(1,), kwargs={"x": 2})
        assert req.name == "layer0"
        assert req.module is module
        assert req.args == (1,)
        assert req.kwargs == {"x": 2}


class TestBatchProcessRequest:
    """Tests for BatchProcessRequest dataclass."""

    def test_batch_process_request_defaults_datas_outputs_when_omitted(self):
        """场景：仅传 name/module。预期：datas/outputs 默认为 None。"""
        module = nn.Linear(4, 4)
        req = BatchProcessRequest(name="block", module=module)
        assert req.datas is None
        assert req.outputs is None

    def test_batch_process_request_stores_datas_when_provided(self):
        """场景：传入 datas。预期：datas 保留。"""
        module = nn.Linear(4, 4)
        datas = [((1,), {})]
        req = BatchProcessRequest(name="block", module=module, datas=datas)
        assert req.datas is datas


class TestDataUnit:
    """Tests for DataUnit dataclass."""

    def test_data_unit_stores_input_output_when_constructed(self):
        """场景：正常构造。预期：input/output 可读。"""
        unit = DataUnit(input=[1, 2], output=[3, 4])
        assert unit.input == [1, 2]
        assert unit.output == [3, 4]
