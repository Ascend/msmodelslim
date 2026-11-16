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

import pytest

from msmodelslim.core.context.summary import QuantStatisticsObserver


def test_quant_statistics_observer_cannot_be_instantiated_when_contract_is_unimplemented():
    with pytest.raises(TypeError):
        QuantStatisticsObserver()  # pylint: disable=abstract-class-instantiated


def test_subclass_cannot_be_instantiated_when_format_is_missing():
    class _WithoutFormat(QuantStatisticsObserver):
        key = "demo"

        def aggregate(self, context):
            return None

    with pytest.raises(TypeError):
        _WithoutFormat()  # pylint: disable=abstract-class-instantiated


def test_subclass_is_accepted_when_key_aggregate_and_format_are_declared():
    class _DemoObserver(QuantStatisticsObserver):
        key = "demo"

        def aggregate(self, context):
            return None

        def format(self, result):
            return "demo"

    observer = _DemoObserver()
    assert observer.key == "demo"
    assert observer.aggregate(None) is None
    assert observer.format(None) == "demo"
