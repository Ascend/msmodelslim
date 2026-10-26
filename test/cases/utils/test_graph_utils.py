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

from unittest.mock import Mock

from torch import nn

from msmodelslim.utils import graph_utils


def test_extract_dag_forwards_arguments_and_returns_adapter(monkeypatch):
    adapter = Mock()
    constructor = Mock(return_value=adapter)
    monkeypatch.setattr(graph_utils, "TorchDAGAdapter", constructor)
    model = nn.Linear(2, 2)
    dummy_input = (model.weight,)
    hook_nodes = ["layer.0"]
    anti_method = "m2"

    result = graph_utils.extract_dag(
        model,
        dummy_input=dummy_input,
        hook_nodes=hook_nodes,
        anti_method=anti_method,
    )

    assert result is adapter
    constructor.assert_called_once_with(
        model,
        dummy_input,
        hook_nodes=hook_nodes,
        anti_method=anti_method,
    )


def test_extract_dag_uses_none_for_optional_arguments(monkeypatch):
    adapter = Mock()
    constructor = Mock(return_value=adapter)
    monkeypatch.setattr(graph_utils, "TorchDAGAdapter", constructor)
    model = nn.Identity()

    assert graph_utils.extract_dag(model) is adapter
    constructor.assert_called_once_with(model, None, hook_nodes=None, anti_method=None)
