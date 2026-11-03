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

import torch

from msmodelslim.format.interface import ExportContext


def test_ExportContext_defaults_state_dict_when_omitted(tmp_path):
    ctx = ExportContext(save_directory=tmp_path)
    assert ctx.state_dict is None


def test_ExportContext_defaults_rank_when_omitted(tmp_path):
    ctx = ExportContext(save_directory=tmp_path)
    assert ctx.rank == 0


def test_ExportContext_defaults_world_size_when_omitted(tmp_path):
    ctx = ExportContext(save_directory=tmp_path)
    assert ctx.world_size == 1


def test_ExportContext_keeps_state_dict_when_provided(tmp_path):
    weights = {"layer.weight": torch.ones(2, 2)}
    ctx = ExportContext(save_directory=Path(tmp_path), state_dict=weights)
    assert ctx.state_dict is weights


def test_ExportContext_keeps_empty_state_dict_when_provided(tmp_path):
    weights = {}
    ctx = ExportContext(save_directory=Path(tmp_path), state_dict=weights)
    assert ctx.state_dict is weights
