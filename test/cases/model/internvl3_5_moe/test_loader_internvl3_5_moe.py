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

from msmodelslim.model.internvl3_5_moe.loader import InternVL3_5MoeAdapterLoader


def test_adapter_class_path_given_loader_when_called_then_return_expected():
    assert (
        InternVL3_5MoeAdapterLoader.ADAPTER_CLASS_PATH
        == "msmodelslim.model.internvl3_5_moe.model_adapter:InternVL3_5MoeModelAdapter"
    )


def test_loader_instantiation_given_default_when_called_then_succeed():
    loader = InternVL3_5MoeAdapterLoader()
    assert loader.ADAPTER_CLASS_PATH == "msmodelslim.model.internvl3_5_moe.model_adapter:InternVL3_5MoeModelAdapter"
