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

Pytest config for precheck tests.

注册预检查插件，使 BasePrecheckConfig 按 type 字段派发到具体子类完成校验。
"""


def pytest_configure(config):
    """注册 expected_answer / garbled_text 预检查插件（幂等）。"""
    from msmodelslim.utils.plugin.plugin_utils import register_plugin
    from msmodelslim.infra.evaluation.precheck.expected_answer_rule import get_plugin as expected_answer_gp
    from msmodelslim.infra.evaluation.precheck.garbled_text_rule import get_plugin as garbled_text_gp

    for getter in (expected_answer_gp, garbled_text_gp):
        register_plugin(getter)
