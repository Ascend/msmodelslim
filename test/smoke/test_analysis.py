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

from unittest.mock import MagicMock, patch  # noqa: F401
import pytest


from .base import invoke_analysis_test


@pytest.mark.smoke
def test_analysis_cli_coverage():
    """测试分析模块CLI主函数代码覆盖"""
    # 测试基本功能
    result = invoke_analysis_test(metrics="kurtosis", patterns=["*"], topk=15)

    # 测试不同算法
    for metrics in ["kurtosis", "std", "quantile"]:
        result = invoke_analysis_test(metrics=metrics, patterns=["*"], topk=10)

    # 测试不同层模式
    for patterns in [["*"], ["*attention*"], ["*mlp*"], ["*attention*", "*mlp*"]]:
        result = invoke_analysis_test(metrics="kurtosis", patterns=patterns, topk=15)

    # 测试不同topk值
    for topk in [5, 10, 15, 20]:
        result = invoke_analysis_test(metrics="kurtosis", patterns=["*"], topk=topk)

    # 测试trust_remote_code（现在已固定为False）
    result = invoke_analysis_test(metrics="kurtosis", patterns=["*"], topk=15)  # noqa: F841
