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
# pylint: disable=no-member

from unittest.mock import patch
import pytest

from msmodelslim.utils.dependency_check import DependencyChecker, require_packages, get_require_packages
from msmodelslim.utils.exception import VersionError


def setup_function():
    """所有测试前清空 _requirements"""
    DependencyChecker._requirements.clear()


# -----------------------------------------
# test add()
# -----------------------------------------
def test_add_requirements():
    DependencyChecker.set_plugin("pluginA", {"torch": ">=2.0"})
    DependencyChecker.set_plugin("pluginA", {"numpy": ">=1.25"})

    assert "pluginA" in DependencyChecker._requirements
    assert DependencyChecker._requirements["pluginA"]["torch"] == ">=2.0"
    assert DependencyChecker._requirements["pluginA"]["numpy"] == ">=1.25"


# -----------------------------------------
# test check_plugin (mock _check_single)
# -----------------------------------------
@patch("msmodelslim.utils.dependency_check.DependencyChecker._check_single")
def test_check_plugin(mock_check):
    # 准备数据
    DependencyChecker.set_plugin("pluginA", {"torch": ">=2.0", "numpy": "<2.0"})

    # 执行
    DependencyChecker.check_plugin("pluginA")

    # 断言：_check_single 被正确调用
    assert mock_check.call_count == 2
    mock_check.assert_any_call("torch", ">=2.0")
    mock_check.assert_any_call("numpy", "<2.0")


# -----------------------------------------
# test _check_single 版本不满足时抛异常
# -----------------------------------------
@patch("msmodelslim.utils.dependency_check.metadata_version")
def test_check_single_version_invalid(mock_metadata):
    # 模拟包已安装，但版本不满足要求
    mock_metadata.return_value = "1.0.0"  # installed version

    with pytest.raises(VersionError):
        DependencyChecker._check_single("torch", ">=2.0")


# -----------------------------------------
# test require_packages decorator
# -----------------------------------------
def test_require_packages_decorator():
    @require_packages({"torch": ">=2.0"})
    class MyModel:
        pass

    assert hasattr(MyModel, "_require_packages")
    assert MyModel._require_packages == {"torch": ">=2.0"}


# -----------------------------------------
# test get_require_packages (内部调用 set_plugin)
# -----------------------------------------
def test_get_require_packages():
    @require_packages({"numpy": ">=1.0"})
    class MyModel:
        pass

    pkg = get_require_packages(MyModel)

    assert pkg == {"numpy": ">=1.0"}
