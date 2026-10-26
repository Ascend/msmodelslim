#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# pylint: disable=use-implicit-booleaness-not-comparison

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

msmodelslim.infra.plugin_practice_dirs 模块的单元测试
"""

from unittest.mock import MagicMock, patch

from msmodelslim.infra.plugin_practice_dirs import discover_plugin_practice_dirs


def _entry_point(name, loader):
    """构造一个等价于 importlib.metadata.EntryPoint 的 mock"""
    ep = MagicMock()
    ep.name = name
    ep.load.return_value = loader
    return ep


class TestDiscoverPluginPracticeDirs:
    """测试 discover_plugin_practice_dirs 的路径收集与过滤"""

    @patch("msmodelslim.infra.plugin_practice_dirs.entry_points")
    def test_return_resolved_dir_when_plugin_returns_path(self, mock_entry_points, tmp_path):
        """正常路径：插件返回存在的目录时，返回解析后的绝对路径"""
        practice_dir = tmp_path / "practice"
        practice_dir.mkdir()
        mock_entry_points.return_value.select.return_value = [_entry_point("p", lambda: practice_dir)]

        result = discover_plugin_practice_dirs()

        assert result == [practice_dir.resolve()]

    @patch("msmodelslim.infra.plugin_practice_dirs.entry_points")
    def test_return_empty_when_callable_returns_none(self, mock_entry_points):
        """边界：插件返回 None 时不产生任何目录"""
        mock_entry_points.return_value.select.return_value = [_entry_point("p", lambda: None)]

        assert discover_plugin_practice_dirs() == []

    @patch("msmodelslim.infra.plugin_practice_dirs.entry_points")
    def test_filter_out_nonexistent_dir(self, mock_entry_points, tmp_path):
        """边界：插件返回不存在的目录时被过滤掉"""
        missing = tmp_path / "missing"
        mock_entry_points.return_value.select.return_value = [_entry_point("p", lambda: missing)]

        assert discover_plugin_practice_dirs() == []

    @patch("msmodelslim.infra.plugin_practice_dirs.entry_points")
    def test_accept_str_and_iterable_paths(self, mock_entry_points, tmp_path):
        """正常路径：插件返回字符串与 Path 的可迭代对象均被收集"""
        first = tmp_path / "first"
        second = tmp_path / "second"
        first.mkdir()
        second.mkdir()
        mock_entry_points.return_value.select.return_value = [_entry_point("p", lambda: [str(first), second])]

        result = discover_plugin_practice_dirs()

        assert result == [first.resolve(), second.resolve()]

    @patch("msmodelslim.infra.plugin_practice_dirs.entry_points")
    def test_log_and_skip_when_loader_raises(self, mock_entry_points):
        """异常：插件回调抛异常时被捕获，仅记录告警并跳过"""

        def _boom():
            raise RuntimeError("boom")

        mock_entry_points.return_value.select.return_value = [_entry_point("p", _boom)]

        assert discover_plugin_practice_dirs() == []
