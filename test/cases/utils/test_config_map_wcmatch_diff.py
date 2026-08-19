#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
-------------------------------------------------------------------------
This file is part of the MindStudio project.
Copyright (c) 2025 Huawei Technologies Co.,Ltd.

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

# 与 wcmatch 的差分测试：验证自研 brace 展开 + 标准库 fnmatch 在"支持的子集"上
# 与 wcmatch BRACE 语义一致（前后一致性）。wcmatch 已从依赖中移除，环境恰好装有
# 时执行本测试；未安装时自动跳过（干净环境不影响）。

import pytest

fnmatch_wcmatch = pytest.importorskip("wcmatch.fnmatch")

from msmodelslim.utils.config_map import _brace_fnmatch  # noqa: E402


Q_WEN3_BRACE = "model.layers.{1,2,3,4,5,6,7,8,30,31,32,43,44,45,46,52,60,61,62,63}.mlp.up_proj"

# 支持的子集：花括号交替（含嵌套/多组/空段/单元素/未闭合）+ 通配符。
# 每个用例断言自研实现与 wcmatch BRACE 结果一致。
SUPPORTED_CASES = [
    ("model.layers.2.mlp.up_proj", "model.layers.{1,2,3}.mlp.up_proj"),
    ("model.layers.9.mlp.up_proj", "model.layers.{1,2,3}.mlp.up_proj"),
    ("a.1", "{a,b}.{1,2}"),
    ("b.2", "{a,b}.{1,2}"),
    ("b", "{a,{b,c}}"),
    ("c", "{a,{b,c}}"),
    ("{a}", "{a}"),
    ("a", "{a}"),
    ("a", "{a,,c}"),
    ("model.{1,2", "model.{1,2"),
    ("layer1", "{layer1,layer2}"),
    ("", ""),
    ("x", "*.x"),
    ("model.layers.0.self_attn.q_proj", "{0,1}.self_attn.*"),
    ("model.layers.2.mlp.up_proj", "model.layers.{1,2,3}.mlp.*"),
    ("model.layers.61.mlp.up_proj", Q_WEN3_BRACE),
    ("model.layers.9.mlp.up_proj", Q_WEN3_BRACE),
    ("model.layers.2.mlp.down_proj", Q_WEN3_BRACE),
]

# 已知差异（有意为之，仓库内无此类真实场景）：
# 1) 区间语法 {1..5}：wcmatch 会展开为 1..5，自研实现按字面量处理；
# 2) 反斜杠转义 \x：wcmatch 还原为字面量 x，标准库 fnmatch 不处理转义、保留反斜杠。
KNOWN_DIFF_CASES = [
    ("model.layers.5.mlp.up_proj", "model.layers.{1..5}.mlp.up_proj"),
    ("a{b,c}", r"a\{b,c}"),
    ("*", r"\*"),
]


class TestWcmatchDiff:
    """自研实现 vs wcmatch BRACE 的差分验证。"""

    @pytest.mark.parametrize("name,pattern", SUPPORTED_CASES)
    def test_brace_fnmatch_matches_wcmatch_when_pattern_in_supported_subset(self, name, pattern):
        expected = fnmatch_wcmatch.fnmatch(name, pattern, flags=fnmatch_wcmatch.BRACE)
        assert _brace_fnmatch(name, pattern) == expected, (
            f"pattern={pattern!r} name={name!r}: ours={_brace_fnmatch(name, pattern)} wcmatch={expected}"
        )

    @pytest.mark.parametrize("name,pattern", KNOWN_DIFF_CASES)
    def test_brace_fnmatch_differs_from_wcmatch_when_pattern_uses_range_or_escape(self, name, pattern):
        """区间与反斜杠转义为已知差异：wcmatch 支持、自研实现不支持，需显式知晓。"""
        expected = fnmatch_wcmatch.fnmatch(name, pattern, flags=fnmatch_wcmatch.BRACE)
        ours = _brace_fnmatch(name, pattern)
        assert expected is True and ours is False, (
            f"pattern={pattern!r} name={name!r} 的行为发生变化，请复核：wcmatch={expected} ours={ours}"
        )
