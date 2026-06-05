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

import unittest
from unittest.mock import patch

from msmodelslim.utils.distributed.dist_helper import is_rank_zero


class TestIsRankZeroWhenDistNotAvailable(unittest.TestCase):
    """测试is_rank_zero在分布式不可用时的行为"""

    @patch("msmodelslim.utils.distributed.dist_helper.dist.is_available", return_value=False)
    def test_is_rank_zero_returns_true_when_dist_not_available(self, _):
        """正常：分布式不可用时应返回True"""
        result = is_rank_zero()
        self.assertTrue(result)


class TestIsRankZeroWhenDistNotInitialized(unittest.TestCase):
    """测试is_rank_zero在分布式未初始化时的行为"""

    @patch("msmodelslim.utils.distributed.dist_helper.dist.is_available", return_value=True)
    @patch("msmodelslim.utils.distributed.dist_helper.dist.is_initialized", return_value=False)
    def test_is_rank_zero_returns_true_when_dist_not_initialized(self, *_):
        """正常：分布式未初始化时应返回True"""
        result = is_rank_zero()
        self.assertTrue(result)


class TestIsRankZeroWhenRankIsZero(unittest.TestCase):
    """测试is_rank_zero在rank为0时的行为"""

    @patch("msmodelslim.utils.distributed.dist_helper.dist.is_available", return_value=True)
    @patch("msmodelslim.utils.distributed.dist_helper.dist.is_initialized", return_value=True)
    @patch("msmodelslim.utils.distributed.dist_helper.dist.get_rank", return_value=0)
    def test_is_rank_zero_returns_true_when_rank_is_zero(self, *_):
        """正常：rank为0时应返回True"""
        result = is_rank_zero()
        self.assertTrue(result)


class TestIsRankZeroWhenRankIsNonZero(unittest.TestCase):
    """测试is_rank_zero在rank非0时的行为"""

    @patch("msmodelslim.utils.distributed.dist_helper.dist.is_available", return_value=True)
    @patch("msmodelslim.utils.distributed.dist_helper.dist.is_initialized", return_value=True)
    @patch("msmodelslim.utils.distributed.dist_helper.dist.get_rank", return_value=3)
    def test_is_rank_zero_returns_false_when_rank_is_nonzero(self, *_):
        """异常：rank非0时应返回False"""
        result = is_rank_zero()
        self.assertFalse(result)
