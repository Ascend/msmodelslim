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

from msmodelslim.core.runner.base import BaseRunner


class TestBaseRunner:
    """Tests for BaseRunner abstract base."""

    def test_run_is_abstract_method_when_inspected(self):
        """场景：检查 BaseRunner.run。预期：标记为抽象方法。"""
        assert getattr(BaseRunner.run, "__isabstractmethod__", False) is True

    def test_add_processor_is_abstract_method_when_inspected(self):
        """场景：检查 BaseRunner.add_processor。预期：标记为抽象方法。"""
        assert getattr(BaseRunner.add_processor, "__isabstractmethod__", False) is True
