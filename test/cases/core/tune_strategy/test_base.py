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

from unittest.mock import MagicMock

import pytest

from msmodelslim.core.tune_strategy.base import BaseTuningStrategy


class TestBaseTuningStrategy:
    """Tests for BaseTuningStrategy abstract contract."""

    def test_cannot_instantiate_subclass_without_generate_practice(self):
        """场景：子类未实现 generate_practice。预期：无法实例化（TypeError）。"""

        class IncompleteStrategy(BaseTuningStrategy):
            pass

        with pytest.raises(TypeError):
            IncompleteStrategy(  # pylint: disable=abstract-class-instantiated
                config=MagicMock(), dataset_loader=MagicMock()
            )
