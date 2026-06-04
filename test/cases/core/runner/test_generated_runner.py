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
from torch import nn

from msmodelslim.core.base.protocol import DataUnit
from msmodelslim.core.runner.generated_runner import GeneratedProcessUnit, GeneratedRunner
from msmodelslim.utils.exception import InvalidDatasetError


class TestGeneratedProcessUnit:
    """Tests for GeneratedProcessUnit."""

    def test_init_generators_raises_invalid_dataset_error_when_not_data_free_and_no_calib(self):
        """场景：processor 非 data_free 且无 calib_data。预期：InvalidDatasetError。"""
        processor = MagicMock()
        processor.is_data_free.return_value = False
        processor.__repr__ = lambda self: "MockProcessor"

        unit = GeneratedProcessUnit(
            model=nn.Linear(2, 2),
            processor=processor,
            pipeline_interface=MagicMock(),
            calib_data=None,
            data_recorder=DataUnit(None, None),
        )
        with pytest.raises(InvalidDatasetError, match="Calib data is needed"):
            unit.init_generators()


class TestGeneratedRunner:
    """Tests for GeneratedRunner."""

    def test_add_processor_appends_config_when_append_true(self):
        """场景：add_processor(append=True)。预期：配置追加到列表末尾。"""
        adapter = MagicMock()
        runner = GeneratedRunner(adapter=adapter)
        cfg1 = MagicMock()
        cfg2 = MagicMock()
        runner.add_processor(cfg1)
        runner.add_processor(cfg2)
        assert runner.process_config_list == [cfg1, cfg2]

    def test_add_processor_inserts_at_head_when_append_false(self):
        """场景：add_processor(append=False)。预期：配置插入列表头部。"""
        adapter = MagicMock()
        runner = GeneratedRunner(adapter=adapter)
        cfg1 = MagicMock()
        cfg2 = MagicMock()
        runner.add_processor(cfg1)
        runner.add_processor(cfg2, append=False)
        assert runner.process_config_list == [cfg2, cfg1]
