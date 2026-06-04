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

from unittest.mock import MagicMock, patch

from msmodelslim.core.const import DeviceType
from msmodelslim.core.runner.pipeline_parallel_runner import PPRunner


class TestPPRunner:
    """Tests for PPRunner."""

    @patch("msmodelslim.core.runner.generated_runner.GeneratedRunner.run")
    @patch("msmodelslim.core.runner.pipeline_parallel_runner.get_logger")
    def test_run_warns_when_device_indices_provided(self, mock_get_logger, mock_parent_run):
        """场景：run 传入 device_indices。预期：记录 device indices 将被忽略的 warning。"""
        adapter = MagicMock()
        runner = PPRunner(adapter=adapter)
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        runner.run(model=None, calib_data=None, device=DeviceType.NPU, device_indices=[0, 1])

        mock_logger.warning.assert_called_once()
        warning_msg = mock_logger.warning.call_args[0][0]
        assert "device indices" in warning_msg.lower() or "Device indices" in warning_msg
        mock_parent_run.assert_called_once()
