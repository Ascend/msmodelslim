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

from torch import nn

from msmodelslim.core.const import DeviceType
from msmodelslim.core.runner.layer_wise_runner import LayerWiseRunner
from msmodelslim.processor import LoadProcessorConfig


class TestLayerWiseRunner:
    """Tests for LayerWiseRunner."""

    def test_preprocess_processor_inserts_load_processor_when_called(self):
        """场景：preprocess_processor。预期：首尾插入 LoadProcessor（load/offload）。"""
        adapter = MagicMock()
        runner = LayerWiseRunner(adapter=adapter, offload_device="meta")
        processor_list = []
        model = nn.Linear(2, 2)

        mock_npu = MagicMock()
        mock_npu.current_device.return_value = 0
        with patch("msmodelslim.core.runner.layer_wise_runner.torch") as mock_torch:
            mock_torch.npu = mock_npu
            runner.preprocess_processor(processor_list, model, device=DeviceType.NPU)

        assert len(processor_list) == 2
        assert isinstance(processor_list[0], LoadProcessorConfig)
        assert processor_list[0].mode == "load"
        assert isinstance(processor_list[-1], LoadProcessorConfig)
        assert processor_list[-1].mode == "offload"
        assert processor_list[-1].device == "meta"

    @patch("msmodelslim.core.runner.layer_wise_runner.get_input_datas")
    @patch.object(LayerWiseRunner, "generated_schedule")
    @patch.object(LayerWiseRunner, "build_process_unit", return_value=[])
    def test_run_calls_get_input_datas_when_invoked(self, mock_build_unit, mock_schedule, mock_get_input_datas):
        """场景：run 调用。预期：mock 的 get_input_datas 被调用。"""
        adapter = MagicMock()
        adapter.init_model.return_value = nn.Linear(2, 2)
        runner = LayerWiseRunner(adapter=adapter)
        calib = [{"x": 1}]
        model = nn.Linear(2, 2)

        runner.run(model=model, calib_data=calib, device=DeviceType.CPU)

        mock_get_input_datas.assert_called_once_with(adapter, calib, DeviceType.CPU)
        mock_build_unit.assert_called_once()
        mock_schedule.assert_called_once()
