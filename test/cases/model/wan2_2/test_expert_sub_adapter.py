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

# pylint: disable=consider-using-from-import

from unittest.mock import Mock

import torch.nn as nn

from msmodelslim.core.graph.adapter_types import AdapterConfig
from msmodelslim.model.base import BaseModelAdapter
from msmodelslim.model.wan2_2.expert_sub_adapter import (
    Wan2_2ExpertSubAdapter,
    Wan2_2HighNoiseSubAdapter,
    Wan2_2LowNoiseSubAdapter,
)
from msmodelslim.processor.quant.fa3.interface import FA3QuantAdapterInterface
from msmodelslim.processor.quarot import OnlineQuaRotInterface


class TestWan2_2ExpertSubAdapter:
    # ===== 接口实现 =====

    def test_implements_quarot_interface_when_low_noise_sub_adapter_created(self):
        parent = Mock()
        parent.get_online_rotation_configs = Mock(return_value={})
        parent.inject_fa3_placeholders = Mock()

        sub = Wan2_2LowNoiseSubAdapter(parent, "low_noise_model")
        sub.bind_module(nn.Linear(4, 4))

        assert isinstance(sub, OnlineQuaRotInterface)
        assert isinstance(sub, FA3QuantAdapterInterface)

        sub.get_online_rotation_configs(sub._module)
        parent.get_online_rotation_configs.assert_called_once_with(sub._module)

    # ===== 继承 BaseModelAdapter =====

    def test_inherits_base_model_adapter_when_expert_sub_adapter_created(self):
        parent = Mock()
        sub = Wan2_2ExpertSubAdapter(parent, "test_expert")
        assert isinstance(sub, BaseModelAdapter)

    def test_inherits_base_model_adapter_when_low_noise_sub_adapter_created(self):
        parent = Mock()
        sub = Wan2_2LowNoiseSubAdapter(parent, "low_noise_model")
        assert isinstance(sub, BaseModelAdapter)

    def test_inherits_base_model_adapter_when_high_noise_sub_adapter_created(self):
        parent = Mock()
        sub = Wan2_2HighNoiseSubAdapter(parent, "high_noise_model")
        assert isinstance(sub, BaseModelAdapter)

    # ===== get_adapter_config_for_subgraph =====

    def test_get_adapter_config_for_subgraph_delegates_to_parent_when_called(self):
        mock_module = Mock()
        mock_module.num_layers = 3
        expected_configs = [Mock(spec=AdapterConfig)]
        parent = Mock()
        parent.get_adapter_config_for_subgraph = Mock(return_value=expected_configs)

        sub = Wan2_2ExpertSubAdapter(parent, "test_expert")
        sub.bind_module(mock_module)

        result = sub.get_adapter_config_for_subgraph()

        parent.get_adapter_config_for_subgraph.assert_called_once_with(3)
        assert result == expected_configs

    def test_get_adapter_config_for_subgraph_passes_module_num_layers_when_low_noise_adapter(self):
        mock_module = Mock()
        mock_module.num_layers = 10
        parent = Mock()
        parent.get_adapter_config_for_subgraph = Mock(return_value=[])

        sub = Wan2_2LowNoiseSubAdapter(parent, "low_noise_model")
        sub.bind_module(mock_module)

        sub.get_adapter_config_for_subgraph()

        parent.get_adapter_config_for_subgraph.assert_called_once_with(10)

    def test_get_adapter_config_for_subgraph_returns_parent_result_when_high_noise_adapter(self):
        mock_module = Mock()
        mock_module.num_layers = 2
        configs = [Mock(spec=AdapterConfig), Mock(spec=AdapterConfig)]
        parent = Mock()
        parent.get_adapter_config_for_subgraph = Mock(return_value=configs)

        sub = Wan2_2HighNoiseSubAdapter(parent, "high_noise_model")
        sub.bind_module(mock_module)

        result = sub.get_adapter_config_for_subgraph()
        assert result is configs
