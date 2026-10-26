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

from typing import List
from unittest.mock import patch

import pytest
import torch
from torch import nn

from msmodelslim.core.base.protocol import BatchProcessRequest
from msmodelslim.core.graph.adapter_types import AdapterConfig, MappingConfig
from msmodelslim.processor.anti_outlier.common.smooth_components import StatKey
from msmodelslim.processor.anti_outlier.flex_smooth import FlexSmoothQuantProcessor, FlexSmoothQuantProcessorConfig
from msmodelslim.processor.anti_outlier.flex_smooth.interface import FlexSmoothQuantInterface
from msmodelslim.utils.exception import SchemaValidateError


class _MockModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(10)
        self.layer1 = nn.Linear(10, 20)
        self.layer2 = nn.Linear(20, 30)

    def forward(self, inputs):
        return self.layer2(self.layer1(self.norm(inputs)))


class _MockFlexSmoothQuantInterface(FlexSmoothQuantInterface):
    def get_adapter_config_for_subgraph(self) -> List[AdapterConfig]:
        return [
            AdapterConfig(
                subgraph_type="norm-linear",
                mapping=MappingConfig(source="norm", targets=["layer1"]),
            )
        ]


class TestFlexSmoothQuantProcessor:
    def _make_processor(self, **config_kwargs):
        model = _MockModel()
        adapter = _MockFlexSmoothQuantInterface()
        config_kwargs.setdefault("alpha", 0.5)
        config_kwargs.setdefault("beta", 0.8)
        config_kwargs.setdefault("enable_subgraph_type", ["norm-linear"])
        config = FlexSmoothQuantProcessorConfig(**config_kwargs)
        return FlexSmoothQuantProcessor(model=model, config=config, adapter=adapter)

    def test_init_when_adapter_and_config_are_valid_then_public_attributes_match(self):
        model = _MockModel()
        adapter = _MockFlexSmoothQuantInterface()
        config = FlexSmoothQuantProcessorConfig(alpha=0.5, beta=0.8, enable_subgraph_type=["norm-linear"])
        processor = FlexSmoothQuantProcessor(model=model, config=config, adapter=adapter)

        assert processor.model is model
        assert processor.config is config
        assert processor.adapter is adapter
        assert processor.config.type == "flex_smooth_quant"

    def test_support_distributed_when_called_then_returns_true(self):
        processor = self._make_processor()

        assert processor.support_distributed() is True

    def test_preprocess_when_adapter_mapping_matches_request_then_installs_statistics_hook(self):
        processor = self._make_processor()
        request = BatchProcessRequest(name="", module=processor.model)

        processor.preprocess(request)
        with torch.no_grad():
            processor.model(torch.randn(2, 10))

        stats = processor.stats_collector.act_stats
        assert list(stats) == ["layer1"]
        assert stats["layer1"][StatKey.TENSOR][0].shape == (2, 10)
        assert stats["layer1"][StatKey.STAT_KEY_SMOOTH_SCALE].shape == (10,)

    def test_preprocess_when_subgraph_type_disabled_then_installs_no_statistics_hook(self):
        processor = self._make_processor(enable_subgraph_type=["linear-linear"])
        request = BatchProcessRequest(name="", module=processor.model)

        processor.preprocess(request)
        with torch.no_grad():
            processor.model(torch.randn(2, 10))

        assert not processor.stats_collector.act_stats

    def test_init_when_subgraph_type_is_invalid_then_raises_schema_validate_error(self):
        model = _MockModel()
        adapter = _MockFlexSmoothQuantInterface()
        config = FlexSmoothQuantProcessorConfig(alpha=0.5, beta=0.8, enable_subgraph_type=["not-a-subgraph"])

        with pytest.raises(SchemaValidateError, match="enable_subgraph_type"):
            FlexSmoothQuantProcessor(model=model, config=config, adapter=adapter)

    def test_postprocess_when_no_statistics_then_does_not_call_flex_smooth_quant(self):
        processor = self._make_processor()
        request = BatchProcessRequest(name="", module=processor.model)
        processor.preprocess(request)

        with patch("msmodelslim.processor.anti_outlier.flex_smooth.processor.flex_smooth_quant") as mock_api:
            processor.postprocess(request)

        mock_api.assert_not_called()

    def test_postprocess_when_statistics_collected_then_calls_flex_smooth_quant_with_configured_alpha_beta(self):
        processor = self._make_processor()
        request = BatchProcessRequest(name="", module=processor.model)
        processor.preprocess(request)
        with torch.no_grad():
            processor.model(torch.randn(2, 10))

        with patch("msmodelslim.processor.anti_outlier.flex_smooth.processor.flex_smooth_quant") as mock_api:
            processor.postprocess(request)

        mock_api.assert_called_once()
        smooth_config = mock_api.call_args.args[1]
        assert smooth_config.alpha == 0.5
        assert smooth_config.beta == 0.8

    def test_postprocess_when_statistics_collected_then_clears_statistics_and_hooks(self):
        processor = self._make_processor()
        request = BatchProcessRequest(name="", module=processor.model)
        processor.preprocess(request)
        with torch.no_grad():
            processor.model(torch.randn(2, 10))

        with patch("msmodelslim.processor.anti_outlier.flex_smooth.processor.flex_smooth_quant"):
            processor.postprocess(request)

        assert not processor.stats_collector.act_stats
        with torch.no_grad():
            processor.model(torch.randn(2, 10))
        assert not processor.stats_collector.act_stats


class TestFlexSmoothQuantProcessorConfig:
    def test_default_config_when_created_then_uses_flex_smooth_quant_defaults(self):
        config = FlexSmoothQuantProcessorConfig()

        assert config.type == "flex_smooth_quant"
        assert config.alpha is None
        assert config.beta is None
        assert config.enable_subgraph_type == ["norm-linear", "linear-linear", "ov", "up-down"]

    def test_custom_config_when_created_then_preserves_supplied_values(self):
        config = FlexSmoothQuantProcessorConfig(
            alpha=0.7,
            beta=0.9,
            enable_subgraph_type=["norm-linear", "linear-linear"],
        )

        assert config.alpha == 0.7
        assert config.beta == 0.9
        assert config.enable_subgraph_type == ["norm-linear", "linear-linear"]

    @pytest.mark.parametrize(
        ("field_name", "value"),
        [("alpha", -1.0), ("alpha", 2.0), ("beta", -0.5), ("beta", 1.5)],
    )
    def test_config_validation_when_value_out_of_range_then_raises_schema_validate_error(self, field_name, value):
        with pytest.raises(SchemaValidateError, match=field_name):
            FlexSmoothQuantProcessorConfig(**{field_name: value})
