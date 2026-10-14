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

from typing import List

import pytest
import torch
import torch.nn as nn

from unittest.mock import MagicMock

from msmodelslim.core.base.protocol import BatchProcessRequest
from msmodelslim.core.context import ContextFactory, ContextManager
from msmodelslim.core.graph.adapter_types import AdapterConfig, MappingConfig
from msmodelslim.core.quantizer.base import QConfig
from msmodelslim.core.quantizer.linear import LinearQConfig
from msmodelslim.ir.qal.qbase import QDType, QScope
from msmodelslim.processor.anti_outlier.awq import AWQInterface
from msmodelslim.processor.anti_outlier.awq.processor import AWQProcessor, AWQProcessorConfig
from msmodelslim.utils.exception import UnsupportedError


@pytest.fixture(autouse=True)
def _context():
    """Ensure a global context is active for every test."""
    ctx = ContextFactory().create(is_distributed=False)
    with ContextManager(ctx):
        yield


class MockMLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))


class MockRMSNorm(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x):
        return x * self.weight


class MockDecoderLayer(nn.Module):
    def __init__(self, hidden_size=64, intermediate_size=128):
        super().__init__()
        self.input_layernorm = MockRMSNorm(hidden_size)
        self.post_attention_layernorm = MockRMSNorm(hidden_size)
        self.mlp = MockMLP(hidden_size, intermediate_size)

    def forward(self, x):
        x = self.input_layernorm(x)
        x = self.mlp(self.post_attention_layernorm(x))
        return x


class MockModel(nn.Module):
    def __init__(self, hidden_size=64, intermediate_size=128, num_layers=2):
        super().__init__()
        self.config = type(
            "Config",
            (),
            {
                "num_hidden_layers": num_layers,
                "num_attention_heads": 4,
                "num_key_value_heads": 4,
                "hidden_size": hidden_size,
            },
        )()
        layers = nn.ModuleList([MockDecoderLayer(hidden_size, intermediate_size) for _ in range(num_layers)])
        self.model = nn.Module()
        self.model.layers = layers

    def forward(self, x):
        for layer in self.model.layers:
            x = layer(x)
        return x


class MockAdapter(AWQInterface):
    def __init__(self, num_layers=2, include_non_fusion=False):
        self.num_layers = num_layers
        self.include_non_fusion = include_non_fusion

    def get_adapter_config_for_subgraph(self) -> List[AdapterConfig]:
        configs = []
        for i in range(self.num_layers):
            prefix = f"model.layers.{i}"
            configs.append(
                AdapterConfig(
                    subgraph_type="norm-linear",
                    mapping=MappingConfig(
                        source=f"{prefix}.post_attention_layernorm",
                        targets=[f"{prefix}.mlp.gate_proj", f"{prefix}.mlp.up_proj"],
                    ),
                )
            )
            configs.append(
                AdapterConfig(
                    subgraph_type="up-down",
                    mapping=MappingConfig(
                        source=f"{prefix}.mlp.up_proj",
                        targets=[f"{prefix}.mlp.down_proj"],
                    ),
                )
            )
            if self.include_non_fusion:
                configs.append(
                    AdapterConfig(
                        subgraph_type="norm-linear",
                        mapping=MappingConfig(
                            source=None,
                            targets=[f"{prefix}.mlp.gate_proj", f"{prefix}.mlp.up_proj"],
                        ),
                    )
                )
        return configs


class MockInvalidAdapter:
    pass


def _make_default_qconfig():
    return LinearQConfig(
        act=QConfig(dtype=QDType.FLOAT, scope=QScope.PER_TENSOR, symmetric=True, method="none"),
        weight=QConfig(dtype=QDType.INT8, scope=QScope.PER_CHANNEL, symmetric=True, method="minmax"),
    )


def _make_processor(
    num_layers=2, n_grid=2, enable_subgraph_type=None, include_non_fusion=False
) -> tuple[AWQProcessor, MockModel, MagicMock]:
    model = MockModel(num_layers=num_layers)
    config = AWQProcessorConfig(
        weight_qconfig=_make_default_qconfig().weight,
        n_grid=n_grid,
        enable_subgraph_type=enable_subgraph_type or ["norm-linear", "up-down"],
    )
    adapter = MockAdapter(num_layers=num_layers, include_non_fusion=include_non_fusion)
    processor = AWQProcessor(model, config, adapter)

    mock_searcher = MagicMock()

    def _mock_search(linears, *_args, **_kwargs):
        scale_dim = linears[0].in_features
        return torch.ones(scale_dim)

    mock_searcher.search.side_effect = _mock_search
    processor.awq_config.awq_searcher = mock_searcher

    return processor, model, mock_searcher


class TestAWQProcessorConfig:
    def test_default_config(self):
        config = AWQProcessorConfig(weight_qconfig=_make_default_qconfig().weight)
        assert config.n_grid == 20
        assert set(config.enable_subgraph_type) == {"norm-linear", "linear-linear", "ov", "up-down"}
        assert config.include is None
        assert config.exclude is None

    def test_invalid_n_grid_rejected(self):
        with pytest.raises(Exception):
            AWQProcessorConfig(weight_qconfig=_make_default_qconfig().weight, n_grid=0)

    def test_invalid_subgraph_type_rejected(self):
        processor, _, _ = _make_processor(enable_subgraph_type=["norm-linear", "banana"])
        with pytest.raises(Exception):
            processor.pre_run()


class TestAWQProcessorInit:
    def test_init_with_valid_adapter(self):
        processor, _, _ = _make_processor()
        assert isinstance(processor, AWQProcessor)
        assert processor.stats_collector is not None
        assert processor.awq_config is not None

    def test_init_with_invalid_adapter_raises(self):
        model = MockModel()
        config = AWQProcessorConfig(weight_qconfig=_make_default_qconfig().weight)
        with pytest.raises(UnsupportedError):
            AWQProcessor(model, config, MockInvalidAdapter())


class TestAWQProcessorPreRun:
    def test_pre_run_populates_global_adapter_config(self):
        processor, _, _ = _make_processor()
        assert processor.global_adapter_config is None
        processor.pre_run()
        assert processor.global_adapter_config is not None
        assert len(processor.global_adapter_config) > 0


class TestAWQProcessorPreprocess:
    def test_preprocess_filters_by_scope(self):
        processor, model, _ = _make_processor(num_layers=2)
        processor.pre_run()
        request = BatchProcessRequest(name="model.layers.0", module=model.model.layers[0])
        processor.preprocess(request)
        # Only layer 0 configs should remain
        adapter_configs = processor.adapter_config
        assert adapter_configs is not None
        for ac in adapter_configs:
            assert "model.layers.0" in ac.mapping.targets[0]

    def test_preprocess_filters_by_subgraph_type(self):
        processor, model, _ = _make_processor(enable_subgraph_type=["up-down"])
        processor.pre_run()
        request = BatchProcessRequest(name="model.layers.0", module=model.model.layers[0])
        processor.preprocess(request)
        adapter_configs = processor.adapter_config
        assert adapter_configs is not None
        for ac in adapter_configs:
            assert ac.subgraph_type == "up-down"

    def test_preprocess_installs_hooks(self):
        processor, model, _ = _make_processor()
        processor.pre_run()
        request = BatchProcessRequest(name="model.layers.0", module=model.model.layers[0])
        processor.preprocess(request)
        # Stats collector should have hooks installed
        assert len(processor.stats_collector.hook_manager.hook_handles) > 0


class TestAWQProcessorPostprocess:
    def test_postprocess_clears_state(self):
        processor, model, mock_searcher = _make_processor()
        processor.pre_run()
        request = BatchProcessRequest(name="model.layers.0", module=model.model.layers[0])
        processor.preprocess(request)

        # Feed one sample through to populate stats
        x = torch.randn(1, 8, 64)
        model.model.layers[0](x)

        processor.postprocess(request)
        assert mock_searcher.search.call_count == 2
        assert processor.adapter_config is None
        assert len(processor.stats_collector.hook_manager.hook_handles) == 0

    def test_two_layers_no_state_leak(self):
        processor, model, mock_searcher = _make_processor(num_layers=2, n_grid=2)
        processor.pre_run()

        for i in range(2):
            layer = model.model.layers[i]
            request = BatchProcessRequest(name=f"model.layers.{i}", module=layer)
            processor.preprocess(request)
            x = torch.randn(1, 8, 64)
            layer(x)
            processor.postprocess(request)

        assert mock_searcher.search.call_count == 4
        # After processing both layers, no residual state
        assert processor.adapter_config is None
        assert len(processor.stats_collector.hook_manager.hook_handles) == 0


class TestAWQProcessorNonFusion:
    def test_non_fusion_subgraph_calls_search(self):
        processor, model, mock_searcher = _make_processor(
            num_layers=1,
            n_grid=2,
            include_non_fusion=True,
            enable_subgraph_type=["norm-linear", "up-down"],
        )
        processor.pre_run()
        layer = model.model.layers[0]
        request = BatchProcessRequest(name="model.layers.0", module=layer)
        processor.preprocess(request)
        x = torch.randn(1, 8, 64)
        layer(x)
        processor.postprocess(request)
        # 2 fusion subgraphs (norm-linear + up-down) + 1 non-fusion = 3 calls
        assert mock_searcher.search.call_count == 3
