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
# pylint: disable=attribute-defined-outside-init,consider-using-from-import

import pytest
import torch
import torch.nn as nn

from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.layer_select import LayerSelector


class SimpleLLM(nn.Module):
    """A simple model with linear layers for testing."""

    def __init__(self, input_dim=8, hidden_dim=8, output_dim=8):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.fc2(x)
        return x


class TestLayerSelector:
    def setup_method(self):
        self.model = SimpleLLM()
        self.calib_data = [[torch.randn(1, 8)], [torch.randn(2, 8)]]

    def test_layer_selector_init(self):
        # Test initialization with default parameters
        selector = LayerSelector(self.model, range_method="std")
        assert selector.model == self.model
        assert len(selector.layer_names) == 2

        # Test initialization with specific layer names
        selector = LayerSelector(self.model, layer_names=['fc1'], range_method="std")
        assert len(selector.layer_names) == 1
        assert 'fc1' in selector.layer_names

        with pytest.raises(TypeError):
            LayerSelector("not a model", range_method="std")

        with pytest.raises(ValueError):
            LayerSelector(self.model, layer_names=['invalid_layer'], range_method="std")

        with pytest.raises(ValueError):
            LayerSelector(self.model, range_method="invalid_method")

    def test_layer_selector_run(self):
        selector = LayerSelector(self.model, range_method="std")
        selector.run(self.calib_data)

        assert len(selector.layer_scores) == 2  # One for each layer

        with pytest.raises(TypeError):
            selector.run("not a list")

    def test_select_layers_by_threshold(self):
        selector = LayerSelector(self.model, range_method="std")
        selector.run(self.calib_data)

        selector.layer_scores = [{'name': 'fc1', 'score': 5.0}, {'name': 'fc2', 'score': 3.0}]

        # Test with threshold that selects all layers
        layers = selector.select_layers_by_threshold(0.5)
        assert len(layers) == 2
        assert 'fc1' in layers
        assert 'fc2' in layers

        # Test with threshold that selects only some layers
        layers = selector.select_layers_by_threshold(4.0)
        assert len(layers) == 1
        assert 'fc1' in layers
        assert 'fc2' not in layers

        # Test with threshold that selects no layers
        layers = selector.select_layers_by_threshold(10.0)
        assert len(layers) == 0

        with pytest.raises(ValueError):
            selector.select_layers_by_threshold(-1.0)

    def test_select_layers_by_disable_level(self):
        selector = LayerSelector(self.model, range_method="std")
        selector.run(self.calib_data)

        selector.layer_groups = [[{'name': 'fc1', 'score': 5.0}], [{'name': 'fc2', 'score': 3.0}]]

        # Test with level that selects only the first group
        layers = selector.select_layers_by_disable_level(1)
        assert len(layers) == 1
        assert 'fc1' in layers

        # Test with level that selects both groups
        layers = selector.select_layers_by_disable_level(2)
        assert len(layers) == 2
        assert 'fc1' in layers
        assert 'fc2' in layers

        # Test with level that exceeds the number of groups
        layers = selector.select_layers_by_disable_level(5)
        assert len(layers) == 2

        # Test with level 0 (should select no layers)
        layers = selector.select_layers_by_disable_level(0)
        assert len(layers) == 0

        with pytest.raises(ValueError):
            selector.select_layers_by_disable_level(-1)
        with pytest.raises(TypeError):
            selector.select_layers_by_disable_level("not an int")

    def test_layers_with_same_score(self):
        selector = LayerSelector(self.model, range_method="std")
        selector.run(self.calib_data)

        selector.layer_groups = [[{'name': 'fc1', 'score': 5.0}, {'name': 'fc2', 'score': 5.0}]]

        # Test that both layers with the same score are selected together
        layers = selector.select_layers_by_disable_level(1)
        assert len(layers) == 2
        assert 'fc1' in layers
        assert 'fc2' in layers

    def test_layer_selector_quantile(self):
        """Test the layer selector with quantile method."""
        # Test initialization with quantile method
        selector = LayerSelector(self.model, range_method="quantile")
        selector.run(self.calib_data)

        # Verify scores were computed
        assert len(selector.layer_scores) == 2
