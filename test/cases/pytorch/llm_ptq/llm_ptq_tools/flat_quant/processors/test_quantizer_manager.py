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
# pylint: disable=attribute-defined-outside-init

import pytest  # noqa: F401
import re
from unittest.mock import Mock
from collections import OrderedDict

from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.flat_quant.processors.quantizer_manager import (
    QuantizerMapper,
    match_pattern,
)
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.flat_quant.processors.flat_quant import (
    FakeQuantizerVisitor,
    FlatQuantQuantizerMapVisitor,
)


class TestMatchPattern:
    def test_match_pattern_string_prefix(self):
        result = match_pattern("layer1.attention.norm", "layer1")
        assert result is True

    def test_match_pattern_string_no_match(self):
        result = match_pattern("layer2.attention.norm", "layer1")
        assert result is False

    def test_match_pattern_regex_match(self):
        pattern = re.compile(r"layer\d+\.attention")
        result = match_pattern("layer1.attention.norm", pattern)
        assert result is not None

    def test_match_pattern_regex_no_match(self):
        pattern = re.compile(r"layer\d+\.mlp")
        result = match_pattern("layer1.attention.norm", pattern)
        assert result is None

    def test_match_pattern_empty_pattern(self):
        result = match_pattern("layer1.attention", "")
        assert result is True


class TestQuantizerMapper:
    def setup_method(self):
        self.mapper = QuantizerMapper()

    def test_initialization(self):
        assert isinstance(self.mapper.pattern_map, OrderedDict)
        assert isinstance(self.mapper.pair_quantizer_map, OrderedDict)
        assert len(self.mapper.pattern_map) == 0
        assert len(self.mapper.pair_quantizer_map) == 0

    def test_register_pattern_chaining(self):
        visitor = Mock(spec=FakeQuantizerVisitor)

        result = self.mapper.register_pattern("layer1", visitor)

        assert result == self.mapper
        assert "layer1" in self.mapper.pattern_map
        assert self.mapper.pattern_map["layer1"] == visitor

    def test_register_pattern_regex(self):
        visitor = Mock(spec=FlatQuantQuantizerMapVisitor)
        pattern = re.compile(r"layer\d+")

        self.mapper.register_pattern(pattern, visitor)

        assert pattern in self.mapper.pattern_map
        assert self.mapper.pattern_map[pattern] == visitor

    def test_apply_quantizer_string_pattern(self):
        visitor = Mock(spec=FakeQuantizerVisitor)
        pair = Mock()
        pair.__str__ = Mock(return_value="layer1.attention.norm")

        pairs_dict = {"AttnNormLinearPair": [pair], "MLPLinearLinearPair": []}

        self.mapper.register_pattern("layer1", visitor)
        self.mapper.apply_quantizer(pairs_dict)

        assert "layer1.attention.norm" in self.mapper.pair_quantizer_map
        assert self.mapper.pair_quantizer_map["layer1.attention.norm"] == visitor
        pair.accept.assert_called_once_with(visitor)

    def test_apply_quantizer_regex_pattern(self):
        visitor = Mock(spec=FlatQuantQuantizerMapVisitor)
        pair = Mock()
        pair.__str__ = Mock(return_value="layer1.attention.norm")

        pairs_dict = {"AttnNormLinearPair": [pair]}

        pattern = re.compile(r"layer\d+\.attention")
        self.mapper.register_pattern(pattern, visitor)
        self.mapper.apply_quantizer(pairs_dict)

        assert "layer1.attention.norm" in self.mapper.pair_quantizer_map
        pair.accept.assert_called_once_with(visitor)

    def test_apply_quantizer_no_match(self):
        visitor = Mock(spec=FakeQuantizerVisitor)
        pair = Mock()
        pair.__str__ = Mock(return_value="layer2.attention.norm")

        pairs_dict = {"AttnNormLinearPair": [pair]}

        self.mapper.register_pattern("layer1", visitor)
        self.mapper.apply_quantizer(pairs_dict)

        assert len(self.mapper.pair_quantizer_map) == 0
        pair.accept.assert_not_called()

    def test_apply_quantizer_clears_previous_mapping(self):
        visitor = Mock(spec=FakeQuantizerVisitor)
        pair = Mock()
        pair.__str__ = Mock(return_value="layer1.attention.norm")

        pairs_dict1 = {"AttnNormLinearPair": [pair]}
        self.mapper.register_pattern("layer1", visitor)
        self.mapper.apply_quantizer(pairs_dict1)

        assert len(self.mapper.pair_quantizer_map) == 1

        pairs_dict2 = {"MLPLinearLinearPair": []}
        self.mapper.apply_quantizer(pairs_dict2)

        assert len(self.mapper.pair_quantizer_map) == 0

    def test_get_quantizer_for_pair(self):
        visitor = Mock(spec=FakeQuantizerVisitor)
        pair = Mock()
        pair.__str__ = Mock(return_value="layer1.attention.norm")

        self.mapper.pair_quantizer_map["layer1.attention.norm"] = visitor

        result_with_pair = self.mapper.get_quantizer_for_pair(pair)
        result_with_string = self.mapper.get_quantizer_for_pair("layer1.attention.norm")
        result_not_found = self.mapper.get_quantizer_for_pair("nonexistent.pair")

        assert result_with_pair == visitor
        assert result_with_string == visitor
        assert result_not_found is None

    def test_to_org_mode(self):
        visitor1 = Mock(spec=FakeQuantizerVisitor)
        visitor2 = Mock(spec=FlatQuantQuantizerMapVisitor)

        self.mapper.pair_quantizer_map["pair1"] = visitor1
        self.mapper.pair_quantizer_map["pair2"] = visitor2

        self.mapper.to_org_mode()

        visitor1.to_org_mode.assert_called_once_with("")
        visitor2.to_org_mode.assert_called_once_with("")

    def test_to_org_mode_with_prefix(self):
        visitor = Mock(spec=FakeQuantizerVisitor)

        self.mapper.pair_quantizer_map["pair1"] = visitor

        self.mapper.to_org_mode(prefix="layer1")

        visitor.to_org_mode.assert_called_once_with("layer1")

    def test_to_org_mode_unique_visitors(self):
        visitor = Mock(spec=FakeQuantizerVisitor)

        self.mapper.pair_quantizer_map["pair1"] = visitor
        self.mapper.pair_quantizer_map["pair2"] = visitor

        self.mapper.to_org_mode()

        visitor.to_org_mode.assert_called_once_with("")

    def test_to_calib_mode(self):
        visitor = Mock(spec=FlatQuantQuantizerMapVisitor)

        self.mapper.pair_quantizer_map["pair1"] = visitor

        self.mapper.to_calib_mode(prefix="layer1")

        visitor.to_calib_mode.assert_called_once_with("layer1")

    def test_to_calib_mode_no_method(self):
        visitor = Mock()
        if hasattr(visitor, 'to_calib_mode'):
            delattr(visitor, 'to_calib_mode')

        self.mapper.pair_quantizer_map["pair1"] = visitor

        self.mapper.to_calib_mode()

    def test_to_eval_mode(self):
        visitor = Mock(spec=FlatQuantQuantizerMapVisitor)

        self.mapper.pair_quantizer_map["pair1"] = visitor

        self.mapper.to_eval_mode(prefix="layer1", quant_weight=False)

        visitor.to_eval_mode.assert_called_once_with(prefix="layer1", quant_weight=False)

    def test_to_eval_mode_no_method(self):
        visitor = Mock()
        if hasattr(visitor, 'to_eval_mode'):
            delattr(visitor, 'to_eval_mode')

        self.mapper.pair_quantizer_map["pair1"] = visitor

        self.mapper.to_eval_mode()

    def test_empty_inputs(self):
        visitor = Mock(spec=FakeQuantizerVisitor)
        pair = Mock()
        pair.__str__ = Mock(return_value="layer1.attention.norm")

        # Test empty pairs dict
        self.mapper.register_pattern("layer1", visitor)
        self.mapper.apply_quantizer({})
        assert len(self.mapper.pair_quantizer_map) == 0

        # Test empty pattern map
        pairs_dict = {"AttnNormLinearPair": [pair]}
        self.mapper.pattern_map.clear()
        self.mapper.apply_quantizer(pairs_dict)
        assert len(self.mapper.pair_quantizer_map) == 0
        pair.accept.assert_not_called()
