#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
-------------------------------------------------------------------------
This file is part of the MindStudio project.
Copyright (c) 2025 Huawei Technologies Co.,Ltd.

MindStudio is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:

         http://license.coscl.org.cn/MulanPSL2

THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details.
-------------------------------------------------------------------------

Unit tests for msmodelslim.format.compressed_tensors_format.quantization.quant_config_builder.
"""

from __future__ import annotations

import re

from torch import nn

from msmodelslim.format.compressed_tensors_format.config.base import QuantizationFormat
from msmodelslim.format.compressed_tensors_format.quantization.quant_args import (
    QuantizationStrategy,
)
from msmodelslim.format.compressed_tensors_format.quantization.quant_config_builder import (
    _compress_module_names_to_regex,
    _resolve_weight_observer,
    _scheme_key,
    apply_runtime_overrides,
    build_scheme_from_module,
    infer_config_groups,
    infer_ignore,
    infer_root_format,
    infer_targets,
)
from msmodelslim.format.compressed_tensors_format.quantization.quant_scheme import (
    preset_name_to_scheme,
)

from test.cases.format.compressed_tensors_format.helpers import (
    MixedQuantFloatModel,
    QuantizedModel,
    make_w8a8_static_module,
)


class TestInferTargets:
    """Tests for infer_targets."""

    def test_infer_targets_return_linear_when_model_has_qir_module(self):
        assert infer_targets(QuantizedModel()) == ["Linear"]

    def test_infer_targets_return_default_linear_when_model_has_no_qir_module(self):
        assert infer_targets(nn.Linear(4, 2)) == ["Linear"]


class TestInferConfigGroups:
    """Tests for infer_config_groups."""

    def test_infer_config_groups_return_one_group_when_single_scheme(self):
        groups = infer_config_groups(QuantizedModel())

        assert len(groups) == 1
        assert list(groups.keys()) == ["group_0"]

    def test_infer_config_groups_return_empty_when_no_qir_module(self):
        assert infer_config_groups(nn.Sequential(nn.Linear(4, 2))) == {}


class TestBuildSchemeFromModule:
    """Tests for build_scheme_from_module."""

    def test_build_scheme_from_module_return_scheme_when_qir_module_supported(self):
        module = make_w8a8_static_module()

        scheme = build_scheme_from_module(module, ["Linear"])

        assert scheme is not None
        assert scheme.weights.strategy == QuantizationStrategy.CHANNEL

    def test_build_scheme_from_module_return_none_when_module_unsupported(self):
        assert build_scheme_from_module(nn.Linear(4, 2), ["Linear"]) is None


class TestApplyRuntimeOverrides:
    """Tests for apply_runtime_overrides."""

    def test_apply_runtime_overrides_set_observer_when_static_weight_scheme(self):
        base_scheme = preset_name_to_scheme("W8A8_STATIC", ["Linear"])
        module = make_w8a8_static_module()

        scheme = apply_runtime_overrides(base_scheme, module)

        assert scheme.weights.observer == "minmax"

    def test_apply_runtime_overrides_clear_observer_when_dynamic_activation(self):
        base_scheme = preset_name_to_scheme("W8A8_DYNAMIC", ["Linear"])
        module = make_w8a8_static_module()

        scheme = apply_runtime_overrides(base_scheme, module)

        assert scheme.input_activations.observer is None


class TestInferRootFormat:
    """Tests for infer_root_format."""

    def test_infer_root_format_return_int_quantized_when_single_group(self):
        groups = infer_config_groups(QuantizedModel())

        assert infer_root_format(groups) == QuantizationFormat.int_quantized.value

    def test_infer_root_format_return_mixed_precision_when_formats_differ(self):
        scheme_a = preset_name_to_scheme("W8A8_STATIC", ["Linear"])
        scheme_b = preset_name_to_scheme("W8A8_STATIC", ["Linear"]).model_copy(
            update={"format": QuantizationFormat.float_quantized}
        )
        groups = {"group_0": scheme_a, "group_1": scheme_b}

        assert infer_root_format(groups) == QuantizationFormat.mixed_precision.value

    def test_infer_root_format_return_default_when_groups_empty(self):
        assert infer_root_format({}) == QuantizationFormat.int_quantized.value


class TestInferIgnore:
    """Tests for infer_ignore."""

    def test_infer_ignore_return_regex_patterns_when_float_linear_coexists(self):
        patterns = infer_ignore(MixedQuantFloatModel())

        assert len(patterns) == 1
        assert patterns[0].startswith("re:")
        assert re.search(patterns[0][3:], "float_linear") is not None

    def test_infer_ignore_return_empty_when_only_quantized_layers(self):
        assert not infer_ignore(QuantizedModel())


class TestResolveWeightObserver:
    """Tests for _resolve_weight_observer."""

    def test_resolve_weight_observer_return_minmax_when_module_has_no_w_scheme(self):
        module = make_w8a8_static_module()

        assert _resolve_weight_observer(module) == "minmax"


class TestSchemeKey:
    """Tests for _scheme_key."""

    def test_scheme_key_return_same_json_when_schemes_identical(self):
        scheme = preset_name_to_scheme("W8A8_STATIC", ["Linear"])

        assert _scheme_key(scheme) == _scheme_key(scheme.model_copy())


class TestCompressModuleNamesToRegex:
    """Tests for _compress_module_names_to_regex."""

    def test_compress_module_names_to_regex_return_empty_when_input_empty(self):
        assert not _compress_module_names_to_regex([])

    def test_compress_module_names_to_regex_merge_numeric_layers_when_shared_prefix(
        self,
    ):
        patterns = _compress_module_names_to_regex(["model.layers.0.linear", "model.layers.1.linear"])

        assert len(patterns) == 1
        body = patterns[0][3:]
        assert re.search(body, "model.layers.0.linear") is not None
        assert re.search(body, "model.layers.1.linear") is not None

    def test_compress_module_names_to_regex_skip_blank_names_when_input_contains_empty(
        self,
    ):
        patterns = _compress_module_names_to_regex(["", "  ", "valid.layer"])

        assert len(patterns) == 1
        assert re.search(patterns[0][3:], "valid.layer") is not None
