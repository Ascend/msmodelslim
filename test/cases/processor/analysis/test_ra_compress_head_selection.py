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

Public-interface behavior tests for the RA compression analysis method.
"""
# pylint: disable=protected-access

import unittest
from types import SimpleNamespace
from typing import Dict
from unittest.mock import patch

import torch
from torch import nn

from msmodelslim.processor.analysis.unary_operator.metrics.ra_compress import (
    PREFIX_TOKEN_ID,
    RaCompressAnalysisMethod,
    resolve_prefix_token_id,
    resolve_v0_prefix_token_id,
)
from msmodelslim.processor.analysis.unary_operator.metrics.ra_compress.interface import (
    RaCompressAnalysisInterface,
)

_IMPL_MODULE = "msmodelslim.processor.analysis.unary_operator.metrics.ra_compress.impl"
_DUMMY_INPUT_LENGTH = 2
_REPET_TIMES = 2
_TOKENS = _DUMMY_INPUT_LENGTH * _REPET_TIMES


class FakeAdapter(RaCompressAnalysisInterface):
    """Minimal public adapter used to exercise the method contract."""

    def get_proj_names(self) -> Dict[str, str]:
        return {"q": "q_proj", "k": "k_proj", "qkv": "qkv_proj"}

    def get_tokenizer(self):
        return None


class _PatternAdapter(RaCompressAnalysisInterface):
    def __init__(self, patterns: Dict[str, str]):
        self.patterns = patterns

    def get_proj_names(self) -> Dict[str, str]:
        return self.patterns

    def get_tokenizer(self):
        return None


class _FakeTokenizer:
    def __init__(self, mapping: Dict[str, object]):
        self.mapping = mapping

    def __call__(self, text: str, **kwargs) -> Dict[str, object]:
        return {"input_ids": torch.tensor([self.mapping[text]])}


class _FakeAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(4, 8, bias=False)
        self.k_proj = nn.Linear(4, 8, bias=False)

    def forward(self, x):
        return self.q_proj(x)


class _FakeLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = _FakeAttention()


class _FakeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(
            num_attention_heads=2,
            num_key_value_heads=2,
            hidden_size=8,
            head_dim=4,
        )
        self.layers = nn.ModuleList([_FakeLayer()])


class _FakeQkvAttention(_FakeAttention):
    def __init__(self):
        super().__init__()
        self.qkv_proj = nn.Linear(4, 16, bias=False)


class _FakeQkvModel(_FakeModel):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([_FakeLayer()])
        self.layers[0].self_attn = _FakeQkvAttention()


def _record_output(method, layer_name: str, output) -> Dict[str, object]:
    stats = {}
    method.get_hook()(None, None, output, layer_name, stats)
    return stats


def _make_method(induction_ratio: float = 0.5, echo_ratio: float = 0.5) -> RaCompressAnalysisMethod:
    return RaCompressAnalysisMethod(
        adapter=FakeAdapter(),
        induction_head_ratio=induction_ratio,
        echo_head_ratio=echo_ratio,
    )


def _uniform_segment_score(num_tokens: int, dummy_length: int, target_offset: int) -> float:
    """Independent expected value for uniform causal attention."""
    matrix_tokens = num_tokens - 1
    per_token_scores = [0.0] * matrix_tokens
    counted = [False] * matrix_tokens
    for step in range(1, matrix_tokens // dummy_length + 1):
        offset = step * dummy_length
        for row in range(offset, matrix_tokens):
            # The truncated matrix row ``row`` corresponds to causal row ``row + 1``.
            per_token_scores[row] += 1.0 / (row + 2)
            counted[row] = True
    active_scores = [value for value, active in zip(per_token_scores, counted) if active]
    return sum(active_scores) / len(active_scores) if active_scores else 0.0


class TestResolveV0PrefixTokenId(unittest.TestCase):
    def test_resolve_v0_prefix_token_id_returns_bos_when_empty_input_nonempty(self):
        tokenizer = _FakeTokenizer({"": [128000], "A": [128000, 32]})
        self.assertEqual(resolve_v0_prefix_token_id(tokenizer), 128000)

    def test_resolve_v0_prefix_token_id_falls_back_to_text_when_empty_input_empty(self):
        tokenizer = _FakeTokenizer({"": [], "A": [32]})
        self.assertEqual(resolve_v0_prefix_token_id(tokenizer), 32)

    def test_resolve_v0_prefix_token_id_returns_none_when_tokenizer_missing(self):
        self.assertIsNone(resolve_v0_prefix_token_id(None))

    def test_resolve_v0_prefix_token_id_returns_none_when_tokenizer_raises(self):
        def raising_tokenizer(text, **kwargs):
            raise RuntimeError("tokenizer unavailable")

        self.assertIsNone(resolve_v0_prefix_token_id(raising_tokenizer))


class TestResolvePrefixTokenId(unittest.TestCase):
    def test_resolve_prefix_token_id_uses_v0_token_when_adapter_has_tokenizer(self):
        adapter = _PatternAdapter({"q": "q_proj", "k": "k_proj", "qkv": "qkv_proj"})
        adapter.get_tokenizer = lambda: _FakeTokenizer({"": [], "A": [32]})
        self.assertEqual(resolve_prefix_token_id(adapter), 32)

    def test_resolve_prefix_token_id_falls_back_when_interface_missing(self):
        self.assertEqual(resolve_prefix_token_id(object()), PREFIX_TOKEN_ID)

    def test_resolve_prefix_token_id_falls_back_when_tokenizer_call_raises(self):
        class BadTokenizer:
            def __call__(self, text, **kwargs):
                raise RuntimeError("bad tokenizer")

        class BadTokenizerAdapter(_PatternAdapter):
            def get_tokenizer(self):
                return BadTokenizer()

        adapter = BadTokenizerAdapter({"q": "q_proj", "k": "k_proj", "qkv": "qkv_proj"})
        self.assertEqual(resolve_prefix_token_id(adapter), PREFIX_TOKEN_ID)


class TestRaCompressTargetLayers(unittest.TestCase):
    def test_get_target_layers_returns_qk_linear_names_when_standard_model(self):
        method = _make_method()
        self.assertEqual(
            method.get_target_layers(_FakeModel()),
            ["layers.0.self_attn.q_proj", "layers.0.self_attn.k_proj"],
        )

    def test_get_target_layers_returns_qkv_name_when_fused_model(self):
        method = _make_method()
        self.assertEqual(
            method.get_target_layers(_FakeQkvModel()),
            [
                "layers.0.self_attn.q_proj",
                "layers.0.self_attn.k_proj",
                "layers.0.self_attn.qkv_proj",
            ],
        )

    def test_get_target_layers_returns_empty_when_all_patterns_empty(self):
        method = RaCompressAnalysisMethod(adapter=_PatternAdapter({"q": "", "k": "", "qkv": ""}))
        self.assertEqual(method.get_target_layers(_FakeModel()), [])


class TestRaCompressHookContract(unittest.TestCase):
    def test_get_hook_records_q_output_when_q_proj_layer(self):
        method = _make_method()
        output = torch.randn(_TOKENS, 8)
        stats = _record_output(method, "layers.0.self_attn.q_proj", output)
        self.assertEqual(len(stats["layers.0.self_attn.q_proj"]["outputs"]), 1)
        self.assertTrue(torch.equal(stats["layers.0.self_attn.q_proj"]["outputs"][0], output))

    def test_get_hook_records_tuple_first_element_when_output_is_tuple(self):
        method = _make_method()
        first = torch.randn(_TOKENS, 8)
        second = torch.randn(_TOKENS, 8)
        stats = _record_output(method, "layers.0.self_attn.q_proj", (first, second))
        self.assertTrue(torch.equal(stats["layers.0.self_attn.q_proj"]["outputs"][0], first))

    def test_get_hook_records_qkv_output_when_fused_layer(self):
        method = _make_method()
        output = torch.randn(_TOKENS, 16)
        stats = _record_output(method, "layers.0.self_attn.qkv_proj", output)
        self.assertEqual(len(stats["layers.0.self_attn.qkv_proj"]["outputs"]), 1)
        self.assertEqual(stats["layers.0.self_attn.qkv_proj"]["layer_name"], "layers.0.self_attn.qkv_proj")


class TestRaCompressScoreContract(unittest.TestCase):
    def _compute_separate(self, tokens: int = _TOKENS):
        method = _make_method(induction_ratio=1.0, echo_ratio=1.0)
        model = _FakeModel()
        method.get_target_layers(model)
        q_output = torch.zeros(tokens, 8)
        k_output = torch.zeros(tokens, 8)
        q_stats = _record_output(method, "layers.0.self_attn.q_proj", q_output)
        _record_output(method, "layers.0.self_attn.k_proj", k_output)
        score = method.compute_score(q_stats["layers.0.self_attn.q_proj"])
        return method, score

    def test_compute_score_returns_uniform_repeat_average_when_qk_separate(self):
        with (
            patch(f"{_IMPL_MODULE}.DUMMY_INPUT_LENGTH", _DUMMY_INPUT_LENGTH),
            patch(f"{_IMPL_MODULE}.REPET_TIMES", _REPET_TIMES),
        ):
            _, score = self._compute_separate()
        expected_prefix = _uniform_segment_score(_TOKENS, _DUMMY_INPUT_LENGTH, 1)
        expected_copying = _uniform_segment_score(_TOKENS, _DUMMY_INPUT_LENGTH, 0)
        self.assertAlmostEqual(score, expected_prefix + expected_copying, places=6)

    def test_compute_score_returns_zero_when_q_has_no_k_output(self):
        with (
            patch(f"{_IMPL_MODULE}.DUMMY_INPUT_LENGTH", _DUMMY_INPUT_LENGTH),
            patch(f"{_IMPL_MODULE}.REPET_TIMES", _REPET_TIMES),
        ):
            method = _make_method()
            method.get_target_layers(_FakeModel())
            q_stats = _record_output(method, "layers.0.self_attn.q_proj", torch.zeros(_TOKENS, 8))
            self.assertEqual(method.compute_score(q_stats["layers.0.self_attn.q_proj"]), 0.0)

    def test_compute_score_returns_zero_when_tokens_insufficient(self):
        with (
            patch(f"{_IMPL_MODULE}.DUMMY_INPUT_LENGTH", _DUMMY_INPUT_LENGTH),
            patch(f"{_IMPL_MODULE}.REPET_TIMES", _REPET_TIMES),
        ):
            method = _make_method()
            method.get_target_layers(_FakeModel())
            q_stats = _record_output(method, "layers.0.self_attn.q_proj", torch.zeros(_TOKENS - 1, 8))
            _record_output(method, "layers.0.self_attn.k_proj", torch.zeros(_TOKENS - 1, 8))
            self.assertEqual(method.compute_score(q_stats["layers.0.self_attn.q_proj"]), 0.0)

    def test_compute_score_returns_zero_when_layer_name_is_not_target(self):
        method = _make_method()
        self.assertEqual(
            method.compute_score({"layer_name": "layers.0.mlp.gate_proj", "outputs": [torch.zeros(_TOKENS, 8)]}),
            0.0,
        )


class TestRaCompressHeadSelectionContract(unittest.TestCase):
    def test_get_compress_heads_returns_empty_maps_when_no_scores(self):
        self.assertEqual(
            _make_method().get_compress_heads(),
            {"prefix_matching": {}, "copying": {}},
        )

    def test_get_compress_heads_returns_all_heads_when_ratios_are_one(self):
        with (
            patch(f"{_IMPL_MODULE}.DUMMY_INPUT_LENGTH", _DUMMY_INPUT_LENGTH),
            patch(f"{_IMPL_MODULE}.REPET_TIMES", _REPET_TIMES),
        ):
            method = _make_method(induction_ratio=1.0, echo_ratio=1.0)
            model = _FakeModel()
            method.get_target_layers(model)
            for layer_name, output in (
                ("layers.0.self_attn.q_proj", torch.zeros(_TOKENS, 8)),
                ("layers.0.self_attn.k_proj", torch.zeros(_TOKENS, 8)),
            ):
                _record_output(method, layer_name, output)
            method.compute_score(
                _record_output(method, "layers.0.self_attn.q_proj", torch.zeros(_TOKENS, 8))[
                    "layers.0.self_attn.q_proj"
                ]
            )
            self.assertEqual(
                method.get_compress_heads(),
                {"prefix_matching": {0: [0, 1]}, "copying": {0: [0, 1]}},
            )

    def test_enrich_layer_scores_attaches_empty_heads_when_no_scores(self):
        method = _make_method()
        layer_scores = [{"name": "layers.0.self_attn.q_proj", "score": 0.0}]
        method.enrich_layer_scores(layer_scores)
        self.assertEqual(layer_scores[0]["induction_heads"], [])
        self.assertEqual(layer_scores[0]["echo_heads"], [])


class TestRaCompressEmptyPatternSemantics(unittest.TestCase):
    def test_get_target_layers_and_compute_score_ignore_empty_patterns(self):
        method = RaCompressAnalysisMethod(adapter=_PatternAdapter({"q": "", "k": "", "qkv": ""}))
        self.assertEqual(method.get_target_layers(_FakeModel()), [])
        self.assertEqual(
            method.compute_score({"layer_name": "layers.0.self_attn.q_proj", "outputs": [torch.zeros(4, 8)]}),
            0.0,
        )

    def test_constructor_logs_warning_when_all_patterns_empty(self):
        with patch(f"{_IMPL_MODULE}.logger") as mock_logger:
            RaCompressAnalysisMethod(adapter=_PatternAdapter({"q": "", "k": "", "qkv": ""}))
        self.assertTrue(mock_logger.warning.called)


if __name__ == "__main__":
    unittest.main()
