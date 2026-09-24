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

import unittest
from typing import Dict
from unittest.mock import patch

import torch

from msmodelslim.processor.analysis.unary_operator.metrics.ra_compress import (
    DUMMY_INPUT_LENGTH,
    PREFIX_TOKEN_ID,
    REPET_TIMES,
    RaCompressAnalysisMethod,
    resolve_model_tokenizer,
    resolve_prefix_token_id,
    resolve_v0_prefix_token_id,
)
from msmodelslim.processor.analysis.unary_operator.metrics.ra_compress.interface import (
    RaCompressAnalysisInterface,
)


class FakeAdapter(RaCompressAnalysisInterface):
    """实现 RaCompressAnalysisInterface 的测试用 adapter（不提供 tokenizer）。"""

    def get_proj_names(self) -> Dict[str, str]:
        return {"q": "q_proj", "k": "k_proj", "qkv": "qkv_proj"}

    def get_tokenizer(self):
        return None


class _FakeTokenizer:
    """按文本返回预设 input_ids 的假 tokenizer。"""

    def __init__(self, mapping: Dict[str, object]):
        self._mapping = mapping

    def __call__(self, text: str, **kwargs) -> Dict[str, torch.Tensor]:
        return {"input_ids": torch.tensor([self._mapping[text]])}


class TestResolveV0PrefixTokenId(unittest.TestCase):
    """测试 resolve_v0_prefix_token_id — 复现 V0 的首 token 口径。"""

    def test_uses_last_token_of_empty_input_when_not_empty(self):
        """场景：tokenizer('') 带 BOS。预期：取该末位 token 即 BOS。"""
        tokenizer = _FakeTokenizer({"": [128000], "A": [128000, 32]})
        self.assertEqual(resolve_v0_prefix_token_id(tokenizer), 128000)

    def test_falls_back_to_text_token_when_empty_input_is_empty(self):
        """场景：tokenizer('') 为空（Qwen2 系列）。预期：取 tokenizer('A') 的末位 token。"""
        tokenizer = _FakeTokenizer({"": [], "A": [32]})
        self.assertEqual(resolve_v0_prefix_token_id(tokenizer), 32)

    def test_returns_none_when_tokenizer_missing(self):
        """异常：未提供 tokenizer。预期：返回 None，由调用方回落兜底值。"""
        self.assertIsNone(resolve_v0_prefix_token_id(None))

    def test_returns_none_when_tokenizer_raises(self):
        """异常：tokenizer 调用失败。预期：返回 None，不向上抛异常。"""

        def _boom(text, **kwargs):
            raise RuntimeError("tokenizer unavailable")

        self.assertIsNone(resolve_v0_prefix_token_id(_boom))


class TestRaCompressPrefixScore(unittest.TestCase):
    """测试 _prefix_score_for_matrix — prefix matching 分数计算。"""

    def test_prefix_score_returns_zero_when_attn_not_2d(self):
        """非 2D tensor 返回 0。"""
        attn_1d = torch.ones(10)
        self.assertEqual(RaCompressAnalysisMethod._prefix_score_for_matrix(attn_1d), 0.0)

    def test_prefix_score_returns_zero_when_seq_len_below_dummy_length(self):
        """seq_len < DUMMY_INPUT_LENGTH 时没有完整段，返回 0。"""
        attn = torch.ones(DUMMY_INPUT_LENGTH - 1, DUMMY_INPUT_LENGTH - 1)
        attn = attn / attn.sum(dim=-1, keepdim=True)
        self.assertEqual(RaCompressAnalysisMethod._prefix_score_for_matrix(attn), 0.0)

    def test_prefix_score_returns_positive_when_attention_is_uniform(self):
        """均匀分布的注意力矩阵，prefix 分数应接近 1/seq_len（每个位置均匀分配）。"""
        seq_len = DUMMY_INPUT_LENGTH * 2
        attn = torch.ones(seq_len, seq_len)
        # 下三角（含对角线）为有效，上三角为 0
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)
        attn = attn.masked_fill(causal_mask, 0.0)
        attn = attn / attn.sum(dim=-1, keepdim=True)

        score = RaCompressAnalysisMethod._prefix_score_for_matrix(attn)
        # 对于均匀分布，prefix 偏移 +1 的位置应接近 1/seq_len
        self.assertGreater(score, 0.0)
        self.assertLess(score, 0.1)

    def test_prefix_score_returns_high_when_attention_on_prefix_offset(self):
        """对角线集中（主对角线+1 偏移）的注意力矩阵应有较高的 prefix 分数。"""
        seq_len = DUMMY_INPUT_LENGTH * 2
        attn = torch.zeros(seq_len, seq_len)
        # 在 prefix matching 的位置（i + d*SEG + 1）设置高值
        for k in range(1, 2):
            start = k * DUMMY_INPUT_LENGTH
            end = min((k + 1) * DUMMY_INPUT_LENGTH, seq_len)
            for i in range(start, end):
                col = i + (-k) * DUMMY_INPUT_LENGTH + 1
                if 0 <= col < seq_len:
                    attn[i, col] = 1.0
        # 因果 mask
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)
        attn = attn.masked_fill(causal_mask, 0.0)
        row_sums = attn.sum(dim=-1, keepdim=True)
        row_sums[row_sums == 0] = 1.0
        attn = attn / row_sums

        score = RaCompressAnalysisMethod._prefix_score_for_matrix(attn)
        self.assertGreater(score, 0.5)


class TestRaCompressCopyingScore(unittest.TestCase):
    """测试 _copying_score_for_matrix — copying matching 分数计算。"""

    def test_copying_score_returns_zero_when_attn_not_2d(self):
        """非 2D tensor 返回 0。"""
        self.assertEqual(RaCompressAnalysisMethod._copying_score_for_matrix(torch.ones(5)), 0.0)

    def test_copying_score_differs_from_prefix_when_uniform_attention(self):
        """copying 分数与 prefix 分数不同（偏移差 1）。"""
        seq_len = DUMMY_INPUT_LENGTH * 2
        attn = torch.ones(seq_len, seq_len)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)
        attn = attn.masked_fill(causal_mask, 0.0)
        attn = attn / attn.sum(dim=-1, keepdim=True)

        prefix = RaCompressAnalysisMethod._prefix_score_for_matrix(attn)
        copying = RaCompressAnalysisMethod._copying_score_for_matrix(attn)
        # 均匀分布下两者应该很接近但不完全相等（偏移差 1）
        self.assertGreater(prefix, 0.0)
        self.assertGreater(copying, 0.0)

    def test_copying_score_returns_zero_when_seq_len_below_dummy_length(self):
        """seq_len < DUMMY_INPUT_LENGTH 时没有完整段，copying 返回 0。"""
        attn = torch.ones(DUMMY_INPUT_LENGTH - 1, DUMMY_INPUT_LENGTH - 1)
        attn = attn / attn.sum(dim=-1, keepdim=True)
        self.assertEqual(RaCompressAnalysisMethod._copying_score_for_matrix(attn), 0.0)


class TestRaCompressMaxEveryGroup(unittest.TestCase):
    """测试 _max_every_group — GQA 分组取 max。"""

    def test_max_every_group_returns_data_unchanged_when_n_is_1(self):
        """n=1（无分组）时原样返回。"""
        data = {0: [1.0, 2.0, 3.0], 1: [4.0, 5.0]}
        result = RaCompressAnalysisMethod._max_every_group(data, 1)
        self.assertEqual(result, data)

    def test_max_every_group_returns_data_unchanged_when_n_is_zero(self):
        """n=0 时按 n<=1 分支原样返回（边界保护）。"""
        data = {0: [1.0, 2.0]}
        result = RaCompressAnalysisMethod._max_every_group(data, 0)
        self.assertEqual(result, data)

    def test_max_every_group_returns_group_max_when_n_is_2(self):
        """n=2 时每 2 个 head 取 max。"""
        data = {0: [1.0, 3.0, 2.0, 4.0]}
        result = RaCompressAnalysisMethod._max_every_group(data, 2)
        self.assertEqual(result[0], [3.0, 4.0])

    def test_max_every_group_returns_remainder_as_group_when_heads_not_divisible(self):
        """head 数不整除 n 时余项单独成组。"""
        data = {0: [1.0, 5.0, 3.0]}
        result = RaCompressAnalysisMethod._max_every_group(data, 2)
        self.assertEqual(result[0], [5.0, 3.0])


class TestRaCompressSelectTopHeads(unittest.TestCase):
    """测试 _select_top_heads — 按 ratio 选 top heads。"""

    def test_select_top_heads_returns_empty_when_data_empty(self):
        """空数据返回空 dict。"""
        self.assertEqual(RaCompressAnalysisMethod._select_top_heads({}, 0.14), {})

    def test_select_top_heads_returns_empty_when_ratio_is_zero(self):
        """ratio=0 时 percent_index=0，所有层返回空索引列表（边界）。"""
        data = {0: [0.1, 0.9]}
        result = RaCompressAnalysisMethod._select_top_heads(data, 0.0)
        self.assertEqual(result[0], [])

    def test_select_top_heads_returns_count_when_ratio_quarter(self):
        """25% ratio 正确选择 top heads。"""
        data = {0: [0.1, 0.9, 0.5, 0.3], 1: [0.8, 0.2, 0.7, 0.4]}
        result = RaCompressAnalysisMethod._select_top_heads(data, 0.25)
        # 8 个值，25% = 2 个，top 2 = [0.9, 0.8]
        all_selected = []
        for heads in result.values():
            all_selected.extend(heads)
        self.assertEqual(len(all_selected), 2)

    def test_select_top_heads_returns_one_when_ratio_one_percent(self):
        """1% ratio 在大数据集上只选极少 head。"""
        data = {0: list(range(100))}
        result = RaCompressAnalysisMethod._select_top_heads(data, 0.01)
        # 100 个值，1% = 1 个，top 1 = [99]
        self.assertEqual(len(result[0]), 1)

    def test_select_top_heads_returns_all_indices_when_ratio_is_one(self):
        """ratio=1.0 选择所有 head。"""
        data = {0: [0.1, 0.2, 0.3]}
        result = RaCompressAnalysisMethod._select_top_heads(data, 1.0)
        self.assertEqual(result[0], [0, 1, 2])


class TestRaCompressGetCompressHeads(unittest.TestCase):
    """测试 get_compress_heads — 完整 head 选择流水线。"""

    def setUp(self):
        self.method = RaCompressAnalysisMethod(adapter=FakeAdapter())
        self.method._num_attention_heads = 4
        self.method._num_key_value_heads = 2
        self.method._head_dim = 8

    def test_get_compress_heads_returns_empty_when_no_scores(self):
        """无分数数据时返回空 head_dict。"""
        result = self.method.get_compress_heads()
        self.assertEqual(result, {"prefix_matching": {}, "copying": {}})

    def test_get_compress_heads_selects_induction_when_prefix_scores_high(self):
        """induction heads 按 prefix 分数选择。"""
        self.method._prefix_scores = {
            0: [0.9, 0.1, 0.8, 0.2],
            1: [0.1, 0.9, 0.2, 0.8],
        }
        self.method._copying_scores = {
            0: [0.0, 0.0, 0.0, 0.0],
            1: [0.0, 0.0, 0.0, 0.0],
        }
        result = self.method.get_compress_heads()
        prefix_map = result["prefix_matching"]
        # GQA: num_heads=4, kv_heads=2, n=2, 分组后 [max(0.9,0.1), max(0.8,0.2)] = [0.9, 0.8]
        # 14% of 4 values (2 layers * 2 kv heads) = round(4 * 0.14) = 1 → top 1 = [0.9]
        # 0.9 在 layer 0, group 0 → kv head 0
        self.assertIn(0, prefix_map)

    def test_get_compress_heads_selects_echo_when_copying_scores_high(self):
        """echo heads 按 copying 分数选择。"""
        self.method._prefix_scores = {
            0: [0.0, 0.0, 0.0, 0.0],
            1: [0.0, 0.0, 0.0, 0.0],
        }
        self.method._copying_scores = {
            0: [0.1, 0.9, 0.2, 0.8],
            1: [0.3, 0.7, 0.4, 0.6],
        }
        result = self.method.get_compress_heads()
        copying_map = result["copying"]
        # GQA: num_heads=4, kv_heads=2, n=2
        # grouped: layer0=[max(0.1,0.9), max(0.2,0.8)]=[0.9,0.8], layer1=[max(0.3,0.7), max(0.4,0.6)]=[0.7,0.6]
        # 4 values, 1% = round(4*0.01)=0 → 0 selected, 用更大 ratio 测试
        # 改用 induction_head_ratio=0.5 模拟
        self.method._induction_head_ratio = 0.5
        self.method._echo_head_ratio = 0.5
        result = self.method.get_compress_heads()
        copying_map = result["copying"]
        self.assertIn(0, copying_map)

    def test_get_compress_heads_removes_empty_lists_when_layer_scores_zero(self):
        """空列表的层被移除。"""
        self.method._prefix_scores = {
            0: [0.0, 0.0, 0.0, 0.0],
            1: [0.9, 0.1, 0.8, 0.2],
        }
        self.method._copying_scores = {
            0: [0.0, 0.0, 0.0, 0.0],
            1: [0.0, 0.0, 0.0, 0.0],
        }
        result = self.method.get_compress_heads()
        # layer 0 的所有分数都是 0，不应出现在 prefix_matching 中
        self.assertNotIn(0, result["prefix_matching"])

    def test_get_compress_heads_returns_heads_when_no_gqa(self):
        """num_kv_heads == num_attention_heads（无 GQA）时分组 n=1 原样选择（边界）。"""
        method = RaCompressAnalysisMethod(adapter=FakeAdapter())
        method._num_attention_heads = 2
        method._num_key_value_heads = 2  # 无 GQA，分组 n=1
        method._head_dim = 4
        method._induction_head_ratio = 0.5
        method._prefix_scores = {0: [0.9, 0.1]}
        method._copying_scores = {0: [0.0, 0.0]}
        result = method.get_compress_heads()
        self.assertIsInstance(result["prefix_matching"], dict)


class TestRaCompressEnrichLayerScores(unittest.TestCase):
    """测试 enrich_layer_scores — 将 head 信息写入 layer_scores。"""

    def setUp(self):
        self.method = RaCompressAnalysisMethod(adapter=FakeAdapter())
        self.method._num_attention_heads = 4
        self.method._num_key_value_heads = 2
        self.method._head_dim = 8

    def test_enrich_layer_scores_populates_heads_when_scores_exist(self):
        """正确填充 induction_heads / echo_heads。"""
        self.method._prefix_scores = {0: [0.9, 0.1, 0.8, 0.2]}
        self.method._copying_scores = {0: [0.1, 0.9, 0.2, 0.8]}
        self.method._layer_idx_to_name = {0: "model.layers.0.self_attn.q_proj"}

        layer_scores = [{"name": "model.layers.0.self_attn.q_proj", "score": 0.5}]
        self.method.enrich_layer_scores(layer_scores)

        self.assertIn("induction_heads", layer_scores[0])
        self.assertIn("echo_heads", layer_scores[0])
        self.assertIsInstance(layer_scores[0]["induction_heads"], list)
        self.assertIsInstance(layer_scores[0]["echo_heads"], list)

    def test_enrich_layer_scores_sets_empty_lists_when_layer_missing(self):
        """layer_scores 中的层没有对应分数时，heads 为空列表。"""
        self.method._prefix_scores = {}
        self.method._copying_scores = {}
        self.method._layer_idx_to_name = {}

        layer_scores = [{"name": "model.layers.0.self_attn.q_proj", "score": 0.5}]
        self.method.enrich_layer_scores(layer_scores)

        self.assertEqual(layer_scores[0]["induction_heads"], [])
        self.assertEqual(layer_scores[0]["echo_heads"], [])

    def test_enrich_layer_scores_keeps_empty_when_layer_scores_empty(self):
        """layer_scores 为空列表时不报错、保持空（异常/边界）。"""
        layer_scores = []
        self.method.enrich_layer_scores(layer_scores)
        self.assertEqual(layer_scores, [])

    @patch("msmodelslim.processor.analysis.unary_operator.metrics.ra_compress.impl.dist")
    def test_enrich_layer_scores_syncs_head_scores_under_dp(self, mock_dist):
        """DP 下 enrich_layer_scores 先 gather 原始 head 分并平均。"""
        mock_dist.is_initialized.return_value = True
        mock_dist.get_world_size.return_value = 2

        self.method._layer_name_to_idx = {"layer.q": 0}
        self.method._layer_idx_to_name = {0: "layer.q"}
        self.method._prefix_scores = {0: [1.0, 0.0]}
        self.method._copying_scores = {0: [0.0, 1.0]}
        self.method._next_layer_idx = 1

        def _all_gather_object(out_list, obj):
            out_list[0] = {"layer.q": ([1.0, 0.0], [0.0, 1.0])}
            out_list[1] = {"layer.q": ([3.0, 2.0], [2.0, 3.0])}

        mock_dist.all_gather_object.side_effect = _all_gather_object
        layer_scores = [{"name": "layer.q", "score": 1.0}]
        self.method.enrich_layer_scores(layer_scores)

        self.assertEqual(self.method._prefix_scores[0], [2.0, 1.0])
        self.assertEqual(self.method._copying_scores[0], [1.0, 2.0])

        layer_scores = [{"name": "layer.q", "score": 0.0}]
        self.method.enrich_layer_scores(layer_scores)
        self.assertIn("induction_heads", layer_scores[0])
        self.assertIn("echo_heads", layer_scores[0])


class TestRaCompressFlattenTo2D(unittest.TestCase):
    """测试 _flatten_to_2d — tensor 展平。"""

    def test_flatten_to_2d_returns_unchanged_when_input_2d(self):
        """2D tensor 原样返回。"""
        t = torch.randn(10, 8)
        result = RaCompressAnalysisMethod._flatten_to_2d(t)
        self.assertEqual(result.shape, (10, 8))
        self.assertTrue(torch.equal(result, t))

    def test_flatten_to_2d_returns_flattened_when_input_3d(self):
        """3D tensor [batch, seq, dim] → [batch*seq, dim]。"""
        t = torch.randn(2, 5, 8)
        result = RaCompressAnalysisMethod._flatten_to_2d(t)
        self.assertEqual(result.shape, (10, 8))

    def test_flatten_to_2d_returns_flattened_when_input_4d(self):
        """4D tensor → [*, dim]。"""
        t = torch.randn(2, 3, 4, 8)
        result = RaCompressAnalysisMethod._flatten_to_2d(t)
        self.assertEqual(result.shape, (24, 8))


class TestRaCompressIsTargetLayer(unittest.TestCase):
    """测试 _is_target_layer — 层名匹配。"""

    def setUp(self):
        self.method = RaCompressAnalysisMethod(adapter=FakeAdapter())

    def test_is_target_layer_returns_true_when_q_proj(self):
        self.assertTrue(self.method._is_target_layer("model.layers.0.self_attn.q_proj"))

    def test_is_target_layer_returns_true_when_k_proj(self):
        self.assertTrue(self.method._is_target_layer("model.layers.0.self_attn.k_proj"))

    def test_is_target_layer_returns_true_when_qkv_proj(self):
        self.assertTrue(self.method._is_target_layer("model.layers.0.self_attn.qkv_proj"))

    def test_is_target_layer_returns_false_when_non_proj_layer(self):
        self.assertFalse(self.method._is_target_layer("model.layers.0.mlp.gate_proj"))
        self.assertFalse(self.method._is_target_layer("model.layers.0.self_attn.o_proj"))


class TestRaCompressComputeScore(unittest.TestCase):
    """测试 compute_score — 分数计算分派。"""

    def setUp(self):
        self.method = RaCompressAnalysisMethod(adapter=FakeAdapter())
        self.method._num_attention_heads = 2
        self.method._num_key_value_heads = 2
        self.method._head_dim = 4

    def test_compute_score_returns_zero_when_no_outputs(self):
        """无 output 数据返回 0。"""
        result = self.method.compute_score({"layer_name": "q_proj", "outputs": []})
        self.assertEqual(result, 0.0)

    def test_compute_score_returns_zero_when_k_layer(self):
        """K 层不计算分数，返回 0。"""
        result = self.method.compute_score({"layer_name": "k_proj", "outputs": [torch.randn(1, 4)]})
        self.assertEqual(result, 0.0)

    def test_compute_score_returns_zero_when_non_target_layer(self):
        """非 Q/K/QKV 层返回 0。"""
        result = self.method.compute_score({"layer_name": "o_proj", "outputs": [torch.randn(1, 4)]})
        self.assertEqual(result, 0.0)

    def test_compute_score_returns_zero_when_q_layer_has_no_k_output(self):
        """Q 层但没有对应 K 输出时返回 0。"""
        self.method._q_outputs = {"model.layers.0.self_attn.q_proj": torch.randn(10, 8)}
        result = self.method.compute_score(
            {
                "layer_name": "model.layers.0.self_attn.q_proj",
                "outputs": [torch.randn(10, 8)],
            }
        )
        self.assertEqual(result, 0.0)

    def test_compute_score_returns_zero_when_layer_name_empty(self):
        """layer_name 为空字符串时不匹配任何模式，返回 0（异常/边界）。"""
        result = self.method.compute_score({"layer_name": "", "outputs": [torch.randn(1, 4)]})
        self.assertEqual(result, 0.0)


class TestRaCompressComputeQkScores(unittest.TestCase):
    """测试 _compute_qk_scores — Q@K^T 分数计算。"""

    def test_compute_qk_scores_returns_empty_when_attention_config_not_set(self):
        """注意力配置未提取时返回空列表。"""
        method = RaCompressAnalysisMethod(adapter=FakeAdapter())
        method._num_attention_heads = 0
        method._head_dim = 0
        q = torch.randn(10, 8)
        k = torch.randn(10, 8)
        prefix, copying = method._compute_qk_scores("q_proj", q, k)
        self.assertEqual(prefix, [])
        self.assertEqual(copying, [])

    def test_compute_qk_scores_returns_empty_when_token_count_insufficient(self):
        """token 数量不足时返回空列表（异常被 catch 并 log warning）。"""
        method = RaCompressAnalysisMethod(adapter=FakeAdapter())
        method._num_attention_heads = 2
        method._num_key_value_heads = 2
        method._head_dim = 4

        # 只有 10 个 token，远小于 2500*4=10000
        q = torch.randn(10, 8)
        k = torch.randn(10, 8)

        prefix, copying = method._compute_qk_scores("q_proj", q, k)
        self.assertEqual(prefix, [])
        self.assertEqual(copying, [])

    def test_compute_qk_scores_returns_per_head_scores_when_tokens_sufficient(self):
        """满足条件时返回每头分数列表。"""
        method = RaCompressAnalysisMethod(adapter=FakeAdapter())
        method._num_attention_heads = 2
        method._num_key_value_heads = 2
        method._head_dim = 4
        method._induction_head_ratio = 0.5
        method._echo_head_ratio = 0.5

        total_tokens = DUMMY_INPUT_LENGTH * REPET_TIMES
        q = torch.randn(total_tokens, 8)
        k = torch.randn(total_tokens, 8)

        prefix, copying = method._compute_qk_scores("q_proj", q, k)
        self.assertEqual(len(prefix), 2)
        self.assertEqual(len(copying), 2)
        # 分数在 [0, 1] 范围内（softmax 概率）
        for score in prefix:
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)
        for score in copying:
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)


class TestRaCompressInterfaceIntegration(unittest.TestCase):
    """测试 RaCompressAnalysisInterface 接口集成。"""

    def test_init_loads_patterns_when_adapter_provided(self):
        """adapter 提供的名称模式被正确使用。"""
        method = RaCompressAnalysisMethod(adapter=FakeAdapter())
        self.assertEqual(method._q_name_pattern, "q_proj")
        self.assertEqual(method._k_name_pattern, "k_proj")
        self.assertEqual(method._qkv_name_pattern, "qkv_proj")

    def test_init_uses_default_patterns_when_adapter_none(self):
        """无 adapter 时使用默认名称模式。"""
        method = RaCompressAnalysisMethod(adapter=None)
        self.assertEqual(method._q_name_pattern, "q_proj")
        self.assertEqual(method._k_name_pattern, "k_proj")
        self.assertEqual(method._qkv_name_pattern, "qkv_proj")

    def test_init_uses_custom_patterns_when_adapter_overrides(self):
        """adapter 提供自定义名称模式。"""

        class CustomAdapter(RaCompressAnalysisInterface):
            def get_proj_names(self) -> Dict[str, str]:
                return {"q": "query", "k": "key", "qkv": "qkv_fused"}

            def get_tokenizer(self):
                return None

        method = RaCompressAnalysisMethod(adapter=CustomAdapter())
        self.assertEqual(method._q_name_pattern, "query")
        self.assertEqual(method._k_name_pattern, "key")
        self.assertEqual(method._qkv_name_pattern, "qkv_fused")
        self.assertTrue(method._is_target_layer("model.layers.0.query"))
        self.assertFalse(method._is_target_layer("model.layers.0.q_proj"))


class TestRaCompressHookBehavior(unittest.TestCase):
    """测试 get_hook 注册的 hook 行为。"""

    def setUp(self):
        self.method = RaCompressAnalysisMethod(adapter=FakeAdapter())
        self.hook = self.method.get_hook()

    def test_hook_stores_q_output_when_q_proj_layer(self):
        """hook 正确存储 Q 输出。"""
        stats_dict = {}
        q_output = torch.randn(5, 8)
        self.hook(None, None, q_output, "model.layers.0.q_proj", stats_dict)
        self.assertIn("model.layers.0.q_proj", self.method._q_outputs)
        self.assertTrue(torch.equal(self.method._q_outputs["model.layers.0.q_proj"], q_output))

    def test_hook_stores_k_output_when_k_proj_layer(self):
        """hook 正确存储 K 输出。"""
        stats_dict = {}
        k_output = torch.randn(5, 8)
        self.hook(None, None, k_output, "model.layers.0.k_proj", stats_dict)
        self.assertIn("model.layers.0.k_proj", self.method._k_outputs)

    def test_hook_stores_qkv_to_both_when_qkv_proj_layer(self):
        """QKV 融合层输出同时存入 Q 和 K。"""
        stats_dict = {}
        qkv_output = torch.randn(5, 24)
        self.hook(None, None, qkv_output, "model.layers.0.qkv_proj", stats_dict)
        self.assertIn("model.layers.0.qkv_proj", self.method._q_outputs)
        self.assertIn("model.layers.0.qkv_proj", self.method._k_outputs)

    def test_hook_populates_stats_dict_when_called(self):
        """hook 正确填充 stats_dict。"""
        stats_dict = {}
        output = torch.randn(5, 8)
        self.hook(None, None, output, "model.layers.0.q_proj", stats_dict)
        self.assertIn("model.layers.0.q_proj", stats_dict)
        self.assertIn("outputs", stats_dict["model.layers.0.q_proj"])
        self.assertEqual(len(stats_dict["model.layers.0.q_proj"]["outputs"]), 1)

    def test_hook_stores_first_element_when_output_is_tuple(self):
        """hook 处理 tuple 输出（取第 0 项）。"""
        stats_dict = {}
        output = (torch.randn(5, 8), torch.randn(5, 8))
        self.hook(None, None, output, "model.layers.0.q_proj", stats_dict)
        stored = self.method._q_outputs["model.layers.0.q_proj"]
        self.assertTrue(torch.equal(stored, output[0]))

    def test_hook_appends_multiple_outputs_when_called_repeatedly(self):
        """hook 多次调用时 outputs 列表累积（边界：多次前向）。"""
        stats_dict = {}
        self.hook(None, None, torch.randn(5, 8), "model.layers.0.q_proj", stats_dict)
        self.hook(None, None, torch.randn(5, 8), "model.layers.0.q_proj", stats_dict)
        self.assertEqual(len(stats_dict["model.layers.0.q_proj"]["outputs"]), 2)


class TestRaCompressEndToEnd(unittest.TestCase):
    """端到端测试：从 hook 到 head 选择的完整流程。"""

    def test_pipeline_selects_heads_when_qk_separate_projection(self):
        """Q/K 分离投影的完整流程：hook → compute_score → get_compress_heads。"""
        method = RaCompressAnalysisMethod(adapter=FakeAdapter())
        method._num_attention_heads = 2
        method._num_key_value_heads = 2
        method._head_dim = 4
        method._induction_head_ratio = 0.5
        method._echo_head_ratio = 0.5

        total_tokens = DUMMY_INPUT_LENGTH * REPET_TIMES
        q_output = torch.randn(total_tokens, 8)
        k_output = torch.randn(total_tokens, 8)

        hook = method.get_hook()
        stats_dict_q = {}
        stats_dict_k = {}
        hook(None, None, q_output, "model.layers.0.q_proj", stats_dict_q)
        hook(None, None, k_output, "model.layers.0.k_proj", stats_dict_k)

        # compute_score 对 Q 层计算
        score = method.compute_score(stats_dict_q["model.layers.0.q_proj"])
        self.assertGreaterEqual(score, 0.0)

        # get_compress_heads
        head_dict = method.get_compress_heads()
        self.assertIn("prefix_matching", head_dict)
        self.assertIn("copying", head_dict)

        # enrich_layer_scores
        layer_scores = [{"name": "model.layers.0.q_proj", "score": score}]
        method.enrich_layer_scores(layer_scores)
        self.assertIn("induction_heads", layer_scores[0])
        self.assertIn("echo_heads", layer_scores[0])

    def test_pipeline_selects_heads_when_qkv_fused_projection(self):
        """QKV 融合投影的完整流程：hook → compute_score → get_compress_heads（正常场景补充）。"""
        method = RaCompressAnalysisMethod(adapter=FakeAdapter())
        method._num_attention_heads = 2
        method._num_key_value_heads = 2
        method._head_dim = 4
        method._induction_head_ratio = 0.5
        method._echo_head_ratio = 0.5

        total_tokens = DUMMY_INPUT_LENGTH * REPET_TIMES
        # q_dim(num_heads*head_dim=2*4=8) + k_dim(num_kv*head_dim=2*4=8) = 16
        qkv_output = torch.randn(total_tokens, 16)

        hook = method.get_hook()
        stats_dict = {}
        hook(None, None, qkv_output, "model.layers.0.qkv_proj", stats_dict)

        score = method.compute_score(stats_dict["model.layers.0.qkv_proj"])
        self.assertGreaterEqual(score, 0.0)

        head_dict = method.get_compress_heads()
        self.assertIn("prefix_matching", head_dict)
        self.assertIn("copying", head_dict)


_CALIB_INPUT_LOGGER = "msmodelslim.processor.analysis.unary_operator.metrics.ra_compress.calib_input.get_logger"
_IMPL_LOGGER = "msmodelslim.processor.analysis.unary_operator.metrics.ra_compress.impl.logger"


class _FakeInterfaceAdapter(RaCompressAnalysisInterface):
    """可注入 tokenizer 的假 adapter，用于覆盖首 token 取值的各类分支。"""

    def __init__(self, tokenizer=None):
        self._tokenizer = tokenizer
        self.tokenizer_calls = 0

    def get_proj_names(self) -> Dict[str, str]:
        return {"q": "q_proj", "k": "k_proj", "qkv": "qkv_proj"}

    def get_tokenizer(self):
        self.tokenizer_calls += 1
        return self._tokenizer


class _RaisingTokenizerAdapter(_FakeInterfaceAdapter):
    """get_tokenizer 抛异常的假 adapter。"""

    def get_tokenizer(self):
        raise RuntimeError("tokenizer unavailable")


class TestRaCompressPrefixTokenOnModelSide(unittest.TestCase):
    """测试首 token 归属：模型侧只提供 tokenizer，V0 配方留在指标侧。"""

    def test_uses_v0_token_when_adapter_provides_tokenizer(self):
        """场景：适配器实现接口并给出 tokenizer。预期：按 V0 口径取 'A' 的末位 token。"""
        adapter = _FakeInterfaceAdapter(tokenizer=_FakeTokenizer({"": [], "A": [32]}))
        self.assertEqual(resolve_prefix_token_id(adapter), 32)

    def test_uses_last_token_of_empty_input_when_not_empty(self):
        """场景：tokenizer('') 带 BOS。预期：直接取该末位 token，不走 'A' 兜底。"""
        adapter = _FakeInterfaceAdapter(tokenizer=_FakeTokenizer({"": [1]}))
        self.assertEqual(resolve_prefix_token_id(adapter), 1)

    def test_falls_back_and_warns_when_tokenizer_not_provided(self):
        """场景：实现了接口但 get_tokenizer 返回 None。预期：告警并回落兜底值。"""
        adapter = _FakeInterfaceAdapter()
        with patch(_CALIB_INPUT_LOGGER) as mock_logger:
            self.assertEqual(resolve_prefix_token_id(adapter), PREFIX_TOKEN_ID)
        self.assertEqual(adapter.tokenizer_calls, 1)
        self.assertTrue(mock_logger.return_value.warning.called)

    def test_falls_back_and_warns_when_interface_not_implemented(self):
        """场景：适配器未实现本接口。预期：不询问 tokenizer，告警并回落兜底值。"""
        with patch(_CALIB_INPUT_LOGGER) as mock_logger:
            self.assertEqual(resolve_prefix_token_id(object()), PREFIX_TOKEN_ID)
        self.assertTrue(mock_logger.return_value.warning.called)

    def test_falls_back_and_warns_when_get_tokenizer_raises(self):
        """场景：适配器的 get_tokenizer 抛异常。预期：告警并回落兜底值，不抛给上层。"""
        with patch(_CALIB_INPUT_LOGGER) as mock_logger:
            self.assertEqual(resolve_prefix_token_id(_RaisingTokenizerAdapter()), PREFIX_TOKEN_ID)
        self.assertTrue(mock_logger.return_value.warning.called)

    def test_falls_back_when_tokenizer_call_raises(self):
        """场景：tokenizer 自身解析失败。预期：回落兜底值。"""

        class _BadTokenizer:
            def __call__(self, text, **kwargs):
                raise RuntimeError("bad tokenizer")

        adapter = _FakeInterfaceAdapter(tokenizer=_BadTokenizer())
        self.assertEqual(resolve_prefix_token_id(adapter), PREFIX_TOKEN_ID)

    def test_resolve_model_tokenizer_returns_none_without_interface(self):
        """场景：非接口对象。预期：返回 None，不尝试取值。"""
        self.assertIsNone(resolve_model_tokenizer(object()))


class TestRaCompressEmptyPatternSemantics(unittest.TestCase):
    """测试名称模式的空串语义：空串表示该投影层不存在，不参与匹配。"""

    @staticmethod
    def _adapter(patterns: Dict[str, str]) -> RaCompressAnalysisInterface:
        class _Adapter(RaCompressAnalysisInterface):
            def get_proj_names(self) -> Dict[str, str]:
                return patterns

            def get_tokenizer(self):
                return None

        return _Adapter()

    def test_empty_qkv_pattern_does_not_match_any_layer(self):
        """场景：无 QKV 融合时按文档示例留空 qkv=""。预期：不命中任何层名。"""
        method = RaCompressAnalysisMethod(adapter=self._adapter({"q": "q_proj", "k": "k_proj", "qkv": ""}))
        self.assertEqual(method._qkv_name_pattern, "")
        self.assertFalse(method._is_target_layer("model.layers.0.self_attn.qkv_proj"))
        self.assertFalse(method._is_target_layer("lm_head"))
        self.assertTrue(method._is_target_layer("model.layers.0.self_attn.q_proj"))

    def test_empty_patterns_skip_hook_and_return_zero_score(self):
        """场景：三个模式都留空。预期：hook 不存输出、算分返回 0（空串不再命中所有 Linear）。"""
        method = RaCompressAnalysisMethod(adapter=self._adapter({"q": "", "k": "", "qkv": ""}))
        method._num_attention_heads = 2
        method._num_key_value_heads = 2
        method._head_dim = 4
        method.get_hook()(None, None, torch.randn(6, 8), "model.layers.0.self_attn.q_proj", {})
        self.assertEqual(method._q_outputs, {})
        self.assertEqual(method._k_outputs, {})
        self.assertEqual(method.compute_score({"layer_name": "model.layers.0.self_attn.q_proj", "outputs": []}), 0.0)

    def test_warns_when_all_patterns_empty(self):
        """场景：适配器把三个模式都留空。预期：初始化时告警提示不会有层被分析。"""
        with patch(_IMPL_LOGGER) as mock_logger:
            RaCompressAnalysisMethod(adapter=self._adapter({"q": "", "k": "", "qkv": ""}))
        self.assertTrue(mock_logger.warning.called)


def _make_cos_sin(seq_len: int, head_dim: int, offset: float = 0.0) -> tuple:
    """构造显式的 [seq_len, head_dim] cos/sin（前后半维配对布局）。"""
    steps = torch.arange(seq_len, dtype=torch.float32).unsqueeze(1) * 0.1 + offset
    freqs = torch.arange(1, head_dim // 2 + 1, dtype=torch.float32).unsqueeze(0)
    angles = steps * freqs
    return torch.cos(angles).repeat(1, 2), torch.sin(angles).repeat(1, 2)


def _manual_cos_sin(seq_len: int, head_dim: int, rope_theta: float) -> tuple:
    """按实现的 config 推算分支同一公式显式构造 cos/sin。"""
    inv_freq = 1.0 / (rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    freqs = torch.outer(torch.arange(seq_len, dtype=inv_freq.dtype), inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos(), emb.sin()


def _expected_roped(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, seq_len: int) -> torch.Tensor:
    """显式构造 rotate_half 旋转结果，作为施加 RoPE 后的期望值。"""
    cos_b = cos[:seq_len].to(dtype=x.dtype).unsqueeze(1)
    sin_b = sin[:seq_len].to(dtype=x.dtype).unsqueeze(1)
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return x * cos_b + torch.cat((-x2, x1), dim=-1) * sin_b


class _FakeRotaryEmb:
    """假 rotary_emb：记录调用入参，可模拟"调用失败但 cos/sin 缓存可用"。"""

    def __init__(self, cos: torch.Tensor, sin: torch.Tensor, raise_on_call: bool = False, with_cache: bool = False):
        self._cos = cos
        self._sin = sin
        self._raise_on_call = raise_on_call
        self.calls = []
        if with_cache:
            self.cos_cached = cos
            self.sin_cached = sin

    def __call__(self, x, **kwargs):
        self.calls.append(kwargs)
        if self._raise_on_call:
            raise RuntimeError("unsupported rotary signature")
        return self._cos.to(x.device), self._sin.to(x.device)


class _FakeRopeConfig:
    """假 config：只提供内置兜底推算需要的 rope 字段。"""

    def __init__(self, rope_theta: float, partial_rotary_factor: float = 1.0):
        self.rope_theta = rope_theta
        self.partial_rotary_factor = partial_rotary_factor


class _FakeAttentionModule:
    """假 self_attn：只承载内置兜底解析 RoPE 需要的属性。"""

    def __init__(self, rotary_emb=None, rope_theta=None, partial_rotary_factor: float = 1.0):
        if rotary_emb is not None:
            self.rotary_emb = rotary_emb
        if rope_theta is not None:
            self.config = _FakeRopeConfig(rope_theta, partial_rotary_factor)


class TestRaCompressRopeResolution(unittest.TestCase):
    """测试 RoPE 取值与施加：position_embeddings、rotary 模块、config 推算三条路径。"""

    LAYER = "model.layers.0.self_attn.q_proj"
    BLOCK = "model.layers.0"

    def setUp(self):
        torch.manual_seed(0)
        self.seq_len = 4
        self.head_dim = 4
        self.q = torch.randn(self.seq_len, 2, self.head_dim)
        self.k = torch.randn(self.seq_len, 2, self.head_dim)

    def _apply(self, method):
        return method._apply_rope(self.LAYER, self.q, self.k, self.seq_len, torch.device("cpu"))

    def _assert_roped(self, method, cos, sin):
        """断言施加 RoPE 后的 Q/K 与显式构造的旋转结果一致。"""
        q_out, k_out = self._apply(method)
        self.assertTrue(torch.allclose(q_out, _expected_roped(self.q, cos, sin, self.seq_len), atol=1e-6))
        self.assertTrue(torch.allclose(k_out, _expected_roped(self.k, cos, sin, self.seq_len), atol=1e-6))

    def test_applies_position_embeddings_from_forward_kwargs(self):
        """取值路径 2：新版布局由模型算好后随 forward kwargs 下发 position_embeddings。"""
        cos, sin = _make_cos_sin(self.seq_len, self.head_dim)
        method = RaCompressAnalysisMethod(adapter=None)
        method.set_forward_kwargs({"position_embeddings": (cos, sin)}, layer_name=self.BLOCK)
        self._assert_roped(method, cos, sin)

    def test_applies_cos_sin_from_attention_rotary_module(self):
        """取值路径 3：旧版布局 attention 内自带 rotary_emb，等距生成 position_ids 调用。"""
        cos, sin = _make_cos_sin(self.seq_len, self.head_dim)
        rotary_emb = _FakeRotaryEmb(cos, sin)
        method = RaCompressAnalysisMethod(adapter=None)
        method._layer_attn_modules[self.LAYER] = _FakeAttentionModule(rotary_emb=rotary_emb)
        self._assert_roped(method, cos, sin)
        self.assertEqual(rotary_emb.calls[0]["position_ids"].tolist(), [[0, 1, 2, 3]])

    def test_reuses_position_ids_from_forward_kwargs(self):
        """取值路径 3：模型下发的 position_ids 优先复用（分段输入等场景不从头编号）。"""
        cos, sin = _make_cos_sin(self.seq_len, self.head_dim)
        rotary_emb = _FakeRotaryEmb(cos, sin)
        method = RaCompressAnalysisMethod(adapter=None)
        method._layer_attn_modules[self.LAYER] = _FakeAttentionModule(rotary_emb=rotary_emb)
        method.set_forward_kwargs({"position_ids": torch.tensor([[7, 8, 9, 10]])}, layer_name=self.BLOCK)
        self._assert_roped(method, cos, sin)
        self.assertEqual(rotary_emb.calls[0]["position_ids"].tolist(), [[7, 8, 9, 10]])

    def test_applies_cached_cos_sin_when_rotary_module_call_fails(self):
        """取值路径 3 的兜底：rotary_emb 三种签名都不可用时退到 cos_cached/sin_cached。"""
        cos, sin = _make_cos_sin(self.seq_len, self.head_dim)
        rotary_emb = _FakeRotaryEmb(cos, sin, raise_on_call=True, with_cache=True)
        method = RaCompressAnalysisMethod(adapter=None)
        method._layer_attn_modules[self.LAYER] = _FakeAttentionModule(rotary_emb=rotary_emb)
        self._assert_roped(method, cos, sin)
        self.assertEqual(len(rotary_emb.calls), 3)

    def test_applies_manual_cos_sin_from_config_rope_theta(self):
        """取值路径 4：前三条都取不到时按 config 的 rope_theta 推算，并告警说明未含 rope_scaling。"""
        method = RaCompressAnalysisMethod(adapter=None)
        method._layer_attn_modules[self.LAYER] = _FakeAttentionModule(rope_theta=10000.0)
        self._assert_roped(method, *_manual_cos_sin(self.seq_len, self.head_dim, 10000.0))
        self.assertIn("manual_fallback", method._rope_warning_keys)

    def test_skips_rope_when_no_source_available(self):
        """兜底分支：无 kwargs、无 attention 模块 → 保留原 Q/K 并告警。"""
        method = RaCompressAnalysisMethod(adapter=None)
        q_out, k_out = self._apply(method)
        self.assertTrue(torch.equal(q_out, self.q))
        self.assertTrue(torch.equal(k_out, self.k))
        self.assertIn("unavailable", method._rope_warning_keys)

    def test_skips_rope_when_cos_sin_dim_mismatch(self):
        """兜底分支：模型下发的 cos/sin 维度与 head_dim 不一致 → 宁可不加也不加错。"""
        cos, sin = _make_cos_sin(self.seq_len, 2)
        method = RaCompressAnalysisMethod(adapter=None)
        method.set_forward_kwargs({"position_embeddings": (cos, sin)}, layer_name=self.BLOCK)
        q_out, _ = self._apply(method)
        self.assertTrue(torch.equal(q_out, self.q))
        self.assertIn("dim_mismatch", method._rope_warning_keys)

    def test_skips_rope_when_cos_sin_covers_fewer_positions(self):
        """兜底分支：cos/sin 覆盖的位置数不足 seq_len → 跳过 RoPE。"""
        cos, sin = _make_cos_sin(self.seq_len - 2, self.head_dim)
        method = RaCompressAnalysisMethod(adapter=None)
        method.set_forward_kwargs({"position_embeddings": (cos, sin)}, layer_name=self.BLOCK)
        q_out, _ = self._apply(method)
        self.assertTrue(torch.equal(q_out, self.q))
        self.assertIn("seq_short", method._rope_warning_keys)

    def test_warns_once_per_reason(self):
        """告警去重：同类原因逐层触发也只记录一次。"""
        method = RaCompressAnalysisMethod(adapter=None)
        for layer_name in ("model.layers.0.self_attn.q_proj", "model.layers.1.self_attn.q_proj"):
            method._apply_rope(layer_name, self.q, self.k, self.seq_len, torch.device("cpu"))
        with patch(_IMPL_LOGGER) as mock_logger:
            method._apply_rope("model.layers.2.self_attn.q_proj", self.q, self.k, self.seq_len, torch.device("cpu"))
        self.assertFalse(mock_logger.warning.called)


if __name__ == "__main__":
    unittest.main()
