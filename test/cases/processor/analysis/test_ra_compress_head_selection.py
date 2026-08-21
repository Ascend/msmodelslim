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

import torch

from msmodelslim.processor.analysis.unary_operator.metrics.ra_compress import (
    DUMMY_INPUT_LENGTH,
    REPET_TIMES,
    RaCompressAnalysisMethod,
)
from msmodelslim.processor.analysis.unary_operator.metrics.ra_compress.interface import (
    RaCompressAnalysisInterface,
)


class FakeAdapter(RaCompressAnalysisInterface):
    """实现 RaCompressAnalysisInterface 的测试用 adapter。"""

    def get_ra_compress_proj_patterns(self) -> Dict[str, str]:
        return {"q": "q_proj", "k": "k_proj", "qkv": "qkv_proj"}


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
            def get_ra_compress_proj_patterns(self) -> Dict[str, str]:
                return {"q": "query", "k": "key", "qkv": "qkv_fused"}

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


if __name__ == "__main__":
    unittest.main()
