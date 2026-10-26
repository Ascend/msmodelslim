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

Unit tests for ra_compress random-segment calibration input builders.
"""

import unittest

import torch

from msmodelslim.processor.analysis.unary_operator.metrics.ra_compress.calib_input import (
    DUMMY_INPUT_LENGTH,
    PLACEHOLDER_DATASET,
    PREFIX_TOKEN_ID,
    REPET_TIMES,
    TOKEN_ID_LOW,
    build_ra_compress_dummy_dataset,
    build_random_token_calib,
    is_random_token_metric,
)
from msmodelslim.processor.analysis.unary_operator.metrics.ra_compress.interface import (
    RaCompressAnalysisInterface,
)


class _Adapter(RaCompressAnalysisInterface):
    def __init__(self):
        self.forward_calls = []

    def get_proj_names(self):
        return {"q": "q_proj", "k": "k_proj", "qkv": "qkv_proj"}

    def get_tokenizer(self):
        return None

    def generate_model_forward(self, model, inputs):
        self.forward_calls.append(inputs)
        return "forward-result"


class TestIsRandomTokenMetric(unittest.TestCase):
    def test_only_ra_compress_is_random_token_metric(self):
        self.assertTrue(is_random_token_metric("ra_compress"))
        self.assertFalse(is_random_token_metric("std"))
        self.assertFalse(is_random_token_metric(""))


class TestBuildRaCompressDummyDataset(unittest.TestCase):
    def test_dataset_shape_prefix_and_determinism(self):
        first = build_ra_compress_dummy_dataset(prefix_token_id=7, dummy_input_length=3, repet_times=2, seed=1)
        second = build_ra_compress_dummy_dataset(prefix_token_id=7, dummy_input_length=3, repet_times=2, seed=1)
        self.assertEqual(len(first), 1)

        input_ids, attention_mask = first[0]
        self.assertEqual(tuple(input_ids.shape), (1, 7))
        self.assertEqual(int(input_ids[0, 0]), 7)
        self.assertTrue(torch.equal(input_ids, second[0][0]))
        self.assertTrue(torch.equal(attention_mask, torch.ones_like(input_ids)))

        random_tokens = input_ids[0, 1:]
        self.assertEqual(random_tokens.numel(), 6)
        self.assertGreaterEqual(int(random_tokens.min()), TOKEN_ID_LOW)
        self.assertLessEqual(int(random_tokens.max()), TOKEN_ID_LOW + 3)

    def test_default_recipe_matches_module_constants(self):
        input_ids, attention_mask = build_ra_compress_dummy_dataset(prefix_token_id=PREFIX_TOKEN_ID)[0]
        expected_length = 1 + DUMMY_INPUT_LENGTH * REPET_TIMES
        self.assertEqual(tuple(input_ids.shape), (1, expected_length))
        self.assertEqual(int(input_ids[0, 0]), PREFIX_TOKEN_ID)
        self.assertEqual(int(attention_mask.sum()), expected_length)
        self.assertGreaterEqual(int(input_ids[0, 1]), TOKEN_ID_LOW)
        self.assertLessEqual(int(input_ids[0, -1]), TOKEN_ID_LOW + DUMMY_INPUT_LENGTH)


class TestBuildRandomTokenCalib(unittest.TestCase):
    def test_context_manager_swaps_forward_and_feeds_random_tokens(self):
        adapter = _Adapter()
        original = adapter.generate_model_forward

        calib, forward_override = build_random_token_calib(adapter)
        self.assertEqual(calib, PLACEHOLDER_DATASET)

        with forward_override:
            self.assertEqual(type(adapter.generate_model_forward).__name__, "_RandomTokenForward")
            result = adapter.generate_model_forward(None, None)

        self.assertEqual(result, "forward-result")
        self.assertEqual(len(adapter.forward_calls), 1)
        input_ids = adapter.forward_calls[0][0]
        self.assertEqual(int(input_ids[0, 0]), PREFIX_TOKEN_ID)
        self.assertEqual(tuple(input_ids.shape), (1, 1 + DUMMY_INPUT_LENGTH * REPET_TIMES))
        self.assertEqual(adapter.generate_model_forward.__func__, original.__func__)


if __name__ == "__main__":
    unittest.main()
