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
from pathlib import Path
from unittest.mock import MagicMock, patch

from msmodelslim.core.const import DeviceType
from msmodelslim.model.transformers.model_adapter import LLMTransformersModel
from msmodelslim.utils.exception import InvalidModelError


class TestLLMTransformersModel(unittest.TestCase):
    def setUp(self):
        self.model_type = "transformers"
        self.model_path = Path(".")

    def _make_adapter(self):
        with patch(
            "msmodelslim.model.transformers.model_adapter.TransformersModel.__init__",
            return_value=None,
        ):
            adapter = LLMTransformersModel(model_type=self.model_type, model_path=self.model_path)
        adapter.model_type = self.model_type
        adapter.model_path = self.model_path
        return adapter

    def test_get_model_type_returns_transformers_when_called(self):
        adapter = self._make_adapter()
        self.assertEqual(adapter.get_model_type(), "transformers")

    def test_get_model_pedigree_returns_llm_transformers_when_called(self):
        adapter = self._make_adapter()
        self.assertEqual(adapter.get_model_pedigree(), "llm_transformers")

    def test_init_model_returns_model_when_load_succeeds(self):
        adapter = self._make_adapter()
        mock_model = MagicMock()
        adapter._load_model = MagicMock(return_value=mock_model)

        result = adapter.init_model(device=DeviceType.NPU)

        self.assertIs(result, mock_model)
        adapter._load_model.assert_called_once_with(DeviceType.NPU)

    def test_init_model_raises_invalid_model_error_when_load_fails(self):
        adapter = self._make_adapter()
        adapter._load_model = MagicMock(side_effect=Exception("Loading failed"))

        with self.assertRaises(InvalidModelError) as ctx:
            adapter.init_model(device=DeviceType.NPU)

        self.assertIn("dedicated model adapter", ctx.exception.action)
        self.assertIn("VLM/DiT", ctx.exception.action)

    def test_handle_dataset_returns_tokenized_data_when_helper_succeeds(self):
        adapter = self._make_adapter()
        mock_dataset = ["data1", "data2"]
        adapter._get_tokenized_data = MagicMock(return_value=mock_dataset)

        result = adapter.handle_dataset(dataset="test_data", device=DeviceType.CPU)

        self.assertEqual(result, mock_dataset)
        adapter._get_tokenized_data.assert_called_once_with("test_data", DeviceType.CPU)

    def test_handle_dataset_raises_invalid_model_error_when_helper_fails(self):
        adapter = self._make_adapter()
        adapter._get_tokenized_data = MagicMock(side_effect=Exception("Processing failed"))

        with self.assertRaises(InvalidModelError):
            adapter.handle_dataset(dataset="test_data", device=DeviceType.CPU)

    @patch("msmodelslim.model.transformers.model_adapter.generated_decoder_layer_visit_func")
    def test_generate_model_visit_yields_requests_when_helper_succeeds(self, mock_visit):
        mock_request = MagicMock()
        mock_visit.return_value = iter([mock_request])
        adapter = self._make_adapter()
        mock_model = MagicMock()

        result = list(adapter.generate_model_visit(mock_model))

        self.assertEqual(result, [mock_request])
        mock_visit.assert_called_once_with(mock_model)

    @patch("msmodelslim.model.transformers.model_adapter.generated_decoder_layer_visit_func")
    def test_generate_model_visit_raises_invalid_model_error_when_helper_fails(self, mock_visit):
        mock_visit.side_effect = Exception("visit failed")
        adapter = self._make_adapter()

        with self.assertRaises(InvalidModelError):
            list(adapter.generate_model_visit(MagicMock()))

    @patch("msmodelslim.model.transformers.model_adapter.transformers_generated_forward_func")
    def test_generate_model_forward_yields_requests_when_helper_succeeds(self, mock_forward):
        mock_request = MagicMock()
        mock_forward.return_value = iter([mock_request])
        adapter = self._make_adapter()
        mock_model = MagicMock()
        mock_inputs = MagicMock()

        result = list(adapter.generate_model_forward(mock_model, mock_inputs))

        self.assertEqual(result, [mock_request])
        mock_forward.assert_called_once_with(mock_model, mock_inputs)

    @patch("msmodelslim.model.transformers.model_adapter.transformers_generated_forward_func")
    def test_generate_model_forward_raises_invalid_model_error_when_helper_fails(self, mock_forward):
        mock_forward.side_effect = Exception("forward failed")
        adapter = self._make_adapter()

        with self.assertRaises(InvalidModelError):
            list(adapter.generate_model_forward(MagicMock(), MagicMock()))

    def test_enable_kv_cache_delegates_to_helper_when_need_kv_cache_true(self):
        adapter = self._make_adapter()
        mock_model = MagicMock()
        adapter._enable_kv_cache = MagicMock(return_value=None)

        adapter.enable_kv_cache(model=mock_model, need_kv_cache=True)

        adapter._enable_kv_cache.assert_called_once_with(mock_model, True)

    def test_enable_kv_cache_delegates_to_helper_when_need_kv_cache_false(self):
        adapter = self._make_adapter()
        mock_model = MagicMock()
        adapter._enable_kv_cache = MagicMock(return_value=None)

        adapter.enable_kv_cache(model=mock_model, need_kv_cache=False)

        adapter._enable_kv_cache.assert_called_once_with(mock_model, False)

    def test_enable_kv_cache_raises_invalid_model_error_when_helper_fails(self):
        adapter = self._make_adapter()
        adapter._enable_kv_cache = MagicMock(side_effect=Exception("Enable KV cache failed"))

        with self.assertRaises(InvalidModelError):
            adapter.enable_kv_cache(model=MagicMock(), need_kv_cache=True)
