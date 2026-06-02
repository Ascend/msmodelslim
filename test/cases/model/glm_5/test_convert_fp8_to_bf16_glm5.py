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
from unittest.mock import patch, MagicMock

import torch
from torch import nn

from msmodelslim.model.glm_5.convert_fp8_to_bf16 import (
    weight_dequant,
    get_inv_weight_map,
    get_inv_tensor,
    auto_convert_module_fp8_to_bf16,
    convert_module_fp8_to_bf16,
    WEIGHT_SCALE_INV,
)


class TestWeightDequant(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_weight_dequant_output_shouldMatchShape_when_weightAndScaleMatch(self):
        m, n, block_size = 256, 256, 128
        weight = torch.randn(m, n)
        scale = torch.randn(m // block_size, n // block_size).abs()

        result = weight_dequant(weight, scale, block_size)
        self.assertEqual(result.shape, (m, n))
        self.assertEqual(result.dtype, torch.bfloat16)

    def test_weight_dequant_output_shouldHandle_when_weightNotMultipleOfBlock(self):
        m, n, block_size = 300, 200, 128
        weight = torch.randn(m, n)
        scale = torch.randn(m // block_size + 1, n // block_size + 1).abs()

        result = weight_dequant(weight, scale, block_size)
        self.assertEqual(result.shape, (m, n))
        self.assertEqual(result.dtype, torch.bfloat16)

    def test_weight_dequant_output_shouldBeBf16_when_inputIsFp32(self):
        m, n, block_size = 128, 128, 64
        weight = torch.randn(m, n, dtype=torch.float32)
        scale = torch.randn(m // block_size, n // block_size, dtype=torch.float32).abs()

        result = weight_dequant(weight, scale, block_size)
        self.assertEqual(result.dtype, torch.bfloat16)

    def test_weight_dequant_shouldScaleUp_when_scaleIsGreaterThanOne(self):
        m, n, block_size = 128, 128, 64
        weight = torch.ones(m, n)
        scale = torch.full((m // block_size, n // block_size), 2.0)

        result = weight_dequant(weight, scale, block_size)
        self.assertTrue(torch.all(result == 2.0))

    def test_weight_dequant_shouldHandle_when_blockSizeIsOne(self):
        m, n = 4, 4
        weight = torch.randn(m, n)
        scale = torch.randn(m, n).abs()

        result = weight_dequant(weight, scale, block_size=1)
        self.assertEqual(result.shape, (m, n))


class TestGetInvWeightMap(unittest.TestCase):
    def setUp(self):
        self.model_path = "/fake/model/path"
        get_inv_weight_map.cache_clear()

    @patch("msmodelslim.model.glm_5.convert_fp8_to_bf16.json_safe_load")
    @patch("msmodelslim.model.glm_5.convert_fp8_to_bf16.os.path.join")
    def test_get_inv_weight_map_shouldReturnFilteredMap_when_indexFileExists(self, mock_join, mock_json_load):
        mock_join.return_value = "/fake/model/path/model.safetensors.index.json"
        mock_json_load.return_value = {
            "weight_map": {
                "layer1.weight": "file1.safetensors",
                "layer1.weight_scale_inv": "file1.safetensors",
                "layer2.weight": "file2.safetensors",
                "layer2.weight_scale_inv": "file2.safetensors",
            }
        }

        result = get_inv_weight_map(self.model_path)

        expected = {
            "layer1": "file1.safetensors",
            "layer2": "file2.safetensors",
        }
        self.assertEqual(result, expected)

    @patch("msmodelslim.model.glm_5.convert_fp8_to_bf16.json_safe_load")
    @patch("msmodelslim.model.glm_5.convert_fp8_to_bf16.os.path.join")
    def test_get_inv_weight_map_shouldReturnEmptyDict_when_noScaleInvKeys(self, mock_join, mock_json_load):
        mock_join.return_value = "/fake/model/path/model.safetensors.index.json"
        mock_json_load.return_value = {
            "weight_map": {
                "layer1.weight": "file1.safetensors",
                "layer2.weight": "file2.safetensors",
            }
        }

        result = get_inv_weight_map(self.model_path)
        self.assertEqual(result, {})

    @patch("msmodelslim.model.glm_5.convert_fp8_to_bf16.json_safe_load")
    @patch("msmodelslim.model.glm_5.convert_fp8_to_bf16.os.path.join")
    def test_get_inv_weight_map_shouldBeCached_when_calledMultipleTimes(self, mock_join, mock_json_load):
        mock_join.return_value = "/fake/model/path/model.safetensors.index.json"
        mock_json_load.return_value = {
            "weight_map": {
                "layer1.weight_scale_inv": "file1.safetensors",
            }
        }

        result1 = get_inv_weight_map(self.model_path)
        result2 = get_inv_weight_map(self.model_path)

        self.assertEqual(result1, result2)
        self.assertEqual(mock_json_load.call_count, 1)


class TestGetInvTensor(unittest.TestCase):
    def setUp(self):
        self.model_path = "/fake/model/path"
        self.weight_map = {
            "layer.weight": "model-00001.safetensors",
        }

    @patch("msmodelslim.model.glm_5.convert_fp8_to_bf16.safe_open")
    @patch("msmodelslim.model.glm_5.convert_fp8_to_bf16.get_valid_read_path")
    @patch("msmodelslim.model.glm_5.convert_fp8_to_bf16.os.path.join")
    @patch("msmodelslim.model.glm_5.convert_fp8_to_bf16.dist.is_initialized")
    def test_get_inv_tensor_shouldReturnTensor_when_singleProcess(
        self, mock_dist_init, mock_join, mock_read_path, mock_safe_open
    ):
        mock_dist_init.return_value = False
        mock_join.return_value = "/fake/model/path/model-00001.safetensors"
        mock_read_path.return_value = "/fake/model/path/model-00001.safetensors"

        mock_file = MagicMock()
        mock_tensor = torch.randn(256, 256)
        mock_file.get_tensor.return_value = mock_tensor
        mock_safe_open.return_value.__enter__.return_value = mock_file

        tensor_name = "layer.weight"
        result = get_inv_tensor(tensor_name, self.model_path, self.weight_map)

        mock_file.get_tensor.assert_called_once_with(tensor_name + WEIGHT_SCALE_INV)
        self.assertIs(result, mock_tensor)

    @patch("msmodelslim.model.glm_5.convert_fp8_to_bf16.get_valid_read_path")
    @patch("msmodelslim.model.glm_5.convert_fp8_to_bf16.os.path.join")
    def test_get_inv_tensor_shouldRaiseError_when_fileNotFound(self, mock_join, mock_read_path):
        mock_join.return_value = "/fake/model/path/missing.safetensors"
        mock_read_path.side_effect = FileNotFoundError("File not found")

        with self.assertRaises(FileNotFoundError):
            get_inv_tensor("layer.weight", self.model_path, self.weight_map)


class TestAutoConvertModuleFp8ToBf16(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.model_path = "/fake/model/path"

    @patch("msmodelslim.model.glm_5.convert_fp8_to_bf16.get_inv_weight_map")
    @patch("msmodelslim.model.glm_5.convert_fp8_to_bf16.convert_module_fp8_to_bf16")
    def test_auto_convert_module_fp8_to_bf16_shouldCallConvert_when_weightMapNotEmpty(self, mock_convert, mock_get_map):
        mock_get_map.return_value = {"sub_module.weight": "file.safetensors"}

        module = nn.Linear(128, 256)
        auto_convert_module_fp8_to_bf16("test_module", module, self.model_path)

        mock_get_map.assert_called_once_with(self.model_path)
        mock_convert.assert_called_once()

    @patch("msmodelslim.model.glm_5.convert_fp8_to_bf16.get_inv_weight_map")
    @patch("msmodelslim.model.glm_5.convert_fp8_to_bf16.convert_module_fp8_to_bf16")
    @patch("msmodelslim.model.glm_5.convert_fp8_to_bf16.get_logger")
    def test_auto_convert_module_fp8_to_bf16_shouldLogWarning_when_keyErrorInSubMap(
        self, mock_logger, mock_convert, mock_get_map
    ):
        mock_get_map.return_value = {"sub_module.weight": "file.safetensors"}

        module = nn.Linear(128, 256)

        def key_error_side_effect(*args, **kwargs):
            raise KeyError("mismatch")

        mock_convert.side_effect = key_error_side_effect

        auto_convert_module_fp8_to_bf16("test_module", module, self.model_path)

        mock_logger.return_value.warning.assert_any_call(
            'Safetensors files not match index.json, please check whether model is of bf16.'
        )
        mock_logger.return_value.warning.assert_any_call('Skip fp8 to bf16.')

    @patch("msmodelslim.model.glm_5.convert_fp8_to_bf16.get_inv_weight_map")
    @patch("msmodelslim.model.glm_5.convert_fp8_to_bf16.convert_module_fp8_to_bf16")
    def test_auto_convert_module_fp8_to_bf16_shouldSkip_when_weightMapEmpty(self, mock_convert, mock_get_map):
        mock_get_map.return_value = {}

        module = nn.Linear(128, 256)
        auto_convert_module_fp8_to_bf16("test_module", module, self.model_path)

        mock_convert.assert_not_called()

    @patch("msmodelslim.model.glm_5.convert_fp8_to_bf16.get_inv_weight_map")
    @patch("msmodelslim.model.glm_5.convert_fp8_to_bf16.convert_module_fp8_to_bf16")
    def test_auto_convert_module_fp8_to_bf16_shouldFilterSubMap_when_moduleHasNoMatchingSubNames(
        self, mock_convert, mock_get_map
    ):
        mock_get_map.return_value = {"other_module.weight": "file.safetensors"}

        module = nn.Linear(128, 256)
        auto_convert_module_fp8_to_bf16("test_module", module, self.model_path)

        mock_convert.assert_called_once()
        call_kwargs = mock_convert.call_args[1]
        self.assertEqual(call_kwargs["weight_map"], {})


class TestConvertModuleFp8ToBf16(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.model_path = "/fake/model/path"

    @patch("msmodelslim.model.glm_5.convert_fp8_to_bf16.get_inv_tensor")
    @patch("msmodelslim.model.glm_5.convert_fp8_to_bf16.weight_dequant")
    def test_convert_module_fp8_to_bf16_shouldDequantWeights_when_subNameInMap(self, mock_dequant, mock_get_inv):
        module = nn.Linear(256, 256)
        mock_scale = torch.randn(256, 256)
        mock_get_inv.return_value = mock_scale
        mock_dequant.return_value = torch.randn(256, 256)

        weight_map = {"test_module": "file.safetensors"}

        convert_module_fp8_to_bf16("test_module", module, self.model_path, weight_map)

        mock_get_inv.assert_called_once_with("test_module", self.model_path, weight_map)
        mock_dequant.assert_called_once()

    @patch("msmodelslim.model.glm_5.convert_fp8_to_bf16.get_inv_tensor")
    def test_convert_module_fp8_to_bf16_shouldSkip_when_subNameNotInMap(self, mock_get_inv):
        module = nn.Linear(256, 256)
        weight_map = {"other_module": "file.safetensors"}

        convert_module_fp8_to_bf16("test_module", module, self.model_path, weight_map)

        mock_get_inv.assert_not_called()

    @patch("msmodelslim.model.glm_5.convert_fp8_to_bf16.get_inv_tensor")
    def test_convert_module_fp8_to_bf16_shouldHandle_when_emptyWeightMap(self, mock_get_inv):
        module = nn.Linear(256, 256)

        convert_module_fp8_to_bf16("test_module", module, self.model_path, {})

        mock_get_inv.assert_not_called()


if __name__ == '__main__':
    unittest.main()
