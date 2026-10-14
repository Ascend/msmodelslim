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
# pylint: disable=unspecified-encoding

import json
import os
import unittest
from unittest.mock import MagicMock, patch

import torch

from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.llm_ptq_utils import QuantType, SAVE_TYPE_SAFE_TENSOR, SAVE_TYPE_ASCENDV1
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.save.saver.ascend_v1 import AscendV1SaverConfig, AscendV1Saver
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.save.saver.factory import SaverFactory
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.save.saver.safetensors import SafetensorsSaverConfig, SafetensorsSaver
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.save.writer import (
    BufferedSafetensorsWriter,
    JsonDescriptionWriter,
    SafetensorsWriter,
)


class TestAscendV1ConfigValidation(unittest.TestCase):
    """Test suite for validating AscendV1SaverConfig functionality"""

    def setUp(self):
        self.base_config = {
            'output_dir': './test_output',
            'model_quant_type': QuantType.W8A8,
            'use_kvcache_quant': False,
            'use_fa_quant': False,
            'safetensors_name': 'test.safetensors',
            'json_name': 'test.json',
            'part_file_size': None,
            'group_size': 128,
        }

    def test_config_creation_with_valid_params_should_succeed(self):
        """Test that creating config with valid parameters succeeds and sets correct values"""
        config = AscendV1SaverConfig(**self.base_config)
        self.assertEqual(config.output_dir, './test_output')
        self.assertEqual(config.model_quant_type, QuantType.W8A8)
        self.assertEqual(config.safetensors_name, 'test.safetensors')
        self.assertEqual(config.json_name, 'test.json')
        self.assertEqual(config.group_size, 128)

    def test_config_creation_with_none_filenames_should_use_defaults(self):
        """Test that config uses default filenames when none are provided"""
        config_dict = self.base_config.copy()
        config_dict['safetensors_name'] = None
        config_dict['json_name'] = None

        config = AscendV1SaverConfig(**config_dict)
        self.assertEqual(config.safetensors_name, 'quant_model_weight_w8a8.safetensors')
        self.assertEqual(config.json_name, 'quant_model_description.json')

    def test_config_from_dict_with_valid_dict_should_succeed(self):
        """Test that creating config from valid dictionary succeeds"""
        config = AscendV1SaverConfig.from_dict(self.base_config)
        self.assertEqual(config.output_dir, './test_output')
        self.assertEqual(config.group_size, 128)

    def test_config_from_dict_with_invalid_input_should_raise_error(self):
        """Test that creating config from invalid input raises TypeError"""
        with self.assertRaises(TypeError):
            AscendV1SaverConfig.from_dict("invalid_input")


class TestAscendV1SaverBasics(unittest.TestCase):
    """Test suite for basic AscendV1Saver functionality"""

    def setUp(self):
        self.config = AscendV1SaverConfig(
            output_dir='./test_output',
            model_quant_type=QuantType.W8A8,
            use_kvcache_quant=False,
            use_fa_quant=False,
            safetensors_name='test.safetensors',
            json_name='test.json',
            part_file_size=None,
            group_size=128,
        )
        os.makedirs('./test_output', exist_ok=True)

    def tearDown(self):
        # Clean up test files
        if os.path.exists('./test_output'):
            for file in os.listdir('./test_output'):
                os.remove(os.path.join('./test_output', file))
            os.rmdir('./test_output')

    def test_saver_initialization_should_create_correct_writers(self):
        """Test that saver initialization creates appropriate writer instances"""
        saver = AscendV1Saver(self.config)
        self.assertIsInstance(saver.weight_writer, SafetensorsWriter)
        self.assertIsInstance(saver.meta_writer, JsonDescriptionWriter)

    def test_saver_with_part_file_should_use_buffered_writer(self):
        """Test that saver uses BufferedSafetensorsWriter when part_file_size is set"""
        config = self.config
        config.part_file_size = 1  # 1GB
        saver = AscendV1Saver(config)
        self.assertIsInstance(saver.weight_writer, BufferedSafetensorsWriter)

    @patch('msmodelslim.pytorch.llm_ptq.llm_ptq_tools.save.writer.SafetensorsWriter.write')
    @patch('msmodelslim.pytorch.llm_ptq.llm_ptq_tools.save.writer.JsonDescriptionWriter.write')
    def test_save_tensor_should_call_both_writers(self, mock_json_write, mock_safetensors_write):
        """Test that saving tensor calls both weight and meta writers"""
        saver = AscendV1Saver(self.config)
        tensor = torch.tensor([1, 2, 3])

        saver.save('test_tensor', QuantType.W8A8, tensor)

        mock_safetensors_write.assert_called_once()
        mock_json_write.assert_called_once()


class TestSaverFactoryCreation(unittest.TestCase):
    """Test suite for SaverFactory creation functionality"""

    def setUp(self):
        self.factory_kwargs = {
            'output_dir': './test_output',
            'cfg': MagicMock(
                model_quant_type=QuantType.W8A8,
                use_kvcache_quant=False,
                use_fa_quant=False,
                enable_communication_quant=False,
            ),
            'safetensors_name': 'test.safetensors',
            'json_name': 'test.json',
            'part_file_size': None,
            'group_size': 128,
            'enable_communication_quant': False,
        }
        os.makedirs('./test_output', exist_ok=True)

    def tearDown(self):
        if os.path.exists('./test_output'):
            for file in os.listdir('./test_output'):
                os.remove(os.path.join('./test_output', file))
            os.rmdir('./test_output')

    def test_create_ascend_v1_saver_should_return_correct_instance(self):
        """Test that factory creates correct AscendV1Saver instance"""
        saver = SaverFactory.create('ascendV1', **self.factory_kwargs)
        self.assertIsInstance(saver, AscendV1Saver)

    def test_create_multi_saver_with_ascend_v1_should_return_correct_instance(self):
        """Test that factory creates correct instance when using list format"""
        saver = SaverFactory.create(['ascendV1'], **self.factory_kwargs)
        self.assertIsInstance(saver, AscendV1Saver)

    def test_create_invalid_save_type_will_raise_value_error(self):
        """Test that factory creates correct instance when using list format"""
        with self.assertRaises(ValueError):
            SaverFactory.create([[[[[[[[['ascendV1']]]]]]]]], **self.factory_kwargs)


class TestEndToEndSaving(unittest.TestCase):
    """Test suite for end-to-end saving functionality"""

    def setUp(self):
        self.test_dir = './test_output'
        os.makedirs(self.test_dir, exist_ok=True)

        self.test_tensor = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)
        self.test_meta = QuantType.W8A8

        self.config = AscendV1SaverConfig(
            output_dir=self.test_dir,
            model_quant_type=QuantType.W8A8,
            use_kvcache_quant=False,
            use_fa_quant=False,
            safetensors_name='test.safetensors',
            json_name='test.json',
            part_file_size=None,
            group_size=128,
        )

    def tearDown(self):
        if os.path.exists(self.test_dir):
            for file in os.listdir(self.test_dir):
                os.remove(os.path.join(self.test_dir, file))
            os.rmdir(self.test_dir)

    def test_complete_save_process_should_create_all_files(self):
        """Test that complete save process creates all expected files"""
        saver = AscendV1Saver(self.config)

        saver.pre_process()
        saver.save('test_tensor', self.test_meta, self.test_tensor)
        saver.post_process()

        self.assertTrue(os.path.exists(os.path.join(self.test_dir, 'test.safetensors')))
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, 'test.json')))


class TestFormatCompatibility(unittest.TestCase):
    """Test suite for comparing AscendV1 and Safetensors format compatibility"""

    def setUp(self):
        self.test_dir = './test_output'
        os.makedirs(self.test_dir, exist_ok=True)

        self.test_tensor = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)
        self.test_meta = QuantType.W8A8

        # Base configuration for both formats
        self.base_saver_config = {
            'output_dir': self.test_dir,
            'part_file_size': None,
            'json_name': None,
        }

        # AscendV1 configuration
        self.ascend_config = AscendV1SaverConfig(
            **self.base_saver_config,
            safetensors_name='safe_v1.safetensors',
            model_quant_type=QuantType.W8A8,
            group_size=128,
            use_kvcache_quant=False,
            use_fa_quant=False,
            enable_communication_quant=False,
        )

        # Safetensors configuration
        self.safe_config = SafetensorsSaverConfig(
            **self.base_saver_config,
            safetensors_name='safe.safetensors',
            model_quant_type=QuantType.W8A8,
            use_kvcache_quant=False,
            use_fa_quant=False,
            enable_communication_quant=False,
        )

        self.factory_kwargs = {
            'output_dir': './test_output',
            'cfg': MagicMock(
                model_quant_type=QuantType.W8A8,
                use_kvcache_quant=False,
                use_fa_quant=False,
                enable_communication_quant=False,
            ),
            'safetensors_name': 'test.safetensors',
            'json_name': None,
            'part_file_size': None,
            'group_size': 128,
            'enable_communication_quant': False,
        }

    def tearDown(self):
        if os.path.exists(self.test_dir):
            for file in os.listdir(self.test_dir):
                os.remove(os.path.join(self.test_dir, file))
            os.rmdir(self.test_dir)

    def test_default_json_names_should_follow_format_conventions(self):
        """Test that default JSON filenames follow format-specific conventions"""
        # Verify AscendV1 default JSON name
        self.assertEqual(self.ascend_config.json_name, "quant_model_description.json")

        # Verify Safetensors default JSON name
        expected_safe_json_name = f"quant_model_description_{self.safe_config.model_quant_type.lower()}.json"
        self.assertEqual(self.safe_config.json_name, expected_safe_json_name)

    def test_basic_format_differences_should_maintain_compatibility(self):
        """Test format differences while maintaining basic compatibility"""
        # Save in both formats
        ascend_saver = AscendV1Saver(self.ascend_config)
        ascend_saver.pre_process()
        ascend_saver.save('test_tensor', self.test_meta, self.test_tensor)
        ascend_saver.post_process()

        safe_saver = SafetensorsSaver(self.safe_config)
        safe_saver.pre_process()
        safe_saver.save('test_tensor', self.test_meta, self.test_tensor)
        safe_saver.post_process()

        # Verify file paths
        ascend_json_path = os.path.join(self.test_dir, "quant_model_description.json")
        safe_json_path = os.path.join(
            self.test_dir, f"quant_model_description_{self.safe_config.model_quant_type.lower()}.json"
        )

        self.assertTrue(os.path.exists(ascend_json_path))
        self.assertTrue(os.path.exists(safe_json_path))

        # Compare contents
        with open(ascend_json_path, 'r') as f:
            ascend_json = json.load(f)
        with open(safe_json_path, 'r') as f:
            safe_json = json.load(f)

        # Verify AscendV1-specific fields
        self.assertIn('version', ascend_json)
        self.assertEqual(ascend_json['version'], '1.0.0')
        self.assertNotIn('version', safe_json)
        self.assertNotIn('group_size', safe_json)

        # Verify common fields
        self.assertIn('model_quant_type', ascend_json)
        self.assertIn('model_quant_type', safe_json)
        self.assertEqual(ascend_json['model_quant_type'], safe_json['model_quant_type'])

        self.assertIn('test_tensor', ascend_json)
        self.assertIn('test_tensor', safe_json)
        self.assertEqual(ascend_json['test_tensor'], safe_json['test_tensor'])

        self.assertIn('group_size', ascend_json)
        self.assertEqual(ascend_json['group_size'], 128)

    def test_factory_format_selection_should_use_correct_defaults(self):
        """Test that factory creates savers with correct default configurations"""
        # Test AscendV1 format
        ascend_saver = SaverFactory.create(
            SAVE_TYPE_ASCENDV1,
            **{
                **self.factory_kwargs,
                'safetensors_name': 'test.safetensors',
            },
        )
        self.assertIsInstance(ascend_saver, AscendV1Saver)
        self.assertEqual(ascend_saver.meta_writer.json_name, "quant_model_description.json")

        # Test Safetensors format
        safe_saver = SaverFactory.create(
            SAVE_TYPE_SAFE_TENSOR, **{**self.factory_kwargs, 'safetensors_name': 'test.safetensors'}
        )
        self.assertIsInstance(safe_saver, SafetensorsSaver)
        self.assertEqual(
            safe_saver.meta_writer.json_name,
            f"quant_model_description_{self.safe_config.model_quant_type.lower()}.json",
        )

    def test_kv_cache_format_differences_should_maintain_compatibility(self):
        """Test KV cache format differences while maintaining compatibility"""
        self.ascend_config = AscendV1SaverConfig(
            **self.base_saver_config,
            safetensors_name='safe_v1.safetensors',
            model_quant_type=QuantType.KV8,
            group_size=128,
            use_kvcache_quant=True,
            use_fa_quant=False,
            enable_communication_quant=False,
        )

        self.safe_config = SafetensorsSaverConfig(
            **self.base_saver_config,
            safetensors_name='safe.safetensors',
            model_quant_type=QuantType.KV8,
            use_kvcache_quant=True,
            use_fa_quant=False,
            enable_communication_quant=False,
        )

        # Save in both formats
        ascend_saver = AscendV1Saver(self.ascend_config)
        ascend_saver.pre_process()
        ascend_saver.save('test_tensor', self.test_meta, self.test_tensor)
        ascend_saver.post_process()

        safe_saver = SafetensorsSaver(self.safe_config)
        safe_saver.pre_process()
        safe_saver.save('test_tensor', self.test_meta, self.test_tensor)
        safe_saver.post_process()

        # Compare contents
        ascend_json_path = os.path.join(self.test_dir, "quant_model_description.json")
        safe_json_path = os.path.join(
            self.test_dir, f"quant_model_description_{self.safe_config.model_quant_type.lower()}.json"
        )

        with open(ascend_json_path, 'r') as f:
            ascend_json = json.load(f)
        with open(safe_json_path, 'r') as f:
            safe_json = json.load(f)

        # Verify format-specific fields
        self.assertIn('version', ascend_json)
        self.assertEqual(ascend_json['version'], '1.0.0')
        self.assertNotIn('version', safe_json)
        self.assertNotIn('group_size', safe_json)

        # Verify common fields
        self.assertEqual(ascend_json['group_size'], 128)

        self.assertIn('kv_quant_type', safe_json)
        self.assertIn('kv_cache_type', safe_json)
        self.assertIn('kv_quant_type', ascend_json)
        self.assertIn('kv_cache_type', ascend_json)

        # Verify KV-specific fields
        self.assertEqual(ascend_json['model_quant_type'], QuantType.KV8)
        self.assertEqual(ascend_json['model_quant_type'], safe_json['model_quant_type'])
        self.assertEqual(ascend_json['test_tensor'], safe_json['test_tensor'])
        self.assertEqual(ascend_json['kv_quant_type'], QuantType.KV8)
        self.assertEqual(ascend_json['kv_quant_type'], safe_json['kv_quant_type'])
        self.assertEqual(ascend_json['kv_cache_type'], safe_json['kv_cache_type'])
        self.assertEqual(ascend_json['kv_cache_type'], QuantType.KV8)

    def test_fa_format_differences_should_maintain_compatibility(self):
        """Test FA format differences while maintaining compatibility"""
        self.ascend_config = AscendV1SaverConfig(
            **self.base_saver_config,
            safetensors_name='safe_v1.safetensors',
            model_quant_type=QuantType.FAQuant,
            group_size=128,
            use_kvcache_quant=False,
            use_fa_quant=True,
            enable_communication_quant=False,
        )

        self.safe_config = SafetensorsSaverConfig(
            **self.base_saver_config,
            safetensors_name='safe.safetensors',
            model_quant_type=QuantType.FAQuant,
            use_kvcache_quant=False,
            use_fa_quant=True,
            enable_communication_quant=False,
        )

        # Save in both formats
        ascend_saver = AscendV1Saver(self.ascend_config)
        ascend_saver.pre_process()
        ascend_saver.save('test_tensor', self.test_meta, self.test_tensor)
        ascend_saver.post_process()

        safe_saver = SafetensorsSaver(self.safe_config)
        safe_saver.pre_process()
        safe_saver.save('test_tensor', self.test_meta, self.test_tensor)
        safe_saver.post_process()

        # Compare contents
        ascend_json_path = os.path.join(self.test_dir, "quant_model_description.json")
        safe_json_path = os.path.join(
            self.test_dir, f"quant_model_description_{self.safe_config.model_quant_type.lower()}.json"
        )

        with open(ascend_json_path, 'r') as f:
            ascend_json = json.load(f)
        with open(safe_json_path, 'r') as f:
            safe_json = json.load(f)

        # Verify format-specific fields
        self.assertIn('version', ascend_json)
        self.assertEqual(ascend_json['version'], '1.0.0')
        self.assertNotIn('version', safe_json)
        self.assertNotIn('group_size', safe_json)

        # Verify common fields
        self.assertIn('group_size', ascend_json)
        self.assertEqual(ascend_json['group_size'], 128)

        self.assertIn('fa_quant_type', safe_json)
        self.assertIn('fa_quant_type', ascend_json)

        # Verify FA-specific fields
        self.assertEqual(ascend_json['model_quant_type'], QuantType.FAQuant)
        self.assertEqual(ascend_json['model_quant_type'], safe_json['model_quant_type'])
        self.assertEqual(ascend_json['test_tensor'], safe_json['test_tensor'])
        self.assertEqual(ascend_json['fa_quant_type'], QuantType.FAQuant)
        self.assertEqual(ascend_json['fa_quant_type'], safe_json['fa_quant_type'])

    def test_communication_format_differences_should_maintain_compatibility(self):
        """Test communication format differences while maintaining compatibility"""
        self.ascend_config = AscendV1SaverConfig(
            **self.base_saver_config,
            safetensors_name='safe_v1.safetensors',
            model_quant_type=QuantType.W8A8,
            group_size=128,
            use_kvcache_quant=False,
            use_fa_quant=False,
            enable_communication_quant=True,
        )

        self.safe_config = SafetensorsSaverConfig(
            **self.base_saver_config,
            safetensors_name='safe.safetensors',
            model_quant_type=QuantType.W8A8,
            use_kvcache_quant=False,
            use_fa_quant=False,
            enable_communication_quant=True,
        )

        # Save in both formats
        ascend_saver = AscendV1Saver(self.ascend_config)
        ascend_saver.pre_process()
        ascend_saver.save('test_tensor', self.test_meta, self.test_tensor)
        ascend_saver.post_process()

        safe_saver = SafetensorsSaver(self.safe_config)
        safe_saver.pre_process()
        safe_saver.save('test_tensor', self.test_meta, self.test_tensor)
        safe_saver.post_process()

        # Compare contents
        ascend_json_path = os.path.join(self.test_dir, "quant_model_description.json")
        safe_json_path = os.path.join(
            self.test_dir, f"quant_model_description_{self.safe_config.model_quant_type.lower()}.json"
        )

        with open(ascend_json_path, 'r') as f:
            ascend_json = json.load(f)
        with open(safe_json_path, 'r') as f:
            safe_json = json.load(f)

        # Verify format-specific fields
        self.assertIn('version', ascend_json)
        self.assertEqual(ascend_json['version'], '1.0.0')
        self.assertIn('reduce_quant_type', ascend_json)
        self.assertEqual(ascend_json['reduce_quant_type'], 'per_channel')

        self.assertNotIn('version', safe_json)
        self.assertNotIn('group_size', safe_json)

        # Verify common fields
        self.assertIn('group_size', ascend_json)
        self.assertEqual(ascend_json['group_size'], 128)

        # Verify basic fields
        self.assertEqual(ascend_json['model_quant_type'], QuantType.W8A8)
        self.assertEqual(ascend_json['model_quant_type'], safe_json['model_quant_type'])
        self.assertEqual(ascend_json['test_tensor'], safe_json['test_tensor'])


if __name__ == '__main__':
    unittest.main()
