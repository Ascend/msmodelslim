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

import sys
import unittest
from typing import Optional
from unittest.mock import MagicMock
from torch import nn
from pydantic import BaseModel
from importlib import import_module

from msmodelslim.utils.exception import SchemaValidateError

# The class under test is imported lazily in setUp() and its symbols are injected into
# module globals (see globals()[var] = getattr(...)); pylint cannot resolve them statically.
# pylint: disable=undefined-variable,global-variable-undefined
# ruff: noqa: F821


# Mocking imports from msmodelslim package
class MockAntiOutlierConfig(BaseModel):
    dev_type: str
    dev_id: int
    anti_method: Optional[str] = None
    flex_config: Optional[dict] = None
    disable_anti_names: Optional[list] = None

    def setup_flex_config(self, flex_config):
        self.flex_config = flex_config
        return flex_config


class MockAntiOutlier:
    def __init__(self, model, calib_data, anti_cfg, norm_class_name=None):
        self.model = model
        self.calib_data = calib_data
        self.anti_cfg = anti_cfg
        self.norm_class_name = norm_class_name

    def process(self):
        pass


class MockQuantConfig(BaseModel):
    dev_type: str
    dev_id: int
    is_dynamic: bool
    mm_tensor: bool
    act_method: int
    disable_names: list
    timestep_sep: int = 0

    def fa_quant(self):
        return self

    def timestep_quant(self, timestep_sep):
        self.timestep_sep = timestep_sep
        return self


class MockCalibrator:
    def __init__(self, model, quant_cfg, calib_data):
        self.model = model
        self.quant_cfg = quant_cfg
        self.calib_data = calib_data

    def run(self):
        pass

    def save(self, output_path, safetensors_name=None, json_name=None, save_type=None, part_file_size=None):
        pass


class TestMultiModal_SD_Quant_Session(unittest.TestCase):
    def setUp(self):
        # 1.备份模块
        self._original_module = {}
        self.modules_to_backup = [
            'msmodelslim.pytorch.llm_ptq',
            'msmodelslim.pytorch.llm_ptq.llm_ptq_tools',
            'msmodelslim.pytorch.llm_ptq.anti_outlier',
            'msmodelslim.pytorch.llm_ptq.anti_outlier.anti_utils',
        ]

        # 备份原始模块（如果存在）
        for module in self.modules_to_backup:
            if module in sys.modules:
                self._original_module[module] = sys.modules[module]
            else:
                self._original_module[module] = None

        # 2.mock模块
        # 父模块
        self.llm_ptq_mock = MagicMock()
        sys.modules['msmodelslim.pytorch.llm_ptq'] = self.llm_ptq_mock
        # 子模块 'anti_outlier'
        self.llm_ptq_mock.anti_outlier = MagicMock(
            AntiOutlierConfig=MockAntiOutlierConfig, AntiOutlier=MagicMock(spec=MockAntiOutlier)
        )
        # 子模块 'llm_ptq_tools'
        self.llm_ptq_mock.llm_ptq_tools = MagicMock(
            QuantConfig=MockQuantConfig, Calibrator=MagicMock(spec=MockCalibrator)
        )

        # 深层模块 'anti_utils'
        self.llm_ptq_mock.anti_outlier.anti_utils = MagicMock()

        # 3.替换 sys.modules
        sys.modules['msmodelslim.pytorch.llm_ptq'] = self.llm_ptq_mock
        sys.modules['msmodelslim.pytorch.llm_ptq.llm_ptq_tools'] = self.llm_ptq_mock.llm_ptq_tools
        sys.modules['msmodelslim.pytorch.llm_ptq.anti_outlier'] = self.llm_ptq_mock.anti_outlier
        sys.modules['msmodelslim.pytorch.llm_ptq.anti_outlier.anti_utils'] = self.llm_ptq_mock.anti_outlier.anti_utils

        global llm_ptq_mock
        llm_ptq_mock = self.llm_ptq_mock

        self.seesion_module = import_module('msmodelslim.quant.session.session')
        global_vars = [
            'process_session_cfg',
            'quant_model',
            'SessionConfig',
            'M3ProcessorConfig',
            'M4ProcessorConfig',
            'M6Config',
            'M6ProcessorConfig',
            'W8A8QuantConfig',
            'W8A8ProcessorConfig',
            'FA3ProcessorConfig',
            'W8A8DynamicQuantConfig',
            'W8A8DynamicProcessorConfig',
            'W8A8TimeStepQuantConfig',
            'W8A8TimeStepProcessorConfig',
            'SaveProcessorConfig',
            'M3',
            'M4',
            'M6',
            'W8A8',
            'FA3',
            'W8A8_DYNAMIC',
            'W8A8_TIMESTEP',
            'SAVE',
        ]
        for var in global_vars:
            globals()[var] = getattr(self.seesion_module, var)

    def tearDown(self):
        # 恢复 sys.modules
        for module in self.modules_to_backup:
            if self._original_module[module] is not None:
                sys.modules[module] = self._original_module[module]
            else:
                del sys.modules[module]

        # 清理测试导入的模块
        if 'msmodelslim.quant.session.session' in sys.modules:
            del sys.modules['msmodelslim.quant.session.session']

        # 清理全局变量
        global_vars = [
            'process_session_cfg',
            'quant_model',
            'SessionConfig',
            'M3ProcessorConfig',
            'M4ProcessorConfig',
            'M6Config',
            'M6ProcessorConfig',
            'W8A8QuantConfig',
            'W8A8ProcessorConfig',
            'FA3ProcessorConfig',
            'W8A8DynamicQuantConfig',
            'W8A8DynamicProcessorConfig',
            'W8A8TimeStepQuantConfig',
            'W8A8TimeStepProcessorConfig',
            'SaveProcessorConfig',
            'M3',
            'M4',
            'M6',
            'W8A8',
            'FA3',
            'W8A8_DYNAMIC',
            'W8A8_TIMESTEP',
            'SAVE',
        ]
        for var in global_vars:
            globals()[var] = None

    def test_process_session_cfg_with_m3_anti_outlier(self):
        # Arrange
        processor_cfg_map = {M3: M3ProcessorConfig()}
        session_cfg = SessionConfig(processor_cfg_map=processor_cfg_map, device='npu')
        expected_anti_cfg = MockAntiOutlierConfig(dev_type='npu', dev_id=0)
        expected_anti_cfg.anti_method = M3

        # Act
        result = process_session_cfg(session_cfg, device_id=0)

        # Assert
        self.assertEqual(result[0].dev_type, expected_anti_cfg.dev_type)
        self.assertEqual(result[0].dev_id, expected_anti_cfg.dev_id)
        self.assertEqual(result[0].anti_method, expected_anti_cfg.anti_method)
        self.assertIsNone(result[1])
        self.assertIsNone(result[2])

    def test_process_session_cfg_with_m4_anti_outlier(self):
        # Arrange
        processor_cfg_map = {M4: M4ProcessorConfig()}
        session_cfg = SessionConfig(processor_cfg_map=processor_cfg_map, device='npu')
        expected_anti_cfg = MockAntiOutlierConfig(dev_type='npu', dev_id=0)
        expected_anti_cfg.anti_method = M4

        # Act
        result = process_session_cfg(session_cfg, device_id=0)

        # Assert
        self.assertEqual(result[0].dev_type, expected_anti_cfg.dev_type)
        self.assertEqual(result[0].dev_id, expected_anti_cfg.dev_id)
        self.assertEqual(result[0].anti_method, expected_anti_cfg.anti_method)
        self.assertIsNone(result[1])
        self.assertIsNone(result[2])

    def test_process_session_cfg_with_m6_anti_outlier(self):
        # Arrange
        m6_config = M6Config(alpha=0.5, beta=0.7)
        m6_processor_config = M6ProcessorConfig(cfg=m6_config)
        processor_cfg_map = {M6: m6_processor_config}
        session_cfg = SessionConfig(processor_cfg_map=processor_cfg_map, device='npu')
        expected_anti_cfg = MockAntiOutlierConfig(dev_type='npu', dev_id=0)
        expected_anti_cfg.anti_method = M6
        expected_anti_cfg.flex_config = {'alpha': 0.5, 'beta': 0.7}

        # Act
        result = process_session_cfg(session_cfg, device_id=0)

        # Assert
        self.assertEqual(result[0].dev_type, expected_anti_cfg.dev_type)
        self.assertEqual(result[0].dev_id, expected_anti_cfg.dev_id)
        self.assertEqual(result[0].anti_method, expected_anti_cfg.anti_method)
        self.assertEqual(result[0].flex_config, expected_anti_cfg.flex_config)
        self.assertIsNone(result[1])
        self.assertIsNone(result[2])

    def test_process_session_cfg_with_w8a8_quantization(self):
        # Arrange
        w8a8_config = W8A8QuantConfig(act_method='minmax')
        w8a8_processor_config = W8A8ProcessorConfig(cfg=w8a8_config, disable_names=['layer1'])
        processor_cfg_map = {W8A8: w8a8_processor_config}
        session_cfg = SessionConfig(processor_cfg_map=processor_cfg_map, device='npu')
        expected_quant_cfg = MockQuantConfig(
            dev_type='npu', dev_id=0, is_dynamic=False, mm_tensor=False, act_method=1, disable_names=['layer1']
        )

        # Act
        result = process_session_cfg(session_cfg, device_id=0)

        # Assert
        self.assertIsNone(result[0])
        self.assertEqual(result[1].dev_type, expected_quant_cfg.dev_type)
        self.assertEqual(result[1].dev_id, expected_quant_cfg.dev_id)
        self.assertEqual(result[1].is_dynamic, expected_quant_cfg.is_dynamic)
        self.assertEqual(result[1].mm_tensor, expected_quant_cfg.mm_tensor)
        self.assertEqual(result[1].act_method, expected_quant_cfg.act_method)
        self.assertEqual(result[1].disable_names, expected_quant_cfg.disable_names)
        self.assertIsNone(result[2])

    def test_process_session_cfg_with_fa3_quantization(self):
        # Arrange
        w8a8_dynamic_config = W8A8DynamicQuantConfig(act_method='histogram')
        w8a8_dynamic_processor_config = W8A8DynamicProcessorConfig(cfg=w8a8_dynamic_config, disable_names=['layer2'])
        fa3_processor_config = FA3ProcessorConfig()
        processor_cfg_map = {W8A8_DYNAMIC: w8a8_dynamic_processor_config, FA3: fa3_processor_config}
        session_cfg = SessionConfig(processor_cfg_map=processor_cfg_map, device='npu')
        expected_quant_cfg = MockQuantConfig(
            dev_type='npu', dev_id=0, is_dynamic=True, mm_tensor=False, act_method=2, disable_names=['layer2']
        )

        # Act
        result = process_session_cfg(session_cfg, device_id=0)

        # Assert
        self.assertIsNone(result[0])
        self.assertEqual(result[1].dev_type, expected_quant_cfg.dev_type)
        self.assertEqual(result[1].dev_id, expected_quant_cfg.dev_id)
        self.assertEqual(result[1].is_dynamic, expected_quant_cfg.is_dynamic)
        self.assertEqual(result[1].mm_tensor, expected_quant_cfg.mm_tensor)
        self.assertEqual(result[1].act_method, expected_quant_cfg.act_method)
        self.assertEqual(result[1].disable_names, expected_quant_cfg.disable_names)
        self.assertIsNone(result[2])

    def test_process_session_cfg_with_w8a8_timestep_quantization(self):
        # Arrange
        w8a8_timestep_config = W8A8TimeStepQuantConfig(act_method='minmax')
        w8a8_timestep_processor_config = W8A8TimeStepProcessorConfig(
            cfg=w8a8_timestep_config, disable_names=['layer3'], timestep_sep=10
        )
        processor_cfg_map = {W8A8_TIMESTEP: w8a8_timestep_processor_config}
        session_cfg = SessionConfig(processor_cfg_map=processor_cfg_map, device='npu')
        expected_quant_cfg = MockQuantConfig(
            dev_type='npu', dev_id=0, is_dynamic=False, mm_tensor=False, act_method=1, disable_names=['layer3']
        )
        expected_quant_cfg.timestep_sep = 10

        # Act
        result = process_session_cfg(session_cfg, device_id=0)

        # Assert
        self.assertIsNone(result[0])
        self.assertEqual(result[1].dev_type, expected_quant_cfg.dev_type)
        self.assertEqual(result[1].dev_id, expected_quant_cfg.dev_id)
        self.assertEqual(result[1].is_dynamic, expected_quant_cfg.is_dynamic)
        self.assertEqual(result[1].mm_tensor, expected_quant_cfg.mm_tensor)
        self.assertEqual(result[1].act_method, expected_quant_cfg.act_method)
        self.assertEqual(result[1].disable_names, expected_quant_cfg.disable_names)
        self.assertEqual(result[1].timestep_sep, expected_quant_cfg.timestep_sep)
        self.assertIsNone(result[2])

    def test_process_session_cfg_with_save(self):
        # Arrange
        save_processor_config = SaveProcessorConfig(output_path='/path/to/save')
        processor_cfg_map = {SAVE: save_processor_config}
        session_cfg = SessionConfig(processor_cfg_map=processor_cfg_map, device='npu')
        expected_save_cfg = save_processor_config

        # Act
        result = process_session_cfg(session_cfg, device_id=0)

        # Assert
        self.assertIsNone(result[0])
        self.assertIsNone(result[1])
        self.assertEqual(result[2], expected_save_cfg)

    def test_process_session_cfg_merges_quant_disable_names_when_m6_and_w8a8_combined(self):
        # Arrange
        m6_processor_config = M6ProcessorConfig(cfg=M6Config(alpha=0.5, beta=0.7))
        w8a8_processor_config = W8A8ProcessorConfig(
            cfg=W8A8QuantConfig(act_method='minmax'), disable_names=['layer1', 'layer2']
        )
        processor_cfg_map = {M6: m6_processor_config, W8A8: w8a8_processor_config}
        session_cfg = SessionConfig(processor_cfg_map=processor_cfg_map, device='npu')

        # Act
        result = process_session_cfg(session_cfg, device_id=0)

        # Assert
        self.assertEqual(result[0].disable_anti_names, ['layer1', 'layer2'])

    def test_process_session_cfg_raises_when_fa3_missing_dynamic_quantization(self):
        # Arrange
        processor_cfg_map = {FA3: FA3ProcessorConfig()}
        session_cfg = SessionConfig(processor_cfg_map=processor_cfg_map, device='npu')

        # Act & Assert
        with self.assertRaises(SchemaValidateError):
            process_session_cfg(session_cfg, device_id=0)

    def test_process_session_cfg_with_invalid_processor(self):
        # Arrange
        invalid_processor_config = M3ProcessorConfig()
        processor_cfg_map = {'invalid': invalid_processor_config}
        session_cfg = SessionConfig(processor_cfg_map=processor_cfg_map, device='npu')

        # Act & Assert
        with self.assertRaises(SchemaValidateError) as context:
            process_session_cfg(session_cfg, device_id=0)
        self.assertTrue("The processor_cfg_map in session_config is not supported" in str(context.exception))

    def test_quant_model_with_anti_outlier_and_quantization(self):
        # Arrange
        class DummyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.device = MagicMock(index=0)

        dummy_model = DummyModel()
        m6_config = M6Config(alpha=0.5, beta=0.7)
        m6_processor_config = M6ProcessorConfig(cfg=m6_config)
        w8a8_config = W8A8QuantConfig(act_method='minmax')
        w8a8_processor_config = W8A8ProcessorConfig(cfg=w8a8_config, disable_names=['layer1'])
        save_processor_config = SaveProcessorConfig(output_path='/path/to/save')
        processor_cfg_map = {M6: m6_processor_config, W8A8: w8a8_processor_config, SAVE: save_processor_config}

        # 重置 mock 调用计数
        llm_ptq_mock.anti_outlier.AntiOutlier.reset_mock()
        llm_ptq_mock.llm_ptq_tools.Calibrator.reset_mock()

        session_cfg = SessionConfig(processor_cfg_map=processor_cfg_map, device='npu', calib_data=[1, 2, 3])

        # Act
        quant_model(dummy_model, session_cfg)

        # Assert
        # 1. 检查 AntiOutlier
        llm_ptq_mock.anti_outlier.AntiOutlier.assert_called_once()
        anti_call_args = llm_ptq_mock.anti_outlier.AntiOutlier.call_args
        self.assertEqual(anti_call_args[0][0], dummy_model)
        self.assertEqual(anti_call_args[0][1], [1, 2, 3])

        # 2. 检查 Calibrator
        llm_ptq_mock.llm_ptq_tools.Calibrator.assert_called_once()
        cal_call_args = llm_ptq_mock.llm_ptq_tools.Calibrator.call_args
        self.assertEqual(cal_call_args[0][0], dummy_model)
        self.assertEqual(cal_call_args[0][2], [1, 2, 3])

        # 3. 检查方法调用
        calibrator_instance = llm_ptq_mock.llm_ptq_tools.Calibrator.return_value
        calibrator_instance.run.assert_called_once()
        calibrator_instance.save.assert_called_once_with(
            output_path='/path/to/save',
            safetensors_name=None,
            json_name=None,
            save_type=['safe_tensor'],
            part_file_size=None,
        )

    def test_quant_model_without_save(self):
        # Arrange
        class DummyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.device = MagicMock(index=0)

        dummy_model = DummyModel()
        m6_config = M6Config(alpha=0.5, beta=0.7)
        m6_processor_config = M6ProcessorConfig(cfg=m6_config)
        w8a8_config = W8A8QuantConfig(act_method='minmax')
        w8a8_processor_config = W8A8ProcessorConfig(cfg=w8a8_config, disable_names=['layer1'])
        processor_cfg_map = {M6: m6_processor_config, W8A8: w8a8_processor_config}

        # 重置 mock 调用计数
        llm_ptq_mock.anti_outlier.AntiOutlier.reset_mock()
        llm_ptq_mock.llm_ptq_tools.Calibrator.reset_mock()

        session_cfg = SessionConfig(processor_cfg_map=processor_cfg_map, device='npu', calib_data=[1, 2, 3])

        # Act
        quant_model(dummy_model, session_cfg)

        # Assert
        # 1. 检查 AntiOutlier
        llm_ptq_mock.anti_outlier.AntiOutlier.assert_called_once()
        anti_call_args = llm_ptq_mock.anti_outlier.AntiOutlier.call_args
        self.assertEqual(anti_call_args[0][0], dummy_model)
        self.assertEqual(anti_call_args[0][1], [1, 2, 3])

        # 2. 检查 Calibrator
        llm_ptq_mock.llm_ptq_tools.Calibrator.assert_called_once()
        cal_call_args = llm_ptq_mock.llm_ptq_tools.Calibrator.call_args
        self.assertEqual(cal_call_args[0][0], dummy_model)
        self.assertEqual(cal_call_args[0][2], [1, 2, 3])

        # 3. 检查方法调用
        calibrator_instance = llm_ptq_mock.llm_ptq_tools.Calibrator.return_value
        calibrator_instance.run.assert_called_once()
        calibrator_instance.save.assert_not_called()


if __name__ == '__main__':
    unittest.main()
