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
# pylint: disable=arguments-differ,consider-using-from-import

# pylint: disable=abstract-method

import unittest
from unittest.mock import MagicMock, patch

import torch
from torch import nn
from contextlib import nullcontext

from msmodelslim.processor.flat_quant.flat_quant import (
    FlatQuantProcessorConfig,
    FlatQuantProcessor,
    QuantStrategyConfig,
    npu_available,
)
from msmodelslim.processor.flat_quant.flat_quant_interface import FlatQuantInterface
from msmodelslim.processor.flat_quant.flat_quant_utils.flat_fake_quant_linear import (
    FlatFakeQuantLinear,
    FlatFakeQuantLinearConfig,
    FlatNormWrapper,
)
from msmodelslim.processor.flat_quant.flat_quant_utils.flat_fake_quant_linear import (
    ForwardMode,  # noqa: F401
)  # noqa: F401
from msmodelslim.core.base.protocol import BatchProcessRequest
from msmodelslim.core.quantizer.linear import LinearQConfig
from msmodelslim.core.quantizer.base import QConfig
from msmodelslim.ir.qal import QDType, QScope
from msmodelslim.utils.exception import SchemaValidateError
from msmodelslim.utils.config_map import ConfigSet


def create_linear_qconfig():
    """创建 LinearQConfig 实例"""
    return LinearQConfig(weight=QConfig(dtype=QDType.FLOAT, scope=QScope.PER_TENSOR, symmetric=True, method="none"))


class MockFlatQuantAdapter(FlatQuantInterface):
    """用于测试的 FlatQuantInterface 模拟实现"""

    def get_flatquant_subgraph(self):  # pylint: disable=arguments-differ
        """返回模拟的结构配置"""
        return []


class SimpleModelWithTieWeights(nn.Module):
    """带有 tie_weights 方法的简单模型"""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(32, 32)

    def tie_weights(self):
        """模拟 tie_weights 方法"""
        pass


# ==================== QuantStrategyConfig 测试 ====================


class TestQuantStrategyConfig(unittest.TestCase):
    """测试 QuantStrategyConfig 类 - 量化策略配置"""

    def test_QuantStrategyConfig_include_is_all_after_init_with_defaults(self):
        """QuantStrategyConfig-初始化后默认-include为星号列表"""
        qconfig = create_linear_qconfig()
        strategy = QuantStrategyConfig(qconfig=qconfig)
        self.assertEqual(strategy.include, ["*"])

    def test_QuantStrategyConfig_exclude_is_empty_after_init_with_defaults(self):
        """QuantStrategyConfig-初始化后默认-exclude为空列表"""
        qconfig = create_linear_qconfig()
        strategy = QuantStrategyConfig(qconfig=qconfig)
        self.assertEqual(strategy.exclude, [])

    def test_QuantStrategyConfig_include_equals_input_after_init_with_custom(self):
        """QuantStrategyConfig-初始化后自定义-include与输入一致"""
        qconfig = create_linear_qconfig()
        strategy = QuantStrategyConfig(qconfig=qconfig, include=["layer1", "layer2"], exclude=["layer3"])
        self.assertEqual(strategy.include, ["layer1", "layer2"])
        self.assertEqual(strategy.exclude, ["layer3"])


# ==================== FlatQuantProcessorConfig 测试 ====================


class TestFlatQuantProcessorConfig(unittest.TestCase):
    """测试 FlatQuantProcessorConfig 类 - FlatQuant处理器配置"""

    # 初始化测试
    def test_FlatQuantProcessorConfig_type_is_flatquant_after_init(self):
        """FlatQuantProcessorConfig-初始化后-type为flatquant"""
        config = FlatQuantProcessorConfig()
        self.assertEqual(config.type, "flatquant")

    def test_FlatQuantProcessorConfig_include_is_all_after_init(self):
        """FlatQuantProcessorConfig-初始化后-include为星号列表"""
        config = FlatQuantProcessorConfig()
        self.assertEqual(config.include, ["*"])

    def test_FlatQuantProcessorConfig_exclude_is_empty_after_init(self):
        """FlatQuantProcessorConfig-初始化后-exclude为空列表"""
        config = FlatQuantProcessorConfig()
        self.assertEqual(config.exclude, [])

    def test_FlatQuantProcessorConfig_seed_is_zero_after_init(self):
        """FlatQuantProcessorConfig-初始化后-seed为0"""
        config = FlatQuantProcessorConfig()
        self.assertEqual(config.seed, 0)

    def test_FlatQuantProcessorConfig_quantization_params_correct_after_init(self):
        """FlatQuantProcessorConfig-初始化后-量化参数正确"""
        config = FlatQuantProcessorConfig()
        self.assertEqual(config.a_bits, 4)
        self.assertEqual(config.w_bits, 4)
        self.assertEqual(config.a_groupsize, -1)
        self.assertEqual(config.w_groupsize, -1)
        self.assertFalse(config.a_asym)
        self.assertFalse(config.w_asym)

    def test_FlatQuantProcessorConfig_training_params_correct_after_init(self):
        """FlatQuantProcessorConfig-初始化后-训练参数正确"""
        config = FlatQuantProcessorConfig()
        self.assertEqual(config.epochs, 10)
        self.assertEqual(config.cali_bsz, 4)
        self.assertEqual(config.flat_lr, 1e-3)
        self.assertIsNone(config.nsamples)

    def test_FlatQuantProcessorConfig_transformation_params_correct_after_init(self):
        """FlatQuantProcessorConfig-初始化后-变换参数正确"""
        config = FlatQuantProcessorConfig()
        self.assertTrue(config.add_diag)
        self.assertTrue(config.lwc)
        self.assertTrue(config.lac)
        self.assertEqual(config.diag_init, "one_style")
        self.assertEqual(config.diag_alpha, 0.3)
        self.assertTrue(config.warmup)
        self.assertEqual(config.tran_type, "svd")

    def test_FlatQuantProcessorConfig_model_config_allows_arbitrary_types(self):
        """FlatQuantProcessorConfig-初始化后-模型配置允许任意类型"""
        config = FlatQuantProcessorConfig()
        self.assertTrue(config.model_config.get("arbitrary_types_allowed", False))

    # dtype 属性测试
    def test_FlatQuantProcessorConfig_dtype_is_float32_when_deactive_amp_true(self):
        """FlatQuantProcessorConfig-deactive_amp为True时-dtype为float32"""
        config = FlatQuantProcessorConfig()
        self.assertEqual(config.deactive_amp, True)
        self.assertEqual(config.dtype, torch.float32)

    # traincast 属性测试
    def test_FlatQuantProcessorConfig_traincast_is_nullcontext_when_deactive_amp_true(self):
        """FlatQuantProcessorConfig-deactive_amp为True时-traincast为nullcontext"""
        config = FlatQuantProcessorConfig()
        self.assertEqual(config.traincast, nullcontext)

    # 验证逻辑测试
    def test_FlatQuantProcessorConfig_passes_validation_when_all_fields_default(self):
        """FlatQuantProcessorConfig-所有字段为默认值时-验证通过"""
        config = FlatQuantProcessorConfig()
        self.assertEqual(config.type, "flatquant")

    def test_model_validate_raises_schema_error_when_internal_field_is_supplied(self):
        """FlatQuantProcessorConfig-通过配置字典传入init=False字段时-抛出SchemaValidateError"""
        with self.assertRaisesRegex(SchemaValidateError, "epochs"):
            FlatQuantProcessorConfig.model_validate({"epochs": 20})

    def test_FlatQuantProcessorConfig_passes_validation_when_init_false_fields_not_modified(self):
        """FlatQuantProcessorConfig-init=False字段未被修改时-验证通过"""
        config = FlatQuantProcessorConfig()
        result = config.validate_init_fields()
        self.assertIs(result, config)


# ==================== FlatQuantProcessor 测试 ====================


class TestFlatQuantProcessor(unittest.TestCase):
    """测试 FlatQuantProcessor 类 - FlatQuant处理器"""

    def setUp(self):
        """设置测试环境"""
        self.model = nn.Sequential(nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 32))
        self.config = FlatQuantProcessorConfig()
        self.adapter = MockFlatQuantAdapter()

    # 初始化测试
    def test_FlatQuantProcessor_model_equals_input_after_init(self):
        """FlatQuantProcessor-初始化后-model与输入一致"""
        processor = FlatQuantProcessor(self.model, self.config, self.adapter)
        self.assertIs(processor.model, self.model)

    def test_FlatQuantProcessor_config_equals_input_after_init(self):
        """FlatQuantProcessor-初始化后-config与输入一致"""
        processor = FlatQuantProcessor(self.model, self.config, self.adapter)
        self.assertIs(processor.config, self.config)

    def test_FlatQuantProcessor_adapter_equals_input_after_init(self):
        """FlatQuantProcessor-初始化后-adapter与输入一致"""
        processor = FlatQuantProcessor(self.model, self.config, self.adapter)
        self.assertIs(processor.adapter, self.adapter)

    def test_FlatQuantProcessor_has_strategies_after_init(self):
        """FlatQuantProcessor-初始化后-strategies属性正确"""
        processor = FlatQuantProcessor(self.model, self.config, self.adapter)
        self.assertEqual(processor.strategies, self.config.strategies)

    def test_FlatQuantProcessor_has_trans_include_after_init(self):
        """FlatQuantProcessor-初始化后-trans_include属性正确"""
        processor = FlatQuantProcessor(self.model, self.config, self.adapter)
        self.assertIsNotNone(processor.trans_include)

    def test_FlatQuantProcessor_has_trans_exclude_after_init(self):
        """FlatQuantProcessor-初始化后-trans_exclude属性正确"""
        processor = FlatQuantProcessor(self.model, self.config, self.adapter)
        self.assertIsNotNone(processor.trans_exclude)

    def test_FlatQuantProcessor_has_structure_config_after_init(self):
        """FlatQuantProcessor-初始化后-structure_config属性正确"""
        processor = FlatQuantProcessor(self.model, self.config, self.adapter)
        self.assertIsNotNone(processor.structure_config)

    def test_FlatQuantProcessor_float_output_is_none_after_init(self):
        """FlatQuantProcessor-初始化后-float_output为None"""
        processor = FlatQuantProcessor(self.model, self.config, self.adapter)
        self.assertIsNone(processor.float_output)

    def test_FlatQuantProcessor_has_layer_trainer_after_init(self):
        """FlatQuantProcessor-初始化后-layer_trainer属性正确"""
        processor = FlatQuantProcessor(self.model, self.config, self.adapter)
        self.assertIsNotNone(processor.layer_trainer)

    # need_kv_cache 方法测试
    def test_FlatQuantProcessor_returns_false_when_need_kv_cache(self):
        """FlatQuantProcessor-调用need_kv_cache-返回False"""
        processor = FlatQuantProcessor(self.model, self.config, self.adapter)
        self.assertFalse(processor.need_kv_cache())

    # pre_run 方法测试
    def test_FlatQuantProcessor_model_in_eval_mode_after_pre_run(self):
        """FlatQuantProcessor-调用pre_run后-模型处于eval模式"""
        processor = FlatQuantProcessor(self.model, self.config, self.adapter)
        processor.pre_run()
        self.assertFalse(self.model.training)

    # post_run 方法测试
    def test_FlatQuantProcessor_calls_tie_weights_when_post_run(self):
        """FlatQuantProcessor-调用post_run时-调用model.tie_weights"""
        model = SimpleModelWithTieWeights()
        processor = FlatQuantProcessor(model, self.config, self.adapter)

        original_tie_weights = model.tie_weights
        call_count = [0]

        def spy_tie_weights():
            call_count[0] += 1
            return original_tie_weights()

        model.tie_weights = spy_tie_weights

        processor.post_run()
        self.assertEqual(call_count[0], 1)

    def test_FlatQuantProcessor_raises_AttributeError_when_post_run_without_tie_weights(self):
        """FlatQuantProcessor-调用post_run时-若模型没有tie_weights方法则抛出AttributeError"""
        model = nn.Linear(32, 32)
        processor = FlatQuantProcessor(model, self.config, self.adapter)

        with self.assertRaises(AttributeError):
            processor.post_run()

    # preprocess 方法测试
    def test_FlatQuantProcessor_freezes_parameters_when_preprocess(self):
        """FlatQuantProcessor-调用preprocess时-冻结参数"""
        processor = FlatQuantProcessor(self.model, self.config, self.adapter)
        module = nn.Linear(32, 32)

        request = BatchProcessRequest(name="test_layer", module=module, datas=None, outputs=[])

        with patch.object(processor, '_run_forward_if_need'):
            with patch('msmodelslim.processor.flat_quant.flat_quant.FlatQuantLayerManager') as MockManager:
                mock_manager_instance = MagicMock()
                MockManager.return_value = mock_manager_instance

                processor.preprocess(request)

                for param in module.parameters():
                    self.assertFalse(param.requires_grad)

    def test_FlatQuantProcessor_creates_layer_quantizer_when_preprocess(self):
        """FlatQuantProcessor-调用preprocess时-创建layer_quantizer"""
        processor = FlatQuantProcessor(self.model, self.config, self.adapter)
        module = nn.Linear(32, 32)
        request = BatchProcessRequest(name="test_layer", module=module, datas=None, outputs=[])

        with patch.object(processor, '_run_forward_if_need'):
            with patch('msmodelslim.processor.flat_quant.flat_quant.FlatQuantLayerManager') as MockManager:
                mock_manager_instance = MagicMock()
                MockManager.return_value = mock_manager_instance

                processor.preprocess(request)
                self.assertIsNotNone(processor.layer_quantizer)

    # process 方法测试
    def test_FlatQuantProcessor_calls_train_layer_when_process(self):
        """FlatQuantProcessor-调用process时-调用layer_trainer.train_layer"""
        processor = FlatQuantProcessor(self.model, self.config, self.adapter)
        module = nn.Linear(32, 32)
        request = BatchProcessRequest(name="test_layer", module=module, datas=None, outputs=None)

        processor.float_output = [torch.randn(2, 32)]

        mock_trainer = MagicMock()
        processor.layer_trainer = mock_trainer

        processor.process(request)
        mock_trainer.train_layer.assert_called_once()

    # postprocess 方法测试
    def test_FlatQuantProcessor_freezes_parameters_when_postprocess(self):
        """FlatQuantProcessor-调用postprocess时-冻结参数"""
        processor = FlatQuantProcessor(self.model, self.config, self.adapter)
        module = nn.Linear(32, 32)
        request = BatchProcessRequest(name="test_layer", module=module, datas=None, outputs=None)

        processor.dtype_dict = {}
        processor.trans_include = set(["*"])
        processor.trans_exclude = set()

        mock_layer_quantizer = MagicMock()
        processor.layer_quantizer = mock_layer_quantizer

        with patch.object(processor, 'set_hook_ir'):
            with patch.object(processor, '_rollback_trans'):
                processor.postprocess(request)

                for param in module.parameters():
                    self.assertFalse(param.requires_grad)

    def test_FlatQuantProcessor_calls_change_mode_eval_when_postprocess(self):
        """FlatQuantProcessor-调用postprocess时-调用change_mode(ForwardMode.EVAL)"""
        processor = FlatQuantProcessor(self.model, self.config, self.adapter)
        module = nn.Linear(32, 32)
        request = BatchProcessRequest(name="test_layer", module=module, datas=None, outputs=None)

        processor.dtype_dict = {}
        processor.trans_include = set(["*"])
        processor.trans_exclude = set()

        mock_layer_quantizer = MagicMock()
        processor.layer_quantizer = mock_layer_quantizer

        with patch.object(processor, 'set_hook_ir'):
            with patch.object(processor, '_rollback_trans'):
                processor.postprocess(request)
                mock_layer_quantizer.change_mode.assert_called_once()

    def test_FlatQuantProcessor_restores_dtype_when_postprocess(self):
        """FlatQuantProcessor-调用postprocess时-恢复参数dtype"""
        processor = FlatQuantProcessor(self.model, self.config, self.adapter)
        module = nn.Linear(4, 4)
        original_dtype = module.weight.dtype
        module.weight.data = module.weight.data.to(torch.float16)

        request = BatchProcessRequest(name="test_layer", module=module, datas=None, outputs=None)

        processor.dtype_dict = {'weight': original_dtype, 'bias': original_dtype}
        processor.trans_include = set(["*"])
        processor.trans_exclude = set()

        mock_layer_quantizer = MagicMock()
        processor.layer_quantizer = mock_layer_quantizer

        with patch.object(processor, 'set_hook_ir'):
            with patch.object(processor, '_rollback_trans'):
                processor.postprocess(request)
                self.assertEqual(module.weight.dtype, original_dtype)

    def test_postprocess_preserves_weight_dtype_when_name_not_in_dtype_dict(self):
        """FlatQuantProcessor-调用postprocess时-参数不在dtype_dict中则保持原dtype"""
        processor = FlatQuantProcessor(self.model, self.config, self.adapter)
        module = nn.Linear(4, 4)
        module.weight.data = module.weight.data.to(torch.float16)

        request = BatchProcessRequest(name="test_layer", module=module, datas=None, outputs=None)

        processor.dtype_dict = {}
        processor.trans_include = set(["*"])
        processor.trans_exclude = set()

        mock_layer_quantizer = MagicMock()
        processor.layer_quantizer = mock_layer_quantizer

        with patch.object(processor, 'set_hook_ir'):
            with patch.object(processor, '_rollback_trans'):
                processor.postprocess(request)

        self.assertEqual(module.weight.dtype, torch.float16)

    # postprocess 回滚变换场景测试
    def test_postprocess_does_not_call_rollback_trans_when_no_flat_fake_quant_linear(self):
        """FlatQuantProcessor-调用postprocess时-没有FlatFakeQuantLinear则不调用rollback_trans"""
        processor = FlatQuantProcessor(self.model, self.config, self.adapter)
        module = nn.Sequential(nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 32))
        request = BatchProcessRequest(name="prefix", module=module, datas=None, outputs=None)

        processor.dtype_dict = {}
        processor.trans_include = ConfigSet(["*"])
        processor.trans_exclude = ConfigSet([])

        mock_layer_quantizer = MagicMock()
        processor.layer_quantizer = mock_layer_quantizer

        with patch.object(processor, 'set_hook_ir'):
            processor.postprocess(request)

        mock_layer_quantizer.rollback_trans.assert_not_called()

    def test_postprocess_calls_rollback_trans_when_name_not_in_trans_include(self):
        """FlatQuantProcessor-调用postprocess时-name不在trans_include中则调用rollback_trans"""
        processor = FlatQuantProcessor(self.model, self.config, self.adapter)
        ffq_linear = FlatFakeQuantLinear(FlatFakeQuantLinearConfig(), nn.Linear(4, 4))
        module = nn.Module()
        module.add_module('test_ffq', ffq_linear)
        request = BatchProcessRequest(name="prefix", module=module, datas=None, outputs=None)

        processor.dtype_dict = {}
        processor.trans_include = ConfigSet(['other_name'])
        processor.trans_exclude = ConfigSet([])

        mock_layer_quantizer = MagicMock()
        processor.layer_quantizer = mock_layer_quantizer

        with patch.object(processor, 'set_hook_ir'):
            processor.postprocess(request)

        mock_layer_quantizer.rollback_trans.assert_called_once_with(pair_name='prefix.test_ffq')

    def test_postprocess_calls_rollback_trans_when_name_in_trans_exclude(self):
        """FlatQuantProcessor-调用postprocess时-name在trans_exclude中则调用rollback_trans"""
        processor = FlatQuantProcessor(self.model, self.config, self.adapter)
        ffq_linear = FlatFakeQuantLinear(FlatFakeQuantLinearConfig(), nn.Linear(4, 4))
        module = nn.Module()
        module.add_module('test_ffq', ffq_linear)
        request = BatchProcessRequest(name="prefix", module=module, datas=None, outputs=None)

        processor.dtype_dict = {}
        processor.trans_include = ConfigSet(['*'])
        processor.trans_exclude = ConfigSet(['prefix.test_ffq'])

        mock_layer_quantizer = MagicMock()
        processor.layer_quantizer = mock_layer_quantizer

        with patch.object(processor, 'set_hook_ir'):
            processor.postprocess(request)

        mock_layer_quantizer.rollback_trans.assert_called_once_with(pair_name='prefix.test_ffq')

    def test_postprocess_does_not_call_rollback_trans_when_trans_include_matches_and_exclude_misses(self):
        """FlatQuantProcessor-调用postprocess时-include命中且exclude未命中则不调用rollback_trans"""
        processor = FlatQuantProcessor(self.model, self.config, self.adapter)
        ffq_linear = FlatFakeQuantLinear(FlatFakeQuantLinearConfig(), nn.Linear(4, 4))
        module = nn.Module()
        module.add_module('test_ffq', ffq_linear)
        request = BatchProcessRequest(name="prefix", module=module, datas=None, outputs=None)

        processor.dtype_dict = {}
        processor.trans_include = ConfigSet(['*'])
        processor.trans_exclude = ConfigSet([])

        mock_layer_quantizer = MagicMock()
        processor.layer_quantizer = mock_layer_quantizer

        with patch.object(processor, 'set_hook_ir'):
            processor.postprocess(request)

        mock_layer_quantizer.rollback_trans.assert_not_called()

    # set_hook_ir 方法测试
    def test_FlatQuantProcessor_preserves_module_structure_when_set_hook_ir(self):
        """FlatQuantProcessor-调用set_hook_ir时-保持模块结构"""
        processor = FlatQuantProcessor(self.model, self.config, self.adapter)
        module = nn.Sequential(nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 32))

        original_num_modules = len(list(module.modules()))

        processor.set_hook_ir(module)

        current_num_modules = len(list(module.modules()))
        self.assertEqual(original_num_modules, current_num_modules)

    def test_FlatQuantProcessor_replaces_ffq_linear_when_set_hook_ir(self):
        """FlatQuantProcessor-调用set_hook_ir时-替换FlatFakeQuantLinear为原始Linear"""
        processor = FlatQuantProcessor(self.model, self.config, self.adapter)
        linear = nn.Linear(4, 4)
        ffq_config = FlatFakeQuantLinearConfig()
        ffq_linear = FlatFakeQuantLinear(ffq_config, linear)

        module = nn.Module()
        module.add_module('test_ffq', ffq_linear)

        processor.set_hook_ir(module)

        self.assertIsInstance(module.test_ffq, nn.Linear)

    def test_FlatQuantProcessor_handles_ffq_linear_with_save_trans_when_set_hook_ir(self):
        """FlatQuantProcessor-调用set_hook_ir时-处理带有save_trans的FlatFakeQuantLinear"""
        processor = FlatQuantProcessor(self.model, self.config, self.adapter)
        linear = nn.Linear(4, 4)
        ffq_config = FlatFakeQuantLinearConfig()
        ffq_linear = FlatFakeQuantLinear(ffq_config, linear)

        mock_save_trans = MagicMock()
        mock_save_trans.get_save_params.return_value = {'test': 'params'}
        ffq_linear.save_trans = mock_save_trans

        module = nn.Module()
        module.add_module('test_ffq', ffq_linear)

        processor.set_hook_ir(module)

        self.assertIsInstance(module.test_ffq, nn.Linear)

    def test_FlatQuantProcessor_replaces_flat_norm_wrapper_when_set_hook_ir(self):
        """FlatQuantProcessor-调用set_hook_ir时-替换FlatNormWrapper为原始Norm"""
        processor = FlatQuantProcessor(self.model, self.config, self.adapter)
        norm = nn.LayerNorm(4)
        norm_wrapper = FlatNormWrapper(norm)

        module = nn.Module()
        module.add_module('test_norm', norm_wrapper)

        processor.set_hook_ir(module)

        self.assertIsInstance(module.test_norm, nn.LayerNorm)

    def test_FlatQuantProcessor_handles_nested_modules_when_set_hook_ir(self):
        """FlatQuantProcessor-调用set_hook_ir时-处理嵌套模块"""
        processor = FlatQuantProcessor(self.model, self.config, self.adapter)
        inner_module = nn.Module()
        linear = nn.Linear(4, 4)
        ffq_config = FlatFakeQuantLinearConfig()
        ffq_linear = FlatFakeQuantLinear(ffq_config, linear)
        inner_module.add_module('inner_ffq', ffq_linear)

        outer_module = nn.Module()
        outer_module.add_module('outer', inner_module)

        processor.set_hook_ir(outer_module)

        self.assertIsInstance(inner_module.inner_ffq, nn.Linear)

    # 完整工作流程测试
    def test_flat_quant_processor_full_workflow_sets_eval_mode_when_run(self):
        """FlatQuantProcessor-完整工作流程-无错误运行"""
        processor = FlatQuantProcessor(self.model, self.config, self.adapter)

        processor.pre_run()
        self.assertFalse(self.model.training)

        module = nn.Linear(4, 4)
        request = BatchProcessRequest(
            name="test_layer", module=module, datas=[(torch.randn(1, 4), {})], outputs=[(torch.randn(1, 4),)]
        )

        with patch.object(processor, '_run_forward_if_need'):
            with patch('msmodelslim.processor.flat_quant.flat_quant.FlatQuantLayerManager') as MockManager:
                mock_manager = MagicMock()
                MockManager.return_value = mock_manager

                processor.preprocess(request)

                processor.float_output = [torch.randn(1, 4)]
                with patch.object(processor.layer_trainer, 'train_layer'):
                    processor.process(request)

                processor.dtype_dict = {}
                processor.trans_include = set(['*'])
                processor.trans_exclude = set()
                with patch.object(processor, 'set_hook_ir'):
                    with patch.object(processor, '_rollback_trans'):
                        processor.postprocess(request)


# ==================== npu_available 测试 ====================


class TestNpuAvailable(unittest.TestCase):
    """测试 torch_npu 导入分支"""

    def test_npu_available_is_boolean(self):
        """npu_available-检查-是布尔值"""
        self.assertIsInstance(npu_available, bool)


if __name__ == '__main__':
    unittest.main()
