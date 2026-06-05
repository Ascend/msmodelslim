#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
-------------------------------------------------------------------------
This file is part of the MindStudio project.
Copyright (c) 2025 Huawei Technologies Co.,Ltd.
...
-------------------------------------------------------------------------
"""

import unittest
from unittest.mock import Mock, patch

import torch
from torch import nn

from msmodelslim.core.base.protocol import BatchProcessRequest
from msmodelslim.ir import FP8FakeQuantActivationPerHead, FakeQuantActivationPerToken
from msmodelslim.processor.quant.fa3 import (
    FA3QuantProcessor,
    FA3QuantProcessorConfig,
    FA3QuantAdapterInterface,
    FA3QuantPlaceHolder,
)
from msmodelslim.processor.quant.fa3.processor import FA3AttentionDetails, _FA3PerHeadObserver
from msmodelslim.utils.exception import UnsupportedError, SpecError, SchemaValidateError
from msmodelslim.core.quantizer.base import QConfig
from msmodelslim.ir.qal.qbase import QDType, QScope
from msmodelslim.ir.qal import QParam, QScheme


# ---------------------------------------------------------------------------
# 辅助工厂函数
# ---------------------------------------------------------------------------
def create_qconfig(dtype, scope, symmetric=True, method="minmax"):
    """创建合法的 QConfig 对象"""
    return QConfig(dtype=dtype, scope=scope, symmetric=symmetric, method=method)


def create_branch_qconfig_dict():
    """创建 FA3AttentionDetails 所需的字典"""
    return {
        "fa_q": {"dtype": "int8", "scope": "per_head", "symmetric": True, "method": "minmax"},
        "fa_k": {"dtype": "fp8_e4m3", "scope": "per_head", "symmetric": True, "method": "minmax"},
    }


def create_processor_config(include=None, exclude=None, qconfig=None, details=None):
    """直接构造 FA3QuantProcessorConfig 实例，补充必要字段"""
    kwargs = {
        "type": "fa3_quant",
        "include": include if include is not None else ["*"],
        "exclude": exclude if exclude is not None else [],
    }
    if qconfig is not None:
        kwargs["qconfig"] = qconfig
    if details is not None:
        kwargs["details"] = details
    return FA3QuantProcessorConfig(**kwargs)


def create_mock_adapter():
    """创建模拟适配器"""
    adapter = Mock(spec=FA3QuantAdapterInterface)
    adapter.inject_fa3_placeholders = Mock()
    return adapter


def create_simple_model():
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(64, 64)
            self.block = None
            self.layer = None

        def forward(self, x):
            return self.linear(x)

    return SimpleModel()


# ---------------------------------------------------------------------------
# 配置测试类
# ---------------------------------------------------------------------------
class TestFA3QuantProcessorConfig(unittest.TestCase):
    """测试 FA3QuantProcessorConfig 的构造和默认行为"""

    def test_qconfig_default_to_INT8_per_head_when_qconfig_not_provided(self):
        """未提供 qconfig 时，默认生成 INT8 per‑head 对称 QConfig"""
        config = create_processor_config()
        self.assertIsInstance(config.qconfig, QConfig)
        self.assertIsNone(config.details)
        self.assertEqual(config.qconfig.dtype, QDType.INT8)
        self.assertEqual(config.qconfig.scope, QScope.PER_HEAD)
        self.assertTrue(config.qconfig.symmetric)

    def test_qconfig_stores_instance_when_qconfig_instance_given(self):
        """传入 QConfig 实例时，正确保存该实例"""
        qconfig = create_qconfig(QDType.FP8_E4M3, QScope.PER_HEAD, symmetric=True)
        config = create_processor_config(qconfig=qconfig)
        self.assertIsInstance(config.qconfig, QConfig)
        self.assertIsNone(config.details)
        self.assertEqual(config.qconfig.dtype, QDType.FP8_E4M3)

    def test_details_stores_attention_details_when_branch_dict_given(self):
        """传入 details 字典时，转换为 FA3AttentionDetails，qconfig 为 None"""
        details_dict = create_branch_qconfig_dict()
        config = create_processor_config(details=details_dict)
        self.assertIsInstance(config.details, FA3AttentionDetails)
        self.assertIsNone(config.qconfig)
        details = config.details
        self.assertIsNotNone(details.fa_q)
        self.assertIsNotNone(details.fa_k)
        self.assertIsNone(details.fa_v)

    def test_mutual_exclusivity_raises_when_both_qconfig_and_details_provided(self):
        """同时提供 qconfig 和 details 时应抛出 SchemaValidateError"""
        qconfig = create_qconfig(QDType.INT8, QScope.PER_HEAD)
        details = create_branch_qconfig_dict()
        with self.assertRaises(SchemaValidateError) as ctx:
            create_processor_config(qconfig=qconfig, details=details)
        self.assertIn("only one of the qconfig and details", str(ctx.exception))

    def test_type_equals_fa3_quant_when_created(self):
        """type 字段固定为 'fa3_quant'"""
        config = create_processor_config()
        self.assertEqual(config.type, "fa3_quant")

    def test_include_exclude_stored_when_provided(self):
        """提供 include/exclude 列表时，正确保存"""
        config = create_processor_config(include=["layer1"], exclude=["layer2"])
        self.assertEqual(config.include, ["layer1"])
        self.assertEqual(config.exclude, ["layer2"])


# ---------------------------------------------------------------------------
# 处理器测试类
# ---------------------------------------------------------------------------
class TestFA3QuantProcessor(unittest.TestCase):
    """测试 FA3QuantProcessor 的初始化、属性和 pre/postprocess 方法"""

    def setUp(self):
        self.adapter = create_mock_adapter()
        self.simple_model = create_simple_model()
        # 禁用 breakpoint 避免源码中遗留的调试断点干扰测试
        self._breakpoint_patcher = patch('builtins.breakpoint', lambda *args, **kwargs: None)
        self._breakpoint_patcher.start()

    def tearDown(self):
        self._breakpoint_patcher.stop()

    # ---------- 初始化（正常/异常） ----------
    def test_init_stores_config_and_adapter_when_valid(self):
        config = create_processor_config()
        processor = FA3QuantProcessor(self.simple_model, config, self.adapter)
        self.assertEqual(processor.config, config)
        self.assertEqual(processor.adapter, self.adapter)
        self.assertIsNotNone(processor.include)
        self.assertIsNotNone(processor.exclude)

    def test_init_raises_UnsupportedError_when_adapter_is_None(self):
        config = create_processor_config()
        with self.assertRaises(UnsupportedError):
            FA3QuantProcessor(self.simple_model, config, adapter=None)

    def test_init_raises_UnsupportedError_when_adapter_not_interface(self):
        config = create_processor_config()
        with self.assertRaises(UnsupportedError):
            FA3QuantProcessor(self.simple_model, config, adapter=Mock())

    # ---------- 属性方法 ----------
    def test_is_data_free_returns_False_when_per_head_qconfig(self):
        config = create_processor_config()
        processor = FA3QuantProcessor(self.simple_model, config, self.adapter)
        self.assertFalse(processor.is_data_free())

    def test_is_data_free_returns_True_when_per_token_qconfig(self):
        config = create_processor_config(qconfig=create_qconfig(QDType.INT8, QScope.PER_TOKEN))
        processor = FA3QuantProcessor(self.simple_model, config, self.adapter)
        self.assertTrue(processor.is_data_free())

    def test_is_data_free_returns_False_when_details_per_head(self):
        details = {
            "fa_q": create_qconfig(QDType.INT8, QScope.PER_HEAD).model_dump(),
            "fa_k": create_qconfig(QDType.FP8_E4M3, QScope.PER_HEAD).model_dump(),
        }
        config = create_processor_config(details=details)
        processor = FA3QuantProcessor(self.simple_model, config, self.adapter)
        self.assertFalse(processor.is_data_free())

    def test_is_data_free_returns_True_when_details_per_token(self):
        details = {
            "fa_q": create_qconfig(QDType.INT8, QScope.PER_TOKEN).model_dump(),
            "fa_v": create_qconfig(QDType.FP8_E4M3, QScope.PER_TOKEN).model_dump(),
        }
        config = create_processor_config(details=details)
        processor = FA3QuantProcessor(self.simple_model, config, self.adapter)
        self.assertTrue(processor.is_data_free())

    def test_support_distributed_returns_True_always(self):
        config = create_processor_config()
        processor = FA3QuantProcessor(self.simple_model, config, self.adapter)
        self.assertTrue(processor.support_distributed())

    # ---------- preprocess ----------
    @patch("msmodelslim.processor.quant.fa3.processor.dist.is_initialized", return_value=False)
    def test_preprocess_calls_adapter_and_replaces_placeholder_when_dist_not_init(self, mock_dist):
        config = create_processor_config(include=["block.fa_q"])
        model = create_simple_model()
        block = nn.Module()
        block.fa_q = FA3QuantPlaceHolder(ratio=0.9)
        model.block = block

        processor = FA3QuantProcessor(model, config, self.adapter)
        request = BatchProcessRequest(name="block", module=block, datas=None, outputs=None)
        self.adapter.inject_fa3_placeholders.side_effect = None

        processor.preprocess(request)

        self.adapter.inject_fa3_placeholders.assert_called_once()
        call_args = self.adapter.inject_fa3_placeholders.call_args[0]
        self.assertEqual(call_args[0], "block")
        self.assertIs(call_args[1], block)
        should_inject = call_args[2]
        self.assertTrue(should_inject("block.fa_q"))
        self.assertFalse(should_inject("block.other"))
        self.assertIsInstance(block.fa_q, _FA3PerHeadObserver)

    @patch("msmodelslim.processor.quant.fa3.processor.dist.is_initialized", return_value=False)
    def test_preprocess_logs_warning_when_adapter_raises(self, mock_dist):
        self.adapter.inject_fa3_placeholders.side_effect = RuntimeError("mock error")
        config = create_processor_config()
        model = create_simple_model()
        block = nn.Module()
        model.block = block
        processor = FA3QuantProcessor(model, config, self.adapter)

        request = BatchProcessRequest(name="block", module=block, datas=None, outputs=None)

        with self.assertLogs('msmodelslim.processor.fa3_quant', level='WARNING') as log:
            processor.preprocess(request)
        self.assertTrue(any("mock error" in msg for msg in log.output))

    # ---------- postprocess ----------
    def _prepare_postprocess_model(self, scope=QScope.PER_HEAD, collect_data=True, qconfig=None, details=None):
        """构建一个包含 observer 子模块的模型，并返回 processor 和相关对象。"""
        model = create_simple_model()
        config_kwargs = {"include": ["layer.fa_q"]}
        if qconfig is not None:
            config_kwargs["qconfig"] = qconfig
        if details is not None:
            config_kwargs["details"] = details
        config = create_processor_config(**config_kwargs)

        processor = FA3QuantProcessor(model, config, self.adapter)

        layer = nn.Module()
        observer = _FA3PerHeadObserver(ratio=1.0, name="layer.fa_q")
        layer.fa_q = observer
        model.layer = layer

        if collect_data:
            observer(torch.randn(2, 4, 10, 16))

        request = BatchProcessRequest(name="layer", module=layer, datas=None, outputs=None)
        return processor, request, layer, observer

    @patch("msmodelslim.processor.quant.fa3.processor.calculate_qparam")
    @patch("msmodelslim.ir.FP8FakeQuantActivationPerHead")
    def test_postprocess_replaces_observer_with_per_head_IR_when_calibrated(self, mock_activation, mock_calc):
        processor, request, layer, observer = self._prepare_postprocess_model(
            scope=QScope.PER_HEAD, qconfig=create_qconfig(QDType.INT8, QScope.PER_HEAD)
        )

        fake_qparam = QParam(
            scheme=QScheme(scope=QScope.PER_HEAD, dtype=QDType.FP8_E4M3, symmetric=True), ext={"scale": torch.rand(4)}
        )
        mock_calc.return_value = fake_qparam
        real_ir = FP8FakeQuantActivationPerHead(fake_qparam)
        mock_activation.create.return_value = real_ir

        processor.postprocess(request)

        mock_calc.assert_called_once()
        _, kwargs = mock_calc.call_args
        self.assertTrue(torch.equal(kwargs["min_val"], observer.min_val.squeeze()))
        self.assertTrue(torch.equal(kwargs["max_val"], observer.max_val.squeeze()))
        self.assertEqual(kwargs["q_dtype"], QDType.INT8)

        self.assertIsInstance(layer.fa_q, FP8FakeQuantActivationPerHead)

        test_input = torch.randn(2, 4, 10, 16)
        with torch.no_grad():
            output = layer.fa_q(test_input)
        self.assertEqual(output.shape, test_input.shape)

    def test_postprocess_raises_SpecError_when_no_calibration_data(self):
        processor, request, _, _ = self._prepare_postprocess_model(
            scope=QScope.PER_HEAD, qconfig=create_qconfig(QDType.INT8, QScope.PER_HEAD), collect_data=False
        )
        with self.assertRaises(SpecError) as ctx:
            processor.postprocess(request)
        self.assertIn("no any update_stats", str(ctx.exception))

    @patch("msmodelslim.ir.auto.AutoFakeQuantActivation.create")
    def test_postprocess_replaces_observer_with_per_token_IR_when_per_token_qconfig(self, mock_create):
        model = create_simple_model()
        config = create_processor_config(
            include=["layer.fa_q"], qconfig=create_qconfig(QDType.FP8_E4M3, QScope.PER_TOKEN, symmetric=True)
        )
        processor = FA3QuantProcessor(model, config, self.adapter)

        layer = nn.Module()
        layer.fa_q = _FA3PerHeadObserver(ratio=1.0)
        model.layer = layer

        request = BatchProcessRequest(name="layer", module=layer, datas=None, outputs=None)

        fake_qparam = QParam(scheme=QScheme(scope=QScope.PER_TOKEN, dtype=QDType.FP8_E4M3, symmetric=True))
        real_ir = FakeQuantActivationPerToken(fake_qparam)
        mock_create.return_value = real_ir

        processor.postprocess(request)
        mock_create.assert_called_once()
        self.assertIs(layer.fa_q, real_ir)

    def test_postprocess_raises_UnsupportedError_when_unsupported_scope(self):
        model = create_simple_model()
        config = create_processor_config(
            include=["layer.fa_q"], qconfig=create_qconfig(QDType.INT8, QScope.PER_CHANNEL, symmetric=True)
        )
        processor = FA3QuantProcessor(model, config, self.adapter)

        layer = nn.Module()
        observer = _FA3PerHeadObserver(ratio=1.0)
        layer.fa_q = observer
        model.layer = layer
        observer(torch.randn(2, 4, 10, 16))

        request = BatchProcessRequest(name="layer", module=layer, datas=None, outputs=None)
        with self.assertRaises(UnsupportedError):
            processor.postprocess(request)

    def test_postprocess_clears_dist_helper_after_execution(self):
        processor, request, _, _ = self._prepare_postprocess_model(qconfig=create_qconfig(QDType.INT8, QScope.PER_HEAD))
        processor.dist_helper = Mock()
        processor.postprocess(request)
        self.assertIsNone(processor.dist_helper)

    # ---------- details 分支的 postprocess 测试 ----------
    @patch("msmodelslim.processor.quant.fa3.processor.calculate_qparam")
    @patch("msmodelslim.ir.FP8FakeQuantActivationPerHead")
    def test_postprocess_uses_details_branch_qconfig_when_details_configured(self, mock_activation, mock_calc):
        """details 模式下，postprocess 根据分支名提取对应的 QConfig 进行处理"""
        details_dict = {
            "fa_q": create_qconfig(QDType.INT8, QScope.PER_HEAD).model_dump(),
            "fa_k": create_qconfig(QDType.FP8_E4M3, QScope.PER_HEAD).model_dump(),
        }
        processor, request, layer, observer = self._prepare_postprocess_model(details=details_dict, collect_data=True)

        fake_qparam = QParam(
            scheme=QScheme(scope=QScope.PER_HEAD, dtype=QDType.FP8_E4M3, symmetric=True), ext={"scale": torch.rand(4)}
        )
        mock_calc.return_value = fake_qparam
        real_ir = FP8FakeQuantActivationPerHead(fake_qparam)
        mock_activation.create.return_value = real_ir

        processor.postprocess(request)

        # 因为分支名称是 'fa_q'，所以使用的应该是 details 中的 fa_q 配置
        mock_calc.assert_called_once()
        _, kwargs = mock_calc.call_args
        self.assertEqual(kwargs["q_dtype"], QDType.INT8)  # fa_q 配置的 dtype 是 INT8
        self.assertIsInstance(layer.fa_q, FP8FakeQuantActivationPerHead)

    def test_postprocess_skips_module_when_qconfig_is_none_in_details(self):
        """当 details 中对应分支的 qconfig 为 None 时，postprocess 应跳过该模块并记录 debug"""
        with patch("msmodelslim.processor.quant.fa3.processor.get_logger") as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger

            # 只配置 fa_q，fa_k 保持 None（模拟不存在的分支）
            details = FA3AttentionDetails(fa_q=create_qconfig(QDType.INT8, QScope.PER_HEAD))
            config = create_processor_config(include=["layer.fa_q", "layer.fa_k"], details=details)
            model = create_simple_model()
            processor = FA3QuantProcessor(model, config, self.adapter)

            layer = nn.Module()
            observer_q = _FA3PerHeadObserver(ratio=1.0, name="layer.fa_q")
            observer_k = _FA3PerHeadObserver(ratio=1.0, name="layer.fa_k")
            layer.fa_q = observer_q
            layer.fa_k = observer_k
            model.layer = layer

            # 为两个 observer 提供校准数据，确保均可进入 postprocess 判断
            observer_q(torch.randn(2, 4, 10, 16))
            observer_k(torch.randn(2, 4, 10, 16))

            request = BatchProcessRequest(name="layer", module=layer, datas=None, outputs=None)

            with (
                patch.object(processor, '_process_per_head') as mock_per_head,
                patch.object(processor, '_process_per_token') as mock_per_token,
            ):
                processor.postprocess(request)

            # fa_q 应该被处理（对应的 qconfig 存在）
            mock_per_head.assert_called_once()
            args, _ = mock_per_head.call_args
            self.assertEqual(args[1], "layer.fa_q")
            mock_per_token.assert_not_called()

            # fa_k 应该被跳过，并记录 debug 日志
            debug_calls = [
                c for c in mock_logger.debug.call_args_list if "layer.fa_k" in str(c) and "skipping" in str(c)
            ]
            self.assertTrue(len(debug_calls) > 0, "Debug log for skipping layer.fa_k not found")


if __name__ == '__main__':
    unittest.main()
