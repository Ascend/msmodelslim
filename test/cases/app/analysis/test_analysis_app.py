#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
-------------------------------------------------------------------------
This file is part of the MindStudio project.
Copyright (c) 2025 Huawei Technologies Co.,Ltd.

MindStudio is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:

         http://license.coscl.org.cn/MulanPSL2

THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details.
-------------------------------------------------------------------------

综合测试用例：验证分析模块的完整功能覆盖
包括CLI、APP和分析服务模块的所有功能
目标覆盖率：>80%
"""

import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from testing_utils.mock import mock_init_config

from msmodelslim.core.analysis_service import AnalysisResult
from msmodelslim.core.const import DeviceType
from msmodelslim.utils.exception import SchemaValidateError

mock_init_config()


class TestComprehensiveAnalysisCoverage(unittest.TestCase):
    """综合测试分析模块的所有功能"""

    def setUp(self):
        """测试前的准备工作"""

        # 1. 保存原始 umask
        original_umask = os.umask(0)  # 临时设为 0 并获取原始值
        try:
            # 2. 设置目标 umask（0o026 对应权限 640/750）
            os.umask(0o026)
            self.temp_dir = tempfile.mkdtemp()
            self.dataset_dir = Path(self.temp_dir) / "lab_calib"
            self.dataset_dir.mkdir()

            # 创建模拟的校准数据集文件
            self.calib_file = self.dataset_dir / "boolq.jsonl"
            with open(self.calib_file, 'w', encoding='utf-8') as f:
                f.write('{"data": "mock calibration data"}')

            # 创建模型文件
            self.model_path = Path(self.temp_dir) / "model"
            self.model_path.mkdir()
        finally:
            # 3. 无论是否出错，都恢复原始 umask
            os.umask(original_umask)

    def tearDown(self):
        """测试后的清理工作"""
        shutil.rmtree(self.temp_dir)


class TestAppAnalysisModule(TestComprehensiveAnalysisCoverage):
    """测试APP分析模块"""

    def test_layer_analysis_application_init(self):
        """测试LayerAnalysisApplication初始化"""
        from msmodelslim.app.analysis.application import LayerAnalysisApplication

        mock_service = MagicMock()
        mock_factory = MagicMock()

        app = LayerAnalysisApplication(mock_service, mock_factory, MagicMock())

        self.assertEqual(app.analysis_service, mock_service)
        self.assertEqual(app.model_factory, mock_factory)

    def test_analyze_parameter_validation_model_type(self):
        """测试analyze方法model_type参数验证"""
        from msmodelslim.app.analysis.application import (
            AnalysisMetrics,
            LayerAnalysisApplication,
            LinearArgs,
        )

        mock_service = MagicMock()
        mock_factory = MagicMock()
        app = LayerAnalysisApplication(mock_service, mock_factory, MagicMock())

        # 测试无效的model_type类型
        with self.assertRaises(SchemaValidateError):
            app.analyze(
                model_type=123,  # 应该是字符串
                model_path=str(self.model_path),
                scope_args=LinearArgs(pattern=["*"], metrics=AnalysisMetrics.STD),
                device=DeviceType.NPU,
                calib_dataset="boolq.jsonl",
                topk=15,
                trust_remote_code=False,
            )

    def test_analyze_parameter_validation_patterns(self):
        """测试analyze方法patterns参数验证"""
        from msmodelslim.app.analysis.application import (
            AnalysisMetrics,
            LayerAnalysisApplication,
            LinearArgs,
        )

        mock_service = MagicMock()
        mock_factory = MagicMock()
        app = LayerAnalysisApplication(mock_service, mock_factory, MagicMock())

        # 测试无效的 pattern 类型（LinearArgs）
        with self.assertRaises(SchemaValidateError):
            app.analyze(
                model_type="Qwen2.5-7B-Instruct",
                model_path=str(self.model_path),
                scope_args=LinearArgs(
                    pattern="not_a_list",  # type: ignore[arg-type]  # intentionally invalid for validation
                    metrics=AnalysisMetrics.STD,
                ),
                device=DeviceType.NPU,
                calib_dataset="boolq.jsonl",
                topk=15,
                trust_remote_code=False,
            )

    def test_analyze_parameter_validation_calib_dataset_format(self):
        """测试analyze方法calib_dataset文件格式验证"""
        from msmodelslim.app.analysis.application import (
            AnalysisMetrics,
            LayerAnalysisApplication,
            LinearArgs,
        )

        mock_service = MagicMock()
        mock_factory = MagicMock()
        app = LayerAnalysisApplication(mock_service, mock_factory, MagicMock())

        # 测试无效的文件格式
        with self.assertRaises(SchemaValidateError):
            app.analyze(
                model_type="Qwen2.5-7B-Instruct",
                model_path=str(self.model_path),
                scope_args=LinearArgs(pattern=["*"], metrics=AnalysisMetrics.STD),
                device=DeviceType.NPU,
                calib_dataset="invalid.txt",  # 无效的文件格式
                topk=15,
                trust_remote_code=False,
            )

    def test_analyze_parameter_validation_topk(self):
        """测试analyze方法topk参数验证"""
        from msmodelslim.app.analysis.application import (
            AnalysisMetrics,
            LayerAnalysisApplication,
            LinearArgs,
        )

        mock_service = MagicMock()
        mock_factory = MagicMock()
        app = LayerAnalysisApplication(mock_service, mock_factory, MagicMock())

        # 测试无效的topk值
        with self.assertRaises(SchemaValidateError):
            app.analyze(
                model_type="Qwen2.5-7B-Instruct",
                model_path=str(self.model_path),
                scope_args=LinearArgs(pattern=["*"], metrics=AnalysisMetrics.STD),
                device=DeviceType.NPU,
                calib_dataset="boolq.jsonl",
                topk=0,  # 无效的topk值
                trust_remote_code=False,
            )

    def test_analyze_parameter_validation_scope_args_type(self):
        """scope_args 必须为 LinearArgs / AttnArgs / LayerArgs"""
        from msmodelslim.app.analysis.application import LayerAnalysisApplication

        mock_service = MagicMock()
        mock_factory = MagicMock()
        app = LayerAnalysisApplication(mock_service, mock_factory, MagicMock())

        with self.assertRaises(SchemaValidateError):
            app.analyze(
                model_type="Qwen2.5-7B-Instruct",
                model_path=str(self.model_path),
                scope_args={"pattern": ["*"]},  # type: ignore[arg-type]
                device=DeviceType.NPU,
                calib_dataset="boolq.jsonl",
                topk=15,
                trust_remote_code=False,
            )

    @patch('msmodelslim.app.analysis.application.get_logger')
    def test_analyze_with_valid_parameters(self, mock_get_logger):
        """测试analyze方法使用有效参数"""
        from msmodelslim.app.analysis.application import (
            AnalysisMetrics,
            LayerAnalysisApplication,
            LinearArgs,
        )
        from msmodelslim.core.analysis_service import AnalysisScope
        from msmodelslim.core.runner.pipeline_interface import PipelineInterface

        mock_service = MagicMock()
        mock_model_factory = MagicMock()
        mock_result_manager = MagicMock()
        mock_model_adapter = MagicMock(spec=PipelineInterface)
        mock_result = MagicMock()
        mock_logger = MagicMock()

        mock_model_factory.create.return_value = mock_model_adapter
        mock_service.analyze.return_value = mock_result
        mock_get_logger.return_value = mock_logger

        app = LayerAnalysisApplication(mock_service, mock_model_factory, mock_result_manager)

        result = app.analyze(
            model_type="Qwen2.5-7B-Instruct",
            model_path=str(self.model_path),
            scope_args=LinearArgs(pattern=["*"], metrics=AnalysisMetrics.STD),
            device=DeviceType.CPU,
            calib_dataset="boolq.jsonl",
            topk=15,
            trust_remote_code=False,
        )

        self.assertEqual(result, mock_result)
        mock_model_factory.create.assert_called_once_with("Qwen2.5-7B-Instruct", self.model_path, False)
        mock_service.analyze.assert_called_once()
        call_kw = mock_service.analyze.call_args.kwargs
        self.assertEqual(call_kw["device"], DeviceType.CPU)
        self.assertEqual(call_kw["model_adapter"], mock_model_adapter)
        ac = call_kw["analysis_config"]
        self.assertEqual(ac.scope, AnalysisScope.LINEAR)
        self.assertEqual(ac.metrics, "std")
        self.assertEqual(ac.calib_dataset, "boolq.jsonl")
        self.assertEqual(ac.linear_pattern, ["*"])
        self.assertIsNone(ac.quant_modules)
        mock_result_manager.display_result.assert_called_once_with(
            mock_result,
            15,
            AnalysisScope.LINEAR,
        )

    @patch('msmodelslim.app.analysis.application.get_logger')
    def test_analyze_with_unsupported_model_adapter(self, mock_get_logger):
        """测试analyze方法使用不支持的模型适配器"""
        from msmodelslim.app.analysis.application import (
            AnalysisMetrics,
            LayerAnalysisApplication,
            LinearArgs,
        )
        from msmodelslim.utils.exception import UnsupportedError

        mock_service = MagicMock()
        mock_model_factory = MagicMock()
        mock_model_adapter = MagicMock()  # 不是PipelineInterface
        mock_logger = MagicMock()

        mock_model_factory.create.return_value = mock_model_adapter
        mock_get_logger.return_value = mock_logger

        app = LayerAnalysisApplication(mock_service, mock_model_factory, MagicMock())

        # 测试不支持的模型适配器
        with self.assertRaises(UnsupportedError):
            app.analyze(
                model_type="Qwen2.5-7B-Instruct",
                model_path=str(self.model_path),
                scope_args=LinearArgs(pattern=["*"], metrics=AnalysisMetrics.STD),
                device=DeviceType.NPU,
                calib_dataset="boolq.jsonl",
                topk=15,
                trust_remote_code=False,
            )

    @patch('msmodelslim.app.analysis.application.get_logger')
    def test_analyze_with_none_result(self, mock_get_logger):
        """测试analyze方法返回None结果"""
        from msmodelslim.app.analysis.application import (
            AnalysisMetrics,
            LayerAnalysisApplication,
            LinearArgs,
        )
        from msmodelslim.core.runner.pipeline_interface import PipelineInterface

        mock_service = MagicMock()
        mock_model_factory = MagicMock()
        mock_model_adapter = MagicMock(spec=PipelineInterface)
        mock_logger = MagicMock()

        mock_model_factory.create.return_value = mock_model_adapter
        mock_service.analyze.return_value = None
        mock_get_logger.return_value = mock_logger
        mock_result_manager = MagicMock()

        app = LayerAnalysisApplication(mock_service, mock_model_factory, mock_result_manager)

        result = app.analyze(
            model_type="Qwen2.5-7B-Instruct",
            model_path=str(self.model_path),
            scope_args=LinearArgs(pattern=["*"], metrics=AnalysisMetrics.KURTOSIS),
            device=DeviceType.NPU,
            calib_dataset="boolq.jsonl",
            topk=15,
            trust_remote_code=False,
        )

        self.assertIsNone(result)
        mock_result_manager.display_result.assert_not_called()


def create_mock_analysis_result(layer_scores: list) -> AnalysisResult:
    """构建 AnalysisResult 对象，用于测试输入。"""
    return AnalysisResult(
        layer_scores=layer_scores,
        method="kurtosis",
        patterns=["conv2d", "linear", "mlp"],
    )


class TestPrintAnalysisResults(unittest.TestCase):
    """测试结果展示（LoggingAnalysisResultDisplayer.display_result）"""

    def test_display_result_logs_and_yaml(self):
        """正常场景：display_result 打印层结果与 YAML 格式"""
        from msmodelslim.infra.logging_analysis_result_displayer import LoggingAnalysisResultDisplayer

        test_layers = [
            {"name": "model.layers.26.mlp.down_proj", "score": 98.7654},
            {"name": "model.layers.4.mlp.down_proj", "score": 87.6543},
            {"name": "model.layers.1.mlp.down_proj", "score": 76.5432},
            {"name": "model.layers.3.mlp.down_proj", "score": 65.4321},
            {"name": "model.layers.2.mlp.down_proj", "score": 54.3210},
        ]
        result = create_mock_analysis_result(test_layers)
        displayer = LoggingAnalysisResultDisplayer()
        mock_logger = MagicMock()

        with patch("msmodelslim.infra.logging_analysis_result_displayer.get_logger", return_value=mock_logger):
            with patch("msmodelslim.infra.logging_analysis_result_displayer.clean_output"):
                displayer.display_result(result, topk=3)

        self.assertTrue(mock_logger.info.called)
        # Build formatted messages (logger.info(fmt, *args) so format with args)
        log_messages = []
        for call in mock_logger.info.call_args_list:
            args = call[0] if call[0] else ()
            if len(args) >= 1:
                try:
                    msg = args[0] % args[1:] if len(args) > 1 else str(args[0])
                except (TypeError, ValueError):
                    msg = str(args[0])
            else:
                msg = ""
            log_messages.append(str(msg))
        self.assertTrue(any("Layer Analysis Results" in msg for msg in log_messages))
        self.assertTrue(any("kurtosis" in msg for msg in log_messages))
        self.assertTrue(any("Total layers analyzed: 5" in msg for msg in log_messages))

    def test_display_result_layer_scope_yaml_suffix(self):
        """layer scope：YAML disable_names 条目应为 ``block.*`` 以匹配整块子模块。"""
        from msmodelslim.core.analysis_service import AnalysisScope
        from msmodelslim.infra.logging_analysis_result_displayer import LoggingAnalysisResultDisplayer

        test_layers = [
            {"name": "model.layers.35", "score": 1.0},
            {"name": "model.layers.0", "score": 0.5},
        ]
        result = create_mock_analysis_result(test_layers)
        displayer = LoggingAnalysisResultDisplayer()
        mock_logger = MagicMock()

        with patch("msmodelslim.infra.logging_analysis_result_displayer.get_logger", return_value=mock_logger):
            with patch("msmodelslim.infra.logging_analysis_result_displayer.clean_output"):
                displayer.display_result(result, topk=10, scope=AnalysisScope.LAYER)

        log_messages = []
        for call in mock_logger.info.call_args_list:
            args = call[0] if call[0] else ()
            if len(args) >= 1:
                try:
                    msg = args[0] % args[1:] if len(args) > 1 else str(args[0])
                except (TypeError, ValueError):
                    msg = str(args[0])
            else:
                msg = ""
            log_messages.append(str(msg))
        self.assertTrue(any("model.layers.35.*" in msg for msg in log_messages))
        self.assertTrue(any("model.layers.0.*" in msg for msg in log_messages))


class TestAnalysisMetrics(unittest.TestCase):
    """测试AnalysisMetrics枚举"""

    def test_analysis_metrics_values(self):
        """测试AnalysisMetrics枚举值"""
        from msmodelslim.app.analysis.application import AnalysisMetrics

        self.assertEqual(AnalysisMetrics.STD.value, 'std')
        self.assertEqual(AnalysisMetrics.QUANTILE.value, 'quantile')
        self.assertEqual(AnalysisMetrics.KURTOSIS.value, 'kurtosis')
        self.assertEqual(AnalysisMetrics.ATTENTION_MSE.value, 'mse')

    def test_analysis_metrics_extended_enum_functionality(self):
        """测试AnalysisMetrics的ExtendedEnum功能"""
        from msmodelslim.app.analysis.application import AnalysisMetrics

        # 测试所有值都是有效的字符串
        for metric in AnalysisMetrics:
            self.assertIsInstance(metric.value, str)
            self.assertGreater(len(metric.value), 0)


if __name__ == '__main__':
    unittest.main()
