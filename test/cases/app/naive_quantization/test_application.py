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

import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from testing_utils.mock import mock_init_config

mock_init_config()

from msmodelslim.core.const import DeviceType, QuantType  # noqa: E402
from msmodelslim.core.practice.interface import Metadata  # noqa: E402
from msmodelslim.utils.exception import EnvError, SchemaValidateError, ToDoError, UnsupportedError  # noqa: E402


class TestNaiveQuantizationAppBase(unittest.TestCase):
    """NaiveQuantizationApplication 测试基类"""

    def setUp(self):
        from msmodelslim.app.naive_quantization import application as app_module

        # Default to the mandatory checks only; patch.object restores the shared config on cleanup.
        config_patcher = patch.object(app_module.msmodelslim_config, "precheck_items", {"quant_precheck_item": set()})
        config_patcher.start()
        self.addCleanup(config_patcher.stop)
        original_umask = os.umask(0)
        try:
            os.umask(0o026)
            self.temp_dir = tempfile.mkdtemp()
            self.model_path = Path(self.temp_dir) / "model"
            self.model_path.mkdir()
            self.save_path = Path(self.temp_dir) / "save"
            self.save_path.mkdir()
        finally:
            os.umask(original_umask)

    def tearDown(self):
        if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)


class TestNaiveQuantizationApplicationInit(TestNaiveQuantizationAppBase):
    """测试 NaiveQuantizationApplication 初始化"""

    def test_init(self):
        """测试初始化"""
        from msmodelslim.app.naive_quantization.application import NaiveQuantizationApplication

        mock_practice_manager = MagicMock()
        mock_quant_service = MagicMock()
        mock_model_factory = MagicMock()

        app = NaiveQuantizationApplication(
            practice_manager=mock_practice_manager,
            quant_service=mock_quant_service,
            model_factory=mock_model_factory,
        )

        self.assertEqual(app.practice_manager, mock_practice_manager)
        self.assertEqual(app.quant_service, mock_quant_service)
        self.assertEqual(app.model_factory, mock_model_factory)


class TestCheckConfig(TestNaiveQuantizationAppBase):
    """测试 check_config 静态方法"""

    def _make_metadata(
        self,
        w_bit=8,
        a_bit=8,
        is_sparse=False,
        kv_cache=False,
        fa_quant=False,
        verified_model_types=None,
        verified_tags=None,
    ):
        return Metadata(
            config_id="test_config",
            score=90,
            label={
                "w_bit": w_bit,
                "a_bit": a_bit,
                "is_sparse": is_sparse,
                "kv_cache": kv_cache,
                "fa_quant": fa_quant,
            },
            verified_model_types=verified_model_types or [],
            verified_tags=verified_tags or {},
        )

    def test_check_config_label_mismatch_w_bit(self):
        """测试 labels 不匹配 w_bit"""
        from msmodelslim.app.naive_quantization.application import NaiveQuantizationApplication
        from msmodelslim.core.practice.interface import ScenarioTagMatch

        metadata = self._make_metadata(w_bit=4)
        result = NaiveQuantizationApplication.check_config(metadata, "Qwen2.5-7B", QuantType.W8A8)
        self.assertEqual(result, ScenarioTagMatch.NO_MATCH)

    def test_check_config_label_mismatch_a_bit(self):
        """测试 labels 不匹配 a_bit"""
        from msmodelslim.app.naive_quantization.application import NaiveQuantizationApplication
        from msmodelslim.core.practice.interface import ScenarioTagMatch

        metadata = self._make_metadata(a_bit=4)
        result = NaiveQuantizationApplication.check_config(metadata, "Qwen2.5-7B", QuantType.W8A8)
        self.assertEqual(result, ScenarioTagMatch.NO_MATCH)

    def test_check_config_verified_model_types_match(self):
        """测试 verified_model_types 匹配时直接返回 True"""
        from msmodelslim.app.naive_quantization.application import NaiveQuantizationApplication
        from msmodelslim.core.practice.interface import ScenarioTagMatch

        metadata = self._make_metadata(verified_model_types=["Qwen2.5-7B"])
        result = NaiveQuantizationApplication.check_config(metadata, "Qwen2.5-7B", QuantType.W8A8)
        self.assertEqual(result, ScenarioTagMatch.MATCH)

    def test_check_config_verified_model_types_no_match(self):
        """测试 verified_model_types 不匹配时返回 False"""
        from msmodelslim.app.naive_quantization.application import NaiveQuantizationApplication
        from msmodelslim.core.practice.interface import ScenarioTagMatch

        metadata = self._make_metadata(verified_model_types=["Other-Model"])
        result = NaiveQuantizationApplication.check_config(metadata, "Qwen2.5-7B", QuantType.W8A8)
        self.assertEqual(result, ScenarioTagMatch.NO_MATCH)

    def test_check_config_scenario_tags_match(self):
        """测试 scenario_tags 匹配 verified_tags"""
        from msmodelslim.app.naive_quantization.application import NaiveQuantizationApplication
        from msmodelslim.core.practice.interface import ScenarioTagMatch

        metadata = self._make_metadata(verified_tags={"Qwen2.5-7B": [["mindie", "npu"], ["vllm", "cpu"]]})
        result = NaiveQuantizationApplication.check_config(
            metadata,
            "Qwen2.5-7B",
            QuantType.W8A8,
            scenario_tags=["mindie", "npu"],
        )
        self.assertEqual(result, ScenarioTagMatch.MATCH)

    def test_check_config_scenario_tags_no_match_returns_standby(self):
        """测试 scenario_tags 不匹配时返回 standby"""
        from msmodelslim.app.naive_quantization.application import NaiveQuantizationApplication
        from msmodelslim.core.practice.interface import ScenarioTagMatch

        metadata = self._make_metadata(verified_tags={"Qwen2.5-7B": [["mindie", "npu"]]})
        result = NaiveQuantizationApplication.check_config(
            metadata,
            "Qwen2.5-7B",
            QuantType.W8A8,
            scenario_tags=["vllm"],
        )
        self.assertEqual(result, ScenarioTagMatch.STANDBY)

    def test_check_config_no_scenario_tags_returns_true(self):
        """测试无 scenario_tags 时返回 True"""
        from msmodelslim.app.naive_quantization.application import NaiveQuantizationApplication
        from msmodelslim.core.practice.interface import ScenarioTagMatch

        metadata = self._make_metadata(verified_tags={"Qwen2.5-7B": [["mindie", "npu"]]})
        result = NaiveQuantizationApplication.check_config(
            metadata,
            "Qwen2.5-7B",
            QuantType.W8A8,
            scenario_tags=None,
        )
        self.assertEqual(result, ScenarioTagMatch.MATCH)

    def test_check_config_w8a8f8_matches_fa_quant_label(self):
        """测试 w8a8f8 与 fa_quant: True 的 label 匹配"""
        from msmodelslim.app.naive_quantization.application import NaiveQuantizationApplication
        from msmodelslim.core.practice.interface import ScenarioTagMatch

        metadata = self._make_metadata(fa_quant=True, verified_model_types=["FLUX.1-dev"])
        result = NaiveQuantizationApplication.check_config(
            metadata,
            "FLUX.1-dev",
            QuantType.W8A8F8,
        )
        self.assertEqual(result, ScenarioTagMatch.MATCH)

    def test_check_config_w8a8f8_mismatch_without_fa_quant_label(self):
        """测试 w8a8f8 与未启用 fa_quant 的 label 不匹配"""
        from msmodelslim.app.naive_quantization.application import NaiveQuantizationApplication
        from msmodelslim.core.practice.interface import ScenarioTagMatch

        metadata = self._make_metadata(verified_model_types=["FLUX.1-dev"])
        result = NaiveQuantizationApplication.check_config(
            metadata,
            "FLUX.1-dev",
            QuantType.W8A8F8,
        )
        self.assertEqual(result, ScenarioTagMatch.NO_MATCH)

    def test_check_config_w8a8c8_matches_kv_cache_label(self):
        """测试 w8a8c8 与 kv_cache: True 的 label 匹配"""
        from msmodelslim.app.naive_quantization.application import NaiveQuantizationApplication
        from msmodelslim.core.practice.interface import ScenarioTagMatch

        metadata = self._make_metadata(kv_cache=True, verified_model_types=["Qwen3-32B"])
        result = NaiveQuantizationApplication.check_config(
            metadata,
            "Qwen3-32B",
            QuantType.W8A8C8,
        )
        self.assertEqual(result, ScenarioTagMatch.MATCH)


class TestGetBestPractice(TestNaiveQuantizationAppBase):
    """测试 get_best_practice"""

    def _make_app(self):
        from msmodelslim.app.naive_quantization.application import NaiveQuantizationApplication

        return NaiveQuantizationApplication(
            practice_manager=MagicMock(),
            quant_service=MagicMock(),
            model_factory=MagicMock(),
        )

    def _get_best_practice_with_mocked_yaml(self, model_type, apiversion, config_id="test_w8a8"):
        mock_model_adapter = MagicMock()
        mock_model_adapter.model_type = model_type
        mock_model_adapter.get_model_type.return_value = model_type
        app = self._make_app()
        mock_config = MagicMock()
        mock_config.apiversion = apiversion
        mock_config.metadata.config_id = config_id
        with (
            patch(
                "msmodelslim.app.naive_quantization.application.yaml_safe_load",
                return_value={},
            ),
            patch(
                "msmodelslim.app.naive_quantization.application.PracticeConfig.model_validate",
                return_value=mock_config,
            ),
        ):
            return app.get_best_practice(
                model_adapter=mock_model_adapter,
                config_path=Path("config.yaml"),
            )

    def test_get_best_practice_with_config_path(self):
        """测试指定 config_path 时直接返回配置"""
        from msmodelslim.app.naive_quantization.application import NaiveQuantizationApplication

        config_yaml = """apiversion: modelslim_v1
metadata:
  config_id: test_w8a8
  score: 90
  label:
    w_bit: 8
    a_bit: 8
    is_sparse: false
    kv_cache: false
spec: {}
"""
        old_umask = os.umask(0)
        try:
            os.umask(0o077)
            config_file = Path(self.temp_dir) / "config.yaml"
            config_file.write_text(config_yaml, encoding="utf-8")
        finally:
            os.umask(old_umask)

        mock_practice_manager = MagicMock()
        mock_quant_service = MagicMock()
        mock_model_factory = MagicMock()
        mock_model_adapter = MagicMock()

        app = NaiveQuantizationApplication(
            practice_manager=mock_practice_manager,
            quant_service=mock_quant_service,
            model_factory=mock_model_factory,
        )

        config = app.get_best_practice(
            model_adapter=mock_model_adapter,
            config_path=config_file,
        )

        self.assertEqual(config.metadata.config_id, "test_w8a8")
        self.assertEqual(config.metadata.label["w_bit"], 8)
        mock_practice_manager.iter_config.assert_not_called()

    def test_get_best_practice_requires_model_info_interface(self):
        """测试 get_best_practice 要求 model_adapter 实现 ModelInfoInterface"""
        from msmodelslim.app.naive_quantization.application import NaiveQuantizationApplication

        mock_practice_manager = MagicMock()
        mock_practice_manager.__contains__ = MagicMock(return_value=True)
        mock_quant_service = MagicMock()
        mock_model_factory = MagicMock()
        mock_model_adapter = MagicMock()  # 不实现 ModelInfoInterface

        app = NaiveQuantizationApplication(
            practice_manager=mock_practice_manager,
            quant_service=mock_quant_service,
            model_factory=mock_model_factory,
        )

        with self.assertRaises(ToDoError):
            app.get_best_practice(
                model_adapter=mock_model_adapter,
                quant_type=QuantType.W8A8,
            )

    def test_get_best_practice_raises_unsupported_error_when_transformers_and_vlm_apiversion(self):
        with self.assertRaises(UnsupportedError) as ctx:
            self._get_best_practice_with_mocked_yaml("transformers", "multimodal_vlm_modelslim_v1")

        self.assertIn("VLM", str(ctx.exception))
        self.assertIn("transformers", str(ctx.exception))

    def test_get_best_practice_raises_unsupported_error_when_transformers_and_sd_apiversion(self):
        with self.assertRaises(UnsupportedError) as ctx:
            self._get_best_practice_with_mocked_yaml("transformers", "multimodal_sd_modelslim_v1")

        self.assertIn("DiT", str(ctx.exception))
        self.assertIn("transformers", str(ctx.exception))

    def test_get_best_practice_returns_config_when_transformers_and_modelslim_v1(self):
        config = self._get_best_practice_with_mocked_yaml("transformers", "modelslim_v1")

        self.assertEqual(config.metadata.config_id, "test_w8a8")
        self.assertEqual(config.apiversion, "modelslim_v1")

    def test_get_best_practice_returns_config_when_other_model_type_and_vlm_apiversion(self):
        config = self._get_best_practice_with_mocked_yaml(
            "Qwen2.5-VL-7B-Instruct",
            "multimodal_vlm_modelslim_v1",
            config_id="vlm_ok",
        )

        self.assertEqual(config.metadata.config_id, "vlm_ok")
        self.assertEqual(config.apiversion, "multimodal_vlm_modelslim_v1")


class TestQuantParameterValidation(TestNaiveQuantizationAppBase):
    """测试 quant 方法参数校验"""

    def _make_app(self):
        from msmodelslim.app.naive_quantization.application import NaiveQuantizationApplication

        return NaiveQuantizationApplication(
            practice_manager=MagicMock(),
            quant_service=MagicMock(),
            model_factory=MagicMock(),
        )

    def test_quant_model_type_validation(self):
        """测试 model_type 参数校验"""
        app = self._make_app()
        with self.assertRaises(SchemaValidateError):
            app.quant(
                model_type=123,
                model_path=str(self.model_path),
                save_path=str(self.save_path),
            )

    def test_quant_device_type_validation(self):
        """测试 device_type 参数校验"""
        app = self._make_app()
        with self.assertRaises(SchemaValidateError):
            app.quant(
                model_type="Qwen2.5-7B",
                model_path=str(self.model_path),
                save_path=str(self.save_path),
                device_type="invalid",
            )

    def test_quant_quant_type_and_config_path_mutual_exclusion(self):
        """测试 quant_type 与 config_path 互斥"""
        app = self._make_app()
        old_umask = os.umask(0)
        try:
            os.umask(0o077)
            config_file = Path(self.temp_dir) / "config.yaml"
            config_file.write_text(
                "apiversion: modelslim_v1\nmetadata:\n  config_id: x\n  label: {}\nspec: {}", encoding="utf-8"
            )
        finally:
            os.umask(old_umask)

        with self.assertRaises(SchemaValidateError):
            app.quant(
                model_type="Qwen2.5-7B",
                model_path=str(self.model_path),
                save_path=str(self.save_path),
                quant_type=QuantType.W8A8,
                config_path=str(config_file),
            )

    def test_quant_trust_remote_code_validation(self):
        """测试 trust_remote_code 参数校验"""
        app = self._make_app()
        with self.assertRaises(SchemaValidateError):
            app.quant(
                model_type="Qwen2.5-7B",
                model_path=str(self.model_path),
                save_path=str(self.save_path),
                trust_remote_code="not_bool",
            )

    def test_quant_scenario_validation(self):
        """测试 tag 参数校验"""
        app = self._make_app()
        with self.assertRaises(SchemaValidateError):
            app.quant(
                model_type="Qwen2.5-7B",
                model_path=str(self.model_path),
                save_path=str(self.save_path),
                tag="not_a_list",
            )

    def test_quant_device_index_non_list_validation(self):
        app = self._make_app()

        with self.assertRaises(SchemaValidateError):
            app.quant(
                model_type="Qwen2.5-7B",
                model_path=str(self.model_path),
                save_path=str(self.save_path),
                device_type=DeviceType.CPU,
                device_index="0",
            )


class TestPrecheckIntegration(TestNaiveQuantizationAppBase):
    """测试量化应用组装通用预检规则"""

    def _make_app(self):
        from msmodelslim.app.naive_quantization.application import NaiveQuantizationApplication

        app = NaiveQuantizationApplication(
            practice_manager=MagicMock(),
            quant_service=MagicMock(),
            model_factory=MagicMock(),
        )
        app._quant = MagicMock()
        return app

    def test_quant_precheck_normalizes_parameters(self):
        from msmodelslim.app.naive_quantization import application as app_module

        app = self._make_app()
        with patch.object(app_module, "msmodelslim_config") as mock_config:
            with (
                patch("msmodelslim.utils.precheck.rules.check_read_permission"),
                patch("msmodelslim.utils.precheck.rules.check_write_permission"),
            ):
                with patch.object(app_module, "convert_to_writable_dir", side_effect=Path):
                    mock_config.precheck_items = {"quant_precheck_item": set()}
                    app.quant(
                        model_type="Qwen2.5-7B",
                        model_path=str(self.model_path),
                        save_path=str(self.save_path),
                        device_type=DeviceType.CPU,
                        device_index=None,
                        quant_type=QuantType.W8A8,
                        config_path=None,
                        trust_remote_code=False,
                        tag=[],
                    )

        args = app._quant.call_args.args
        self.assertEqual(args[1], self.model_path)
        self.assertEqual(args[2], self.save_path)
        self.assertIsNone(args[8])

    def test_quant_precheck_rejects_removed_quant_type(self):
        from msmodelslim.app.naive_quantization import application as app_module

        app = self._make_app()
        with patch.object(app_module, "msmodelslim_config") as mock_config:
            mock_config.precheck_items = {"quant_precheck_item": set()}
            with self.assertRaises(SchemaValidateError):
                app.quant(
                    model_type="Qwen2.5-7B",
                    model_path=str(self.model_path),
                    save_path=str(self.save_path),
                    device_type=DeviceType.CPU,
                    device_index=None,
                    quant_type="w8a8",
                    config_path=None,
                    trust_remote_code=False,
                    tag=None,
                )
        app._quant.assert_not_called()

    def test_quant_precheck_ignores_unknown_configured_item(self):
        from msmodelslim.app.naive_quantization import application as app_module

        app = self._make_app()
        with patch.object(app_module, "msmodelslim_config") as mock_config:
            mock_config.precheck_items = {"quant_precheck_item": {"path.permission"}}
            with (
                patch("msmodelslim.utils.precheck.rules.check_read_permission"),
                patch("msmodelslim.utils.precheck.rules.check_write_permission"),
                patch.object(app_module, "convert_to_writable_dir", side_effect=Path),
            ):
                app.quant(
                    model_type="Qwen2.5-7B",
                    model_path=str(self.model_path),
                    save_path=str(self.save_path),
                    device_type=DeviceType.CPU,
                    quant_type=QuantType.W8A8,
                )

        app._quant.assert_called_once()

    def test_quant_precheck_logs_precheck_banner(self):
        from msmodelslim.app.naive_quantization import application as app_module

        app = self._make_app()
        logger = MagicMock()
        with (
            patch.object(app_module, "msmodelslim_config") as mock_config,
            patch.object(app_module, "get_logger", lambda: logger),
        ):
            mock_config.precheck_items = {"quant_precheck_item": set()}
            with (
                patch("msmodelslim.utils.precheck.rules.check_read_permission"),
                patch("msmodelslim.utils.precheck.rules.check_write_permission"),
                patch.object(app_module, "convert_to_writable_dir", side_effect=Path),
            ):
                app.quant(
                    model_type="Qwen2.5-7B",
                    model_path=str(self.model_path),
                    save_path=str(self.save_path),
                    device_type=DeviceType.CPU,
                    quant_type=QuantType.W8A8,
                )

        info_messages = [call.args[0] for call in logger.info.call_args_list]
        # 预检段落标识需出现在所有预检结果之前，与后续 ANALYSE MODEL / QUANTIZE MODEL 段落一致。
        self.assertEqual(info_messages[0], "===========PRECHECK===========")

    def test_quant_precheck_runs_configured_optional_disk_rule(self):
        from msmodelslim.app.naive_quantization import application as app_module

        app = self._make_app()
        with patch.object(app_module, "msmodelslim_config") as mock_config:
            mock_config.precheck_items = {"quant_precheck_item": {"disk"}}
            with self.assertRaises(EnvError):
                app.quant(
                    model_type="Qwen2.5-7B",
                    model_path=str(self.model_path),
                    save_path=str(self.save_path),
                    device_type=DeviceType.CPU,
                    quant_type=QuantType.W8A8,
                )

        app._quant.assert_not_called()

    def test_quant_precheck_passes_occupancy_threshold_at_registration(self):
        from msmodelslim.app.naive_quantization import application as app_module

        app = self._make_app()
        bind_spy = MagicMock(wraps=app_module.npu_occupancy_check.bind)
        with (
            patch.object(app_module, "msmodelslim_config") as mock_config,
            patch.object(app_module.npu_occupancy_check, "bind", bind_spy),
            patch("msmodelslim.utils.precheck.rules.check_read_permission"),
            patch("msmodelslim.utils.precheck.rules.check_write_permission"),
            patch.object(app_module, "convert_to_writable_dir", side_effect=Path),
        ):
            mock_config.precheck_items = {"quant_precheck_item": set()}
            app.quant(
                model_type="Qwen2.5-7B",
                model_path=str(self.model_path),
                save_path=str(self.save_path),
                device_type=DeviceType.CPU,
                quant_type=QuantType.W8A8,
            )

        bind_spy.assert_called_once_with(
            DeviceType.CPU,
            0,
            free_ratio_min=app_module.NPU_FREE_MEMORY_RATIO_MIN,
            rule_id="npu.NPU:0",
        )

    def test_quant_precheck_treats_model_type_as_optional(self):
        from msmodelslim.app.naive_quantization import application as app_module

        app = self._make_app()
        bind_spy = MagicMock(wraps=app_module.string_field_check.bind)
        with (
            patch.object(app_module, "msmodelslim_config") as mock_config,
            patch.object(app_module.string_field_check, "bind", bind_spy),
            patch(
                "msmodelslim.core.quant_service.modelslim_convert.config_detect.is_modelslim_convert_config",
                return_value=True,
            ),
            patch("msmodelslim.utils.precheck.rules.check_read_permission"),
            patch("msmodelslim.utils.precheck.rules.convert_to_readable_file", side_effect=Path),
            patch("msmodelslim.utils.precheck.rules.check_write_permission"),
            patch.object(app_module, "convert_to_writable_dir", side_effect=Path),
        ):
            mock_config.precheck_items = {"quant_precheck_item": set()}
            app.quant(
                model_type=None,
                model_path=str(self.model_path),
                save_path=str(self.save_path),
                device_type=DeviceType.CPU,
                config_path=str(Path(self.temp_dir) / "convert.yaml"),
            )

        app._quant.assert_called_once()
        model_type_binding = next(call for call in bind_spy.call_args_list if call.args[1] == "model_type")
        self.assertIsNone(model_type_binding.args[0])
        self.assertTrue(model_type_binding.kwargs["skip_when_none"])

    def test_quant_precheck_registers_occupancy_for_valid_unique_indices(self):
        from msmodelslim.app.naive_quantization import application as app_module

        app = self._make_app()
        bind_spy = MagicMock(wraps=app_module.npu_occupancy_check.bind)
        with (
            patch.object(app_module, "msmodelslim_config") as mock_config,
            patch.object(app_module.npu_occupancy_check, "bind", bind_spy),
        ):
            mock_config.precheck_items = {"quant_precheck_item": set()}
            # npu.index rejects the mixed list, but occupancy registration already happened.
            with self.assertRaises(SchemaValidateError):
                app.quant(
                    model_type="Qwen2.5-7B",
                    model_path=str(self.model_path),
                    save_path=str(self.save_path),
                    device_type=DeviceType.NPU,
                    device_index=[2, 0, 2, -1, True, "1"],
                )

        self.assertEqual(
            [call.kwargs["rule_id"] for call in bind_spy.call_args_list],
            ["npu.NPU:2", "npu.NPU:0"],
        )

    def test_quant_precheck_failure_does_not_create_save_path(self):
        from msmodelslim.app.naive_quantization import application as app_module

        app = self._make_app()
        save_path = Path(self.temp_dir) / "new" / "save"
        with (
            patch.object(app_module, "msmodelslim_config") as mock_config,
            patch("msmodelslim.utils.precheck.rules.get_valid_path", side_effect=lambda value: value),
        ):
            mock_config.precheck_items = {"quant_precheck_item": set()}
            with self.assertRaises(SchemaValidateError):
                app.quant(
                    model_type="Qwen2.5-7B",
                    model_path=str(self.model_path),
                    save_path=str(save_path),
                    device_type=DeviceType.CPU,
                    quant_type=QuantType.W8A8,
                    trust_remote_code="invalid",
                )

        self.assertFalse(save_path.exists())
        app._quant.assert_not_called()

    def test_invalid_device_type_fails_only_quant_arguments(self):
        from msmodelslim.app.naive_quantization import application as app_module
        from msmodelslim.utils.precheck import NPU_INDEX_GROUP, PrecheckManager, PrecheckStatus, npu_index_check

        manager = PrecheckManager()
        manager.register(app_module.check_quant_arguments.bind("invalid", None, None, False, None, "model"))
        manager.register(npu_index_check.bind("invalid", None))

        report = manager.run()

        self.assertEqual(
            [result.check_id for result in report.results if result.status == PrecheckStatus.FAIL],
            ["quant.arguments"],
        )
        npu_result = next(result for result in report.results if result.group == NPU_INDEX_GROUP)
        self.assertEqual(npu_result.status, PrecheckStatus.SKIP)

    def test_quant_argument_group_is_stable(self):
        from msmodelslim.app.naive_quantization import application as app_module

        self.assertEqual(app_module.QUANT_ARGUMENT_GROUP, "quant.arguments")
        self.assertEqual(app_module.check_quant_arguments.group, app_module.QUANT_ARGUMENT_GROUP)

    def test_quant_creates_save_path_after_all_prechecks_pass(self):
        from msmodelslim.app.naive_quantization import application as app_module

        app = self._make_app()
        save_path = Path(self.temp_dir) / "new" / "save"
        with patch.object(app_module, "msmodelslim_config") as mock_config:
            mock_config.precheck_items = {"quant_precheck_item": set()}
            app.quant(
                model_type="Qwen2.5-7B",
                model_path=str(self.model_path),
                save_path=str(save_path),
                device_type=DeviceType.CPU,
                quant_type=QuantType.W8A8,
            )

        self.assertTrue(save_path.is_dir())
        self.assertEqual(app._quant.call_args.args[2], save_path)


if __name__ == "__main__":
    unittest.main()
