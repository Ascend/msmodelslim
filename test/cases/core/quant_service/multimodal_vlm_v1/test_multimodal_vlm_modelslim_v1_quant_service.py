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
"""
# pylint: disable=duplicate-code

from pathlib import Path
from typing import List
from unittest.mock import Mock, MagicMock, patch, call

import pytest

from msmodelslim.core.const import RunnerType, DeviceType
from msmodelslim.core.quant_service.interface import BaseQuantConfig
from msmodelslim.core.quant_service.multimodal_vlm_v1.quant_config import MultimodalVLMModelslimV1QuantConfig
from msmodelslim.core.quant_service.multimodal_vlm_v1.quant_service import (
    MultimodalVLMModelslimV1QuantService,
    MultimodalVLMModelslimV1QuantServiceConfig,
)
from msmodelslim.core.runner.pipeline_interface import PipelineInterface
from msmodelslim.core.runner.optional_interface import LayerWiseOffloadOptionalInterface
from msmodelslim.utils.exception import SchemaValidateError
from msmodelslim.core.context import IContextFactory


class TestMultimodalVLMModelslimV1QuantService:  # pylint: disable=attribute-defined-outside-init
    """测试 `MultimodalVLMModelslimV1QuantService` 的关键行为。"""

    def setup_method(self):
        # 轻量级 dataset_loader mock，仅暴露测试需要的接口
        self.dataset_loader = Mock()
        # quant_process 会对加载结果做非空 text 校验，mock 需返回带 text 的样本列表
        self.mock_dataset = [{"text": "hello"}]
        self.dataset_loader.get_dataset_by_name.return_value = self.mock_dataset

        # QuantService 初始化使用 QuantServiceConfig（仅 apiversion），不是 QuantConfig
        self.quant_service_config = MultimodalVLMModelslimV1QuantServiceConfig()
        self.context_factory = Mock(spec=IContextFactory)
        self.service = MultimodalVLMModelslimV1QuantService(
            self.quant_service_config, self.dataset_loader, self.context_factory
        )

        # 模型适配器使用 PipelineInterface 的 spec，确保类型检查分支可被命中
        self.model_adapter = Mock(spec=PipelineInterface)

        # 公共参数
        self.save_path = Path("/tmp/quant")
        self.device = DeviceType.NPU

    def test_backend_name_constant(self):
        """验证 backend_name 常量值是否符合预期。"""
        assert self.service.backend_name == "multimodal_vlm_modelslim_v1"

    @pytest.mark.parametrize(
        "case_name, quant_cfg, model_adap, save_p, device, expected_msg",
        [
            (
                "invalid_quant_config",
                Mock(),  # 非 BaseQuantConfig
                Mock(spec=PipelineInterface),
                Path("test"),
                DeviceType.NPU,
                "task is not a BaseTask",
            ),
            (
                "invalid_model_adapter",
                Mock(spec=BaseQuantConfig),
                Mock(),  # 非 PipelineInterface
                Path("test"),
                DeviceType.NPU,
                "model_adapter must be a PipelineInterface",
            ),
            (
                "invalid_save_path",
                Mock(spec=BaseQuantConfig),
                Mock(spec=PipelineInterface),
                "not-a-path",  # type: ignore[arg-type]
                DeviceType.NPU,
                "save_path must be a Path or None",
            ),
            (
                "invalid_device",
                Mock(spec=BaseQuantConfig),
                Mock(spec=PipelineInterface),
                Path("test"),
                "invalid-device",  # type: ignore[arg-type]
                "device must be a DeviceType",
            ),
        ],
    )
    def test_quantize_invalid_params(self, case_name, quant_cfg, model_adap, save_p, device, expected_msg):
        """参数化测试：验证所有非法参数场景都会抛出 SchemaValidateError。"""
        with pytest.raises(SchemaValidateError) as exc_info:
            self.service.quantize(quant_cfg, model_adap, save_p, device=device)
        assert expected_msg in str(exc_info.value)

    @patch.object(MultimodalVLMModelslimV1QuantConfig, "from_base")
    def test_quantize_normal_flow_calls_quant_process(self, mock_from_base):
        """验证量化主入口在校验通过时会正确调用 `quant_process`。"""
        converted_cfg = Mock()
        mock_from_base.return_value = converted_cfg

        self.service.quant_process = Mock(return_value="ok")

        base_cfg = Mock(spec=BaseQuantConfig)
        result = self.service.quantize(
            base_cfg,
            self.model_adapter,
            self.save_path,
            device=self.device,
            device_indices=[0],
        )

        assert result == "ok"
        mock_from_base.assert_called_once_with(base_cfg)
        self.service.quant_process.assert_called_once_with(
            converted_cfg,
            self.model_adapter,
            self.save_path,
            self.device,
            [0],
        )

    @patch("msmodelslim.core.quant_service.multimodal_vlm_v1.quant_service.seed_all")
    def test_quant_process_raises_when_loaded_sample_text_missing(self, mock_seed_all):
        """加载后的校准样本若缺少非空 text，应在进入 runner 前失败。"""
        from msmodelslim.utils.exception import InvalidDatasetError

        self.dataset_loader.get_dataset_by_name.return_value = [{"text": ""}]
        spec = Mock()
        spec.dataset = "empty_text_dataset"
        spec.default_text = "unused"
        spec.process = []
        spec.save = []
        spec.runner = RunnerType.LAYER_WISE
        quant_cfg = Mock()
        quant_cfg.spec = spec

        with pytest.raises(InvalidDatasetError) as exc_info:
            self.service.quant_process(
                quant_config=quant_cfg,
                model_adapter=self.model_adapter,
                save_path=None,
                device=DeviceType.CPU,
            )
        assert "text data is missing" in str(exc_info.value)

    @pytest.mark.parametrize(
        "runner_value, device_indices, expected_type",
        [
            (RunnerType.LAYER_WISE, None, RunnerType.LAYER_WISE),
            (RunnerType.LAYER_WISE, [0, 1], RunnerType.LAYER_WISE),
            (RunnerType.DP_LAYER_WISE, [0, 1], RunnerType.DP_LAYER_WISE),
            (RunnerType.AUTO, [0, 1, 2], RunnerType.DP_LAYER_WISE),
            (RunnerType.AUTO, [0], RunnerType.LAYER_WISE),
            (RunnerType.AUTO, None, RunnerType.LAYER_WISE),
            (RunnerType.MODEL_WISE, None, RunnerType.LAYER_WISE),
            ("unknown", None, RunnerType.LAYER_WISE),
        ],
    )
    @patch("msmodelslim.core.quant_service.multimodal_vlm_v1.quant_service.get_logger")
    def test_choose_runner_type_with_logging(self, mock_get_logger, runner_value, device_indices, expected_type):
        """覆盖 `_choose_runner_type` 的 DP / 单卡 / model_wise 回退分支。"""
        quant_cfg = MagicMock()
        quant_cfg.spec.runner = runner_value

        adapter = Mock(spec=PipelineInterface)
        result = self.service._choose_runner_type(quant_cfg, adapter, device_indices)

        assert result == expected_type
        mock_logger = mock_get_logger.return_value
        assert mock_logger.info.called or mock_logger.warning.called

    @patch("msmodelslim.core.quant_service.multimodal_vlm_v1.quant_service.get_logger")
    @patch("msmodelslim.core.quant_service.multimodal_vlm_v1.quant_service.LayerWiseRunner")
    @patch("msmodelslim.core.quant_service.multimodal_vlm_v1.quant_service.torch")
    @patch("msmodelslim.core.quant_service.multimodal_vlm_v1.quant_service.seed_all")
    def test_quant_process_full_flow_with_save_path_and_npu(
        self,
        mock_seed_all,
        mock_torch,
        mock_runner_cls,
        mock_get_logger,
    ):
        """
        完整流程测试（带 save_path & NPU 设备，单卡 layer_wise）：
        - 校验随机种子设置
        - 校验 NPU 编译模式配置
        - 校验数据集加载与 default_text 赋值
        - 校验持久化配置与 Processor 注册
        - 校验最终的 runner.run 调用参数（含 device_indices）
        """
        # 准备 quant_config.spec 的各项字段
        process_cfgs: List[Mock] = [Mock(), Mock()]
        save_cfgs: List[Mock] = [Mock(), Mock()]

        spec = Mock()
        spec.dataset = "mock_dataset_path"
        spec.default_text = "default prompt"
        spec.process = process_cfgs
        spec.save = save_cfgs
        spec.runner = RunnerType.LAYER_WISE

        quant_cfg = Mock()
        quant_cfg.spec = spec

        # LayerWiseRunner 返回的 runner mock
        runner = Mock()
        mock_runner_cls.return_value = runner

        # 调用被测方法
        result = self.service.quant_process(
            quant_config=quant_cfg,
            model_adapter=self.model_adapter,
            save_path=self.save_path,
            device=self.device,
            device_indices=[0],
        )

        # quant_process 不需要返回值，这里只确认流程顺利结束
        assert result is None

        # 随机种子设置
        mock_seed_all.assert_called_once()

        # NPU 编译模式设置
        mock_torch.npu.set_compile_mode.assert_called_once_with(jit_compile=False)

        # 数据集加载与 default_text 设置
        assert self.dataset_loader.default_text == "default prompt"
        self.dataset_loader.get_dataset_by_name.assert_called_once_with("mock_dataset_path")

        # save_path 有效时，save_cfg 需要被更新目录，并附加到最终 process 列表
        for save_cfg in save_cfgs:
            save_cfg.set_save_directory.assert_called_once_with(self.save_path)

        # Runner 创建与 Processor 注册
        mock_runner_cls.assert_called_once_with(adapter=self.model_adapter, offload_device="cpu")
        expected_calls = [call(processor_cfg=cfg) for cfg in process_cfgs + save_cfgs]
        runner.add_processor.assert_has_calls(expected_calls, any_order=False)

        # 最终量化执行（透传 device_indices）
        runner.run.assert_called_once_with(calib_data=self.mock_dataset, device=self.device, device_indices=[0])
        self.context_factory.create.assert_called_once_with(is_distributed=False)
        # 确保结束日志被打印
        assert mock_get_logger.return_value.info.called

    @patch("msmodelslim.core.quant_service.multimodal_vlm_v1.quant_service.get_logger")
    @patch("msmodelslim.core.runner.dp_layer_wise_runner.DPLayerWiseRunner")
    @patch("msmodelslim.core.quant_service.multimodal_vlm_v1.quant_service.torch")
    @patch("msmodelslim.core.quant_service.multimodal_vlm_v1.quant_service.seed_all")
    def test_quant_process_auto_multi_device_uses_dp_runner(
        self,
        mock_seed_all,
        mock_torch,
        mock_dp_runner_cls,
        mock_get_logger,
    ):
        """runner=auto 且多卡 device_indices 时创建 DPLayerWiseRunner 并开启分布式 context。"""
        process_cfgs: List[Mock] = [Mock()]
        save_cfgs: List[Mock] = []

        spec = Mock()
        spec.dataset = "mock_dataset_path"
        spec.default_text = "default prompt"
        spec.process = process_cfgs
        spec.save = save_cfgs
        spec.runner = RunnerType.AUTO

        quant_cfg = Mock()
        quant_cfg.spec = spec

        runner = Mock()
        mock_dp_runner_cls.return_value = runner

        self.service.quant_process(
            quant_config=quant_cfg,
            model_adapter=self.model_adapter,
            save_path=None,
            device=self.device,
            device_indices=[0, 1, 2, 3],
        )

        mock_dp_runner_cls.assert_called_once_with(adapter=self.model_adapter, offload_device="cpu")
        runner.run.assert_called_once_with(
            calib_data=self.mock_dataset,
            device=self.device,
            device_indices=[0, 1, 2, 3],
        )
        self.context_factory.create.assert_called_once_with(is_distributed=True)

    @patch("msmodelslim.core.quant_service.multimodal_vlm_v1.quant_service.get_logger")
    @patch("msmodelslim.core.runner.dp_layer_wise_runner.DPLayerWiseRunner")
    @patch("msmodelslim.core.quant_service.multimodal_vlm_v1.quant_service.torch")
    @patch("msmodelslim.core.quant_service.multimodal_vlm_v1.quant_service.seed_all")
    def test_quant_process_uses_dp_runner_when_dp_layer_wise(
        self,
        mock_seed_all,
        mock_torch,
        mock_dp_runner_cls,
        mock_get_logger,
    ):
        """runner=dp_layer_wise 时创建 DPLayerWiseRunner，与 modelslim_v1 行为对齐。"""
        process_cfgs: List[Mock] = [Mock()]
        save_cfgs: List[Mock] = []

        spec = Mock()
        spec.dataset = "mock_dataset_path"
        spec.default_text = "default prompt"
        spec.process = process_cfgs
        spec.save = save_cfgs
        spec.runner = RunnerType.DP_LAYER_WISE

        quant_cfg = Mock()
        quant_cfg.spec = spec

        runner = Mock()
        mock_dp_runner_cls.return_value = runner

        self.service.quant_process(
            quant_config=quant_cfg,
            model_adapter=self.model_adapter,
            save_path=None,
            device=self.device,
            device_indices=[0, 1],
        )

        mock_dp_runner_cls.assert_called_once_with(adapter=self.model_adapter, offload_device="cpu")
        runner.run.assert_called_once_with(
            calib_data=self.mock_dataset,
            device=self.device,
            device_indices=[0, 1],
        )
        self.context_factory.create.assert_called_once_with(is_distributed=True)

    @patch("msmodelslim.core.quant_service.multimodal_vlm_v1.quant_service.get_logger")
    @patch("msmodelslim.core.quant_service.multimodal_vlm_v1.quant_service.LayerWiseRunner")
    @patch("msmodelslim.core.quant_service.multimodal_vlm_v1.quant_service.seed_all")
    def test_quant_process_without_save_path_and_unknown_runner(
        self,
        mock_seed_all,
        mock_runner_cls,
        mock_get_logger,
    ):
        """
        测试以下组合场景：
        - device 为 CPU（跳过 NPU 特有逻辑）
        - runner 类型未知（回退 layer_wise）
        - save_path 为 None（不注册 save processor）
        """
        process_cfgs: List[Mock] = [Mock()]
        save_cfgs: List[Mock] = [Mock()]

        spec = Mock()
        spec.dataset = "cpu_dataset"
        spec.default_text = "cpu prompt"
        spec.process = process_cfgs
        spec.save = save_cfgs
        spec.runner = "not-layer-wise"

        quant_cfg = Mock()
        quant_cfg.spec = spec

        runner = Mock()
        mock_runner_cls.return_value = runner
        cpu_device = DeviceType.CPU

        # 注意：这里不提供 save_path
        self.service.quant_process(
            quant_config=quant_cfg,
            model_adapter=self.model_adapter,
            save_path=None,
            device=cpu_device,
            device_indices=None,
        )

        # CPU 场景下，seed_all 仍应调用
        mock_seed_all.assert_called_once()

        # save_path 为 None，save_cfg 不应触发 set_save_directory
        for save_cfg in save_cfgs:
            save_cfg.set_save_directory.assert_not_called()

        # Runner 仍然需要创建并执行
        mock_runner_cls.assert_called_once_with(adapter=self.model_adapter, offload_device="cpu")
        runner.add_processor.assert_has_calls(
            [call(processor_cfg=cfg) for cfg in process_cfgs],
            any_order=False,
        )
        runner.run.assert_called_once_with(calib_data=self.mock_dataset, device=cpu_device, device_indices=None)

    @staticmethod
    def _make_quant_config_for_offload_case():
        spec = Mock()
        spec.dataset = "cpu_dataset"
        spec.default_text = "cpu prompt"
        spec.process = []
        spec.save = []
        spec.runner = "layer_wise"

        quant_cfg = Mock()
        quant_cfg.spec = spec
        return quant_cfg

    @staticmethod
    def _make_adapter_for_offload_case(preferred_offload=None, with_optional_interface=True):
        def handle_dataset(*_args, **_kwargs):
            return []

        def init_model(*_args, **_kwargs):
            return Mock()

        def generate_model_visit(*_args, **_kwargs):
            yield from ()

        def generate_model_forward(*_args, **_kwargs):
            yield from ()

        def enable_kv_cache(*_args, **_kwargs):
            return None

        attrs = {
            "model_type": property(lambda self: "test_model"),
            "model_path": property(lambda self: Path("/tmp/test_model")),
            "trust_remote_code": property(lambda self: False),
            "handle_dataset": handle_dataset,
            "init_model": init_model,
            "generate_model_visit": generate_model_visit,
            "generate_model_forward": generate_model_forward,
            "enable_kv_cache": enable_kv_cache,
        }
        if with_optional_interface:
            attrs["get_layer_wise_offload_device"] = lambda self: preferred_offload

        bases = (
            (PipelineInterface, LayerWiseOffloadOptionalInterface) if with_optional_interface else (PipelineInterface,)
        )
        adapter_cls = type("AdapterForOffloadCase", bases, attrs)
        return adapter_cls()  # pylint: disable=abstract-class-instantiated

    @pytest.mark.parametrize(
        "preferred_offload, with_optional_interface, expected_offload, expect_invalid_warning",
        [
            ("meta", True, "meta", False),
            ("gpu", True, "cpu", True),
            ("", True, "cpu", False),
            (None, False, "cpu", False),
        ],
    )
    @patch("msmodelslim.core.quant_service.multimodal_vlm_v1.quant_service.get_logger")
    @patch("msmodelslim.core.quant_service.multimodal_vlm_v1.quant_service.LayerWiseRunner")
    @patch("msmodelslim.core.quant_service.multimodal_vlm_v1.quant_service.seed_all")
    def test_quant_process_when_model_adapter_offload_setting_changes_then_runner_uses_expected_device(
        self,
        mock_seed_all,
        mock_runner_cls,
        mock_get_logger,
        preferred_offload,
        with_optional_interface,
        expected_offload,
        expect_invalid_warning,
    ):
        quant_cfg = self._make_quant_config_for_offload_case()
        adapter = self._make_adapter_for_offload_case(
            preferred_offload=preferred_offload,
            with_optional_interface=with_optional_interface,
        )

        runner = Mock()
        mock_runner_cls.return_value = runner

        self.service.quant_process(
            quant_config=quant_cfg,
            model_adapter=adapter,
            save_path=None,
            device=DeviceType.CPU,
            device_indices=None,
        )

        mock_seed_all.assert_called_once()
        mock_runner_cls.assert_called_once()
        assert mock_runner_cls.call_args.kwargs["offload_device"] == expected_offload

        mock_logger = mock_get_logger.return_value
        has_invalid_warning = any(
            "Invalid offload device" in str(args[0]) for args, _ in mock_logger.warning.call_args_list
        )
        assert has_invalid_warning is expect_invalid_warning
