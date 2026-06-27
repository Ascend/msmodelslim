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

import functools
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, call

import pytest

from msmodelslim.core.const import DeviceType
from msmodelslim.core.quant_service.interface import BaseQuantConfig
from msmodelslim.utils.exception import UnsupportedError
from msmodelslim.core.quant_service.multimodal_sd_v1.legacy_pipeline_interface import (
    LegacyMultimodalPipelineInterface,
)
from msmodelslim.core.quant_service.multimodal_sd_v1.quant_config import (
    DumpConfig,
    MultimodalSDModelslimV1QuantConfig,
    MultimodalPipelineInterface,
    MultimodalSDConfig,
    SchemaValidateError,
)
from msmodelslim.core.quant_service.multimodal_sd_v1.quant_service import (
    MultimodalSDModelslimV1QuantService,
    MultimodalSDModelslimV1QuantServiceConfig,
)
from msmodelslim.core.context import IContextFactory


class TestQuantProcessComplete:  # pylint: disable=attribute-defined-outside-init
    mock_model = None
    mock_model1 = None
    mock_model2 = None

    @pytest.fixture(autouse=True)
    def setup_method(self, tmp_path):
        # 基础服务与核心依赖
        self.dataset_loader = Mock()

        # 1. 简化配置层级：用MagicMock自动支持属性访问
        self.mock_quant_spec = MagicMock()
        # 量化配置核心属性
        self.legacy_inference_raw = {"sample_steps": 40}
        self.mock_quant_spec.multimodal_sd_config = MultimodalSDConfig(
            dump_config=DumpConfig(capture_mode='args', dump_data_dir="/tmp/test_dump", enable_dump=True),
            model_config=self.legacy_inference_raw,
        )
        # 处理/保存配置（2个实例，覆盖多配置场景）
        self.mock_quant_spec.process = [Mock(), Mock()]
        self.mock_quant_spec.save = [Mock(), Mock()]
        self.mock_quant_spec.runner = "layer_wise"

        # 2. 只保留BaseQuantConfig的spec（推荐，符合实际类型约束）
        self.quant_config = Mock(spec=BaseQuantConfig)
        self.quant_config.apiversion = "multimodal_sd_modelslim_v1"
        # 给quant_config添加spec属性，模拟原意图（关联mock_quant_spec）
        self.quant_config.spec = self.mock_quant_spec

        # QuantService 初始化使用 QuantServiceConfig（仅 apiversion），不是 QuantConfig
        self.quant_service_config = MultimodalSDModelslimV1QuantServiceConfig()
        self.context_factory = Mock(spec=IContextFactory)
        self.service = MultimodalSDModelslimV1QuantService(
            self.quant_service_config, self.dataset_loader, self.context_factory
        )

        # 模型适配器：走 Legacy 编排路径（load_pipeline / apply_quantization / enable_dump 交互）
        self.model_adapter = Mock(spec=LegacyMultimodalPipelineInterface)
        self.mock_model = Mock()
        self.mock_model1 = Mock()
        self.mock_model2 = Mock()
        self.model_adapter.init_model.return_value = {'': self.mock_model}
        self.model_adapter.model_args = Mock()
        self.model_adapter.model_args.task_config = ''
        self.model_adapter.transformer = Mock()

        # 3. 测试固定参数
        self.save_path = Path("/tmp/test_save")
        self.device = DeviceType.NPU

    def test_backend_name(self):
        """验证后端名称"""
        assert self.service.backend_name == "multimodal_sd_modelslim_v1"

    @pytest.mark.parametrize(
        "case",
        [
            # 参数化合并4个无效参数测试
            (
                "invalid_quant_config",
                Mock(),
                Mock(spec=MultimodalPipelineInterface),
                Path("test"),
                "task must be a BaseTask",
            ),
            (
                "invalid_model_adapter",
                Mock(spec=BaseQuantConfig),
                Mock(),
                Path("test"),
                "model must implement LegacyMultimodalPipelineInterface or MultimodalPipelineInterface",
            ),
            (
                "invalid_save_path",
                Mock(spec=BaseQuantConfig),
                Mock(spec=MultimodalPipelineInterface),
                "invalid_path",
                "save_path must be a Path or None",
            ),
            (
                "invalid_device",
                Mock(spec=BaseQuantConfig),
                Mock(spec=MultimodalPipelineInterface),
                Path("test"),
                "device must be a DeviceType",
            ),
        ],
    )
    def test_quantize_invalid_params(self, case):
        """参数化测试：合并所有无效参数场景"""
        name, quant_cfg, model_adap, save_p, err_msg = case
        with pytest.raises(SchemaValidateError) as excinfo:
            # 只有invalid_device场景传入无效device，其他用合法DeviceType
            device = "invalid" if name == "invalid_device" else DeviceType.NPU
            self.service.quantize(quant_cfg, model_adap, save_p, device=device)
        assert err_msg in str(excinfo.value)

    @patch.object(MultimodalSDModelslimV1QuantConfig, 'from_base')
    def test_quantize_normal_flow(self, mock_from_base):
        """测试正常量化流程"""
        mock_from_base.return_value = self.quant_config
        self.service.quant_process = Mock(return_value="result")

        # 执行测试
        self.service.quantize(Mock(spec=BaseQuantConfig), self.model_adapter, self.save_path, self.device)

        # 核心验证：配置转换+流程调用
        mock_from_base.assert_called_once()
        self.service.quant_process.assert_called_once_with(
            self.quant_config, self.model_adapter, self.save_path, self.device
        )

    @patch("msmodelslim.core.quant_service.multimodal_sd_v1.quant_service.load_cached_data_for_models")
    @patch("msmodelslim.core.quant_service.multimodal_sd_v1.quant_service.to_device")
    @patch("msmodelslim.core.quant_service.multimodal_sd_v1.quant_service.LayerWiseRunner")
    def test_quant_process_full_flow(self, mock_runner_cls, mock_to_device, mock_load_cache):
        """简化完整流程测试：聚焦核心步骤"""
        # 1. 简化依赖Mock行为
        self.model_adapter.init_model.return_value = {'model1': self.mock_model1, 'model2': self.mock_model2}
        self.model_adapter.model_args.task_config = 'mock'
        mock_load_cache.return_value = "raw_calib_data"
        calib_data = {
            'model1': '/tmp/test_dump/calib_data_mock_model1.pth',
            'model2': '/tmp/test_dump/calib_data_mock_model2.pth',
        }
        mock_to_device.return_value = calib_data
        mock_runner = Mock()
        mock_runner_cls.return_value = mock_runner

        # 2. 执行测试
        self.service.quant_process(self.quant_config, self.model_adapter, self.save_path, self.device)

        # 3. 核心步骤验证（按执行顺序）
        # 模型配置与加载
        self.model_adapter.set_model_args.assert_called_once_with(self.legacy_inference_raw)
        self.model_adapter.load_pipeline.assert_called_once()
        # 数据加载与设备转换（路径用 Path 规范化，避免 Windows/Unix 分隔符差异）
        expected_pth_list = {
            'model1': Path('/tmp/test_dump/calib_data_mock_model1.pth'),
            'model2': Path('/tmp/test_dump/calib_data_mock_model2.pth'),
        }
        model = {'model1': self.mock_model1, 'model2': self.mock_model2}
        assert mock_load_cache.call_count == 1
        call_kw = mock_load_cache.call_args[1]
        actual_pth_list = {k: Path(v).as_posix() for k, v in call_kw["pth_file_path_list"].items()}
        assert actual_pth_list == {k: Path(v).as_posix() for k, v in expected_pth_list.items()}
        assert call_kw["generate_func"] == self.model_adapter.run_calib_inference
        assert call_kw["models"] == model
        assert call_kw["dump_config"] == self.mock_quant_spec.multimodal_sd_config.dump_config
        mock_to_device.assert_called_once_with("raw_calib_data", self.device.value)
        # 保存配置与Runner处理
        expert_save_path1 = self.save_path.joinpath("model1")
        expert_save_path2 = self.save_path.joinpath("model2")
        for save_cfg in self.mock_quant_spec.save:
            save_cfg.set_save_directory.assert_has_calls(
                [call(expert_save_path1), call(expert_save_path2)], any_order=False
            )
        mock_runner.add_processor.assert_has_calls(
            [call(processor_cfg=cfg) for cfg in self.mock_quant_spec.process], any_order=False
        )
        # 量化应用（验证partial函数参数）
        partial_func = self.model_adapter.apply_quantization.call_args[0][0]
        assert isinstance(partial_func, functools.partial)
        assert partial_func.func == mock_runner.run
        assert partial_func.keywords == {
            "calib_data": "/tmp/test_dump/calib_data_mock_model2.pth",
            "device": self.device,
            "model": self.mock_model2,
        }

    @patch("msmodelslim.core.quant_service.multimodal_sd_v1.quant_service.load_cached_data_for_models")
    @patch("msmodelslim.core.quant_service.multimodal_sd_v1.quant_service.to_device")
    @patch("msmodelslim.core.quant_service.multimodal_sd_v1.quant_service.LayerWiseRunner")
    def test_quant_process_full_flow_not_moe(self, mock_runner_cls, mock_to_device, mock_load_cache):
        """简化完整流程测试：聚焦核心步骤"""
        # 1. 简化依赖Mock行为
        self.model_adapter.init_model.return_value = {'': self.mock_model1}
        self.model_adapter.model_args.task_config = ''
        mock_load_cache.return_value = "raw_calib_data"
        calib_data = {'': '/tmp/test_dump/calib_data__.pth'}
        mock_to_device.return_value = calib_data
        mock_runner = Mock()
        mock_runner_cls.return_value = mock_runner

        # 2. 执行测试
        self.service.quant_process(self.quant_config, self.model_adapter, self.save_path, self.device)

        # 3. 核心步骤验证（按执行顺序）
        # 模型配置与加载
        self.model_adapter.set_model_args.assert_called_once_with(self.legacy_inference_raw)
        self.model_adapter.load_pipeline.assert_called_once()
        # 数据加载与设备转换（路径用 Path 规范化，避免 Windows/Unix 分隔符差异）
        expected_pth_list = {'': Path('/tmp/test_dump/calib_data__.pth')}
        model = {'': self.mock_model1}
        assert mock_load_cache.call_count == 1
        call_kw = mock_load_cache.call_args[1]
        actual_pth_list = {k: Path(v).as_posix() for k, v in call_kw["pth_file_path_list"].items()}
        assert actual_pth_list == {k: Path(v).as_posix() for k, v in expected_pth_list.items()}
        assert call_kw["generate_func"] == self.model_adapter.run_calib_inference
        assert call_kw["models"] == model
        assert call_kw["dump_config"] == self.mock_quant_spec.multimodal_sd_config.dump_config
        mock_to_device.assert_called_once_with("raw_calib_data", self.device.value)
        # 保存配置与Runner处理
        expert_save_path1 = self.save_path
        for save_cfg in self.mock_quant_spec.save:
            save_cfg.set_save_directory.assert_has_calls([call(expert_save_path1)], any_order=False)
        mock_runner.add_processor.assert_has_calls(
            [call(processor_cfg=cfg) for cfg in self.mock_quant_spec.process], any_order=False
        )
        # 量化应用（验证partial函数参数）
        partial_func = self.model_adapter.apply_quantization.call_args[0][0]
        assert isinstance(partial_func, functools.partial)
        assert partial_func.func == mock_runner.run
        assert partial_func.keywords == {
            "calib_data": "/tmp/test_dump/calib_data__.pth",
            "device": self.device,
            "model": self.mock_model1,
        }

    @patch("msmodelslim.core.quant_service.multimodal_sd_v1.quant_service.load_cached_data_for_models")
    @patch("msmodelslim.core.quant_service.multimodal_sd_v1.quant_service.to_device")
    @patch("msmodelslim.core.quant_service.multimodal_sd_v1.quant_service.LayerWiseRunner")
    def test_quant_process_legacy_uses_inference_config_when_model_config_absent(
        self, mock_runner_cls, mock_to_device, mock_load_cache
    ):
        """Legacy 路径应通过 resolve_inference_raw 读取 inference_config。"""
        inference_raw = {
            "size": "1280*720",
            "frame_num": 81,
            "sample_steps": 40,
            "convert_model_dtype": True,
            "task": "t2v-A14B",
        }
        self.mock_quant_spec.multimodal_sd_config = MultimodalSDConfig(
            dump_config=DumpConfig(enable_dump=False),
            inference_config=inference_raw,
        )
        self.model_adapter.init_model.return_value = {'': self.mock_model1}
        mock_to_device.return_value = {'': None}
        mock_runner_cls.return_value = Mock()

        with patch("builtins.input", return_value="y"):
            self.service.quant_process(self.quant_config, self.model_adapter, self.save_path, self.device)

        self.model_adapter.set_model_args.assert_called_once_with(inference_raw)

    def test_quant_process_legacy_raises_when_inference_config_missing(self):
        """Legacy 路径在 inference_config/model_config 均缺失时应明确报错。"""
        self.mock_quant_spec.multimodal_sd_config = MultimodalSDConfig(
            dump_config=DumpConfig(enable_dump=False),
        )

        with pytest.raises(SchemaValidateError) as excinfo:
            self.service.quant_process(self.quant_config, self.model_adapter, self.save_path, self.device)

        assert "Legacy pipeline requires inference_config or model_config" in str(excinfo.value)

    @patch("msmodelslim.core.quant_service.multimodal_sd_v1.quant_service.load_cached_data_for_models")
    @patch("msmodelslim.core.quant_service.multimodal_sd_v1.quant_service.get_logger")
    @patch("msmodelslim.core.quant_service.multimodal_sd_v1.quant_service.LayerWiseRunner")
    def test_quant_process_special_scenarios(self, mock_runner_cls, mock_get_logger, mock_load_cache):
        """合并特殊场景测试：彻底隔离Mock调用记录，避免干扰"""
        # 1. 初始化依赖
        mock_load_cache.return_value = {"": "raw_calib_data"}
        mock_logger = mock_get_logger.return_value
        mock_runner_cls.return_value = Mock()
        dump_config = self.mock_quant_spec.multimodal_sd_config.dump_config

        # -------------------------- 场景1：默认路径（dump_data_dir为空，save_path有效）--------------------------
        # 初始化场景1的save_config（独立实例，避免干扰）
        scene1_save_cfgs = [Mock(), Mock()]
        self.mock_quant_spec.save = scene1_save_cfgs  # 替换为场景1专属的save_config

        dump_config.dump_data_dir = ""
        self.service.quant_process(self.quant_config, self.model_adapter, self.save_path, self.device)

        # 验证场景1的save_config被调用（符合预期）
        for save_cfg in scene1_save_cfgs:
            save_cfg.set_save_directory.assert_called_once_with(self.save_path)

        # -------------------------- 场景2：非layer_wise runner --------------------------
        # 初始化场景2的save_config（新实例，隔离场景1）
        scene2_save_cfgs = [Mock(), Mock()]
        self.mock_quant_spec.save = scene2_save_cfgs

        self.mock_quant_spec.runner = "other_runner"
        self.service.quant_process(self.quant_config, self.model_adapter, self.save_path, self.device)

        # 验证场景2的save_config被调用（save_path有效，符合预期）
        for save_cfg in scene2_save_cfgs:
            save_cfg.set_save_directory.assert_called_once_with(self.save_path)
        # 验证警告日志
        mock_logger.warning.assert_has_calls(
            [call("runner for multimodal_sd_v1 is not layer_wise, will be converted to layer_wise.")]
        )

        # -------------------------- 场景3：无save_path --------------------------
        # 重置关键Mock：避免前场景干扰
        mock_load_cache.reset_mock()
        scene3_save_cfgs = [Mock(), Mock()]
        self.mock_quant_spec.save = scene3_save_cfgs
        dump_config.dump_data_dir = "./test"

        # 执行无save_path的流程
        self.service.quant_process(self.quant_config, self.model_adapter, save_path=None, device=self.device)
        for save_cfg in scene3_save_cfgs:
            save_cfg.set_save_directory.assert_not_called()

    @patch("builtins.input", return_value="y")
    @patch("msmodelslim.core.quant_service.multimodal_sd_v1.quant_service.load_cached_data_for_models")
    @patch("msmodelslim.core.quant_service.multimodal_sd_v1.quant_service.to_device")
    @patch("msmodelslim.core.quant_service.multimodal_sd_v1.quant_service.LayerWiseRunner")
    def test_quant_process_enable_dump_false(self, mock_runner_cls, mock_to_device, mock_load_cache, mock_input):
        """enable_dump=False 时不调用 load_cached_data_for_models，calib_data 为各 expert 对应 None；需交互确认后继续"""
        self.model_adapter.init_model.return_value = {"": self.mock_model1}
        self.model_adapter.model_args.task_config = ""
        self.mock_quant_spec.multimodal_sd_config.dump_config.enable_dump = False
        mock_to_device.side_effect = lambda x, _: x  # 透传，便于断言 calib_data 内容
        mock_runner_cls.return_value = Mock()

        captured_config = []

        def capture_config(config):
            captured_config.append(config)

        with patch.object(self.service, "quantize_multi_expert_models", side_effect=capture_config):
            self.service.quant_process(self.quant_config, self.model_adapter, self.save_path, self.device)

        mock_load_cache.assert_not_called()
        mock_input.assert_called_once()
        assert len(captured_config) == 1
        assert captured_config[0].calib_data == {"": None}
        mock_to_device.assert_called_once()
        call_args = mock_to_device.call_args[0]
        assert call_args[0] == {"": None}

    @patch("builtins.input", return_value="n")
    @patch("msmodelslim.core.quant_service.multimodal_sd_v1.quant_service.load_cached_data_for_models")
    @patch("msmodelslim.core.quant_service.multimodal_sd_v1.quant_service.LayerWiseRunner")
    def test_quant_process_enable_dump_false_user_declines(self, mock_runner_cls, mock_load_cache, mock_input):
        """enable_dump=False 时用户输入非 y，应抛出 UnsupportedError 并退出"""
        self.model_adapter.init_model.return_value = {"": self.mock_model1}
        self.model_adapter.model_args.task_config = ""
        self.mock_quant_spec.multimodal_sd_config.dump_config.enable_dump = False
        mock_runner_cls.return_value = Mock()

        with pytest.raises(UnsupportedError) as excinfo:
            self.service.quant_process(self.quant_config, self.model_adapter, self.save_path, self.device)

        assert "enable_dump" in str(excinfo.value) or "dump" in str(excinfo.value).lower()
        mock_load_cache.assert_not_called()
