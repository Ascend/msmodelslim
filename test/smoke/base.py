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
# pylint: disable=no-name-in-module,too-many-ancestors

import os
from functools import lru_cache
from pathlib import Path
from typing import List
from unittest.mock import patch, MagicMock

import torch
from resources.fake_llama.fake_llama import get_fake_llama_model_and_tokenizer
from torch import nn
from transformers import PretrainedConfig, PreTrainedTokenizerBase

from msmodelslim.core.const import DeviceType
from msmodelslim.model.qwen3.model_adapter import Qwen3ModelAdapter
from msmodelslim.processor.kv_smooth import KVSmoothFusedUnit, KVSmoothFusedType


@lru_cache(maxsize=1)
def is_npu_available():
    try:
        import torch_npu  # noqa: F401

        return torch.npu.is_available()
    except ImportError:
        return False


@lru_cache(maxsize=1)
def is_cuda_available():
    try:
        return torch.cuda.is_available()
    except ImportError:
        return False


class FakeLlamaModelAdapter(Qwen3ModelAdapter):
    def __init__(self, model_type: str, model_path: Path, trust_remote_code: bool = False):
        model, tokenizer = get_fake_llama_model_and_tokenizer()
        self.loaded_config = model.config
        self.loaded_model = model
        self.loaded_tokenizer = tokenizer
        Qwen3ModelAdapter.__init__(self, model_type, model_path, trust_remote_code)

    def _load_config(self, trust_remote_code=False) -> PretrainedConfig:
        return self.loaded_config

    def _load_model(self, device: DeviceType) -> nn.Module:
        return self.loaded_model

    def _load_tokenizer(self, trust_remote_code=False) -> PreTrainedTokenizerBase:
        return self.loaded_tokenizer

    def get_kvcache_smooth_fused_subgraph(self) -> List[KVSmoothFusedUnit]:
        return [
            KVSmoothFusedUnit(
                attention_name=f"model.layers.{i}.self_attn",
                layer_idx=i,
                fused_from_query_states_name="q_proj",
                fused_from_key_states_name="k_proj",
                fused_type=KVSmoothFusedType.StateViaRopeToLinear,
            )
            for i in range(self.config.num_hidden_layers)
        ]


def invoke_test(config_name: str, model_save_path: str, device: str = 'cpu', offload_device: str = 'cpu'):
    """使用真正的CLI parser来模拟命令行参数并返回model_adapter"""
    import sys
    from msmodelslim.cli.__main__ import main as cli_main

    # 保存原始的sys.argv
    original_argv = sys.argv.copy()

    # 用于存储model_adapter的变量
    captured_model_adapter = None

    fake_ep = MagicMock()
    fake_ep.name = "fake_llama"
    fake_ep.load.return_value = FakeLlamaModelAdapter

    try:
        # 构建命令行参数
        config_path = os.path.join(os.path.dirname(__file__), "configs", config_name)
        sys.argv = [
            'msmodelslim',
            'quant',
            '--model_type',
            'fake_llama',
            '--model_path',
            './',
            '--save_path',
            model_save_path,
            '--device',
            device,
            '--config_path',
            config_path,
            '--trust_remote_code',
            'False',
        ]

        # 使用patch来模拟copy_files调用并拦截model_adapter
        with (
            patch("msmodelslim.model.plugin_factory.plugin_model_factory.entry_points") as mock_entry_points,
            patch("msmodelslim.core.quant_service.modelslim_v1.save.ascendv1.copy_files") as mock_copy_files,  # noqa: F841
            patch("msmodelslim.model.plugin_factory.base_loader.DependencyChecker.check_plugin") as mock_check_plugin,
        ):
            mock_entry_points.return_value.select.return_value = [fake_ep]
            mock_check_plugin.return_value = None

            # 获取原始的quantize方法
            from msmodelslim.core.quant_service.proxy import QuantServiceProxy

            original_quantize = QuantServiceProxy.quantize

            # 创建包装函数来捕获model_adapter但不影响原始流程
            def capture_model_adapter(
                self, quant_config, model_adapter, save_path=None, device=None, device_indices=None
            ):
                nonlocal captured_model_adapter
                captured_model_adapter = model_adapter
                # 调用原始方法，保持原始流程不变
                return original_quantize(self, quant_config, model_adapter, save_path, device, device_indices)

            # 临时替换方法
            QuantServiceProxy.quantize = capture_model_adapter

            # Mock LayerWiseRunner 构造时的 offload_device 参数
            from msmodelslim.core.runner.layer_wise_runner import LayerWiseRunner

            # 保存原始的 __init__ 方法
            original_init = LayerWiseRunner.__init__

            # 创建新的 __init__ 方法来设置 offload_device 参数
            def mock_init(self, adapter, offload_device=offload_device):
                original_init(self, adapter, offload_device)

            # 替换 __init__ 方法
            LayerWiseRunner.__init__ = mock_init

            try:
                # 直接调用CLI main函数，它会解析sys.argv
                cli_main()
            finally:
                # 恢复原始方法
                QuantServiceProxy.quantize = original_quantize
                # 恢复原始的 __init__ 方法
                LayerWiseRunner.__init__ = original_init
    finally:
        # 恢复原始的sys.argv
        sys.argv = original_argv

    return captured_model_adapter


def invoke_analysis_test(metrics: str = "kurtosis", patterns: list = None, topk: int = 15):
    """
    使用真正的CLI parser来模拟分析模块命令行参数并返回分析结果

    Args:
        metrics: 分析算法
        patterns: 层模式列表
        topk: 输出topk敏感层

    Returns:
        分析结果
    """
    import sys
    from msmodelslim.cli.__main__ import main as cli_main

    # 保存原始的sys.argv
    original_argv = sys.argv.copy()

    # 用于存储分析结果的变量
    captured_result = None

    try:
        # 构建命令行参数
        sys.argv = [
            'msmodelslim',
            'analyze',
            '--model_type',
            'fake_llama',
            '--model_path',
            './',
            '--device',
            'cpu',
            '--metrics',
            metrics,
            '--calib_dataset',
            'boolq.jsonl',
            '--topk',
            str(topk),
            '--trust_remote_code',
            'False',
        ]

        # 添加patterns参数
        if patterns:
            sys.argv.extend(['--pattern'])
            sys.argv.extend(patterns)

        # Mock整个分析流程
        with patch('msmodelslim.cli.analysis.__main__.LayerAnalysisApplication') as analysis_app:
            mock_app_instance = MagicMock()
            mock_app_instance.analyze.return_value = "mock_analysis_result"
            analysis_app.return_value = mock_app_instance

            # Mock数据集加载器
            with patch('msmodelslim.cli.analysis.__main__.FileDatasetLoader'):
                # Mock分析服务
                with patch('msmodelslim.cli.analysis.__main__.PipelineAnalysisService'):
                    # 使用patch来捕获分析结果
                    from msmodelslim.cli.analysis.__main__ import main as analysis_main

                    original_analysis_main = analysis_main

                    def capture_result(args):
                        nonlocal captured_result
                        try:
                            captured_result = original_analysis_main(args)
                            return captured_result
                        except Exception as e:
                            captured_result = e
                            return None

                    # 临时替换方法
                    import msmodelslim.cli.analysis.__main__ as analysis_module

                    analysis_module.main = capture_result

                    try:
                        cli_main()
                    finally:
                        analysis_module.main = original_analysis_main

    finally:
        sys.argv = original_argv

    return captured_result
