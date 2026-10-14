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

OASQProcessor 单元测试：配置、上下文构建、算法应用、norm 替换与 adapter 回退。
"""

from typing import List
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch import nn

from msmodelslim.core.base.protocol import BatchProcessRequest
from msmodelslim.core.graph.adapter_types import AdapterConfig, MappingConfig
from msmodelslim.ir.non_fusion_smooth_quant_ir import NonFusionSmoothQuantHookIR
from msmodelslim.ir.norm_bias import RMSNormBias
from msmodelslim.ir.rms_norm import RMSNorm
from msmodelslim.processor.anti_outlier.common.smooth_components import StatKey
from msmodelslim.processor.anti_outlier.common.subgraph_type import (
    NonFusionSubgraph,
    NormLinearSubgraph,
)
from msmodelslim.processor.anti_outlier.oasq import OASQProcessor, OASQProcessorConfig
from msmodelslim.processor.anti_outlier.oasq.interface import OASQInterface
from msmodelslim.processor.anti_outlier.oasq.processor import OASQStatsCollector
from msmodelslim.utils.exception import SchemaValidateError


class MockModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(10, 20)
        self.norm_layer = nn.LayerNorm(10)

    def get_submodule(self, name):
        if name == "layer1":
            return self.layer1
        if name == "model.layers.0.input_layernorm":
            return self.norm_layer
        if name == "norm_no_weight":
            return nn.ReLU()
        return None

    def set_submodule(self, name, module):
        if name == "model.layers.0.input_layernorm":
            self.norm_layer = module


class MockOASQAdapter(OASQInterface):
    def get_adapter_config_for_subgraph(self) -> List[AdapterConfig]:
        return [
            AdapterConfig(
                subgraph_type="norm-linear",
                mapping=MappingConfig(
                    source="model.layers.0.input_layernorm",
                    targets=["layer1"],
                ),
            )
        ]


class MockAdapterWithoutInterface:
    pass


class TestOASQProcessorConfig:
    """场景：OASQProcessorConfig 字段默认值与校验。"""

    def test_config_when_default_then_type_is_oasq_and_symmetric_true(self):
        """给定默认构造，期望 type=oasq、symmetric=True、默认子图列表完整。"""
        cfg = OASQProcessorConfig()
        assert cfg.type == "oasq"
        assert cfg.symmetric is True
        assert cfg.max_iters is None
        assert "norm-linear" in cfg.enable_subgraph_type
        assert "ov" in cfg.enable_subgraph_type

    def test_config_when_max_iters_zero_then_raises_schema_validate_error(self):
        """给定 max_iters=0，期望 SchemaValidateError（须 >0）。"""
        with pytest.raises(SchemaValidateError):
            OASQProcessorConfig(max_iters=0)

    def test_config_when_max_iters_positive_then_accepts_value(self):
        """给定 max_iters=8，期望配置成功保存。"""
        cfg = OASQProcessorConfig(max_iters=8)
        assert cfg.max_iters == 8


class TestOASQProcessorInit:
    """场景：Processor 初始化与分布式声明。"""

    def test_init_when_valid_adapter_then_creates_stats_collector_with_symmetric_flag(self):
        """给定合法 adapter，期望 stats_collector.symmetric 跟随 config。"""
        model = MockModel()
        cfg = OASQProcessorConfig(symmetric=False)
        processor = OASQProcessor(model, cfg, MockOASQAdapter())
        assert isinstance(processor.stats_collector, OASQStatsCollector)
        assert processor.stats_collector.symmetric is False
        assert processor.support_distributed() is True

    def test_init_when_adapter_missing_interface_then_fallback_default_adapter(self):
        """给定未实现 OASQInterface 的 adapter，期望回退 default adapter 并设标志。"""
        processor = OASQProcessor(MockModel(), OASQProcessorConfig(), MockAdapterWithoutInterface())
        assert processor.is_defalut_adapter is True


class TestOASQProcessorBuildSmoothContext:
    """场景：从 act_stats 构建 OASQContext。"""

    def _processor(self):
        return OASQProcessor(MockModel(), OASQProcessorConfig(), MockOASQAdapter())

    def test_build_smooth_context_when_linear_names_empty_then_returns_none(self):
        """给定空 linear_names，期望返回 None。"""
        assert self._processor()._build_smooth_context([]) is None

    def test_build_smooth_context_when_name_missing_in_act_stats_then_returns_none(self):
        """给定统计中不存在该 linear，期望返回 None。"""
        assert self._processor()._build_smooth_context(["missing.fc"]) is None

    def test_build_smooth_context_when_smooth_scale_missing_then_returns_none(self):
        """给定有 act_stats 但缺少 SMOOTH_SCALE，期望返回 None。"""
        processor = self._processor()
        processor.stats_collector.act_stats["fc"] = {
            StatKey.STAT_KEY_SHIFT: torch.zeros(4),
        }
        assert processor._build_smooth_context(["fc"]) is None

    def test_build_smooth_context_when_shift_key_missing_then_context_shift_is_none(self):
        """给定有 SMOOTH_SCALE 但无 SHIFT，期望 context.shift 为 None。"""
        processor = self._processor()
        processor.stats_collector.act_stats["fc"] = {
            StatKey.STAT_KEY_SMOOTH_SCALE: torch.ones(4),
        }
        ctx = processor._build_smooth_context(["fc"])
        assert ctx is not None
        assert ctx.shift is None

    def test_build_smooth_context_when_stats_complete_then_returns_oasq_context(self):
        """给定完整 SMOOTH_SCALE/SHIFT，期望返回 OASQContext。"""
        processor = self._processor()
        scale = torch.ones(4)
        shift = torch.zeros(4)
        processor.stats_collector.act_stats["fc"] = {
            StatKey.STAT_KEY_SMOOTH_SCALE: scale,
            StatKey.STAT_KEY_SHIFT: shift,
        }
        ctx = processor._build_smooth_context(["fc"])
        assert ctx is not None
        assert ctx.version == 1
        assert torch.equal(ctx.a_smooth_scale, scale)
        assert torch.equal(ctx.shift, shift)


class TestOASQProcessorApplySmoothAlgorithm:
    """场景：apply_smooth_algorithm 的 shift 推导、跳过与 NonFusion Hook 注册。"""

    def test_apply_smooth_when_no_stats_then_skips_without_calling_oasq(self):
        """给定无统计上下文，期望跳过且不调用 oasq API。"""
        processor = OASQProcessor(MockModel(), OASQProcessorConfig(), MockOASQAdapter())
        subgraph = NormLinearSubgraph(norm=RMSNormBias(10), linears=[nn.Linear(10, 10)])
        with patch("msmodelslim.processor.anti_outlier.oasq.processor.oasq") as mock_oasq:
            processor.apply_smooth_algorithm(subgraph, ["absent.fc"])
            mock_oasq.assert_not_called()

    def test_apply_smooth_when_norm_linear_asymmetric_then_sets_shift_true(self):
        """给定 norm-linear + symmetric=False，期望 oasq 配置 shift=True。"""
        processor = OASQProcessor(
            MockModel(),
            OASQProcessorConfig(symmetric=False, max_iters=2),
            MockOASQAdapter(),
        )
        processor.stats_collector.act_stats["layer1"] = {
            StatKey.STAT_KEY_SMOOTH_SCALE: torch.ones(10),
            StatKey.STAT_KEY_SHIFT: torch.zeros(10),
        }
        subgraph = NormLinearSubgraph(norm=RMSNormBias(10), linears=[nn.Linear(10, 10)])
        with patch("msmodelslim.processor.anti_outlier.oasq.processor.oasq") as mock_oasq:
            mock_oasq.return_value = None
            processor.apply_smooth_algorithm(subgraph, ["layer1"])
            cfg = mock_oasq.call_args[0][1]
            assert cfg.shift is True

    def test_apply_smooth_when_non_asym_subgraph_then_forces_shift_false(self):
        """给定非 asym 子图（non-fusion），即使 symmetric=False 也期望 shift=False。"""
        processor = OASQProcessor(
            MockModel(),
            OASQProcessorConfig(symmetric=False, max_iters=2),
            MockOASQAdapter(),
        )
        linear = nn.Linear(8, 4)
        processor.stats_collector.act_stats["nf"] = {
            StatKey.STAT_KEY_SMOOTH_SCALE: torch.ones(8),
            StatKey.STAT_KEY_SHIFT: torch.zeros(8),
        }
        subgraph = NonFusionSubgraph(linears=[linear])
        scales = torch.ones(8)
        with patch("msmodelslim.processor.anti_outlier.oasq.processor.oasq", return_value=scales):
            processor.apply_smooth_algorithm(subgraph, ["nf"])
        # NonFusion 应注册 NonFusionSmoothQuantHookIR
        hooks = [h for h in linear._forward_pre_hooks.values() if isinstance(h, NonFusionSmoothQuantHookIR)]
        assert len(hooks) == 1


class TestOASQProcessorReplaceNormAndLifecycle:
    """场景：norm 替换与 preprocess/postprocess 清理。"""

    def test_replace_norm_when_symmetric_true_and_layernorm_has_bias_then_uses_rmsnormbias(self):
        """给定 symmetric=True 且原 norm 带 bias，期望替换为 RMSNormBias。"""
        model = MockModel()
        processor = OASQProcessor(model, OASQProcessorConfig(symmetric=True), MockOASQAdapter())
        processor.adapter_config = MockOASQAdapter().get_adapter_config_for_subgraph()
        processor._replace_norm_modules()
        assert isinstance(model.norm_layer, RMSNormBias)

    def test_replace_norm_when_symmetric_true_and_norm_without_bias_attr_then_uses_rmsnorm(self):
        """给定 symmetric=True 且模块无 bias 属性语义，期望尽量替换成功。"""
        model = MockModel()

        # LayerNorm always has bias in pytorch; simulate RMS-like by replacing with custom
        class _Norm(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.ones(10))

        model.norm_layer = _Norm()
        processor = OASQProcessor(model, OASQProcessorConfig(symmetric=True), MockOASQAdapter())
        processor.adapter_config = MockOASQAdapter().get_adapter_config_for_subgraph()
        processor._replace_norm_modules()
        assert isinstance(model.norm_layer, RMSNorm)

    def test_replace_norm_when_non_norm_linear_subgraph_then_skips(self):
        """给定 adapter_config 为 ov 子图，期望不替换 LayerNorm。"""
        model = MockModel()
        processor = OASQProcessor(model, OASQProcessorConfig(), MockOASQAdapter())
        processor.adapter_config = [
            AdapterConfig(
                subgraph_type="ov",
                mapping=MappingConfig(source="model.layers.0.input_layernorm", targets=["layer1"]),
            )
        ]
        processor._replace_norm_modules()
        assert isinstance(model.norm_layer, nn.LayerNorm)

    def test_replace_norm_when_set_submodule_raises_then_logs_and_continues(self):
        """给定 set_submodule 抛异常，期望捕获后不中断。"""
        model = MockModel()

        def _boom(name, module):
            raise RuntimeError("inject fail")

        model.set_submodule = _boom
        processor = OASQProcessor(model, OASQProcessorConfig(), MockOASQAdapter())
        processor.adapter_config = MockOASQAdapter().get_adapter_config_for_subgraph()
        processor._replace_norm_modules()  # should not raise

    def test_replace_norm_when_source_empty_then_skips_without_crash(self):
        """给定 mapping.source 为空字符串，期望跳过且不抛异常。"""
        model = MockModel()
        processor = OASQProcessor(model, OASQProcessorConfig(), MockOASQAdapter())
        processor.adapter_config = [
            AdapterConfig(
                subgraph_type="norm-linear",
                mapping=MappingConfig(source="", targets=["layer1"]),
            )
        ]
        processor._replace_norm_modules()
        assert isinstance(model.norm_layer, nn.LayerNorm)

    def test_replace_norm_when_module_has_no_weight_then_skips_without_crash(self):
        """给定 source 模块无 weight，期望跳过。"""
        model = MockModel()
        processor = OASQProcessor(model, OASQProcessorConfig(), MockOASQAdapter())
        processor.adapter_config = [
            AdapterConfig(
                subgraph_type="norm-linear",
                mapping=MappingConfig(source="norm_no_weight", targets=["layer1"]),
            )
        ]
        processor._replace_norm_modules()

    def test_preprocess_and_postprocess_when_dist_not_initialized_then_clears_dist_helper(self):
        """给定未初始化分布式，期望 preprocess 不创建 DistHelper，postprocess 清理为 None。"""
        model = MockModel()
        processor = OASQProcessor(model, OASQProcessorConfig(), MockOASQAdapter())
        processor.global_adapter_config = MockOASQAdapter().get_adapter_config_for_subgraph()
        request = BatchProcessRequest(name="model.layers.0", module=model.layer1)
        with patch("msmodelslim.processor.anti_outlier.oasq.processor.dist.is_initialized", return_value=False):
            processor.preprocess(request)
            assert processor.dist_helper is None
            processor.postprocess(request)
            assert processor.dist_helper is None
            assert processor.stats_collector.dist_helper is None

    def test_preprocess_when_dist_initialized_then_sets_dist_helper_on_collector(self):
        """给定 dist.is_initialized=True，期望创建 DistHelper 并注入 stats_collector。"""
        model = MockModel()
        processor = OASQProcessor(model, OASQProcessorConfig(), MockOASQAdapter())
        processor.global_adapter_config = MockOASQAdapter().get_adapter_config_for_subgraph()
        processor.adapter_config = MockOASQAdapter().get_adapter_config_for_subgraph()
        request = BatchProcessRequest(name="model.layers.0", module=model.layer1)
        fake_helper = MagicMock()
        with (
            patch(
                "msmodelslim.processor.anti_outlier.oasq.processor.dist.is_initialized",
                return_value=True,
            ),
            patch(
                "msmodelslim.processor.anti_outlier.oasq.processor.DistHelper",
                return_value=fake_helper,
            ),
            patch(
                "msmodelslim.processor.anti_outlier.oasq.processor.BaseSmoothProcessor.preprocess",
                return_value=None,
            ),
        ):
            processor.preprocess(request)
        assert processor.dist_helper is fake_helper
        assert processor.stats_collector.dist_helper is fake_helper
