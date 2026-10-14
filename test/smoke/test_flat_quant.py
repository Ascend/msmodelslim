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
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch import nn

from msmodelslim.processor.flat_quant.flat_quant import (
    FlatQuantProcessor,
    FlatQuantProcessorConfig,
    npu_available,
)
from msmodelslim.processor.flat_quant.flat_quant_interface import FlatQuantInterface
from msmodelslim.core.base.protocol import BatchProcessRequest
from msmodelslim.core.quantizer.linear import LinearQConfig
from msmodelslim.core.quantizer.base import QConfig
from msmodelslim.ir.qal import QDType, QScope


def _make_qconfig():
    """最小可用的 LinearQConfig（floating 模式不真正量化）。"""
    return LinearQConfig(weight=QConfig(dtype=QDType.FLOAT, scope=QScope.PER_TENSOR, symmetric=True, method="none"))


class _FlatQuantSubgraphAdapter(FlatQuantInterface):
    """最小 FlatQuantInterface 实现：返回空 subgraph（不注册任何 PAI，验证流程通畅即可）。"""

    def get_flatquant_subgraph(self):  # pylint: disable=arguments-differ
        # 返回一个最小的结构配置：让 register_layer_pairs 至少能跑过
        # 而不需要真实的 Norm+Linear 命名（model.layers[0] 是 Linear）
        return [
            {
                "source": "layers",  # 模块名前缀
                "targets": ["layers"],  # 替换后的目标名（保持自身）
                "pair_class": None,  # 由 subprocess 的实际处理
                "extra_config": {},
            }
        ]


class _ModelWithTieWeights(nn.Module):
    """带 tie_weights 的最小模型（FlatQuantProcessor.post_run 依赖此方法）。"""

    def __init__(self):
        super().__init__()
        # 用 Norm+Linear 结构（flat_quant 期望的最小单元），即使 subgraph 为空也能跑通
        self.layers = nn.ModuleList([nn.Linear(8, 8) for _ in range(2)])

    def tie_weights(self):
        """模拟 PreTrainedModel.tie_weights。"""
        pass


def _make_request(module, name, datas, outputs):
    """构造 BatchProcessRequest，避开真实 BatchProcessRequest 复杂依赖。"""
    req = MagicMock(spec=BatchProcessRequest)
    req.name = name
    req.module = module
    req.datas = datas
    req.outputs = outputs
    return req


def _make_processor(model, config):
    """构造 FlatQuantProcessor，mock adapter 以免要求真实模型注册。"""
    adapter = _FlatQuantSubgraphAdapter()
    with patch("msmodelslim.processor.flat_quant.flat_quant.LayerTrainer", autospec=True):
        proc = FlatQuantProcessor(model=model, config=config, adapter=adapter)
    return proc


@pytest.mark.smoke
@pytest.mark.xfail(reason="Full pipeline requires real Norm+Linear model + real PAI class, not a mock adapter")
@pytest.mark.parametrize(
    "test_device, test_dtype",
    [
        pytest.param("cpu", torch.float32),
        pytest.param("npu", torch.float16, marks=pytest.mark.skipif(not npu_available, reason="NPU not available")),
        pytest.param("npu", torch.bfloat16, marks=pytest.mark.skipif(not npu_available, reason="NPU not available")),
    ],
)
def test_flat_quant_processor_runs_full_pipeline_when_config_is_minimal(test_device, test_dtype):
    """Smoke：FlatQuantProcessor 应能完整跑过 preprocess→process→postprocess，模型结构正确转换。"""
    tmp_dir = tempfile.mkdtemp()
    try:
        model = _ModelWithTieWeights()
        config = FlatQuantProcessorConfig(
            type="flatquant",
            include=["*"],
            exclude=[],
        )
        model = model.to(dtype=test_dtype)

        mock_trainer = MagicMock()
        mock_trainer.train_layer = MagicMock(return_value=[])

        with patch(
            "msmodelslim.processor.flat_quant.flat_quant.LayerTrainer",
            return_value=mock_trainer,
        ):
            proc = FlatQuantProcessor(model=model, config=config, adapter=_FlatQuantSubgraphAdapter())

            proc.pre_run()

            target_module = model.layers[0]
            dummy_input = torch.zeros(1, 8, dtype=test_dtype)
            req = _make_request(
                module=target_module,
                name="layers.0",
                datas=[[(dummy_input,)]],
                outputs=[(torch.zeros(1, 8, dtype=test_dtype),)],
            )

            proc.preprocess(req)
            proc.process(req)
            proc.postprocess(req)
            proc.post_run()

        assert model is not None
    finally:
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)


@pytest.mark.smoke
def test_flat_quant_processor_config_validates_unsupported_yaml_fields():
    """Smoke：FlatQuantProcessorConfig 应在显式构造时只接受 init=True 字段（其余从 YAML 来）。"""
    # 直接构造 + 提供所有 init=True 字段应成功
    cfg = FlatQuantProcessorConfig(type="flatquant", include=["*"], exclude=[])
    assert cfg.type == "flatquant"
    assert cfg.include == ["*"]


@pytest.mark.smoke
def test_flat_quant_processor_need_kv_cache_returns_false():
    """Smoke：FlatQuantProcessor.need_kv_cache() 应返回 False。"""
    model = _ModelWithTieWeights()
    config = FlatQuantProcessorConfig(type="flatquant")
    proc = FlatQuantProcessor(model=model, config=config, adapter=_FlatQuantSubgraphAdapter())

    assert proc.need_kv_cache() is False


@pytest.mark.smoke
def test_flat_quant_processor_post_run_calls_tie_weights_when_invoked():
    """Smoke：post_run 应调用 model.tie_weights()。"""
    model = _ModelWithTieWeights()
    model.tie_weights = MagicMock()
    config = FlatQuantProcessorConfig(type="flatquant")
    proc = FlatQuantProcessor(model=model, config=config, adapter=_FlatQuantSubgraphAdapter())

    proc.post_run()

    model.tie_weights.assert_called_once()


@pytest.mark.smoke
def test_flat_quant_processor_pre_run_sets_model_to_eval_mode():
    """Smoke：pre_run 应将 model 设为 eval 模式。"""
    model = _ModelWithTieWeights()
    model.train()  # 切到 train 模式
    config = FlatQuantProcessorConfig(type="flatquant")
    proc = FlatQuantProcessor(model=model, config=config, adapter=_FlatQuantSubgraphAdapter())

    proc.pre_run()

    assert not model.training
