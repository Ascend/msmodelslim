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

msmodelslim.core.quant_service.modelslim_convert.impl.group_runner 模块的单元测试
"""

from unittest.mock import MagicMock

import torch
from torch import nn

from msmodelslim.core.convert.config import ConvertConfig, ParallelConfig
from msmodelslim.core.convert.edges import TransformEdge
from msmodelslim.core.convert.protocol import ConvertContext
from msmodelslim.core.convert.router import IRRouter
from msmodelslim.core.convert.tasks import IRTask, RoutedTask
from msmodelslim.core.convert.types import IRKind, LossLevel, SourceIR, TensorRef
from msmodelslim.core.quant_service.modelslim_convert.impl.group_runner import (
    DependencyGroupRunner,
    estimate_task_bytes,
    prepare_result,
)
from msmodelslim.core.convert.tasks import IRResult, PortableTensor
from msmodelslim.core.quant_service.modelslim_convert.virtual_module import ModelFreeLinear


def _make_routed_linear_task(module_path: str = "layers.0.q_proj") -> tuple[nn.Module, RoutedTask]:
    root = nn.Module()
    linear = ModelFreeLinear(
        full_name=module_path,
        tensor_bindings={
            "weight": TensorRef("weight", "w", "s0", "bf16", (2, 2)),
        },
        source_ir=SourceIR(kind=IRKind.FLOAT),
    )
    linear.register_parameter("weight", nn.Parameter(torch.ones(2, 2)))
    linear.lazy_initialized = True
    layer0 = nn.Module()
    layer0.add_module("q_proj", linear)
    layers = nn.Module()
    layers.add_module("0", layer0)
    root.add_module("layers", layers)

    edge = TransformEdge(
        src_ir=IRKind.FLOAT,
        dst_ir=IRKind.W8A8_MXFP8,
        processor_name="MockProcessor",
        loss_level=LossLevel.LOSSY,
    )
    task = IRTask(
        module_path=module_path,
        source_ir=SourceIR(kind=IRKind.FLOAT),
        target_ir=IRKind.W8A8_MXFP8,
        tensor_bindings=linear.tensor_bindings,
        inverse_weight_map={},
    )
    routed = RoutedTask(task=task, route=[edge], route_ir_names=[IRKind.FLOAT, IRKind.W8A8_MXFP8])
    return root, routed


class TestPrepareResult:
    """测试 prepare_result 函数"""

    def test_prepare_result_return_portable_state_dict_when_return_mode_state_dict(self):
        mod = nn.Linear(2, 2, bias=False)
        mod.weight = nn.Parameter(torch.ones(2, 2))
        result = IRResult(module_path="x", final_ir=IRKind.FLOAT, module=mod)
        prepared = prepare_result(result, return_mode="state_dict")
        assert prepared.module is None  # 校验 module 被剥离
        assert isinstance(prepared.state_dict["weight"], PortableTensor)  # 校验 tensor 打包

    def test_prepare_result_return_unchanged_when_return_mode_module(self):
        mod = nn.Linear(2, 2, bias=False)
        result = IRResult(module_path="x", final_ir=IRKind.FLOAT, module=mod)
        assert prepare_result(result, return_mode="module") is result  # 校验 module 模式不转换


class TestEstimateTaskBytes:
    """测试 estimate_task_bytes 函数"""

    def test_estimate_task_bytes_return_byte_count_when_ref_has_shape(self):
        task = IRTask(
            module_path="x",
            source_ir=SourceIR(kind=IRKind.FLOAT),
            target_ir=IRKind.FLOAT,
            tensor_bindings={
                "weight": TensorRef("weight", "w", "s0", "bf16", (10, 20)),
            },
            inverse_weight_map={},
        )
        assert estimate_task_bytes(task, None) == 400  # 校验 float16 粗算 10*20*2


class TestDependencyGroupRunner:
    """测试 DependencyGroupRunner 类"""

    def test_run_one_apply_processor_chain_when_route_given(self):
        root, routed = _make_routed_linear_task()
        router = IRRouter()
        mock_proc = MagicMock()
        deployed = nn.Linear(2, 2, bias=False)
        mock_proc.transform.return_value = deployed
        mock_proc.name = "MockProcessor"
        router._processors["MockProcessor"] = mock_proc

        config = ConvertConfig(model_path="/m", save_path="/o")
        context = ConvertContext(config=config)
        context.virtual_tree = root
        context.reader = MagicMock()

        result, timing = DependencyGroupRunner(router)._run_one(context, routed)
        assert result.module_path == "layers.0.q_proj"  # 校验任务路径
        assert timing.lazy_init_s == 0.0  # 校验已 lazy_init 不再加载
        assert result.final_ir == IRKind.W8A8_MXFP8  # 校验目标 IR
        assert result.loss_level == "lossy"  # 校验损失等级

    def test_run_group_yield_results_when_single_task(self):
        root, routed = _make_routed_linear_task()
        router = IRRouter()
        mock_proc = MagicMock()
        mock_proc.transform.return_value = nn.Linear(2, 2, bias=False)
        router._processors["MockProcessor"] = mock_proc

        config = ConvertConfig(
            model_path="/m",
            save_path="/o",
            parallel=ParallelConfig(max_workers=1, task_granularity="ir_task"),
        )
        context = ConvertContext(config=config)
        context.virtual_tree = root
        context.reader = MagicMock()

        results = DependencyGroupRunner(router).run_group(
            context, [routed], max_workers=1, budget=None, catalog=None, return_mode="module"
        )
        assert len(results) == 1  # 校验单任务组返回一条结果
        assert results[0].module_path == "layers.0.q_proj"

    def test_run_group_return_all_results_when_multi_worker(self):
        router = IRRouter()
        mock_proc = MagicMock()
        mock_proc.transform.return_value = nn.Linear(2, 2, bias=False)
        router._processors["MockProcessor"] = mock_proc

        def _task(path: str) -> RoutedTask:
            edge = TransformEdge(
                src_ir=IRKind.FLOAT,
                dst_ir=IRKind.W8A8_MXFP8,
                processor_name="MockProcessor",
                loss_level=LossLevel.LOSSY,
            )
            task = IRTask(
                module_path=path,
                source_ir=SourceIR(kind=IRKind.FLOAT),
                target_ir=IRKind.W8A8_MXFP8,
                tensor_bindings={
                    "weight": TensorRef("weight", "w", "s0", "bf16", (2, 2)),
                },
                inverse_weight_map={},
            )
            return RoutedTask(task=task, route=[edge], route_ir_names=[IRKind.FLOAT, IRKind.W8A8_MXFP8])

        config = ConvertConfig(model_path="/m", save_path="/o")
        context = ConvertContext(config=config)
        context.virtual_tree = None
        reader = MagicMock()
        reader.load_tensors.return_value = {"w": torch.ones(2, 2)}
        context.reader = reader

        results = DependencyGroupRunner(router).run_group(
            context,
            [_task("layers.0.q_proj"), _task("layers.1.q_proj")],
            max_workers=2,
            budget=None,
            catalog=None,
            return_mode="module",
        )
        assert len(results) == 2  # 校验多 worker 完成全部任务
