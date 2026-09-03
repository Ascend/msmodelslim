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

msmodelslim.core.quant_service.modelslim_convert.impl.executor 模块的单元测试
"""

from unittest.mock import MagicMock, patch

import torch
from torch import nn

from msmodelslim.core.convert.catalog import DependencyMap
from msmodelslim.core.convert.config import ConvertConfig, ParallelConfig
from msmodelslim.core.convert.edges import TransformEdge
from msmodelslim.core.convert.protocol import ConvertContext
from msmodelslim.core.convert.router import IRRouter
from msmodelslim.core.convert.tasks import IRResult, IRTask, RoutedTask
from msmodelslim.core.convert.types import IRKind, LossLevel, SourceIR, TensorRef
from msmodelslim.core.quant_service.modelslim_convert.impl.executor import (
    ConvertExecutor,
    _schedule_groups,
)
from msmodelslim.core.quant_service.modelslim_convert.virtual_module import (
    ModelFreeLinear,
)


class TestScheduleGroups:
    """测试 ConvertExecutor 内部 _schedule_groups 函数"""

    def test_schedule_groups_group_by_fused_from_when_dependency_present(self):
        task1 = IRTask(
            module_path="a",
            source_ir=SourceIR(kind=IRKind.FLOAT),
            target_ir=IRKind.FLOAT,
            tensor_bindings={
                "weight": TensorRef("weight", "k1", "s0", "bf16", (), meta={"fused_from": "fused"}),
            },
            inverse_weight_map={},
        )
        task2 = IRTask(
            module_path="b",
            source_ir=SourceIR(kind=IRKind.FLOAT),
            target_ir=IRKind.FLOAT,
            tensor_bindings={
                "weight": TensorRef("weight", "k2", "s0", "bf16", (), meta={"fused_from": "fused"}),
            },
            inverse_weight_map={},
        )
        dep = DependencyMap()
        dep.add_owner("k1", "s0")
        dep.add_owner("k2", "s0")
        dep.add_dependency("k1", "fused")
        dep.add_dependency("k2", "fused")
        groups = _schedule_groups(
            [
                RoutedTask(task=task1, route=[], route_ir_names=[]),
                RoutedTask(task=task2, route=[], route_ir_names=[]),
            ],
            dep,
        )
        assert len(groups) == 1  # 校验 fused 依赖合并为一组
        assert len(groups[0]) == 2

    def test_schedule_groups_group_by_shard_when_no_fused_dependency(self):
        task1 = IRTask(
            module_path="a",
            source_ir=SourceIR(kind=IRKind.FP8_BLOCK),
            target_ir=IRKind.W8A8_MXFP8,
            tensor_bindings={"weight": TensorRef("weight", "k1", "shard-a", "fp8", (2, 2))},
            inverse_weight_map={"shard-a.safetensors": ["k1", "k1_scale"]},
        )
        task2 = IRTask(
            module_path="b",
            source_ir=SourceIR(kind=IRKind.FP8_BLOCK),
            target_ir=IRKind.W8A8_MXFP8,
            tensor_bindings={"weight": TensorRef("weight", "k2", "shard-a", "fp8", (2, 2))},
            inverse_weight_map={"shard-a.safetensors": ["k2", "k2_scale"]},
        )
        task3 = IRTask(
            module_path="c",
            source_ir=SourceIR(kind=IRKind.FP8_BLOCK),
            target_ir=IRKind.W8A8_MXFP8,
            tensor_bindings={"weight": TensorRef("weight", "k3", "shard-b", "fp8", (2, 2))},
            inverse_weight_map={"shard-b.safetensors": ["k3", "k3_scale"]},
        )
        groups = _schedule_groups(
            [
                RoutedTask(task=task1, route=[], route_ir_names=[]),
                RoutedTask(task=task2, route=[], route_ir_names=[]),
                RoutedTask(task=task3, route=[], route_ir_names=[]),
            ],
            DependencyMap(),
        )
        assert len(groups) == 2  # 校验按 shard 分为两组
        assert sorted(len(g) for g in groups) == [1, 2]

    def test_schedule_groups_return_single_task_group_when_no_shard_map(self):
        task = IRTask(
            module_path="solo",
            source_ir=SourceIR(kind=IRKind.FLOAT),
            target_ir=IRKind.FLOAT,
            tensor_bindings={"weight": TensorRef("weight", "k", "s", "bf16", (2, 2))},
            inverse_weight_map={},
        )
        groups = _schedule_groups([RoutedTask(task=task, route=[], route_ir_names=[])], DependencyMap())
        assert len(groups) == 1  # 校验单任务独立成组
        assert groups[0][0].task.module_path == "solo"


class TestConvertExecutor:
    """测试 ConvertExecutor 类"""

    def test_run_yield_results_when_single_worker_and_one_task(self):
        root = nn.Module()
        linear = ModelFreeLinear(
            full_name="layers.0.q_proj",
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

        router = IRRouter()
        mock_proc = MagicMock()
        mock_proc.transform.return_value = nn.Linear(2, 2, bias=False)
        router._processors["MockProcessor"] = mock_proc

        edge = TransformEdge(
            src_ir=IRKind.FLOAT,
            dst_ir=IRKind.FLOAT,
            processor_name="MockProcessor",
            loss_level=LossLevel.LOSSLESS,
        )
        task = IRTask(
            module_path="layers.0.q_proj",
            source_ir=SourceIR(kind=IRKind.FLOAT),
            target_ir=IRKind.FLOAT,
            tensor_bindings=linear.tensor_bindings,
            inverse_weight_map={},
        )
        routed = RoutedTask(task=task, route=[edge], route_ir_names=[IRKind.FLOAT, IRKind.FLOAT])

        config = ConvertConfig(
            model_path="/m",
            save_path="/o",
            parallel=ParallelConfig(max_workers=1, task_granularity="ir_task"),
        )
        context = ConvertContext(config=config)
        context.virtual_tree = root
        context.reader = MagicMock()

        results = list(ConvertExecutor(router=router).run(context, [routed]))
        assert len(results) == 1  # 校验 run 流式产出一条结果
        assert results[0].module_path == "layers.0.q_proj"

    def test_run_uses_multinpu_when_parallel_mode_is_npu_multi(self):
        task = IRTask(
            module_path="layers.0.q_proj",
            source_ir=SourceIR(kind=IRKind.FLOAT),
            target_ir=IRKind.FLOAT,
            tensor_bindings={"weight": TensorRef("weight", "k", "s", "bf16", (2, 2))},
            inverse_weight_map={},
        )
        routed = [RoutedTask(task=task, route=[], route_ir_names=[])]

        config = ConvertConfig(
            model_path="/m",
            save_path="/o",
            parallel=ParallelConfig(
                max_workers=1,
                task_granularity="ir_task",
                device_indices=[0, 1],
            ),
        )
        context = ConvertContext(config=config)
        context.parallel_mode = "npu_multi"

        mock_result = IRResult(module_path="layers.0.q_proj", final_ir=IRKind.FLOAT)
        mock_scheduler = MagicMock()
        mock_scheduler.run.return_value = iter([mock_result])
        mock_scheduler.summaries = []
        mock_scheduler.worker_metas = []

        with patch(
            "msmodelslim.core.quant_service.modelslim_convert.impl.executor.MultiNpuConvertScheduler",
            return_value=mock_scheduler,
        ) as mock_cls:
            results = list(ConvertExecutor(router=IRRouter()).run(context, routed))

        assert len(results) == 1  # 校验 npu_multi 分支产出结果
        assert results[0].module_path == "layers.0.q_proj"
        # 校验 device_indices 来自 ConvertConfig.parallel，不再拆设备串
        assert mock_cls.call_args.kwargs["device_indices"] == [0, 1]
        assert mock_cls.call_args.kwargs["save_path"] == "/o"
        assert mock_cls.call_args.kwargs["return_mode"] == "module"

    def test_run_multinpu_skip_direct_write_when_dst_format_is_hf(self):
        """场景：npu_multi 但 dst_format 不是 AscendV1。
        预期：不打开 worker 直写，结果仍走 state_dict 回传。
        """
        task = IRTask(
            module_path="layers.0.q_proj",
            source_ir=SourceIR(kind=IRKind.FLOAT),
            target_ir=IRKind.FLOAT,
            tensor_bindings={"weight": TensorRef("weight", "k", "s", "bf16", (2, 2))},
            inverse_weight_map={},
        )
        routed = [RoutedTask(task=task, route=[], route_ir_names=[])]
        config = ConvertConfig(
            model_path="/m",
            save_path="/o",
            dst_format="huggingface",
            parallel=ParallelConfig(max_workers=1, device_indices=[0, 1]),
        )
        context = ConvertContext(config=config)
        context.parallel_mode = "npu_multi"
        mock_scheduler = MagicMock()
        mock_scheduler.run.return_value = iter([])
        mock_scheduler.summaries = []
        mock_scheduler.worker_metas = []
        with patch(
            "msmodelslim.core.quant_service.modelslim_convert.impl.executor.MultiNpuConvertScheduler",
            return_value=mock_scheduler,
        ) as mock_cls:
            list(ConvertExecutor(router=IRRouter()).run(context, routed))
        assert mock_cls.call_args.kwargs["save_path"] is None
        assert mock_cls.call_args.kwargs["return_mode"] == "state_dict"
