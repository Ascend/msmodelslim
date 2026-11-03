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
MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details.
-------------------------------------------------------------------------

msmodelslim.core.quant_service.modelslim_convert.impl.worker 模块的单元测试
"""

from unittest.mock import MagicMock, patch

from msmodelslim.core.convert.config import ConvertConfig
from msmodelslim.core.quant_service.modelslim_convert.impl.worker import bootstrap_convert_worker


def test_bootstrap_convert_worker_registers_paretoq_processors_when_available():
    router = MagicMock()
    runner = MagicMock()
    with (
        patch(
            "msmodelslim.processor.convert.registry.register_convert_processors",
            return_value=router,
        ),
        patch(
            "msmodelslim.infra.io.checkpoint_reader.CheckpointReader",
            return_value=MagicMock(),
        ),
        patch(
            "msmodelslim.core.quant_service.modelslim_convert.impl.worker.DependencyGroupRunner",
            return_value=runner,
        ),
        patch(
            "msmodelslim.processor.paretoq.impl.ParetoQImpl.get_convert_processors",
            return_value=[MagicMock(name="w4a8"), MagicMock(name="w4a4")],
        ) as get_procs,
    ):
        ctx, out_runner = bootstrap_convert_worker(
            model_path="/tmp/model",
            config=ConvertConfig(model_path="/tmp/model", save_path="/tmp/out", dst_format="ascendv1"),
            device="cpu",
        )
    assert out_runner is runner
    assert ctx.resolved_worker_device == "cpu"
    assert router.register_processor.call_count == 2
    get_procs.assert_called_once()


def test_bootstrap_convert_worker_continues_when_paretoq_unavailable():
    router = MagicMock()
    runner = MagicMock()
    with (
        patch(
            "msmodelslim.processor.convert.registry.register_convert_processors",
            return_value=router,
        ),
        patch(
            "msmodelslim.infra.io.checkpoint_reader.CheckpointReader",
            return_value=MagicMock(),
        ),
        patch(
            "msmodelslim.core.quant_service.modelslim_convert.impl.worker.DependencyGroupRunner",
            return_value=runner,
        ),
        patch(
            "msmodelslim.processor.paretoq.impl.ParetoQImpl.get_convert_processors",
            side_effect=ImportError("no paretoq"),
        ),
    ):
        _, out_runner = bootstrap_convert_worker(
            model_path="/tmp/model",
            config=ConvertConfig(model_path="/tmp/model", save_path="/tmp/out", dst_format="ascendv1"),
            device="cpu",
        )
    assert out_runner is runner
    router.register_processor.assert_not_called()
