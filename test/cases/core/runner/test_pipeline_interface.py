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

# pylint: disable=useless-parent-delegation

from pathlib import Path
from typing import Any, Generator, List

import pytest
from torch import nn

from msmodelslim.core.base.protocol import ProcessRequest
from msmodelslim.core.const import DeviceType
from msmodelslim.core.runner.pipeline_interface import PipelineInterface
from msmodelslim.utils.exception import ToDoError


class _TodoPipeline(PipelineInterface):
    """Pipeline stub that delegates abstract methods to base ToDoError paths."""

    @property
    def model_path(self):
        return Path("/tmp")

    @property
    def model_type(self) -> str:
        return "test"

    @property
    def trust_remote_code(self) -> bool:
        return False

    def handle_dataset(self, dataset: Any, device: DeviceType = DeviceType.NPU) -> List[Any]:
        return super().handle_dataset(dataset, device)

    def init_model(self, device: DeviceType = DeviceType.NPU) -> nn.Module:
        return super().init_model(device=device)

    def generate_model_visit(self, model: nn.Module) -> Generator[ProcessRequest, Any, None]:
        return super().generate_model_visit(model)

    def generate_model_forward(self, model: nn.Module, inputs: Any) -> Generator[ProcessRequest, Any, None]:
        return super().generate_model_forward(model, inputs)

    def enable_kv_cache(self, model: nn.Module, need_kv_cache: bool) -> None:
        return super().enable_kv_cache(model, need_kv_cache)


class TestPipelineInterface:
    """Tests for PipelineInterface."""

    def test_handle_dataset_raises_todo_error_when_not_implemented(self):
        """场景：调用默认 handle_dataset。预期：ToDoError。"""
        adapter = _TodoPipeline()
        with pytest.raises(ToDoError, match="generate dataset"):
            adapter.handle_dataset([])

    def test_init_model_raises_todo_error_when_not_implemented(self):
        """场景：调用默认 init_model。预期：ToDoError。"""
        adapter = _TodoPipeline()
        with pytest.raises(ToDoError, match="init model"):
            adapter.init_model()
