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

from unittest.mock import MagicMock

import torch
from torch import nn

from msmodelslim.core.base.processor import BaseProcessor
from msmodelslim.core.base.protocol import BatchProcessRequest


class TestBaseProcessor:
    """Tests for BaseProcessor."""

    def test_process_populates_outputs_when_module_forward_has_args(self):
        """场景：请求含模块与前向参数。预期：outputs 写入前向结果。"""
        linear = nn.Linear(2, 2)
        processor = BaseProcessor(linear)
        request = BatchProcessRequest(
            name="linear",
            module=linear,
            datas=[((torch.randn(3, 2),), {})],
            outputs=None,
        )
        processor.process(request)
        assert request.outputs is not None
        assert len(request.outputs) == 1
        assert request.outputs[0].shape == (3, 2)

    def test_process_leaves_outputs_empty_when_no_args_or_kwargs(self):
        """场景：请求无 args/kwargs。预期：outputs 为空列表。"""
        module = MagicMock()
        processor = BaseProcessor(nn.Linear(2, 2))
        request = BatchProcessRequest(
            name="mock",
            module=module,
            datas=[((), {})],
            outputs=None,
        )
        processor.process(request)
        assert request.outputs == []

    def test_pre_run_post_run_noop_when_called(self):
        """场景：调用 pre_run/post_run 默认实现。预期：不抛异常。"""
        processor = BaseProcessor(nn.Linear(2, 2))
        processor.pre_run()
        processor.post_run()
