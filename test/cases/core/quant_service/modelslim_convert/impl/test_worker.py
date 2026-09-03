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

msmodelslim.core.quant_service.modelslim_convert.impl.worker 模块的单元测试
"""

import pytest

from msmodelslim.core.convert.config import ConvertConfig
from msmodelslim.core.quant_service.modelslim_convert.impl.worker import (
    GroupWorkPayload,
    convert_dependency_group,
)


class TestConvertDependencyGroup:
    """测试 convert_dependency_group 函数"""

    def test_convert_dependency_group_raise_runtime_error_when_queue_not_initialized(self, monkeypatch):
        import msmodelslim.core.quant_service.modelslim_convert.impl.worker as worker_mod

        monkeypatch.setattr(worker_mod, "_RESULT_QUEUE", None)
        payload = GroupWorkPayload(
            model_path="/m",
            routed_tasks=[],
            config=ConvertConfig(model_path="/m", save_path="/o"),
            worker_threads=1,
            budget=None,
            return_mode="module",
        )
        with pytest.raises(RuntimeError, match="result queue is not initialized"):
            convert_dependency_group(payload)
