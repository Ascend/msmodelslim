#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# pylint: disable=use-implicit-booleaness-not-comparison,abstract-class-instantiated

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

from pathlib import Path

import pytest

from msmodelslim.core.convert.config import ConvertConfig
from msmodelslim.core.convert.protocol import ConvertContext, ICheckpointReader, IIRTransformProcessor
from msmodelslim.core.convert.types import IRKind


def test_convert_context_resolves_paths_and_runtime_defaults():
    context = ConvertContext(ConvertConfig(model_path="input-model", save_path="output-model"))

    assert context.model_path == Path("input-model")
    assert context.save_path == Path("output-model")
    assert context.resolved_worker_device == "cpu"
    assert context.parallel_mode == "thread"
    assert context.resolved_worker_devices == []


def test_checkpoint_reader_remains_abstract():
    with pytest.raises(TypeError):
        ICheckpointReader()


def test_ir_transform_processor_protocol_accepts_complete_processor():
    class Processor:
        name = "identity"
        src_ir = IRKind.FLOAT
        dst_ir = IRKind.FLOAT
        requires_forward = False
        requires_calibration = False
        loss_level = "lossless"

        def transform(self, module, context):
            return module

    class IncompleteProcessor:
        name = "incomplete"

    assert isinstance(Processor(), IIRTransformProcessor)
    assert not isinstance(IncompleteProcessor(), IIRTransformProcessor)
