#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# pylint: disable=abstract-class-instantiated

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
from torch import nn

from msmodelslim.core.const import DeviceType
from msmodelslim.core.quant_service.modelslim_v0.pipeline_interface import PipelineInterface
from msmodelslim.utils.exception import UnsupportedError


class _ConcretePipeline(PipelineInterface):
    @property
    def model_type(self) -> str:
        return "unit-test"

    @property
    def model_path(self) -> Path:
        return Path("unit-test")

    @property
    def trust_remote_code(self) -> bool:
        return False

    def handle_dataset(self, dataset, device=DeviceType.NPU):
        return list(dataset)

    def handle_dataset_by_batch(self, dataset, batch_size, device=DeviceType.NPU):
        return [list(dataset)[:batch_size]]

    def load_model(self, device=DeviceType.NPU) -> nn.Module:
        return nn.Identity()


def test_abstract_methods_reject_direct_instantiation_and_report_unsupported():
    with pytest.raises(TypeError, match="abstract"):
        PipelineInterface()

    with pytest.raises(UnsupportedError, match="This model does not support generate dataset"):
        PipelineInterface.handle_dataset(object(), [1])


def test_concrete_pipeline_uses_default_image_and_save_hooks(tmp_path):
    interface = _ConcretePipeline()

    assert interface.uses_image_calib() is False
    assert interface.modelslim_v0_save_postprocess(str(tmp_path)) is None
    assert not list(tmp_path.iterdir())
