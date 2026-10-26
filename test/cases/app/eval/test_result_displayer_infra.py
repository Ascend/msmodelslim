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

from typing import Optional

import pytest

from msmodelslim.app.eval.result_displayer_infra import InferenceResultDisplayerInfra


class _RecordingInferenceResultDisplayer(InferenceResultDisplayerInfra):
    def __init__(self):
        self.calls = []

    def display_result(self, result, device_indices: Optional[list[int]] = None) -> None:
        self.calls.append((result, device_indices))


def test_inference_result_displayer_infra_cannot_instantiate_when_display_result_is_abstract():
    with pytest.raises(TypeError, match="abstract"):
        InferenceResultDisplayerInfra()  # pylint: disable=abstract-class-instantiated


def test_display_result_receives_result_and_device_indices_when_concrete_subclass_is_called():
    result = object()
    displayer = _RecordingInferenceResultDisplayer()

    displayer.display_result(result, device_indices=[0, 2])

    assert displayer.calls == [(result, [0, 2])]
