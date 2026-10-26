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

from testing_utils.mock import mock_init_config

mock_init_config()

from msmodelslim.app.analysis.result_displayer_infra import AnalysisResultDisplayerInfra  # noqa: E402
from msmodelslim.core.analysis_service import AnalysisScope  # noqa: E402


class _RecordingAnalysisResultDisplayer(AnalysisResultDisplayerInfra):
    def __init__(self):
        self.calls = []

    def display_result(
        self,
        result,
        topk,
        scope: Optional[AnalysisScope] = None,
        save_path: Optional[str] = None,
        model_type: Optional[str] = None,
    ) -> None:
        self.calls.append((result, topk, scope, save_path, model_type))


def test_analysis_result_displayer_infra_cannot_instantiate_when_display_result_is_abstract():
    with pytest.raises(TypeError, match="abstract"):
        AnalysisResultDisplayerInfra()  # pylint: disable=abstract-class-instantiated


def test_display_result_receives_all_arguments_when_concrete_subclass_is_called():
    result = object()
    displayer = _RecordingAnalysisResultDisplayer()

    displayer.display_result(
        result,
        7,
        AnalysisScope.LAYER,
        save_path="/tmp/result.yaml",
        model_type="TinyLlama",
    )

    assert displayer.calls == [
        (result, 7, AnalysisScope.LAYER, "/tmp/result.yaml", "TinyLlama"),
    ]
