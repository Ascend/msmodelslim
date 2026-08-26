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

from unittest.mock import patch

import pytest

from msmodelslim.processor.analysis.distributed_utils import check_distributed_analysis_supported
from msmodelslim.processor.analysis.methods_base import LayerAnalysisMethod
from msmodelslim.processor.analysis.unary_operator.metrics.kurtosis import KurtosisAnalysisMethod
from msmodelslim.utils.exception import UnsupportedError


class _UnsupportedDpMethod(LayerAnalysisMethod):
    @property
    def name(self) -> str:
        return "unsupported_dp"

    def get_hook(self):
        return lambda *args, **kwargs: None


class TestSupportsDistributed:
    def test_default_is_false(self):
        assert _UnsupportedDpMethod().supports_distributed is False

    def test_kurtosis_declares_support(self):
        assert KurtosisAnalysisMethod().supports_distributed is True


class TestCheckDistributedAnalysisSupported:
    @patch("msmodelslim.processor.analysis.distributed_utils.dist")
    def test_raises_when_distributed_and_not_supported(self, mock_dist):
        mock_dist.is_initialized.return_value = True
        mock_dist.get_world_size.return_value = 4

        with pytest.raises(UnsupportedError, match="unsupported_dp"):
            check_distributed_analysis_supported(False, "unsupported_dp")

    @patch("msmodelslim.processor.analysis.distributed_utils.dist")
    def test_noop_when_single_device(self, mock_dist):
        mock_dist.is_initialized.return_value = False

        check_distributed_analysis_supported(False, "unsupported_dp")

    @patch("msmodelslim.processor.analysis.distributed_utils.dist")
    def test_noop_when_distributed_and_supported(self, mock_dist):
        mock_dist.is_initialized.return_value = True
        mock_dist.get_world_size.return_value = 2

        check_distributed_analysis_supported(True, "kurtosis")
