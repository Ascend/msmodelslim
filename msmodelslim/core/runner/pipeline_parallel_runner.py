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

from typing import Optional, List, Any

from torch import nn

from msmodelslim.core.const import DeviceType
from msmodelslim.core.runner.generated_runner import GeneratedRunner
from msmodelslim.utils.logging import logger_setter, get_logger


@logger_setter()
class PPRunner(GeneratedRunner):
    def run(
        self,
        model: nn.Module = None,
        calib_data: Optional[List[Any]] = None,
        device: DeviceType = DeviceType.NPU,
        device_indices: Optional[List[int]] = None,
    ):
        if device_indices is not None:
            get_logger().warning(
                "Specifying device indices is not supported in model_wise runner. Device indices will be ignored. "
            )
        super().run(model, calib_data, device, device_indices)
