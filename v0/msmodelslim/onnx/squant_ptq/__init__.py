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
"""
__all__ = ['QuantConfig']

from msmodelslim.onnx.squant_ptq.quant_config import QuantConfig

try:
    from msmodelslim.onnx.squant_ptq.onnx_quant_tools import OnnxCalibrator
except ModuleNotFoundError as exception:
    from msmodelslim import logger
    logger.warning("Can not import OnnxCalibrator from: %s", exception)
else:
    __all__ += ['OnnxCalibrator']

