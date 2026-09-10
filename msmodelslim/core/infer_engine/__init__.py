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

from .dp_fake_quant import DPFakeQuantInferenceEngine
from .fake_quant import FakeQuantInferenceEngine, PrefillLoop, Session, WeightManager
from .interface import IInferenceEngine, InferenceConfig, InferenceResult

__all__ = [
    "FakeQuantInferenceEngine",
    "DPFakeQuantInferenceEngine",
    "IInferenceEngine",
    "InferenceConfig",
    "InferenceResult",
    "PrefillLoop",
    "Session",
    "WeightManager",
]
