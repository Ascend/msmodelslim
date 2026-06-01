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

from abc import ABC, abstractmethod
from typing import Tuple

import torch
from torch import nn


class AscendV1GlobalModelDtypeInterface(ABC):
    """Interface for adapters that expose the global model torch dtype (e.g. for Saver to decide deq_scale int64)."""

    @abstractmethod
    def get_global_model_torch_dtype(self) -> torch.dtype:
        """
        Return the global torch dtype used for model loading/calibration.
        Used by Saver and other components to infer precision (e.g. whether bfloat16).
        """
        pass


class AscendV1SaveInterface(ABC):
    def ascendv1_save_postprocess(self, model: nn.Module, save_directory: str) -> None:
        """
        导出件后处理
        @param model: 量化模型
        @param save_directory: 包含导出件（如config.json，quant_model_description.json等）的量化模型存储路径
        """
        pass

    def ascendv1_save_module_preprocess(
        self, prefix: str, module: nn.Module, model: nn.Module
    ) -> Tuple[str, nn.Module]:
        """
        在保存模块前，对模块进行预处理，返回新的前缀和模块
        @param prefix: 模块的前缀路径
        @param module: 待处理的模块
        @param model: 模型
        @return: 返回(prefix, module)
        """
        pass
