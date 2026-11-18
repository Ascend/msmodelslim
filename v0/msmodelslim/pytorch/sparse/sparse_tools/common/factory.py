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

from torch import nn as nn

from msmodelslim.pytorch.sparse.sparse_tools.sparse_kia.wrapper import SparseModelWrapper
from msmodelslim.pytorch.sparse.sparse_tools.common.abstract_class import CompressModelWrapper
from msmodelslim.pytorch.sparse.sparse_tools.common.config import Config


class ModelWrapperFactory(object):
    @staticmethod
    def create_model_wrapper(wrapper_type: str,
                             model: nn.Module,
                             cfg: Config = None,
                             logger=None,
                             **kwargs) -> CompressModelWrapper:
        if wrapper_type.lower() == 'sparse':
            return SparseModelWrapper(model,
                                      cfg=cfg,
                                      logger=logger,
                                      dataset=kwargs.get("dataset"))
        else:
            raise NotImplementedError(f'{wrapper_type} wrapper is not supported')
