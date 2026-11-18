# Copyright Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.

from __future__ import division, absolute_import, print_function

from torch import nn as nn

from msmodelslim.pytorch.quant.qat_tools.qat_kia.wrapper import QatModelWrapper
from msmodelslim.pytorch.quant.qat_tools.common.abstract_class import CompressModelWrapper
from msmodelslim.pytorch.quant.qat_tools.common.config import Config


class ModelWrapperFactory(object):
    @staticmethod
    def create_model_wrapper(wrapper_type: str,
                             model: nn.Module,
                             cfg: Config = None,
                             logger=None,
                             dummy_input=None,
                             **kwargs) -> CompressModelWrapper:
        if wrapper_type.lower() == 'aqt':
            return QatModelWrapper(model, cfg=cfg, logger=logger, dummy_input=dummy_input)
        else:
            raise NotImplementedError(f'{wrapper_type} wrapper is not supported')
