# Copyright Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.

from __future__ import division, absolute_import, print_function

from torch import nn as nn

from ascend_utils.common.security import check_type
from msmodelslim.pytorch.quant.qat_tools.qat_kia.wrapper import QatModelWrapper
from msmodelslim.pytorch.quant.qat_tools.common.abstract_class import CompressModelWrapper


class ModelWrapperFactory(object):
    @staticmethod
    def create_model_wrapper(wrapper_type,
                             model,
                             cfg=None,
                             logger=None) -> CompressModelWrapper:
        check_type(wrapper_type, str, param_name="wrapper_type")
        check_type(model, nn.Module, param_name="model")
        if wrapper_type.lower() == 'qat':
            return QatModelWrapper(model, cfg=cfg, logger=logger)
        else:
            raise NotImplementedError(f'{wrapper_type} wrapper is not supported')
