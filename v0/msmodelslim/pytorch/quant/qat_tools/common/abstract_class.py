# Copyright Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.

import copy
from abc import ABC, abstractmethod
from typing import List, Union, Tuple
from torch import nn as nn
import torch

from msmodelslim import logger as msmodelslim_logger
from msmodelslim.pytorch.quant.qat_tools.utils.utils import CallParams
from msmodelslim.pytorch.quant.qat_tools.qat_kia.quantize import QuantBaseOperation


class CompressModelWrapper(ABC):
    def __init__(self,
                 model: nn.Module,
                 cfg=None,
                 logger=None,
                 dummy_input: Union[
                     torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor],
                     CallParams
                 ] = None):
        super().__init__()
        self.model = model
        self.cfg = cfg
        self.logger = logger

        if self.logger is None:
            self.logger = msmodelslim_logger
        self.disable_names = cfg.disable_names
        self.orin_model = copy.deepcopy(model)
        self._dummy_input = dummy_input

    @staticmethod
    def _setattr(model: nn.Module,
                 module_name: str,
                 module: nn.Module):
        QuantBaseOperation.setattr_(model, module_name, module)

    @staticmethod
    def _parse_dummy_input(dummy_input: Union[
        torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor],
        CallParams, None
    ] = None):
        if isinstance(dummy_input, CallParams):
            ret_input = (item for item in dummy_input.args)
            for _, val in dummy_input.kwargs:
                ret_input.append(val)
            return ret_input

        return dummy_input

    @abstractmethod
    def wrap(self):
        raise NotImplementedError

    @abstractmethod
    def model_export(self):
        raise NotImplementedError