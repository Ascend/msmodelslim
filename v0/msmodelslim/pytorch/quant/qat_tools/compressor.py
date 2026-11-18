# Copyright Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.

from __future__ import absolute_import, division, print_function
from typing import Union, List, Tuple
import logging

import torch.nn as nn
import torch

from ascend_utils.common.security.pytorch import check_torch_module
from ascend_utils.common.security import check_type, get_valid_read_path, get_valid_write_path
from msmodelslim import logger as msmodelslim_logger
from msmodelslim.pytorch.quant.qat_tools.common.factory import ModelWrapperFactory
from msmodelslim.pytorch.quant.qat_tools.common.config import Config
from msmodelslim.pytorch.quant.qat_tools.utils.utils import CallParams
from .compression.qat.qat_config import QatConfig



class Compressor(object):
    """ Compressor to manage the compression process."""

    def __init__(self,
                 model: nn.Module,
                 cfg: Config,
                 logger=None):
        check_type(cfg, Config, param_name="cfg")
        check_torch_module(model)
        self.model = model
        if cfg is None:
            raise ValueError('Compress related config is needed!')
        self.cfg = cfg
        self.logger = logger
        if self.logger is None:
            self.logger = msmodelslim_logger
        self.model_wrapper = None
        self._dummy_input = None
        self.is_compressed = False

    @property
    def compressed_model(self):
        if self.is_compressed:
            return self.model
        else:
            raise Exception('model has not been compressed yet!')

    def compress(self,
                 dummy_input: Union[
                     torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor],
                     CallParams, None
                 ] = None,
                 **kwargs):
        """ Wrap the model for compression."""
        self._dummy_input = dummy_input

        try:
            self.model_wrapper = ModelWrapperFactory.create_model_wrapper(self.cfg.method,
                                                                          self.model,
                                                                          cfg=self.cfg,
                                                                          logger=self.logger,
                                                                          dummy_input=self._dummy_input,
                                                                          **kwargs)
        except NotImplementedError as e:
            self.logger.error(e)
            raise

        self.model_wrapper.wrap()
        self.is_compressed = True

    def train(self, trainer=None):
        """ Train the model to adapt to compression."""
        if trainer is not None:
            try:
                trainer.train()
            except AttributeError as e:
                self.logger.error(e)
        else:
            raise ValueError('should input a trainer to train the compressed model')

    def export(self, dummy_input=None):
        """ Export the onnx for deployment."""
        if self._dummy_input is None and dummy_input is None:
            self.logger.error("Shall set dummy input for onnx export")
            return

        if dummy_input is not None:
            self._dummy_input = dummy_input

        self.model_wrapper.model_export(self._dummy_input)


def qsin_qat(model, quant_config, quant_logger):
    if not isinstance(model, nn.Module):
        raise TypeError("model must be nn.Module!")
    if not isinstance(quant_config, Config):
        raise TypeError("quant_config must be Config!")
    if not isinstance(quant_logger, logging.Logger):
        raise TypeError("quant_logging must be logging.Logger!")

    compressor = Compressor(model, quant_config, quant_logger)
    compressor.compress()
    model = compressor.model

    return model


def save_qsin_qat_model(model, save_onnx_name, dummy_input, saved_ckpt, input_names):
    if not isinstance(model, nn.Module):
        raise TypeError("model must be nn.Module!")
    if not isinstance(dummy_input, torch.Tensor):
        raise TypeError("dummy_input must be torch.Tensor!")
    saved_ckpt = get_valid_read_path(saved_ckpt)
    save_onnx_name = get_valid_write_path(save_onnx_name)

    quant_config = QatConfig(compressed_model_checkpoint=saved_ckpt,
                             quant_mode=False,
                             save_onnx_name=save_onnx_name,
                             input_names=input_names)
    compressor = Compressor(model, quant_config)
    compressor.compress(dummy_input=dummy_input)
    compressor.export()
