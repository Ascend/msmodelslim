# Copyright Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.

__all__ = ['qsin_qat', 'save_qsin_qat_model', 'QatConfig', 'get_logger', 'GISGD', 'QatModelWrapper', 'GraphOpt'] + \
          ['kl_loss', 'dkd_loss', 'rkd_loss', 'sckd_loss']

from .qat_kia.kd import kl_loss, dkd_loss, rkd_loss, sckd_loss
from .qat_kia.GISGD import GISGD
from .qat_kia.graph_opt import GraphOpt
from .compressor import qsin_qat, save_qsin_qat_model
from .qat_kia.wrapper import QatModelWrapper
from .compression.qat.qat_config import QatConfig


