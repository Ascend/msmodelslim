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

import os.path
from dataclasses import dataclass, field
from logging import Logger
from typing import Optional, Union

from msmodelslim import logger as msmodelslim_logger
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.llm_ptq_utils import SAVE_TYPE_SAFE_TENSOR
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.save.saver.safetensors import SafetensorsSaverConfig, SafetensorsSaver
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.save.saver.base import BaseSaver
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.save.writer import BufferedSafetensorsWriter, JsonDescriptionWriter, \
    SafetensorsWriter


@dataclass
class AscendV1SaverConfig(SafetensorsSaverConfig):
    group_size: int = 0

    def __post_init__(self):
        if not isinstance(self.safetensors_name, str):
            default_safetensors_name = f"quant_model_weight_{self.model_quant_type.lower()}.safetensors"
            self.logger.warning(f"invalid `safetensors_name`, defaulting to `{default_safetensors_name}`")
            self.safetensors_name = default_safetensors_name
        if not isinstance(self.json_name, str):
            default_json_name = f"quant_model_description.json"
            self.logger.warning(f"invalid `json_name`, defaulting to `{default_json_name}`")
            self.json_name = default_json_name

    @staticmethod
    def from_dict(d: dict):
        if isinstance(d, AscendV1SaverConfig):
            return d
        if not isinstance(d, dict):
            raise TypeError(f'AscendV1 save config must be an instance of dict, but got {type(d).__name__}')
        return AscendV1SaverConfig(**d)

    def get_saver(self):
        return AscendV1Saver(self)


class AscendV1Saver(SafetensorsSaver):
    type_ = SAVE_TYPE_SAFE_TENSOR
    version_name_ = "1.0.0"

    def __init__(self, cfg: Union[AscendV1SaverConfig, dict]):
        super().__init__(cfg)

        cfg = AscendV1SaverConfig.from_dict(cfg)
        self.logger = cfg.logger

        if cfg.part_file_size is None:
            file_path = os.path.join(cfg.output_dir, cfg.safetensors_name)
            self.weight_writer = SafetensorsWriter(logger=self.logger, file_path=file_path)
        else:
            file_name_prefix = cfg.safetensors_name.replace('.safetensors', '')
            self.weight_writer = BufferedSafetensorsWriter(logger=self.logger,
                                                           max_gb_size=cfg.part_file_size,
                                                           save_directory=cfg.output_dir,
                                                           save_prefix=file_name_prefix)
        self.meta_writer = JsonDescriptionWriter(logger=self.logger,
                                                 model_quant_type=cfg.model_quant_type,
                                                 json_name=cfg.json_name,
                                                 save_directory=cfg.output_dir,
                                                 use_kvcache_quant=cfg.use_kvcache_quant,
                                                 use_fa_quant=cfg.use_fa_quant,
                                                 version_name=self.version_name_,
                                                 group_size=cfg.group_size,
                                                 enable_communication_quant=cfg.enable_communication_quant)
