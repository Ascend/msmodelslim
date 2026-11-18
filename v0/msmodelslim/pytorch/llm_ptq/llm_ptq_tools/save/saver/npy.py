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


import fnmatch
from dataclasses import dataclass, field
from logging import Logger
from typing import Union

from msmodelslim import logger as msmodelslim_logger
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.llm_ptq_utils import SAVE_TYPE_NUMPY
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.save.saver.base import BaseSaver
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.save.writer import NpyWriter


@dataclass
class NpySaverConfig:
    logger: Logger = field(init=False, default=msmodelslim_logger)
    save_directory: str

    @staticmethod
    def from_dict(d: dict):
        if isinstance(d, NpySaverConfig):
            return d
        if not isinstance(d, dict):
            raise TypeError(f'Npy save cfg must be an instance of dict, but got {type(d).__name__}')
        return NpySaverConfig(**d)

    def get_saver(self):
        return NpySaver(self)


class NpySaver(BaseSaver):
    type_ = SAVE_TYPE_NUMPY

    def __init__(self, cfg: Union[NpySaverConfig, dict]):
        super().__init__()

        cfg = NpySaverConfig.from_dict(cfg)
        self.logger = cfg.logger
        self.writer_list = []

        def create_helper(file_name: str):
            writer = NpyWriter(logger=self.logger, save_directory=cfg.save_directory, file_name=file_name)
            self.writer_list.append(writer)
            return writer

        self.quant_weight_writer = create_helper(file_name="quant_weight.npy")
        self.input_scale_writer = create_helper(file_name="input_scale.npy")
        self.input_offset_writer = create_helper(file_name="input_offset.npy")
        self.quant_bias_writer = create_helper(file_name="quant_bias.npy")
        self.deq_scale_writer = create_helper(file_name="deq_scale.npy")
        self.weight_scale_writer = create_helper(file_name="weight_scale.npy")
        self.weight_offset_writer = create_helper(file_name="weight_offset.npy")
        self.kv_cache_scale_writer = create_helper(file_name="kv_cache_scale.npy")
        self.kv_cache_offset_writer = create_helper(file_name="kv_cache_offset.npy")
        self.fa_quant_scale_writer = create_helper(file_name="fa_quant_scale.npy")
        self.fa_quant_offset_writer = create_helper(file_name="fa_quant_offset.npy")
        self.anti_fp_norm_writer = create_helper(file_name="anti_fp_norm.npy")

    def pre_process(self) -> None:
        pass

    def save(self, name, meta, data) -> None:
        # keep suffix
        if fnmatch.fnmatch(name, '*.fa_*scale*'):
            self.fa_quant_scale_writer.write(name, data)
        elif fnmatch.fnmatch(name, '*.fa_*offset*'):
            self.fa_quant_offset_writer.write(name, data)
        elif fnmatch.fnmatch(name, '*norm.weight') or fnmatch.fnmatch(name, '*norm.bias'):
            self.anti_fp_norm_writer.write(name, data)
        elif name.endswith('.kv_cache_scale'):
            self.kv_cache_scale_writer.write(name, data)
        elif name.endswith('.kv_cache_offset'):
            self.kv_cache_offset_writer.write(name, data)
        # remove suffix
        elif name.endswith('.weight') or name.endswith('.bias'):
            new_name = name.replace('.weight', '').replace('.bias', '')
            self.quant_weight_writer.write(new_name, data)
        elif name.endswith('.quant_bias'):
            new_name = name.replace('.quant_bias', '')
            self.quant_bias_writer.write(new_name, data)
        elif name.endswith('.deq_scale'):
            new_name = name.replace('.deq_scale', '')
            self.deq_scale_writer.write(new_name, data)
        elif name.endswith('.input_scale'):
            new_name = name.replace('.input_scale', '')
            self.input_scale_writer.write(new_name, data)
        elif name.endswith('.input_offset'):
            new_name = name.replace('.input_offset', '')
            self.input_offset_writer.write(new_name, data)
        elif name.endswith('.weight_scale'):
            new_name = name.replace('.weight_scale', '')
            self.weight_scale_writer.write(new_name, data)
        elif name.endswith('.weight_offset'):
            new_name = name.replace('.weight_offset', '')
            self.weight_offset_writer.write(new_name, data)

    def post_process(self) -> None:
        for writer in self.writer_list:
            writer.close()

        self.logger.info('Numpy weight saved successfully')
