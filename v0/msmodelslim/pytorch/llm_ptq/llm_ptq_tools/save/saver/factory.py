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

from ascend_utils.common.security import check_element_type
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.llm_ptq_utils import SAVE_TYPE_SAFE_TENSOR, \
    SAVE_TYPE_NUMPY, SAVE_TYPE_ASCENDV1, QuantType
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.save.saver.ascend_v1 \
    import AscendV1SaverConfig, AscendV1Saver
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.save.saver.base import BaseSaver
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.save.saver.multi import MultiSaver
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.save.saver.npy import NpySaverConfig, NpySaver
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.save.saver.safetensors import SafetensorsSaverConfig, SafetensorsSaver


class SaverFactory:
    @staticmethod
    def create(typ, **kwargs) -> BaseSaver:
        if isinstance(typ, list):
            check_element_type(typ, str)
            if len(typ) == 1:
                return SaverFactory.create(typ[0], **kwargs)
            return SaverFactory.create_multi_saver(typ, **kwargs)

        if not isinstance(typ, str):
            raise TypeError(f'Unsupported save type: {type(typ).__name__}')

        if typ == SAVE_TYPE_SAFE_TENSOR:
            return SaverFactory.create_safetensors_saver(typ, **kwargs)
        if typ == SAVE_TYPE_NUMPY:
            return SaverFactory.create_npy_saver(typ, **kwargs)
        if typ == SAVE_TYPE_ASCENDV1:
            return SaverFactory.create_ascend_v1_saver(typ, **kwargs)
        raise ValueError(f'Unsupported save type: {type(typ).__name__}')

    @staticmethod
    def create_multi_saver(types, **kwargs) -> MultiSaver:
        saver = MultiSaver()
        for typ in types:
            saver.register(SaverFactory.create(typ, **kwargs))
        return saver

    @staticmethod
    def create_ascend_v1_saver(typ, **kwargs) -> AscendV1Saver:
        output_dir = kwargs['output_dir']
        cfg = kwargs['cfg']
        safetensors_name = kwargs['safetensors_name']
        json_name = kwargs['json_name']
        part_file_size = kwargs['part_file_size']
        group_size = kwargs['group_size']
        enable_communication_quant = kwargs['enable_communication_quant']

        model_quant_type = QuantType.W8A8_MIX \
            if cfg.model_quant_type == QuantType.W8A8 and cfg.pdmix else cfg.model_quant_type

        return AscendV1SaverConfig(output_dir=output_dir,
                                   model_quant_type=model_quant_type,
                                   use_kvcache_quant=cfg.use_kvcache_quant,
                                   use_fa_quant=cfg.use_fa_quant,
                                   safetensors_name=safetensors_name,
                                   json_name=json_name,
                                   part_file_size=part_file_size,
                                   group_size=group_size,
                                   enable_communication_quant=enable_communication_quant,
                                   ).get_saver()

    @staticmethod
    def create_safetensors_saver(typ, **kwargs) -> SafetensorsSaver:
        output_dir = kwargs['output_dir']
        cfg = kwargs['cfg']
        safetensors_name = kwargs['safetensors_name']
        json_name = kwargs['json_name']
        part_file_size = kwargs['part_file_size']

        return SafetensorsSaverConfig(output_dir=output_dir,
                                      model_quant_type=cfg.model_quant_type,
                                      use_kvcache_quant=cfg.use_kvcache_quant,
                                      use_fa_quant=cfg.use_fa_quant,
                                      safetensors_name=safetensors_name,
                                      json_name=json_name,
                                      part_file_size=part_file_size).get_saver()

    @staticmethod
    def create_npy_saver(typ, **kwargs) -> NpySaver:
        output_dir = kwargs['output_dir']
        return NpySaverConfig(save_directory=output_dir).get_saver()
