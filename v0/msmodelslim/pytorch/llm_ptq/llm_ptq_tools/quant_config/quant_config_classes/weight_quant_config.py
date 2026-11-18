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


from ascend_utils.common.security import check_type

from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.quant_config.quant_config_classes import BaseConfig
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.quant_config.quant_config_classes.config_utils import \
    check_and_generate_config_param

BLOCK_SIZE_LIST = [64, 128, 256, 512, 1024, 2048, 4096]


class WeightQuantConfig(BaseConfig):
    """
    权重量化的Config
    包含：W8A16、W4A16
    """

    def __init__(self,
                 last_config: BaseConfig,
                 w_method: str = 'MinMax',
                 mm_tensor: bool = True,
                 w_sym: bool = True,
                 group_size: int = 64,
                 block_size: int = 64,
                 ):
        """
        Args:
            last_config: 上一个config，基于该config进行修改
            w_method: weight量化的方式
            mm_tensor: matmul的tensor（即weight）是per-tensor还是per-channel
            w_sym: 权重是否对称量化
            group_size: per-group场景下group的大小，通常取64或128
        """
        super().__init__()
        # 获取上一个config的所有属性
        self._init_attribute_by_config(last_config)

        # 更新当前特有的属性
        self.w_method = w_method
        self.mm_tensor = mm_tensor  # 权重量化为per-tensor，False为per-channel
        self.w_sym = w_sym  # 权重是否对称量化，只有在权重量化中支持调整
        self.group_size = group_size
        self.block_size = block_size

        # 校验参数
        check_type(self.block_size, int, param_name="block_size")
        if self.block_size not in BLOCK_SIZE_LIST:
            raise ValueError(f"block_size must be among choice {BLOCK_SIZE_LIST}, please check it.")
        check_and_generate_config_param(self)
