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


from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.quant_config.quant_config_classes import BaseConfig
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.quant_config.quant_config_classes.config_utils import \
    check_and_generate_config_param


class SparseQuantConfig(BaseConfig):
    """
    稀疏量化的Config
    """

    def __init__(self,
                 last_config: BaseConfig,
                 act_method: int = 1,
                 fraction: float = 0.01,
                 nonuniform: bool = False,
                 is_lowbit: bool = False,
                 model_type: str = "",
                 do_smooth: bool = False,
                 use_sigma: bool = False,
                 sigma_factor: float = 3.0
                 ):
        """
        Args:
            last_config: 上一个config，基于该config进行修改
            act_method: activation量化的方式
            fraction: 稀疏量化的调优参数
            nonuniform: 稀疏量化中默认为非均匀量化，决定权重是否均匀量化。建议采用False
            is_lowbit: 是否使用稀疏量化的low bit算法
            model_type: 稀疏量化low bit中指定模型类别
            do_smooth: 稀疏量化low bit中是否对activation使用smooth算法
            use_sigma: 稀疏量化low bit的outlier选取范围，False时使用fraction，True使用sigma_factor
            sigma_factor:  稀疏量化low bit的调优参数
        """
        super().__init__()
        # 获取上一个config的所有属性
        self._init_attribute_by_config(last_config)

        # 更新当前特有的属性
        self.act_method = act_method  # 只能1或者2
        self.fraction = fraction  # 稀疏量化调优参数
        self.nonuniform = nonuniform  # 稀疏量化中默认为非均匀量化，决定权重是否均匀量化。建议采用False
        self.is_lowbit = is_lowbit  # 是否调用lowbit稀疏量化
        self.model_type = model_type  # lowbit才会用到，即将取消
        self.do_smooth = do_smooth  # lowbit 采用smooth quant进行activation的outlier处理
        self.use_sigma = use_sigma  # lowbit outlier 选取范围处理，False 时使用 fraction， True 使用 sigma_factor
        self.sigma_factor = sigma_factor  # lowbit 调优参数

        # 一旦调用 SparseQuantConfig，默认为True
        self.co_sparse = True

        # 校验参数
        check_and_generate_config_param(self)
