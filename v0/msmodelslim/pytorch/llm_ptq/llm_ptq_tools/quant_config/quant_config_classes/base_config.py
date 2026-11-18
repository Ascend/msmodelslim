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


from msmodelslim import logger as msmodelslim_logger
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.quant_config.quant_config_classes.config_utils import \
    check_and_generate_config_param


class BaseConfig:
    """
    量化Config的基础类，存放通用参数

    为保障前向兼容性，BaseConfig开放所有QuantConfig的可调参数
    """

    def __init__(self,
                 w_bit: int = 8,
                 a_bit: int = 8,
                 act_method: int = 1,
                 w_method: str = 'MinMax',
                 disable_names: object = None,
                 pr: float = 1.0,
                 mm_tensor: bool = True,
                 dev_type: object = 'cpu',
                 dev_id: int = None,
                 fraction: float = 0.01,
                 nonuniform: bool = False,
                 co_sparse: bool = False,
                 w_sym: bool = True,
                 is_lowbit: bool = False,
                 do_smooth: bool = False,
                 use_sigma: bool = False,
                 sigma_factor: float = 3.0,
                 disable_last_linear: bool = True,
                 use_kvcache_quant: bool = False,
                 open_outlier: bool = True,
                 is_dynamic: bool = False,
                 group_size: int = 64,
                 percdamp: float = 0.01,
                 pdmix: bool = False,
                 ):
        """
        Args:
            w_bit: 量化的weight bit位数
            a_bit: 量化的activation bit位数
            act_method: activation量化的方式
            w_method: weight量化的方式
            disable_names: 指定量化回退层的名称
            pr: 量化的一个概率参数，非1时量化生成的参数带有随机性
            mm_tensor: matmul的tensor（即weight）是per-tensor还是per-channel
            dev_type: 运行的设备
            dev_id: 只有在非cpu上才需要指定id
            fraction: 稀疏量化的调优参数
            nonuniform: 稀疏量化中默认为非均匀量化，决定权重是否均匀量化。建议采用False
            co_sparse: 是否启用稀疏量化
            w_sym: 权重是否对称量化
            is_lowbit: 是否使用稀疏量化的low bit算法
            do_smooth: 稀疏量化low bit中是否对activation使用smooth算法
            use_sigma: 稀疏量化low bit的 outlier 选取范围，False 时使用 fraction，True 使用 sigma_factor
            sigma_factor: 稀疏量化low bit 的调优参数
            disable_last_linear: 是否自动回退最后一个Linear，通常是lm head层
            use_kvcache_quant: 是否使用kv cache量化
            open_outlier: 是否开启权重异常值划分
            is_dynamic: 是否使用动态量化，即w8a8中的activation动态生成
            group_size: per-group场景下group的大小，通常取64或128
            percdamp: GPTQ所使用的矩阵正定偏置系数,当GPTQ运行出现非正定矩阵导致的报错时可以增大该参数
        """
        # 建议在只调整下列6个参数，其余参数在对应特性的Config中设置
        self.w_bit = w_bit
        self.a_bit = a_bit
        self.disable_names = disable_names if disable_names is not None else []
        self.disable_last_linear = disable_last_linear
        self.dev_type = dev_type
        self.dev_id = dev_id

        # 下面的参数，应该在特定的量化方法中调用。为了兼容老的调用方式，开放给用户
        self.act_method = act_method
        self.w_method = w_method
        self.pr = pr
        self.mm_tensor = mm_tensor
        self.w_sym = w_sym
        self.fraction = fraction
        self.nonuniform = nonuniform
        self.co_sparse = co_sparse
        self.is_lowbit = is_lowbit
        self.do_smooth = do_smooth
        self.use_sigma = use_sigma
        self.sigma_factor = sigma_factor
        self.use_kvcache_quant = use_kvcache_quant
        self.is_dynamic = is_dynamic
        self.open_outlier = open_outlier
        self.group_size = group_size
        self.percdamp = percdamp
        self.pdmix = pdmix
        # 所有校验都置于该函数，基于入参新生成的参数也置于该函数
        check_and_generate_config_param(self)

    def _init_attribute_by_config(self, config):
        """
        基于一个已有的BaseConfig对象，为当前BaseConfig对象赋值
        Args:
            config: BaseConfig的实例化对象
        """
        if not isinstance(config, BaseConfig):
            msmodelslim_logger.error("The current config must be an instance of BaseConfig.")
        config_attribute = config.__dict__
        self.__dict__.update(**config_attribute)

