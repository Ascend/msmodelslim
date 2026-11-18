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

from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.quant_config.quant_config_factory import QuantConfigFactory


class QuantConfig:
    """
    初始化时只完成少数核心参数的初始化
    如果要做稀疏量化、权重量化、kvcache量化、动态量化，调用相关接口完成。

    quant_config = QuantConfig()

    quant_config = QuantConfig().weight_quant()
    quant_config = QuantConfig().weight_quant().kv_quant()
    quant_config = QuantConfig().sparse_quant()

    quant_config = QuantConfig(
        w_bit=4,
        disable_names=disable_names,
        dev_type='npu',
        dev_id=0,
        act_method=3,
        pr=2.0,
        fraction=0.011,
        nonuniform=False,
        mm_tensor=False,
        co_sparse=True
        )

    quant_config = QuantConfig(w_bit=4,disable_names=disable_names,dev_type='npu',dev_id=0).
                    sparse_quant()
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
            percdamp: GPTQ所使用的矩阵正定偏置系数,当GPTQ运行出现非正定矩阵导致的报错时可以增大该参数
        """
        self._cur_config = QuantConfigFactory. \
            get_quant_config('base', w_bit=w_bit, a_bit=a_bit, act_method=act_method, w_method=w_method,
                             disable_names=disable_names, pr=pr, mm_tensor=mm_tensor, dev_type=dev_type, dev_id=dev_id,
                             fraction=fraction, nonuniform=nonuniform, co_sparse=co_sparse, w_sym=w_sym,
                             is_lowbit=is_lowbit, do_smooth=do_smooth, use_sigma=use_sigma,
                             sigma_factor=sigma_factor, disable_last_linear=disable_last_linear,
                             use_kvcache_quant=use_kvcache_quant, open_outlier=open_outlier, is_dynamic=is_dynamic,
                             group_size=group_size, percdamp=percdamp, pdmix=pdmix)
        self._modify_quant_param()

    def weight_quant(self,
                     w_method: str = 'MinMax',
                     mm_tensor: bool = True,
                     w_sym: bool = True,
                     group_size: int = 64,
                     block_size: int = 64,
                     ):
        """
        权重量化的参数初始化，即 W8A16 或 W4A16

        Args:
            w_method: weight量化的方式
            mm_tensor: matmul的tensor（即weight）是per-tensor还是per-channel
            w_sym: 权重是否对称量化
            group_size: per-group场景下group的大小，通常取64或128,
            block_size: NF4场景下每个block的大小，通常取64或128,

        """
        # 权重量化的参数, 所有量化Config的父类，存放通用参数
        self._cur_config = QuantConfigFactory.get_quant_config('weight', last_config=self._cur_config,
                                                               w_method=w_method, mm_tensor=mm_tensor,
                                                               w_sym=w_sym, group_size=group_size,
                                                               block_size=block_size)
        self._modify_quant_param()
        return self

    def weight_activation_quant(self,
                                act_method: int = 1,
                                pr: float = 1.0,
                                is_dynamic: bool = False):
        """
        权重量化+激活量化的参数初始化, 即 W8A8

        Args:
            act_method: activation量化的方式
            pr: 量化的一个概率参数，非1时量化生成的参数带有随机性
            is_dynamic: 是否使用动态量化，即w8a8中的activation动态生成
        """
        self._cur_config = QuantConfigFactory.get_quant_config('weight_activation', last_config=self._cur_config,
                                                               act_method=act_method, pr=pr, is_dynamic=is_dynamic)
        self._modify_quant_param()
        return self

    def sparse_quant(self,
                     act_method: int = 1,
                     fraction: float = 0.01,
                     nonuniform: bool = False,
                     is_lowbit: bool = False,
                     do_smooth: bool = False,
                     use_sigma: bool = False,
                     sigma_factor: float = 3.0):
        """
        稀疏量化的参数初始化

        Args:
            act_method: activation量化的方式
            fraction: 稀疏量化的调优参数
            nonuniform: 稀疏量化中默认为非均匀量化，决定权重是否均匀量化。建议采用False
            is_lowbit: 是否使用稀疏量化的low bit算法
            do_smooth: 稀疏量化low bit中是否对activation使用smooth算法
            use_sigma: 稀疏量化low bit的outlier选取范围，False时使用fraction，True使用sigma_factor
            sigma_factor:  稀疏量化low bit的调优参数
        """
        self._cur_config = QuantConfigFactory.get_quant_config('sparse', last_config=self._cur_config,
                                                               act_method=act_method, fraction=fraction,
                                                               nonuniform=nonuniform, is_lowbit=is_lowbit,
                                                               do_smooth=do_smooth, use_sigma=use_sigma,
                                                               sigma_factor=sigma_factor)
        self._modify_quant_param()
        return self

    def kv_quant(self, kv_sym: bool = True):
        """
        kv cache 量化的参数初始化，
        无需输入参数，调用本函数即可，会自动将 use_kvcache_quant 置为True
        """
        self._cur_config = QuantConfigFactory.get_quant_config('kv', last_config=self._cur_config, kv_sym=kv_sym)
        self._modify_quant_param()
        return self

    def fa_quant(self,
                 fa_amp: int = 0
                 ):
        """
        FA量化的参数初始化
        调用fa_quant()，默认自动打开

        Arg:
            fa_amp: 自动回退层数，以整个attention为单位进行回退
        """
        self._cur_config = QuantConfigFactory.get_quant_config('fa_quant',
                                                               last_config=self._cur_config,
                                                               fa_amp=fa_amp
                                                               )
        self._modify_quant_param()
        return self

    def timestep_quant(self,
                       max_dynamic_step: int = None,
                       ):
        """
        时间步量化的初始化
        """
        self._cur_config = QuantConfigFactory.get_quant_config('timestep_quant',
                                                               last_config=self._cur_config,
                                                               max_dynamic_step=max_dynamic_step,
                                                               )
        self._modify_quant_param()
        return self

    def simulate_tp(self,
                    tp_size,
                    enable_communication_quant=True,
                    enable_per_device_quant=True,
                    ):
        """
        多卡量化模拟的参数初始化

        Arg:
            tp_size: 模拟多卡量化的卡数
            allreduce_quant: 是否启用模拟通信量化
            quant_per_tp: 模拟多卡通信量化，每张卡是否使用同一个scale
        """
        self._cur_config = QuantConfigFactory.get_quant_config('simulate_tp',
                                                               last_config=self._cur_config,
                                                               tp_size=tp_size,
                                                               enable_communication_quant=enable_communication_quant,
                                                               enable_per_device_quant=enable_per_device_quant)
        self._modify_quant_param()
        return self


    def _modify_quant_param(self):
        """
        用于把当前的量化BaseConfig对象 _cur_config 的所有参数赋值给当前的QuantConfig类，
        兼容Calibrator中对QuantConfig的使用
        """
        config_attribute = self._cur_config.__dict__
        self.__dict__.update(**config_attribute)
