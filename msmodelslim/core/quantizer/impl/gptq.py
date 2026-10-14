#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
-------------------------------------------------------------------------
This file is part of the MindStudio project.
Copyright (c) 2026 Huawei Technologies Co.,Ltd.

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
# pylint: disable=redefined-builtin

import math
from typing import Optional

import torch

import msmodelslim.ir as qir
from msmodelslim.core.observer import MsMinMaxObserver, MinMaxObserverConfig
from msmodelslim.ir.api import quantize, fake_quantize, calculate_qparam
from msmodelslim.ir.qal import QABCRegistry, QDType, QStorage, QParam, QScope, QScheme
from msmodelslim.utils.exception import SpecError
from msmodelslim.utils.logging import logger_setter
from msmodelslim.utils.validation.value import greater_than_zero, int_greater_than_zero
from ..base import AutoWeightQuantizer, QConfig

PERCDAMP_KEY = "percdamp"  # percdamp值配置key
PERCDAMP_DEFAULT = 0.01  # 默认percdamp值
BLOCK_SIZE_KEY = "block_size"  # blockSize值配置key
BLOCK_SIZE_DEFAULT = 128  # 默认blockSize值
GROUP_SIZE_KEY = "group_size"  # groupSize值配置key
GROUP_SIZE_DEFAULT = 128  # 默认groupSize值


def get_ext_value(config: Optional[QConfig], attrName: str, default):
    value = default
    if config is not None and config.ext is not None:
        temp = config.ext.get(attrName, default)
        if temp is not None:
            value = temp
    return value


def add_batch(hessian: torch.Tensor, nsamples: float, input: Optional[torch.Tensor]):
    tmp_hessian = hessian
    tmp_nsamples = nsamples
    if hessian is None:
        columns = input.shape[-1]
        tmp_hessian = torch.zeros((columns, columns), device=input.device)
    if len(input.shape) == 3:
        input = input.reshape((-1, input.shape[-1]))
    input = input.float()
    tmp = input.shape[0]
    tmp_hessian *= tmp_nsamples / (tmp_nsamples + tmp)
    tmp_nsamples += tmp
    input = math.sqrt(2 / tmp_nsamples) * input.float()
    tmp_hessian += input.t().matmul(input)
    return tmp_hessian, tmp_nsamples


def calculate_hessian_inv(hessian: torch.Tensor, percdamp: float, weight_tensor: torch.Tensor) -> torch.Tensor:
    dead = torch.diag(hessian) == 0
    hessian[dead, dead] = 1
    weight_tensor[:, dead] = 0
    columns = weight_tensor.shape[1]

    # 添加阻尼
    damp = percdamp * torch.mean(torch.diag(hessian))
    diag = torch.arange(columns, device=weight_tensor.device)
    hessian[diag, diag] += damp

    # cholesky分解重构
    need_recovering = False
    if "cpu" not in str(hessian.device):
        # cholesky分解重构不支持NPU运算，需要将数据搬迁到CPU中执行
        hessian = hessian.to("cpu")
        need_recovering = True
    hessian = torch.linalg.cholesky(hessian)
    hessian_inv = torch.cholesky_inverse(hessian)
    hessian_inv = torch.linalg.cholesky(hessian_inv, upper=True).to(weight_tensor.dtype)
    if need_recovering:
        hessian_inv = hessian_inv.to(weight_tensor.device)
    return hessian_inv


def _validate_ext_config(config: QConfig):
    """
    GPTQ参数配置校验
    """
    if config.ext is None:
        return
    if "percdamp" in config.ext:
        greater_than_zero(config.ext["percdamp"], "percdamp")
    if "block_size" in config.ext:
        int_greater_than_zero(config.ext["block_size"], "block_size")
    if "group_size" in config.ext:
        int_greater_than_zero(config.ext["group_size"], "group_size")


@QABCRegistry.multi_register(
    dispatch_key=[
        (qir.int8_per_channel_sym, "gptq"),
        (qir.int8_per_channel_asym, "gptq"),
        (qir.int4_per_channel_sym, "gptq"),
        (qir.int4_per_channel_asym, "gptq"),
    ],
    abc_type=AutoWeightQuantizer,
)
@logger_setter()
class WeightPerChannelGPTQ(AutoWeightQuantizer):
    """
    Per-Channel GPTQ量化器

    特点：
    1. 每个通道使用独立的量化参数（scale和offset）
    2. 使用GPTQ算法优化权重，减少量化误差
    3. 支持对称和非对称量化
    """

    def __init__(self, config: QConfig):
        super().__init__()
        self.config = config
        minmax_config = MinMaxObserverConfig(dim=0, keepdim=False)
        self.minmax_observer = MsMinMaxObserver(minmax_config)
        self.weight: Optional[QStorage] = None
        self.bias: Optional[torch.Tensor] = None
        self.w_q_param: Optional[QParam] = None
        self.w_q_storage: Optional[QStorage] = None
        self.hessian: torch.Tensor = None
        self.nsamples: float = 0
        self.percdamp = get_ext_value(self.config, PERCDAMP_KEY, PERCDAMP_DEFAULT)
        self.block_size = get_ext_value(self.config, BLOCK_SIZE_KEY, BLOCK_SIZE_DEFAULT)

    def forward(self, x: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播：执行量化或返回反量化结果

        Args:
            x: 输入张量

        Returns:
            torch.Tensor: 反量化后的权重张量
        """
        # 收集并合并激活值到hessian矩阵中
        if x is not None and x.numel() != 0:
            self.hessian, self.nsamples = add_batch(self.hessian, self.nsamples, x.clone())
        return self.weight.value

    def is_data_free(self) -> bool:
        """
        GPTQ权重算法需要使用激活值计算hessian矩阵，因此需要返回False
        """
        return False

    def support_distributed(self) -> bool:
        """
        GPTQ权重算法不支持DP分布式量化
        """
        return False

    def validate_ext_config(self):
        """
        GPTQ参数配置校验
        """
        _validate_ext_config(self.config)

    def init_weight(self, weight: QStorage, bias: Optional[torch.Tensor] = None) -> None:
        self.weight = weight
        self.bias = bias

    def get_q_storage(self) -> QStorage:
        if self.hessian is None:
            raise SpecError("No hessian was set", action="Please call forward first")
        if self.w_q_storage is None:
            self.w_q_storage, self.w_q_param = self.__gptq_per_channel_quantize()
        return self.w_q_storage

    def get_q_param(self) -> QParam:
        if self.hessian is None:
            raise SpecError("No hessian was set", action="Please call forward first")
        if self.w_q_param is None:
            self.w_q_storage, self.w_q_param = self.__gptq_per_channel_quantize()
        return self.w_q_param

    def __gptq_per_channel_quantize(self):
        """GPTQ量化单个线性层"""
        weight_tensor = self.weight.value.clone()
        # 计算w_q_param
        self.w_q_param = self.__calculate_w_q_param()
        # 对海森矩阵进行分解重构
        Hinv = calculate_hessian_inv(self.hessian, self.percdamp, weight_tensor)

        # 分块处理
        columns = weight_tensor.shape[1]
        # w_q_param = None
        for start_idx in range(0, columns, self.block_size):
            end_idx = min(start_idx + self.block_size, columns)
            count = end_idx - start_idx

            # 处理当前组
            block_weight = weight_tensor[:, start_idx:end_idx].clone()
            block_Hinv = Hinv[start_idx:end_idx, start_idx:end_idx]

            # 初始化量化误差
            block_quant_error = torch.zeros_like(block_weight)
            block_weight_dequant = torch.zeros_like(block_weight)

            # 量化当前组
            for idx in range(count):
                # 计算权重更新
                channel_weight = block_weight[:, idx].clone().unsqueeze(1)
                # 进行伪量化
                channel_weight_dequant = fake_quantize(self.weight.same_like(channel_weight.t()), self.w_q_param).T
                block_weight_dequant[:, idx] = channel_weight_dequant.value.flatten()

                # 计算量化误差
                channel_quant_error = (channel_weight - channel_weight_dequant.value) / block_Hinv[idx, idx]
                block_weight[:, idx:] -= channel_quant_error.matmul(block_Hinv[idx, idx:].unsqueeze(0))
                block_quant_error[:, idx] = channel_quant_error.flatten()

            # 更新权重数据
            weight_tensor[:, start_idx:end_idx] = block_weight_dequant
            weight_tensor[:, end_idx:] -= block_quant_error.matmul(Hinv[start_idx:end_idx, end_idx:])

        tmp_weight_tensor = self.weight.same_like(weight_tensor)
        w_q_storage = quantize(tmp_weight_tensor.T, self.w_q_param).T
        return w_q_storage, self.w_q_param

    def __calculate_w_q_param(self) -> QParam:
        # 使用MinMax观察器计算权重的统计信息
        self.minmax_observer.update(self.weight.T.value)
        # 转置后更新，确保正确的维度
        min_val, max_val = self.minmax_observer.get_min_max()

        # 计算初始的量化参数（scale和offset）
        return calculate_qparam(
            min_val=min_val,
            max_val=max_val,
            q_dtype=QDType(self.config.dtype),
            q_scope=QScope(self.config.scope),
            symmetric=self.config.symmetric,
        )


@QABCRegistry.multi_register(
    dispatch_key=[
        (qir.int8_per_group_sym, "gptq"),
        (qir.int8_per_group_asym, "gptq"),
        (qir.int4_per_group_sym, "gptq"),
        (qir.int4_per_group_asym, "gptq"),
    ],
    abc_type=AutoWeightQuantizer,
)
@logger_setter()
class WeightPerGroupGPTQ(AutoWeightQuantizer):
    """
    Per-Group GPTQ量化器

    特点：
    1. 每个分组使用独立的量化参数（scale和offset）
    2. 使用GPTQ算法优化权重，减少量化误差
    3. 支持对称和非对称量化
    """

    def __init__(self, config: QConfig):
        super().__init__()
        self.config = config
        minmax_config = MinMaxObserverConfig(dim=0, keepdim=False)
        self.minmax_observer = MsMinMaxObserver(minmax_config)
        self.weight: Optional[QStorage] = None
        self.bias: Optional[torch.Tensor] = None
        self.w_q_param: Optional[QParam] = None
        self.w_q_storage: Optional[QStorage] = None
        self.hessian: torch.Tensor = None
        self.nsamples: float = 0
        self.percdamp = get_ext_value(self.config, PERCDAMP_KEY, PERCDAMP_DEFAULT)
        self.block_size = get_ext_value(self.config, BLOCK_SIZE_KEY, BLOCK_SIZE_DEFAULT)
        self.group_size = get_ext_value(self.config, GROUP_SIZE_KEY, GROUP_SIZE_DEFAULT)

    def forward(self, x: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播：执行量化或返回反量化结果

        Args:
            x: 输入张量

        Returns:
            torch.Tensor: 反量化后的权重张量
        """
        # 收集并合并激活值到hessian矩阵中
        if x is not None and x.numel() != 0:
            self.hessian, self.nsamples = add_batch(self.hessian, self.nsamples, x.clone())
        return self.weight.value

    def is_data_free(self) -> bool:
        """
        GPTQ权重算法需要使用激活值计算hessian矩阵，因此需要返回False
        """
        return False

    def support_distributed(self) -> bool:
        """
        GPTQ权重算法不支持DP分布式量化
        """
        return False

    def validate_ext_config(self):
        """
        GPTQ参数配置校验
        """
        _validate_ext_config(self.config)

    def init_weight(self, weight: QStorage, bias: Optional[torch.Tensor] = None) -> None:
        if weight.value.shape[1] % self.group_size != 0:
            raise SpecError(
                "Input channels must be divisible by group_size in per-group GPTQ",
                action=f"Please set group_size to divide input channels ({weight.value.shape[1]})",
            )
        self.weight = weight
        self.bias = bias

    def get_q_storage(self) -> QStorage:
        if self.hessian is None:
            raise SpecError("No hessian was set", action="Please call forward first")
        if self.w_q_storage is None:
            self.w_q_storage, self.w_q_param = self.__gptq_per_group_quantize()
        return self.w_q_storage

    def get_q_param(self) -> QParam:
        if self.hessian is None:
            raise SpecError("No hessian was set", action="Please call forward first")
        if self.w_q_param is None:
            self.w_q_storage, self.w_q_param = self.__gptq_per_group_quantize()
        return self.w_q_param

    def __gptq_per_group_quantize(self):
        """GPTQ量化单个线性层"""
        weight_tensor = self.weight.value.clone()
        # 初始化w_q_param
        self.w_q_param = self.__init_w_q_param()
        # 对海森矩阵进行分解重构
        Hinv = calculate_hessian_inv(self.hessian, self.percdamp, weight_tensor)

        # 分块处理
        columns = weight_tensor.shape[1]
        for start_idx in range(0, columns, self.block_size):
            end_idx = min(start_idx + self.block_size, columns)
            count = end_idx - start_idx

            # 处理当前组
            block_weight = weight_tensor[:, start_idx:end_idx].clone()
            block_Hinv = Hinv[start_idx:end_idx, start_idx:end_idx]

            # 初始化量化误差
            block_quant_error = torch.zeros_like(block_weight)
            block_weight_dequant = torch.zeros_like(block_weight)

            # 量化当前组
            for idx in range(count):
                # 计算权重更新
                channel_weight = block_weight[:, idx].clone().unsqueeze(1)
                # 计算分组权重参数
                channel_w_q_param = self.__calculate_group_q_param(weight_tensor, start_idx + idx)
                # 进行伪量化
                channel_weight_dequant = fake_quantize(self.weight.same_like(channel_weight.t()), channel_w_q_param).T
                block_weight_dequant[:, idx] = channel_weight_dequant.value.flatten()

                # 计算量化误差
                channel_quant_error = (channel_weight - channel_weight_dequant.value) / block_Hinv[idx, idx]
                block_weight[:, idx:] -= channel_quant_error.matmul(block_Hinv[idx, idx:].unsqueeze(0))
                block_quant_error[:, idx] = channel_quant_error.flatten()

            # 更新权重数据
            weight_tensor[:, start_idx:end_idx] = block_weight_dequant
            weight_tensor[:, end_idx:] -= block_quant_error.matmul(Hinv[start_idx:end_idx, end_idx:])

        tmp_weight_tensor = self.weight.same_like(weight_tensor)
        w_q_storage = quantize(tmp_weight_tensor.T, self.w_q_param).T
        return w_q_storage, self.w_q_param

    def __init_w_q_param(self) -> QParam:
        weight_shape = self.weight.value.shape
        return QParam(
            scheme=QScheme(
                dtype=self.config.dtype,
                scope=self.config.scope,
                symmetric=self.config.symmetric,
            ),
            ext={
                "scale": torch.zeros(
                    weight_shape[0], weight_shape[1] // self.group_size, device=self.weight.value.device
                ),
                "offset": torch.zeros(
                    weight_shape[0], weight_shape[1] // self.group_size, device=self.weight.value.device
                ),
                "group_size": self.group_size,
            },
        )

    def __calculate_group_q_param(self, weight_tensor: torch.Tensor, current_idx: int) -> QParam:
        if current_idx % self.group_size == 0:
            start_idx = (current_idx // self.group_size) * self.group_size
            end_idx = min(start_idx + self.group_size, weight_tensor.shape[1])
            group_weight_tensor = weight_tensor[:, start_idx:end_idx].clone()
            # 使用MinMax观察器计算权重的统计信息
            self.minmax_observer.reset()
            self.minmax_observer.update(group_weight_tensor.t())
            # 转置后更新，确保正确的维度
            min_val, max_val = self.minmax_observer.get_min_max()

            # 计算初始的量化参数（scale和offset）,对权重分组后，分组内部采用PER_CHANNEL进行量化
            group_q_param = calculate_qparam(
                min_val=min_val,
                max_val=max_val,
                q_dtype=QDType(self.config.dtype),
                q_scope=QScope.PER_CHANNEL,
                symmetric=self.config.symmetric,
            )
            # 更新局部变量中的scale和offset值
            group_idx = current_idx // self.group_size
            self.w_q_param.ext["scale"][:, group_idx] = group_q_param.ext["scale"]
            self.w_q_param.ext["offset"][:, group_idx] = group_q_param.ext["offset"]
            return group_q_param
        else:
            group_idx = current_idx // self.group_size
            # 对权重分组后，分组内部采用PER_CHANNEL进行量化
            return QParam(
                scheme=QScheme(
                    dtype=self.w_q_param.scheme.dtype,
                    scope=QScope.PER_CHANNEL,
                    symmetric=self.w_q_param.scheme.symmetric,
                ),
                ext={
                    "scale": self.w_q_param.ext["scale"][:, group_idx],
                    "offset": self.w_q_param.ext["offset"][:, group_idx],
                },
            )
