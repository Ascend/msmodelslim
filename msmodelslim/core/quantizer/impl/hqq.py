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

# Per-Channel HQQ 权重量化器。
# 算法要点：
#   1. 使用 MinMax Observer 计算初始 scale/offset
#   2. 固定 scale，通过半二次分裂迭代搜索最优 offset，最小化量化误差
#   3. 贪心通道更新：每个通道独立保留更优参数
#   4. 收敛早停：相对下降阈值 + 绝对变化阈值，所有通道收敛即提前退出
#   5. 仅支持非对称量化：symmetric=True 时配置校验直接报错（不进入 MinMax 流程）

from typing import Optional

import torch

import msmodelslim.ir as qir
from msmodelslim.ir.api import quantize, dequantize, fake_quantize, calculate_qparam
from msmodelslim.ir.qal import QABCRegistry, QDType, QStorage, QParam, QScope
from msmodelslim.core.observer import MsMinMaxObserver, MinMaxObserverConfig
from msmodelslim.utils.exception import SpecError
from msmodelslim.utils.logging import logger_setter
from ..base import AutoWeightQuantizer, QConfig


SCALE_SEARCH_ITER_NUM = 20
SCALE_SEARCH_CONVERGE_THRESHOLD = 1e-10
SCALE_SEARCH_MIN_SCALE = 1e-5
EXT_SCALE_NAME = "scale"
EXT_OFFSET_NAME = "offset"
HQQ_SHRINK_P = 0.7  # 为了稀疏性，p <= 1
BETA = 1000


def set_ext_scale(q_param: QParam, scale: torch.Tensor) -> QParam:
    q_param.ext[EXT_SCALE_NAME] = scale
    return q_param


def set_ext_offset(q_param: QParam, offset: torch.Tensor) -> QParam:
    q_param.ext[EXT_OFFSET_NAME] = offset
    return q_param


def get_ext_scale(q_param: QParam) -> torch.Tensor:
    return q_param.ext[EXT_SCALE_NAME]


def get_ext_offset(q_param: QParam) -> torch.Tensor:
    return q_param.ext[EXT_OFFSET_NAME]


def hqq_calculate_qparam(
    weight: QStorage,
    q_param: QParam,
    n_iter: int = SCALE_SEARCH_ITER_NUM,
    beta: float = BETA,
    lp_norm: float = HQQ_SHRINK_P,
    converge_threshold: float = SCALE_SEARCH_CONVERGE_THRESHOLD,
    min_scale: float = SCALE_SEARCH_MIN_SCALE,
) -> QParam:
    """
    HQQ (Half-Quadratic Quantization) 量化算法，通过迭代，固定 scale，搜索最优的 offset 来最小化量化误差。

    算法原理：
    1. 获取初始量化参数
    2. 使用量化参数对权重 W 进行伪量化（量化后反量化），得到量化后的权重 W_q 和伪量化后的权重 W_r
    3. 计算量化误差 u = W - W_r
    4. 计算广义软阈值运算符 W_e = shrink(u, beta, p) = sign(u) * relu(|u| - |u| ^ (p-1) / beta)
    5. 计算当前最优的 offset = E[W_q - (W - W_e) / scale]
    6. 比较新旧参数的量化误差，保留更好的参数（逐通道贪心更新）
    7. 重复步骤 2-6 直到收敛或达到最大迭代次数

    Args:
        weight: 输入张量，类型为 QStorage，表示待量化的权重张量
        q_param: 初始量化参数，类型为 QParam，包含初始的 scale 和 offset
        n_iter: 最大迭代轮数（默认 20）
        beta: 正则化系数，控制对量化误差的惩罚强度（默认 1000）
        lp_norm: Lp 范数（HQQ_SHRINK_P），用于控制稀疏性（默认 0.7）
        converge_threshold: 绝对收敛阈值（默认 1e-10）
        min_scale: 相对收敛阈值（默认 1e-5）

    Returns:
        QParam: 优化后的量化参数，包含原 scale 和 最优的 offset
    """

    # 检查权重是否为 2D 张量
    if weight.value.ndim != 2:
        raise SpecError("Weight must be a 2D tensor", action="Please check the weight shape")

    # 初始化最优参数：使用输入的初始参数作为起点
    scale = get_ext_scale(q_param)  # 最优缩放因子（固定不更新）
    best_offset = get_ext_offset(q_param)  # 最优偏移量

    # 使用初始参数进行量化和反量化
    quant_weight = quantize(weight, q_param)  # 量化后的权重
    dequant_weight = fake_quantize(weight, q_param)  # 反量化后的权重

    # 计算初始量化误差：使用 p 范数作为损失函数，并省略开 p 次方根
    best_pnorm = torch.mean(torch.pow(torch.abs((weight.value - dequant_weight.value)), lp_norm), dim=0, keepdim=True)

    # 当前迭代的量化权重，初始化为最优量化权重
    weight_tensor = weight.value

    # 主迭代循环：最多迭代 n_iter 次
    for _ in range(n_iter):
        quant_weight_tensor = quant_weight.value.to(weight_tensor.dtype)
        dequant_weight_tensor = dequant_weight.value.to(weight_tensor.dtype)

        # 计算最优 offset
        error_weight_tensor = weight_tensor - dequant_weight_tensor
        shrink_weight_tensor = torch.sign(error_weight_tensor) * torch.relu(
            torch.abs(error_weight_tensor) - torch.abs(error_weight_tensor) ** (lp_norm - 1) / beta
        )
        next_offset = torch.mean(
            quant_weight_tensor - (weight_tensor - shrink_weight_tensor) / scale, dim=0, keepdim=True
        )  # 计算当前最优的 offset

        # 更新量化参数并重新量化
        next_q_param = set_ext_offset(q_param, next_offset.squeeze())
        quant_weight = quantize(weight, next_q_param)

        # 评估当前参数的量化效果并更新最优参数
        dequant_weight = fake_quantize(weight, next_q_param)  # 使用当前参数进行反量化
        new_pnorm = torch.mean(
            torch.pow(torch.abs((weight.value - dequant_weight.value)), lp_norm), dim=0, keepdim=True
        ).squeeze()  # 计算当前 p 范数

        # 创建掩码：标记哪些通道的误差得到了改善
        mask = (new_pnorm < best_pnorm).to(torch.int32)  # 1 表示改善，0 表示没有改善

        # 贪心更新策略：只保留更好的参数
        # 对于每个通道，如果当前误差更小，则更新为当前参数；否则保持原参数
        best_pnorm_next = best_pnorm * (1 - mask) + new_pnorm * mask  # 更新最优 p 范数

        # 收敛性检查：判断是否达到收敛条件
        # 使用两种判断标准来确保收敛的稳定性

        # 判断 1：相对下降幅度是否足够小
        # 这表示误差的相对改善幅度小于阈值，适用于误差较大的情况
        mask1 = (best_pnorm - best_pnorm_next) / best_pnorm.clamp(min=1e-4) < min_scale

        # 判断 2：绝对变化量是否足够小
        # 这表示误差的绝对变化量小于阈值，适用于误差较小的情况
        mask2 = torch.abs(best_pnorm - best_pnorm_next) < converge_threshold

        # 综合判断：只有当所有通道都满足收敛条件时才提前退出
        # logical_and(logical_not(mask1), logical_not(mask2)) 表示既不满足相对条件也不满足绝对条件
        # 如果所有通道都满足至少一个条件，则 sum 为 0，可以退出循环
        if torch.sum(torch.logical_and(torch.logical_not(mask1), torch.logical_not(mask2))) == 0:
            break  # 提前退出：所有通道都已收敛

        best_pnorm = best_pnorm_next
        # 更新最优量化权重：只更新改善的通道
        best_offset = (best_offset * (1 - mask) + next_offset * mask).squeeze()
        best_q_param = set_ext_offset(q_param, best_offset)
        quant_weight = quantize(weight, best_q_param)
        dequant_weight = fake_quantize(weight, best_q_param)

    # 返回最优的量化参数
    q_param = set_ext_offset(q_param, best_offset)  # 设置最优偏移量
    return q_param


@QABCRegistry.multi_register(
    dispatch_key=[
        (qir.int8_per_channel_asym, "hqq"),  # int8 非对称 per-channel HQQ
    ],
    abc_type=AutoWeightQuantizer,
)
@logger_setter(__name__)
class WeightPerChannelHQQ(AutoWeightQuantizer):
    """
    Per-Channel HQQ 量化器

    特点：
    1. 每个通道使用独立的量化参数（scale 和 offset）
    2. 使用 HQQ 算法优化量化参数，减少量化误差
    3. 仅支持非对称量化（symmetric=True 时配置校验直接报错）
    """

    def __init__(self, config: QConfig):
        super().__init__()
        # 配置 MinMax 观察器，用于计算初始的量化范围
        minmax_config = MinMaxObserverConfig(dim=0, keepdim=False)  # 沿着第一个维度计算 min/max
        self.config = config
        self.minmax_observer = MsMinMaxObserver(minmax_config)
        # 初始化成员变量
        self.weight: Optional[QStorage] = None  # 原始权重
        self.bias: Optional[torch.Tensor] = None  # 偏置项
        self.w_q_param: Optional[QParam] = None  # 量化参数
        self.w_q_storage: Optional[QStorage] = None  # 量化后的权重存储
        self.is_quantized = False  # 标记是否已完成量化

    def validate_ext_config(self):
        """
        配置校验：HQQ 仅支持非对称量化。

        Raises:
            SpecError: 配置为对称量化（symmetric=True）时抛出。
        """
        if self.config.symmetric:
            raise SpecError(
                "HQQ only supports asymmetric quantization, got symmetric=True",
                action="Please set weight.symmetric to False in the yaml config "
                "(HQQ does not support symmetric quantization).",
            )

    def forward(self, x: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播：执行量化或返回反量化结果

        Args:
            x: 输入张量（在此量化器中不使用）

        Returns:
            torch.Tensor: 反量化后的权重张量
        """
        # 检查是否已初始化权重
        if not self.is_quantized and self.weight is None:
            raise SpecError("No weight was set", action="Please call init_weight first")

        # 如果还未量化，执行量化过程
        if not self.is_quantized:
            # 使用 MinMax 观察器计算权重的统计信息
            self.minmax_observer.update(self.weight.T.value)  # 转置后更新，确保正确的维度
            min_val, max_val = self.minmax_observer.get_min_max()

            # 计算初始的量化参数（scale 和 offset），HQQ 仅走非对称路径
            self.w_q_param = calculate_qparam(
                min_val=min_val,
                max_val=max_val,
                q_dtype=QDType(self.config.dtype),
                q_scope=QScope(self.config.scope),
                symmetric=False,
            )

            # HQQ 迭代优化 offset（固定参数：n_iter=20, beta=1000, lp_norm=0.7）
            self.w_q_param = hqq_calculate_qparam(self.weight.T, self.w_q_param)

            # 使用优化后的参数进行量化，并存储结果
            self.w_q_storage = quantize(self.weight.T, self.w_q_param).T

            # 标记为已量化，并释放原始权重内存
            self.is_quantized = True
            del self.weight
            self.weight = None

        # 返回反量化后的权重（用于推理）
        return dequantize(self.w_q_storage.T, self.w_q_param).T.value

    def init_weight(self, weight: QStorage, bias: Optional[torch.Tensor] = None) -> None:
        """
        初始化权重和偏置

        Args:
            weight: 待量化的权重张量
            bias: 偏置项（可选）
        """
        self.weight = weight
        self.bias = bias

    def get_q_storage(self) -> QStorage:
        """
        获取量化后的权重存储

        Returns:
            QStorage: 量化后的权重存储

        Raises:
            SpecError: 如果还未执行量化
        """
        if self.w_q_storage is None:
            _ = self.forward(None)
        return self.w_q_storage

    def get_q_param(self) -> QParam:
        """
        获取量化参数

        Returns:
            QParam: 量化参数（包含 scale 和 offset）

        Raises:
            SpecError: 如果还未执行量化
        """
        if self.w_q_param is None:
            _ = self.forward(None)
        if self.config.symmetric:
            return self.w_q_param
        # 非对称量化：HQQ 内部使用 (q - offset) * scale 反量化（offset 为负零点），
        # 而 vllm 的 npu_weight_quant_batchmatmul 使用 (w + offset) * scale 反量化。
        # 因此导出/部署时需对 offset 取反，使其匹配 vllm 语义（offset 为正零点）。
        # 同时返回 PER_CHANNEL_NEG_OFFSET 专用 scheme（scope=PER_CHANNEL_NEG_OFFSET，描述 offset 取反
        # 语义），使 AutoFakeQuantLinear.create 按 (float_per_tensor_sym, int8_per_channel_asym_neg_offset)
        # 命中 W8A16PerChannelNegOffsetFakeQuantLinear，其 forward 内将取反 offset 再取反
        # 还原，保证框架 (q - offset) * scale 正确重构权重。
        return QParam(
            scheme=qir.int8_per_channel_asym_neg_offset,
            ext={**self.w_q_param.ext, "offset": -self.w_q_param.ext["offset"]},
        )

    def is_data_free(self) -> bool:
        """HQQ 是 data-free 方法"""
        return True

    def support_distributed(self) -> bool:
        return True
