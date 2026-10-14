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
# pylint: disable=unsubscriptable-object

from typing import Optional

import torch
from pydantic import validate_call
from torch import distributed as dist

import msmodelslim.ir as qir
from msmodelslim.ir.api import fake_quantize, quantize, dequantize, calculate_qparam
from msmodelslim.ir.qal import QABCRegistry, QDType, QStorage, QParam, QScope, QScheme
from msmodelslim.core.observer import MsMinMaxObserver, MinMaxObserverConfig
from msmodelslim.utils.distributed.task_scheduler.sync import broadcast_tensor_process_group_safe
from msmodelslim.utils.distributed.task_scheduler.types import TaskExecutionRecord, TaskSyncContext
from msmodelslim.utils.exception import SpecError
from msmodelslim.utils.logging import logger_setter
from ..base import AutoWeightQuantizer, QConfig

SCALE_SEARCH_ITER_NUM = 50
SCALE_SEARCH_CONVERGE_THRESHOLD = 1e-10
SCALE_SEARCH_MIN_SCALE = 1e-30
EXT_SCALE_NAME = "scale"
EXT_OFFSET_NAME = "offset"


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


def ssz_calculate_qparam(
    weight: QStorage,
    q_param: QParam,
    config: Optional[QConfig] = None,
) -> QParam:
    """
    SSZ (Scan-Scale-Zero) 量化算法，通过迭代搜索最优的 scale 和 offset 来最小化量化误差

    算法原理：
    1. 使用初始的量化参数进行量化
    2. 通过最小二乘法计算当前最优的 scale 和 offset
    3. 比较新旧参数的量化误差，保留更好的参数
    4. 重复步骤2-3直到收敛或达到最大迭代次数

    Args:
        weight: 输入张量，类型为QStorage，表示待量化的权重张量
        q_param: 初始量化参数，类型为QParam，包含初始的 scale 和 offset

    Returns:
        QParam: 优化后的量化参数，包含最优的 scale 和 offset
    """

    # 检查权重是否为2D张量
    if weight.value.ndim != 2:
        raise SpecError("Weight must be a 2D tensor", action="Please check the weight shape")

    # 初始化最优参数：使用输入的初始参数作为起点
    best_scale = get_ext_scale(q_param)  # 最优缩放因子
    best_offset = get_ext_offset(q_param)  # 最优偏移量

    # 使用初始参数进行量化和反量化
    best_quant_weight = quantize(weight, q_param)  # 量化后的权重
    best_dequant_weight = fake_quantize(weight, q_param)  # 反量化后的权重（用于计算误差）

    # 计算初始量化误差：使用均方误差(MSE)作为损失函数
    # dim=0表示沿着第一个维度计算，keepdim=True保持维度
    best_mse = torch.mean(torch.pow(torch.abs((weight.value - best_dequant_weight.value)), 2), dim=0, keepdim=True)

    # 当前迭代的量化权重，初始化为最优量化权重
    quant_weight = best_quant_weight
    # 确定迭代次数：优先使用 config.ext 中的 step 值
    iter_num = SCALE_SEARCH_ITER_NUM  # 默认迭代次数
    if config is not None and hasattr(config, 'ext') and config.ext is not None:
        step = config.ext.get('step', 0)
        if isinstance(step, int) and step > 0:
            iter_num = step

    # 主迭代循环：最多迭代50次
    for _ in range(iter_num):
        # 根据量化方案选择不同的参数更新策略
        if q_param.scheme.symmetric:
            # 对称量化：offset固定为0，只优化scale
            quant_weight_tensor = quant_weight.value.to(weight.value.dtype)  # 转换为浮点类型用于计算

            # 使用最小二乘法计算最优scale
            current_scale = torch.sum(weight.value * quant_weight_tensor, dim=0, keepdim=True) / torch.sum(
                quant_weight_tensor * quant_weight_tensor, dim=0, keepdim=True
            ).clamp(min=SCALE_SEARCH_MIN_SCALE)

            # 更新量化参数并重新量化
            q_param = set_ext_scale(q_param, current_scale.squeeze())  # 移除多余的维度
            quant_weight = quantize(weight, q_param)  # 使用新scale重新量化
            current_offset = get_ext_offset(q_param)  # 对称量化中offset保持为0

        else:
            # 非对称量化：同时优化scale和offset
            quant_weight_tensor = quant_weight.value.to(weight.value.dtype)

            # 计算去除offset后的量化权重
            quant_weight_minus_offset = quant_weight_tensor - q_param.ext[EXT_OFFSET_NAME]

            # 使用最小二乘法计算最优scale
            current_scale = torch.sum(weight.value * quant_weight_minus_offset, dim=0, keepdim=True) / torch.sum(
                quant_weight_minus_offset * quant_weight_minus_offset, dim=0, keepdim=True
            ).clamp(min=SCALE_SEARCH_MIN_SCALE)  # 防止除零

            # 计算最优offset
            current_offset = torch.sum(quant_weight_tensor * current_scale - weight.value, dim=0, keepdim=True) / (
                weight.value.shape[0] * current_scale
            )

            # 更新量化参数并重新量化
            q_param = set_ext_scale(q_param, current_scale.squeeze())
            q_param = set_ext_offset(q_param, current_offset.squeeze())
            quant_weight = quantize(weight, q_param)

        # 评估当前参数的量化效果并更新最优参数
        current_dequant_weight = fake_quantize(weight, q_param)  # 使用当前参数进行反量化
        current_mse = torch.mean(
            torch.pow(torch.abs((weight.value - current_dequant_weight.value)), 2), dim=0, keepdim=True
        ).squeeze()  # 计算当前MSE

        # 收敛性检查：判断是否达到收敛条件
        # 使用两种判断标准来确保收敛的稳定性

        # 判断1：相对下降幅度是否足够小
        # 这表示误差的相对改善幅度小于阈值，适用于误差较大的情况
        mask1 = (best_mse - current_mse) / best_mse.clamp(min=1e-4) < SCALE_SEARCH_CONVERGE_THRESHOLD

        # 判断2：绝对变化量是否足够小
        # 这表示误差的绝对变化量小于阈值，适用于误差较小的情况
        mask2 = torch.abs(best_mse - current_mse) < SCALE_SEARCH_CONVERGE_THRESHOLD

        # 综合判断：只有当所有通道都满足收敛条件时才提前退出
        # logical_and(torch.logical_not(mask1), torch.logical_not(mask2)) 表示既不满足相对条件也不满足绝对条件
        # 如果所有通道都满足至少一个条件，则sum为0，可以退出循环
        # 这种双重判断机制可以覆盖不同量级的损失场景，提供更稳定的收敛判断
        if torch.sum(torch.logical_and(torch.logical_not(mask1), torch.logical_not(mask2))) == 0:
            break  # 提前退出：所有通道都已收敛

        # 创建掩码：标记哪些通道的误差得到了改善
        mask = (current_mse < best_mse).to(torch.int32)  # 1表示改善，0表示没有改善

        # 贪心更新策略：只保留更好的参数
        # 对于每个通道，如果当前误差更小，则更新为当前参数；否则保持原参数
        best_mse = best_mse * (1 - mask) + current_mse * mask  # 更新最优MSE
        best_scale = (best_scale * (1 - mask) + current_scale * mask).squeeze()  # 更新最优scale

        if q_param.scheme.symmetric:
            # 对称量化：offset固定为0
            best_offset = get_ext_offset(q_param)
        else:
            # 非对称量化：更新最优offset
            best_offset = (best_offset * (1 - mask) + current_offset * mask).squeeze()

        # 更新最优量化权重：只更新改善的通道
        best_quant_weight.value = best_quant_weight.value * (1 - mask) + quant_weight.value * mask

    # 返回最优的量化参数
    q_param = set_ext_scale(q_param, best_scale)  # 设置最优缩放因子
    q_param = set_ext_offset(q_param, best_offset)  # 设置最优偏移量
    return q_param


@QABCRegistry.multi_register(
    dispatch_key=[
        (qir.int8_per_channel_sym, "ssz"),  # 注册为int8对称per-channel SSZ量化器
        (qir.int4_per_channel_sym, "ssz"),  # 注册int4对称per-channel SSZ量化器
    ],
    abc_type=AutoWeightQuantizer,
)
@logger_setter(__name__)
class WeightPerChannelSsz(AutoWeightQuantizer):
    """
    Per-Channel SSZ量化器

    特点：
    1. 每个通道使用独立的量化参数（scale和offset）
    2. 使用SSZ算法优化量化参数，减少量化误差
    3. 支持对称和非对称量化
    """

    def __init__(self, config: QConfig):
        super().__init__()
        # 配置MinMax观察器，用于计算初始的量化范围
        minmax_config = MinMaxObserverConfig(dim=0, keepdim=False)  # 沿着第一个维度计算min/max
        self.config = config
        self.minmax_observer = MsMinMaxObserver(minmax_config)

        # 初始化成员变量
        self.weight: Optional[QStorage] = None  # 原始权重
        self.bias: Optional[torch.Tensor] = None  # 偏置项
        self.w_q_param: Optional[QParam] = None  # 量化参数
        self.w_q_storage: Optional[QStorage] = None  # 量化后的权重存储
        self.is_quantized = False  # 标记是否已完成量化

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
            # 使用MinMax观察器计算权重的统计信息
            self.minmax_observer.update(self.weight.T.value)  # 转置后更新，确保正确的维度
            min_val, max_val = self.minmax_observer.get_min_max()

            # 计算初始的量化参数（scale和offset）
            self.w_q_param = calculate_qparam(
                min_val=min_val,
                max_val=max_val,
                q_dtype=QDType(self.config.dtype),
                q_scope=QScope(self.config.scope),
                symmetric=self.config.symmetric,
            )

            # 使用SSZ算法优化量化参数，减少量化误差
            self.w_q_param = ssz_calculate_qparam(self.weight.T, self.w_q_param, config=self.config)

            # 使用优化后的参数进行量化，并存储结果
            self.w_q_storage = quantize(self.weight.T, self.w_q_param).T

            # 标记为已量化，并释放原始权重内存
            self.is_quantized = True
            del self.weight
            self.weight = None

        # 返回反量化后的权重（用于推理）
        return dequantize(self.w_q_storage.T, self.w_q_param).T.value

    @validate_call(config=dict(arbitrary_types_allowed=True))
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
            QParam: 量化参数（包含scale和offset）

        Raises:
            SpecError: 如果还未执行量化
        """
        if self.w_q_param is None:
            _ = self.forward(None)
        return self.w_q_param

    # ── DTSMixin: distributed_sync ──────────────────────────────────────

    def distributed_sync(self, record: TaskExecutionRecord, sync_ctx: TaskSyncContext) -> None:
        """分布式同步：owner rank broadcast 量化状态到所有 rank。"""
        if not dist.is_initialized() or sync_ctx.world_size <= 1:
            return
        _broadcast_quantizer_state(self, record.executor_rank)

    def load_quantized_from_broadcast_tensors(
        self,
        scale: torch.Tensor,
        offset: torch.Tensor,
        w_q_storage_value: torch.Tensor,
        q_storage_dtype: QDType,
    ) -> None:
        """在非 owner rank 上重建量化状态（与 forward 首次量化后一致）。"""
        scheme = QScheme(
            scope=QScope(self.config.scope),
            dtype=QDType(self.config.dtype),
            symmetric=self.config.symmetric,
        )
        q_param = QParam(scheme=scheme, ext={})
        q_param = set_ext_scale(q_param, scale)
        q_param = set_ext_offset(q_param, offset)
        self.w_q_param = q_param
        self.w_q_storage = QStorage(dtype=q_storage_dtype, value=w_q_storage_value)
        self.is_quantized = True
        if self.weight is not None:
            del self.weight
            self.weight = None


def _broadcast_quantizer_state(wq, owner_rank: int) -> None:
    """
    Broadcast 权重量化器状态（scale、offset、量化权重）从 owner rank 到所有 rank。
    使用 broadcast_object_list 传元数据，broadcast_tensor_process_group_safe 传张量。
    """
    rank = dist.get_rank()

    meta = [None]
    if rank == owner_rank:
        if not wq.is_quantized:
            raise SpecError(
                "Cannot broadcast quantizer state: weight quantizer has not been initialized. "
                "Call forward() first to trigger quantization."
            )
        meta[0] = {
            "scale_shape": list(get_ext_scale(wq.w_q_param).shape),
            "scale_dtype": str(get_ext_scale(wq.w_q_param).dtype),
            "offset_shape": list(get_ext_offset(wq.w_q_param).shape),
            "offset_dtype": str(get_ext_offset(wq.w_q_param).dtype),
            "storage_shape": list(wq.w_q_storage.value.shape),
            "storage_dtype": str(wq.w_q_storage.value.dtype),
            "storage_qdtype": wq.w_q_storage.dtype.name,
        }
    dist.broadcast_object_list(meta, src=owner_rank)

    if rank == owner_rank:
        scale = get_ext_scale(wq.w_q_param).contiguous()
        offset = get_ext_offset(wq.w_q_param).contiguous()
        storage = wq.w_q_storage.value.contiguous()
    else:
        m = meta[0]
        scale = torch.empty(m["scale_shape"], dtype=getattr(torch, m["scale_dtype"].split(".")[-1]))
        offset = torch.empty(m["offset_shape"], dtype=getattr(torch, m["offset_dtype"].split(".")[-1]))
        storage = torch.empty(m["storage_shape"], dtype=getattr(torch, m["storage_dtype"].split(".")[-1]))

    broadcast_tensor_process_group_safe(scale, src=owner_rank)
    broadcast_tensor_process_group_safe(offset, src=owner_rank)
    broadcast_tensor_process_group_safe(storage, src=owner_rank)

    if rank != owner_rank:
        qdt = getattr(QDType, m["storage_qdtype"], QDType.INT8)
        wq.load_quantized_from_broadcast_tensors(scale, offset, storage, qdt)
