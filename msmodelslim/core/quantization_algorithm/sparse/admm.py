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
# pylint: disable=consider-using-from-import

import torch
import torch.nn as nn

import numpy as np


KEEP_BITS = 2
KEEP_PROPORTION = 0.02
PERCDAMP = 0.1
ITERATIVE_PRUNE = 15
ITERS = 20


def quantize_l2(keep_mask: torch.Tensor, para: torch.Tensor, threshold: int = 2, rtn: bool = True) -> torch.Tensor:
    """
    L2量化函数：对参数进行量化，同时保持指定位置的精度

    Args:
        keep_mask: 需要保持精度的位置掩码
        para: 待量化的参数张量
        threshold: 指数阈值，控制量化精度
        rtn: 是否使用四舍五入(True)或截断(False)

    Returns:
        量化后的参数张量
    """
    para = para.to(dtype=torch.half)
    mant, exp = split_half(para)  # 分离尾数和指数
    exp2fp8 = exp.clone()
    exp2fp8[:] = threshold
    sign = para.sign()  # 获取符号
    mant = mant.abs()  # 取绝对值

    # 定义不同量化区域
    region2zero = exp < -14  # 指数过小，量化为0
    region2one = (exp > -15) & (exp < threshold - 10)  # 指数适中，量化为1

    quanted_mant = mant.clone()
    torch.ldexp(quanted_mant, exp2fp8, out=quanted_mant)  # 左移指数
    if rtn:
        quanted_mant = quanted_mant.round().to(dtype=torch.half)  # 四舍五入
    else:
        quanted_mant = quanted_mant.trunc().to(dtype=torch.half)  # 截断
    torch.ldexp(quanted_mant, -exp2fp8, out=quanted_mant)  # 右移指数

    # 应用量化规则
    quanted_mant[region2zero] = 0.0  # 过小值量化为0
    quanted_mant[keep_mask] = mant[keep_mask]  # 保持指定位置的精度
    quanted_mant[region2one] = 1.0  # 适中值量化为1
    region2two = region2one & (mant > 1.5)  # 大于1.5的值量化为2
    quanted_mant[region2two] = 2.0
    res = torch.ldexp(quanted_mant, exp).to(dtype=torch.half)  # 恢复指数
    res *= sign  # 恢复符号

    return res.to(dtype=torch.float16)


def quantize_clip(para: torch.Tensor, threshold: int = 2, rtn: bool = True) -> torch.Tensor:
    """
    裁剪量化函数：对参数进行量化，不保持特定位置精度

    Args:
        para: 待量化的参数张量
        threshold: 指数阈值
        rtn: 是否使用四舍五入

    Returns:
        量化后的参数张量
    """
    para = para.to(dtype=torch.half)
    mant, exp = split_half(para)
    exp2fp8 = exp.clone()
    exp2fp8[:] = threshold
    sign = para.sign()
    mant = mant.abs()

    region2zero = exp < -14
    region2one = (exp > -15) & (exp < threshold - 10)

    quanted_mant = mant.clone()
    torch.ldexp(quanted_mant, exp2fp8, out=quanted_mant)
    if rtn:
        quanted_mant = quanted_mant.round().to(dtype=torch.half)
    else:
        quanted_mant = quanted_mant.trunc().to(dtype=torch.half)
    torch.ldexp(quanted_mant, -exp2fp8, out=quanted_mant)

    quanted_mant[region2zero] = 0.0
    quanted_mant[region2one] = 1.0
    region2two = region2one & (mant > 1.5)
    quanted_mant[region2two] = 2.0
    res = torch.ldexp(quanted_mant, exp).to(dtype=torch.half)
    res *= sign

    return res


def split_half(para: torch.Tensor, zero_exp_value=0) -> tuple[torch.Tensor, torch.Tensor]:
    """
    将半精度浮点数分离为尾数和指数

    Args:
        para: 输入张量或数组
        zero_exp_value: 零值对应的指数值

    Returns:
        mant: 尾数
        exp: 指数
    """
    if isinstance(para, torch.Tensor):
        mant, exp = para.frexp()  # 分离尾数和指数
    elif isinstance(para, np.ndarray):
        mant, exp = np.frexp(para)
    else:
        raise TypeError("input must be torch.Tensor or np.ndarray")

    # 处理特殊值
    region_inf = (para == np.half("inf")) | (para == np.half("-inf"))  # 无穷大
    region_nan = para == np.half("Nan")  # NaN
    zero_exp = para == 0  # 零值
    exp = exp - 1
    exp[exp < -14] = -15  # 次正规数
    exp[region_inf | region_nan] = 16  # 无穷大和NaN
    region_subnormal = exp == -15  # 次正规数区域
    exp[zero_exp] = zero_exp_value  # 零值指数
    mant = mant * 2
    mant[region_subnormal] = (para * 2**15)[region_subnormal]  # 次正规数处理
    return mant, exp


class AdmmPruner:
    """
    ADMM稀疏器：使用交替方向乘子法进行模型稀疏化

    ADMM是一种优化算法，
    用于解决带有约束的优化问题，在模型稀疏化中用于找到最优的权重稀疏模式。
    """

    def __init__(self, layer: nn.Linear):
        """
        初始化ADMM稀疏器

        Args:
            layer: 需要稀疏化的 nn.Linear 层
        """
        self.layer = layer
        self.dev = self.layer.weight.device
        self.rows = layer.weight.shape[0]  # 权重矩阵行数
        self.columns = layer.weight.shape[1]  # 权重矩阵列数
        self.hessian = torch.zeros((self.columns, self.columns), device=self.dev)  # Hessian矩阵
        self.nsamples = 0  # 样本数量
        self.scaler_row = torch.zeros((self.columns), device=self.dev)  # 行缩放因子

    def add_batch(self, inp: torch.Tensor):
        """
        添加一批输入数据，更新统计信息

        Args:
            inp: 输入数据
        """
        inp_reshape = inp.reshape(-1, inp.shape[-1]).float()
        self.hessian += inp_reshape.T.matmul(inp_reshape)  # 累积Hessian矩阵

        inp_tmp = inp.clone()
        if len(inp_tmp.shape) == 2:
            inp_tmp = inp_tmp.unsqueeze(0)
        tmp = inp_tmp.shape[0]
        if isinstance(self.layer, nn.Linear):
            if len(inp_tmp.shape) == 3:
                inp_tmp = inp_tmp.reshape((-1, inp_tmp.shape[-1]))
            inp_tmp = inp_tmp.t()

        # 更新行缩放因子
        self.scaler_row *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp

        inp_tmp = inp_tmp.type(torch.float32)
        self.scaler_row += torch.norm(inp_tmp, p=2, dim=1) ** 2 / self.nsamples

    @torch.no_grad()
    def fasterprune(self, sparse_ratio: float):
        """
        执行快速稀疏化

        Args:
            sparse_ratio: 稀疏比例
        """
        hessian = self.hessian
        norm = torch.diag(hessian).sqrt() + 1e-8  # 计算归一化因子
        hessian = hessian / norm
        hessian = (hessian.T / norm).T
        normalized_weights = (self.layer.weight.float().detach() * norm).T

        # 设置ADMM参数
        rho0 = PERCDAMP * torch.diag(hessian).mean()  # 初始惩罚参数
        diag = torch.arange(hessian.shape[0], device=hessian.device)
        hessian[diag, diag] += rho0

        rho = 1

        hessian_weight_mult = hessian.matmul(normalized_weights)
        hessian[diag, diag] += rho

        # 计算逆矩阵
        o_device = hessian.device
        hessian = hessian.to("cpu")
        hessian_inv = torch.inverse(hessian).to(o_device)
        self.hessian = None
        del hessian

        lagrange_multiplier = torch.zeros_like(normalized_weights)  # 拉格朗日乘子

        # ADMM主循环
        for itt in range(ITERS):
            if ITERATIVE_PRUNE > 0 and itt < ITERATIVE_PRUNE:
                # 迭代稀疏：选择绝对值最小的元素进行稀疏化
                topk = torch.topk(
                    (normalized_weights + lagrange_multiplier).abs().flatten(),
                    k=int(normalized_weights.numel() * sparse_ratio),
                    largest=False,
                )
                mask = torch.ones(normalized_weights.numel(), dtype=torch.bool, device=normalized_weights.device)
                mask[topk.indices] = 0
                mask = mask.reshape(normalized_weights.shape)
                del topk
            sparse_weights = (normalized_weights + lagrange_multiplier) * mask  # 投影到稀疏空间

            lagrange_multiplier = lagrange_multiplier + (normalized_weights - sparse_weights)  # 更新拉格朗日乘子

            normalized_weights = hessian_inv.matmul(
                hessian_weight_mult + rho * (sparse_weights - lagrange_multiplier)
            )  # 更新权重

        sparse_weights = (normalized_weights + lagrange_multiplier) * mask  # 最终稀疏结果
        out = sparse_weights.T / norm  # 反归一化

        self.layer.weight.data = out.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype).contiguous()

        def get_keep_mask2(keep_proportion: float = 0.02) -> torch.Tensor:
            """
            获取需要保持精度的位置掩码

            Args:
                keep_proportion: 保持精度的比例

            Returns:
                保持精度的位置掩码
            """
            weights_fp8 = quantize_clip(para=self.layer.weight.data, threshold=KEEP_BITS)
            # 计算量化误差与缩放因子的乘积作为保持精度的度量
            keep_metric = torch.abs(self.layer.weight.data - weights_fp8) * torch.sqrt(self.scaler_row.reshape((1, -1)))
            value2keep = keep_metric.flatten().sort(descending=True)[0][int(keep_metric.numel() * keep_proportion - 1)]
            region2keep = keep_metric >= value2keep
            return region2keep

        keep_mask = get_keep_mask2(keep_proportion=KEEP_PROPORTION)
        # 应用L2量化，保持重要位置的精度
        self.layer.weight.data = quantize_l2(keep_mask=keep_mask, para=self.layer.weight.data, threshold=KEEP_BITS).to(
            self.layer.weight.data.dtype
        )

    def free(self):
        """释放内存"""
        self.hessian = None
        # 检查 torch.npu 是否可用，如果可用则清理缓存
        if hasattr(torch, 'npu'):
            torch.npu.empty_cache()
