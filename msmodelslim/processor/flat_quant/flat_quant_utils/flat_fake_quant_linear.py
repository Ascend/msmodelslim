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

import torch
from torch import nn
from typing import Any, ClassVar, Optional
import torch.nn.functional as F
from msmodelslim.processor.flat_quant.flat_quant_utils.fake_clip_quantizer import WeightQuantizer, ActivationQuantizer
from pydantic import BaseModel, Field


class ForwardMode(BaseModel):
    ORG: ClassVar[str] = "org"
    CALIB: ClassVar[str] = "calib"
    EVAL: ClassVar[str] = "eval"

    @classmethod
    def get_description(cls, mode: str) -> str:
        descriptions = {"org": "原生模式", "calib": "校准模式", "eval": "推理模式"}
        return descriptions.get(mode, "未知模式")


class FlatFakeQuantLinearConfig(BaseModel):
    """
    伪量化线性层的配置类，用于控制权重与激活的量化行为。
    """

    w_bits: int = Field(default=16, description="权重位宽")
    a_bits: int = Field(default=16, description="激活位宽")
    w_asym: bool = Field(default=False, description="权重是否使用非对称量化")
    a_asym: bool = Field(default=False, description="激活是否使用非对称量化")
    lwc: bool = Field(default=False, description="是否启用权重的逐层量化（Layer-wise Weight Quantization）")
    lac: bool = Field(default=False, description="是否启用激活的逐层量化（Layer-wise Activation Quantization）")
    a_groupsize: int = Field(default=-1, description="激活分组大小（-1 表示按张量整体量化）")
    a_per_tensor: bool = Field(
        default=False, description="激活是否按张量整体量化（True 表示 per-tensor，False 表示 per-channel）"
    )


class FlatFakeQuantLinear(nn.Module):
    """支持变换矩阵的伪量化线性层，具备模式切换与量化前向能力"""

    def __init__(
        self,
        config: FlatFakeQuantLinearConfig,
        linear: nn.Linear,
    ):
        super().__init__()
        self.config = config
        self.linear = linear
        self.weight_quantizer = WeightQuantizer(
            bits=config.w_bits,
            in_size=linear.weight.shape[1],
            out_size=linear.weight.shape[0],
            perchannel=True,
            sym=not config.w_asym,
            lwc=config.lwc,
        )
        self.act_quantizer = ActivationQuantizer(
            bits=config.a_bits,
            sym=not config.a_asym,
            lac=config.lac,
            groupsize=config.a_groupsize,
            per_tensor=config.a_per_tensor,
        )

        self._mode = ForwardMode.ORG
        self.weight_in_trans = None
        self.weight_out_trans = None
        self.act_in_trans = None
        self.save_trans = None

    def unwrapper(self) -> nn.Linear:
        """返回原始线性层，用于恢复或调试"""
        return self.linear

    def del_linear(self) -> None:
        """删除内部线性层及其相关量化器，释放内存"""
        del self.linear, self.weight_quantizer, self.act_quantizer

    @property
    def weight(self) -> torch.Tensor:
        """获取当前层的权重张量"""
        return self.linear.weight

    @property
    def bias(self) -> Optional[torch.Tensor]:
        """获取当前层的偏置张量"""
        return self.linear.bias

    def extra_repr(self) -> str:
        """返回该模块的可读性表示，用于打印调试"""
        return f"weight shape: {tuple(self.weight.shape)}, bias={self.bias is not None}"

    def set_act_clip_factor(self, clip_factor: float) -> None:
        """设置激活值剪裁因子，用于动态校准"""
        if self.act_quantizer.lac:
            self.act_quantizer.clip_factor = clip_factor

    def set_trans(
        self,
        weight_in_trans: Optional[Any] = None,
        weight_out_trans: Optional[Any] = None,
        act_in_trans: Optional[Any] = None,
        save_trans: Optional[Any] = None,
    ) -> None:
        """设置量化自适应变换矩阵，用于处理权重与激活的变换"""
        self.weight_in_trans = weight_in_trans
        self.weight_out_trans = weight_out_trans
        self.act_in_trans = act_in_trans
        self.save_trans = save_trans

    def fake_quant_weight(self) -> None:
        """对权重应用伪量化（原地操作），用于校准阶段"""
        self.linear.weight.data = self.weight_quantizer.get_fake_quant_weight(self.linear.weight.data)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """前向传播，根据当前模式选择原始、校准或评估路径"""
        if self._mode == ForwardMode.EVAL:
            hidden_states = self.act_quantizer(hidden_states)
            return F.linear(hidden_states, self.weight, self.bias)
        elif self._mode == ForwardMode.ORG:
            return F.linear(hidden_states, self.weight, self.bias)
        else:
            return self._calib_forward(hidden_states)

    def _calib_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """校准模式下的前向传播，处理变换矩阵与量化操作"""
        if self.act_in_trans is not None:
            hidden_states = self.act_in_trans(hidden_states)

        weight = self.weight.data
        if self.weight_in_trans is not None:
            weight = self.weight_in_trans(weight, inv_t=True)

        if self.weight_quantizer.lwc:
            weight = self.weight_quantizer.apply_wclip(weight)

        if self.weight_out_trans is not None:
            weight = self.weight_out_trans(weight.T).T

        if self.weight_out_trans is not None and self.bias is not None:
            bias = self.weight_out_trans(self.bias.data)
        else:
            bias = self.bias

        hidden_states = self.act_quantizer(hidden_states)
        weight = self.weight_quantizer(weight)
        return F.linear(hidden_states, weight, bias)

    def change_mode(self, mod: ForwardMode) -> None:
        """统一的模式切换接口，用于在不同阶段切换计算路径"""
        if mod == ForwardMode.EVAL:
            self.weight_quantizer.reparameterize()
            self.act_quantizer.reparameterize()
            self._reparameterize()
        self._mode = mod

    def _reparameterize(self) -> None:
        """将权重转换为评估模式下的最终形式，移除变换矩阵"""
        if self._mode == ForwardMode.EVAL:
            return

        weight = self.weight.data
        ori_dtype = weight.dtype
        weight = weight.to(torch.float64)

        if self.weight_in_trans is not None:
            weight = self.weight_in_trans(weight, inv_t=True)

        if self.weight_quantizer.lwc:
            weight = self.weight_quantizer.apply_wclip(weight)

        if self.weight_out_trans is not None:
            weight = self.weight_out_trans(weight.T).T

        if self.weight_out_trans is not None and self.bias is not None:
            self.bias.data = self.weight_out_trans(self.bias.data)

        self.weight.data = weight.to(ori_dtype)

        del self.weight_in_trans
        del self.weight_out_trans
        self.weight_in_trans = None
        self.weight_out_trans = None


class FlatNormWrapper(nn.Module):
    """支持变换矩阵的归一化层包装器，用于在量化流程中处理归一化"""

    def __init__(
        self,
        norm: nn.Module,
        trans: Any = None,
    ):
        super().__init__()
        self.norm = norm
        self.trans = trans
        self._mode = ForwardMode.ORG

    def unwrapper(self) -> nn.Module:
        """返回原始线性层，用于恢复或调试"""
        return self.norm

    def del_norm(self) -> None:
        """删除内部线性层及其相关量化器，释放内存"""
        del self.norm

    @property
    def weight(self) -> torch.Tensor:
        """获取归一化层的权重张量"""
        return self.norm.weight

    @property
    def bias(self) -> Optional[torch.Tensor]:
        """获取归一化层的偏置张量"""
        return self.norm.bias

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """前向传播，根据当前模式选择原始或校准/评估路径"""
        if self._mode == ForwardMode.ORG:
            return self._ori_forward(hidden_states)
        else:
            return self._calib_eval_forward(hidden_states)

    def change_mode(self, mod: ForwardMode) -> None:
        """统一的模式切换接口，用于切换归一化模块的运行模式"""
        if mod == ForwardMode.EVAL:
            self.trans = None
        self._mode = mod

    def _ori_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """原始前向传播，不应用任何变换"""
        return self.norm(hidden_states)

    def _calib_eval_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """校准或评估模式下的前向传播，应用变换矩阵"""
        hidden_states = self.norm(hidden_states)
        if self.trans is not None:
            hidden_states = self.trans(hidden_states)
        return hidden_states
