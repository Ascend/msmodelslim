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
MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details.
-------------------------------------------------------------------------
"""

# qparam 参数分组：定义“哪些参数是量化参数”的命名约定，并据此把可训练参数
# 切分为 qparam / decay / no-decay 三组优化器参数组（可注入的分组策略）。
# log_qparams=true 时使用 RelativeAlphaAdamW：对数空间更新 + 相对初始 alpha clamp。

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import torch
from torch import nn

from msmodelslim.utils.exception import SchemaValidateError
from msmodelslim.utils.logging import get_logger


def matches_qparam(param_name: str, qparam_names: Sequence[str]) -> bool:
    """参数名是否命中 qparam 命名约定（全等或点分隔后缀匹配）。"""
    return any(param_name == qname or param_name.endswith("." + qname) for qname in qparam_names)


@dataclass(frozen=True)
class ParamGroupResult:
    """分组结果：优化器 param groups 与各组参数计数。"""

    groups: List[Dict[str, Any]] = field(default_factory=list)
    n_decay: int = 0
    n_no_decay: int = 0
    n_qparam: int = 0


class RelativeAlphaAdamW(torch.optim.AdamW):
    """AdamW with log-space LSQ alpha updates and relative clamps.

    Parameters in a group tagged ``qat_group="qparam"`` keep fp32 optimizer
    state and an fp32 master copy. Updates follow:

        alpha <- alpha * exp(-lr * m_hat / (sqrt(v_hat) + eps))

    then clamped to ``[alpha_min_ratio, alpha_max_ratio] * alpha0``.
    Weight groups are updated by the stock AdamW path.
    """

    def __init__(
        self,
        params: Any,
        *args: Any,
        alpha_ref: Optional[Dict[int, torch.Tensor]] = None,
        alpha_min_ratio: float = 0.5,
        alpha_max_ratio: float = 2.0,
        **kwargs: Any,
    ):
        super().__init__(params, *args, **kwargs)
        self._alpha_ref = dict(alpha_ref or {})
        self._alpha_min_ratio = float(alpha_min_ratio)
        self._alpha_max_ratio = float(alpha_max_ratio)

    def _init_state(self, param: Any) -> Dict[str, Any]:
        state = self.state[param]
        if "exp_avg" not in state:
            state["step"] = 0
            state["exp_avg"] = torch.zeros_like(param, dtype=torch.float32)
            state["exp_avg_sq"] = torch.zeros_like(param, dtype=torch.float32)
            state["alpha_master"] = param.detach().to(torch.float32).clone()
        return state

    @torch.no_grad()
    def _update_qparam(self, param: Any, grad: torch.Tensor, group: Dict[str, Any]) -> None:
        state = self._init_state(param)
        state["step"] += 1
        beta1, beta2 = group["betas"]
        eps = float(group["eps"])
        lr = float(group["lr"])
        exp_avg = state["exp_avg"]
        exp_avg_sq = state["exp_avg_sq"]
        grad_fp32 = grad.to(torch.float32)
        exp_avg.mul_(beta1).add_(grad_fp32, alpha=1.0 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad_fp32, grad_fp32, value=1.0 - beta2)
        bias_correction1 = 1.0 - beta1 ** state["step"]
        bias_correction2 = 1.0 - beta2 ** state["step"]
        denom = exp_avg_sq.div(bias_correction2).sqrt_().add_(eps)
        direction = exp_avg.div(bias_correction1).div_(denom)
        master = state["alpha_master"]
        master.mul_(torch.exp(-lr * direction))
        min_ratio = float(group.get("alpha_min_ratio", self._alpha_min_ratio))
        max_ratio = float(group.get("alpha_max_ratio", self._alpha_max_ratio))
        ref = self._alpha_ref.get(id(param))
        if ref is None:
            ref = state.get("alpha_ref")
        if ref is None:
            state["alpha_ref"] = master.clone()
            ref = state["alpha_ref"]
        master.clamp_(min=min_ratio * ref, max=max_ratio * ref)
        param.copy_(master.to(param.dtype))

    @torch.no_grad()
    def step(self, closure: Any = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        deferred = []
        for group in self.param_groups:
            if group.get("qat_group") != "qparam":
                continue
            for param in group["params"]:
                if param.grad is None:
                    continue
                deferred.append((param, param.grad, group))
                param.grad = None

        super().step()

        for param, grad, group in deferred:
            self._update_qparam(param, grad, group)
        return loss


class QParamGroupBuilder:
    """按 qparam 命名约定构建分组优化器参数组。

    Groups:
      - qparam (name endswith ``qparam_names``): ``lr * qparam_lr_scale``, no decay
      - regular decayed params: base lr, ``weight_decay``
      - regular no-decay params (bias / norm): base lr, no decay
    """

    def __init__(
        self,
        qparam_names: Sequence[str],
        qparam_lr_scale: float,
        *,
        alpha_min_ratio: float = 0.5,
        alpha_max_ratio: float = 2.0,
    ):
        self.qparam_names = tuple(qparam_names)
        self.qparam_lr_scale = float(qparam_lr_scale)
        self.alpha_min_ratio = float(alpha_min_ratio)
        self.alpha_max_ratio = float(alpha_max_ratio)

    def matches(self, param_name: str) -> bool:
        """参数名是否命中本组配置的 qparam 命名约定。"""
        return matches_qparam(param_name, self.qparam_names)

    def build(
        self,
        model: nn.Module,
        *,
        decay_names: Sequence[str],
        base_lr: float,
        weight_decay: float,
    ) -> ParamGroupResult:
        """遍历 ``model`` 可训练参数并产出三组 param groups。"""
        qparam_params: List[Any] = []
        decay_params: List[Any] = []
        no_decay_params: List[Any] = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if self.matches(name):
                qparam_params.append(param)
            elif name in decay_names:
                decay_params.append(param)
            else:
                no_decay_params.append(param)

        grouped_parameters: List[Dict[str, Any]] = []
        if decay_params:
            grouped_parameters.append(
                {
                    "params": decay_params,
                    "lr": base_lr,
                    "weight_decay": weight_decay,
                    "qat_group": "weight",
                }
            )
        if no_decay_params:
            grouped_parameters.append(
                {
                    "params": no_decay_params,
                    "lr": base_lr,
                    "weight_decay": 0.0,
                    "qat_group": "weight",
                }
            )
        if qparam_params:
            grouped_parameters.append(
                {
                    "params": qparam_params,
                    "lr": base_lr * self.qparam_lr_scale,
                    "weight_decay": 0.0,
                    "qat_group": "qparam",
                    "alpha_min_ratio": self.alpha_min_ratio,
                    "alpha_max_ratio": self.alpha_max_ratio,
                }
            )
        else:
            get_logger().warning(
                "QParamGroupBuilder: no trainable qparam matched %s; "
                "fake-quant injection may be missing or params are frozen.",
                self.qparam_names,
            )
        if not grouped_parameters:
            raise SchemaValidateError(
                "QParamGroupBuilder found no trainable parameters",
                action="Please check the model requires_grad state before train().",
            )
        return ParamGroupResult(
            groups=grouped_parameters,
            n_decay=len(decay_params),
            n_no_decay=len(no_decay_params),
            n_qparam=len(qparam_params),
        )
