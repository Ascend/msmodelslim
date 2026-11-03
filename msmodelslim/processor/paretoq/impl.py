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

from __future__ import annotations

import fnmatch
from typing import Any, List, Literal, Optional, Tuple

import torch
import torch.nn as nn  # pylint: disable=consider-using-from-import
import torch.nn.functional as F

from msmodelslim.app.train.qat.qat_linear_algorithm import QATLinearAlgorithm
from msmodelslim.core.convert.protocol import ConvertContext, IIRTransformProcessor
from msmodelslim.ir import (
    AutoFakeQuantLinear,
    int4_per_channel_sym,
    int4_per_token_sym,
    int8_per_token_sym,
)
from msmodelslim.ir.qal import QDType, QParam, QStorage
from msmodelslim.processor.paretoq.config import ParetoQConfig
from msmodelslim.processor.paretoq.core import (
    fake_quant_weight,
    init_weight_clip_from_weight,
)
from msmodelslim.utils.config_map import ConfigSet
from msmodelslim.utils.exception import SchemaValidateError
from msmodelslim.utils.logging import get_logger
from msmodelslim.utils.plugin import register_plugin

DeployAct = Literal["a8", "a4"]

_DEPLOY_ACT_SCHEMES = {
    "a8": int8_per_token_sym,
    "a4": int4_per_token_sym,
}

_QPARAM_NAMES: Tuple[str, ...] = ("weight_clip_val",)


class ParetoQFakeQuantLinear(nn.Linear):
    """ParetoQ fake-quant linear：训练期 weight-only LSQ 壳；部署走 ``deploy()``。"""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        *,
        w_bits: int = 16,
        weight_layerwise: bool = False,
        deploy_act: DeployAct = "a8",
        weight_clip_val: Optional[torch.Tensor] = None,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__(in_features, out_features, bias=bias, **factory_kwargs)
        self.w_bits = w_bits
        self.weight_layerwise = weight_layerwise
        self.deploy_act = deploy_act
        if self.w_bits < 16:
            if weight_clip_val is not None:
                self.weight_clip_val = nn.Parameter(
                    weight_clip_val.detach().to(device=self.weight.device, dtype=self.weight.dtype).clone()
                )
            else:
                # 占位；QAT 路径由 ``ParetoQImpl.init_weight_clip_val`` 填充。
                self.weight_clip_val = nn.Parameter(
                    torch.zeros(
                        self.out_features,
                        1,
                        device=self.weight.device,
                        dtype=self.weight.dtype,
                    )
                )

    @classmethod
    def from_qat_tensors(
        cls,
        weight: torch.Tensor,
        weight_clip_val: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        *,
        w_bits: int,
        weight_layerwise: bool = False,
        deploy_act: DeployAct = "a8",
    ) -> "ParetoQFakeQuantLinear":
        """从 checkpoint 张量（W + 已训练 α）构建可部署壳。"""
        if w_bits >= 16:
            raise ValueError(f"from_qat_tensors requires w_bits < 16 for deploy, got {w_bits}")
        if weight.ndim != 2:
            raise ValueError(f"from_qat_tensors expects 2D weight, got shape {tuple(weight.shape)}")
        module = cls(
            weight.shape[1],
            weight.shape[0],
            bias=bias is not None,
            w_bits=w_bits,
            weight_layerwise=weight_layerwise,
            deploy_act=deploy_act,
            weight_clip_val=weight_clip_val,
            device=weight.device,
            dtype=weight.dtype,
        )
        with torch.no_grad():
            module.weight.copy_(weight.detach())
            if bias is not None:
                module.bias.copy_(bias.detach())
        return module

    def forward(self, input_):
        if self.w_bits >= 16:
            weight = self.weight
        else:
            weight = fake_quant_weight(
                self.weight,
                self.weight_clip_val,
                self.w_bits,
                self.weight_layerwise,
            ).to(input_.dtype)
        return F.linear(input_, weight, self.bias)

    def deploy(self) -> nn.Module:
        """将训练好的浮点 W + α 转为可部署 IR（如 W4A8 / W4A4）。不原地修改 ``self``。"""
        if self.w_bits >= 16:
            raise ValueError(
                "Cannot deploy fp16/passthrough ParetoQFakeQuantLinear "
                f"(w_bits={self.w_bits}); expected quantized shell with w_bits < 16"
            )

        act_scheme = _DEPLOY_ACT_SCHEMES[self.deploy_act]
        alpha = self.weight_clip_val.detach()
        q_w = self._quantize_weight(self.weight.detach(), alpha, self.w_bits)
        scale = alpha.squeeze(-1) if alpha.dim() > 1 else alpha
        x_q_param = QParam(scheme=act_scheme, ext={"axes": -1})
        w_ext = {"scale": scale}
        if self.deploy_act == "a4":
            w_ext["offset"] = torch.zeros_like(scale, dtype=torch.int32)
        w_q_param = QParam(scheme=int4_per_channel_sym, ext=w_ext)
        w_q = QStorage(dtype=QDType.INT4, value=q_w)
        bias = self.bias.detach() if self.bias is not None else None
        return AutoFakeQuantLinear.create(x_q_param, w_q_param, w_q, bias)

    @staticmethod
    def _quantize_weight(
        weight: torch.Tensor,
        weight_clip_val: torch.Tensor,
        w_bits: int,
    ) -> torch.Tensor:
        eps = torch.tensor(1e-5, device=weight_clip_val.device, dtype=weight_clip_val.dtype)
        alpha = torch.where(weight_clip_val > eps, weight_clip_val, eps)
        if w_bits in (0, 1):
            qn, qp = -1, 1
        else:
            qn = -(2 ** (w_bits - 1))
            qp = 2 ** (w_bits - 1) - 1
        return (weight / alpha).round().clamp(qn, qp)


def _set_submodule(root: nn.Module, full_name: str, module: nn.Module) -> None:
    set_child = getattr(root, "set_submodule", None)
    if callable(set_child):
        set_child(full_name, module)
        return
    if not full_name:
        raise ValueError("Cannot replace the root module without set_submodule")
    parent_name, _, child = full_name.rpartition(".")
    parent = root.get_submodule(parent_name) if parent_name else root
    setattr(parent, child, module)


class ParetoQImpl(QATLinearAlgorithm):
    """
    ParetoQ 对 ``QATLinearAlgorithm`` 的实现（由 ``AlgorithmTrainHook`` 委托）：

    - ``replace_linears``：替换匹配的 ``nn.Linear`` 为 FakeQuant 壳
    - ``init_weight_clip_val``：初始化可学习 α
    - convert：从虚拟模块物化壳并 ``deploy()``
    """

    def __init__(self, config: ParetoQConfig, model: Optional[nn.Module] = None) -> None:
        self.config = config
        self.model = model
        self.include = ConfigSet(list(config.include))
        self.exclude = ConfigSet(list(config.exclude))

    def qparam_names(self) -> Tuple[str, ...]:
        return _QPARAM_NAMES

    def replace_linears(self, model: nn.Module) -> List[ParetoQFakeQuantLinear]:
        """按 include/exclude 将精确类型为 ``nn.Linear`` 的模块替换为 FakeQuant 壳。"""
        self.model = model
        cfg = self.config
        layerwise = cfg.weight_scope == "layerwise"
        targets: List[Tuple[str, nn.Linear]] = []
        for name, module in list(model.named_modules()):
            # Exact type: subclasses such as the fake-quant shell must not be replaced again.
            if not name or type(module) is not nn.Linear:  # pylint: disable=unidiomatic-typecheck
                continue
            if name not in self.include or name in self.exclude:
                continue
            targets.append((name, module))

        shells: List[ParetoQFakeQuantLinear] = []
        for name, linear in targets:
            shell = ParetoQFakeQuantLinear(
                linear.in_features,
                linear.out_features,
                bias=linear.bias is not None,
                w_bits=cfg.w_bits,
                weight_layerwise=layerwise,
                deploy_act=cfg.deploy_act,
                device=linear.weight.device,
                dtype=linear.weight.dtype,
            )
            with torch.no_grad():
                shell.weight.copy_(linear.weight)
                if linear.bias is not None:
                    shell.bias.copy_(linear.bias)
            _set_submodule(model, name, shell)
            shells.append(shell)
            get_logger().debug("ParetoQ: %s -> ParetoQFakeQuantLinear", name)

        get_logger().info(
            "ParetoQ replace_linears: replaced %s Linear(s) (w_bits=%s)",
            len(shells),
            cfg.w_bits,
        )
        return shells

    def init_weight_clip_val(self, shells: List[ParetoQFakeQuantLinear]) -> None:
        """按 ``weight_clip_init`` 策略，用浮点权重初始化每层可学习 α。"""
        init_method = self.config.weight_clip_init
        w_bits = self.config.w_bits
        for shell in shells:
            with torch.no_grad():
                shell.weight_clip_val.copy_(
                    init_weight_clip_from_weight(shell.weight, w_bits, init_method).to(
                        device=shell.weight.device,
                        dtype=shell.weight.dtype,
                    )
                )
        get_logger().info(
            "ParetoQ init_weight_clip_val: inited %s shell(s) method=%s",
            len(shells),
            init_method,
        )

    @staticmethod
    def _module_defaults(module_path: str, context: ConvertContext | None) -> dict[str, Any]:
        if context is None:
            return {}
        rules = context.config.module_rules
        for rule in rules:
            if rule.match == module_path and rule.defaults:
                return dict(rule.defaults)
        for rule in rules:
            if rule.match != module_path and fnmatch.fnmatch(module_path, rule.match) and rule.defaults:
                return dict(rule.defaults)
        return {}

    @classmethod
    def materialize_from_virtual(
        cls,
        module: nn.Module,
        context: ConvertContext | None = None,
    ) -> ParetoQFakeQuantLinear:
        """convert 侧：从虚拟模块（W + 已训练 α）物化 ParetoQ 壳；禁止 minmax 重算 α。"""
        # 懒导入，避免 impl ↔ modelslim_convert 包初始化环。
        from msmodelslim.core.quant_service.modelslim_convert.virtual_module import (
            ModelFreeLinear,
        )

        if isinstance(module, ParetoQFakeQuantLinear):
            return module
        if not isinstance(module, ModelFreeLinear):
            raise SchemaValidateError(
                f"ParetoQ convert expects ModelFreeLinear or ParetoQFakeQuantLinear, got {type(module).__name__}",
                action="Please ensure QAT deploy module_rules mark the layer as PARETOQ_FLOAT.",
            )

        weight = getattr(module, "weight", None)
        if weight is None or weight.ndim != 2:
            raise SchemaValidateError(
                f"ParetoQ convert requires a 2D weight on {module.full_name!r}",
                action="Please ensure the QAT checkpoint safetensors contains the module weight.",
            )

        alpha = getattr(module, "weight_clip_val", None)
        if alpha is None:
            raise SchemaValidateError(
                f"ParetoQ convert requires weight_clip_val on {module.full_name!r}",
                action="Please ensure the QAT checkpoint safetensors contains trained α.",
            )

        meta = cls._module_defaults(module.full_name, context)
        w_bits = int(meta.get("w_bits", 4))
        if w_bits < 0 or w_bits > 4:
            raise SchemaValidateError(
                f"ParetoQ convert cannot deploy module {module.full_name!r} with w_bits={w_bits}",
                action="Please use a quantized shell with w_bits in [0, 4].",
            )
        deploy_act = str(meta.get("deploy_act") or "a8")
        if deploy_act not in ("a8", "a4"):
            raise SchemaValidateError(
                f"Unsupported deploy_act={deploy_act!r} for module {module.full_name!r}",
                action="Please use deploy_act a8 or a4.",
            )

        return ParetoQFakeQuantLinear.from_qat_tensors(
            weight,
            alpha,
            getattr(module, "bias", None),
            w_bits=w_bits,
            weight_layerwise=bool(meta.get("weight_layerwise", False)),
            deploy_act=deploy_act,
        )

    @classmethod
    def transform_for_convert(
        cls,
        module: nn.Module,
        context: ConvertContext,
    ) -> nn.Module:
        """convert 侧：物化壳后 ``deploy()`` 到目标 IR。"""
        return cls.materialize_from_virtual(module, context).deploy()

    @classmethod
    def get_convert_processors(cls) -> List[IIRTransformProcessor]:
        """
        算法侧闭环：提供 QAT→deploy 的 ``IIRTransformProcessor`` 边。

        Application / convert worker 仅查询并注册，不硬编码具体 Processor 类型。
        """
        # 懒导入，避免 impl ↔ convert 模块初始化环。
        from msmodelslim.processor.paretoq.convert import get_paretoq_convert_processors

        return list(get_paretoq_convert_processors())


def get_plugin():
    """Plugin:train_algorithm entry — (ParetoQConfig, ParetoQImpl) 算法插件。"""
    return ParetoQConfig, ParetoQImpl


register_plugin(get_plugin)
