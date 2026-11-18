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
from dataclasses import dataclass
from typing import Any, List, Mapping, Optional, Tuple

import torch
import torch.nn as nn  # pylint: disable=consider-using-from-import
import torch.nn.functional as F

from msmodelslim.app.train.qat.qat_linear_algorithm import QATLinearAlgorithm
from msmodelslim.core.convert.protocol import ConvertContext, IIRTransformProcessor
from msmodelslim.core.quantization_algorithm.paretoq.config import DeployAct, ParetoQConfig, ParetoQQuantSpec
from msmodelslim.core.quantization_algorithm.paretoq.core import (
    fake_quant_weight,
    init_weight_clip_from_weight,
)
from msmodelslim.core.quantization_schema import (
    AutoFakeQuantActivation,
    AutoFakeQuantDynamicCache,
    AutoFakeQuantLinear,
    WrapperIR,
    int4_per_channel_sym,
    int4_per_token_sym,
    int8_per_token_sym,
)
from msmodelslim.core.quantization_schema.qal import QDType, QParam, QScheme, QStorage
from msmodelslim.utils.config_map import ConfigSet
from msmodelslim.utils.exception import SchemaValidateError, UnsupportedError
from msmodelslim.utils.logging import get_logger
from msmodelslim.utils.plugin import register_plugin

_QPARAM_NAMES: Tuple[str, ...] = ("weight_clip_val",)


@dataclass(frozen=True)
class _DeployRuntime:
    qscheme: QScheme
    weight_offset_dtype: Optional[torch.dtype] = None


_DEPLOY_RUNTIMES = {
    DeployAct.A4: _DeployRuntime(qscheme=int4_per_token_sym, weight_offset_dtype=torch.int32),
    DeployAct.A8: _DeployRuntime(qscheme=int8_per_token_sym),
}


class ParetoQFakeQuantLinear(nn.Linear):
    """ParetoQ 训练壳：浮点 master weight + 可学习 ``weight_clip_val``。"""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        *,
        quant_spec: ParetoQQuantSpec,
        weight_clip_val: Optional[torch.Tensor] = None,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__(in_features, out_features, bias=bias, **factory_kwargs)
        self.quant_spec = quant_spec
        if self.w_bits >= 16:
            return
        if weight_clip_val is None:
            weight_clip_val = torch.zeros(
                self.out_features,
                1,
                device=self.weight.device,
                dtype=self.weight.dtype,
            )
        self.weight_clip_val = nn.Parameter(
            weight_clip_val.detach().to(device=self.weight.device, dtype=self.weight.dtype).clone()
        )

    @property
    def w_bits(self) -> int:
        return self.quant_spec.w_bits

    @property
    def weight_layerwise(self) -> bool:
        return self.quant_spec.weight_layerwise

    @property
    def deploy_act(self) -> DeployAct:
        return self.quant_spec.deploy_act

    @classmethod
    def from_qat_tensors(
        cls,
        weight: torch.Tensor,
        weight_clip_val: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        *,
        quant_spec: ParetoQQuantSpec,
    ) -> "ParetoQFakeQuantLinear":
        """从 QAT checkpoint 的浮点 weight、α 和可选 bias 还原训练壳。"""
        if quant_spec.w_bits >= 16:
            raise ValueError(f"from_qat_tensors requires w_bits < 16 for deploy, got {quant_spec.w_bits}")
        if weight.ndim != 2:
            raise ValueError(f"from_qat_tensors expects 2D weight, got shape {tuple(weight.shape)}")

        module = cls(
            weight.shape[1],
            weight.shape[0],
            bias=bias is not None,
            quant_spec=quant_spec,
            weight_clip_val=weight_clip_val,
            device=weight.device,
            dtype=weight.dtype,
        )
        with torch.no_grad():
            module.weight.copy_(weight.detach())
            if bias is not None:
                module.bias.copy_(bias.detach())
        return module

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
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
        """将浮点 W + α 转为目标伪量化 IR；不原地修改 ``self``。"""
        if self.w_bits >= 16:
            raise ValueError(
                "Cannot deploy fp16/passthrough ParetoQFakeQuantLinear "
                f"(w_bits={self.w_bits}); expected quantized shell with w_bits < 16"
            )

        deploy_runtime = _DEPLOY_RUNTIMES[self.deploy_act]
        alpha = self.weight_clip_val.detach()
        q_w = self._quantize_weight(self.weight.detach(), alpha, self.w_bits)
        scale = alpha.squeeze(-1) if alpha.dim() > 1 else alpha
        x_q_param = QParam(scheme=deploy_runtime.qscheme, ext={"axes": -1})
        w_ext = {"scale": scale}
        if deploy_runtime.weight_offset_dtype is not None:
            w_ext["offset"] = torch.zeros_like(scale, dtype=deploy_runtime.weight_offset_dtype)
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


def _spec_from_checkpoint_meta(meta: Mapping[str, Any]) -> ParetoQQuantSpec:
    return ParetoQQuantSpec(
        w_bits=int(meta.get("w_bits", 4)),
        weight_layerwise=bool(meta.get("weight_layerwise", False)),
        deploy_act=meta.get("deploy_act") or DeployAct.A8,
    )


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


def _build_shell_from_float(
    impl: "ParetoQImpl",
    name: str,
    source: nn.Linear,
) -> ParetoQFakeQuantLinear:
    shell = ParetoQFakeQuantLinear(
        source.in_features,
        source.out_features,
        bias=source.bias is not None,
        quant_spec=impl.config.quant_spec,
        device=source.weight.device,
        dtype=source.weight.dtype,
    )
    if not impl._module_is_meta(source):
        with torch.no_grad():
            shell.weight.copy_(source.weight)
            if source.bias is not None:
                shell.bias.copy_(source.bias)
            shell.weight_clip_val.copy_(
                init_weight_clip_from_weight(
                    source.weight,
                    impl.config.quant_spec.w_bits,
                    impl.config.weight.qparam_init,
                ).to(
                    device=shell.weight_clip_val.device,
                    dtype=shell.weight_clip_val.dtype,
                )
            )
    get_logger().debug("ParetoQ: %s -> ParetoQFakeQuantLinear", name)
    return shell


# Source-type dispatch table: add a PTQ IR builder here when a new input IR is supported.
_SOURCE_BUILDERS = {
    nn.Linear: _build_shell_from_float,
}


class ParetoQImpl(QATLinearAlgorithm):
    """
    ParetoQ 对 ``QATLinearAlgorithm`` 的实现（由 ``AlgorithmTrainHook`` 委托）：

    - ``replace_linears``：按源模块类型分发表替换 FLOAT Linear（及后续注册的 PTQ IR），
      并在建壳时完成可学习 α 初始化
    - convert：从虚拟模块物化壳并 ``deploy()``
    """

    def __init__(self, config: ParetoQConfig, model: Optional[nn.Module] = None) -> None:
        self.config = config
        self.model = model
        self.include = ConfigSet(list(config.include))
        self.exclude = ConfigSet(list(config.exclude))

    def qparam_names(self) -> Tuple[str, ...]:
        return _QPARAM_NAMES

    def replace_linears(self, model: nn.Module) -> List[Tuple[str, ParetoQFakeQuantLinear]]:
        """按源模块类型分发替换，并在建壳时完成 α 初始化。"""
        self.model = model
        shells: List[Tuple[str, ParetoQFakeQuantLinear]] = []
        for name, module in list(model.named_modules()):
            if not name:
                continue
            # Exact type lookup prevents replacing ParetoQ shells via the nn.Linear row.
            builder = _SOURCE_BUILDERS.get(type(module))  # pylint: disable=unidiomatic-typecheck
            if builder is None:
                if isinstance(
                    module,
                    (AutoFakeQuantLinear, AutoFakeQuantActivation, AutoFakeQuantDynamicCache, WrapperIR),
                ):
                    raise UnsupportedError(
                        f"ParetoQ replace_linears does not support IR {type(module).__name__} at '{name}'",
                        action="Only FLOAT nn.Linear modules are supported in this release.",
                    )
                continue

            if name not in self.include or name in self.exclude:
                # Registered PTQ IR outside include/exclude must fail closed.
                if isinstance(module, AutoFakeQuantLinear):
                    raise UnsupportedError(
                        f"Quant IR '{name}' is outside the ParetoQ include/exclude scope",
                        action="Please include every quantized PTQ layer, or export those layers as FLOAT.",
                    )
                continue

            shell = builder(self, name, module)
            _set_submodule(model, name, shell)
            shells.append((name, shell))

        get_logger().info(
            "ParetoQ replace_linears: replaced %s Linear(s) (w_bits=%s)",
            len(shells),
            self.config.quant_spec.w_bits,
        )
        return shells

    @staticmethod
    def _module_is_meta(module: nn.Module) -> bool:
        return any(p.is_meta for p in module.parameters())

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
        try:
            quant_spec = _spec_from_checkpoint_meta(meta)
        except ValueError as exc:
            raise SchemaValidateError(
                f"Invalid ParetoQ metadata for module {module.full_name!r}: {exc}",
                action="Please use w_bits in [0, 4] and deploy_act a8 or a4.",
            ) from exc

        return ParetoQFakeQuantLinear.from_qat_tensors(
            weight,
            alpha,
            getattr(module, "bias", None),
            quant_spec=quant_spec,
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
        from msmodelslim.core.quantization_algorithm.paretoq.convert import get_paretoq_convert_processors

        return list(get_paretoq_convert_processors())


def get_plugin():
    """Plugin:train_algorithm entry — (ParetoQConfig, ParetoQImpl) 算法插件。"""
    return ParetoQConfig, ParetoQImpl


register_plugin(get_plugin)
