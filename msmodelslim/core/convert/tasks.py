#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
IR-level work units for parallel conversion.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn

from msmodelslim.core.convert.edges import RouteConstraints, TransformEdge
from msmodelslim.core.convert.types import IRKind, SourceIR, TensorRef

_MXFP8_DEPLOY_KEYS = frozenset({"weight", "weight_scale", "weight_offset"})


@dataclass(frozen=True)
class PortableTensor:
    """
    跨进程按值传输的 tensor 表示。

    多进程回传时若直接 pickle ``torch.Tensor``，torch 的 multiprocessing reduction 会
    走共享内存 + mmap（``rebuild_storage_fd``）；大规模 MoE 下 mmap 区域数会超过
    ``vm.max_map_count`` 触发 ``Cannot allocate memory``。

    这里改为「原始字节(uint8) + dtype 名 + shape」，仅传输纯 bytes，不占共享内存/mmap，
    主进程用 ``to_tensor`` 精确还原（含 bf16/fp8 等无 numpy 对应的 torch 专有 dtype）。
    """

    raw: bytes
    dtype_name: str
    shape: tuple[int, ...]

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor) -> "PortableTensor":
        contiguous = tensor.detach().cpu().contiguous()
        byte_view = contiguous.view(torch.uint8).reshape(-1)
        return cls(
            raw=bytes(byte_view.numpy().tobytes()),
            dtype_name=str(contiguous.dtype).removeprefix("torch."),
            shape=tuple(contiguous.shape),
        )

    def to_tensor(self) -> torch.Tensor:
        dtype = getattr(torch, self.dtype_name)
        flat = torch.frombuffer(bytearray(self.raw), dtype=torch.uint8).clone()
        return flat.view(dtype).reshape(self.shape)


def _restore_state_dict(state_dict: dict[str, Any]) -> dict[str, Any]:
    """把回传 state_dict 中的 ``PortableTensor`` 还原为 ``torch.Tensor``。"""
    return {
        key: (value.to_tensor() if isinstance(value, PortableTensor) else value) for key, value in state_dict.items()
    }


def is_mxfp8_deploy_state(state_dict: dict[str, Any]) -> bool:
    """判断 state_dict 是否来自 ``W8A8MXDynamicPerBlockFakeQuantLinear.deploy()``。"""
    return _MXFP8_DEPLOY_KEYS.issubset(state_dict.keys())


def _float_module_from_state_dict(state_dict: dict[str, Any]) -> nn.Module:
    """从 state_dict 重建 FLOAT 模块；直接挂 Parameter，避免 ``nn.Linear.load_state_dict`` 把 bf16 升成 float32。"""
    weight = state_dict.get("weight")
    if weight is not None and weight.ndim == 2:
        bias = state_dict.get("bias")
        linear = nn.Linear(weight.shape[1], weight.shape[0], bias=bias is not None)
        linear.weight = nn.Parameter(weight.detach(), requires_grad=False)
        if bias is not None:
            linear.bias = nn.Parameter(bias.detach(), requires_grad=False)
        return linear

    mod = nn.Module()
    for name, tensor in state_dict.items():
        mod.register_parameter(name, nn.Parameter(tensor.detach(), requires_grad=False))
    return mod


@dataclass
class IRTask:
    """
    One convertible unit (typically one linear module on the virtual tree).

    Scheduling granularity is ``dependency_group`` by default so partner tensors
    across shards load together (cf. model_free_ptq inverse_weight_map).
    """

    module_path: str
    source_ir: SourceIR
    target_ir: IRKind
    tensor_bindings: dict[str, TensorRef]
    inverse_weight_map: dict[str, list[str] | None]
    route_constraints: RouteConstraints | None = None
    estimated_bytes: int = 0
    device: str = "cpu"
    meta: dict[str, Any] = field(default_factory=dict)

    def create_empty_module(self) -> nn.Module:
        """Build placeholder module for this task; filled by lazy_init in worker."""
        from msmodelslim.core.quant_service.modelslim_convert.virtual_module import (
            create_model_free_module,
        )

        return create_model_free_module(
            module_path=self.module_path,
            tensor_bindings=self.tensor_bindings,
            source_ir=self.source_ir,
            target_ir=self.target_ir,
        )


@dataclass
class RoutedTask:
    """IR task plus resolved processor route."""

    task: IRTask
    route: list[TransformEdge]
    route_ir_names: list[IRKind]


@dataclass
class IRResult:
    """Output of one IR task after transform chain."""

    module_path: str
    final_ir: IRKind
    module: nn.Module | None = None
    state_dict: dict[str, Any] | None = None
    already_saved: bool = False
    loss_level: str = "lossy"
    route_ir_names: list[IRKind] = field(default_factory=list)

    def resolve_module(self) -> nn.Module:
        """返回可写入虚拟树的模块；多进程 state_dict 路径会重建 FakeQuant 类型。"""
        if self.module is not None:
            return self.module
        if self.state_dict is None:
            raise ValueError(f"IRResult for {self.module_path} has no module or state_dict")
        state_dict = _restore_state_dict(self.state_dict)
        if self.final_ir == IRKind.FLOAT:
            return _float_module_from_state_dict(state_dict)
        if self.final_ir == IRKind.W8A8_MXFP8:
            if is_mxfp8_deploy_state(state_dict):
                from msmodelslim.ir.w8a8_mx_dynamic import (
                    W8A8MXDynamicPerBlockFakeQuantLinear,
                )

                return W8A8MXDynamicPerBlockFakeQuantLinear.from_deploy_state_dict(state_dict)
            return _float_module_from_state_dict(state_dict)
        raise ValueError(
            f"Cannot rebuild module for {self.module_path} with final_ir={self.final_ir!r} from state_dict"
        )

    def materialize_to_module(self, target: nn.Module) -> None:
        """Copy weights/IR fields from result into an existing virtual-tree module."""
        if self.module is None and self.state_dict is None:
            raise ValueError(f"IRResult for {self.module_path} has no payload")
        if self.module is not None:
            target.load_state_dict(self.module.state_dict(), strict=False)
        elif self.state_dict is not None:
            target.load_state_dict(_restore_state_dict(self.state_dict), strict=False)
