#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
虚拟 nn.Module 节点（convert_design.md §8）。

用途：
  - 为 SaveProcessor 提供 ``named_modules()`` 遍历结构；
  - 为 IR 任务提供 ``lazy_init`` 绑定的权重属性名（weight / weight_scale_inv 等）。

不做 forward；转换后 module 类型可能变为 qir.FakeQuantLinear 等。
"""

from __future__ import annotations

import torch
from torch import nn

from msmodelslim.core.quant_service.modelslim_convert.weight_mapping.fused_load import load_logical_tensor
from msmodelslim.core.convert.protocol import ICheckpointReader
from msmodelslim.core.convert.types import IRKind, SourceIR, TensorRef


class ModelFreeModule(nn.Module):  # pylint: disable=abstract-method
    """
    Convert 专用模块基类：元数据 + ``tensor_bindings``，首次访问权重时 ``lazy_init``。

    ``tensor_bindings`` 的 key 为逻辑名（如 weight），value 为 ``TensorRef``（含 checkpoint key、
    shard、以及 preprocess 写入的 ``meta``，例如 ``fused_from``）。
    """

    full_name: str
    source_format: str | None
    source_ir: SourceIR
    target_ir: IRKind | None
    tensor_bindings: dict[str, TensorRef]
    lazy_initialized: bool

    def __init__(
        self,
        full_name: str,
        tensor_bindings: dict[str, TensorRef] | None = None,
        source_format: str | None = None,
        source_ir: SourceIR | None = None,
        target_ir: IRKind | None = None,
    ) -> None:
        super().__init__()
        self.full_name = full_name
        self.source_format = source_format
        self.source_ir = source_ir or SourceIR(kind=IRKind.UNKNOWN)
        self.target_ir = target_ir
        self.tensor_bindings = tensor_bindings or {}
        self.lazy_initialized = False

    def lazy_init(self, reader: ICheckpointReader, device: torch.device | str = "cpu") -> None:
        """
        从 checkpoint 加载绑定张量到 Parameter/Buffer。

        - 普通 key：按 shard 批量 ``reader.load_tensors``。
        - ``meta.fused_from``：走 ``load_logical_tensor``，只读 fused 张量再切片（§6.3）。
        """
        if self.lazy_initialized:
            return
        dev = str(device)
        direct: dict[str, list[str]] = {}
        for logical, ref in self.tensor_bindings.items():
            meta = ref.meta or {}
            if meta.get("fused_from"):
                tensor = load_logical_tensor(reader, ref.key, meta, device=dev)
                self._register_logical(logical, tensor)
                continue
            direct.setdefault(ref.shard, []).append(ref.key)

        if direct:
            merged = {s: sorted(set(k)) for s, k in direct.items()}
            tensors = reader.load_tensors(merged, device=dev)
            for logical, ref in self.tensor_bindings.items():
                if ref.meta.get("fused_from"):
                    continue
                tensor = tensors.get(ref.key)
                if tensor is None:
                    continue
                self._register_logical(logical, tensor)
        self.lazy_initialized = True

    def _register_logical(self, logical: str, tensor: torch.Tensor) -> None:
        if logical in ("weight", "bias"):
            self.register_parameter(logical, nn.Parameter(tensor, requires_grad=False))
        else:
            self.register_buffer(logical, tensor, persistent=True)


class ModelFreeLinear(ModelFreeModule):  # pylint: disable=abstract-method
    """Linear 语义虚拟模块；convert_rules 仅对此类生成 IRTask。"""

    pass


class PassthroughModule(ModelFreeModule):  # pylint: disable=abstract-method
    """Norm / embedding / catalog 余量：原样 FLOAT 落盘。

    全部挂 Parameter，供 AscendV1 与 HuggingFace 的 ``on_float_module`` 共用
    ``named_parameters(recurse=False, prefix=...)`` 写出 checkpoint key。
    """

    def _register_logical(self, logical: str, tensor: torch.Tensor) -> None:
        # AscendV1 仅遍历 named_parameters；非 weight/bias 的 A_log 等也注册为 Parameter
        safe = logical.replace(".", "_")
        self.register_parameter(safe, nn.Parameter(tensor, requires_grad=False))

    def _single_bound_parameter(self, prefix: str) -> tuple[str, nn.Parameter] | None:
        if prefix != self.full_name or len(self.tensor_bindings) != 1:
            return None
        logical, ref = next(iter(self.tensor_bindings.items()))
        if ref.key not in (self.full_name, prefix):
            return None
        safe = logical.replace(".", "_")
        param = self._parameters.get(safe)
        if param is None:
            return None
        return prefix, param

    def named_parameters(self, prefix: str = "", recurse: bool = True, remove_duplicate: bool = True):
        """
        单子量且 ``full_name`` 与 checkpoint key 一致时（如 ``...experts.down_proj``），
        写出名须为 prefix 本身，不能是 ``prefix.down_proj``。
        """
        if not recurse:
            single = self._single_bound_parameter(prefix)
            if single is not None:
                yield single
                return
        yield from super().named_parameters(prefix=prefix, recurse=recurse, remove_duplicate=remove_duplicate)


def create_model_free_module(
    module_path: str,
    tensor_bindings: dict[str, TensorRef],
    source_ir: SourceIR,
    target_ir: IRKind,
    source_format: str | None = None,
    module_kind: str = "linear",
) -> ModelFreeModule:
    """按 ``module_kind`` 选择具体虚拟模块类。"""
    cls = PassthroughModule if module_kind in ("norm", "embedding", "lm_head", "passthrough") else ModelFreeLinear
    return cls(
        full_name=module_path,
        tensor_bindings=tensor_bindings,
        source_format=source_format,
        source_ir=source_ir,
        target_ir=target_ir,
    )


def set_submodule_by_path(root: nn.Module, path: str, module: nn.Module) -> None:
    """按点分路径挂载子模块，缺失的中间节点自动创建为 ``nn.Module()``。"""
    parts = path.split(".")
    parent = root
    for part in parts[:-1]:
        if not hasattr(parent, part) or getattr(parent, part) is None:
            setattr(parent, part, nn.Module())
        parent = getattr(parent, part)
    setattr(parent, parts[-1], module)
