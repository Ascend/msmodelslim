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
# pylint: disable=useless-import-alias

import torch
from torch import nn as nn
from torch.nn import functional as F

from msmodelslim.core.convert.protocol import ICheckpointReader
from msmodelslim.core.convert.types import IRKind, SourceIR, TensorRef
from msmodelslim.core.observer import MsMinMaxBlockObserver, MinMaxBlockObserverConfig
from msmodelslim.core.quantization_schema.api import calculate_qparam, fake_quantize, dequantize
from msmodelslim.core.quantization_schema.qal import QABCRegistry, QDType, QParam, QScheme, QScope, QStorage
from msmodelslim.core.quantization_schema.utils import reshape_to_blocks, undo_reshape_to_blocks
from msmodelslim.core.quantization_schema import AutoFakeQuantLinear
from msmodelslim.core.quantization_schema import mxfp4_per_block_sym, mxfp8_per_block_sym
from msmodelslim.utils.logging import logger_setter


@QABCRegistry.multi_register(
    dispatch_key=[
        (mxfp8_per_block_sym, mxfp4_per_block_sym),
    ],
    abc_type=AutoFakeQuantLinear,
)
@logger_setter()
class W4A8MXDynamicPerBlockFakeQuantLinear(AutoFakeQuantLinear):
    def __init__(self, x_q_param: QParam, w_q_param: QParam, w_q: QStorage, bias: torch.Tensor):
        super().__init__()
        self.w_scheme = w_q_param.scheme
        self.w_mx_finfo = w_q_param.scheme.dtype.mx_finfo
        self.w_axes = w_q_param.ext.get("axes")

        self.x_scheme = x_q_param.scheme
        self.x_mx_finfo = x_q_param.scheme.dtype.mx_finfo
        self.x_axes = x_q_param.ext.get("axes")

        self.weight_scale = nn.Parameter(w_q_param.ext.get("scale"), requires_grad=False)
        self.weight_offset = nn.Parameter(w_q_param.ext.get("offset"), requires_grad=False)
        self.weight = nn.Parameter(w_q.value, requires_grad=False)
        self.bias = nn.Parameter(bias, requires_grad=False) if bias is not None else None

        minmax_config = MinMaxBlockObserverConfig(axes=self.x_axes)
        self.x_minmax_block_observer = MsMinMaxBlockObserver(minmax_config)

    def __repr__(self) -> str:
        return f"W4A8MXDynamicPerBlockFakeQuantLinear(symmetric={self.w_scheme.symmetric})"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        axes = self.x_axes
        axes = [axes] if isinstance(axes, int) else axes
        axes = [a + x.ndim if a < 0 else a for a in axes]
        x, axes_, orig_shape, padded_shape = reshape_to_blocks(x, axes, self.x_mx_finfo.block_size)
        shared_exp_axes = [x + 1 for x in axes_] if self.x_mx_finfo.block_size > 0 else axes_
        self.x_minmax_block_observer.update(x, shared_exp_axes=shared_exp_axes)
        x_min_val, x_max_val = self.x_minmax_block_observer.get_min_max()
        x_q_param = calculate_qparam(
            x_min_val,
            x_max_val,
            q_dtype=self.x_scheme.dtype,
            q_scope=self.x_scheme.scope,
            symmetric=self.x_scheme.symmetric,
        )
        x_q_dq = fake_quantize(QStorage(QDType.FLOAT, x), x_q_param)
        x_q_dq.value = undo_reshape_to_blocks(x_q_dq.value, padded_shape, orig_shape, axes)

        w_q_storage = QStorage(dtype=self.w_scheme.dtype, value=self.weight)
        axes = self.w_axes
        axes = [axes] if isinstance(axes, int) else axes
        axes = [a + w_q_storage.value.ndim if a < 0 else a for a in axes]
        w_q_storage.value, _, w_orig_shape, w_padded_shape = reshape_to_blocks(
            w_q_storage.value, axes, self.w_mx_finfo.block_size
        )
        w_q_param = QParam(
            scheme=QScheme(scope=QScope.PER_BLOCK, dtype=QDType.MXFP4, symmetric=True),
            ext={"scale": self.weight_scale.data},
        )
        weight_q_dq = dequantize(w_q_storage, w_q_param)
        weight_q_dq.value = undo_reshape_to_blocks(weight_q_dq.value, w_padded_shape, w_orig_shape, axes)

        return F.linear(x_q_dq.value, weight_q_dq.value, self.bias)


class PackedW4A8MXDynamicPerBlockFakeQuantLinear(AutoFakeQuantLinear):
    """已打包的 W4A8 MX per-block 线性层。

    继承 ``AutoFakeQuantLinear``，但不注册量化分发，也不做伪量化。
    权重按 checkpoint key 懒加载，参数名保持 checkpoint 叶子名。
    Ascend saver 的 ``on_packed_w4a8_mx_dynamic_per_block`` 遍历 ``named_parameters`` 原样写出，
    不把 ``scale`` 改成 ``weight_scale``，也不 ``pack_fp4_to_uint8`` 或给 scale 加 127。
    仅当 ``config.json`` 的 ``quant_method`` 为 ``fp8`` 且 ``activation_scheme`` 为
    ``dynamic`` 时，虚拟树才会把未匹配的 MXFP4 挂成这个类型。
    """

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
        """按 checkpoint key 加载已打包权重，数值不做转换。"""
        if self.lazy_initialized:
            return
        from msmodelslim.core.quant_service.modelslim_convert.weight_mapping.fused_load import (  # pylint: disable=import-outside-toplevel
            load_logical_tensor,
        )

        dev = str(device)
        direct: dict[str, list[str]] = {}
        logical_to_ckpt: dict[str, str] = {}
        for logical, ref in self.tensor_bindings.items():
            meta = ref.meta or {}
            if meta.get("fused_from"):
                self._register_logical(logical, load_logical_tensor(reader, ref.key, meta, device=dev))
                continue
            ckpt_key = meta.get("renamed_from") or ref.key
            logical_to_ckpt[logical] = ckpt_key
            direct.setdefault(ref.shard, []).append(ckpt_key)

        if direct:
            tensors = reader.load_tensors({shard: sorted(set(keys)) for shard, keys in direct.items()}, device=dev)
            for logical, ref in self.tensor_bindings.items():
                if (ref.meta or {}).get("fused_from"):
                    continue
                tensor = tensors.get(logical_to_ckpt.get(logical, ref.key))
                if tensor is None:
                    continue
                self._register_logical(logical, tensor)
        self.lazy_initialized = True

    def _register_logical(self, logical: str, tensor: torch.Tensor) -> None:
        """参数名用 checkpoint 叶子名，saver 按这个名字写出。"""
        self.register_parameter(logical.replace(".", "_"), nn.Parameter(tensor, requires_grad=False))

    def _single_bound_parameter(self, prefix: str) -> tuple[str, nn.Parameter] | None:
        if prefix != self.full_name or len(self.tensor_bindings) != 1:
            return None
        logical, ref = next(iter(self.tensor_bindings.items()))
        if ref.key not in (self.full_name, prefix):
            return None
        param = self._parameters.get(logical.replace(".", "_"))
        if param is None:
            return None
        return prefix, param

    def named_parameters(self, prefix: str = "", recurse: bool = True, remove_duplicate: bool = True):
        """单张量且 full_name 就是 checkpoint key 时，写出名用 prefix 本身。"""
        if not recurse:
            single = self._single_bound_parameter(prefix)
            if single is not None:
                yield single
                return
        yield from super().named_parameters(prefix=prefix, recurse=recurse, remove_duplicate=remove_duplicate)
