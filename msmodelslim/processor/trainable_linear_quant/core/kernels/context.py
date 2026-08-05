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

# 伪量化流水线基类上下文、选项与 listener（INT/MX 子类见 ``int`` / ``mxfp``）。

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Literal, Optional, TYPE_CHECKING, Union

import torch

from msmodelslim.core.quantizer.base import QConfig
from msmodelslim.ir.qal import QDType, QParam

KernelFamily = Literal["int", "mxfp"]

_INT_QDTYPES = frozenset({QDType.INT4, QDType.INT8})
_MXFP_QDTYPES = frozenset({QDType.MXFP4, QDType.MXFP8})


def kernel_family_from_config(config: QConfig) -> KernelFamily:
    """由 ``QConfig.dtype`` 推导 TLQ 行为族（与 kernel 注册表 dispatch 一致）。"""
    q_dtype = QDType(config.dtype)
    if q_dtype in _INT_QDTYPES:
        return "int"
    if q_dtype in _MXFP_QDTYPES:
        return "mxfp"
    raise TypeError(f"unsupported TLQ kernel dtype {q_dtype!r}")


if TYPE_CHECKING:
    from .int import IntFakeQuantContext
    from .mxfp import MxFakeQuantContext

FakeQuantContext = Union["IntFakeQuantContext", "MxFakeQuantContext", "FakeQuantPipelineContext"]
ContextFactory = Callable[
    [QConfig, torch.Tensor, Dict[str, torch.Tensor]],
    FakeQuantContext,
]


@dataclass
class FakeQuantPipelineContext:
    """各 dtype 流水线共享的可变状态（纯 ``torch.Tensor``，不依赖 ``QStorage``）。"""

    config: QConfig
    float_tensor: torch.Tensor
    train_tensors: Dict[str, torch.Tensor] = field(default_factory=dict)
    normed: Optional[torch.Tensor] = None
    q_param: Optional[QParam] = None
    quantized: Optional[torch.Tensor] = None

    @property
    def working_tensor(self) -> torch.Tensor:
        return self.float_tensor

    def set_working_tensor(self, tensor: torch.Tensor) -> None:
        """Listener 改写流水线输入浮点张量。"""
        self.float_tensor = tensor


@dataclass
class FakeQuantResult:
    """``TLQKernel.fake_quantize`` 返回值。"""

    tensor: torch.Tensor
    q_param: QParam
    quantized: Optional[torch.Tensor] = None

    @property
    def scale(self) -> torch.Tensor:
        return self.q_param.ext["scale"]

    @property
    def offset(self) -> torch.Tensor:
        return self.q_param.ext.get("offset", torch.zeros_like(self.scale))
