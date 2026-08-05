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

# 可监听检查点的 TLQ Kernel 工具类（listener 框架，dtype 数学由注入的 driver 实现）。

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from enum import Enum
from typing import Callable, Dict, Generator, List, Optional

import torch

from msmodelslim.core.quantizer.base import QConfig

from .context import (
    ContextFactory,
    FakeQuantContext,
    FakeQuantResult,
)


class FakeQuantStage(str, Enum):
    """伪量化 forward 路径上的可监听检查点（INT / MXFP 共用枚举，语义见各 driver）。"""

    QUANT_INPUT = "quant_input"
    BEFORE_QPARAM = "before_qparam"
    QPARAM_READY = "qparam_ready"
    PRE_ROUND = "pre_round"
    QUANTIZED = "quantized"
    DEQUANT_OUTPUT = "dequant_output"


FakeQuantListener = Callable[[FakeQuantStage, FakeQuantContext], None]
FakeQuantDriver = Callable[
    [FakeQuantContext],
    Generator[FakeQuantStage, None, FakeQuantResult],
]


@dataclass
class _ListenerEntry:
    op_id: str
    fn: FakeQuantListener


class TLQKernel:
    """可监听伪量化工具类。

    职责：

    1. **监听回调管理**：通过 ``add_listener`` / ``remove_listeners`` 注册与移除训练 Op
       注入的 listener。
    2. **回调事件循环**：驱动 dtype 专有 ``driver`` 生成器；每个检查点 yield 后，将
       ``(stage, ctx)`` 原样分发给全部 listener，由 listener 自行判断是否处理并直接改写
       共享 ``ctx``；driver 在后续步骤读取已更新的流水线状态。

    act/weight 区分由 Wrapper 持有的 ``x_kernel`` / ``w_kernel`` 两个实例承担，
    而非 kernel 内部字段。

    dtype 相关的伪量化数学由注入的无状态 ``driver`` 与 ``context_factory`` 实现。
    """

    def __init__(
        self,
        config: QConfig,
        driver: FakeQuantDriver,
        context_factory: ContextFactory,
    ) -> None:
        self.config = config
        self._driver = driver
        self._context_factory = context_factory
        self._listeners: List[_ListenerEntry] = []

    def add_listener(self, fn: FakeQuantListener, op_id: str) -> None:
        """注册 listener；``op_id`` 用于 ``remove_listeners`` 批量移除。"""
        self._listeners.append(_ListenerEntry(op_id=op_id, fn=fn))

    def remove_listeners(self, op_id: str) -> None:
        self._listeners = [e for e in self._listeners if e.op_id != op_id]

    def fake_quantize(
        self,
        float_tensor: torch.Tensor,
        train_tensors: Optional[Dict[str, torch.Tensor]] = None,
        return_quantized_weight: bool = False,
    ) -> FakeQuantResult:
        """对浮点张量执行一次伪量化。

        Args:
            float_tensor: 待伪量化的浮点输入（激活或权重）。
            train_tensors: 本次调用注入 listener 的可训练张量（如 round_tune 的 ``value``）。
            return_quantized_weight: 为 ``True`` 时 driver 直接返回 ``quantized`` 网格（unwrap 写回 layer 用）。
        """
        ctx = self._context_factory(
            self.config,
            float_tensor,
            dict(train_tensors or {}),
        )
        driver = partial(self._driver, return_quantized_weight=return_quantized_weight)(ctx)
        try:
            while True:
                stage = next(driver)
                self._dispatch(stage, ctx)
        except StopIteration as stop:
            result: FakeQuantResult = stop.value
            return result

    def _dispatch(self, stage: FakeQuantStage, ctx: FakeQuantContext) -> None:
        for entry in self._listeners:
            entry.fn(stage, ctx)
