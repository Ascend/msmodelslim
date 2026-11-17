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

# Lightweight SignSGD: θ ← θ - lr * sign(g).
#
# Behavior aligns with AutoRound's SignSGD under the default path
# (momentum=0, weight_decay=0). Slim reimplementation for TLQ — not a
# verbatim copy of AutoRound or ``torch.optim.SGD``.

from __future__ import annotations

from typing import Any, Iterable, Optional, Union

import torch
from torch.optim.optimizer import Optimizer


class SignSGD(Optimizer):
    """Sign stochastic gradient descent.

    Args:
        params: Iterable of parameters or param-group dicts.
        lr: Learning rate (per-coordinate step size is ±lr).
    """

    def __init__(
        self,
        params: Iterable[Union[torch.Tensor, dict[str, Any]]],
        lr: float,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        super().__init__(params, dict(lr=lr))

    @torch.no_grad()
    def step(self, closure: Optional[Any] = None) -> Optional[float]:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.add_(p.grad.sign(), alpha=-lr)

        return loss
