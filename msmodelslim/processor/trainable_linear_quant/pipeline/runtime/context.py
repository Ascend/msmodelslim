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

# Per-block session state for trainable linear quant processing.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, TYPE_CHECKING

import torch

from msmodelslim.processor.trainable_linear_quant.data import BlockOutput

if TYPE_CHECKING:
    from msmodelslim.processor.trainable_linear_quant.core.ops.base import TLQOp
    from msmodelslim.processor.trainable_linear_quant.core.train import BlockTrainResult
    from msmodelslim.processor.trainable_linear_quant.core.wrapper import TrainableLinearQuantWrapper


@dataclass
class BlockTLQContext:
    """Single-block session state across preprocess → process → postprocess."""

    block_name: str
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))

    # Forward captures
    teacher_outputs: List[BlockOutput] = field(default_factory=list)
    train_result: Optional["BlockTrainResult"] = None

    # Block setup (populated during preprocess)
    wrappers_by_path: Dict[str, TrainableLinearQuantWrapper] = field(default_factory=dict)
    ops: List[TLQOp] = field(default_factory=list)

    def require_ops(self) -> List[TLQOp]:
        if not self.ops:
            raise RuntimeError(f"block {self.block_name!r} has no TLQ ops installed; run preprocess first")
        return self.ops

    def release(self) -> None:
        """Drop training-time references after block finalize (free device memory)."""
        for op in self.ops:
            op.release_cached_params()
            op.target_modules.clear()
        self.ops.clear()
        self.wrappers_by_path.clear()
        self.teacher_outputs.clear()
        self.train_result = None


__all__ = ["BlockTLQContext"]
