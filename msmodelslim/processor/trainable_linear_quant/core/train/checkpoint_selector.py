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

# Checkpoint selection strategies for TLQ block training.

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional, Sequence, Tuple, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from msmodelslim.processor.trainable_linear_quant.config.train_config import BlockTrainConfig
    from msmodelslim.processor.trainable_linear_quant.core.ops.base import TLQOp


@dataclass(frozen=True)
class BlockTrainResult:
    """Summary of a completed block training run.

    When ``completed_iters`` is 0 (``iters=0``), no optimization ran; ``best_iter`` is 0
    and losses are 0 — the saved snapshot is bind-time initialized parameters.
    """

    best_iter: int
    init_loss: float
    best_loss: float
    completed_iters: int


def save_best_params(tlq_ops: Sequence[TLQOp]) -> None:
    """Snapshot live trainable params for each op into ``TLQOp._best_params``."""
    for op in tlq_ops:
        op.save_best_params()


def _should_early_stop(iter_idx: int, last_best_iter: int, patience: int) -> bool:
    return 0 < patience <= (iter_idx - last_best_iter)


class CheckpointSelector(ABC):
    """Select when to snapshot TLQ op trainable parameters during block training."""

    def __init__(self, iters: int) -> None:
        self._iters = iters
        self._last_saved_iter = -1
        self._init_loss: Optional[float] = None
        self._last_avg_loss = 0.0

    @classmethod
    def from_config(cls, config: BlockTrainConfig) -> CheckpointSelector:
        select_best = config.select_best
        if select_best.mode == "last":
            return LastIterSelector(iters=config.iters)
        if select_best.mode == "ema":
            return EmaCheckpointSelector(
                iters=config.iters,
                ema_beta=select_best.ema_beta,
                ema_window_size=select_best.ema_window_size,
                early_stop_patience=select_best.early_stop_patience,
            )
        return MinLossCheckpointSelector(
            iters=config.iters,
            early_stop_patience=select_best.early_stop_patience,
        )

    @property
    def last_saved_iter(self) -> int:
        return self._last_saved_iter

    @property
    def last_avg_loss(self) -> float:
        return self._last_avg_loss

    def select(self, iter_idx: int, mean_loss: float) -> Tuple[bool, bool]:
        """Return ``(should_save, should_stop)`` after forward/backward, before ``optimizer.step``."""
        saved = False
        if iter_idx == 0:
            self._init_loss = mean_loss
            saved = self._save(iter_idx)
        saved = self._maybe_save(iter_idx, mean_loss) or saved
        self._last_avg_loss = mean_loss
        return saved, False

    @abstractmethod
    def get_result(
        self,
        last_mean_loss: float,
        completed_iters: Optional[int] = None,
    ) -> BlockTrainResult: ...

    @abstractmethod
    def _maybe_save(self, iter_idx: int, mean_loss: float) -> bool: ...

    def _save(self, iter_idx: int) -> bool:
        self._last_saved_iter = iter_idx
        return True


class EmaCheckpointSelector(CheckpointSelector):
    def __init__(
        self,
        iters: int,
        ema_beta: float,
        ema_window_size: int,
        early_stop_patience: int = -1,
    ) -> None:
        super().__init__(iters=iters)
        self._early_stop_patience = early_stop_patience
        self._ema_beta = ema_beta
        self._last_best_iter = 0
        self._best_loss = torch.finfo(torch.float).max
        self._loss_history: Deque[float] = deque(maxlen=ema_window_size)

    def select(self, iter_idx: int, mean_loss: float) -> Tuple[bool, bool]:
        saved, _ = super().select(iter_idx, mean_loss)
        if self._loss_history:
            self._last_avg_loss = sum(self._loss_history) / len(self._loss_history)
        should_stop = _should_early_stop(
            iter_idx,
            self._last_best_iter,
            self._early_stop_patience,
        )
        return saved, should_stop

    def _maybe_save(self, iter_idx: int, mean_loss: float) -> bool:
        if iter_idx == 0:
            self._best_loss = mean_loss
        self._loss_history.append(mean_loss)
        avg_loss = sum(self._loss_history) / len(self._loss_history)
        if avg_loss < self._best_loss:
            self._best_loss = self._ema_beta * self._best_loss + (1.0 - self._ema_beta) * avg_loss
            self._last_best_iter = iter_idx
            self._save(iter_idx)
            return True
        return False

    def get_result(
        self,
        last_mean_loss: float,
        completed_iters: Optional[int] = None,
    ) -> BlockTrainResult:
        return BlockTrainResult(
            best_iter=self._last_best_iter,
            init_loss=self._init_loss if self._init_loss is not None else last_mean_loss,
            best_loss=self._best_loss,
            completed_iters=completed_iters if completed_iters is not None else self._iters,
        )


class MinLossCheckpointSelector(CheckpointSelector):
    """Save checkpoint when per-iter ``mean_loss`` hits a new minimum."""

    def __init__(self, iters: int, early_stop_patience: int = -1) -> None:
        super().__init__(iters=iters)
        self._early_stop_patience = early_stop_patience
        self._last_best_iter = 0
        self._best_loss = torch.finfo(torch.float).max

    def select(self, iter_idx: int, mean_loss: float) -> Tuple[bool, bool]:
        saved, _ = super().select(iter_idx, mean_loss)
        should_stop = _should_early_stop(
            iter_idx,
            self._last_best_iter,
            self._early_stop_patience,
        )
        return saved, should_stop

    def _maybe_save(self, iter_idx: int, mean_loss: float) -> bool:
        if mean_loss < self._best_loss:
            self._best_loss = mean_loss
            self._last_best_iter = iter_idx
            self._save(iter_idx)
            return True
        return False

    def get_result(
        self,
        last_mean_loss: float,
        completed_iters: Optional[int] = None,
    ) -> BlockTrainResult:
        return BlockTrainResult(
            best_iter=self._last_best_iter,
            init_loss=self._init_loss if self._init_loss is not None else last_mean_loss,
            best_loss=self._best_loss,
            completed_iters=completed_iters if completed_iters is not None else self._iters,
        )


class LastIterSelector(CheckpointSelector):
    """Save init checkpoint at iter 0 and live params before the final optimizer step."""

    def __init__(self, iters: int, early_stop_patience: int = -1) -> None:
        super().__init__(iters=iters)
        _ = early_stop_patience

    def _maybe_save(self, iter_idx: int, mean_loss: float) -> bool:
        _ = mean_loss
        if iter_idx == self._iters - 1:
            self._save(iter_idx)
            return True
        return False

    def get_result(
        self,
        last_mean_loss: float,
        completed_iters: Optional[int] = None,
    ) -> BlockTrainResult:
        return BlockTrainResult(
            best_iter=self._iters - 1,
            init_loss=self._init_loss if self._init_loss is not None else last_mean_loss,
            best_loss=last_mean_loss,
            completed_iters=completed_iters if completed_iters is not None else self._iters,
        )


__all__ = [
    "BlockTrainResult",
    "CheckpointSelector",
    "EmaCheckpointSelector",
    "LastIterSelector",
    "MinLossCheckpointSelector",
    "save_best_params",
]
