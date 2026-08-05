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

# Thin block-level trainer orchestration for trainable linear quant.

from __future__ import annotations

import copy
import logging
from collections import defaultdict
from typing import Any, Dict, List, Sequence, Set, Tuple, Union

import torch
from torch import nn

from msmodelslim.processor.trainable_linear_quant.config.train_config import BlockTrainConfig
from msmodelslim.processor.trainable_linear_quant.data import (
    BlockInput,
    BlockOutput,
    TLQBlockDataInterface,
)
from msmodelslim.utils.logging import get_logger
from .sign_sgd import SignSGD
from .checkpoint_selector import (
    BlockTrainResult,
    CheckpointSelector,
    save_best_params,
)
from .loss_evaluator import BlockLossEvaluator
from msmodelslim.processor.trainable_linear_quant.core.ops.base import TLQOp, format_tensor_dbg
from msmodelslim.utils.seed import seed_all

_BATCH_SIZE = 1


class TrainableLinearQuantBlockTrainer:
    """块级量化参数优化训练（trainable linear quant 路径）。"""

    def __init__(
        self,
        config: BlockTrainConfig,
        block_data: TLQBlockDataInterface,
        lr_scheduler=None,
    ) -> None:
        if block_data is None:
            raise ValueError("block_data is required for TrainableLinearQuantBlockTrainer")
        self._config = config
        self._lr_scheduler = lr_scheduler
        self._loss_evaluator = BlockLossEvaluator.from_config(
            block_data,
            config,
            batch_size=_BATCH_SIZE,
        )

    @property
    def config(self) -> BlockTrainConfig:
        return self._config

    @staticmethod
    def get_optimizer():
        return SignSGD

    @staticmethod
    def step(optimizer, lr_schedule) -> None:
        optimizer.step()
        optimizer.zero_grad()
        lr_schedule.step()

    def _handle_skipped_training(
        self,
        tlq_ops: Sequence[TLQOp],
        block_name: str,
    ) -> BlockTrainResult:
        save_best_params(tlq_ops)
        get_logger().info(
            "block %s: training skipped (iters=0), using initialized parameters",
            block_name,
        )
        return BlockTrainResult(
            best_iter=0,
            init_loss=0.0,
            best_loss=0.0,
            completed_iters=0,
        )

    def _build_optimizer_param_groups(
        self,
        ops: Sequence[TLQOp],
    ) -> List[Dict[str, Any]]:
        """Group TLQ op trainable parameters by learning rate for SignSGD."""
        global_lr = self._config.lr
        buckets: Dict[float, List[torch.nn.Parameter]] = defaultdict(list)
        seen: Set[int] = set()
        for op in ops:
            lr = getattr(op.config, "lr", None) or global_lr
            for p in op.train_params.values():
                pid = id(p)
                if pid in seen:
                    continue
                seen.add(pid)
                buckets[float(lr)].append(p)
        if not buckets:
            return [{"params": []}]
        if len(buckets) == 1:
            lr = next(iter(buckets))
            params = buckets[lr]
            if abs(lr - global_lr) < 1e-12:
                groups: List[Dict[str, Any]] = [{"params": params}]
            else:
                groups = [{"params": params, "lr": lr}]
        else:
            groups = [{"params": buckets[lr], "lr": lr} for lr in sorted(buckets)]
        return groups

    def _create_optimizer_and_schedule(
        self,
        tlq_ops: Sequence[TLQOp],
    ) -> Tuple[Any, Any]:
        optimizer = self.get_optimizer()(
            self._build_optimizer_param_groups(tlq_ops),
            lr=self._config.lr,
        )
        if self._lr_scheduler is None:
            lr_schedule = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=1.0,
                end_factor=0.0,
                total_iters=self._config.iters,
            )
        else:
            lr_schedule = copy.deepcopy(self._lr_scheduler)
        return optimizer, lr_schedule

    @torch.no_grad()
    def _log_ops_param_dbg(
        self,
        block_name: str,
        iter_idx: int,
        tlq_ops: Sequence[TLQOp],
        phase: str,
    ) -> None:
        """Per-op trainable param (/grad) snapshot; only emitted at DEBUG level."""
        # logger.debug 会先求值实参；非 DEBUG 时跳过，避免每 iter 做张量统计。
        if not get_logger().isEnabledFor(logging.DEBUG):
            return
        with_grad = phase == "pre_step"
        for op in tlq_ops:
            fields = [
                f"{key}({format_tensor_dbg(tensor, include_abs_mean=True, include_grad=with_grad)})"
                for key, tensor in op.train_params.items()
                if isinstance(tensor, torch.Tensor)
            ]
            if not fields:
                continue
            get_logger().debug(
                "block %s iter %d [%s] op %s: %s",
                block_name,
                iter_idx,
                phase,
                op.op_id,
                " ".join(fields),
            )

    @torch.enable_grad()
    def train_block(
        self,
        block: nn.Module,
        all_datas: List[BlockInput],
        float_output: List[BlockOutput],
        device: Union[str, torch.device],
        tlq_ops: Sequence[TLQOp],
        block_name: str,
    ) -> BlockTrainResult:
        seed_all(self._config.train_seed)
        if self._config.iters == 0:
            return self._handle_skipped_training(tlq_ops, block_name)

        gas = max(self._config.gradient_accumulate_steps, 1)
        nsamples = len(all_datas)
        samples_per_iter = min(nsamples, _BATCH_SIZE * gas)
        planned_iters = self._config.iters
        selector = CheckpointSelector.from_config(self._config)
        optimizer, lr_schedule = self._create_optimizer_and_schedule(tlq_ops)

        def train_one_iter(iter_idx: int) -> Tuple[float, bool]:
            indices = torch.randperm(nsamples)[:samples_per_iter]

            if iter_idx == 0:
                optimizer.zero_grad(set_to_none=True)

            mean_loss = self._loss_evaluator.eval_and_backward(
                block,
                all_datas,
                float_output,
                device,
                indices,
            )

            should_save, should_stop = selector.select(iter_idx, mean_loss)
            get_logger().debug(
                "block %s iter %d: loss=%.6f avg=%.6f save=%s stop=%s",
                block_name,
                iter_idx,
                mean_loss,
                selector.last_avg_loss,
                should_save,
                should_stop,
            )
            # DEBUG：step 前看参数+梯度，确认各 Op 是否收到有效 grad。
            self._log_ops_param_dbg(block_name, iter_idx, tlq_ops, phase="pre_step")
            if should_save:
                save_best_params(tlq_ops)

            if should_stop:
                optimizer.zero_grad(set_to_none=True)
                return mean_loss, True

            self.step(optimizer, lr_schedule)
            # DEBUG：step 后看参数是否被 SignSGD 更新。
            self._log_ops_param_dbg(block_name, iter_idx, tlq_ops, phase="post_step")
            return mean_loss, False

        last_mean_loss = 0.0
        completed_iters = 0
        for i in range(planned_iters):
            last_mean_loss, early_stopped = train_one_iter(i)
            completed_iters = i + 1
            if early_stopped:
                break

        result = selector.get_result(
            last_mean_loss=last_mean_loss,
            completed_iters=completed_iters,
        )
        get_logger().info(
            "Training completed for block %s: loss iter 0=%.6f -> iter %d=%.6f",
            block_name,
            result.init_loss,
            result.best_iter,
            result.best_loss,
        )
        if 0 < result.completed_iters < planned_iters:
            get_logger().info(
                "block %s early stopped after %d iters (last saved iter %d)",
                block_name,
                result.completed_iters,
                selector.last_saved_iter,
            )
        return result


__all__ = ["TrainableLinearQuantBlockTrainer"]
