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

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

from transformers import PreTrainedTokenizerBase

from msmodelslim.core.train.protocol import (
    ITrainHook,
    TrainHookContext,
)
from msmodelslim.core.train.task import CoreTrainTask, CoreTrainTaskConfig
from msmodelslim.model import IModel
from msmodelslim.utils.exception import SchemaValidateError, UnsupportedError
from msmodelslim.utils.logging import get_logger


class RankWorker:
    """Per-rank pipeline: process group -> load -> dataset -> pre_train -> train."""

    @staticmethod
    def entry(
        task: CoreTrainTaskConfig,
        model_adapter: IModel,
        save_path: Union[str, Path],
        hooks: Sequence[ITrainHook],
    ) -> None:
        # task / hooks 已在 setup 侧组装；worker 只消费，不做 schema 二次校验与钩子重建。
        RankWorker(task, model_adapter, save_path, hooks).run()

    def __init__(
        self,
        task: CoreTrainTaskConfig,
        model_adapter: IModel,
        save_path: Union[str, Path],
        hooks: Sequence[ITrainHook],
    ):
        self.task_cfg = task
        self.core_task = CoreTrainTask(config=task, save_path=Path(save_path))
        self.model_adapter = model_adapter
        self.save_path = Path(save_path)
        self.hooks: List[ITrainHook] = list(hooks)
        self.model_max_length = int(task.train.hyperparameters.model_max_length)
        self.dist_info: Dict[str, Any] = {}
        self.tokenizer: Optional[PreTrainedTokenizerBase] = None
        self.model: Any = None
        self.train_dataset: Any = None
        self.eval_dataset: Any = None

    def run(self) -> None:
        self._assert_launcher_env()
        self.dist_info = self._init_process_group()
        self.core_task.dist_info = dict(self.dist_info)
        self._load_tokenizer()
        self._build_datasets()
        self._load_model()
        self._invoke_pre_train()
        self._train()
        get_logger().info(
            "RankWorker finished train: qparam_names=%s save_path=%s train_samples=%s eval_samples=%s",
            self.core_task.qparam_names,
            self.save_path,
            len(self.train_dataset) if self.train_dataset is not None else 0,
            len(self.eval_dataset) if self.eval_dataset is not None else 0,
        )

    def _fail_stage(self, stage: str, **context: Any) -> None:
        """按阶段记录失败上下文后重新抛出，便于定位卡在哪一步。"""
        parts = " ".join(f"{key}={value!r}" for key, value in context.items())
        get_logger().exception(
            "RankWorker failed at stage=%s%s",
            stage,
            f" {parts}" if parts else "",
        )

    def _assert_launcher_env(self) -> None:
        local_rank = os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0"))
        world_size = os.environ.get("WORLD_SIZE", "1")
        try:
            local_rank_i = int(local_rank)
            world_size_i = int(world_size)
        except (TypeError, ValueError) as exc:
            raise SchemaValidateError(
                f"Invalid distributed env LOCAL_RANK={local_rank!r} WORLD_SIZE={world_size!r}",
                action="Please launch QAT workers via accelerate so LOCAL_RANK / WORLD_SIZE are set.",
            ) from exc
        if world_size_i < 1 or local_rank_i < 0:
            raise SchemaValidateError(
                f"Invalid ranks: LOCAL_RANK={local_rank_i}, WORLD_SIZE={world_size_i}",
                action="Please check accelerate launch / device_id configuration.",
            )

    def _init_process_group(self) -> Dict[str, Any]:
        try:
            from accelerate import PartialState
        except ImportError as exc:
            raise UnsupportedError(
                "accelerate is required to initialize the QAT process group",
                action="Please install accelerate before running QAT training.",
            ) from exc

        import torch.distributed as dist

        state = PartialState()
        dist_info = {
            "local_rank": int(state.local_process_index),
            "process_index": int(state.process_index),
            "world_size": int(state.num_processes),
            "device": str(state.device),
            "dist_initialized": bool(dist.is_available() and dist.is_initialized()),
        }
        if dist_info["world_size"] > 1 and not dist_info["dist_initialized"]:
            raise UnsupportedError(
                f"Process group was not initialized after PartialState() (world_size={dist_info['world_size']})",
                action="Please launch via accelerate notebook_launcher / debug_launcher.",
            )
        get_logger().info(
            "Process group ready: local_rank=%s process_index=%s world_size=%s device=%s",
            dist_info["local_rank"],
            dist_info["process_index"],
            dist_info["world_size"],
            dist_info["device"],
        )
        return dist_info

    def _load_tokenizer(self) -> None:
        try:
            self.tokenizer = self.model_adapter.load_tokenizer_for_qat(
                model_max_length=self.model_max_length,
            )
        except Exception:
            self._fail_stage(
                "load_tokenizer",
                adapter=type(self.model_adapter).__name__,
                model_max_length=self.model_max_length,
                model_type=getattr(self.model_adapter, "model_type", None),
            )
            raise
        get_logger().info(
            "HF tokenizer loaded for QAT: adapter=%s tokenizer_max_length=%s",
            type(self.model_adapter).__name__,
            self.model_max_length,
        )

    def _load_model(self) -> None:
        try:
            self.model = self.model_adapter.load_model_for_qat()
            if getattr(self.model, "config", None) is not None:
                self.model.config.use_cache = False
        except Exception:
            self._fail_stage(
                "load_model",
                adapter=type(self.model_adapter).__name__,
                model_type=getattr(self.model_adapter, "model_type", None),
                save_path=str(self.save_path),
            )
            raise
        get_logger().info(
            "HF model loaded for QAT: adapter=%s model=%s",
            type(self.model_adapter).__name__,
            type(self.model).__name__,
        )

    def _build_datasets(self) -> None:
        # 训练数据构建模块按需懒加载（与 callback/trainer 同属 train 运行时依赖）。
        from msmodelslim.core.train.dataset_build import (  # pylint: disable=no-name-in-module
            build_qat_tokenized_datasets,
        )

        dataset_cfg = self.task_cfg.dataset
        world_size = int(self.dist_info.get("world_size") or 1)
        try:
            self.train_dataset, self.eval_dataset = build_qat_tokenized_datasets(
                path=dataset_cfg.path,
                ratio=float(dataset_cfg.ratio_eval_samples),
                tokenizer=self.tokenizer,
                max_length=self.model_max_length,
                num_proc=world_size,
            )
        except Exception:
            self._fail_stage(
                "build_datasets",
                dataset_path=dataset_cfg.path,
                ratio_eval_samples=dataset_cfg.ratio_eval_samples,
                model_max_length=self.model_max_length,
                world_size=world_size,
            )
            raise

    def _invoke_pre_train(self) -> None:
        if not self.hooks:
            raise SchemaValidateError(
                "RankWorker received empty pre_train hooks",
                action="Please assemble ITrainHook list in setup (e.g. QATTask.assemble_hooks) "
                "and pass them via TrainLauncher.",
            )
        ctx = TrainHookContext(
            model=self.model,
            core_task=self.core_task,
        )
        try:
            for hook in self.hooks:
                hook.pre_train(ctx)
            self.core_task.qparam_names = list(ctx.extra.get("qparam_names") or [])
        except Exception:
            self._fail_stage(
                "pre_train",
                hooks=[type(h).__name__ for h in self.hooks],
                model=type(self.model).__name__ if self.model is not None else None,
            )
            raise
        get_logger().info(
            "pre_train finished: hooks=%s qparam_names=%s",
            [type(h).__name__ for h in self.hooks],
            self.core_task.qparam_names,
        )

    def _train(self) -> None:
        # callback / trainer 按需懒加载，避免 import RankWorker 时强依赖重训练栈。
        from msmodelslim.core.train.callback import QATCallback  # pylint: disable=no-name-in-module
        from msmodelslim.core.train.trainer import (  # pylint: disable=no-name-in-module
            DEFAULT_QPARAM_NAMES,
            build_qat_trainer,
        )

        qparam_names = tuple(self.core_task.qparam_names) or DEFAULT_QPARAM_NAMES
        algorithm_cfg = []
        for algo in self.task_cfg.algorithm:
            if hasattr(algo, "model_dump"):
                algorithm_cfg.append(algo.model_dump(mode="python", by_alias=False))
            elif isinstance(algo, dict):
                algorithm_cfg.append(dict(algo))

        fake_quant_after_n_steps = getattr(
            self.task_cfg.train.hyperparameters,
            "fake_quant_after_n_steps",
            None,
        )
        try:
            trainer = build_qat_trainer(
                model=self.model,
                train_cfg=self.task_cfg.train,
                output_dir=str(self.save_path),
                train_dataset=self.train_dataset,
                eval_dataset=self.eval_dataset,
                tokenizer=self.tokenizer,
                qparam_names=qparam_names,
                algorithm_cfg=algorithm_cfg,
                callbacks=[QATCallback(fake_quant_after_n_steps=fake_quant_after_n_steps)],
            )
            source = self.task_cfg.source_model_path
            if source:
                trainer.source_model_path = str(source)

            get_logger().info(
                "===========QAT TRAIN LOOP=========== output_dir=%s qparam_names=%s",
                self.save_path,
                qparam_names,
            )
            trainer.train()
        except Exception:
            self._fail_stage(
                "train",
                save_path=str(self.save_path),
                qparam_names=list(qparam_names),
                train_samples=len(self.train_dataset) if self.train_dataset is not None else 0,
                eval_samples=len(self.eval_dataset) if self.eval_dataset is not None else 0,
            )
            raise


rank_worker_entry = RankWorker.entry
