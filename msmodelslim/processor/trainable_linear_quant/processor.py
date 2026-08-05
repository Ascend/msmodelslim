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

import json
from typing import Dict, List, Optional, Sequence

import torch
from torch import nn

from msmodelslim.core.base.protocol import BatchProcessRequest
from msmodelslim.core.quantizer.linear import LinearQConfig
from msmodelslim.ir.qal import QABCRegistry
from msmodelslim.processor.base import AutoSessionProcessor
from msmodelslim.utils.exception import SchemaValidateError
from msmodelslim.utils.logging import get_logger, logger_setter

from .config import TrainableLinearQuantProcessorConfig
from .core.ops.base import operations_need_adapter_subgraph
from .pipeline.resolve import StrategyResolver
from .data import (
    BlockOutput,
    TLQBlockDataInterface,
    propagate_outputs_to_inputs,
    resolve_tlq_block_data_interface,
)
from .pipeline.finalize import finalize_block
from .pipeline.runtime import (
    BlockTLQContext,
    capture_float_teacher,
    capture_quant_propagation,
    log_npu_mem,
)
from .core.train import TrainableLinearQuantBlockTrainer
from .pipeline.setup import BlockSetup


@QABCRegistry.register(dispatch_key=TrainableLinearQuantProcessorConfig, abc_class=AutoSessionProcessor)
@logger_setter(prefix="msmodelslim.processor.trainable_linear_quant")
class TrainableLinearQuantProcessor(AutoSessionProcessor):
    def __init__(
        self,
        model: nn.Module,
        config: TrainableLinearQuantProcessorConfig,
        adapter: Optional[object] = None,
    ) -> None:
        super().__init__(model)
        self.model = model
        self.config = config
        self.adapter = adapter

        self.train_config = config.train_config
        self._train_operation_configs = list(config.operations)

        self.layer_qconfigs: Dict[str, LinearQConfig] = {}
        self._strategy_resolver: Optional[StrategyResolver] = None

        self._sessions: Dict[str, BlockTLQContext] = {}
        self._propagation_outputs: Optional[List[BlockOutput]] = None
        self._global_adapter_configs: List = []
        self._block_setup: Optional[BlockSetup] = None
        self._block_data: TLQBlockDataInterface = resolve_tlq_block_data_interface(adapter)

    def support_distributed(self) -> bool:
        return False

    def pre_run(self) -> None:
        get_logger().info(
            "TLQ config: %s",
            json.dumps(
                self.config.model_dump(mode="json", exclude_none=True),
                ensure_ascii=False,
                separators=(",", ":"),
            ),
        )

        for _, param in self.model.named_parameters():
            param.requires_grad_(False)

        if operations_need_adapter_subgraph(self._train_operation_configs):
            self._global_adapter_configs = self._load_global_adapter_configs()
        else:
            self._global_adapter_configs = []

        # Keep strategies for per-block lazy match: layer-wise loaders only
        # materialize a subset of modules at pre_run (e.g. Qwen3.5 → layers.0).
        self._strategy_resolver = StrategyResolver(self.config.strategies)
        self.layer_qconfigs = {}
        get_logger().info(
            "TLQ strategies ready (%d); layer qconfigs resolve on each block wrap",
            len(self.config.strategies),
        )

        self._block_setup = BlockSetup(
            model=self.model,
            operation_configs=self._train_operation_configs,
            layer_qconfigs=self.layer_qconfigs,
            strategy_resolver=self._strategy_resolver,
            train_with_act_quant=self.config.train_with_act_quant,
            global_adapter_configs=self._global_adapter_configs,
            adapter=self.adapter,
        )
        get_logger().debug(
            "Loaded %d global adapter subgraph configs",
            len(self._global_adapter_configs),
        )

    def install(
        self,
        request: BatchProcessRequest,
        teacher_outputs: Optional[Sequence[BlockOutput]] = None,
    ) -> BlockTLQContext:
        """Wrap linears, bind ops, and store per-block session. No training / unwrap."""
        if self._block_setup is None:
            raise RuntimeError("BlockSetup not initialized; run pre_run first")

        param_device = next(request.module.parameters()).device
        ctx = BlockTLQContext(
            block_name=request.name,
            device=param_device,
        )
        if teacher_outputs is None:
            teacher_outputs = capture_float_teacher(request)
        ctx.teacher_outputs = list(teacher_outputs)
        get_logger().info(
            "block %s: captured %d float teacher outputs",
            request.name,
            len(ctx.teacher_outputs),
        )

        with torch.device(device=param_device):
            ctx.wrappers_by_path = self._block_setup.wrap_linears(request.name, request.module)
            ctx.ops = self._block_setup.install_ops(request.name, request.module, ctx.wrappers_by_path)
        log_npu_mem("AFTER_WRAPPING")

        self._sessions[request.name] = ctx
        return ctx

    def train_block(self, request: BatchProcessRequest) -> None:
        """Run TLQ training for the installed block session."""
        ctx = self._get_session(request.name)

        if self.config.enable_quanted_input and self._propagation_outputs is not None:
            self._inject_propagation_inputs(request)

        # Strategies may exclude every Linear in this block (e.g. model.visual);
        # skip training instead of failing require_ops().
        if not ctx.ops:
            get_logger().info(
                "block %s: no TLQ ops installed, skip training",
                request.name,
            )
            return

        log_npu_mem("BEFORE_TRAINING")
        ops = ctx.require_ops()
        trainer = TrainableLinearQuantBlockTrainer(self.train_config, self._block_data)
        all_datas = request.datas
        get_logger().info(
            "block %s: starting TLQ training (%d iters, %d samples)",
            request.name,
            self.train_config.iters,
            len(all_datas),
        )
        ctx.train_result = trainer.train_block(
            block=request.module,
            all_datas=all_datas,
            float_output=ctx.teacher_outputs,
            device=ctx.device,
            tlq_ops=ops,
            block_name=request.name,
        )

    def finalize(self, request: BatchProcessRequest) -> None:
        """Load best params, unwrap wrappers, export IR."""
        ctx = self._get_session(request.name)

        if ctx.ops:
            get_logger().debug(
                "block %s: applying best parameters and unwrapping",
                request.name,
            )
            with torch.no_grad(), torch.device(device=ctx.device):
                finalize_block(
                    request.name,
                    request.module,
                    ctx,
                    model=self.model,
                )
        else:
            get_logger().info(
                "block %s: no TLQ ops installed, skip finalize",
                request.name,
            )

    def release(self, request: BatchProcessRequest) -> None:
        """Drop per-block session state."""
        ctx = self._sessions.pop(request.name, None)
        if ctx is None:
            return
        ctx.release()
        get_logger().debug("block %s: released TLQ session", request.name)

    def preprocess(self, request: BatchProcessRequest) -> None:
        self.install(request)

    def process(self, request: BatchProcessRequest) -> None:
        self.train_block(request)

    def postprocess(self, request: BatchProcessRequest) -> None:
        ctx = self._get_session(request.name)

        if self.config.enable_quanted_input:
            if self._propagation_outputs is not None:
                self._inject_propagation_inputs(request)
            if ctx.ops:
                self._load_best_op_params(ctx)
            self._propagation_outputs = capture_quant_propagation(request)

        self.finalize(request)

        # 始终把浮点 teacher 交给 Runner，保证下一层 preprocess 采 teacher 仍用 FP 输入。
        request.outputs = list(ctx.teacher_outputs)
        self.release(request)

    def post_run(self) -> None:
        if self._strategy_resolver is not None:
            self._strategy_resolver.warn_unmatched()
            get_logger().info(
                "TLQ session done: %d linear layers received a qconfig",
                len(self.layer_qconfigs),
            )
        self._sessions.clear()
        self._propagation_outputs = None
        self.layer_qconfigs = {}
        self._strategy_resolver = None
        self._block_setup = None

    def _load_best_op_params(self, ctx: BlockTLQContext) -> None:
        for op in ctx.require_ops():
            op.load_best_params()

    def _inject_propagation_inputs(self, request: BatchProcessRequest) -> None:
        """将上一层量化前向结果注入本层 datas（仅影响训练 / 量化传播，不影响 teacher）。"""
        get_logger().debug("block %s: inject propagation inputs", request.name)
        propagate_outputs_to_inputs(
            self._block_data,
            request.datas,
            self._propagation_outputs,
        )

    def _load_global_adapter_configs(self) -> list:
        action = (
            "When trainable_smooth (or another subgraph op) is enabled, provide a model "
            "adapter with get_adapter_config_for_subgraph returning a non-empty List[AdapterConfig]"
        )
        adapter = self.adapter
        if adapter is None or not hasattr(adapter, "get_adapter_config_for_subgraph"):
            raise SchemaValidateError(
                "subgraph TLQ operations require get_adapter_config_for_subgraph on the model adapter",
                action=action,
            )
        configs = adapter.get_adapter_config_for_subgraph()
        if not isinstance(configs, list) or not configs:
            got = "empty list" if isinstance(configs, list) else type(configs).__name__
            raise SchemaValidateError(
                f"get_adapter_config_for_subgraph must return a non-empty list, got {got}",
                action=action,
            )
        return configs

    def _get_session(self, block_name: str) -> BlockTLQContext:
        ctx = self._sessions.get(block_name)
        if ctx is None:
            raise RuntimeError(f"no TLQ session for block {block_name!r}; run preprocess first")
        return ctx
