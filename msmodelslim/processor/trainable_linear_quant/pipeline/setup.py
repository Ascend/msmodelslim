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

# Block linear wrapping and TLQ op installation.

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple, Type

from torch import nn

from msmodelslim.processor.anti_outlier.common.subgraph_type import Subgraph
from msmodelslim.core.quantizer.linear import LinearQConfig
from msmodelslim.utils.exception import UnsupportedError
from msmodelslim.utils.logging import get_logger

from msmodelslim.processor.trainable_linear_quant.core.ops import (
    TLQOp,
    TLQOpConfig,
    create_linear_tlq_op,
    create_subgraph_tlq_op,
    is_subgraph_op_config,
    load_tlq_op_class,
)
from msmodelslim.processor.trainable_linear_quant.core.subgraph import (
    resolve_subgraphs_for_op,
)
from msmodelslim.processor.trainable_linear_quant.core.wrapper import TrainableLinearQuantWrapper
from msmodelslim.processor.trainable_linear_quant.pipeline.resolve import StrategyResolver


def _subgraph_types_for_op_install(
    op_cls: Type[TLQOp],
    op_config: TLQOpConfig,
) -> Sequence[str]:
    """Subgraph types enabled for a subgraph TLQ op in the current block."""
    enabled = getattr(op_config, "enable_subgraph_type", None)
    if enabled is not None:
        return tuple(enabled)
    return tuple(getattr(op_cls, "SUPPORTED_SUBGRAPH_TYPES", ()))


def _linear_install_key(op_type: str, layer_path: str) -> str:
    return f"{op_type}:linear:{layer_path}"


def _subgraph_install_key(op_type: str, subgraph: Subgraph) -> str:
    return f"{op_type}:{type(subgraph).__name__}"


class OpInstallReporter:
    """Records TLQ op installation outcomes and surfaces failed binds to users."""

    def __init__(self) -> None:
        self._skip_reasons: List[Tuple[str, str]] = []
        self._installed = 0

    @property
    def installed(self) -> int:
        return self._installed

    @property
    def skipped(self) -> int:
        return len(self._skip_reasons)

    def record_installed(self) -> None:
        self._installed += 1

    def record_skip(self, key: str, reason: str) -> None:
        self._skip_reasons.append((key, reason))
        # Config / subgraph mismatches must be visible; debug-only hides YAML mistakes.
        get_logger().warning("Skipped TLQ op install for %s: %s", key, reason)

    def finish(self, ops: List[TLQOp], *, block_name: str = "") -> List[TLQOp]:
        skipped = len(self._skip_reasons)
        get_logger().info(
            "TLQ op binding completed: %d installed, %d skipped, %d total instances",
            self._installed,
            skipped,
            len(ops),
        )
        # Attempts existed but every bind failed → hard error (empty ops with wraps
        # from unmatched subgraph-only configs is OK: skipped stays 0).
        if self._installed == 0 and skipped > 0:
            preview = "; ".join(f"{key}: {reason}" for key, reason in self._skip_reasons[:5])
            extra = f" (+{skipped - 5} more)" if skipped > 5 else ""
            raise UnsupportedError(
                f"block {block_name!r}: all TLQ op installs failed "
                f"({skipped} skipped, 0 installed). Examples: {preview}{extra}"
            )
        return ops


class BlockSetup:
    """Wrap block linears and install configured TLQ ops."""

    def __init__(
        self,
        model: nn.Module,
        operation_configs: List[TLQOpConfig],
        layer_qconfigs: Dict[str, LinearQConfig],
        strategy_resolver: Optional[StrategyResolver] = None,
        train_with_act_quant: bool = False,
        global_adapter_configs: Optional[List] = None,
        adapter: Optional[object] = None,
    ) -> None:
        self._model = model
        self._operation_configs = list(operation_configs)
        self._layer_qconfigs = layer_qconfigs
        self._strategy_resolver = strategy_resolver
        self._train_with_act_quant = train_with_act_quant
        self._global_adapter_configs = global_adapter_configs or []
        self._adapter = adapter

    def _qconfig_for_layer(self, layer_name: str) -> Optional[LinearQConfig]:
        cached = self._layer_qconfigs.get(layer_name)
        if cached is not None:
            return cached
        if self._strategy_resolver is None:
            return None
        matched = self._strategy_resolver.match(layer_name)
        if matched is not None:
            self._layer_qconfigs[layer_name] = matched
        return matched

    def wrap_linears(
        self,
        block_name: str,
        block: nn.Module,
    ) -> Dict[str, TrainableLinearQuantWrapper]:
        wrappers_by_path: Dict[str, TrainableLinearQuantWrapper] = {}
        quantized_count = 0

        for layer_name, m in block.named_modules(prefix=block_name):
            if not isinstance(m, nn.Linear):
                continue
            qc = self._qconfig_for_layer(layer_name)
            if qc is None:
                continue
            wrapper = TrainableLinearQuantWrapper(
                m,
                linear_qconfig=qc,
                train_with_act_quant=self._train_with_act_quant,
            )
            wrapper.layer_path = layer_name
            self._model.set_submodule(layer_name, wrapper)
            wrappers_by_path[layer_name] = wrapper
            quantized_count += 1
            get_logger().debug("Layer '%s' wrapped for quantization", layer_name)

        get_logger().info(
            "Block wrapping completed: %d layers wrapped",
            quantized_count,
        )
        return wrappers_by_path

    def install_ops(
        self,
        block_name: str,
        block: nn.Module,
        wrappers_by_path: Dict[str, TrainableLinearQuantWrapper],
    ) -> List[TLQOp]:
        _ = block
        ops: List[TLQOp] = []
        reporter = OpInstallReporter()

        for op_config in self._operation_configs:
            op_cls = load_tlq_op_class(op_config)
            if is_subgraph_op_config(op_config):
                bindings = resolve_subgraphs_for_op(
                    op_cls,
                    op_config,
                    model=self._model,
                    block_name=block_name,
                    wrappers_by_path=wrappers_by_path,
                    global_adapter_configs=self._global_adapter_configs,
                    subgraph_types=_subgraph_types_for_op_install(op_cls, op_config),
                    adapter=self._adapter,
                )
                for subgraph, target_modules in bindings:
                    key = _subgraph_install_key(op_config.type, subgraph)
                    try:
                        op = create_subgraph_tlq_op(
                            op_config,
                            subgraph=subgraph,
                            target_modules=target_modules,
                        )
                        op.bind()
                        ops.append(op)
                        reporter.record_installed()
                    except UnsupportedError as exc:
                        reporter.record_skip(key, str(exc))
            else:
                for layer_path, wrapper in sorted(wrappers_by_path.items()):
                    key = _linear_install_key(op_config.type, layer_path)
                    try:
                        op = create_linear_tlq_op(
                            op_config,
                            layer_path=layer_path,
                            wrapper=wrapper,
                        )
                        op.bind()
                        ops.append(op)
                        reporter.record_installed()
                    except UnsupportedError as exc:
                        reporter.record_skip(key, str(exc))

        return reporter.finish(ops, block_name=block_name)


__all__ = [
    "BlockSetup",
    "OpInstallReporter",
]
