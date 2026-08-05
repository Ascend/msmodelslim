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

# Subgraph TLQ op install: resolve adapter subgraphs to wrapped targets.

from __future__ import annotations

from typing import AbstractSet, Dict, Iterator, List, Optional, Sequence, Tuple, Type, TYPE_CHECKING

from torch import nn

from msmodelslim.core.graph.adapter_filter import filter_adapter_configs_for_processor
from msmodelslim.core.graph.adapter_types import AdapterConfig
from msmodelslim.core.graph.subgraph_builder import build_subgraph_from_adapter
from msmodelslim.processor.anti_outlier.common.subgraph_type import (
    LinearLinearSubgraph,
    NonFusionSubgraph,
    NormLinearSubgraph,
    OVSubgraph,
    Subgraph,
    UpDownSubgraph,
)
from msmodelslim.utils.exception import UnsupportedError
from msmodelslim.utils.logging import get_logger

from msmodelslim.processor.trainable_linear_quant.core.wrapper import TrainableLinearQuantWrapper

if TYPE_CHECKING:
    from msmodelslim.processor.trainable_linear_quant.core.ops.base import (
        TLQOp,
        TLQOpConfig,
    )

SubgraphBinding = Tuple[Subgraph, Dict[str, TrainableLinearQuantWrapper]]

__all__ = [
    "SubgraphBinding",
    "resolve_subgraphs_for_op",
]


def _iter_subgraph_target_modules(
    subgraph: Subgraph,
) -> Iterator[Tuple[str, nn.Module]]:
    """遍历子图中作为量化/训练目标的 ``(layer_path, module)``。"""
    if isinstance(subgraph, (NonFusionSubgraph, NormLinearSubgraph)):
        names = subgraph.linear_names or []
        for idx, linear in enumerate(subgraph.linears):
            yield (names[idx] if idx < len(names) else ""), linear
        return
    if isinstance(subgraph, LinearLinearSubgraph):
        yield subgraph.linear2_name or "", subgraph.linear2
        return
    if isinstance(subgraph, OVSubgraph):
        yield subgraph.o_proj_name or "", subgraph.o_proj
        return
    if isinstance(subgraph, UpDownSubgraph):
        yield subgraph.down_proj_name or "", subgraph.down_proj
        return
    raise UnsupportedError(f"unsupported subgraph type: {type(subgraph).__name__}")


def _wrappers_from_subgraph(subgraph: Subgraph) -> Dict[str, TrainableLinearQuantWrapper]:
    """从子图 target 字段收集 ``layer_path -> wrapper``（须已在 block 内 wrap）。"""
    result: Dict[str, TrainableLinearQuantWrapper] = {}
    for layer_path, module in _iter_subgraph_target_modules(subgraph):
        if not isinstance(module, TrainableLinearQuantWrapper):
            path = layer_path or getattr(module, "layer_path", "")
            raise UnsupportedError(
                f"TLQ target {path!r} must be TrainableLinearQuantWrapper after block wrap, got {type(module).__name__}"
            )
        path = layer_path or module.layer_path
        if not path:
            raise UnsupportedError(f"cannot resolve layer path for wrapper in {type(subgraph).__name__}")
        result[path] = module
    if not result:
        raise UnsupportedError(f"no TLQ targets in subgraph {type(subgraph).__name__}")
    return result


def _subgraph_dedup_key(cfg: AdapterConfig) -> str:
    targets = "+".join(sorted(cfg.mapping.targets))
    return f"{cfg.subgraph_type}:{cfg.mapping.source or 'none'}->{targets}"


def _resolve_wrapped_subgraph(
    model: nn.Module,
    cfg: AdapterConfig,
    wrapped_paths: AbstractSet[str],
    adapter: Optional[object] = None,
) -> SubgraphBinding:
    """wrap 后由 adapter 建子图；校验 target 均在 ``wrapped_paths`` 且为 wrapper。"""
    if not cfg.mapping.targets or not any(t in wrapped_paths for t in cfg.mapping.targets):
        raise UnsupportedError(f"adapter targets not wrapped in block: {cfg}")
    built = build_subgraph_from_adapter(model, cfg, adapter=adapter)
    if built is None:
        raise UnsupportedError(f"failed to build subgraph from adapter: {cfg}")
    target_modules = _wrappers_from_subgraph(built.subgraph)
    if not target_modules.keys() <= set(wrapped_paths):
        extra = set(target_modules) - set(wrapped_paths)
        raise UnsupportedError(f"subgraph targets not in wrapped_paths: {extra}")
    return built.subgraph, target_modules


def _iter_resolved_subgraphs(
    subgraph_type: str,
    model: nn.Module,
    wrapped_paths: AbstractSet[str],
    adapter_configs: Sequence[AdapterConfig],
    adapter: Optional[object] = None,
) -> Iterator[Tuple[SubgraphBinding, str]]:
    for adapter_cfg in adapter_configs:
        if adapter_cfg.subgraph_type != subgraph_type:
            continue
        if not any(p in wrapped_paths for p in adapter_cfg.mapping.targets):
            continue
        dedup_key = _subgraph_dedup_key(adapter_cfg)
        try:
            binding = _resolve_wrapped_subgraph(model, adapter_cfg, wrapped_paths, adapter=adapter)
        except UnsupportedError as exc:
            get_logger().warning(
                "Skipped subgraph resolve for %s: %s",
                dedup_key,
                exc,
            )
            continue
        yield binding, dedup_key


def resolve_subgraphs_for_op(
    op_cls: Type["TLQOp"],
    op_config: "TLQOpConfig",
    model: nn.Module,
    block_name: str,
    wrappers_by_path: Dict[str, TrainableLinearQuantWrapper],
    global_adapter_configs: Sequence[AdapterConfig],
    subgraph_types: Sequence[str],
    adapter: Optional[object] = None,
) -> List[SubgraphBinding]:
    """Filter adapter configs and resolve subgraphs with wrapped targets for one subgraph op."""
    if not hasattr(op_config, "enable_subgraph_type"):
        return []

    adapter_configs = filter_adapter_configs_for_processor(
        global_adapter_configs,
        op_config,
        block_name,
    )
    wrapped_paths = set(wrappers_by_path)
    bindings: List[SubgraphBinding] = []
    seen: set[str] = set()

    for subgraph_type in subgraph_types:
        for binding, dedup_key in _iter_resolved_subgraphs(
            subgraph_type,
            model=model,
            wrapped_paths=wrapped_paths,
            adapter_configs=adapter_configs,
            adapter=adapter,
        ):
            if dedup_key in seen:
                get_logger().debug("Skipped duplicate subgraph binding: %s", dedup_key)
                continue
            seen.add(dedup_key)
            bindings.append(binding)

    get_logger().debug(
        "Resolved %d subgraph bindings for op %s in block %s (types=%s)",
        len(bindings),
        op_config.type,
        block_name,
        list(subgraph_types),
    )
    return bindings
