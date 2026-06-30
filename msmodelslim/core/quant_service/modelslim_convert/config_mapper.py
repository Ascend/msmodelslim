#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
将 ``apiversion: modelslim_convert`` 的 spec 映射为 ``ConvertConfig``。

新 YAML spec 字段：
  - preprocess: rename / convert（chunk、merge）
  - linears: 匹配线性层并指定 target IR 与 route
  - save: 落盘格式（ascend_v1 等）
  - parallel: 并行参数
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Tuple, Union

from pydantic import BaseModel, ConfigDict, Field

from msmodelslim.core.convert.config import (
    ConvertConfig,
    ConvertDefaults,
    ConvertRule,
    ModuleRule,
    ParallelConfig,
    WeightMappingRule,
    WeightOpConfig,
)
from msmodelslim.core.convert.types import IRKind


class RenamePattern(BaseModel):
    model_config = ConfigDict(extra="forbid")

    from_: str = Field(alias="from")
    to: str


class RenamePreprocessConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["rename"] = "rename"
    patterns: List[RenamePattern] = Field(default_factory=list)


class ConvertOpConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: str
    dim: Optional[int] = None
    projections: Optional[List[str]] = None


class ConvertPreprocessConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["convert"] = "convert"
    source: List[str] = Field(default_factory=list)
    target: List[str] = Field(default_factory=list)
    ops: List[ConvertOpConfig] = Field(default_factory=list)


class LinearConvertConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    match: List[str] = Field(default_factory=list)
    target: IRKind
    route: Union[List[IRKind], Literal["auto"]] = "auto"


class SaveConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: str = "ascend_v1"
    part_file_size: int = 4


class ParallelSpecConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # workers=1：单进程组内线程（可配 NPU）；workers>1：组间多进程 + 组内线程（CPU，突破 GIL）
    workers: int = 1
    # 单组最大任务数；超过则拆成多个子组分散到不同进程，缓解 MoE 大组拖尾
    max_group_size: Optional[int] = None
    # 仅 workers=1 且 worker_device 指向 NPU 时生效
    worker_device: str = "cpu"
    npu_max_workers: int = 1


class ModelslimConvertServiceConfig(BaseModel):
    """modelslim_convert quant_service 的 spec 结构。"""

    model_config = ConfigDict(extra="allow")

    preprocess: List[Dict[str, Any]] = Field(default_factory=list)
    linears: List[LinearConvertConfig] = Field(default_factory=list)
    save: List[SaveConfig] = Field(default_factory=list)
    parallel: ParallelSpecConfig = Field(default_factory=ParallelSpecConfig)
    defaults: ConvertDefaults = Field(default_factory=ConvertDefaults)


_SAVE_TYPE_MAP = {
    "ascend_v1": "ascendv1",
    "ascendv1": "ascendv1",
    "ascendv1_saver": "ascendv1",
    "huggingface": "huggingface",
    "hf": "huggingface",
    "compressed_tensors": "compressed_tensors",
}


# 固定策略：同 shard / fused 依赖的任务分组，组内共享 shard 句柄与 fused 缓存。
_DEFAULT_TASK_GRANULARITY = "dependency_group"
_DEFAULT_SHARD_CACHE_SIZE = 1
_DEFAULT_WORKER_THREADS = 4


def _preprocess_to_rules(spec: ModelslimConvertServiceConfig) -> List[WeightMappingRule]:
    rules: List[WeightMappingRule] = []
    for idx, raw in enumerate(spec.preprocess):
        ptype = raw.get("type")
        if ptype == "rename":
            cfg = RenamePreprocessConfig.model_validate(raw)
            for pat_idx, pat in enumerate(cfg.patterns):
                rules.append(
                    WeightMappingRule(
                        id=f"rename_{idx}_{pat_idx}",
                        source_patterns=[pat.from_],
                        target_patterns=[pat.to],
                        ops=[WeightOpConfig(type="rename")],
                    ),
                )
        elif ptype == "convert":
            cfg = ConvertPreprocessConfig.model_validate(raw)
            ops = _map_convert_ops(cfg.ops)
            rules.append(
                WeightMappingRule(
                    id=f"convert_{idx}",
                    source_patterns=list(cfg.source),
                    target_patterns=list(cfg.target),
                    ops=ops,
                    module_kind="linear",
                    reversible=True,
                ),
            )
        else:
            raise ValueError(f"Unsupported preprocess type: {ptype!r}")
    return rules


def _map_convert_ops(ops: List[ConvertOpConfig]) -> List[WeightOpConfig]:
    mapped: List[WeightOpConfig] = []
    for op in ops:
        if op.type == "chunk":
            mapped.append(
                WeightOpConfig(
                    type="split_fused_gate_up",
                    params={
                        "split_dim": op.dim if op.dim is not None else 1,
                        "projections": op.projections or ["gate_proj", "up_proj"],
                    },
                ),
            )
        elif op.type == "merge":
            mapped.append(
                WeightOpConfig(
                    type="merge_gate_up",
                    params={"split_dim": op.dim if op.dim is not None else 0},
                ),
            )
        else:
            mapped.append(WeightOpConfig(type=op.type, params=op.model_dump(exclude={"type"})))
    return mapped


# 源 IR -> (source_format, 额外 tensor 绑定)。决定虚拟树如何绑定权重并供 router 选路。
_SOURCE_IR_BINDINGS: Dict[IRKind, Tuple[str, Dict[str, str]]] = {
    IRKind.FP8_BLOCK: (
        "fp8_block",
        {"weight": "{module}.weight", "weight_scale_inv": "{module}.weight_scale_inv"},
    ),
    IRKind.INT4_PACKED: (
        "int4_per_group",
        {
            "weight_packed": "{module}.weight_packed",
            "weight_scale": "{module}.weight_scale",
            "weight_shape": "{module}.weight_shape",
            "bias": "{module}.bias",
        },
    ),
    IRKind.FLOAT: ("bf16", {"weight": "{module}.weight"}),
}


def _infer_source_ir(route: Union[List[IRKind], str]) -> Optional[IRKind]:
    """显式 route 的首元素即源 IR；route=auto 时由虚拟树按 catalog dtype 推断。"""
    if route == "auto":
        return None
    if route:
        return route[0]
    return IRKind.FLOAT


def _module_rule_fields_for_route(
    route: Union[List[IRKind], str],
) -> Tuple[Optional[str], Optional[IRKind], Dict[str, str]]:
    """Return (source_format, source_ir, tensor_map) for a linear route spec."""
    source_ir = _infer_source_ir(route)
    if source_ir is None:
        return (
            None,
            None,
            {
                "weight": "{module}.weight",
                "weight_scale_inv": "{module}.weight_scale_inv",
            },
        )
    source_format, tensor_map = _SOURCE_IR_BINDINGS.get(source_ir, _SOURCE_IR_BINDINGS[IRKind.FLOAT])
    return source_format, source_ir, dict(tensor_map)


def _linears_to_module_and_convert_rules(
    linears: List[LinearConvertConfig],
) -> Tuple[List[ModuleRule], List[ConvertRule]]:
    module_rules: List[ModuleRule] = []
    convert_rules: List[ConvertRule] = []
    for linear in linears:
        source_format, source_ir, tensor_map = _module_rule_fields_for_route(linear.route)
        for pattern in linear.match:
            module_rules.append(
                ModuleRule(
                    match=pattern,
                    module_kind="linear",
                    source_format=source_format,
                    source_ir=source_ir,
                    tensor_map=dict(tensor_map),
                ),
            )
            convert_rules.append(
                ConvertRule(
                    match=pattern,
                    target_ir=linear.target,
                    route=linear.route,
                ),
            )
    return module_rules, convert_rules


def _resolve_dst_format(save: List[SaveConfig], defaults: ConvertDefaults) -> str:
    if save:
        return _SAVE_TYPE_MAP.get(save[0].type.lower(), save[0].type.lower())
    return defaults.dst_format


def spec_to_convert_config(
    spec: Union[ModelslimConvertServiceConfig, Dict[str, Any]],
    model_path: str,
    save_path: str,
    model_family: Optional[str] = None,
) -> ConvertConfig:
    """将 quant spec 转为可执行的 ``ConvertConfig``。"""
    if not isinstance(spec, ModelslimConvertServiceConfig):
        spec = ModelslimConvertServiceConfig.model_validate(spec)

    module_rules, convert_rules = _linears_to_module_and_convert_rules(spec.linears)
    parallel = ParallelConfig(
        max_workers=spec.parallel.workers,
        task_granularity=_DEFAULT_TASK_GRANULARITY,
        worker_backend="process" if spec.parallel.workers > 1 else "thread",
        worker_threads=_DEFAULT_WORKER_THREADS,
        max_group_size=spec.parallel.max_group_size,
        shard_cache_size=_DEFAULT_SHARD_CACHE_SIZE,
        worker_device=spec.parallel.worker_device,
        npu_max_workers=spec.parallel.npu_max_workers,
    )

    return ConvertConfig(
        model_path=model_path,
        save_path=save_path,
        model_family=model_family,
        dst_format=_resolve_dst_format(spec.save, spec.defaults),
        defaults=spec.defaults,
        preprocess_rules=_preprocess_to_rules(spec),
        module_rules=module_rules,
        convert_rules=convert_rules,
        parallel=parallel,
    )


def load_specific_config(yaml_spec: object) -> ModelslimConvertServiceConfig:
    """从 YAML spec 加载 modelslim_convert 配置。"""
    if isinstance(yaml_spec, ModelslimConvertServiceConfig):
        return yaml_spec
    if not isinstance(yaml_spec, dict):
        raise ValueError("task spec must be dict")
    return ModelslimConvertServiceConfig.model_validate(yaml_spec)
