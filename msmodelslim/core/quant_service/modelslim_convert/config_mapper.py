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
    """权重张量名重命名规则：把匹配 `from` 的权重名改写为 `to`。"""

    model_config = ConfigDict(extra="forbid")

    from_: str = Field(alias="from", description="源权重名模式，支持通配符；匹配到的权重名将被改写。")
    to: str = Field(description="改写后的目标权重名模式。")


class RenamePreprocessConfig(BaseModel):
    """`modelslim_convert` 预处理步骤之一：批量重命名权重张量。"""

    model_config = ConfigDict(extra="forbid")

    type: Literal["rename"] = Field(default="rename", description="预处理类型，固定为 `rename`。")
    patterns: List[RenamePattern] = Field(default_factory=list, description="重命名规则列表，逐条应用到匹配的权重名。")


class ConvertOpConfig(BaseModel):
    """`convert` 预处理步骤中的权重算子，如拆分/合并 fused 的 gate/up 投影。"""

    model_config = ConfigDict(extra="allow")

    type: str = Field(description="算子类型：`chunk`（拆分 fused gate/up）、`merge`（合并 gate/up）或其他映射算子。")
    dim: Optional[int] = Field(
        default=None, description="拆分/合并维度：不指定时按算子类型自动推断，`chunk` 为 1，`merge` 为 0。"
    )
    projections: Optional[List[str]] = Field(
        default=None, description="`chunk` 拆出的投影名列表：不指定时自动推断为 `gate_proj`、`up_proj`。"
    )


class ConvertPreprocessConfig(BaseModel):
    """`modelslim_convert` 预处理步骤之一：对匹配的线性层做权重变换（拆分/合并等）。"""

    model_config = ConfigDict(extra="forbid")

    type: Literal["convert"] = Field(default="convert", description="预处理类型，固定为 `convert`。")
    source: List[str] = Field(default_factory=list, description="源权重名模式列表（待变换的线性层）。")
    target: List[str] = Field(default_factory=list, description="目标权重名模式列表（变换结果）。")
    ops: List[ConvertOpConfig] = Field(default_factory=list, description="权重变换算子列表，如 `chunk`、`merge`。")


class LinearConvertConfig(BaseModel):
    """指定匹配的线性层转换到目标 IR 的规则。"""

    model_config = ConfigDict(extra="forbid")

    match: List[str] = Field(default_factory=list, description="匹配的线性层名称模式列表。")
    target: IRKind = Field(description="转换目标 IR 类型，如 `W8A8_MXFP8`、`INT4_PACKED` 等。")
    route: Union[List[IRKind], Literal["auto"]] = Field(
        default="auto", description="转换路径：显式 IR 列表（首元素为源 IR），或 `auto` 由虚拟树按权重 dtype 推断。"
    )


class SaveConfig(BaseModel):
    """`modelslim_convert` 的保存格式配置。"""

    model_config = ConfigDict(extra="allow")

    type: str = Field(
        default="ascend_v1",
        description="保存格式：`ascend_v1`（昇腾，与 `ConvertDefaults.dst_format` 的 `ascendv1` 等价）；`compressed_tensors`（HF 兼容 safetensors）；`huggingface`/`hf` 是 `compressed_tensors` 的别名。",
    )
    part_file_size: int = Field(default=4, description="分片文件大小，单位 GB；0 表示不分片。")


class ParallelSpecConfig(BaseModel):
    """`modelslim_convert` 的并行执行配置。"""

    model_config = ConfigDict(extra="forbid")

    # workers=1：单进程组内线程（可配 NPU）；workers>1：组间多进程 + 组内线程（CPU，突破 GIL）
    workers: int = Field(
        default=1,
        description="并行 worker 数：1 表示单进程组内线程（可配 NPU）；大于1 表示组间多进程 + 组内线程（CPU）。",
    )
    # 单组最大任务数；超过则拆成多个子组分散到不同进程，缓解 MoE 大组拖尾
    max_group_size: Optional[int] = Field(
        default=None, description="单个依赖组的最大任务数，超过则拆成多个子组分散到不同进程；不设置表示不拆分。"
    )
    # 仅 workers=1 且 worker_device 指向 NPU 时生效
    worker_device: str = Field(default="cpu", description="worker 运行设备：`cpu` 或 `npu`。")
    npu_max_workers: int = Field(
        default=1, description="仅 `workers=1` 且 `worker_device=npu` 时生效，限制组内并发以防显存溢出。"
    )


class ModelslimConvertServiceConfig(BaseModel):
    """`modelslim_convert` 服务的 spec 结构。

    声明权重名重命名/变换（`preprocess`）、线性层转换规则（`linears`）、
    保存格式（`save`）、并行执行（`parallel`）与默认值（`defaults`）。
    """

    model_config = ConfigDict(extra="allow")

    preprocess: List[Dict[str, Any]] = Field(
        default_factory=list, description="预处理步骤列表，每项 `type` 为 `rename` 或 `convert`。"
    )
    linears: List[LinearConvertConfig] = Field(default_factory=list, description="线性层转换规则列表。")
    save: List[SaveConfig] = Field(default_factory=list, description="保存格式配置列表，取首个生效。")
    parallel: ParallelSpecConfig = Field(default_factory=ParallelSpecConfig, description="并行执行配置。")
    defaults: ConvertDefaults = Field(default_factory=ConvertDefaults, description="字段缺省时的全局默认值。")


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


def _resolve_part_file_size(save: List[SaveConfig]) -> int:
    """从 YAML ``spec.save[0].part_file_size`` 读取分片大小；未配置时默认 4GB。"""
    if save:
        return save[0].part_file_size
    return 4


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
        part_file_size=_resolve_part_file_size(spec.save),
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
