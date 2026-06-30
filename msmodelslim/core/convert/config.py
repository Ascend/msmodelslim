#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
Pydantic configuration schemas for ``msmodelslim convert``.

Three rule families are kept separate on purpose:
  - ``preprocess_rules``: structural changes to the weight map
  - ``module_rules``: virtual tree construction and tensor bindings
  - ``convert_rules``: per-layer target IR and routing constraints
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from msmodelslim.core.convert.types import IRKind

# 产品约束：W8A8_MXFP8 权重仅在昇腾 NPU 上运行，落盘须走 AscendV1，不用 HF/compressed_tensors。
_MXFP8_TARGET_IR = IRKind.W8A8_MXFP8
_ASCENDV1_DST_FORMATS = frozenset({"ascendv1", "ascendv1_saver"})


class WeightOpConfig(BaseModel):
    """Declarative weight-map operation (chunk, concat, rename, ...)."""

    model_config = ConfigDict(extra="forbid")

    type: str
    params: Dict[str, Any] = Field(default_factory=dict)


class WeightMappingRule(BaseModel):
    """
      Preprocess rule: map checkpoint keys to logical keys via structural ops.

      Inspired by HuggingFace ``WeightConverter`` / ``ConversionOps``; applied
    to the catalog before virtual tree construction.
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    source_patterns: List[str]
    target_patterns: List[str]
    ops: List[WeightOpConfig] = Field(default_factory=list)
    module_kind: str = "linear"
    reversible: bool = True


class ModuleRule(BaseModel):
    """
    Virtual-tree rule: which modules exist and how checkpoint keys bind to IR fields.
    """

    model_config = ConfigDict(extra="forbid")

    match: str
    module_kind: str = "linear"
    source_format: Optional[str] = None
    source_ir: Optional[IRKind] = None
    tensor_map: Dict[str, str] = Field(default_factory=dict)
    convert: bool = True
    defaults: Dict[str, Any] = Field(default_factory=dict)


class ConvertRule(BaseModel):
    """Per-layer conversion target and optional explicit route."""

    model_config = ConfigDict(extra="forbid")

    match: str
    target_ir: IRKind
    route: Union[List[IRKind], Literal["auto"]] = "auto"
    action: Literal["transform", "passthrough", "skip"] = "transform"


class ConvertDefaults(BaseModel):
    """Global defaults when rules omit fields."""

    model_config = ConfigDict(extra="forbid")

    src_format: str = "auto"
    dst_format: str = "ascendv1"
    dst_ir: Optional[IRKind] = None


class ParallelConfig(BaseModel):
    """Worker pool and memory budget for IR-task execution."""

    model_config = ConfigDict(extra="forbid")

    max_workers: int = 1
    max_inflight_bytes: Optional[int] = None
    max_tensor_bytes_per_task: Optional[int] = None
    shard_cache_size: int = 1
    worker_device: str = "cpu"
    # thread: 组内 ThreadPoolExecutor（受 GIL 限制）；process: 组间 ProcessPoolExecutor，纯 CPU 计算并行
    worker_backend: Literal["thread", "process"] = "process"
    # 每个 worker 进程内的线程数（YAML 层固定为 4，不经配置暴露）
    worker_threads: int = 4
    # 仅 worker_backend=thread 且 worker_device 指向 NPU 时生效，限制组内并发以防显存 OOM
    npu_max_workers: int = 1
    task_granularity: Literal["ir_task", "dependency_group"] = "dependency_group"
    # 单个 dependency group 的最大任务数；超过则按任务切成多个子组分散到不同进程并行，
    # 缓解 MoE 大组（一层 512 个 expert 任务）只能单进程承包导致的收尾拖尾、多核空闲。
    # None 或 <=0 表示不拆分（保持整组，fused 缓存复用率最高）。
    max_group_size: Optional[int] = None


class ConvertConfig(BaseModel):
    """
    Top-level convert job configuration loaded from CLI/YAML.
    """

    model_config = ConfigDict(extra="forbid")

    model_path: str
    save_path: str
    model_family: Optional[str] = None
    dst_format: str = "ascendv1"
    defaults: ConvertDefaults = Field(default_factory=ConvertDefaults)
    preprocess_rules: List[WeightMappingRule] = Field(default_factory=list)
    module_rules: List[ModuleRule] = Field(default_factory=list)
    convert_rules: List[ConvertRule] = Field(default_factory=list)
    parallel: ParallelConfig = Field(default_factory=ParallelConfig)

    @field_validator("model_path", "save_path", mode="after")
    @classmethod
    def _non_empty_path(cls, v: str) -> str:
        if not v or not str(v).strip():
            raise ValueError("path must be non-empty")
        return v

    @model_validator(mode="after")
    def _mxfp8_requires_ascendv1(self) -> ConvertConfig:
        """任意 convert_rule 目标为 W8A8_MXFP8 时，dst_format 必须为 ascendv1。"""
        wants_mxfp8 = any(r.action == "transform" and r.target_ir == _MXFP8_TARGET_IR for r in self.convert_rules)
        if not wants_mxfp8:
            return self
        dst = self.dst_format.lower()
        if dst not in _ASCENDV1_DST_FORMATS:
            raise ValueError(
                f"target_ir W8A8_MXFP8 requires dst_format ascendv1 (Ascend NPU deployment); "
                f"got dst_format={self.dst_format!r}. Use huggingface/compressed_tensors only for "
                f"FLOAT targets (e.g. fp8_block -> bf16)."
            )
        return self
