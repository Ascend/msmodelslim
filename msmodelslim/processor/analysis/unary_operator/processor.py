#!/usr/bin/env python
# -*- coding: UTF-8 -*-

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

import functools
import inspect
from typing import Annotated, Any, Callable, Dict, List, Literal, Optional

from pydantic import Field, AfterValidator, model_validator
from torch import nn

from msmodelslim.core.base.protocol import BatchProcessRequest
from msmodelslim.core.context import get_current_context
from msmodelslim.ir.qal.qregistry import QABCRegistry
from msmodelslim.processor.base import AutoProcessorConfig, AutoSessionProcessor
from msmodelslim.utils.validation.pydantic import validate_str_length
from msmodelslim.processor.analysis.distributed_utils import (
    check_distributed_analysis_supported,
    merge_packed_layer_stats_across_ranks,
    publish_layer_analysis_result,
    write_layer_analysis_result,
)
from msmodelslim.processor.analysis.unary_operator.metrics.factory import UnaryAnalysisMethodFactory
from msmodelslim.processor.analysis.unary_operator.metrics.ra_compress import (
    RA_COMPRESS_METRIC,
    validate_metric_params,
)
from msmodelslim.utils.logging import get_logger
from msmodelslim.utils.exception import SchemaValidateError, UnexpectedError

# 各指标专属超参的校验器：参数白名单与取值约束由指标自身声明，processor 只按 metrics 分派
_METRIC_PARAMS_VALIDATORS: Dict[str, Callable[[Dict[str, Any]], None]] = {
    RA_COMPRESS_METRIC: validate_metric_params,
}


def _accepts_block_name(setter: Callable) -> bool:
    """判断 set_forward_kwargs 实现是否接受第二个入参（当前块名）。"""
    try:
        return len(inspect.signature(setter).parameters) >= 2
    except (TypeError, ValueError):  # 内建函数等无法取签名时按不支持处理
        return False


class UnaryAnalysisProcessorConfig(AutoProcessorConfig):
    """一元（无量化）敏感度分析处理器配置。

    位于 `spec.process[]`，由 `type: unary_analysis` 分派；基于激活分布统计量
    （分位数/标准差/峰度）评估各层对量化的敏感度。
    """

    type: Literal["unary_analysis"] = Field(
        default="unary_analysis", description="处理器类型，固定为 `unary_analysis`。"
    )
    metrics: str = Field(
        default="kurtosis",
        description="分析指标：`quantile`（分位数）、`std`（标准差）、`kurtosis`（峰度）、"
        "`ra_compress`（RA Compress 长序列压缩头筛选）",
    )
    patterns: List[Annotated[str, AfterValidator(validate_str_length())]] = Field(
        default_factory=lambda: ["*"],
        description="待分析的层名模式列表，默认 `*` 匹配全部",
    )
    metric_params: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "指标专属超参，仅由 `metrics` 指定的分析方法解析。当前仅 `metrics=ra_compress` 支持："
            "`induction_head_ratio`（默认 0.14）、`echo_head_ratio`（默认 0.01），取值范围均为 [0, 1]。"
        ),
    )

    @model_validator(mode="after")
    def _validate_metric_params(self) -> "UnaryAnalysisProcessorConfig":
        """校验指标专属超参，非法使用直接报错。"""
        params = self.metric_params or {}
        if not params:
            return self

        validate = _METRIC_PARAMS_VALIDATORS.get(self.metrics)
        if validate is None:
            raise SchemaValidateError(
                f"metrics={self.metrics!r} does not accept metric_params, but got {sorted(params)}",
                action="Please remove metric_params, or use a metrics that supports it "
                f"(supported: {sorted(_METRIC_PARAMS_VALIDATORS)}).",
            )
        validate(params)
        return self


@QABCRegistry.register(dispatch_key=UnaryAnalysisProcessorConfig, abc_class=AutoSessionProcessor)
class UnaryAnalysisProcessor(AutoSessionProcessor):
    """
    Layer sensitivity analysis using unary activation (single forward).

    Lifecycle orchestration (postprocess / post_run) lives here. The analysis
    method supplies ``compute_score`` and optionally ``pack`` / ``merge`` for DP.
    """

    def __init__(
        self,
        model: nn.Module,
        config: UnaryAnalysisProcessorConfig,
        adapter: Optional[object] = None,
    ):
        super().__init__(model)
        self.config = config
        self._analysis_method = UnaryAnalysisMethodFactory.create_method(
            config.metrics,
            adapter=adapter,
            **dict(config.metric_params or {}),
        )
        self._target_layers: List[str] = []
        self._layer_stats: Dict[str, Any] = {}
        self._pending_packed_stats: Dict[str, Any] = {}
        self._layer_scores: List[Dict[str, Any]] = []
        self._hook_handles: Dict[str, Any] = {}

    def support_distributed(self) -> bool:
        return True

    def pre_run(self) -> None:
        ctx = get_current_context()
        if ctx is None:
            raise UnexpectedError("No context is working.")
        check_distributed_analysis_supported(
            self._analysis_method.supports_distributed,
            self._analysis_method.name,
        )

    def _forward_method_kwargs(self, request: BatchProcessRequest) -> None:
        """把当前层的 forward kwargs 透传给分析方法（如 ra_compress 需要 RoPE）。

        可选回调：方法实现 ``set_forward_kwargs`` 才会被调用，其它指标不受影响。
        回调可选地接收第二个入参（当前块名，如 ``model.layers.0``），用于按块缓存
        模型下发的位置编码等信息；只接收 kwargs 的实现仍然兼容。
        多样本时各样本的层 kwargs 相同，取第一份即可。
        """
        setter = getattr(self._analysis_method, 'set_forward_kwargs', None)
        if not callable(setter):
            return
        if not request.datas:
            return
        try:
            if _accepts_block_name(setter):
                setter(request.datas[0][1], request.name)
            else:
                setter(request.datas[0][1])
        except Exception as error:  # pylint: disable=broad-except
            get_logger().warning(
                "Failed to forward layer kwargs to analysis method %s: %s",
                self._analysis_method.name,
                error,
            )

    def preprocess(self, request: BatchProcessRequest) -> None:
        all_layers = self._analysis_method.get_target_layers(request.module, request.name)
        self._target_layers = self._analysis_method.filter_layers_by_patterns(all_layers, self.config.patterns)
        self._forward_method_kwargs(request)
        get_logger().debug(
            "UnaryAnalysisProcessor preprocess: %d target layers (metrics=%s)",
            len(self._target_layers),
            self._analysis_method.name,
        )

        if len(self._target_layers) == 0:
            get_logger().warning(
                "No target layers/modules found matching the specified patterns for %s. "
                "Please check the patterns %s and the model structure, to ensure it meets expectations.",
                request.name,
                self.config.patterns,
            )

        # Runner 按块下发 request（如 request.name='model.layers.0'），target_layers 是叶子 Linear 全名
        # 遍历当前块下属于 _target_layers 的 nn.Linear 子模块，逐个注册 hook
        hook_fn = self._analysis_method.get_hook()
        for sub_name, sub_module in request.module.named_modules(prefix=request.name):
            if sub_name not in self._target_layers:
                continue
            if not isinstance(sub_module, nn.Linear):
                continue
            bound_hook = functools.partial(
                hook_fn,
                layer_name=sub_name,
                stats_dict=self._layer_stats,
            )
            handle = sub_module.register_forward_hook(bound_hook)
            self._hook_handles[sub_name] = handle

    def postprocess(self, request: BatchProcessRequest) -> None:
        # 卸 hook；按 method.pack 决定缓存统计量或本地算分，并释放 raw 激活
        keys_to_remove = [k for k in self._hook_handles if k == request.name or k.startswith(request.name + ".")]
        for k in keys_to_remove:
            handle = self._hook_handles.pop(k, None)
            if handle is not None:
                handle.remove()
            if k in self._layer_stats and k in self._target_layers:
                packed = self._analysis_method.pack_stats_for_distributed_merge(self._layer_stats[k])
                if packed is not None:
                    self._pending_packed_stats[k] = packed
                    get_logger().debug("%s: packed stats for distributed merge", k)
                else:
                    score = self._analysis_method.compute_score(self._layer_stats[k])
                    self._layer_scores.append({"name": k, "score": score})
                    get_logger().debug("%s: %s", k, score)
                del self._layer_stats[k]

    def post_run(self) -> None:
        merge_across_ranks = True
        if self._pending_packed_stats:
            merged_stats = merge_packed_layer_stats_across_ranks(
                self._pending_packed_stats,
                self._analysis_method.merge_distributed_stats,
            )
            self._layer_scores = []
            for name in sorted(merged_stats.keys()):
                score = self._analysis_method.compute_score(merged_stats[name])
                self._layer_scores.append({"name": name, "score": score})
                get_logger().debug("%s: %s", name, score)
            self._pending_packed_stats.clear()
            # Scores already come from globally reduced stats.
            merge_across_ranks = False

        self._layer_scores = publish_layer_analysis_result(
            self._layer_scores,
            self._analysis_method.name,
            patterns=self.config.patterns,
            merge_across_ranks=merge_across_ranks,
        )
        self._analysis_method.enrich_layer_scores(self._layer_scores)
        self._layer_scores = write_layer_analysis_result(
            self._layer_scores,
            self._analysis_method.name,
            patterns=self.config.patterns,
        )

        get_logger().info(
            "UnaryAnalysisProcessor post_run: %d layer scores computed (%s)",
            len(self._layer_scores),
            self._analysis_method.name,
        )

        if not self._layer_scores:
            get_logger().warning(
                "No statistics collected. This may be caused by empty calibration data "
                "or incompatible patterns with the model structure."
            )

    def get_layer_scores(self) -> List[Dict[str, Any]]:
        """Return the computed layer scores (for tests or when context is not used)."""
        return self._layer_scores
