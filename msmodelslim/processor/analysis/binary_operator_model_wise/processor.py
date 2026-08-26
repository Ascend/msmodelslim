"""
-------------------------------------------------------------------------
This file is part of the MindStudio project.
Copyright (c) 2026 Huawei Technologies Co.,Ltd.

MindStudio is licensed under Mulan PSL v2.
You may use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:

         http://license.coscl.org.cn/MulanPSL2

THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details.
"""

from typing import Annotated, Any, Dict, List, Literal, Optional, Tuple

import torch
from pydantic import Field, AfterValidator
from torch import nn

from msmodelslim.core.base.protocol import BatchProcessRequest
from msmodelslim.core.context import get_current_context
from msmodelslim.ir.qal.qregistry import QABCRegistry
from msmodelslim.processor.base import AutoProcessorConfig, AutoProcessorConfigList, AutoSessionProcessor
from msmodelslim.processor.analysis.distributed_utils import (
    check_distributed_analysis_supported,
    maybe_barrier_before_linear_quant,
    publish_layer_analysis_result,
)
from msmodelslim.processor.analysis.binary_operator_model_wise.metrics.mse_model_wise.block_data import (
    resolve_mse_model_wise_block_data,
)
from msmodelslim.utils.logging import get_logger
from msmodelslim.utils.exception import UnexpectedError, UnsupportedError
from msmodelslim.utils.validation.pydantic import validate_str_length

from .metrics.factory import ModelWiseMethodFactory


class BinaryOperatorModelWiseProcessorConfig(AutoProcessorConfig):
    """模型级敏感度分析配置（对比模型最终输出，使用 MSE 指标）"""

    type: Literal["binary_operator_model_wise"] = Field(
        default="binary_operator_model_wise", description="处理器类型，固定为 `binary_operator_model_wise`。"
    )
    metrics: str = Field(
        default="mse_model_wise",
        description="分析指标：`mse_model_wise`（对比模型最终输出）",
    )
    quant_modules: List[Annotated[str, AfterValidator(validate_str_length())]] = Field(
        default_factory=lambda: ["*"],
        description=(
            "与 linear_quant.include、CLI --quant_modules 一致；"
            "用于结果展示名后缀，如 model.layers.2 (*mlp*)。实际量化范围以 linear_quant 为准。"
        ),
    )
    configs: AutoProcessorConfigList = Field(
        default_factory=list,
        description="量化子处理器配置列表，用于进行量化-反量化",
    )


@QABCRegistry.register(dispatch_key=BinaryOperatorModelWiseProcessorConfig, abc_class=AutoSessionProcessor)
class BinaryOperatorModelWiseProcessor(AutoSessionProcessor):
    """模型级敏感度分析"""

    def __init__(
        self,
        model: nn.Module,
        config: BinaryOperatorModelWiseProcessorConfig,
        adapter: Optional[object] = None,
    ):
        super().__init__(model)
        self.config = config
        self.adapter = adapter
        self.quant_processors = [AutoSessionProcessor.from_config(model, cfg, adapter) for cfg in config.configs]
        self._analysis_method = ModelWiseMethodFactory.create_method(config.metrics, adapter=adapter)
        self._block_data = resolve_mse_model_wise_block_data(adapter)
        self._base_data_count: int = 0
        self._block_names: List[str] = []
        self._float_outputs: List[Any] = []
        self._quant_inputs: List[Any] = []
        self._merged_outputs: List[Any] = []
        # Skip switch for non-chainable layers. Once enabled, this and all subsequent blocks are skipped.
        self._skip_remaining_blocks: bool = False
        self._skipped_request_names: List[str] = []

    def support_distributed(self) -> bool:
        return all(processor.support_distributed() for processor in self.quant_processors)

    def pre_run(self) -> None:
        ctx = get_current_context()
        if ctx is None:
            raise UnexpectedError("No context is working.")
        check_distributed_analysis_supported(
            self._analysis_method.supports_distributed,
            self._analysis_method.name,
        )
        for processor in self.quant_processors:
            processor.pre_run()

    def preprocess(self, request: BatchProcessRequest) -> None:
        if self._skip_remaining_blocks:
            self._skipped_request_names.append(request.name)
            get_logger().warning(
                "BinaryOperatorModelWiseProcessor: skip layer %s (already in skip mode).",
                request.name,
            )
            # In skip mode, still run forward once to keep the generator chain alive.
            if request.datas is not None:
                self._run_forward_if_need(request)
            return

        # Auto-detect non-chainable layers (e.g. MTP input)
        # once chaining check fails, warn and skip this and all subsequent layers.
        try:
            request.datas = self._replace_request_datas_with_merged_outputs_if_need(request.datas)
        except UnsupportedError as e:
            self._skipped_request_names.append(request.name)
            get_logger().warning(
                "BinaryOperatorModelWiseProcessor: enter skip mode at %s; skip this and subsequent layers. reason=%s",
                request.name,
                str(e),
            )
            # Keep the forward chain runnable.
            if request.datas is not None:
                self._run_forward_if_need(request)
            self._skip_remaining_blocks = True
            return

        self._block_names.append(request.name)

        if self._base_data_count == 0:
            self._base_data_count = len(request.datas)

        float_inputs, quant_inputs = self._build_float_quant_inputs(request.datas)

        request.datas = float_inputs
        self._run_forward_if_need(request)
        self._float_outputs = request.outputs
        self._quant_inputs = quant_inputs

    def process(self, request: BatchProcessRequest) -> None:
        if self._skip_remaining_blocks:
            return
        maybe_barrier_before_linear_quant()
        request.datas = self._quant_inputs
        for qp in self.quant_processors:
            qp.preprocess(request)
            qp.process(request)
            qp.postprocess(request)

    def postprocess(self, request: BatchProcessRequest) -> None:
        if self._skip_remaining_blocks:
            return
        request.datas = self._quant_inputs
        self._run_forward_if_need(request)

        # 将纯浮点输出与带量化输出结果拼接
        self._merged_outputs = [*self._float_outputs, *request.outputs]

    def post_run(self) -> None:
        for processor in self.quant_processors:
            processor.post_run()

        self._validate_merged_outputs()
        layer_scores = self._compute_layer_scores()
        layer_scores = publish_layer_analysis_result(
            layer_scores,
            self._analysis_method.name,
            quant_modules=self.config.quant_modules,
        )

        if self._skipped_request_names:
            get_logger().warning(
                "BinaryOperatorModelWiseProcessor: skipped %d layers (ranking excludes them). skipped_layers=%s",
                len(self._skipped_request_names),
                ", ".join(self._skipped_request_names),
            )

        get_logger().info(
            "BinaryOperatorModelWiseProcessor post_run: %d layer scores (%s), quant_modules=%s",
            len(layer_scores),
            self._analysis_method.name,
            self.config.quant_modules,
        )

    def _validate_merged_outputs(self) -> None:
        base_count = self._base_data_count
        num_layers = len(self._block_names)
        expected = base_count * (num_layers + 1) if base_count > 0 else 0

        if base_count <= 0 or len(self._merged_outputs) < expected:
            raise UnexpectedError(
                "BinaryOperatorModelWiseProcessor post_run got invalid merged outputs: "
                f"base_count={base_count}, merged={len(self._merged_outputs)}, "
                f"num_layers={num_layers}, expected={expected}."
            )

    def _compute_layer_scores(self) -> List[Dict[str, Any]]:
        layer_scores: List[Dict[str, Any]] = []
        base_count = self._base_data_count
        ref_outputs: List[Any] = self._merged_outputs[:base_count]
        for layer_idx, layer_name in enumerate(self._block_names):
            block_base = base_count * (layer_idx + 1)
            cand_outputs: List[Any] = self._merged_outputs[block_base : block_base + base_count]

            score = self._analysis_method.compute_score(ref_outputs, cand_outputs)
            layer_scores.append(
                {
                    "name": layer_name,
                    "score": score,
                }
            )
        return layer_scores

    def _replace_request_datas_with_merged_outputs_if_need(
        self, datas: Optional[List[Tuple[tuple, dict]]]
    ) -> Optional[List[Tuple[tuple, dict]]]:
        """若存在上一层 merged_outputs，则用其 hidden_states 重建当前层 datas。"""
        merged_outputs = self._merged_outputs
        base_data_count = self._base_data_count

        if not merged_outputs or datas is None:
            return datas

        old_rows = datas

        if base_data_count > 0:
            if len(old_rows) < base_data_count or len(merged_outputs) < base_data_count:
                raise UnexpectedError(
                    "BinaryOperatorModelWiseProcessor got inconsistent tensor counts for hidden_states "
                    f"consistency check: base_data_count={base_data_count}, "
                    f"datas={len(old_rows)}, merged_outputs={len(merged_outputs)}."
                )

            base_rows = old_rows[:base_data_count]
            tail_outputs = merged_outputs[-base_data_count:]
            for row, out in zip(base_rows, tail_outputs):
                try:
                    req_hidden = self._block_data.extract_hidden_states(row)
                    merged_hidden = self._block_data.extract_hidden_states(out)
                except UnsupportedError:
                    get_logger().debug(
                        "BinaryOperatorModelWiseProcessor: skip output chaining; use generator datas.",
                    )
                    return datas

                merged_hidden = merged_hidden.to(
                    device=req_hidden.device,
                    dtype=req_hidden.dtype,
                )
                if req_hidden.shape != merged_hidden.shape:
                    get_logger().debug(
                        "BinaryOperatorModelWiseProcessor: skip output chaining due to shape mismatch.",
                    )
                    return datas
                if not torch.allclose(req_hidden, merged_hidden):
                    raise UnsupportedError(
                        "Model-wise chaining broken: current layer input hidden_states != previous layer output."
                    )

        new_rows: List[Tuple[tuple, dict]] = []
        for idx, out in enumerate(merged_outputs):
            try:
                hidden = self._block_data.extract_hidden_states(out)
            except UnsupportedError:
                get_logger().debug(
                    "BinaryOperatorModelWiseProcessor: skip output chaining; use generator datas.",
                )
                return datas
            _, template_kwargs = old_rows[idx % len(old_rows)]
            new_rows.append(((hidden,), template_kwargs))

        return new_rows

    def _build_float_quant_inputs(
        self,
        datas: Optional[List[Tuple[tuple, dict]]],
    ) -> Tuple[List[Tuple[tuple, dict]], List[Tuple[tuple, dict]]]:
        """float 用全部行，quant 用前 ``quant_source_count`` 行。"""
        num_datas = self._base_data_count or None
        return list(datas), datas[:num_datas]
