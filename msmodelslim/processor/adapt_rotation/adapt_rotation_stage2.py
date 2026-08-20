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

from typing import Annotated, List, Literal

import torch
from torch import nn
from pydantic import BaseModel, Field, AfterValidator, field_validator

from msmodelslim.core.context import get_current_context
from msmodelslim.utils.logging import get_logger, logger_setter
from msmodelslim.utils.validation.pydantic import in_range
from msmodelslim.processor.quarot.offline_quarot.quarot_interface import (
    QuaRotInterface,
    RotatePair,
    get_rotate_command,
)
from msmodelslim.processor.quarot.offline_quarot.quarot import QuaRotProcessor
from msmodelslim.processor.quarot.common.quarot_utils import is_power_of_two
from msmodelslim.utils.exception import SchemaValidateError


class AdaptRotationStage2ProcessorConfig(BaseModel):
    """adapt_rotation 阶段2的配置。

    父配置 `stage: 2` 时生效，在阶段1基础上支持在线旋转，可指定 down_proj 在线层与最大 TP 并行度。
    YAML 中这些字段平铺在 `spec.process[]` 的 adapt_rotation 处理器下（由 before-validator
    归一化为 `stage_config` 内部对象）。
    """

    type: Literal["_adapt_rotation_stage2"] = Field(
        default="_adapt_rotation_stage2", description="阶段配置类型，内部标识，无需配置。"
    )
    online: bool = Field(default=False, description="是否启用在线旋转")
    block_size: int = Field(default=-1, description="块大小，-1 表示 hidden_dim")
    down_proj_online_layers: List[int] = Field(
        default_factory=list,
        description="down_proj 在线层索引列表",
    )
    max_tp_size: Annotated[int, AfterValidator(in_range(min_val=1, max_val=64))] = Field(
        default=4, description="最大 TP 并行度"
    )

    @field_validator('down_proj_online_layers')
    @classmethod
    def validate_down_proj_online_layers(cls, v: List[int]) -> List[int]:
        """校验 down_proj_online_layers：每个元素为非负整数"""
        for i, x in enumerate(v):
            if not isinstance(x, int):
                raise SchemaValidateError(f"down_proj_online_layers[{i}] must be int, got {type(x).__name__}")
            if x < 0:
                raise SchemaValidateError(f"down_proj_online_layers[{i}] must be >= 0, got {x}")
        return v

    @field_validator('max_tp_size')
    @classmethod
    def validate_max_tp_size(cls, v: int) -> int:
        """校验 max_tp_size：必须大于等于1且为2的幂"""
        if v < 1 or not is_power_of_two(v):
            raise SchemaValidateError(f"max_tp_size must be a positive power of 2 or equal to 1, got {v}")
        return v

    @field_validator('block_size')
    @classmethod
    def validate_block_size(cls, v: int) -> int:
        """校验 block_size：取值范围为-1或2的非负整数次幂"""
        if v == -1:
            return v
        if v <= 0 or not is_power_of_two(v):
            raise SchemaValidateError(f"block_size must be -1 or a positive power of 2, got {v}")
        return v


@logger_setter(prefix="msmodelslim.processor.adapt_rotation")
class AdaptRotationStage2Processor(QuaRotProcessor):
    """
    AdaptRotation stage2: equivalent to QuaRotProcessor, but uses optimized rotation matrix
    from stage1 (via context) instead of creating it. Only hidden_size-dim rotations are
    replaced; rot_uv (head_dim) etc. remain as created by adapter.
    """

    def __init__(
        self, model: nn.Module, config: AdaptRotationStage2ProcessorConfig, adapter: QuaRotInterface, **kwargs
    ) -> None:
        super().__init__(model, config, adapter, **kwargs)
        self.rotate_pairs = None

    def _overlay_adapted_matrix(self, rotate_pairs: List[RotatePair], adapted_matrix: torch.Tensor) -> None:
        """
        Overlay adapted_matrix from stage1 onto rotations with matching dimension.
        """
        target_size = adapted_matrix.shape[0]
        for pair in rotate_pairs:
            for layer_name, rot in list(pair.right_rot.items()):
                if isinstance(rot, (list, tuple)):
                    continue  # e.g. [eye, rot_uv], skip
                if hasattr(rot, 'shape') and rot.shape[0] == target_size:
                    pair.right_rot[layer_name] = adapted_matrix
            for layer_name, rot in list(pair.left_rot.items()):
                if isinstance(rot, (list, tuple)):
                    continue
                if hasattr(rot, 'shape') and rot.shape[0] == target_size:
                    pair.left_rot[layer_name] = adapted_matrix

    def pre_run(self) -> None:
        # Get adapted_matrix from stage1 via context
        ctx = get_current_context()
        adapted_matrix_from_ctx = None
        if ctx is None:
            get_logger().warning(
                "AdaptRotation stage2: context is None, cannot get adapted_matrix from stage1. "
                "Using adapter default rotations (degraded). Ensure stage1 runs in prior stage with ContextManager."
            )
        else:
            ns = ctx.get("adapt_rotation")
            if ns is not None and "adapted_matrix" in ns.state:
                adapted_matrix_from_ctx = ns.state["adapted_matrix"]
            else:
                get_logger().warning(
                    "AdaptRotation stage2: no adapted_matrix in context. "
                    "Stage1 may not have run or failed. Using adapter default rotations (degraded)."
                )

        pre_run_fused_ln, self.fused_map = self.adapter.get_ln_fuse_map()
        pre_run_bake_names, self.bake_names = self.adapter.get_bake_names()
        pre_run_pairs, self.rotate_pairs = self.adapter.get_rotate_map(block_size=self.config.block_size)

        # Overlay adapted matrix from context (hidden_size-dim only; rot_uv etc. unchanged)
        if adapted_matrix_from_ctx is not None:
            get_logger().info(
                "Overlaying stage1 adapted_matrix (shape=%s) onto rotations", adapted_matrix_from_ctx.shape
            )
            self._overlay_adapted_matrix(pre_run_pairs, adapted_matrix_from_ctx)
            self._overlay_adapted_matrix(self.rotate_pairs, adapted_matrix_from_ctx)

        pre_run_commands = get_rotate_command(pre_run_pairs)
        self._fuse_norm(pre_run_fused_ln)
        self._bake_mean(pre_run_bake_names)
        self._rotate(pre_run_commands)
        self.rotate_commands = get_rotate_command(self.rotate_pairs)
        if self.config.online:
            self.online_processor.pre_run()
