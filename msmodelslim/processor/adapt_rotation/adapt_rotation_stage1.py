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

from collections import defaultdict
from typing import List, Literal, Dict

import functools
import torch
import torch.distributed as dist
from torch import nn
from pydantic import BaseModel, ConfigDict, Field, field_validator

from msmodelslim.core.base.protocol import BatchProcessRequest
from msmodelslim.core.context import get_current_context
from msmodelslim.processor.base import AutoSessionProcessor
from msmodelslim.utils.distributed import DistHelper
from msmodelslim.utils.exception import SchemaValidateError, UnsupportedError
from msmodelslim.utils.logging import get_logger, logger_setter
from msmodelslim.processor.quarot.offline_quarot.quarot_interface import QuaRotInterface
from msmodelslim.processor.quarot.common.quarot_utils import fuse_ln_linear, is_power_of_two
from .interface import AdaptRotationInterface
from .iterative_hadamard_optimization import HadamardOptimizer

LAYER_TYPE_STR_MAX_LEN = 128


class AdaptRotationStage1ProcessorConfig(BaseModel):
    """Internal config for stage1 (BaseModel, not in AutoProcessorConfig registry to avoid union recursion)."""
    model_config = ConfigDict(extra="forbid")
    type: Literal["_adapt_rotation_stage1"] = "_adapt_rotation_stage1"
    steps: int = Field(default=20, ge=1, le=1000, description="迭代优化步数")
    quant_dtype: Literal["int4", "int8"] = Field(
        default="int4",
        description="量化比特数，应与下游量化中激活值量化类型一致（如 w4a4 用 int4，w8a8 用 int8）",
    )
    layer_type: List[str] = Field(
        default_factory=lambda: ["up_proj"],
        min_length=1,
        description="要收集激活的层名子串列表",
    )
    block_size: int = Field(default=-1, description="块大小，-1 表示 hidden_dim")
    max_samples: int = Field(default=2048, ge=1, le=100000, description="每层最大采样数")

    @field_validator('layer_type')
    @classmethod
    def validate_layer_type(cls, v: List[str]) -> List[str]:
        """校验 layer_type：每个元素为非空字符串且长度 <= 128"""
        if not v:
            raise SchemaValidateError("layer_type must not be empty")
        for i, s in enumerate(v):
            if not isinstance(s, str):
                raise SchemaValidateError(f"layer_type[{i}] must be str, got {type(s).__name__}")
            if not s.strip():
                raise SchemaValidateError(f"layer_type[{i}] must not be empty string")
            if len(s) > LAYER_TYPE_STR_MAX_LEN:
                raise SchemaValidateError(f"layer_type[{i}] length must be <= {LAYER_TYPE_STR_MAX_LEN}, got {len(s)}")
        return v

    @field_validator('block_size')
    @classmethod
    def validate_block_size(cls, v: int) -> int:
        """校验 block_size：取值范围为-1或大于0且为2的幂的整数"""
        if v == -1:
            return v
        if v <= 0 or not is_power_of_two(v):
            raise SchemaValidateError(f"block_size must be -1 or a positive power of 2, got {v}")
        return v


@logger_setter(prefix="msmodelslim.processor.adapt_rotation")
class AdaptRotationStage1Processor(AutoSessionProcessor):
    def __init__(
        self,
        model: nn.Module,
        config: AdaptRotationStage1ProcessorConfig,
        adapter: AdaptRotationInterface,
        **kwargs
    ) -> None:
        super().__init__(model)
        self.config = config
        self.model = model
        self.adapter = adapter
        self.act_dict = defaultdict(list)
        self._stat_hooks: list = []
        self.dist_helper = None
        self.rot_matrix = None
        if not isinstance(adapter, AdaptRotationInterface):
            raise UnsupportedError(
                f'{adapter.__class__.__name__} does not support AdaptRotationInterface',
                action='AdaptRotationStage1Processor depends on AdaptRotationInterface. '
                       'Please provide a valid model adapter which implements AdaptRotationInterface',
            )

    def stat_tensor(self, name, tensor):
        tensor = tensor.detach()
        tensor = tensor.mean(dim=1).cpu()
        self.act_dict[name].append(tensor)

    def support_distributed(self) -> bool:
        return True

    def is_data_free(self) -> bool:
        return False

    # ---- distributed helpers --------------------------------------------

    def _setup_dist_helper(self, request: BatchProcessRequest) -> None:
        if dist.is_initialized():
            self.dist_helper = DistHelper(request.module, prefix=request.name)

    def _teardown_dist_helper(self) -> None:
        self.dist_helper = None

    def _gather_activations_across_ranks(self) -> None:
        ws = dist.get_world_size()
        keys = [None] * ws
        dist.all_gather_object(keys, set(self.act_dict.keys()))
        all_keys = set()
        for k in keys:
            if k:
                all_keys.update(k)
        merged = defaultdict(list)
        for key in sorted(all_keys):
            local = self.act_dict.get(key, [])
            cat = torch.cat(local, dim=0) if local else torch.empty(0)
            gathered = [None] * ws
            dist.all_gather_object(gathered, cat)
            parts = [t for t in gathered if isinstance(t, torch.Tensor) and t.numel() > 0]
            if parts:
                merged[key] = [torch.cat(parts, dim=0)]
        self.act_dict = merged

    # ---- end distributed helpers ----------------------------------------

    def pre_run(self) -> None:
        _, pre_run_fused_ln = self.adapter.get_ln_fuse_map()
        self.rot_matrix = QuaRotInterface.get_rotate_command(
            size=self.adapter.get_hidden_dim(),
            block_size=self.config.block_size,
            mode=QuaRotInterface.QuaRotMode.HADAMARD,
        )
        self._fuse_norm(pre_run_fused_ln)

    def preprocess(self, request: BatchProcessRequest) -> None:
        """注册前向钩子，供 process 阶段前向传播时收集激活。"""
        self._setup_dist_helper(request)
        prefix = request.name
        prefix = f"{prefix}." if prefix != "" else ""

        def stat_input_hook(m, x, y, name):
            if isinstance(x, tuple):
                x = x[0]
            if x.dim() == 2:
                x = x.unsqueeze(1)
            self.stat_tensor(name, x)

        self._stat_hooks = []
        for name, m in request.module.named_modules():
            if isinstance(m, torch.nn.Linear) and any(layer in name for layer in self.config.layer_type):
                self._stat_hooks.append(
                    m.register_forward_hook(
                        functools.partial(stat_input_hook, name=prefix + name)
                    )
                )

    def postprocess(self, request: BatchProcessRequest) -> None:
        """移除本层注册的前向钩子。"""
        if self._stat_hooks:
            for h in self._stat_hooks:
                h.remove()
            self._stat_hooks = []
        self._teardown_dist_helper()

    def post_run(self) -> None:
        if dist.is_initialized() and dist.get_world_size() > 1:
            self._gather_activations_across_ranks()

        act_matrix = {}
        for name, tensors in self.act_dict.items():
            all_acts = torch.cat(tensors, dim=0)
            total = all_acts.shape[0]
            if total <= self.config.max_samples:
                act_matrix[name] = all_acts
            else:
                idx = torch.linspace(0, total - 1, steps=self.config.max_samples).long()
                act_matrix[name] = all_acts[idx]

        if not act_matrix:
            raise UnsupportedError(
                "AdaptRotation stage1 collected no activations; act_dict is empty.",
                action="Check layer_type config matches model layer names. "
            )

        optimizer = HadamardOptimizer(
            quant_dtype=self.config.quant_dtype,
            max_samples=self.config.max_samples,
            steps=self.config.steps,
        )
        if getattr(torch, 'npu', None) is not None and torch.npu.is_available():
            device = torch.device(f"npu:{torch.npu.current_device()}")
        else:
            device = torch.device("cpu")
        adapted_matrix = optimizer.optimize(
            act_matrix,
            self.rot_matrix,
            device=device,
        )

        ctx = get_current_context()
        if ctx is None:
            raise UnsupportedError(
                "AdaptRotation stage1 requires context to store adapted_matrix for stage2, but get_current_context() returned None.",
                action="Ensure AdaptRotation stage1 runs within quant_service prior stage (ContextManager).",
            )
        ns = ctx["adapt_rotation"]  # pylint: disable=unsubscriptable-object
        adapted_matrix_cpu = adapted_matrix.cpu() if hasattr(adapted_matrix, "cpu") else adapted_matrix
        ns.state["adapted_matrix"] = adapted_matrix_cpu

    def _fuse_norm(self, fused_map: Dict[str, str]):
        for key, value in fused_map.items():
            get_logger().debug("start to fuse layer norm and linear: %s and %s", key, value)
            layernorms = []
            if isinstance(key, (list, tuple)):
                for k in key:
                    layernorms.append(self.model.get_submodule(k))
            else:
                layernorms.append(self.model.get_submodule(key))
            linears = []

            if isinstance(value, (list, tuple)):
                for v in value:
                    linears.append(self.model.get_submodule(v))
            else:
                linears.append(self.model.get_submodule(value))
            try:
                fuse_ln_linear(layernorms, linears)
            except UnsupportedError as e:
                raise UnsupportedError(
                    "fuse layer norm and linear error!",
                    action=f"Please check the {key} and {value} size!",
                ) from e
            get_logger().debug("successfully fuse layer norm and linear: %s and %s", key, value)
