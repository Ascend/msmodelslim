#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
-------------------------------------------------------------------------
This file is part of the MindStudio project.
Copyright (c) 2025 Huawei Technologies Co.,Ltd.

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

import operator
from typing import List, Literal, Dict

import torch
from torch import nn
from pydantic import field_validator, Field

import msmodelslim.ir as qir
from msmodelslim.core.base.protocol import BatchProcessRequest
from msmodelslim.ir.qal.qregistry import QABCRegistry
from msmodelslim.processor.base import AutoProcessorConfig, AutoSessionProcessor
from msmodelslim.processor.lossless.protocol import LosslessCheckPlan
from msmodelslim.utils.exception import SchemaValidateError
from msmodelslim.utils.exception import UnsupportedError
from msmodelslim.utils.logging import get_logger
from .laos_online import LAOSOnlineRotationProcessor
from .quarot_interface import QuaRotInterface, RotSide, RotateCommand, get_rotate_command
from ..common.quarot_utils import (
    fuse_ln_linear,
    rotate_linear,
    is_power_of_two,
    bake_mean_into_linear,
    rotate_weight,
)


def _request_target_matches(target: str, prefix: str) -> bool:
    if not prefix:
        return True
    return target == prefix.rstrip(".") or target.startswith(prefix)


class QuaRotProcessorConfig(AutoProcessorConfig):
    """QuaRot（离线旋转）处理器配置。

    位于 `spec.process[]`，由 `type: quarot` 分派；通过随机正交旋转消除激活离群值
    的维度结构，降低低比特量化的精度损失。
    """

    type: Literal["quarot"] = Field(default="quarot", description="处理器类型，固定为 `quarot`。")
    online: bool = Field(default=False, description="是否在线旋转（默认离线）。")
    block_size: int = Field(default=-1, description="旋转块大小，-1 表示按 hidden_dim 整块旋转。")
    down_proj_online_layers: List[int] = Field(
        default_factory=lambda: [], description="需要在线旋转的 down_proj 层索引列表。"
    )
    max_tp_size: int = Field(default=4, description="最大 TP（Tensor Parallelism，张量并行）并行度，必须为2的幂。")
    """为 True 时在 pre_run 中向首个旋转目标模块注入 QuaRotExtraInfoHookIR，用于导出 optional.quarot.global_rotation。"""
    export_extra_info: bool = Field(
        default=True, description="是否导出 `optional.quarot.global_rotation` 旋转信息，用于下游部署。"
    )

    @field_validator("max_tp_size")
    @classmethod
    def validate_max_tp_size(cls, v):
        """校验 max_tp_size：必须大于等于1且为2的幂"""
        if v < 1 or not is_power_of_two(v):
            raise SchemaValidateError(f"max_tp_size must be a positive power of 2 or equal to 1, got {v}")
        return v

    @field_validator("block_size")
    @classmethod
    def validate_block_size(cls, v):
        """校验 block_size：取值范围为-1或2的非负整数次幂"""
        if v == -1:
            return v
        if v <= 0 or not is_power_of_two(v):
            raise SchemaValidateError(f"block_size must be -1 or a positive power of 2, got {v}")
        return v


@QABCRegistry.register(dispatch_key=QuaRotProcessorConfig, abc_class=AutoSessionProcessor)
class QuaRotProcessor(AutoSessionProcessor):
    def __init__(
        self,
        model: nn.Module,
        config: QuaRotProcessorConfig,
        adapter: QuaRotInterface,
        **kwargs,
    ) -> None:
        super().__init__(model)
        self.config = config
        self.model = model
        self.adapter = adapter
        self.fused_map = {}
        self.bake_names = []
        self.rotate_commands = []
        self.rotate_pairs = []
        self._plan_ready = False
        self._pre_run_fused_map = {}
        self._pre_run_bake_names = []
        self._pre_run_pairs = []
        self._lossless_pre_run_fused_map = {}
        self._lossless_pre_run_bake_names = []
        self._lossless_fused_map = {}
        self._lossless_bake_names = []
        self._lossless_pre_run_commands = []
        self._lossless_rotate_commands = []
        self._lossless_global_rotation = None
        self._lossless_check_active = False
        self._lossless_adapter_request_state = None
        self._deferred_pre_run_fused_map = {}
        self._deferred_pre_run_bake_names = []
        self._deferred_pre_run_commands = []
        self._deferred_pre_run_export_commands = []
        self._lossless_export_hook_injected = False
        if not isinstance(adapter, QuaRotInterface):
            raise UnsupportedError(
                f"{adapter.__class__.__name__} does not support QuaRot",
                action="Please provide a valid model adapter which implements QuaRotInterface",
            )
        if self.config.online:
            self.online_processor = LAOSOnlineRotationProcessor(model, config, adapter)

    def support_distributed(self) -> bool:
        return True

    def is_data_free(self) -> bool:
        return True

    def prepare_lossless_check(self) -> LosslessCheckPlan:
        if self.config.online:
            raise UnsupportedError(
                "Lossless check only supports offline QuaRot",
                action="Please set online to False in the nested QuaRot config",
            )
        self._prepare_plan()
        if self._lossless_adapter_request_state is None:
            enable_global_requests = getattr(self.adapter, "_enable_lossless_check_global_requests", None)
            if callable(enable_global_requests):
                self._lossless_adapter_request_state = enable_global_requests()
        self._lossless_check_active = True
        self._deferred_pre_run_fused_map = _copy_mapping(self._pre_run_fused_map)
        self._deferred_pre_run_bake_names = list(self._pre_run_bake_names)
        self._deferred_pre_run_commands = [
            _clone_rotate_command(command) for command in get_rotate_command(self._pre_run_pairs)
        ]
        self._deferred_pre_run_export_commands = [
            _clone_rotate_command(command) for command in self._deferred_pre_run_commands
        ]
        self._lossless_export_hook_injected = False
        expected_targets = _collect_operation_targets(
            self._lossless_pre_run_fused_map,
            self._lossless_pre_run_bake_names,
            self._lossless_pre_run_commands,
        )
        expected_targets.update(
            _collect_operation_targets(
                self._lossless_fused_map,
                self._lossless_bake_names,
                self._lossless_rotate_commands,
            )
        )
        return LosslessCheckPlan(expected_targets=tuple(sorted(expected_targets)))

    def create_lossless_check_context(self, request: BatchProcessRequest, reference_datas):
        from .quarot_lossless import create_quarot_lossless_check_context

        if not self._plan_ready:
            raise UnsupportedError("QuaRot lossless-check plan is not prepared")
        return create_quarot_lossless_check_context(self, request, reference_datas)

    def align_lossless_check_io(self, context, target, reference, processed):
        from .quarot_lossless import align_quarot_lossless_check_io

        return align_quarot_lossless_check_io(context, target, reference, processed)

    def pre_run(self) -> None:
        self._prepare_plan()
        self._record_debug_info(self._pre_run_pairs, self.rotate_pairs)
        if self._lossless_check_active:
            return
        pre_run_commands = get_rotate_command(self._pre_run_pairs)
        self._fuse_norm(self._pre_run_fused_map)
        self._bake_mean(self._pre_run_bake_names)
        self._rotate(pre_run_commands)
        self._inject_global_rotation_export_hook(pre_run_commands)

        if self.config.online:
            self.online_processor.pre_run()

    def _prepare_plan(self) -> None:
        if self._plan_ready:
            return
        self._pre_run_fused_map, fused_map = self.adapter.get_ln_fuse_map()
        self._pre_run_bake_names, bake_names = self.adapter.get_bake_names()
        self._pre_run_pairs, self.rotate_pairs = self.adapter.get_rotate_map(block_size=self.config.block_size)
        self.fused_map = _copy_mapping(fused_map)
        self.bake_names = list(bake_names)
        self.rotate_commands = get_rotate_command(self.rotate_pairs)
        self._lossless_pre_run_fused_map = _copy_mapping(self._pre_run_fused_map)
        self._lossless_pre_run_bake_names = list(self._pre_run_bake_names)
        self._lossless_fused_map = _copy_mapping(fused_map)
        self._lossless_bake_names = list(bake_names)
        self._lossless_pre_run_commands = [
            _clone_rotate_command(command) for command in get_rotate_command(self._pre_run_pairs)
        ]
        self._lossless_rotate_commands = [_clone_rotate_command(command) for command in self.rotate_commands]
        for command in get_rotate_command(self._pre_run_pairs):
            if command.side == RotSide.RIGHT:
                self._lossless_global_rotation = _clone_rotation(command.rot)
                break
        self._plan_ready = True

    def _inject_global_rotation_export_hook(self, pre_run_commands: List) -> None:
        """向首个旋转目标模块注入 QuaRotExtraInfoHookIR，用于导出 optional.quarot.global_rotation。
        note: 后续若会对全局旋转矩阵有所调整，如 RotationTune 算法，需要调整此处注入的内容。
        """
        if self.config.export_extra_info and pre_run_commands:
            first_cmd = pre_run_commands[0]
            global_rotation = first_cmd.rot.detach().clone()
            rotation_info = qir.QuarotOfflineRotationInfo(global_rotation=global_rotation)
            hook_ir = qir.QuaRotExtraInfoHookIR(rotation_info)
            sub_module = self.model.get_submodule(first_cmd.target)
            hook_handle = sub_module.register_forward_pre_hook(hook_ir)
            hook_ir.set_hook_handle(hook_handle)
            get_logger().info("Injected QuaRotExtraInfoHookIR for optional.quarot.global_rotation export")
        elif self.config.export_extra_info and not pre_run_commands:
            get_logger().warning(
                "export_extra_info is True but pre_run_commands is empty; "
                "no global rotation matrix available for export (optional.quarot.global_rotation)."
            )

    def _record_debug_info(self, pre_run_pairs, rotate_pairs):
        from msmodelslim.core.context import get_current_context

        ctx = get_current_context()

        if ctx is not None and ctx.is_enable_debug():
            ns = operator.getitem(ctx, "quarot_rotate_matrices")

            for pre_run in pre_run_pairs:
                self._record_rotate_pair_mapping(pre_run, ns)

            for rotate_pair in rotate_pairs:
                self._record_rotate_pair_mapping(rotate_pair, ns)

    def _record_rotate_pair_mapping(self, rotate_pair, ns):
        for side_name, rot_dict in [
            ("left", rotate_pair.left_rot),
            ("right", rotate_pair.right_rot),
        ]:
            for layer_name, rot_tensor in rot_dict.items():
                key = f"{layer_name}.{side_name}"

                if isinstance(rot_tensor, list):
                    ns.debug[key] = [m.cpu().detach() for m in rot_tensor]
                else:
                    ns.debug[key] = rot_tensor.cpu().detach()

    def preprocess(self, request: BatchProcessRequest) -> None:
        if self._lossless_check_active:
            return
        self._process_request(request)
        if self.config.online:
            self.online_processor.preprocess(request)

    def process(self, request: BatchProcessRequest) -> None:
        if not self._lossless_check_active:
            super().process(request)
            return
        self._apply_deferred_pre_run()
        self._process_request(request)
        request.outputs = None

    def _process_request(self, request: BatchProcessRequest) -> None:
        prefix = request.name
        prefix = f"{prefix}." if prefix != "" else ""
        fused_map = self._filter_fused_map(prefix)
        bake_names = self._filter_bake_names(prefix)
        rotate_commands = self._filter_commands(prefix)
        self._fuse_norm(fused_map)
        self._bake_mean(bake_names)
        self._rotate(rotate_commands)

    def post_run(self) -> None:
        try:
            if self._lossless_check_active:
                self._apply_deferred_pre_run()
            self._fuse_norm(self.fused_map)
            self.fused_map = {}
            self._bake_mean(self.bake_names)
            self.bake_names = []
            self._rotate(self.rotate_commands)
            self.rotate_commands = []
            if self.config.online:
                self.online_processor.post_run()
        finally:
            self.abort_lossless_check()

    def abort_lossless_check(self) -> None:
        """Discard deferred self-check work and restore adapter-only request state."""
        if not self._lossless_check_active and self._lossless_adapter_request_state is None:
            return
        try:
            if self._lossless_adapter_request_state is not None:
                restore_global_requests = getattr(self.adapter, "_restore_lossless_check_global_requests", None)
                if callable(restore_global_requests):
                    restore_global_requests(self._lossless_adapter_request_state)
        finally:
            self._lossless_adapter_request_state = None
            self._lossless_check_active = False
            self._deferred_pre_run_fused_map = {}
            self._deferred_pre_run_bake_names = []
            self._deferred_pre_run_commands = []
            self._deferred_pre_run_export_commands = []
            self._lossless_export_hook_injected = False
            self.fused_map = {}
            self.bake_names = []
            self.rotate_commands = []

    def _apply_deferred_pre_run(self) -> None:
        if not self._lossless_check_active:
            return
        self._fuse_norm(self._deferred_pre_run_fused_map)
        self._deferred_pre_run_fused_map = {}
        self._bake_mean(self._deferred_pre_run_bake_names)
        self._deferred_pre_run_bake_names = []
        self._rotate(self._deferred_pre_run_commands)
        self._deferred_pre_run_commands = []
        if not self._lossless_export_hook_injected:
            self._inject_global_rotation_export_hook(self._deferred_pre_run_export_commands)
            self._lossless_export_hook_injected = True

    def _filter_fused_map(self, prefix: str) -> Dict[str, str]:
        res = {}
        for key, value in self.fused_map.items():
            select = False
            if isinstance(value, list):
                for v in value:
                    if _request_target_matches(v, prefix):
                        select = True
            else:
                if _request_target_matches(value, prefix):
                    select = True
            if select:
                res[key] = value
        self.fused_map = {k: v for k, v in self.fused_map.items() if k not in res}
        return res

    def _filter_bake_names(self, prefix: str):
        res = [name for name in self.bake_names if _request_target_matches(name, prefix)]
        for name in res:
            self.bake_names.remove(name)
        return res

    def _filter_commands(self, prefix: str):
        res = [command for command in self.rotate_commands if _request_target_matches(command.target, prefix)]
        for command in res:
            self.rotate_commands.remove(command)
        return res

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

    def _bake_mean(self, bake_names: List[str]):
        for name in bake_names:
            get_logger().debug("start to bake mean into linear: %s", name)
            mod = self.model.get_submodule(name)
            if isinstance(mod, nn.Linear):
                bake_mean_into_linear(mod)
                get_logger().debug("successfully bake mean into linear: %s", name)
            else:
                raise UnsupportedError(
                    "bake mean into linear error!",
                    action=f"Please check the {name} type and model adapter implementation!",
                )

    def _rotate(self, rotate_commands: List[str]):
        for command in rotate_commands:
            get_logger().debug("start to %s rotate linear: %s", command.side.value, command.target)
            try:
                mod = self.model.get_submodule(command.target)
                rotate_linear(mod, command.rot, command.side == RotSide.RIGHT)
            except AttributeError:
                path_list = command.target.split(".")
                mod = self.model.get_submodule(".".join(path_list[:-1]))
                weight = getattr(mod, path_list[-1])
                rotate_weight(weight, command.rot, command.side == RotSide.RIGHT)
            except UnsupportedError as e:
                raise UnsupportedError(
                    f"{command.side.value} rotate linear error!",
                    action=f"Please check whether the {command.target} size is equal \
                                    to the rotate matrix size: {command.rot.shape[0]}!",
                ) from e
            get_logger().debug("%s rotate linear success: %s", command.side.value, command.target)


def _copy_mapping(mapping):
    copied = {}
    for key, value in mapping.items():
        if isinstance(value, list):
            copied[key] = list(value)
        elif isinstance(value, tuple):
            copied[key] = tuple(value)
        else:
            copied[key] = value
    return copied


def _clone_rotate_command(command):
    return RotateCommand(
        target=command.target,
        rot=_clone_rotation(command.rot),
        side=command.side,
    )


def _clone_rotation(rotation):
    if isinstance(rotation, torch.Tensor):
        return rotation.detach().clone()
    if isinstance(rotation, list):
        return [_clone_rotation(item) for item in rotation]
    if isinstance(rotation, tuple):
        return tuple(_clone_rotation(item) for item in rotation)
    return rotation


def _collect_operation_targets(fused_map, bake_names, commands):
    targets = set(bake_names)
    targets.update(command.target for command in commands)
    for value in fused_map.values():
        if isinstance(value, (list, tuple)):
            targets.update(value)
        else:
            targets.add(value)
    return targets
