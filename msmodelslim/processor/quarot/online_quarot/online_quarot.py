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

from typing import Annotated, List, Literal, Optional, Dict, Callable

from torch import nn
from pydantic import Field, field_validator, AfterValidator

import msmodelslim.ir as qir
from msmodelslim.ir.qal.qregistry import QABCRegistry
from msmodelslim.core.base.protocol import BatchProcessRequest
from msmodelslim.processor.base import AutoProcessorConfig, AutoSessionProcessor
from msmodelslim.utils.validation.pydantic import validate_str_length
from msmodelslim.utils.config_map import ConfigSet
from msmodelslim.utils.exception import SchemaValidateError, UnsupportedError
from msmodelslim.utils.logging import get_logger
from msmodelslim.ir.quarot import RotationType
from .online_quarot_interface import OnlineQuaRotInterface, RotationConfig
from ..common.quarot_utils import create_rot, is_power_of_two, rotate_linear


class OnlineQuaRotProcessorConfig(AutoProcessorConfig):
    """在线 QuaRot 旋转处理器配置。

    位于 `spec.process[]`，由 `type: online_quarot` 分派；在推理过程中在线应用随机正交旋转，
    消除激活离群值，可被 RotationConfig 中的 block_size 覆盖。
    """

    type: Literal["online_quarot"] = Field(default="online_quarot", description="处理器类型，固定为 `online_quarot`。")
    include: Optional[List[Annotated[str, AfterValidator(validate_str_length())]]] = Field(
        default=None, description="包含的模块名称模式；不设置表示全部匹配。"
    )
    exclude: Optional[List[Annotated[str, AfterValidator(validate_str_length())]]] = Field(
        default=None, description="排除的模块名称模式，优先级高于 `include`。"
    )
    block_size: int = Field(
        default=-1, description="旋转块大小，-1 表示按 hidden_dim 整块旋转；可被 RotationConfig 覆盖。"
    )

    @field_validator('block_size')
    @classmethod
    def validate_block_size(cls, v):
        """校验 block_size：取值范围为-1或2的非负整数次幂"""
        if v == -1:
            return v
        if v <= 0 or not is_power_of_two(v):
            raise SchemaValidateError(f"block_size must be -1 or a positive power of 2, got {v}")
        return v


def _get_full_module_name(target_module: nn.Module, request: BatchProcessRequest) -> str:
    """
    通过遍历request.module获取完整的模块名称。
    """
    for name, module in request.module.named_modules():
        if module is target_module:
            # 拼接完整路径：request.name + 相对路径
            full_name = f"{request.name}.{name}" if name else request.name
            return full_name

    # 如果找不到，抛出UnsupportedError
    raise UnsupportedError(f"Cannot find full module name for {target_module}")


def _convert_hookir_to_wrapper(model: nn.Module) -> None:
    """
    将模型中的HookIR转换为Wrapper

    Args:
        model: 要处理的模型
    """
    # 遍历模型中的所有子模块
    for name, sub_module in model.named_modules():
        if hasattr(sub_module, '_forward_pre_hooks'):
            # 遍历模块的所有前向钩子
            for hook in sub_module._forward_pre_hooks.values():
                # 检查是否是HookIR类型
                if isinstance(hook, qir.HookIR):
                    # 将hook_ir转换为wrapper
                    wrapper = hook.wrapper_module(sub_module)
                    # 将wrapper替换模块
                    model.set_submodule(name, wrapper)
                    get_logger().info("Converted %s to wrapper for module: %s", type(hook), name)

        if hasattr(sub_module, '_forward_hooks'):
            # 遍历模块的所有后向钩子
            for hook in sub_module._forward_hooks.values():
                # 检查是否是HookIR类型
                if isinstance(hook, qir.HookIR):
                    # 将hook_ir转换为wrapper
                    wrapper = hook.wrapper_module(sub_module)
                    # 将wrapper替换模块
                    model.set_submodule(name, wrapper)
                    get_logger().info("Converted %s to wrapper for module: %s", type(hook), name)


@QABCRegistry.register(dispatch_key=OnlineQuaRotProcessorConfig, abc_class=AutoSessionProcessor)
class OnlineQuaRotProcessor(AutoSessionProcessor):
    """
    简化的在线旋转处理器。

    支持四种旋转模式：
    - input: 在模块输入处旋转（使用 forward_pre_hook）

    - output: 在模块输出处旋转（使用 forward_hook）

    - replace: 替换模块，只执行旋转（使用 WrapperIR）

    - offline: 离线旋转，直接作用在模型参数上（使用 rotate_linear）

    """

    def __init__(
        self, model: nn.Module, config: OnlineQuaRotProcessorConfig, adapter: OnlineQuaRotInterface, **kwargs
    ) -> None:
        super().__init__(model)
        self.config = config
        self.model = model
        self.adapter = adapter

        if not isinstance(adapter, OnlineQuaRotInterface):
            raise UnsupportedError(
                f'{adapter.__class__.__name__} does not support OnlineQuaRotInterface',
                action='Please provide a valid model adapter which implements OnlineQuaRotInterface',
            )

        # 获取旋转配置（传递 model 以便 adapter 可以挂载模块）
        self.rotation_configs: Dict[str, RotationConfig] = adapter.get_online_rotation_configs(model)

        # 初始化 include/exclude 过滤
        self.include_set = ConfigSet(config.include) if config.include else ConfigSet(["*"])
        self.exclude_set = ConfigSet(config.exclude) if config.exclude else ConfigSet([])

        # 存储旋转信息
        self.rotation_infos: Dict[str, qir.OnlineRotationInfo] = {}

        # 初始化旋转策略映射
        self._rotation_strategies = self._get_rotation_strategies()

    def support_distributed(self) -> bool:
        return True

    def is_data_free(self) -> bool:
        return True

    def pre_run(self) -> None:
        """预处理阶段：初始化旋转矩阵"""
        get_logger().info("Initializing online rotation matrices")

        # 为每个配置生成或验证旋转矩阵
        for module_name, rot_config in self.rotation_configs.items():
            if not self._should_apply_rotation(module_name):
                continue

            # 获取或生成旋转矩阵
            if rot_config.rotation_matrix is not None:
                # 用户提供的旋转矩阵
                rotation_matrix = rot_config.rotation_matrix
            else:
                # 自动生成旋转矩阵
                if rot_config.rotation_mode is None:
                    raise UnsupportedError(
                        f"Rotation matrix for {module_name} is not provided and "
                        "rotation_mode is not specified in RotationConfig",
                        action="Please provide rotation_matrix or specify rotation_mode in RotationConfig",
                    )

                rotation_matrix = create_rot(
                    mode=rot_config.rotation_mode,
                    size=rot_config.rotation_size,
                    block_size=rot_config.block_size if rot_config.block_size != -1 else self.config.block_size,
                    rot_step=rot_config.rot_step,
                    eye_step=rot_config.eye_step,
                    seed=rot_config.seed,
                    dtype=rot_config.dtype,
                )

            # 将字符串类型的 rotation_type 转换为 RotationType 枚举
            rotation_type_enum = RotationType(rot_config.rotation_type)

            # 创建旋转信息对象
            rotation_info = qir.OnlineRotationInfo(
                rotation_matrix=rotation_matrix,
                rotation_type=rotation_type_enum,
                layer_name=module_name,
                rotation_side=rot_config.rotation_side,
            )
            self.rotation_infos[module_name] = rotation_info

        get_logger().info("Initialized %d rotation configurations", len(self.rotation_infos))

    def preprocess(self, request: BatchProcessRequest) -> None:
        """为每个层应用旋转"""
        prefix = request.name
        prefix = f"{prefix}." if prefix != "" else ""

        # 过滤出当前层范围内的旋转配置
        filtered_configs = {}
        for module_name, rot_config in self.rotation_configs.items():
            if module_name.startswith(prefix):
                # 移除前缀，获取相对名称
                relative_name = module_name[len(prefix) :] if prefix else module_name
                filtered_configs[relative_name] = (module_name, rot_config)

        # 为每个配置应用旋转
        for relative_name, (full_module_name, rot_config) in filtered_configs.items():
            if not self._should_apply_rotation(full_module_name):
                continue
            # 获取旋转信息
            rotation_info = self.rotation_infos[full_module_name]
            # 根据旋转类型应用旋转（模块获取逻辑移到各个策略函数中）
            self._apply_rotation(relative_name, full_module_name, rotation_info, request)

    def post_run(self) -> None:
        """后处理阶段：转换 HookIR 为 WrapperIR"""
        _convert_hookir_to_wrapper(self.model)
        get_logger().info("Converted all HookIR to WrapperIR")

    def _should_apply_rotation(self, module_name: str) -> bool:
        """
        检查是否应该对模块应用旋转。
        """
        # 检查 include
        if module_name not in self.include_set:
            return False

        # 检查 exclude（优先级高于 include）
        if module_name in self.exclude_set:
            return False

        return True

    def _get_rotation_strategies(self) -> Dict[RotationType, Callable]:
        """
        返回旋转类型到处理方法的映射
        """
        return {
            RotationType.INPUT: self._apply_input_rotation,
            RotationType.OUTPUT: self._apply_output_rotation,
            RotationType.REPLACE: self._apply_replace_rotation,
            RotationType.OFFLINE: self._apply_offline_rotation,
        }

    def _apply_input_rotation(
        self,
        relative_name: str,
        full_module_name: str,
        rotation_info: qir.OnlineRotationInfo,
        request: BatchProcessRequest,
    ) -> None:
        """
        处理 INPUT 类型旋转：在模块输入处旋转（使用 forward_pre_hook）。
        """
        # 获取模块
        module = request.module.get_submodule(relative_name)
        hook_ir = qir.OnlineRotationInputHookIR(full_module_name, rotation_info)
        hook_handle = module.register_forward_pre_hook(hook_ir)
        hook_ir.set_hook_handle(hook_handle)
        get_logger().debug("Applied input rotation to %s", full_module_name)

    def _apply_output_rotation(
        self,
        relative_name: str,
        full_module_name: str,
        rotation_info: qir.OnlineRotationInfo,
        request: BatchProcessRequest,
    ) -> None:
        """
        处理 OUTPUT 类型旋转：在模块输出处旋转（使用 forward_hook）。
        """
        # 获取模块
        module = request.module.get_submodule(relative_name)
        hook_ir = qir.OnlineRotationOutputHookIR(full_module_name, rotation_info)
        hook_handle = module.register_forward_hook(hook_ir)
        hook_ir.set_hook_handle(hook_handle)
        get_logger().debug("Applied output rotation to %s", full_module_name)

    def _apply_replace_rotation(
        self,
        relative_name: str,
        full_module_name: str,
        rotation_info: qir.OnlineRotationInfo,
        request: BatchProcessRequest,
    ) -> None:
        """
        处理 REPLACE 类型旋转：替换模块，只执行旋转（使用 WrapperIR）。
        1. 首先尝试获取目标模块并包装替换
        2. 如果目标模块不存在，在父模块中挂载 Identity 模块后再包装替换
        """
        # 计算相对于 request.module 的模块路径
        if request.name and full_module_name.startswith(request.name + "."):
            relative_path = full_module_name[len(request.name) + 1 :]
        else:
            relative_path = full_module_name

        try:
            # 尝试直接获取目标模块
            target_module = request.module.get_submodule(relative_name)
            wrapper = qir.OnlineRotationWrapper(target_module, full_module_name, rotation_info)
            request.module.set_submodule(relative_path, wrapper)
            get_logger().debug("Applied replace rotation to %s", full_module_name)

        except AttributeError:
            # 模块不存在，尝试在父模块中挂载 Identity
            parent_path, _, child_name = relative_name.rpartition(".")

            if not parent_path:  # 如果没有父模块路径
                get_logger().error("Cannot find parent module for %s", relative_name)
                return

            # 查找父模块
            parent_module = None
            for name, module in request.module.named_modules():
                if name.endswith(parent_path):
                    parent_module = module
                    break

            if not parent_module:
                get_logger().error("Parent module %s not found", parent_path)
                return

            # 挂载 Identity 模块
            parent_module.register_module(child_name, nn.Identity())
            identity_module = parent_module.get_submodule(child_name)

            # 包装 Identity 模块
            wrapper = qir.OnlineRotationWrapper(identity_module, child_name, rotation_info)
            request.module.set_submodule(relative_path, wrapper)
            get_logger().debug("Applied replace rotation to %s (with Identity module)", full_module_name)

    def _apply_offline_rotation(
        self,
        relative_name: str,
        full_module_name: str,
        rotation_info: qir.OnlineRotationInfo,
        request: BatchProcessRequest,
    ) -> None:
        """
        处理 OFFLINE 类型旋转：直接作用在模型参数上（使用 rotate_linear）。
        """
        # 获取模块
        module = request.module.get_submodule(relative_name)

        if not hasattr(module, 'weight') or module.weight is None or module.weight.dim() != 2:
            raise UnsupportedError(
                f"Offline rotation only supports modules with weight attribute and weight dimension is 2, "
                f"got {type(module)} for {full_module_name}",
                action="Please use offline rotation only on modules with weight attribute and weight dimension is 2",
            )

        # 获取旋转方向和矩阵
        rotation_side = rotation_info.rotation_side or "right"
        right_rotate = rotation_side == "right"
        rotation_matrix = rotation_info.rotation_matrix

        # 应用旋转到线性层权重
        try:
            rotate_linear(module, rotation_matrix, right_rotate=right_rotate)
            get_logger().debug("Applied offline %s rotation to %s", rotation_side, full_module_name)
        except UnsupportedError as e:
            raise UnsupportedError(
                f"Failed to apply offline rotation to {full_module_name}: {e}",
                action="Please check the rotation matrix size matches the weight dimensions",
            ) from e

    def _handle_unknown_rotation_type(self, rotation_type: RotationType, full_module_name: str) -> None:
        raise UnsupportedError(
            f"Unknown rotation type: {rotation_type}",
            action=f"Please use one of ['input', 'output', 'replace', 'offline'] for {full_module_name}",
        )

    def _apply_rotation(
        self,
        relative_name: str,
        full_module_name: str,
        rotation_info: qir.OnlineRotationInfo,
        request: BatchProcessRequest,
    ) -> None:
        """
        应用旋转到模块。

        根据 rotation_type 分发到对应的策略方法。子类可以通过重写
        `_get_rotation_strategies()` 来扩展新的旋转类型，或重写特定的
        策略方法来修改处理逻辑。
        """
        rotation_type = rotation_info.rotation_type
        strategy = self._rotation_strategies.get(rotation_type)

        if strategy:
            strategy(relative_name, full_module_name, rotation_info, request)
        else:
            self._handle_unknown_rotation_type(rotation_type, full_module_name)
