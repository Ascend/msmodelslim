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
# pylint: disable=attribute-defined-outside-init

from typing import Any, Dict, List, Optional, Tuple
from enum import Enum
from dataclasses import dataclass

import torch
from torch import nn

from msmodelslim.utils.logging import logger_setter
from .wrapper import WrapperIR, HookIR


class QuarotOnlineRotationInfo:
    """
    Quarot旋转矩阵信息。

    该类负责管理全局共享的旋转矩阵和层索引信息。
    """

    def __init__(
        self,
        rotation_o_proj: Optional[torch.Tensor],
        rotation_o_proj_eye: Optional[torch.Tensor],
        rotation_down_proj_m: Optional[torch.Tensor],
        rotation_down_proj_n: Optional[torch.Tensor],
        max_tp_size: int,
    ):
        """
        初始化QuarotRotationInfo。

        Args:
            rotation_o_proj: 普通旋转矩阵
            rotation_down_proj_m: Kronecker旋转矩阵M
            rotation_down_proj_n: Kronecker旋转矩阵N
        """
        self.heads_rotation = rotation_o_proj
        self.heads_rotation_eye = rotation_o_proj_eye
        self.kronecker_rotation_m = rotation_down_proj_m
        self.kronecker_rotation_n = rotation_down_proj_n
        self.max_tp_size = max_tp_size

        self.heads_rotation_layers: List[str] = []
        self.kronecker_rotation_layers: List[str] = []

    def add_rotation_layer(self, layer_name: str) -> None:
        """添加使用全局旋转矩阵的层名称。"""
        self.heads_rotation_layers.append(layer_name)

    def add_kronecker_rotation_layer(self, layer_name: str) -> None:
        """添加使用全局Kronecker旋转矩阵的层名称。"""
        self.kronecker_rotation_layers.append(layer_name)

    def get_quarot_save_info(self) -> Dict[str, Any]:
        """
        获取quarot相关的保存信息。

        Returns:
            包含旋转矩阵和层名称的字典
        """
        return {
            "max_tp_size": self.max_tp_size,
            "heads_rotation": {"layers": self.heads_rotation_layers.copy()},
            "kronecker_rotation": {"layers": self.kronecker_rotation_layers.copy()},
        }


class QuarotOfflineRotationInfo:
    """
    QuaRot 离线/导出用旋转矩阵信息。
    仅包含 global_rotation 一个成员（torch.Tensor），
    供 QuaRotExtraInfoHookIR / QuaRotExtraInfoWrapperIR 与 Saver 导出到 optional.quarot.global_rotation。
    """

    def __init__(self, global_rotation: torch.Tensor):
        self.global_rotation = global_rotation


@logger_setter()
class QuarotOnlineHeadRotationWrapper(WrapperIR):
    """
    直接进行旋转运算的包装器。

    该类继承自WrapperIR，包装AutoFakeQuantLinear实例，使用全局共享的旋转矩阵，
    在forward前添加旋转运算。
    """

    def __init__(self, module: nn.Module, layer_name: str, rotation_info: QuarotOnlineRotationInfo):
        """
        初始化RotationWrapper包装器。

        Args:
            module: 被包装的AutoFakeQuantLinear实例
            layer_name: 层名称，用于保存时标识
            rotation_info: 旋转矩阵信息
        """
        super().__init__(module)
        self.layer_name = layer_name
        self.rotation_info = rotation_info
        self.rotation_info.add_rotation_layer(layer_name)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播，在AutoFakeQuantLinear前添加旋转运算。

        Args:
            x: 输入张量

        Returns:
            经过旋转运算和线性变换的输出张量
        """
        x_rotated = self._apply_rotation(x)
        return self.wrapped_module(x_rotated)

    def extra_repr(self) -> str:
        """
        返回额外的字符串表示，描述旋转矩阵信息。

        Returns:
            包含旋转矩阵信息的字符串
        """
        rot_1 = self.rotation_info.heads_rotation
        rot_2 = self.rotation_info.heads_rotation_eye

        return f"heads_rotation(Q:{rot_1.shape[0]}x{rot_1.shape[1]}, I:{rot_2.shape[0]}x{rot_2.shape[1]})"

    def _apply_rotation(self, x: torch.Tensor) -> torch.Tensor:
        """
        应用旋转运算。

        Args:
            x: 输入张量

        Returns:
            旋转后的张量
        """
        rot_1 = self.rotation_info.heads_rotation
        rot_2 = self.rotation_info.heads_rotation_eye
        dtype = x.dtype
        device = x.device
        init_shape = x.shape
        scaled_x = x.reshape(-1, rot_1.shape[0], rot_2.shape[0])
        scaled_x = torch.matmul(rot_1.to(device, dtype).T, scaled_x).reshape(init_shape)
        return scaled_x.reshape(init_shape)


@logger_setter()
class QuarotOnlineKroneckerRotationWrapper(WrapperIR):
    """
    按Kronecker Product方式进行旋转的包装器。

    该类继承自WrapperIR，包装AutoFakeQuantLinear实例，使用全局共享的旋转矩阵，
    通过Kronecker Product组合后进行旋转运算。
    """

    def __init__(self, module: nn.Module, layer_name: str, rotation_info: QuarotOnlineRotationInfo):
        """
        初始化KroneckerRotationWrapper包装器。

        Args:
            module: 被包装的AutoFakeQuantLinear实例
            layer_name: 层名称，用于保存时标识
            rotation_info: 旋转矩阵信息
        """
        super().__init__(module)
        self.layer_name = layer_name
        self.rotation_info = rotation_info
        self.rotation_info.add_kronecker_rotation_layer(layer_name)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播，在AutoFakeQuantLinear前添加Kronecker旋转运算。

        Args:
            x: 输入张量

        Returns:
            经过Kronecker旋转运算和线性变换的输出张量
        """
        x_rotated = self._apply_kronecker_rotation(x)
        return self.wrapped_module(x_rotated)

    def extra_repr(self) -> str:
        """
        返回额外的字符串表示，描述Kronecker旋转矩阵信息。

        Returns:
            包含Kronecker旋转矩阵信息的字符串
        """
        rot_1 = self.rotation_info.kronecker_rotation_m
        rot_2 = self.rotation_info.kronecker_rotation_n

        return f"kronecker_rotation(M:{rot_1.shape[0]}x{rot_1.shape[1]}, N:{rot_2.shape[0]}x{rot_2.shape[1]})"

    def _apply_kronecker_rotation(self, x: torch.Tensor) -> torch.Tensor:
        """
        应用Kronecker Product旋转运算。

        Args:
            x: 输入张量

        Returns:
            Kronecker旋转后的张量
        """
        rot_1 = self.rotation_info.kronecker_rotation_m
        rot_2 = self.rotation_info.kronecker_rotation_n
        dtype = x.dtype
        device = x.device
        init_shape = x.shape
        scaled_x = x.reshape(-1, rot_1.shape[0], rot_2.shape[0])
        scaled_x = torch.matmul(scaled_x, rot_2.to(device, dtype))
        scaled_x = torch.matmul(rot_1.to(device, dtype).T, scaled_x).reshape(init_shape)
        return scaled_x.reshape(init_shape)


@logger_setter()
class QuarotHeadsRotationHookIR(HookIR):
    """
    Quarot专用的HookIR实现，用于替代直接使用register_forward_pre_hook。

    该类实现了HookIR抽象基类，将hook信息转换为QuarotOnlineRotationWrapper。
    """

    def __init__(self, layer_name: str, rotation_info: QuarotOnlineRotationInfo):
        """
        初始化QuarotHookIR。

        Args:
            layer_name: 层名称
            rotation_info: 旋转矩阵信息
        """
        super().__init__()
        self.layer_name = layer_name
        self.rotation_info = rotation_info

    def __call__(
        self,
        module: nn.Module,
        args: Tuple[Any, ...],
    ) -> Tuple[Any, ...]:
        """
        实现Callable接口，作为hook函数被调用。

        Args:
            module: 被hook的模块
            args: 模块的输入参数

        Returns:
            处理后的输入参数和关键字参数
        """
        # 执行普通旋转运算
        dtype = args[0].dtype
        device = args[0].device
        x = args[0]
        init_shape = x.shape

        # 应用旋转运算
        rot_1 = self.rotation_info.heads_rotation
        rot_2 = self.rotation_info.heads_rotation_eye

        scaled_x = x.reshape(-1, rot_1.shape[0], rot_2.shape[0])
        scaled_x = torch.matmul(rot_1.to(device, dtype).T, scaled_x).reshape(init_shape)

        return (scaled_x,) + args[1:]

    def wrapper_module(self, module: nn.Module) -> WrapperIR:
        """
        实现HookIR抽象方法，返回QuarotOnlineRotationWrapper。

        Args:
            module: 要包装的模块

        Returns:
            QuarotOnlineRotationWrapper实例
        """
        # 将hook信息转换为WrapperIR
        self.remove_hook()
        return QuarotOnlineHeadRotationWrapper(module, self.layer_name, self.rotation_info)


@logger_setter()
class QuarotKroneckerRotationHookIR(HookIR):
    """
    Quarot专用的Kronecker HookIR实现，用于down_proj的Kronecker旋转。

    该类实现了HookIR抽象基类，将hook信息转换为QuarotOnlineKroneckerRotationWrapper。
    """

    def __init__(self, layer_name: str, rotation_info: QuarotOnlineRotationInfo):
        """
        初始化QuarotKroneckerHookIR。

        Args:
            layer_name: 层名称
            rotation_info: 旋转矩阵信息
        """
        super().__init__()
        self.layer_name = layer_name
        self.rotation_info = rotation_info

    def __call__(
        self,
        module: nn.Module,
        args: Tuple[Any, ...],
    ) -> Tuple[Any, ...]:
        """
        实现Callable接口，作为hook函数被调用。

        Args:
            module: 被hook的模块
            args: 模块的输入

        Returns:
            处理后的输入
        """
        # 执行Kronecker旋转运算
        dtype = args[0].dtype
        device = args[0].device
        x = args[0]
        init_shape = x.shape

        # 使用rotation_info中的Kronecker旋转矩阵进行运算
        rot_1 = self.rotation_info.kronecker_rotation_m
        rot_2 = self.rotation_info.kronecker_rotation_n

        scaled_x = x.reshape(-1, rot_1.shape[0], rot_2.shape[0])
        scaled_x = torch.matmul(scaled_x, rot_2.to(device, dtype))
        scaled_x = torch.matmul(rot_1.to(device, dtype).T, scaled_x).reshape(init_shape)

        return (scaled_x,) + args[1:]

    def wrapper_module(self, module: nn.Module) -> WrapperIR:
        """
        实现HookIR抽象方法，返回QuarotOnlineKroneckerRotationWrapper。

        Args:
            module: 要包装的模块

        Returns:
            QuarotOnlineKroneckerRotationWrapper实例
        """
        # 将hook信息转换为WrapperIR
        self.remove_hook()
        return QuarotOnlineKroneckerRotationWrapper(module, self.layer_name, self.rotation_info)


@logger_setter()
class QuaRotExtraInfoHookIR(HookIR):
    """
    QuaRot 导出用额外信息的 HookIR。
    用于在量化过程中不改变模型树路径的前提下，承载全局旋转矩阵等需导出到
    optional.quarot.global_rotation 的信息；导出时由 Saver 转为 QuaRotExtraInfoWrapperIR 处理。
    接受 QuarotOfflineRotationInfo。
    """

    def __init__(self, rotation_info: QuarotOfflineRotationInfo):
        super().__init__()
        self.rotation_info = rotation_info

    def __call__(
        self,
        module: nn.Module,
        args: Tuple[Any, ...],
    ) -> Tuple[Any, ...]:
        """Hook 不改变前向结果，仅承载信息。"""
        return args

    def wrapper_module(self, module: nn.Module) -> "QuaRotExtraInfoWrapperIR":
        self.remove_hook()
        return QuaRotExtraInfoWrapperIR(module, self.rotation_info)


@logger_setter()
class QuaRotExtraInfoWrapperIR(WrapperIR):
    """
    QuaRot 导出用额外信息的 WrapperIR。
    承载 QuarotOfflineRotationInfo（全局旋转矩阵等），供 AscendV1Saver 写入
    optional.quarot.global_rotation 的 JSON 与独立 safetensors 文件。
    额外导出模式（is_atomic() 为 False）：先按 prefix 单独导出 wrapped_module，再单独处理本 WrapperIR 自身。
    """

    def __init__(self, wrapped_module: nn.Module, rotation_info: QuarotOfflineRotationInfo):
        super().__init__(wrapped_module)
        self.rotation_info = rotation_info

    @staticmethod
    def is_atomic() -> bool:
        return False

    def forward(self, *args, **kwargs) -> Any:
        return self.wrapped_module(*args, **kwargs)


class RotationType(str, Enum):
    INPUT = "input"
    OUTPUT = "output"
    REPLACE = "replace"
    OFFLINE = "offline"


@dataclass
class OnlineRotationInfo:
    """
    简化的在线旋转矩阵信息类。

    该类负责管理旋转矩阵和配置信息，支持四种旋转模式：
    - input: 在模块输入处旋转
    - output: 在模块输出处旋转
    - replace: 替换模块，只执行旋转
    - offline: 离线旋转，直接作用在模型参数上

    Args:
        rotation_matrix: 旋转矩阵 R
        rotation_type: 旋转类型 ("input", "output", "replace", "offline")
        layer_name: 层名称
        rotation_side: 旋转方向 ("left" 或 "right")，仅用于 offline 类型
    """

    rotation_matrix: torch.Tensor
    rotation_type: RotationType  # "input", "output", "replace", "offline"
    layer_name: str
    rotation_side: Optional[str] = None  # "left" or "right"，仅用于 offline 类型


class BaseOnlineRotation:
    def _init_rotation_common(self, layer_name: str, rotation_info: OnlineRotationInfo) -> None:
        """
        Args:
            layer_name: 层名称
            rotation_info: 旋转矩阵信息
        """
        self.layer_name = layer_name
        self.rotation_info = rotation_info

    def _apply_rotation(self, x: torch.Tensor) -> torch.Tensor:
        """
        应用旋转运算 x @ R.T。
        """
        rot = self.rotation_info.rotation_matrix
        dtype = x.dtype
        device = x.device
        init_shape = x.shape

        # 确保 x 的最后一个维度与旋转矩阵匹配
        # x: [..., dim] -> [..., dim] @ R.T
        if len(init_shape) == 1:
            # 1D tensor: [dim]
            x_reshaped = x.unsqueeze(0)  # [1, dim]
            rotated = torch.matmul(x_reshaped, rot.T.to(device, dtype))
            return rotated.squeeze(0)
        elif len(init_shape) == 2:
            # 2D tensor: [batch, dim]
            rotated = torch.matmul(x, rot.T.to(device, dtype))
            return rotated
        else:
            # 3D+ tensor: [..., batch, dim] -> flatten last two dims
            x_flat = x.reshape(-1, init_shape[-1])
            rotated_flat = torch.matmul(x_flat, rot.T.to(device, dtype))
            return rotated_flat.reshape(*init_shape[:-1], rotated_flat.shape[-1])


@logger_setter()
class OnlineRotationWrapper(BaseOnlineRotation, WrapperIR):
    """
    通用的在线旋转包装器，用于替换模块。

    该类继承自WrapperIR，用于替换模块，只执行旋转运算而不调用原模块。
    """

    def __init__(self, module: nn.Module, layer_name: str, rotation_info: OnlineRotationInfo):
        """
        初始化OnlineRotationWrapper。

        Args:
            module: 被包装的模块（用于保存时保留原模块信息）
            layer_name: 层名称，用于保存时标识
            rotation_info: 旋转矩阵信息
        """
        WrapperIR.__init__(self, module)
        self._init_rotation_common(layer_name, rotation_info)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._apply_rotation(x)

    def extra_repr(self) -> str:
        rot = self.rotation_info.rotation_matrix
        return f"online_rotation(type:{self.rotation_info.rotation_type}, R:{rot.shape[0]}x{rot.shape[1]})"


@logger_setter()
class OnlineRotationInputHookIR(BaseOnlineRotation, HookIR):
    """
    通用的在线旋转输入HookIR实现，用于输入旋转。

    该类实现了HookIR抽象基类，支持在模块输入处应用旋转（forward_pre_hook）。
    """

    def __init__(self, layer_name: str, rotation_info: OnlineRotationInfo):
        """
        初始化OnlineRotationInputHookIR。

        Args:
            layer_name: 层名称
            rotation_info: 旋转矩阵信息
        """
        HookIR.__init__(self)
        self._init_rotation_common(layer_name, rotation_info)

    def __call__(
        self,
        module: nn.Module,
        args: Tuple[Any, ...],
    ) -> Tuple[Any, ...]:
        # 输入旋转：x @ R.T
        if len(args) == 0:
            raise ValueError("Input rotation hook requires input arguments")
        x = args[0]
        rotated_x = self._apply_rotation(x)
        return (rotated_x,) + args[1:]

    def wrapper_module(self, module: nn.Module) -> WrapperIR:
        self.remove_hook()

        # 对于input类型，创建一个包装器，在forward中应用旋转但保留原模块功能
        class RotationInputModuleWrapper(BaseOnlineRotation, WrapperIR):
            def __init__(self, wrapped_module, layer_name, rotation_info):
                WrapperIR.__init__(self, wrapped_module)
                self._init_rotation_common(layer_name, rotation_info)

            def forward(self, *args, **kwargs):
                # 输入旋转：先旋转输入，再执行原模块
                if len(args) > 0:
                    rotated_input = self._apply_rotation(args[0])
                    new_args = (rotated_input,) + args[1:]
                    return self.wrapped_module(*new_args, **kwargs)
                else:
                    raise ValueError("Input rotation hook requires input arguments")

        return RotationInputModuleWrapper(module, self.layer_name, self.rotation_info)


@logger_setter()
class OnlineRotationOutputHookIR(BaseOnlineRotation, HookIR):
    """
    通用的在线旋转输出HookIR实现，用于输出旋转。

    该类实现了HookIR抽象基类，支持在模块输出处应用旋转（forward_hook）。
    """

    def __init__(self, layer_name: str, rotation_info: OnlineRotationInfo):
        """
        初始化OnlineRotationOutputHookIR。

        Args:
            layer_name: 层名称
            rotation_info: 旋转矩阵信息
        """
        HookIR.__init__(self)
        self._init_rotation_common(layer_name, rotation_info)

    def __call__(
        self,
        module: nn.Module,
        args: Tuple[Any, ...],
        output: torch.Tensor,
    ) -> torch.Tensor:
        # 输出旋转：output @ R.T
        rotated_output = self._apply_rotation(output)
        return rotated_output

    def wrapper_module(self, module: nn.Module) -> WrapperIR:
        self.remove_hook()

        # 对于output类型，创建一个包装器，在forward中应用旋转但保留原模块功能
        class RotationOutputModuleWrapper(BaseOnlineRotation, WrapperIR):
            def __init__(self, wrapped_module, layer_name, rotation_info):
                WrapperIR.__init__(self, wrapped_module)
                self._init_rotation_common(layer_name, rotation_info)

            def forward(self, *args, **kwargs):
                # 输出旋转：先执行原模块，再旋转输出
                output = self.wrapped_module(*args, **kwargs)
                if isinstance(output, tuple):
                    rotated_output = self._apply_rotation(output[0])
                    return (rotated_output,) + output[1:]
                return self._apply_rotation(output)

        return RotationOutputModuleWrapper(module, self.layer_name, self.rotation_info)
