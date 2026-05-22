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

from abc import abstractmethod
from collections.abc import Callable
from typing import Optional, Dict, Type, Any, Set, List, Literal

import torch.distributed as dist
from pydantic import SerializeAsAny
from pydantic.functional_validators import BeforeValidator
from torch import nn
from typing_extensions import Annotated

import msmodelslim.ir as qir
from msmodelslim.core.base.protocol import BatchProcessRequest
from msmodelslim.processor.base import AutoSessionProcessor, AutoProcessorConfig
from msmodelslim.utils.distributed import DistHelper
from msmodelslim.utils.logging import get_logger


class AutoSaverBaseConfig(AutoProcessorConfig):
    type: Literal['_auto_save'] = "_auto_save"

    @abstractmethod
    def set_save_directory(self, save_directory):
        pass


def validate_auto_saver_processor_config_list(v: Any) -> List['AutoProcessorConfig']:
    if isinstance(v, list):
        validated_configs = []
        for item in v:
            if isinstance(item, dict):
                validated_configs.append(AutoSaverBaseConfig.model_validate(item))
            elif isinstance(item, AutoSaverBaseConfig):
                validated_configs.append(item)
            else:
                raise ValueError(f"Invalid config item type: {type(item)}")
        if not isinstance(validated_configs[-1], AutoSaverBaseConfig):
            raise TypeError("The save config you provide is not a saver config")
        return validated_configs
    raise ValueError("Expected a list of AutoSaverBaseConfig or dict")


AutoSaverConfigList = Annotated[
    List[SerializeAsAny[AutoSaverBaseConfig]],
    BeforeValidator(validate_auto_saver_processor_config_list)
]


def _convert_hookir_to_wrapper(module: nn.Module) -> None:
    """
    将模块中的HookIR转换为Wrapper

    Args:
        module: 要处理的模块
    """
    # 遍历模块中的所有子模块
    for name, sub_module in module.named_modules():
        if hasattr(sub_module, '_forward_pre_hooks'):
            # 遍历模块的所有前向钩子
            for hook in sub_module._forward_pre_hooks.values():
                # 检查是否是HookIR类型
                if isinstance(hook, qir.HookIR):
                    # 将hook_ir转换为wrapper
                    wrapper = hook.wrapper_module(sub_module)
                    # 将wrapper替换模块
                    module.set_submodule(name, wrapper)
                    get_logger().info(f"Converted {type(hook)} to wrapper for module: {name}")


class AutoSaverProcessor(AutoSessionProcessor):

    def __init__(self, model: nn.Module, config: AutoProcessorConfig, adapter: object, **kwargs: Dict[str, Any]):
        super().__init__(model)
        self.config = config
        self.adapter = adapter
        self.dist_helper: Optional[DistHelper] = None
        self.processed_modules: Set[nn.Module] = set()
        self.process_map: Dict[Type[nn.Module], Callable[[str, nn.Module], None]] = {
            qir.W8A8StaticFakeQuantLinear: self.on_w8a8_static,
            qir.W8A16StaticPerChannelFakeQuantLinear: self.on_w8a16_static_per_channel,
            qir.W8A16StaticPerGroupFakeQuantLinear: self.on_w8a16_static_per_group,
            qir.W8A8DynamicPerChannelFakeQuantLinear: self.on_w8a8_dynamic_per_channel,
            qir.W8A8PDMixFakeQuantLinear: self.on_w8a8_pd_mix,
            qir.W8A8DynamicPerGroupFakeQuantLinear: self.on_w8a8_dynamic_per_group,
            qir.W4A4DynamicPerChannelFakeQuantLinear: self.on_w4a4_dynamic_per_channel,
            qir.W4A4DynamicPerGroupFakeQuantLinear: self.on_w4a4_dynamic_per_group,
            qir.W8A8MXDynamicPerBlockFakeQuantLinear: self.on_w8a8_mx_dynamic_per_block,
            qir.W4A8MXDynamicPerBlockFakeQuantLinear: self.on_w4a8_mx_dynamic_per_block,
            qir.W4A4MXDynamicPerBlockFakeQuantLinear: self.on_w4a4_mx_dynamic_per_block,
            qir.W4A4MXDynamicDualScaleFakeQuantLinear: self.on_w4a4_mx_dynamic_dual_scale,
            qir.W4A8DynamicFakeQuantLinear: self.on_w4a8_dynamic,
            nn.Linear: self.on_float_linear,
            nn.Module: self.on_float_module,
            qir.FakeQuantDynamicCache: self.on_dynamic_cache,
            qir.FakeQuantActivationPerHead: self.on_activation_per_head,
            qir.FakeQuantActivationPerToken: self.on_activation_per_token,
            qir.W16A16sLinear: self.on_w16a16s,
            qir.QuarotOnlineHeadRotationWrapper: self.on_rotation_wrapper,
            qir.QuarotOnlineKroneckerRotationWrapper: self.on_kronecker_rotation_wrapper,
            qir.QuaRotExtraInfoWrapperIR: self.on_quarot_extra_info_wrapper,
            qir.OnlineRotationWrapper: self.on_online_rotation_wrapper,
            qir.WFP8AFP8DynamicPerChannelFakeQuantLinear: self.on_wfp8afp8_dynamic_per_channel,
            qir.FlatQuantOnlineWrapper: self.on_flat_clip_wrapper,
            qir.NonFusionSmoothQuantWrapper: self.on_non_fusion_smooth_quant_wrapper,
        }

    def support_distributed(self) -> bool:

        """
        是否支持分布式校准模式下保存导出件。

        目前分布式校准仅使用DP模式。

        分布式校准启用后，每个rank都会实例化一个AutoSaverProcessor，每个rank的
        实例将会收到全量的on_xxx系列调用，子类应当自行筛选属于本rank的内容进行保存。
        
        最后，在post_run中，会调用一次merge_ranks方法，用于合并各个rank的导出件。
        """

        return False

    def is_data_free(self) -> bool:
        return True

    def pre_run(self) -> None:
        pass

    def post_run(self) -> None:

        for name, sub_module in self.model.named_modules(memo=self.processed_modules):
            self.on_float_module(name, sub_module)

        if self.support_distributed() and dist.is_initialized():
            self.merge_ranks()

    def postprocess(self, request: BatchProcessRequest) -> None:
        prefix, module = request.name, request.module

        # 处理hookir转换
        _convert_hookir_to_wrapper(module)

        for name, sub_module in module.named_modules(memo=self.processed_modules, prefix=prefix):
            self._process_module_maybe_wrapper_ir(name, sub_module)

    @abstractmethod
    def merge_ranks(self) -> None:

        """
        合并各个rank所保存的导出件。
        """
        pass

    def on_w8a8_static(self, prefix: str, module: qir.W8A8StaticFakeQuantLinear):
        raise NotImplementedError(f"You should implement the on_w8a8_static method for {self.__class__.__name__}")

    def on_w8a16_static_per_channel(self, prefix: str, module: qir.W8A16StaticPerChannelFakeQuantLinear):
        raise NotImplementedError(f"You should implement the on_w8a16_static_per_channel method for {self.__class__.__name__}")

    def on_w8a16_static_per_group(self, prefix: str, module: qir.W8A16StaticPerGroupFakeQuantLinear):
        raise NotImplementedError(f"You should implement the on_w8a16_static_per_group method for {self.__class__.__name__}")

    def on_w8a8_dynamic_per_channel(self, prefix: str, module: qir.W8A8DynamicPerChannelFakeQuantLinear):
        raise NotImplementedError(
            f"You should implement the on_w8a8_dynamic_per_channel method for {self.__class__.__name__}")

    def on_w8a8_pd_mix(self, prefix: str, module: qir.W8A8PDMixFakeQuantLinear):
        raise NotImplementedError(
            f"You should implement the on_w8a8_pd_mix method for {self.__class__.__name__}")

    def on_w8a8_dynamic_per_group(self, prefix: str, module: qir.W8A8DynamicPerGroupFakeQuantLinear):
        raise NotImplementedError(
            f"You should implement the on_w8a8_dynamic_per_group method for {self.__class__.__name__}")

    def on_wfp8afp8_dynamic_per_channel(self, prefix: str, module: qir.WFP8AFP8DynamicPerChannelFakeQuantLinear):
        raise NotImplementedError(
            f"You should implement the on_wfp8afp8_dynamic_per_channel method for {self.__class__.__name__}")

    def on_w4a4_dynamic_per_channel(self, prefix: str, module: qir.W4A4DynamicPerChannelFakeQuantLinear):
        raise NotImplementedError(
            f"You should implement the on_w4a4_dynamic_per_channel method for {self.__class__.__name__}")

    def on_w4a4_dynamic_per_group(self, prefix: str, module: qir.W4A4DynamicPerGroupFakeQuantLinear):
        raise NotImplementedError(
            f"You should implement the on_w4a4_dynamic_per_group method for {self.__class__.__name__}")

    def on_w4a4_mx_dynamic_per_block(self, prefix: str, module: qir.W4A4MXDynamicPerBlockFakeQuantLinear):
        raise NotImplementedError(
            f"You should implement the on_w4a4_mx_dynamic_per_block method for {self.__class__.__name__}")

    def on_w8a8_mx_dynamic_per_block(self, prefix: str, module: qir.W8A8MXDynamicPerBlockFakeQuantLinear):
        raise NotImplementedError(
            f"You should implement the on_w8a8_mx_dynamic_per_block method for {self.__class__.__name__}")

    def on_w4a8_mx_dynamic_per_block(self, prefix: str, module: qir.W4A8MXDynamicPerBlockFakeQuantLinear):
        raise NotImplementedError(
            f"You should implement the on_w4a8_mx_dynamic_per_block method for {self.__class__.__name__}")

    def on_w4a4_mx_dynamic_dual_scale(self, prefix: str, module: qir.W4A4MXDynamicDualScaleFakeQuantLinear):
        raise NotImplementedError(
            f"You should implement the on_w4a4_mx_dynamic_dual_scale method for {self.__class__.__name__}")

    def on_w4a8_dynamic(self, prefix: str, module: qir.W4A8DynamicFakeQuantLinear):
        raise NotImplementedError(f"You should implement the on_w4a8_dynamic method for {self.__class__.__name__}")

    def on_float_linear(self, prefix: str, module: nn.Linear):
        raise NotImplementedError(f"You should implement the on_linear method for {self.__class__.__name__}")

    def on_float_module(self, prefix: str, module: nn.Module):
        raise NotImplementedError(f"You should implement the on_float_module method for {self.__class__.__name__}")

    def on_dynamic_cache(self, prefix: str, module: qir.FakeQuantDynamicCache):
        raise NotImplementedError(f"You should implement the on_dynamic_cache method for {self.__class__.__name__}")
    
    def on_activation_per_head(self, prefix: str, module: qir.FakeQuantActivationPerHead):
        raise NotImplementedError(
            f"You should implement the on_activation_per_head method for {self.__class__.__name__}"
            )
    
    def on_activation_per_token(self, prefix: str, module: qir.FakeQuantActivationPerToken):
        raise NotImplementedError(
            f"You should implement the on_activation_per_token method for {self.__class__.__name__}"
            )

    def on_rotation_wrapper(self, prefix: str, module: qir.QuarotOnlineHeadRotationWrapper):
        """
        处理RotationWrapper类型的模块。

        RotationWrapper使用全局共享的旋转矩阵，在保存时只需要处理被包装的模块，
        旋转矩阵本身会在全局保存逻辑中处理。

        Args:
            prefix: 模块名称前缀
            module: RotationWrapper模块实例
        """
        # 只处理被包装的模块，旋转矩阵由全局逻辑处理
        self._process_module(prefix, module.wrapped_module)

    def on_kronecker_rotation_wrapper(self, prefix: str, module: qir.QuarotOnlineKroneckerRotationWrapper):
        """
        处理KroneckerRotationWrapper类型的模块。

        KroneckerRotationWrapper使用全局共享的旋转矩阵，在保存时只需要处理被包装的模块，
        旋转矩阵本身会在全局保存逻辑中处理。

        Args:
            prefix: 模块名称前缀
            module: KroneckerRotationWrapper模块实例
        """
        # 只处理被包装的模块，旋转矩阵由全局逻辑处理
        self._process_module(prefix, module.wrapped_module)

    def on_quarot_extra_info_wrapper(self, prefix: str, module: qir.QuaRotExtraInfoWrapperIR):
        """
        处理 QuaRot 额外信息（全局旋转矩阵等）的导出。
        子类应实现：将 rotation_info 写入独立 safetensors，并在 JSON 中写入 optional.quarot.global_rotation 路径。
        """
        raise NotImplementedError(
            f"You should implement the on_quarot_extra_info_wrapper method for {self.__class__.__name__}")

    def on_flat_clip_wrapper(self, prefix: str, module: qir.FlatQuantOnlineWrapper):
        """
        处理FlatQuantOnlineWrapper类型的模块。

        保存旋转矩阵到left_trans, right_trans, clip_factor, 并在JSON中添加相应的描述。

        Args:
            prefix: 模块名称前缀
            module: FlatQuantOnlineWrapper模块实例
        """
        self._process_module(prefix, module.wrapped_module)
    
    def on_non_fusion_smooth_quant_wrapper(self, prefix: str, module: qir.NonFusionSmoothQuantWrapper):
        """
        Args:
            prefix: 模块名称前缀
            module: NonFusionSmoothQuantWrapper模块实例
        """
        raise NotImplementedError(
            f"You should implement the on_non_fusion_smooth_quant_wrapper method for {self.__class__.__name__}")

    def on_wrapper_ir(self, prefix: str, module: qir.WrapperIR):
        """
        处理WrapperIR类型的模块。

        WrapperIR是一个包装器，它持有一个被包装的nn.Module。在保存时，
        根据包装器的原子性决定处理策略。

        处理策略：
        - 如果包装器不是原子性的（is_atomic()返回False），先处理被包装模块，再处理包装器自身
        - 如果包装器是原子性的（is_atomic()返回True），只处理包装器自身，跳过被包装模块

        这样设计的好处：
        - 支持原子性和非原子性包装器的不同处理需求
        - 原子性包装器作为整体处理，避免重复处理
        - 非原子性包装器可以分别处理被包装模块和包装器自身

        Args:
            prefix: 模块名称前缀
            module: WrapperIR模块实例
        """
        # 根据原子性决定处理策略
        wrapped_module = module.wrapped_module
        if not module.is_atomic():
            # 非原子性：先处理被包装模块，再处理包装器自身
            self._process_module(prefix, wrapped_module)
        # 处理包装器自身
        self._process_module(prefix, module)

    def on_w16a16s(self, prefix: str, module: qir.W16A16sLinear):
        raise NotImplementedError(f"You should implement the on_w16a16s method for {self.__class__.__name__}")

    def on_online_rotation_wrapper(self, prefix: str, module: qir.OnlineRotationWrapper):
        """
        处理OnlineRotationWrapper类型的模块。
        
        OnlineRotationWrapper使用旋转矩阵替换模块，在保存时需要保存旋转矩阵。
        
        Args:
            prefix: 模块名称前缀
            module: OnlineRotationWrapper模块实例
        """
        raise NotImplementedError(
            f"You should implement the on_online_rotation_wrapper method for {self.__class__.__name__}"
        )

    def _process_module(self, prefix: str, module: nn.Module):
        """
        使用process_map处理模块的通用方法。

        Args:
            prefix: 模块名称前缀
            module: 要处理的模块
        """

        get_logger().debug("Processing module %r for %r", type(module).__name__, prefix)

        if type(module) in self.process_map:
            self.process_map[type(module)](prefix, module)
        else:
            self.on_float_module(prefix, module)

        self.processed_modules.add(module)
        
    def _process_module_maybe_wrapper_ir(self, prefix: str, module: nn.Module):
        """
        处理模块的通用方法，如果模块是WrapperIR，则调用on_wrapper_ir方法，否则调用_process_module方法。
        
        Args:
            prefix: 模块名称前缀
            module: 要处理的模块
        """
        if isinstance(module, qir.WrapperIR):
            self.on_wrapper_ir(prefix, module)
        else:
            self._process_module(prefix, module)
