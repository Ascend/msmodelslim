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
import gc
from typing import Optional, Literal

import torch
from pydantic import Field
from torch import nn

from msmodelslim.ir.qal import QABCRegistry
from msmodelslim.core.base.protocol import BatchProcessRequest
from msmodelslim.processor.base import AutoSessionProcessor, AutoProcessorConfig
from msmodelslim.utils.logging import get_logger
from msmodelslim.utils.memory import get_module_param_size, get_device_allocated_memory, get_device_reserved_memory, \
    format_memory_size, register_device_alignment_hook, unregister_device_alignment_hook


class LoadProcessorConfig(AutoProcessorConfig):
    type: Literal["load"] = "load"
    device: str = Field(default="cpu", description="目标设备")
    non_blocking: bool = Field(default=False, description="是否非阻塞加载")
    mode: Literal['load', 'offload'] = Field(default="load", description="加载模式")
    cleanup: bool = Field(default=False, description="是否清理缓存")
    post_offload: bool = Field(default=False, description="是否offload激活值")


@QABCRegistry.register(dispatch_key=LoadProcessorConfig, abc_class=AutoSessionProcessor)
class LoadProcessor(AutoSessionProcessor):
    """
    模块加载处理器，用于将模块加载到指定设备上。
    
    该处理器在preprocess阶段将请求中的模块移动到指定的设备上，
    支持CPU、GPU等设备类型。
    """

    def __init__(
            self,
            model: nn.Module,
            config: LoadProcessorConfig,
            adapter: Optional[object] = None,
    ):
        super().__init__(model)
        self.config = config
        self.adapter = adapter
        self.device = config.device
        self.non_blocking = config.non_blocking

    def __repr__(self) -> str:
        return f"LoadProcessor(device={self.device}, non_blocking={self.non_blocking})"

    def support_distributed(self) -> bool:
        return True
        
    def preprocess(self, request: BatchProcessRequest) -> None:
        if self.config.mode == "load":
            self.to_device(request=request)

    def postprocess(self, request: BatchProcessRequest) -> None:
        if self.config.mode == "offload":
            self.to_device(request=request)

    def to_device(self, request: BatchProcessRequest) -> None:
        """
        在预处理阶段将模块移动到指定设备上。
        
        Args:
            request: 批量处理请求，包含需要处理的模块
        """
        if request.module is not None:
            # 将模块移动到指定设备

            get_logger().debug("Move {} with size {} to {}".format(
                request.name,
                format_memory_size(get_module_param_size(request.module)),
                self.device)
            )

            get_logger().debug("Before move: allocated={}, reserved={}".format(
                format_memory_size(get_device_allocated_memory()),
                format_memory_size(get_device_reserved_memory())
            ))

            if self.config.mode == "load":
                register_device_alignment_hook(request.module, with_kwargs=True, name=request.name,
                                               post_offload=self.config.post_offload)

            if self.config.mode == "offload":
                unregister_device_alignment_hook(request.module, name=request.name)

            request.module.to(torch.device(self.device), non_blocking=self.non_blocking)

            if self.config.cleanup:

                gc.collect()

                if hasattr(torch, 'cuda') and torch.cuda.is_available():
                    torch.cuda.empty_cache()

                try:
                    import torch_npu
                    torch_npu.npu.empty_cache()
                except ImportError:
                    get_logger().warning("torch_npu module not available, skipping NPU cache cleanup")
                except AttributeError as e:
                    get_logger().warning(f"torch_npu.npu.empty_cache() not available: {e}")
                except Exception as e:
                    get_logger().warning(f"Failed to clear NPU cache: {e}")

            get_logger().debug("After move: allocated={}, reserved={}".format(
                format_memory_size(get_device_allocated_memory()),
                format_memory_size(get_device_reserved_memory())
            ))

    def is_data_free(self) -> bool:
        """
        判断处理器是否需要数据。
        
        Returns:
            True，因为加载处理器不需要输入数据
        """
        return True
