#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# pylint: disable=duplicate-code

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

import functools
import inspect
import json
import os
import shutil
from typing import Dict, Any, List, Literal

import torch
import torch.distributed as dist
from torch import nn

from ascend_utils.common.security.path import json_safe_dump
from msmodelslim import logger
from msmodelslim.core.base.protocol import BatchProcessRequest
from msmodelslim.ir.qal.qregistry import QABCRegistry
from msmodelslim.processor.base import AutoSessionProcessor
from msmodelslim.utils.distributed import DistHelper, get_distributed_task_work_queue
from msmodelslim.utils.logging import logger_setter, get_logger
from .ascendv1 import AscendV1Saver, AscendV1Config, ASCENDV1_DESC_JSON_NAME, copy_files, remove_quantization_config
from .interface import AscendV1SaveInterface
from .saver import _convert_hookir_to_wrapper
from .utils.json import JsonWriter


class DistributedAscendV1Config(AscendV1Config):
    """
    支持分布式保存的 ascendV1 量化模型保存器配置。

    该类继承自 AscendV1Config，用于在分布式模式下自动选择 DistributedAscendV1Saver。
    """

    type: Literal['ascendv1_saver_distributed'] = "ascendv1_saver_distributed"


def convert_to_distributed_config_if_needed(configs: List[AscendV1Config]) -> List:
    """
    如果启用了分布式，将列表中的 AscendV1Config 转换为 DistributedAscendV1Config

    Args:
        configs: 配置对象列表

    Returns:
        转换后的配置对象列表
    """
    if not dist.is_initialized() or not configs:
        return configs

    converted_configs = []
    for cfg in configs:
        if isinstance(cfg, AscendV1Config) and not isinstance(cfg, DistributedAscendV1Config):
            distributed_cfg = DistributedAscendV1Config(
                type="ascendv1_saver_distributed",
                save_directory=cfg.save_directory,
                part_file_size=cfg.part_file_size,
                ext=cfg.ext,
            )
            converted_configs.append(distributed_cfg)
            logger.info("Converted AscendV1Config to DistributedAscendV1Config for distributed saving")
        else:
            converted_configs.append(cfg)

    return converted_configs


def save_this_rank_only():
    """
    该函数用于装饰on_xxx系列方法，用于在分布式模式下，过滤掉不应属于当前rank的保存动作。

    该装饰器会自动判断是否属于当前rank，若不属于当前rank，则不会调用被装饰的函数。
    """

    def decorator(func):
        # check function signature
        if inspect.signature(func).parameters.keys() != {'self', 'prefix', 'module'}:
            raise ValueError(
                f"Function {func.__name__} has incorrect signature that cannot be decorated by save_this_rank_only"
            )

        @functools.wraps(func)
        def wrapper(self_instance: "DistributedAscendV1Saver", prefix: str, module: nn.Module) -> None:
            if not dist.is_initialized():
                func(self_instance, prefix, module)
                return

            if self_instance.dist_helper is None:
                if dist.get_rank() == 0:
                    func(self_instance, prefix, module)
                return

            is_local_only = self_instance.dist_helper.is_local_only(prefix)
            is_in_shared_modules_slice = prefix in self_instance.shared_modules_slice
            save_on_this_rank = is_local_only or is_in_shared_modules_slice

            if not save_on_this_rank:
                return

            func(self_instance, prefix, module)
            return

        return wrapper

    return decorator


def decorate_on_methods(cls):
    """
    类装饰器：自动为所有以 'on_' 开头的方法添加 save_this_rank_only 装饰器
    """

    @functools.wraps(cls, updated=())
    def wrapper_class():
        # 获取所有以 'on_' 开头的方法，包括继承的方法
        for attr_name in dir(cls):
            if attr_name.startswith('on_') and callable(getattr(cls, attr_name)):
                attr = getattr(cls, attr_name)
                # 检查是否是实例方法、类方法或静态方法
                if hasattr(attr, '__func__'):  # 实例方法
                    # 对于继承的方法，也需要装饰，但要避免重复装饰
                    # 检查方法是否已经被 save_this_rank_only 装饰过
                    if not hasattr(attr, '__wrapped__') or 'save_this_rank_only' not in str(attr):
                        setattr(cls, attr_name, save_this_rank_only()(attr))
                elif inspect.isfunction(attr):  # 类方法或静态方法
                    if not hasattr(attr, '__wrapped__') or 'save_this_rank_only' not in str(attr):
                        setattr(cls, attr_name, save_this_rank_only()(attr))
        return cls

    return wrapper_class()


class _DistIterCtx:
    """_iter_tasks 内部使用的轻量分布上下文。step 1 构造，step 5 销毁。"""

    def __init__(self, prefix: str, module: nn.Module):
        self.prefix = prefix
        self.module = module
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()

        self.queue = get_distributed_task_work_queue()

        # 构造 DistHelper，通过 all_gather_object 识别本 rank 的 local_only 模块
        self._dist_helper = DistHelper(module, prefix=prefix)
        self.local_only_names = self._dist_helper._local_only_modules

    def destroy(self):
        self._dist_helper = None


@QABCRegistry.register(dispatch_key=DistributedAscendV1Config, abc_class=AutoSessionProcessor)
@logger_setter(prefix='msmodelslim.saver.ascend_v1_distributed')  # 4-level: msmodelslim.core.quant_service.modelslim_v1
class DistributedAscendV1Saver(AscendV1Saver):
    """
    支持分布式保存的 ascendV1 量化模型保存器。

    该类继承自 AscendV1Saver，并添加了分布式保存支持。
    在分布式模式下，每个rank会保存自己的部分到独立的目录中，最后在rank 0上合并所有文件。
    """

    def __init__(self, model: nn.Module, config: AscendV1Config, adapter: object, **kwargs: Dict[str, Any]):
        super().__init__(model, config, adapter, **kwargs)

        self.save_directory = self.get_rank_save_directory() if dist.is_initialized() else config.save_directory

        self.json_writer = JsonWriter(self.save_directory, ASCENDV1_DESC_JSON_NAME)
        self.safetensors_writer = self.get_safetensors_writer(config)

        if dist.is_initialized():
            self.file_mappings = [{} for _ in range(dist.get_world_size())]
        else:
            self.file_mappings = [{}]

    def support_distributed(self) -> bool:
        return True

    def _iter_tasks(self, prefix: str, module: nn.Module, memo=None):
        """生成本 rank 应处理的 (全名, 子模块) 序列。"""
        # 1. 根据输入构造Helper
        if not dist.is_initialized() or dist.get_world_size() == 1:
            yield from module.named_modules(prefix=prefix, memo=memo)
            return

        ctx = _DistIterCtx(prefix, module)

        # 2. 获取全部未处理模块
        all_modules = list(module.named_modules(prefix=prefix, memo=memo))

        # 3. 过滤出未处理模块中的 local_only 模块并遍历（EP 场景下本 rank 独占的模块）
        for name, sub in all_modules:
            if name in ctx.local_only_names:
                yield name, sub

        # 4. 过滤出未处理模块中的 shared 模块并分发
        shared_items = [(n, s) for n, s in all_modules if n not in ctx.local_only_names]
        if ctx.queue is not None:
            # 动态分发（默认）：通过 queue 抢任务
            shared_map = dict(shared_items)
            if ctx.rank == 0:
                for name, _ in shared_items:
                    ctx.queue.put(name)
                for _ in range(ctx.world_size):
                    ctx.queue.put(None)
            while True:
                try:
                    name = ctx.queue.get(timeout=300)
                except Exception:
                    break
                if name is None:
                    break
                yield name, shared_map[name]
        else:
            # 静态轮询（无共享 queue 时的 fallback）
            for idx, (name, sub) in enumerate(shared_items):
                if idx % ctx.world_size == ctx.rank:
                    yield name, sub

        # 5. 销毁Helper
        ctx.destroy()

    def postprocess(self, request: BatchProcessRequest) -> None:
        prefix, module = request.name, request.module
        _convert_hookir_to_wrapper(module)
        for name, sub in self._iter_tasks(prefix, module, memo=self.processed_modules):
            self._process_module_maybe_wrapper_ir(name, sub)

    def get_rank_save_directory(self) -> str:
        return os.path.join(self.config.save_directory, f"rank_{dist.get_rank()}")

    def post_run(self) -> None:
        _convert_hookir_to_wrapper(self.model)
        for name, sub in self._iter_tasks('', self.model, memo=self.processed_modules):
            self._process_module_maybe_wrapper_ir(name, sub)

        for key, val in self.json_append.items():
            self.json_writer.write(key, val)

        if self.quarot_info is not None:
            self.metadata['quarot'] = self.quarot_info.get_quarot_save_info()

        self.json_writer.write("version", self.version)
        self.json_writer.write("model_quant_type", self.model_quant_type)
        self.json_writer.write("metadata", self.metadata)
        self.json_writer.write("group_size", self.group_size)
        self.json_writer.write(
            "optional",
            {scope: scope_info.model_dump(mode='json') for scope, scope_info in self.json_optional_infos.items()},
        )

        self.json_writer.close()
        self.safetensors_writer.close()

        if self.support_distributed() and dist.is_initialized():
            self.merge_ranks()

        if dist.get_rank() != 0:
            return

        copy_files(self.adapter.model_path, self.config.save_directory)
        remove_quantization_config(self.config.save_directory)

        if isinstance(self.adapter, AscendV1SaveInterface):
            self.adapter.ascendv1_save_postprocess(self.model, self.config.save_directory)

    def merge_ranks(self) -> None:
        """合并所有rank保存的文件"""

        # 统计所有rank的文件数量
        file_counts = [torch.zeros(1, dtype=torch.int64)] * dist.get_world_size()
        local_count = len([f for f in os.listdir(self.save_directory) if f.endswith('.safetensors')])
        dist.all_gather_object(file_counts, local_count)

        dist.barrier()

        if dist.get_rank() != 0:
            return

        # 合并所有rank的文件
        self._merge_safetensor_files(file_counts)
        self._merge_index_files()
        self._merge_json_files()
        self._cleanup_rank_dirs()

        get_logger().info("Merge ranks completed successfully. Final weights saved to: %s", self.config.save_directory)

    def _merge_safetensor_files(self, file_counts):
        """合并所有rank的safetensors文件"""
        for rank in range(dist.get_world_size()):
            rank_dir = os.path.join(self.config.save_directory, f"rank_{rank}")
            safetensors_files = sorted([f for f in os.listdir(rank_dir) if f.endswith('.safetensors')])

            # 计算当前rank的文件偏移
            offset = sum(file_counts[:rank])

            # 重命名并移动文件
            for i, file in enumerate(safetensors_files):
                src = os.path.join(rank_dir, file)
                # 保持原有的文件命名格式
                if self.config.part_file_size is None:
                    dst = os.path.join(self.config.save_directory, self.safetensors_writer.save_prefix)
                else:
                    file_name_prefix = self.safetensors_writer.save_prefix
                    dst = os.path.join(
                        self.config.save_directory,
                        f"{file_name_prefix}-{offset + i + 1:05d}-of-{sum(file_counts):05d}.safetensors",
                    )
                # 记录文件映射关系
                self.file_mappings[rank][file] = os.path.basename(dst)
                shutil.move(src, dst)

    def _merge_index_files(self):
        """合并所有rank的index文件"""
        index_files = []
        for rank in range(dist.get_world_size()):
            rank_dir = os.path.join(self.config.save_directory, f"rank_{rank}")
            index_file = os.path.join(rank_dir, f"{self.safetensors_writer.save_prefix}.safetensors.index.json")
            if os.path.exists(index_file):
                index_files.append((rank, index_file))

        if index_files:
            # 合并index文件
            merged_index = {"metadata": {"total_size": 0}, "weight_map": {}}
            for rank, index_file in index_files:
                with open(index_file, "r", encoding="utf-8") as f:
                    index_data = json.load(f)
                    merged_index["metadata"]["total_size"] += index_data["metadata"]["total_size"]
                    # 更新weight_map中的文件路径
                    for key, value in index_data["weight_map"].items():
                        # 从原始路径中提取文件名
                        original_file = os.path.basename(value)
                        # 使用对应rank的文件映射查找新文件名
                        new_file = self.file_mappings[rank].get(original_file)
                        if new_file is None:
                            logger.warning("File %s not found in rank %s", original_file, rank)
                            continue
                        merged_index["weight_map"][key] = new_file

            # 保存合并后的index文件
            final_index_path = os.path.join(
                self.config.save_directory, f"{self.safetensors_writer.save_prefix}.safetensors.index.json"
            )
            json_safe_dump(merged_index, final_index_path, indent=2)

    def _merge_json_files(self):
        """合并所有rank的json文件"""
        json_files = []
        for rank in range(dist.get_world_size()):
            rank_dir = os.path.join(self.config.save_directory, f"rank_{rank}")
            json_file = os.path.join(rank_dir, self.json_writer.file_name)
            if os.path.exists(json_file):
                json_files.append(json_file)

        if json_files:
            # 合并json文件
            merged_meta = {}
            for json_file in json_files:
                with open(json_file, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                    merged_meta.update(meta)

            # 对键进行排序
            sorted_meta = dict(sorted(merged_meta.items()))

            # 保存合并后的json文件
            final_json_path = os.path.join(self.config.save_directory, self.json_writer.file_name)
            json_safe_dump(sorted_meta, final_json_path, indent=2)

    def _cleanup_rank_dirs(self):
        """清理所有rank的保存目录"""
        for rank in range(dist.get_world_size()):
            rank_dir = os.path.join(self.config.save_directory, f"rank_{rank}")
            shutil.rmtree(rank_dir)
