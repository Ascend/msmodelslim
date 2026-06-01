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
import os
from typing import Dict, Any, Optional, List, Literal

import torch
import torch.distributed as dist
from pydantic import Field
from torch import nn

import msmodelslim.ir as qir
from msmodelslim.ir.qal.qregistry import QABCRegistry
from msmodelslim.model import IModel
from msmodelslim.core.base.protocol import BatchProcessRequest
from msmodelslim.processor.base import AutoSessionProcessor
from msmodelslim.utils.distributed import DistHelper
from msmodelslim.utils.exception import UnsupportedError, SchemaValidateError
from msmodelslim.utils.logging import logger
from msmodelslim.utils.security import safe_copy_file
from msmodelslim.ir.qal import QDType, QScope
from .saver import AutoSaverProcessor, AutoSaverBaseConfig
from .utils.json import JsonWriter
from .utils.safetensors import SafetensorsWriter, BufferedSafetensorsWriter
from .utils.pack import pack_fp4_to_uint8


def copy_files(input_path, output_path):
    """
    复制模型配置文件
    @param input_path: 源目录
    @param output_path: 目标目录
    """
    for file in os.listdir(input_path):
        if not any((file.endswith(subfix) for subfix in ['.json', '.py'])):
            continue

        if any((file.endswith(subfix) for subfix in ['index.json'])):
            continue

        ori_file = os.path.join(input_path, file)
        dest_file = os.path.join(output_path, file)
        safe_copy_file(src_path=ori_file, dest_path=dest_file)
        os.chmod(dest_file, int("600", 8))


class ValidJsonExt:
    JSON_APPEND = "json_append"


class MindIEFormatConfig(AutoSaverBaseConfig):
    """
    MindIEFormat 量化模型保存器配置。该配置用于配置 MindIEFormat 量化模型保存器。

    该配置包含以下字段：
        - type: 量化模型保存器类型，固定为"mindie_format_saver"
        - save_directory: 量化模型保存目录，默认为"."
        - part_file_size: 量化模型权重文件大小，默认为4，单位为GB，若part_file_size为0，则不进行分文件保存
        - ext: 扩展配置，用于配置量化模型保存器的扩展功能

    Notes:

    在MindIEFormat格式中，标准导出件包括：
        - 量化模型描述文件，使用json格式描述量化模型
        - 量化模型权重文件，使用safetensors格式保存量化模型权重（可能包含多个）
        - safetensors index文件，使用json格式保存safetensors文件的索引（可选）

    量化模型描述文件中，包含以下字段：
        - 量化模型权重键值对，键为权重名称，值为权重所属的量化类型
            example:
            {
                "model.layers.0.self_attn.q_proj.weight": "W8A8",
                "model.layers.0.self_attn.q_proj.input_scale": "W8A8",
                "model.layers.0.self_attn.q_proj.input_offset": "W8A8",
                "model.layers.0.self_attn.q_proj.deq_scale": "W8A8",
                "model.layers.0.self_attn.q_proj.quant_bias": "W8A8",
                "model.layers.0.self_attn.q_proj.bias": "FLOAT",
            }

            该例子中表明了model.layers.0.self_attn.q_proj所代表的nn.Linear被量化为W8A8类型。

    量化模型权重文件以safetensors格式保存权重参数。考虑到文件大小的限制，可能会存在多个权重文件，
    此时则会生成safetensors index文件，用于记录各个权重所处的safetensors文件。

    """

    type: Literal['mindie_format_saver'] = "mindie_format_saver"
    save_directory: str = Field(default=".", exclude=True)
    part_file_size: int = 4
    ext: Dict[str, Any] = Field(default_factory=dict, exclude_if=lambda v: not v)

    def set_save_directory(self, save_directory: str):
        self.save_directory = str(save_directory)


DEFAULT_DESC_JSON_NAME = "quant_model_description.json"
DEFAULT_SAFETENSORS_NAME = "quant_model_weight.safetensors"
DEFAULT_GROUP_SIZE = 32


def save_this_rank_only():
    """

    该函数用于装饰on_xxx系列方法，用于在分布式模式下，过滤掉不应属于当前rank的保存动作。

    example:

    @save_this_rank_only
    def on_w8a8_static(self, prefix: str, module: qir.W8A8StaticFakeQuantLinear):
        pass

    该装饰器会自动入参判断是否属于当前rank，若不属于当前rank，则不会调用被装饰的函数。

    """

    def decorator(func):
        # check function signature
        if inspect.signature(func).parameters.keys() != {'self', 'prefix', 'module'}:
            raise SchemaValidateError(
                f"Function {func.__name__} has incorrect signature that cannot be decorated by save_this_rank_only"
            )

        @functools.wraps(func)
        def wrapper(self_instance: 'MindIEFormatSaver', prefix: str, module: nn.Module) -> None:
            if not dist.is_initialized():
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


@QABCRegistry.register(dispatch_key=MindIEFormatConfig, abc_class=AutoSessionProcessor)
class MindIEFormatSaver(AutoSaverProcessor):
    """
    mindie_format 量化模型保存器。该保存器将量化模型保存为 MindieFormat 格式。

    关于该格式的更多信息，请参考 MindIEFormatConfig 中的说明。
    """

    def __init__(self, model: nn.Module, config: MindIEFormatConfig, adapter: object, **kwargs: Dict[str, Any]):
        super().__init__(model, config, adapter, **kwargs)
        self.config = config
        self.adapter: IModel = adapter
        self.json_append = dict()
        self.save_directory = self.get_rank_save_directory() if dist.is_initialized() else config.save_directory
        self.json_writer = JsonWriter(config.save_directory, DEFAULT_DESC_JSON_NAME)
        self.safetensors_writer = self.get_safetensors_writer(config)
        self.dist_helper: Optional[DistHelper] = None
        self.shared_modules_slice: Optional[List[str]] = None
        self.group_size = None

    def support_distributed(self) -> bool:
        return True

    def post_run(self) -> None:
        super().post_run()

        if ValidJsonExt.JSON_APPEND in self.json_append:
            json_append = self.json_append.get(ValidJsonExt.JSON_APPEND)
            for key, val in json_append.items():
                self.json_writer.write(key, val)

            # 更改为 mindie_format 格式
            model_quant_type = self.json_append[ValidJsonExt.JSON_APPEND]['model_quant_type']
            self.json_writer.file_name = f"quant_model_description_{model_quant_type.lower()}.json"
            self.safetensors_writer.file_path = os.path.join(
                self.save_directory, f"quant_model_weight_{model_quant_type.lower()}.safetensors"
            )

        self.json_writer.close()
        self.safetensors_writer.close()

        copy_files(self.adapter.model_path, self.config.save_directory)

    def preprocess(self, request: BatchProcessRequest) -> None:
        if dist.is_initialized():
            self.prepare_for_distributed(request)

    def postprocess(self, request: BatchProcessRequest) -> None:
        super().postprocess(request)
        self.cleanup_for_distributed()

    def prepare_for_distributed(self, request: BatchProcessRequest) -> None:
        self.dist_helper = DistHelper(request.module)
        self.shared_modules_slice = self.dist_helper.get_shared_modules_slice(prefix=request.name)

    def cleanup_for_distributed(self) -> None:
        self.dist_helper = None

    def get_safetensors_writer(self, config: MindIEFormatConfig) -> SafetensorsWriter:
        if config.part_file_size > 0:
            return BufferedSafetensorsWriter(
                logger=logger,
                max_gb_size=config.part_file_size,
                save_directory=self.save_directory,
                save_prefix=DEFAULT_SAFETENSORS_NAME.removesuffix('.safetensors'),
            )
        elif config.part_file_size == 0:
            return SafetensorsWriter(
                logger=logger,
                file_path=os.path.join(self.save_directory, DEFAULT_SAFETENSORS_NAME),
            )
        else:
            raise SchemaValidateError(
                "The save parameter part_file_size must be greater than or equal to 0.Please check."
            )

    def get_rank_save_directory(self) -> str:
        return os.path.join(self.config.save_directory, f"rank_{dist.get_rank()}")

    def write_tensor(self, prefix: str, desc: str, tensor: torch.Tensor):
        self.json_writer.write(prefix, desc)
        self.safetensors_writer.write(prefix, tensor)

    def merge_ranks(self) -> None:
        if dist.get_rank() != 0:
            return
        raise UnsupportedError("merge_ranks for mindie_format is not implemented now")

    def _raise_ascendv1_saver_recommended(self, method_name: str) -> None:
        raise UnsupportedError(
            f"{method_name} is not supported by MindIEFormatSaver.",
            action="Please use AscendV1Saver instead.",
        )

    def on_w8a16_static_per_channel(self, prefix: str, module: qir.W8A16StaticPerChannelFakeQuantLinear):
        self._raise_ascendv1_saver_recommended("on_w8a16_static_per_channel")

    def on_w8a16_static_per_group(self, prefix: str, module: qir.W8A16StaticPerGroupFakeQuantLinear):
        self._raise_ascendv1_saver_recommended("on_w8a16_static_per_group")

    def on_w8a8_pd_mix(self, prefix: str, module: qir.W8A8PDMixFakeQuantLinear):
        self._raise_ascendv1_saver_recommended("on_w8a8_pd_mix")

    def on_w8a8_dynamic_per_group(self, prefix: str, module: qir.W8A8DynamicPerGroupFakeQuantLinear):
        self._raise_ascendv1_saver_recommended("on_w8a8_dynamic_per_group")

    def on_wfp8afp8_dynamic_per_channel(self, prefix: str, module: qir.WFP8AFP8DynamicPerChannelFakeQuantLinear):
        self._raise_ascendv1_saver_recommended("on_wfp8afp8_dynamic_per_channel")

    def on_w4a4_dynamic_per_channel(self, prefix: str, module: qir.W4A4DynamicPerChannelFakeQuantLinear):
        self._raise_ascendv1_saver_recommended("on_w4a4_dynamic_per_channel")

    def on_w4a4_dynamic_per_group(self, prefix: str, module: qir.W4A4DynamicPerGroupFakeQuantLinear):
        self._raise_ascendv1_saver_recommended("on_w4a4_dynamic_per_group")

    def on_w4a4_mx_dynamic_per_block(self, prefix: str, module: qir.W4A4MXDynamicPerBlockFakeQuantLinear):
        self._raise_ascendv1_saver_recommended("on_w4a4_mx_dynamic_per_block")

    def on_w4a8_mx_dynamic_per_block(self, prefix: str, module: qir.W4A8MXDynamicPerBlockFakeQuantLinear):
        self._raise_ascendv1_saver_recommended("on_w4a8_mx_dynamic_per_block")

    def on_w4a8_dynamic(self, prefix: str, module: qir.W4A8DynamicFakeQuantLinear):
        self._raise_ascendv1_saver_recommended("on_w4a8_dynamic")

    def on_dynamic_cache(self, prefix: str, module: qir.FakeQuantDynamicCache):
        self._raise_ascendv1_saver_recommended("on_dynamic_cache")

    def on_activation_per_head(self, prefix: str, module: qir.FakeQuantActivationPerHead):
        self._raise_ascendv1_saver_recommended("on_activation_per_head")

    def on_quarot_extra_info_wrapper(self, prefix: str, module: qir.QuaRotExtraInfoWrapperIR):
        self._raise_ascendv1_saver_recommended("on_quarot_extra_info_wrapper")

    def on_non_fusion_smooth_quant_wrapper(self, prefix: str, module: qir.NonFusionSmoothQuantWrapper):
        self._raise_ascendv1_saver_recommended("on_non_fusion_smooth_quant_wrapper")

    def on_w16a16s(self, prefix: str, module: qir.W16A16sLinear):
        self._raise_ascendv1_saver_recommended("on_w16a16s")

    @save_this_rank_only()
    def on_w8a8_static(self, prefix: str, module: qir.W8A8StaticFakeQuantLinear):
        with torch.device(module.weight.device):
            input_scale, input_offset = module.input_scale, module.input_offset
            input_scale = input_scale.unsqueeze(0) if input_scale.ndim == 0 else input_scale
            input_offset = input_offset.unsqueeze(0) if input_offset.ndim == 0 else input_offset
            weight_scale = module.weight_scale
            quant_weight = module.weight
            deq_scale = input_scale * weight_scale
            deq_scale = deq_scale.squeeze(1) if deq_scale.ndim > 1 else deq_scale
            fp_weight_bias = module.bias if module.bias is not None else torch.zeros(module.weight.shape[0])
            correction = quant_weight.to(torch.float32).sum(dim=1) * input_offset.to(torch.float32)
            quant_bias = torch.round(fp_weight_bias / deq_scale - correction).to(torch.int32)
            self.write_tensor(prefix + ".weight", "W8A8", quant_weight.to(torch.int8))
            self.write_tensor(prefix + ".quant_bias", "W8A8", quant_bias.to(torch.int32))
            self.write_tensor(prefix + ".input_scale", "W8A8", input_scale.to(torch.float32))
            self.write_tensor(prefix + ".input_offset", "W8A8", input_offset.to(torch.float32))
            self.write_tensor(prefix + ".deq_scale", "W8A8", deq_scale.to(torch.float32))
            if module.bias is not None:
                self.write_tensor(prefix + ".bias", "FLOAT", module.bias.to(torch.float32))

        if ValidJsonExt.JSON_APPEND not in self.json_append:
            self.json_append[ValidJsonExt.JSON_APPEND] = dict()
        self.json_append[ValidJsonExt.JSON_APPEND]['model_quant_type'] = "W8A8"

    @save_this_rank_only()
    def on_w8a8_dynamic_per_channel(self, prefix: str, module: qir.W8A8DynamicPerChannelFakeQuantLinear):
        with torch.device(module.weight.device):
            weight_scale = module.weight_scale.unsqueeze(-1)
            self.write_tensor(prefix + ".weight", "W8A8_DYNAMIC", module.weight.to(torch.int8))
            self.write_tensor(prefix + ".weight_scale", "W8A8_DYNAMIC", weight_scale.to(torch.float32))
            self.write_tensor(
                prefix + ".weight_offset", "W8A8_DYNAMIC", torch.zeros_like(weight_scale).to(torch.float32)
            )
            if module.bias is not None:
                self.write_tensor(prefix + ".bias", "FLOAT", module.bias.to(torch.float32))

        if ValidJsonExt.JSON_APPEND not in self.json_append:
            self.json_append[ValidJsonExt.JSON_APPEND] = dict()
        self.json_append[ValidJsonExt.JSON_APPEND]['model_quant_type'] = "W8A8_DYNAMIC"

    @save_this_rank_only()
    def on_w8a8_mx_dynamic_per_block(self, prefix: str, module: qir.W8A8MXDynamicPerBlockFakeQuantLinear):
        with torch.device(module.weight.device):
            if not (isinstance(module.w_axes, (int, list))):
                raise SchemaValidateError("w_axes must be int or list[int].")
            weight_scale = module.weight_scale
            self.group_size = DEFAULT_GROUP_SIZE
            self.write_tensor(prefix + ".weight", "W8A8_MXFP8", module.weight.cpu().to(torch.float8_e4m3fn))
            self.write_tensor(
                prefix + ".weight_scale",
                "W8A8_MXFP8",
                (weight_scale.squeeze(dim=module.w_axes) + 127).to(torch.uint8),
                # +127 是对 weight_scale 进行偏移处理，使其从-127~128偏移到0~255，正好覆盖torch_uint8的取值范围
            )
            if module.bias is not None:
                self.write_tensor(prefix + ".bias", "FLOAT", module.bias.to(torch.float32))

        if ValidJsonExt.JSON_APPEND not in self.json_append:
            self.json_append[ValidJsonExt.JSON_APPEND] = dict()
        self.json_append[ValidJsonExt.JSON_APPEND]['model_quant_type'] = "W8A8_MXFP8"

    def on_w4a4_mx_dynamic_dual_scale(self, prefix: str, module: qir.W4A4MXDynamicDualScaleFakeQuantLinear):
        with torch.device(module.weight.device):
            if not (isinstance(module.w_axes, (int, list))):
                raise SchemaValidateError("w_axes must be int or list[int].")
            weight_scale = module.weight_scale
            weight_dual_scale = module.weight_dual_scale
            self.group_size = 32
            self.write_tensor(prefix + ".weight", "W4A4_MXFP4_DUALSCALE", pack_fp4_to_uint8(module.weight.cpu()))
            self.write_tensor(
                prefix + ".weight_scale",
                "W4A4_MXFP4_DUALSCALE",
                (weight_scale.squeeze(dim=module.w_axes) + 127).cpu().to(torch.uint8),
                # +127 是对 weight_scale 进行偏移处理，使其从-127~128偏移到0~255，正好覆盖torch_uint8的取值范围
            )
            self.write_tensor(
                prefix + ".weight_dual_scale", "W4A4_MXFP4_DUALSCALE", weight_dual_scale.cpu().to(torch.float32)
            )
            if module.bias is not None:
                self.write_tensor(prefix + ".bias", "FLOAT", module.bias.to(torch.float32))

        if ValidJsonExt.JSON_APPEND not in self.json_append:
            self.json_append[ValidJsonExt.JSON_APPEND] = dict()
        self.json_append[ValidJsonExt.JSON_APPEND]['model_quant_type'] = "W4A4_MXFP4_DUALSCALE"

    def on_online_rotation_wrapper(self, prefix: str, module: qir.OnlineRotationWrapper):
        """
        处理OnlineRotationWrapper类型的模块。
        """
        rotation_matrix = module.rotation_info.rotation_matrix
        # 保存旋转矩阵，标签为 FLOAT
        self.write_tensor(f"{prefix}", "FLOAT", rotation_matrix.clone())

    @save_this_rank_only()
    def on_float_linear(self, prefix: str, module: nn.Linear):
        return self.on_float_module(prefix, module)

    @save_this_rank_only()
    def on_float_module(self, prefix: str, module: nn.Module):
        for name, param in module.named_parameters(recurse=False, prefix=prefix):
            self.write_tensor(name, "FLOAT", param)

    def on_activation_per_token(self, prefix: str, module: qir.FakeQuantActivationPerToken):
        # 对于FP8 per-token动态量化，保存quant_type标签
        if module.x_q_scheme.dtype == QDType.FP8_E4M3 and module.x_q_scheme.scope == QScope.PER_TOKEN:
            # 保存格式为 self_attn.quant_type，而不是 fa3_q.quant_type
            # 提取父路径，例如 model.layers.0.self_attn.fa3_q -> model.layers.0.self_attn
            parent_prefix = prefix.rsplit('.', 1)[0]
            quant_type_key = parent_prefix + ".quant_type"

            self.json_writer.write(quant_type_key, "FP8_DYNAMIC")
        else:
            raise SchemaValidateError(f"FakeQuantActivationPerToken Unsupported dtype: {module.x_q_scheme.dtype}")
