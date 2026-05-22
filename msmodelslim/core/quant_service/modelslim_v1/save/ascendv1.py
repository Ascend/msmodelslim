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

import os
from typing import Dict, Any, Optional, List, Literal
from unittest.mock import patch

import torch
import torch.distributed as dist
from pydantic import Field, BaseModel
from torch import nn

from msmodelslim import logger, ir as qir
from msmodelslim.ir.qal.qregistry import QABCRegistry
from msmodelslim.model import IModel
from msmodelslim.processor.base import AutoSessionProcessor
from msmodelslim.utils.distributed import DistHelper
from msmodelslim.utils.exception import ToDoError, SchemaValidateError
from msmodelslim.utils.logging import logger_setter
from msmodelslim.utils.security import safe_copy_file, get_write_directory
from msmodelslim.utils.security import json_safe_load, json_safe_dump
from .interface import AscendV1SaveInterface, AscendV1GlobalModelDtypeInterface
from .saver import AutoSaverProcessor, AutoSaverBaseConfig, _convert_hookir_to_wrapper
from .utils.deqscale import deqscale2int64_by_dtype
from .utils.json import JsonWriter
from .utils.pack import w4a8_pack_int4, process_scale, pack_fp4_to_uint8
from .utils.safetensors import SafetensorsWriter, BufferedSafetensorsWriter


def copy_files(input_path, output_path):
    """
    复制模型配置文件
    @param input_path: 源目录
    @param output_path: 目标目录
    """
    for file in os.listdir(input_path):
        if not any((file.endswith(subfix) for subfix in ['.json', '.py', '.txt', '.jinja'])):
            continue

        if any((file.endswith(subfix) for subfix in ['index.json'])):
            continue

        ori_file = os.path.join(input_path, file)
        dest_file = os.path.join(output_path, file)
        safe_copy_file(src_path=ori_file, dest_path=dest_file)
        os.chmod(dest_file, int("600", 8))


def remove_quantization_config(output_path):
    """
    从config.json文件中移除quantization_config字段（包括顶层和text_config中的）
    @param output_path: 目标目录
    """
    config_file = os.path.join(output_path, "config.json")

    if not os.path.exists(config_file):
        return

    try:
        config_data = json_safe_load(config_file, check_user_stat=True)

        removed = False
        if 'quantization_config' in config_data:
            del config_data['quantization_config']
            removed = True
        if 'text_config' in config_data and isinstance(config_data['text_config'], dict):
            if 'quantization_config' in config_data['text_config']:
                del config_data['text_config']['quantization_config']
                removed = True

        if removed:
            json_safe_dump(config_data, config_file, indent=2, check_user_stat=True)
    except Exception as e:
        logger.warning(f"Failed to remove quantization_config in config.json!")


class AscendV1Config(AutoSaverBaseConfig):
    """
    ascendV1 量化模型保存器配置。该配置用于配置ascendV1量化模型保存器。

    该配置包含以下字段：
        - type: 量化模型保存器类型，固定为"ascendv1_saver"
        - save_directory: 量化模型保存目录，默认为"."
        - part_file_size: 量化模型权重文件大小，默认为4，单位为GB，若part_file_size为0，则不进行分文件保存
        - ext: 扩展配置，用于配置量化模型保存器的扩展功能

    Notes:

    在ascendV1格式中，标准导出件包括：
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
            }

            该例子中表明了model.layers.0.self_attn.q_proj所代表的nn.Linear被量化为W8A8类型，
            该类型由5个参数完全描述。

    量化模型权重文件以safetensors格式保存权重参数。考虑到文件大小的限制，可能会存在多个权重文件，
    此时则会生成safetensors index文件，用于记录各个权重所处的safetensors文件。

    """

    def set_save_directory(self, save_directory: str):
        self.save_directory = str(save_directory)

    type: Literal['ascendv1_saver'] = "ascendv1_saver"
    save_directory: str = Field(default=".", exclude=True)
    part_file_size: int = 4
    ext: Dict[str, Any] = Field(default_factory=dict, exclude_if=lambda v: not v)


class QuaRotOptionalScopeInfo(BaseModel):
    rotation_map: Dict[str, str] = Field(default_factory=dict)

ASCENDV1_DESC_JSON_NAME = "quant_model_description.json"
ASCENDV1_SAFETENSORS_NAME = "quant_model_weights.safetensors"


@QABCRegistry.register(dispatch_key=AscendV1Config, abc_class=AutoSessionProcessor)
@logger_setter(prefix='msmodelslim.saver.ascend_v1')  # 4-level: msmodelslim.core.quant_service.modelslim_v1
class AscendV1Saver(AutoSaverProcessor):
    """
    ascendV1 量化模型保存器。该保存器将量化模型保存为AscendV1格式。

    关于该格式的更多信息，请参考 AscendV1Config 中的说明。
    """
    # W4A8_DYNAMIC is hidden（不加入列表，混合时不作为 model_quant_type）。
    # 比特位数越低优先级越高（列表中越靠后 index 越大，混合时越优先被记录）。
    # 同比特内：W8A8 优先于 W8A8_DYNAMIC、W8A8_MIX，与 V0 一致。
    QUANT_TYPE_PRIORITY = [
        'FLOAT', 'W16A16S', 'W8A16',                           # 高比特 → 低优先级
        'W8A8_DYNAMIC', 'W8A8_MIX', 'W8A8',                    # 8w8a，W8A8 优先于 W8A8_DYNAMIC/W8A8_MIX
        'WFP8AFP8_DYNAMIC', 'W8A8_MXFP8',                      # 8-bit 浮点量化
        'W4A8_MXFP',                                           # 4w 浮点量化
        'W4A4_DYNAMIC', 'W4A4_MXFP4', 'W4A4_MXFP4_DUALSCALE'                          # 4w4a → 高优先级
    ]

    def __init__(self, model: nn.Module, config: AscendV1Config, adapter: object, **kwargs: Dict[str, Any]):
        super().__init__(model, config, adapter, **kwargs)
        self.json_append = dict()
        self.metadata = dict()
        self.json_optional_infos: Dict[str, BaseModel] = dict()
        self.save_directory = self.get_rank_save_directory() if dist.is_initialized() else config.save_directory
        self.optional_save_directory = os.path.join(config.save_directory, "optional")
        self.json_writer = JsonWriter(self.save_directory, ASCENDV1_DESC_JSON_NAME)
        self.safetensors_writer = self.get_safetensors_writer(config)
        self.dist_helper: Optional[DistHelper] = None
        self.shared_modules_slice: Optional[List[str]] = None
        self.quarot_info: Optional[qir.QuarotOnlineRotationInfo] = None

        self.version = "1.0.0"
        self.model_quant_type = "Unknown"
        self.group_size = 0
        # If global torch dtype is bfloat16, convert deq_scale to int64.
        self._global_torch_dtype_is_bf16 = self._resolve_is_bf16_from_adapter(adapter)

    def support_distributed(self) -> bool:
        return True

    def post_run(self) -> None:

        _convert_hookir_to_wrapper(self.model)
        for name, sub_module in self.model.named_modules(memo=self.processed_modules):
            self._process_module_maybe_wrapper_ir(name, sub_module)

        for key, val in self.json_append.items():
            self.json_writer.write(key, val)

        if self.quarot_info is not None:
            self.metadata['quarot'] = self.quarot_info.get_quarot_save_info()

        self.json_writer.write("version", self.version)
        self.json_writer.write("model_quant_type", self.model_quant_type)
        self.json_writer.write("metadata", self.metadata)
        self.json_writer.write("group_size", self.group_size)
        self.json_writer.write("optional", {
            scope: scope_info.model_dump(mode='json')
            for scope, scope_info in self.json_optional_infos.items()
        })

        self.json_writer.close()
        self.safetensors_writer.close()

        if not isinstance(self.adapter, IModel):
            raise ToDoError(f'Model Adapter does NOT has attr model_path',
                            action=f'Please implement IModel for saving')
        copy_files(self.adapter.model_path, self.config.save_directory)
        remove_quantization_config(self.config.save_directory)

        if isinstance(self.adapter, AscendV1SaveInterface):
            self.adapter.ascendv1_save_postprocess(self.model, self.config.save_directory)

    def get_safetensors_writer(self, config: AscendV1Config) -> SafetensorsWriter:
        if config.part_file_size > 0:
            return BufferedSafetensorsWriter(
                logger=logger,
                max_gb_size=config.part_file_size,
                save_directory=self.save_directory,
                save_prefix=ASCENDV1_SAFETENSORS_NAME.removesuffix('.safetensors')
            )
        else:
            return SafetensorsWriter(
                logger=logger,
                file_path=os.path.join(self.save_directory, ASCENDV1_SAFETENSORS_NAME),
            )

    def get_rank_save_directory(self) -> str:
        return os.path.join(self.config.save_directory, f"rank_{dist.get_rank()}")

    def _resolve_is_bf16_from_adapter(self, adapter: object) -> bool:
        """从 adapter 的 AscendV1GlobalModelDtypeInterface 或 config 解析原始模型是否为 bfloat16，用于 deq_scale 是否转 int64。"""
        if isinstance(adapter, AscendV1GlobalModelDtypeInterface):
            try:
                return adapter.get_global_model_torch_dtype() == torch.bfloat16
            except Exception as e:
                logger.warning(
                    "Failed to resolve torch_dtype from adapter %s, falling back to config. Error: %s",
                    type(adapter).__name__,
                    e,
                )
        logger.warning("Can not resolve torch_dtype from model config, using False as default.")
        return False

    def write_tensor(self, prefix: str, desc: str, tensor: torch.Tensor):
        self.json_writer.write(prefix, desc)
        self.safetensors_writer.write(prefix, tensor)

    def on_w8a8_static(self, prefix: str, module: qir.W8A8StaticFakeQuantLinear):
        self.update_quant_type("W8A8")

        with torch.device(module.weight.device):
            input_scale, input_offset = module.input_scale, module.input_offset
            input_scale = input_scale.unsqueeze(0) if input_scale.ndim == 0 else input_scale
            input_offset = input_offset.unsqueeze(0) if input_offset.ndim == 0 else input_offset
            weight_scale = module.weight_scale
            quant_weight = module.weight
            deq_scale = input_scale * weight_scale
            deq_scale = deq_scale.squeeze(1) if deq_scale.ndim > 1 else deq_scale
            fp_weight_bias = module.bias if module.bias is not None else torch.zeros(module.weight.shape[0])
            fp_weight_bias = fp_weight_bias.unsqueeze(1) if deq_scale.ndim > 1 else fp_weight_bias
            correction = quant_weight.to(torch.float32).sum(dim=1) * input_offset.to(torch.float32)
            correction = correction.unsqueeze(1) if deq_scale.ndim > 1 else correction
            quant_bias = torch.round(fp_weight_bias / deq_scale - correction).to(torch.int32)
            deq_scale_to_write = deqscale2int64_by_dtype(
                deq_scale.to(torch.float32), self._global_torch_dtype_is_bf16
            )
            self.write_tensor(prefix + ".weight", "W8A8", quant_weight.to(torch.int8))
            self.write_tensor(prefix + ".quant_bias", "W8A8", quant_bias.to(torch.int32))
            self.write_tensor(prefix + ".input_scale", "W8A8", input_scale.to(torch.float32))
            self.write_tensor(prefix + ".input_offset", "W8A8", input_offset.to(torch.float32))
            self.write_tensor(prefix + ".deq_scale", "W8A8", deq_scale_to_write)
            if module.bias is not None:
                self.write_tensor(prefix + ".bias", "FLOAT", module.bias.to(torch.float32))

    def on_w8a16_static_per_channel(self, prefix: str, module: qir.W8A16StaticPerChannelFakeQuantLinear):
        self.update_quant_type("W8A16")

        with torch.device(module.weight.device):
            weight_scale = module.weight_scale.unsqueeze(-1) if module.weight_scale.ndim == 1 else module.weight_scale
            weight_offset = module.weight_offset.unsqueeze(-1) if module.weight_offset.ndim == 1 else module.weight_offset
            self.write_tensor(prefix + ".weight", "W8A16", module.weight.to(torch.int8))
            self.write_tensor(prefix + ".weight_scale", "W8A16", weight_scale.to(torch.float32))
            self.write_tensor(prefix + ".weight_offset", "W8A16", weight_offset.to(torch.float32))
            if module.bias is not None:
                self.write_tensor(prefix + ".bias", "FLOAT", module.bias.to(torch.float32))


    def on_w8a16_static_per_group(self, prefix: str, module: qir.W8A16StaticPerGroupFakeQuantLinear):
        self.update_quant_type("W8A16")
        self.group_size = module.group_size

        with torch.device(module.weight.device):
            weight_scale = module.weight_scale.unsqueeze(-1) if module.weight_scale.ndim == 1 else module.weight_scale
            weight_offset = module.weight_offset.unsqueeze(-1) if module.weight_offset.ndim == 1 else module.weight_offset
            self.write_tensor(prefix + ".weight", "W8A16", module.weight.to(torch.int8))
            self.write_tensor(prefix + ".weight_scale", "W8A16", weight_scale.to(torch.float32))
            self.write_tensor(prefix + ".weight_offset", "W8A16", weight_offset.to(torch.float32))
            if module.bias is not None:
                self.write_tensor(prefix + ".bias", "FLOAT", module.bias.to(torch.float32))

    def on_w8a8_dynamic_per_channel(self, prefix: str, module: qir.W8A8DynamicPerChannelFakeQuantLinear):
        self.update_quant_type("W8A8_DYNAMIC")

        with torch.device(module.weight.device):
            weight_scale = module.weight_scale.unsqueeze(-1)
            self.write_tensor(prefix + ".weight", "W8A8_DYNAMIC", module.weight.to(torch.int8))
            self.write_tensor(prefix + ".weight_scale", "W8A8_DYNAMIC", weight_scale.to(torch.float32))
            self.write_tensor(prefix + ".weight_offset", "W8A8_DYNAMIC",
                              torch.zeros_like(weight_scale).to(torch.float32))
            if module.bias is not None:
                self.write_tensor(prefix + ".bias", "FLOAT", module.bias.to(torch.float32))

    def on_w8a8_pd_mix(self, prefix: str, module: qir.W8A8PDMixFakeQuantLinear):
        self.update_quant_type("W8A8_MIX")

        with torch.device(module.weight.device):
            input_scale, input_offset = module.input_scale, module.input_offset
            input_scale = input_scale.unsqueeze(0) if input_scale.ndim == 0 else input_scale
            input_offset = input_offset.unsqueeze(0) if input_offset.ndim == 0 else input_offset
            weight_scale = module.weight_scale
            quant_weight = module.weight
            deq_scale = input_scale * weight_scale
            deq_scale = deq_scale.squeeze(1) if deq_scale.ndim > 1 else deq_scale
            fp_weight_bias = module.bias if module.bias is not None else torch.zeros(module.weight.shape[0])
            fp_weight_bias = fp_weight_bias.unsqueeze(1) if deq_scale.ndim > 1 else fp_weight_bias
            correction = quant_weight.to(torch.float32).sum(dim=1) * input_offset.to(torch.float32)
            correction = correction.unsqueeze(1) if deq_scale.ndim > 1 else correction
            quant_bias = torch.round(fp_weight_bias / deq_scale - correction).to(torch.int32)
            deq_scale_to_write = deqscale2int64_by_dtype(
                deq_scale.to(torch.float32), self._global_torch_dtype_is_bf16
            )
            self.write_tensor(prefix + ".weight", "W8A8_MIX", quant_weight.to(torch.int8))
            self.write_tensor(prefix + ".quant_bias", "W8A8_MIX", quant_bias.to(torch.int32))
            self.write_tensor(prefix + ".input_scale", "W8A8_MIX", input_scale.to(torch.float32))
            self.write_tensor(prefix + ".input_offset", "W8A8_MIX", input_offset.to(torch.float32))
            self.write_tensor(prefix + ".deq_scale", "W8A8_MIX", deq_scale_to_write)

            weight_scale = weight_scale.unsqueeze(-1)
            self.write_tensor(prefix + ".weight_scale", "W8A8_MIX", weight_scale.to(torch.float32))
            self.write_tensor(prefix + ".weight_offset", "W8A8_MIX",
                              torch.zeros_like(weight_scale).to(torch.float32))
            if module.bias is not None:
                self.write_tensor(prefix + ".bias", "FLOAT", module.bias.to(torch.float32))

    def on_w8a8_dynamic_per_group(self, prefix: str, module: qir.W8A8DynamicPerGroupFakeQuantLinear):
        self.update_quant_type("W8A8_DYNAMIC")
        self.group_size = module.group_size

        with torch.device(module.weight.device):
            weight_scale = module.weight_scale
            self.write_tensor(prefix + ".weight", "W8A8_DYNAMIC", module.weight.to(torch.int8))
            self.write_tensor(prefix + ".weight_scale", "W8A8_DYNAMIC", weight_scale.to(torch.float32))
            self.write_tensor(prefix + ".weight_offset", "W8A8_DYNAMIC",
                              torch.zeros_like(weight_scale).to(torch.float32))
            if module.bias is not None:
                self.write_tensor(prefix + ".bias", "FLOAT", module.bias.to(torch.float32))

    def on_wfp8afp8_dynamic_per_channel(self, prefix: str, module: qir.WFP8AFP8DynamicPerChannelFakeQuantLinear):
        self.update_quant_type("WFP8AFP8_DYNAMIC")
        with torch.device(module.weight.device):
            weight_scale = module.weight_scale.unsqueeze(-1)
            self.write_tensor(prefix + ".weight", "WFP8AFP8_DYNAMIC", module.weight.cpu().to(torch.float8_e4m3fn))
            self.write_tensor(prefix + ".weight_scale", "WFP8AFP8_DYNAMIC", weight_scale.to(torch.float32))
            self.write_tensor(prefix + ".weight_offset", "WFP8AFP8_DYNAMIC",
                              torch.zeros_like(weight_scale).to(torch.float32))
            if module.bias is not None:
                self.write_tensor(prefix + ".bias", "FLOAT", module.bias.to(torch.float32))

    def on_w8a8_mx_dynamic_per_block(self, prefix: str, module: qir.W8A8MXDynamicPerBlockFakeQuantLinear):
        self.update_quant_type("W8A8_MXFP8")

        with torch.device(module.weight.device):
            if not (isinstance(module.w_axes, (int, list))):
                raise SchemaValidateError("w_axes must be int or list[int].")
            weight_scale = module.weight_scale
            self.group_size = 32
            self.write_tensor(prefix + ".weight", "W8A8_MXFP8", module.weight.cpu().to(torch.float8_e4m3fn))
            self.write_tensor(
                prefix + ".weight_scale",
                "W8A8_MXFP8",
                (weight_scale.squeeze(dim=module.w_axes) + 127).to(torch.uint8)
                # +127 是对 weight_scale 进行偏移处理，使其从-127~128偏移到0~255，正好覆盖torch_uint8的取值范围
            )
            if module.bias is not None:
                self.write_tensor(prefix + ".bias", "FLOAT", module.bias.to(torch.float32))

    def on_w4a8_dynamic(self, prefix: str, module: qir.W4A8DynamicFakeQuantLinear):
        self.update_quant_type("W4A8_DYNAMIC")

        with torch.device(module.weight.device):
            weight_scale = module.weight_scale.unsqueeze(-1)
            weight = module.weight.to(torch.float32)
            deq_weight = weight.T * module.weight_scale
            scale_bias = process_scale(prefix, deq_weight.T, 16)
            self.write_tensor(prefix + ".weight", "W4A8_DYNAMIC", w4a8_pack_int4(module.weight.to(torch.int8)))
            self.write_tensor(prefix + ".weight_scale", "W4A8_DYNAMIC", weight_scale.to(torch.float32))
            self.write_tensor(prefix + ".weight_offset", "W4A8_DYNAMIC",
                              torch.zeros_like(weight_scale).to(torch.float32))
            self.write_tensor(prefix + '.scale_bias', "W4A8_DYNAMIC", scale_bias.to(torch.float32))
            if module.bias is not None:
                self.write_tensor(prefix + ".bias", "FLOAT", module.bias.to(torch.float32))

    def on_w4a4_dynamic_per_channel(self, prefix: str, module: qir.W4A4DynamicPerChannelFakeQuantLinear):
        self.update_quant_type("W4A4_DYNAMIC")

        with torch.device(module.weight.device):
            weight_scale = module.weight_scale.unsqueeze(-1)
            weight_offset = module.weight_offset.unsqueeze(-1)
            self.write_tensor(prefix + ".weight", "W4A4_DYNAMIC", module.weight.to(torch.int8))
            self.write_tensor(prefix + ".weight_scale", "W4A4_DYNAMIC", weight_scale.to(torch.float32))
            self.write_tensor(prefix + ".weight_offset", "W4A4_DYNAMIC", weight_offset.to(torch.float32))
            if module.bias is not None:
                self.write_tensor(prefix + ".bias", "FLOAT", module.bias.to(torch.float32))

    def on_w4a4_dynamic_per_group(self, prefix: str, module: qir.W4A4DynamicPerGroupFakeQuantLinear):
        self.update_quant_type("W4A4_DYNAMIC")
        self.group_size = module.group_size

        with torch.device(module.weight.device):
            self.write_tensor(prefix + ".weight", "W4A4_DYNAMIC", module.weight.to(torch.int8))
            self.write_tensor(prefix + ".weight_scale", "W4A4_DYNAMIC", module.weight_scale.to(torch.float32))
            self.write_tensor(prefix + ".weight_offset", "W4A4_DYNAMIC", module.weight_offset.to(torch.float32))
            if module.bias is not None:
                self.write_tensor(prefix + ".bias", "FLOAT", module.bias.to(torch.float32))

    def on_w4a4_mx_dynamic_per_block(self, prefix: str, module: qir.W4A4MXDynamicPerBlockFakeQuantLinear):
        self.update_quant_type("W4A4_MXFP4")

        with torch.device(module.weight.device):
            if not (isinstance(module.w_axes, (int, list))):
                raise SchemaValidateError("w_axes must be int or list[int].")
            weight_scale = module.weight_scale
            self.group_size = 32
            self.write_tensor(prefix + ".weight", "W4A4_MXFP4", pack_fp4_to_uint8(module.weight.cpu()))
            self.write_tensor(
                prefix + ".weight_scale",
                "W4A4_MXFP4",
                (weight_scale.squeeze(dim=module.w_axes) + 127).to(torch.uint8)
                # +127 是对 weight_scale 进行偏移处理，使其从-127~128偏移到0~255，正好覆盖torch_uint8的取值范围
            )
            if module.bias is not None:
                self.write_tensor(prefix + ".bias", "FLOAT", module.bias.to(torch.float32))

    def on_w4a4_mx_dynamic_dual_scale(self, prefix: str, module: qir.W4A4MXDynamicDualScaleFakeQuantLinear):
        self.update_quant_type("W4A4_MXFP4_DUALSCALE")

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
                (weight_scale.squeeze(dim=module.w_axes) + 127).cpu().to(torch.uint8)
                # +127 是对 weight_scale 进行偏移处理，使其从-127~128偏移到0~255，正好覆盖torch_uint8的取值范围
            )
            self.write_tensor(prefix + ".weight_dual_scale", "W4A4_MXFP4_DUALSCALE", weight_dual_scale.cpu().to(torch.float32))
            if module.bias is not None:
                self.write_tensor(prefix + ".bias", "W4A4_MXFP4_DUALSCALE", module.bias.to(torch.float32))

    def on_w4a8_mx_dynamic_per_block(self, prefix: str, module: qir.W4A8MXDynamicPerBlockFakeQuantLinear):
        self.update_quant_type("W4A8_MXFP")

        with torch.device(module.weight.device):
            if not (isinstance(module.w_axes, (int, list))):
                raise SchemaValidateError("w_axes must be int or list[int].")
            weight_scale = module.weight_scale
            self.group_size = 32
            self.write_tensor(prefix + ".weight", "W4A8_MXFP", pack_fp4_to_uint8(module.weight.cpu()))
            self.write_tensor(
                prefix + ".weight_scale",
                "W4A8_MXFP",
                (weight_scale.squeeze(dim=module.w_axes) + 127).to(torch.uint8)
                # +127 是对 weight_scale 进行偏移处理，使其从-127~128偏移到0~255，正好覆盖torch_uint8的取值范围
            )
            if module.bias is not None:
                self.write_tensor(prefix + ".bias", "FLOAT", module.bias.to(torch.float32))

    def on_float_linear(self, prefix: str, module: nn.Linear):
        self.update_quant_type("FLOAT")

        return self.on_float_module(prefix, module)

    def on_float_module(self, prefix: str, module: nn.Module):
        for name, param in module.named_parameters(recurse=False, prefix=prefix):
            self.write_tensor(name, "FLOAT", param)

    def on_dynamic_cache(self, prefix: str, module: qir.FakeQuantDynamicCache):
        prefix_list = prefix.split(".")
        prefix_no_last = '.'.join(prefix_list[:-1])
        if "key_states" in prefix_list[-1]:
            self.write_tensor(prefix_no_last + ".k_proj.kv_cache_scale", "C8", module.kv_cache_scale)
            self.write_tensor(prefix_no_last + ".k_proj.kv_cache_offset", "C8", module.kv_cache_offset)
        elif "value_states" in prefix_list[-1]:
            self.write_tensor(prefix_no_last + ".v_proj.kv_cache_scale", "C8", module.kv_cache_scale)
            self.write_tensor(prefix_no_last + ".v_proj.kv_cache_offset", "C8", module.kv_cache_offset)
        else:
            raise ValueError(f"Unknown dynamic cache prefix: {prefix}")
        self.json_append['kv_cache_type'] = "C8"
        self.json_append['kv_quant_type'] = "C8"

    def on_w16a16s(self, prefix: str, module: qir.W16A16sLinear):
        self.update_quant_type("W16A16S")

        for name, param in module.named_parameters(recurse=False, prefix=prefix):
            self.write_tensor(name, "W16A16S", param)

    def on_activation_per_head(self, prefix: str, module: qir.FakeQuantActivationPerHead):
        scale = module.input_scale.to(torch.float32).unsqueeze(-1)
        # 对于1维张量（fa_k.scale, fa_v.scale），转化为2维（与fa_q.scale对齐维数）
        if scale.dim() == 1:
            scale = scale.unsqueeze(-1)
        offset = torch.zeros_like(scale, dtype=torch.int8)
        self.write_tensor(prefix + ".scale", "FAQuant", scale)
        self.write_tensor(prefix + ".offset", "FAQuant", offset)
        self.json_append['fa_quant_type'] = "FAKQuant"

    def on_activation_per_token(self, prefix: str, module: qir.FakeQuantActivationPerToken):
        # 对于 per-token 动态量化，保存 quant_type 标签
        from msmodelslim.ir.qal import QDType, QScope
        # 保存格式为 self_attn.quant_type，而不是 fa3_q.quant_type
        # 提取父路径，例如 model.layers.0.self_attn.fa3_q -> model.layers.0.self_attn
        parent_prefix = prefix.rsplit('.', 1)[0]
        quant_type_key = parent_prefix + ".quant_type"
        
        if (module.x_q_scheme.dtype == QDType.FP8_E4M3 and 
            module.x_q_scheme.scope == QScope.PER_TOKEN):
            self.json_writer.write(quant_type_key, "FP8_DYNAMIC")
        elif (module.x_q_scheme.dtype == QDType.INT8 and 
            module.x_q_scheme.scope == QScope.PER_TOKEN):
            self.json_writer.write(quant_type_key, "INT8_DYNAMIC")
        else:
            raise SchemaValidateError(f"FakeQuantActivationPerToken Unsupported dtype: {module.x_q_scheme.dtype}")

    def on_online_rotation_wrapper(self, prefix: str, module: qir.OnlineRotationWrapper):
        """
        处理OnlineRotationWrapper类型的模块。
        """
        rotation_matrix = module.rotation_info.rotation_matrix
        # 保存旋转矩阵，标签为 FLOAT
        self.write_tensor(f"{prefix}", "FLOAT", rotation_matrix.clone())

    def on_rotation_wrapper(self, prefix: str, module: qir.QuarotOnlineHeadRotationWrapper):
        """
        处理RotationWrapper类型的模块。

        保存旋转矩阵到model.rotation，并在JSON中添加相应的描述。

        Args:
            prefix: 模块名称前缀
            module: RotationWrapper模块实例
        """
        self.quarot_info = module.rotation_info
        self.safetensors_writer.write(f"{prefix}.heads_rotation", self.quarot_info.heads_rotation.clone())

    def on_kronecker_rotation_wrapper(self, prefix: str, module: qir.QuarotOnlineKroneckerRotationWrapper):
        """
        处理KroneckerRotationWrapper类型的模块。

        保存旋转矩阵到model.rotation_m和model.rotation_n，并在JSON中添加相应的描述。

        Args:
            prefix: 模块名称前缀
            module: KroneckerRotationWrapper模块实例
        """
        self.quarot_info = module.rotation_info
        self.safetensors_writer.write(f"{prefix}.kronecker_rotation_m", self.quarot_info.kronecker_rotation_m.clone())
        self.safetensors_writer.write(f"{prefix}.kronecker_rotation_n", self.quarot_info.kronecker_rotation_n.clone())

    def on_quarot_extra_info_wrapper(self, prefix: str, module: qir.QuaRotExtraInfoWrapperIR):
        """
        导出 QuaRot 全局旋转矩阵到独立 safetensors 文件，并在 JSON 中写入 optional.quarot.global_rotation 路径。
        module.rotation_info 类型为 QuarotOfflineRotationInfo，其唯一成员 global_rotation 为 torch.Tensor。
        """
        self.optional_save_directory = get_write_directory(self.optional_save_directory)
        offline_info: qir.QuarotOfflineRotationInfo = module.rotation_info
        scope = "quarot"
        scope_tensor_file_name = f"{scope}.safetensors"
        scope_tensor_file_path = os.path.join(self.optional_save_directory, scope_tensor_file_name)
        relative_file_path = os.path.relpath(scope_tensor_file_path, self.save_directory)
        writer = SafetensorsWriter(logger=logger, file_path=scope_tensor_file_path)
        writer.write("global_rotation", offline_info.global_rotation)
        writer.close()
        scope_info = QuaRotOptionalScopeInfo(rotation_map={"global_rotation": relative_file_path})
        self.json_optional_infos.setdefault(scope, scope_info)

    def on_flat_clip_wrapper(self, prefix: str, module: qir.FlatQuantOnlineWrapper):
        """
        处理FlatQuantOnlineWrapper类型的模块。

        保存旋转矩阵到left_trans, right_trans, clip_factor, 并在JSON中添加相应的描述。

        Args:
            prefix: 模块名称前缀
            module: FlatQuantOnlineWrapper模块实例
        """
        wrapped_module = module.wrapped_module
        original_write_tensor = self.write_tensor
        self.desc_quant = ''

        def flat_write_tensor(prefix: str, desc: str, tensor: torch.Tensor):
            # FlatQuant INT 量化命名格式为：WXAX_FLATQUANT_DYNAMIC; MXFP 量化命名格式为：WXAX_MXFP_FLATQUANT
            if desc.endswith("_DYNAMIC"):
                desc = desc.replace("_DYNAMIC", "_FLATQUANT_DYNAMIC")
            else:
                desc += '_FLATQUANT'
            self.desc_quant = desc
            original_write_tensor(prefix, desc, tensor)

        with patch.object(self, 'write_tensor', wraps=flat_write_tensor):
            self._process_module(prefix, wrapped_module)
        self.update_quant_type(self.desc_quant)

        if module.save_trans is not None:
            save_trans = module.save_trans
            for key, trans in save_trans.items():
                self.write_tensor(f"{prefix}.{key}", self.desc_quant, trans)
            self.write_tensor(f"{prefix}.clip_ratio", self.desc_quant, module.clip_factor)

    def on_non_fusion_smooth_quant_wrapper(self, prefix: str, module: qir.NonFusionSmoothQuantWrapper):
        wrapped_module = module.wrapped_module
        self.write_tensor(prefix + ".div.mul_scale", "FLOAT", module.scales)
        prefix = prefix + ".linear"
        self._process_module(prefix, wrapped_module)

    def update_quant_type(self, quant_type: str):
        if quant_type not in self.QUANT_TYPE_PRIORITY:
            return
        if self.model_quant_type not in self.QUANT_TYPE_PRIORITY:
            self.model_quant_type = quant_type
            return
        if self.QUANT_TYPE_PRIORITY.index(quant_type) > self.QUANT_TYPE_PRIORITY.index(self.model_quant_type):
            self.model_quant_type = quant_type

    def _process_module(self, prefix: str, module: nn.Module):
        if isinstance(self.adapter, AscendV1SaveInterface):
            self.processed_modules.add(module)
            prefix, module = self.adapter.ascendv1_save_module_preprocess(prefix, module, self.model) or (prefix, module)
        super()._process_module(prefix=prefix, module=module)
