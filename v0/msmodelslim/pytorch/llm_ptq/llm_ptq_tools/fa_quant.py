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


import logging
from collections import defaultdict
from enum import Enum
from typing import Tuple, List, Optional

import torch
from transformers.configuration_utils import PretrainedConfig

from ascend_utils.common.security import check_type
from msmodelslim import logger as msmodelslim_logger
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.quant_funcs import linear_quantization_params

_SUPPORT_RECALL_WINDOW = False
try:
    from msmodelslim.pytorch.lowbit.atomic_power_outlier import recall_window
except ImportError:
    msmodelslim_logger.warning(
        "The current CANN version does not support recall_window method."
    )
else:
    _SUPPORT_RECALL_WINDOW = True


class TensorType(Enum):
    Q = "q"
    K = "k"
    V = "v"

    @classmethod
    def get_values(cls):
        return [t.value for t in cls]


class QKVQuantizer:
    def __init__(self):
        self.bit = None
        self.sym = None
        self.num_head = None
        self._min_values = None
        self._max_values = None
        self.scale = None
        self.offset = None
        self.is_record = False
        self.single_record = None
        self.ratio = None

    def configure(self, bit: int, sym: bool, num_head: int, ratio: float):
        self.bit = bit
        self.sym = sym
        self.num_head = num_head
        self.ratio = ratio

    def update(self, samples: torch.Tensor, tp_size: int):
        if self.is_record:
            self.single_record = samples
            self.is_record = False

        batch_size, num_head, seq_len, head_dim = samples.shape
        if tp_size == 0:
            raise ZeroDivisionError("tp size can not be zero")
        if not _SUPPORT_RECALL_WINDOW:
            num_head_per_device = self.num_head // tp_size
            samples = samples.contiguous().view(tp_size * num_head_per_device, -1)
            samples_max = samples.max(dim=-1, keepdim=True)[0]
            samples_min = samples.min(dim=-1, keepdim=True)[0]
        else:
            samples = samples.contiguous().view(tp_size * num_head, -1)
            samples_min, samples_max = recall_window(samples, self.ratio, -1, True)

        ## min value
        if self._min_values is None:
            self._min_values = samples_min
        else:
            index = samples_min < self._min_values
            self._min_values[index] = samples_min[index]
        ## max value
        if self._max_values is None:
            self._max_values = samples_max
        else:
            index = samples_max > self._max_values
            self._max_values[index] = samples_max[index]

    def get_scale_offset(
            self,
            states_tensor: torch.Tensor = None,
            tp_size: int = None,
            update: bool = False
    ):
        if update:
            self.update(states_tensor, tp_size)
            self.scale, self.offset = self._calculate_scale_offset()
        return self.scale, self.offset

    def is_calibrated(self):
        return self.scale is not None and self.offset is not None

    def reset_quant_params(self):
        self.scale, self.offset = None, None
    
    def enable_record(self):
        self.is_record = True
    
    def clear_record(self):
        self.single_record = None

    def _calculate_scale_offset(self, integral_zero_point: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        dtype = self._max_values.dtype
        scale, offset = linear_quantization_params(
            self.bit,
            self._min_values,
            self._max_values,
            sym=self.sym,
            integral_zero_point=integral_zero_point
        )
        if dtype == torch.bfloat16:
            scale = scale.to(dtype)
            offset = offset.to(dtype)
        return scale, offset


class FAQuantizer:
    def __init__(self, config, logger):
        def check_config(config):
            required_attrs = ["num_attention_heads", "hidden_size"]
            for attr in required_attrs:
                if not hasattr(config, attr):
                    raise AttributeError(f"FAQuantizer's config needs attributes: '{attr}'")
            
            check_type(config.num_attention_heads, int, "num_attention_heads in config")
            check_type(config.hidden_size, int, "hidden_size in config")
            
            # 如果不存在num_key_value_heads，则设置为num_attention_heads的值
            if not hasattr(config, "num_key_value_heads"):
                config.num_key_value_heads = config.num_attention_heads
                logger.warning("Failed to obtain `num_key_value_heads`, assuming Multi-head Attention by default.")
            else:
                check_type(config.num_key_value_heads, int, "num_key_value_heads in config")

        check_type(logger, logging.Logger, param_name="logger")
        check_config(config)
        
        self.tp_size = None
        self.num_head = None
        self.num_kv_head = None # GQA模型的KV所用到的注意力头数量，如果是MHA模型则与num_head相同
        self.head_dim = None
        self.is_calib = False
        self.dequant_infer = False
        self.logger = logger
        self.debug_mode = False

        self.q_observer = QKVQuantizer()
        self.k_observer = QKVQuantizer()
        self.v_observer = QKVQuantizer()

        self._set_head_params(config)

        self.processed_types = set()

    def quant(self, states_tensor: torch.Tensor, qkv: TensorType):
        self.processed_types.add(qkv)
        check_type(states_tensor, torch.Tensor, param_name="states_tensor")

        if not self.is_calib and not self.dequant_infer:
            return states_tensor

        if self.tp_size == 0:
            raise ValueError("tp size can not be zero")
        if TensorType(qkv) == TensorType.Q:
            scale, offset = self.q_observer.get_scale_offset(
                states_tensor=states_tensor,
                tp_size=self.tp_size,
                update=not self.dequant_infer
            )
            num_head_per_device = self.q_observer.num_head // self.tp_size
        elif TensorType(qkv) == TensorType.K:
            scale, offset = self.k_observer.get_scale_offset(
                states_tensor=states_tensor,
                tp_size=self.tp_size,
                update=not self.dequant_infer
            )
            num_head_per_device = self.k_observer.num_head // self.tp_size
        elif TensorType(qkv) == TensorType.V:
            scale, offset = self.v_observer.get_scale_offset(
                states_tensor=states_tensor,
                tp_size=self.tp_size,
                update=not self.dequant_infer
            )
            num_head_per_device = self.v_observer.num_head // self.tp_size
        else:
            values_string = ", ".join(TensorType.get_values())
            raise ValueError(f"Unsupported current TensorType. "
                             f"Please confirm if the parameter is in `{values_string}`")

        expected_types = {"q", "k", "v"}
        if self.processed_types != expected_types:
            missing = expected_types - self.processed_types
            raise RuntimeError(f"Missing qkv types:{missing}. "
                               f"Please ensure all {expected_types} are processed.")

        # only observe the quant params, not quantize and dequantize the tensor
        return states_tensor

    def configure(self, bit: int, sym: bool, tp_size: int = 1):
        self.tp_size = tp_size

        self.q_observer.configure(bit, sym, self.num_head, 0.9999)
        self.k_observer.configure(bit, sym, self.num_kv_head, 0.9999)
        self.v_observer.configure(bit, sym, self.num_kv_head, 1.0)
        
    def set_head_params(self, num_head: int, head_dim: int, num_kv_head: int):
        self.num_head = num_head
        self.head_dim = head_dim
        self.num_kv_head = num_kv_head
    
    def is_calibrated(self):
        return self.q_observer.is_calibrated() and self.k_observer.is_calibrated() and self.v_observer.is_calibrated()

    def disable_calibration(self):
        self.is_calib = False
        if self.is_calibrated():
            self.dequant_infer = True

    def enable_calibration(self):
        self.is_calib = True
        self.q_observer.reset_quant_params()
        self.k_observer.reset_quant_params()
        self.v_observer.reset_quant_params()
    
    def record_once(self):
        self.q_observer.enable_record()
        self.k_observer.enable_record()
        self.v_observer.enable_record()
    
    def reset(self):
        self.q_observer.reset_quant_params()
        self.k_observer.reset_quant_params()
        self.v_observer.reset_quant_params()

    def export_quant_params(self) -> Tuple[Tuple[torch.Tensor, ...], Tuple[torch.Tensor, ...]]:
        q_scale, q_offset = self.q_observer.get_scale_offset()
        k_scale, k_offset = self.k_observer.get_scale_offset()
        v_scale, v_offset = self.v_observer.get_scale_offset()
        return (
            q_scale.to(device="cpu"),
            k_scale.to(device="cpu"),
            v_scale.to(device="cpu")
        ), (
            q_offset.to(device="cpu", dtype=torch.int8),
            k_offset.to(device="cpu", dtype=torch.int8),
            v_offset.to(device="cpu", dtype=torch.int8)
        )

    def _set_head_params(self, config):  
        num_head = config.num_attention_heads
        hidden_size = config.hidden_size
        
        if num_head == 0:
            raise ValueError("`num_attention_heads` cannot be zero.")
        head_dim = hidden_size // num_head
        
        num_key_value_heads = config.num_key_value_heads
        
        # if GQA（num_key_value_heads != num_head）
        if num_key_value_heads != num_head:
            self.logger.info(
                "Using Grouped Query Attention (GQA): "
                "num_key_value_heads=%r, num_head=%r",
                num_key_value_heads,
                num_head
            )
        
        self.set_head_params(num_head, head_dim, num_key_value_heads)


def configure_fa(model: torch.nn.Module, bit: int = 8, sym: bool = True, tp_size: int = 1):
    for name, module in model.named_modules():
        if is_attn_module_and_then_check_quantizer(module, name):
            module.fa_quantizer.configure(bit, sym, tp_size)


def enable_fa_calibration(model: torch.nn.Module, skip_modules: List[str] = None):
    """
    启用FA校准
    
    Args:
        model: 需要校准的模型
        skip_modules: 需要跳过的模块名称列表
    """
    if skip_modules is None:
        skip_modules = []

    for name, module in model.named_modules():
        if name in skip_modules:
            continue

        if is_attn_module_and_then_check_quantizer(module, name):
            module.fa_quantizer.enable_calibration()


def disable_fa_calibration(model: torch.nn.Module):
    for name, module in model.named_modules():
        if is_attn_module_and_then_check_quantizer(module, name):
            module.fa_quantizer.disable_calibration()


def enable_fa_quantizer_record(model: torch.nn.Module):
    for name, module in model.named_modules():
        if is_attn_module_and_then_check_quantizer(module, name):
            module.fa_quantizer.record_once()


def collect_fa_quantizer_record(model: torch.nn.Module):
    records = {}
    for name, module in model.named_modules():
        if is_attn_module_and_then_check_quantizer(module, name):
            records[f"{name}.query_states"] = module.fa_quantizer.q_observer.single_record
            records[f"{name}.key_states"] = module.fa_quantizer.k_observer.single_record
            records[f"{name}.value_states"] = module.fa_quantizer.v_observer.single_record
            module.fa_quantizer.q_observer.clear_record()
            module.fa_quantizer.k_observer.clear_record()
            module.fa_quantizer.v_observer.clear_record()
    return records
    

def export_fa_quant_params(module: torch.nn.Module, name: str) -> Tuple[dict, ...]:
    quant_params_scale = defaultdict(torch.Tensor)
    quant_params_offset = defaultdict(torch.Tensor)
    attach_map = defaultdict(list)

    param_types = TensorType.get_values()

    if module.fa_quantizer.is_calibrated():
        scales, offsets = module.fa_quantizer.export_quant_params()

    for i, param_type in enumerate(param_types):
        scale_name = f"{name}.fa_{param_type}.scale"
        offset_name = f"{name}.fa_{param_type}.offset"
        
        if module.fa_quantizer.is_calibrated():
            quant_params_scale[scale_name] = scales[i]
            quant_params_offset[offset_name] = offsets[i]

        attach_map[name].extend([scale_name, offset_name])

    return quant_params_scale, quant_params_offset, attach_map


def is_attn_module_and_then_check_quantizer(module: torch.nn.Module, module_name: str) -> bool:
    # flux有Attention类,hunyuan的attn是在MMSingleStreamBlock, MMDoubleStreamBlock中
    if all(x not in module.__class__.__name__ for x in ["Attention", "MMSingleStreamBlock", "MMDoubleStreamBlock"]):
        return False
    if hasattr(module, "fa_quantizer") and isinstance(module.fa_quantizer, FAQuantizer):
        return True
    if hasattr(module, "fa_quantizer") and not isinstance(module.fa_quantizer, FAQuantizer):
        raise AttributeError(f"`FAQuantizer` is not detected in {module_name}. "
                                f"Please check the modeling file and insert FAQuantizer in the correct place.")
    return False


class ModelType(Enum):
    FLUX = "flux"
    HYVIDEO = "hyvideo"
    DEFAULT = "default"


class ModelTypeDetector:
    """模型类型检测器"""
    
    @staticmethod
    def detect_model_type(model: Optional[torch.nn.Module], logger: logging.Logger) -> 'ModelType':
        """
        检测模型类型，支持处理 None、非 Module 对象、空模型等异常情况
        """
        # 1. 检查 None 或无效输入
        if model is None:
            return ModelType.DEFAULT  # 或 raise ValueError("Model cannot be None")
        
        # 2. 验证类型
        if not isinstance(model, torch.nn.Module):
            raise ValueError(f"Expected torch.nn.Module, got {type(model).__name__}")

        try:
            # 3. 安全获取第一个子模块的类名
            modules = list(model.named_modules())
            if not modules:  # 空模型（仅有自身）
                first_module = model
            else:
                first_module = modules[0][1]  # (name, module) 中的 module
            
            module_class_name = first_module.__class__.__name__

            # 4. 类型判断（不区分大小写，提高容错）
            if "Flux" in module_class_name:
                return ModelType.FLUX
            elif "HYVideo" in module_class_name:
                return ModelType.HYVIDEO
            else:
                return ModelType.DEFAULT

        except Exception as e:  # 捕获可能的动态模型异常
            logger.warning(
                "Model type detection failed, fallback to DEFAULT. Error: %s",
                str(e)
                )
            return ModelType.DEFAULT


def install_fa_quantizer(
    model: torch.nn.Module,
    config: PretrainedConfig,
    logger: logging.Logger,
    skip_layers: Optional[List[str]] = None,
):
    """为模型安装FAQuantizer
    
    Args:
        model: 需要安装FAQuantizer的模型
        config: 模型配置
        logger: 日志记录器
        skip_layers: 需要跳过的层名列表
    """
    skip_layers = skip_layers or []
    model_type = ModelTypeDetector.detect_model_type(model, logger)

    if model_type == ModelType.FLUX:
        from .fa_quant_adapter.flux import install_for_flux_model
        install_for_flux_model(model, config, logger, skip_layers)
    elif model_type == ModelType.HYVIDEO:
        from .fa_quant_adapter.hyvideo import install_for_hyvideo_model
        install_for_hyvideo_model(model, config, logger, skip_layers)
    else:
        from .fa_quant_adapter.default_model import install_for_default_model
        install_for_default_model(model, config, logger, skip_layers)
