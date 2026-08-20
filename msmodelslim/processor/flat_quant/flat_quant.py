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

import torch
import functools
from torch import nn
from contextlib import nullcontext, AbstractContextManager
from pydantic import BaseModel, Field, computed_field, model_validator, AfterValidator
from typing import Annotated, List, Optional, Literal, Callable, Any, Union
import msmodelslim.ir as qir
from msmodelslim.ir.qal.qregistry import QABCRegistry
from msmodelslim.core.base.protocol import BatchProcessRequest
from msmodelslim.core.quantizer.linear import LinearQConfig
from msmodelslim.processor.base import AutoSessionProcessor, AutoProcessorConfig
from msmodelslim.utils.validation.pydantic import validate_str_length
from msmodelslim.processor.flat_quant.trainer import LayerTrainer
from msmodelslim.processor.flat_quant.flat_quant_interface import FlatQuantInterface
from msmodelslim.utils.config_map import ConfigSet
from msmodelslim.processor.flat_quant.flat_quant_utils import utils
from msmodelslim.processor.flat_quant.flat_quant_utils.flat_quant_manager import FlatQuantLayerManager
from msmodelslim.processor.flat_quant.flat_quant_utils.flat_fake_quant_linear import (
    ForwardMode,
    FlatFakeQuantLinear,
    FlatNormWrapper,
)
from msmodelslim.utils.exception import SchemaValidateError, UnsupportedError

import importlib

npu_available = importlib.util.find_spec("torch_npu") is not None


class QuantStrategyConfig(BaseModel):
    """量化策略配置：定义量化参数、包含/排除模块规则"""

    qconfig: LinearQConfig = Field(description="量化配置参数")
    include: List[Annotated[str, AfterValidator(validate_str_length())]] = Field(
        default_factory=lambda: ["*"], description="要包含的模块名称（支持通配符 *）"
    )
    exclude: List[Annotated[str, AfterValidator(validate_str_length())]] = Field(
        default_factory=list, description="要排除的模块名称（优先于 include）"
    )


class FlatQuantProcessorConfig(AutoProcessorConfig):
    """FlatQuant处理器配置：定义量化训练参数、策略、混合精度等"""

    type: Literal["flatquant"] = Field(default="flatquant", description="处理器类型标识，固定为 'flatquant'")
    include: List[Annotated[str, AfterValidator(validate_str_length())]] = Field(
        default_factory=lambda: ["*"], init=True, description="包含的模块名称"
    )
    exclude: List[Annotated[str, AfterValidator(validate_str_length())]] = Field(
        default_factory=lambda: [], init=True, description="排除的模块名称"
    )
    strategies: List[QuantStrategyConfig] = Field(default=[], init=False, description="量化策略配置列表")

    seed: int = Field(default=0, init=False, description="随机种子，用于复现结果")
    diag_relu: bool = Field(default=True, init=False, description="是否启用 diag_relu 激活函数实现变换矩阵")
    amp_dtype: str = Field(default="bfloat16", init=False, description="混合精度类型，用于加速训练")

    a_bits: int = Field(default=4, init=False, description="校准训练时激活量化的位宽（如 4bit）")
    a_groupsize: int = Field(default=-1, init=False, description="校准训练时激活量化的组大小（-1 表示按张量分组）")
    a_asym: bool = Field(default=False, init=False, description="校准训练时激活量化是否为非对称量化")
    a_per_tensor: bool = Field(default=False, init=False, description="校准训练时激活量化是否按张量进行（而非按通道）")

    w_bits: int = Field(default=4, init=False, description="校准训练时权重量化的位宽（如 4bit）")
    w_groupsize: int = Field(default=-1, init=False, description="校准训练时权重量化的组大小（-1 表示按张量分组）")
    w_asym: bool = Field(default=False, init=False, description="校准训练时权重量化是否为非对称量化")

    epochs: int = Field(default=10, init=False, description="校准训练的总轮数")
    nsamples: Optional[int] = Field(default=None, init=False, description="用于校准的样本数量")
    cali_bsz: int = Field(default=4, init=False, description="校准阶段的批次大小")
    flat_lr: float = Field(default=1e-3, init=False, description="FlatQuant 量化训练的学习率")
    add_diag: bool = Field(default=True, init=False, description="是否启用对角缩放矩阵，用于全局缩放")
    lwc: bool = Field(default=True, init=False, description="是否启用权重校准（训练权重量化参数）")
    lac: bool = Field(default=True, init=False, description="是否启用激活校准（训练激活量化参数）")
    diag_init: str = Field(
        default="one_style", init=False, description="对角缩放矩阵的初始化方式,支持sq_style以及one_style"
    )
    diag_alpha: float = Field(default=0.3, init=False, description="对角线缩放参数，控制缩放强度")
    warmup: bool = Field(default=True, init=False, description="是否启用训练预热机制，提升稳定性")
    deactive_amp: bool = Field(default=True, init=False, description="是否禁用混合精度训练（用于调试）")
    tran_type: str = Field(default="svd", init=False, description="变换矩阵实现方式：svd 表示基于 SVD 分解")

    @model_validator(mode='after')
    def validate_init_fields(self):
        """
        校验逻辑：带 `init=False` 的内部字段不允许在 YAML 中显式赋值；如果赋值则抛出错误。
        """
        errors = []
        for field_name, field_info in self.model_fields.items():
            field_value = getattr(self, field_name)
            default_value = field_info.default

            if not field_info.init:
                if field_value != default_value:
                    errors.append(f"{field_name}")

        if errors:
            raise SchemaValidateError(
                "Configuration validation failed: YAML configuration contains unsupported parameters: ".join(errors)
            )
        return self

    @computed_field(return_type=torch.dtype)
    @property
    def dtype(self) -> torch.dtype:
        """获取当前配置的精度类型（float32 / bfloat16 / float16）"""
        if self.deactive_amp:
            return torch.float32
        else:
            if self.amp_dtype == "bfloat16":
                return torch.bfloat16
            elif self.amp_dtype == "float16":
                return torch.float16
            else:
                raise UnsupportedError(
                    f"Unsupported mixed-precision dtype: {self.amp_dtype}. Only 'float16', 'bfloat16' are supported."
                )

    @computed_field
    @property
    def traincast(self) -> Union[AbstractContextManager, Callable[..., Any]]:
        """获取用于训练的上下文管理器（自动混合精度或无）"""
        if self.deactive_amp:
            return nullcontext
        else:
            device_type = "npu" if torch.npu.is_available() else "cuda"
            return functools.partial(torch.amp.autocast, device_type=device_type, dtype=self.dtype)

    model_config = {"arbitrary_types_allowed": True}


@QABCRegistry.register(dispatch_key=FlatQuantProcessorConfig, abc_class=AutoSessionProcessor)
class FlatQuantProcessor(AutoSessionProcessor):
    """FlatQuant处理器：实现逐层量化校准训练，支持动态变换矩阵与IR结构封装"""

    def __init__(
        self, model: torch.nn.Module, config: FlatQuantProcessorConfig, adapter: FlatQuantInterface, **kwargs
    ) -> None:
        """初始化FlatQuant处理器，加载模型结构、量化策略和适配器，准备训练环境"""
        super().__init__(model)
        self.model = model
        self.config = config
        self.strategies = config.strategies
        self.trans_include = ConfigSet(config.include)
        self.trans_exclude = ConfigSet(config.exclude)
        self.adapter = adapter
        self.structure_config = adapter.get_flatquant_subgraph()
        self.float_output = None
        self.layer_trainer = LayerTrainer(self.config)
        self.layer_quantizer = None
        self.device = ''
        self.dtype_dict = {}

    def pre_run(self) -> None:
        """模型预处理阶段：设置为评估模式，冻结参数，执行全局量化插入并清空缓存"""
        self.model.eval()

    def preprocess(self, request: BatchProcessRequest) -> None:
        """为当前层准备训练：记录原始输出、拷贝层、应用FlatQuant结构并切换至原始模式"""
        for param in request.module.parameters():
            param.requires_grad = False

        self.layer_quantizer = FlatQuantLayerManager(request.module, self.config)
        self.layer_quantizer.register_layer_pairs(self.structure_config, request.name)
        self.device = next(request.module.parameters()).device.type

        self.layer_quantizer.wrap_linear(self.device)
        self.layer_quantizer.change_mode(ForwardMode.ORG)
        self._run_forward_if_need(request)
        self.float_output = [output[0] for output in request.outputs]
        self.layer_quantizer.change_mode(ForwardMode.CALIB)

        self.dtype_dict = {name: param.dtype for name, param in request.module.named_parameters()}

    def process(self, request: BatchProcessRequest) -> None:
        """对当前层执行量化训练：切换到校准模式，使用训练器优化量化参数"""
        self.layer_trainer.train_layer(request=request, float_output=self.float_output)

    def postprocess(self, request: BatchProcessRequest) -> None:
        """恢复层数据类型，回滚仿射变换，切换至评估模式，并将层转换为IR结构"""
        for name, param in request.module.named_parameters():
            param.requires_grad = False
            if name in self.dtype_dict:
                param.data = param.to(self.dtype_dict[name])

        self._rollback_trans(request.name, request.module)
        self.layer_quantizer.change_mode(ForwardMode.EVAL)
        self.set_hook_ir(request.module)

    def post_run(self) -> None:
        """模型最终处理：绑定权重以确保共享参数一致性，并清空缓存"""
        self.model.tie_weights()

    def _rollback_trans(self, prefix: str, module: nn.Module) -> None:
        for name, submodule in module.named_modules(prefix=prefix):
            if not isinstance(submodule, FlatFakeQuantLinear):
                continue
            should_rollback = False
            if name not in self.trans_include:
                should_rollback = True

            if name in self.trans_exclude:
                should_rollback = True

            if should_rollback:
                self.layer_quantizer.rollback_trans(pair_name=name)

    @torch.no_grad()
    def set_hook_ir(self, block: torch.nn.Module) -> None:
        """递归遍历模型模块，将FlatFakeQuantLinear替换为FlatQuantOnlineWrapper包装的IR层"""
        for name, child in list(block.named_children()):
            if isinstance(child, FlatFakeQuantLinear):
                clip_ratio = child.act_quantizer.get_clip_ratio()
                save_trans = None

                if hasattr(child, "save_trans") and child.save_trans is not None:
                    save_trans = child.save_trans.get_save_params()

                ori_linear: torch.nn.Linear = child.unwrapper()
                child.del_linear()

                if save_trans is None:
                    block._modules[name] = ori_linear
                else:
                    hook_ir = qir.FlatQuantOnlineHookIR(clip_ratio, save_trans)
                    hook_handle = ori_linear.register_forward_pre_hook(hook_ir)
                    hook_ir.set_hook_handle(hook_handle)
                    block._modules[name] = ori_linear
            elif isinstance(child, FlatNormWrapper):
                norm: torch.nn.Module = child.unwrapper()
                child.del_norm()
                block._modules[name] = norm
            else:
                self.set_hook_ir(child)

            utils.empty_cache()
            import gc

            gc.collect()

    def need_kv_cache(self):
        """判断当前处理器是否需要键值缓存"""
        return False
