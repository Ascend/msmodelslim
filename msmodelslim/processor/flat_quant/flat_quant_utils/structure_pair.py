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
# pylint: disable=attribute-defined-outside-init,logging-fstring-interpolation

import torch
import functools
from torch import nn
from typing import Union, Callable, List, Any
from typing import Dict  # noqa: F401
from msmodelslim.utils.logging import get_logger
from msmodelslim.utils.exception import SchemaValidateError, UnexpectedError, UnsupportedError

from msmodelslim.processor.flat_quant.flat_quant_utils.utils import (
    get_module_by_name,
    set_module_by_name,
    get_decompose_dim,
    get_init_scale,
    stat_input_hook,
)

from msmodelslim.processor.flat_quant.flat_quant_utils.flat_fake_quant_linear import (
    ForwardMode,
    FlatNormWrapper,
    FlatFakeQuantLinear,
)

from msmodelslim.processor.flat_quant.flat_quant_utils.trans_matrix import GeneralMatrixTrans


class StructurePair(nn.Module):
    """
    管理一组结构对（如 Norm + Linear 或 Linear + Linear），
    支持统一的量化操作接口。
    """

    support_structure_pairs = []

    def __init_subclass__(cls, register=True, **kwargs):
        super().__init_subclass__(**kwargs)

        if kwargs.get('register') is False:
            return

        if getattr(cls, '_register', True) is False:
            return

        cls.support_structure_pairs.append(cls)

    def __init__(
        self,
        sources: str,
        targets: List[str],
        prefix_name: str,
        model: nn.Module,
    ) -> None:
        """
        初始化结构对管理器。

        Args:
            sources: 前置模块名称（如 Norm 层或前一个 Linear）。
            targets: 后续模块名称列表（如目标 Linear 层）。
            prefix_name: 模块前缀路径。
            model: 整体模型实例。
        """
        if not isinstance(sources, str):
            raise SchemaValidateError(f"sources must be a string, but got {type(sources).__name__}")
        if not isinstance(targets, list):
            raise SchemaValidateError(f"targets must be a list of strings, but got {type(targets).__name__}")

        super().__init__()
        self.model = model
        self.model_config = getattr(model, 'config', None)
        self.source_modules = sources
        self.target_modules = targets
        self.prefix_name = prefix_name
        self.name = f"{prefix_name}.{self._name}"
        self.linear_trans = None
        self.act_stats = {}

    def wrap_linear(self) -> None:
        """
        替换目标线性层为量化版本，并插入变换模块（trans）。
        """
        self._create_flatquant_linear()
        self._create_trans()

    def rollback_trans(self, pair_name: str) -> None:
        """
        移除已插入的变换（trans），恢复原始结构。
        """
        pass

    def change_mode(self, mod: ForwardMode) -> None:
        """
        统一切换模块运行模式（ORG/EVAL/CALIB），影响量化行为。

        行为说明：
            1. 清除钩子
            2. 对 Linear 的参数进行融合
            3. NormLinear 切换模式
            4. EVAL 模式下：融合在线 trans 的 diag_scale，trans 融合自身参数
            5. CALIB 模式下：初始化 trans 的 diag_scale
        """
        self._remove_forward_hook()
        self._call_method_on_modules(
            module_names=self.target_modules,
            method="change_mode",
            mod=mod,
        )

        if mod == ForwardMode.EVAL:
            self._reparameterize_act_diag_scale()
            self.linear_trans.to_eval_mode()
        elif mod == ForwardMode.CALIB:
            if self.config.diag_init == "sq_style":
                self._init_diag_scale(diag_alpha=self.config.diag_alpha)

    def _call_method_on_modules(
        self,
        module_names: Union[str, List[str]],
        method: Union[str, Callable],
        *args,
        **kwargs,
    ) -> None:
        """
        在指定模块上执行给定方法。支持单个或多个模块名。
        """
        if isinstance(module_names, str):
            module_names = [module_names]
        elif not isinstance(module_names, list):
            raise SchemaValidateError(
                f"module_names must be a str or a list of str, but got {type(module_names).__name__}"
            )

        for module_name in module_names:
            module = get_module_by_name(self.model, module_name, prefix=self.prefix_name)
            if not hasattr(module, method):
                raise UnsupportedError(f"Module '{module_name}' does not have a method named '{method}'")

            if isinstance(method, str):
                target_method = getattr(module, method)
            else:
                target_method = method
            target_method(*args, **kwargs)

    def _create_trans(self) -> None:
        """
        根据前置模块形状创建变换矩阵（trans），用于量化插件。
        """
        pre_linear_module = self._get_pre_linear_module()
        pre_dim_left, pre_dim_right = get_decompose_dim(pre_linear_module.weight.shape[0])
        self.linear_trans = GeneralMatrixTrans(
            pre_dim_left,
            pre_dim_right,
            add_diag=self.config.add_diag,
            diag_relu=self.config.diag_relu,
            tran_type=self.config.tran_type,
        )

    def _create_flatquant_linear(self) -> None:
        """
        将目标线性层替换为支持量化和变换的 FlatFakeQuantLinear。
        """
        for linear_name in self.target_modules:
            linear = get_module_by_name(self.model, linear_name, prefix=self.prefix_name)
            flat_linear = FlatFakeQuantLinear(self.config, linear)
            if self.config.diag_init == "sq_style":
                self._register_forward_hook(flat_linear, linear_name)
            set_module_by_name(self.model, linear_name, flat_linear, prefix=self.prefix_name)

    def _register_forward_hook(self, linear, name):
        """
        注册前向钩子，用于统计输入激活的最大值。
        """
        self.hooks = {}
        self.act_stats[name] = {}
        self.act_stats[name]['input_max'] = torch.full([linear.weight.shape[1]], 1e-5, dtype=linear.weight.dtype)
        self.hooks[name] = linear.register_forward_hook(
            functools.partial(stat_input_hook, name=name, act_stats=self.act_stats)
        )

    def _remove_forward_hook(self):
        """
        移除前向钩子，避免内存泄漏。
        """
        if not hasattr(self, 'hooks'):
            return
        for name, hook in self.hooks.items():
            hook.remove()

    def _reparameterize_act_diag_scale(self) -> None:
        """
        在 EVAL 模式下，将变换的对角缩放因子应用到原始权重。
        """
        if hasattr(self.linear_trans, 'diag_trans') and self.linear_trans.diag_trans is not None:
            pre_linear_name = self.source_modules
            pre_linear_module = get_module_by_name(self.model, pre_linear_name, prefix=self.prefix_name)
            weight = pre_linear_module.weight.data
            ori_dtype = weight.dtype
            if weight.dim() == 2:
                weight = weight.to(torch.float32) * self.linear_trans.diag_trans.diag_scale.data.to(
                    torch.float32
                ).unsqueeze(1)
            elif weight.dim() == 1:
                weight = weight.to(torch.float32) * self.linear_trans.diag_trans.diag_scale.data.to(torch.float32)
            else:
                raise UnexpectedError(f"Unsupported weight dimension: {weight.dim()}")
            pre_linear_module.weight.data = weight.to(ori_dtype)

    def _init_diag_scale(self, diag_alpha: float = 0.5) -> None:
        """
        在 CALIB 模式下，基于输入和权重最大值初始化对角缩放因子。
        """
        if self.linear_trans.diag_trans is None:
            return
        pre_linear_name = self.source_modules  # noqa: F841
        post_linear_names = self.target_modules

        weights = []
        for linear_name in post_linear_names:
            linear_module = get_module_by_name(self.model, linear_name, prefix=self.prefix_name)
            weights.append(linear_module.weight)
            input_max = self.act_stats[linear_name].get('input_max', None)

        if input_max is None:
            input_max = torch.full([linear_module.weight.shape[1]], 1e-5, dtype=linear_module.weight.dtype)

        weights_max = torch.cat(weights, dim=0).abs().max(dim=0)[0]
        weights_max = weights_max.to(self.linear_trans.diag_trans.diag_scale)
        input_max = input_max.to(self.linear_trans.diag_trans.diag_scale)
        self.linear_trans.diag_trans.diag_scale.data = get_init_scale(weights_max, input_max, diag_alpha)

    def contain(self, name: str) -> bool:
        """
        判断给定模块名是否属于该结构对的目标模块之一。
        """
        return any(name == target for target in self.target_modules)

    def _get_pre_linear_module(self) -> nn.Module:
        """
        获取前置模块（如 Norm 层或前一个 Linear）的实例。
        """
        pre_linear_name = self.source_modules
        return get_module_by_name(self.model, pre_linear_name, prefix=self.prefix_name)

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return self.name


class NormLinearPair(StructurePair):
    """
    Norm 层与后续线性层之间的结构对，支持嵌入变换模块。
    wrap_linear / change_mode / rollback_trans 都有变化。
    """

    _register = False

    def __init__(
        self,
        config: Any,
        norm_name: str,
        linear_name: str,
        prefix_name: str,
        model: nn.Module,
    ) -> None:
        super().__init__(norm_name, linear_name, prefix_name, model)
        self.config = config

    def wrap_linear(self) -> None:
        """
        替换线性层为量化版本，并将变换模块嵌入 Norm 层包装器中。
        """
        super().wrap_linear()
        self.norm_module = self._get_pre_linear_module()
        flat_norm = FlatNormWrapper(self.norm_module, self.linear_trans)
        set_module_by_name(self.model, self.source_modules, flat_norm, prefix=self.prefix_name)

        self._call_method_on_modules(
            module_names=self.target_modules,
            method="set_trans",
            weight_in_trans=self.linear_trans,
            save_trans=self.linear_trans,
        )
        self.norm_module = self._get_pre_linear_module()

    def change_mode(self, mod: ForwardMode) -> None:
        """
        统一切换模式，包括 Norm 层和线性层的行为。
        """
        self._remove_forward_hook()
        self._call_method_on_modules(
            module_names=self.target_modules,
            method="change_mode",
            mod=mod,
        )
        self.norm_module.change_mode(mod)

        if mod == ForwardMode.EVAL:
            self._reparameterize_act_diag_scale()
            self.linear_trans.to_eval_mode()
        elif mod == ForwardMode.CALIB:
            if self.config.diag_init == "sq_style":
                self._init_diag_scale(diag_alpha=self.config.diag_alpha)

    def rollback_trans(self, pair_name: str) -> None:
        """
        移除变换模块，恢复原始 Norm 层结构。
        """
        if not self.contain(pair_name):
            return
        ori_norm = self.norm_module.norm
        set_module_by_name(self.model, self.source_modules, ori_norm, prefix=self.prefix_name)

        get_logger().info(
            f"Rolling back affine transformation matrices for all modules in {self._name}: {self.target_modules}"
        )
        self._call_method_on_modules(
            module_names=self.target_modules,
            method="set_trans",
            weight_in_trans=None,
            save_trans=None,
        )


class AttnNormLinearPair(NormLinearPair):
    """
    注意力模块中的 Norm 层与 QKV 线性层结构对，用于统一量化管理。
    """

    _register = True
    _name = "attn.norm_linear"

    def __init__(
        self,
        config: Any,
        attn_norm_name: str,
        linear_name: List[str],
        prefix_name: str,
        model: nn.Module,
    ) -> None:
        super().__init__(config, attn_norm_name, linear_name, prefix_name, model)
        self.config = config


class AttnLinearLinearPair(StructurePair):
    """
    注意力模块中前一个 Linear 与后一个 Linear 之间的结构对，
    支持变换插入。无 diag，相关操作去除。
    """

    _name = "attn.linear_linear"

    def __init__(
        self,
        config: Any,
        pre_linear_name: str,
        post_linear_name: str,
        prefix_name: str,
        model: nn.Module,
        head_dim: int = None,
        num_attention_heads: int = None,
    ) -> None:
        super().__init__(pre_linear_name, post_linear_name, prefix_name, model)
        self.config = config
        self.head_dim = head_dim
        self.num_attention_heads = num_attention_heads

    def _create_trans(self) -> None:
        """
        根据注意力头维度创建变换矩阵，支持分头处理。
        """
        if self.head_dim is None or self.num_attention_heads is None:
            raise SchemaValidateError(
                "head_dim and num_attention_heads must not be empty. Please check that they are correctly assigned in the model adapter."
            )

        self.linear_trans = GeneralMatrixTrans(
            self.num_attention_heads,
            self.head_dim,
            add_diag=False,
            diag_relu=self.config.diag_relu,
            tran_type=self.config.tran_type,
        )

    def _reparameterize_act_diag_scale(self) -> None:
        """
        忽略对角缩放重参数化（因无对角变换）。
        """
        pass

    def _init_diag_scale(self, diag_alpha: float = 0.5) -> None:
        """
        忽略对角缩放初始化（因无对角变换）。
        """
        pass

    def wrap_linear(self) -> None:
        """
        替换目标线性层并插入变换模块，同时配置前向传播的 trans 路径。
        """
        super().wrap_linear()

        pre_linear_module = self._get_pre_linear_module()
        pre_linear_module.set_trans(
            weight_in_trans=pre_linear_module.weight_in_trans,
            act_in_trans=pre_linear_module.act_in_trans,
            save_trans=pre_linear_module.save_trans,
            weight_out_trans=self.linear_trans.right_trans,
        )

        self._call_method_on_modules(
            module_names=self.target_modules,
            method="set_trans",
            weight_in_trans=self.linear_trans,
            act_in_trans=self.linear_trans.left_trans,
            save_trans=self.linear_trans.left_trans,
        )

    def rollback_trans(self, pair_name: str) -> None:
        """
        回退变换模块的插入，恢复原始线性层结构。
        """
        if not self.contain(pair_name):
            return
        pre_linear_module = self._get_pre_linear_module()
        pre_linear_module.set_trans(
            weight_in_trans=pre_linear_module.weight_in_trans,
            act_in_trans=pre_linear_module.act_in_trans,
            save_trans=pre_linear_module.save_trans,
            weight_out_trans=None,
        )

        self._call_method_on_modules(
            module_names=self.target_modules,
            method="set_trans",
            weight_in_trans=None,
            act_in_trans=None,
            save_trans=None,
        )


class MLPNormLinearPair(NormLinearPair):
    """
    MLP 模块中 Norm 层与 Gate-Up 线性层的结构对，用于量化处理。
    """

    _register = True
    _name = "mlp.norm_linear"

    def __init__(
        self,
        config: Any,
        mlp_norm_name: str,
        linear_name: str,
        prefix_name: str,
        model: nn.Module,
    ) -> None:
        super().__init__(config, mlp_norm_name, linear_name, prefix_name, model)
        self.config = config


class MLPLinearLinearPair(StructurePair):
    """
    MLP 模块中前一个 Linear 与后一个 Linear 之间的结构对，用于量化变换。
    """

    _name = "mlp.linear_linear"

    def __init__(
        self,
        config: Any,
        pre_linear_name: str,
        post_linear_name: str,
        prefix_name: str,
        model: nn.Module,
    ) -> None:
        super().__init__(pre_linear_name, post_linear_name, prefix_name, model)
        self.config = config

    def wrap_linear(self) -> None:
        """
        替换目标线性层并插入变换模块，配置权重与激活的 trans 路径。
        """
        super().wrap_linear()

        self._call_method_on_modules(
            module_names=self.target_modules,
            method="set_trans",
            weight_in_trans=self.linear_trans,
            act_in_trans=self.linear_trans,
            save_trans=self.linear_trans,
        )

    def rollback_trans(self, pair_name: str) -> None:
        """
        移除变换模块的插入，还原为原始线性层结构。
        """
        if not self.contain(pair_name):
            return

        self._call_method_on_modules(
            module_names=self.target_modules,
            method="set_trans",
            weight_in_trans=None,
            act_in_trans=None,
            save_trans=None,
        )
