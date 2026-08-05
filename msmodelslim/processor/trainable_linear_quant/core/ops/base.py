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

# 可训练 TLQ Op 基类、配置与工厂。

import copy
from abc import ABC, abstractmethod
from typing import ClassVar, Dict, List, Optional, Sequence, Type

import torch
from pydantic import Field
from torch import nn

from msmodelslim.processor.anti_outlier.common import SubgraphRegistry
from msmodelslim.processor.anti_outlier.common.subgraph_type import Subgraph
from msmodelslim.utils.exception import UnsupportedError
from msmodelslim.utils.plugin import TypedConfig, TypedFactory
from msmodelslim.utils.plugin.plugin_utils import list_registered_plugin_types, load_plugin_component_class

from msmodelslim.processor.trainable_linear_quant.core.wrapper import TrainableLinearQuantWrapper

TRAIN_OPERATION_ENTRY_GROUP = "msmodelslim.train_operation.plugins"


@TypedConfig.plugin_entry(entry_point_group=TRAIN_OPERATION_ENTRY_GROUP)
class TLQOpConfig(TypedConfig):
    """可训练 TLQ Op 插件配置。"""

    type: TypedConfig.TypeField
    lr: Optional[float] = Field(
        default=None,
        gt=0.0,
        description="该 Op 可训练参数学习率；未指定时使用 train_config.lr",
    )


class TLQOp(ABC):
    """可训练 TLQ Op 基类。

    安装方式由子类区分：

    - ``LinearTLQOp``：单 layer + wrapper
    - ``SubgraphTLQOp``：adapter 子图 + 多个 wrapper
    """

    op_id: str
    config: TLQOpConfig
    target_modules: Dict[str, TrainableLinearQuantWrapper]

    def __init__(self) -> None:
        self._best_params: Optional[Dict[str, torch.Tensor]] = None

    @abstractmethod
    def bind(self) -> None:
        """创建可训练参数并向 ``TLQKernel`` 注册 listener。"""

    @abstractmethod
    def unbind(self) -> None:
        """训练收尾：解除伪量化监听，并将 Op 对模型的持久影响写回模型。

        由 Processor 在 ``load_best_params`` 与各 ``wrapper.unwrapper()`` **之后**调用。

        - Per-linear Op 通常只解除 listener。
        - Subgraph Op 可能还需把可训练参数融合进子图结构（见 ``SubgraphTLQOp`` 子类）。
        """

    @property
    @abstractmethod
    def train_params(self) -> Dict[str, torch.Tensor]:
        """可训练参数（key → ``nn.Parameter``）；须在 ``bind`` 之后访问。"""

    @property
    def best_params(self) -> Optional[Dict[str, torch.Tensor]]:
        """训练期保存的最优可训练参数快照（``save_best_params`` 写入）。"""
        return self._best_params

    @torch.no_grad()
    def reset_params(self, params: Dict[str, torch.Tensor]) -> None:
        """将外部快照写回 ``train_params``。"""
        for key, param in self.train_params.items():
            value = params.get(key)
            if value is None:
                continue
            param.copy_(value.to(device=param.device, dtype=param.dtype))

    @torch.no_grad()
    def save_best_params(self) -> None:
        """将当前 ``train_params`` 深拷贝到本 Op 内部 ``_best_params``（由 Trainer 在最优 loss 步调用）。"""
        snap: Dict[str, torch.Tensor] = {}
        for key, tensor in self.train_params.items():
            if isinstance(tensor, nn.Parameter):
                snap[key] = copy.deepcopy(tensor.data)
            else:
                snap[key] = copy.deepcopy(tensor)
        self._best_params = snap

    @torch.no_grad()
    def load_best_params(self) -> None:
        """finalize 前将 ``_best_params`` 写回 ``train_params``。"""
        if self._best_params is None:
            raise UnsupportedError(
                f"TLQOp {self.op_id!r} has no saved best params; call save_best_params during training"
            )
        self.reset_params(self._best_params)

    def release_cached_params(self) -> None:
        """Release checkpoint snapshots held after block finalize."""
        self._best_params = None


class LinearTLQOp(TLQOp):
    """Per-linear TLQ Op：直接绑定单个 ``TrainableLinearQuantWrapper``。"""

    layer_path: str

    def __init__(
        self,
        config: TLQOpConfig,
        layer_path: str,
        wrapper: TrainableLinearQuantWrapper,
    ) -> None:
        super().__init__()
        self.config = config
        self.layer_path = layer_path
        self.target_modules = {layer_path: wrapper}
        self.op_id = f"{layer_path}.{config.type}"


class SubgraphTLQOp(TLQOp):
    """Subgraph TLQ Op：通过 adapter 子图拓扑绑定一个或多个 wrapper。"""

    SUPPORTED_SUBGRAPH_TYPES: ClassVar[Sequence[str]] = ()

    subgraph: Subgraph
    subgraph_type: str

    def __init__(
        self,
        config: TLQOpConfig,
        subgraph: Subgraph,
        target_modules: Dict[str, TrainableLinearQuantWrapper],
    ) -> None:
        super().__init__()
        self.config = config
        self.subgraph = subgraph
        self.target_modules = target_modules
        subgraph_type = SubgraphRegistry.get_name(type(subgraph))
        if subgraph_type == "unknown":
            raise UnsupportedError(f"unknown subgraph type: {type(subgraph).__name__}")
        self.subgraph_type = subgraph_type
        paths = "+".join(sorted(self.target_modules))
        self.op_id = f"{paths}.{config.type}"


_tlq_op_factory = TypedFactory[TLQOp](config_base_class=TLQOpConfig)


def is_subgraph_op_config(op_config: TLQOpConfig) -> bool:
    """Subgraph ops require adapter subgraph configs (e.g. trainable_smooth)."""
    return hasattr(op_config, "enable_subgraph_type")


def operations_need_adapter_subgraph(operation_configs: Sequence[TLQOpConfig]) -> bool:
    """Return whether any configured op needs ``get_adapter_config_for_subgraph``."""
    return any(is_subgraph_op_config(cfg) for cfg in operation_configs)


def load_tlq_op_class(config: TLQOpConfig) -> Type[TLQOp]:
    """按 ``config.type`` 加载已注册的 TLQ Op 类（不实例化）。"""
    return load_plugin_component_class(TRAIN_OPERATION_ENTRY_GROUP, config.type)


def create_linear_tlq_op(
    config: TLQOpConfig,
    layer_path: str,
    wrapper: TrainableLinearQuantWrapper,
) -> TLQOp:
    """Create a per-linear TLQ op bound to one wrapped layer."""
    op_cls = load_tlq_op_class(config)
    return op_cls(config, layer_path=layer_path, wrapper=wrapper)


def create_subgraph_tlq_op(
    config: TLQOpConfig,
    subgraph: Subgraph,
    target_modules: Dict[str, TrainableLinearQuantWrapper],
) -> SubgraphTLQOp:
    """Create a subgraph TLQ op bound to an adapter-resolved subgraph."""
    return _tlq_op_factory.create(
        config,
        subgraph=subgraph,
        target_modules=target_modules,
    )


def registered_tlq_op_types() -> List[str]:
    return list_registered_plugin_types(TRAIN_OPERATION_ENTRY_GROUP)


def format_tensor_dbg(
    tensor: Optional[torch.Tensor],
    include_abs_mean: bool = False,
    include_nnz: bool = False,
    include_grad: bool = False,
) -> str:
    """Format tensor stats for bind/finalize/DEBUG logs.

    Base fields: ``min/max/mean``. Optional ``abs_mean``/``nnz`` for sparse offsets;
    ``include_grad`` for trainer pre_step snapshots.
    """
    if tensor is None:
        return "none"
    t = tensor.detach().float().reshape(-1)
    parts = [
        f"min={float(t.min()):.4g}",
        f"max={float(t.max()):.4g}",
        f"mean={float(t.mean()):.4g}",
    ]
    if include_abs_mean:
        parts.append(f"abs_mean={float(t.abs().mean()):.4g}")
    if include_nnz:
        parts.append(f"nnz={int((t.abs() > 0).sum())}/{t.numel()}")
    if include_grad:
        grad = tensor.grad
        if grad is None:
            parts.append("grad=None")
        else:
            g = grad.detach().float().reshape(-1)
            parts.append(f"grad_abs_mean={float(g.abs().mean()):.4g}")
            parts.append(f"grad_nnz={int((g.abs() > 0).sum())}/{g.numel()}")
    return ",".join(parts)
