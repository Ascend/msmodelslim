#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
离线权重转换（msmodelslim convert）领域协议定义。

本文件只做一件事：定义“阶段接口”和“阶段之间共享的数据契约”，不包含具体实现。
你可以把它看成 convert 子系统的“接口说明书 + 数据流规范”。

设计目标：
1. 让 app 层编排器（ConvertApplication）只依赖抽象接口，而不依赖具体实现类。
2. 让不同实现（reader / preprocess / executor / save）可替换、可测试。
3. 把“输入是什么、输出是什么、阶段间怎么交接”写清楚，减少隐式约定。

重要约束（由 router 注册阶段强制校验）：
- convert 专用 Processor 必须是离线数据无关型，
  即 ``requires_forward=False`` 且 ``requires_calibration=False``。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Iterator, Literal, Protocol, runtime_checkable

from torch import nn

from msmodelslim.core.convert.catalog import PreprocessResult, TensorCatalog
from msmodelslim.core.convert.config import ConvertConfig, WeightMappingRule
from msmodelslim.core.convert.tasks import IRResult, IRTask, RoutedTask
from msmodelslim.core.convert.types import IRKind


class ConvertContext:
    """
    单次 convert 运行的可变上下文（全流程共享的“运行时状态”）。

    这个对象贯穿整个流水线：
    CLI/Fatory 创建后，Preprocess / TreeBuilder / TaskBuilder / Executor / Save
    都会在它上面读写各自阶段的中间结果。

    字段说明：
    - config: ``ConvertConfig``
      本次任务的静态配置（路径、规则、并行参数、目标格式等）。
    - model_path / save_path: ``Path``
      源 checkpoint 目录与输出目录（从 config 预解析为 Path）。
    - reader: ``ICheckpointReader | None``
      读盘器，负责 index/header/tensor 读取。
    - virtual_tree: ``nn.Module | None``
      虚拟模块树（不做 forward）；executor 会把转换后的模块写回此树。
    - preprocess_result: ``PreprocessResult | None``
      预处理结果，含目录改写与 DependencyMap。
    - catalog: ``TensorCatalog | None``
      当前有效“权重目录”（一般为 preprocess 后目录）。
    """

    def __init__(
        self,
        config: ConvertConfig,
        reader: ICheckpointReader | None = None,
    ) -> None:
        self.config = config
        self.model_path = Path(config.model_path)
        self.save_path = Path(config.save_path)
        self.reader = reader
        self.virtual_tree: nn.Module | None = None
        self.preprocess_result: PreprocessResult | None = None
        self.catalog: TensorCatalog | None = None
        # resolve_worker_device(config.parallel.worker_device) 的结果；executor / processor 共用。
        # worker_backend=process 时固定为 "cpu"。
        self.resolved_worker_device: str = "cpu"
        # 并行模式。npu_multi = 每张 NPU 卡一个子进程（torch.npu.set_device 绑卡）。
        # AscendV1 时 worker 直接写盘，主进程只写 passthrough 并 merge staging。
        # 默认 thread：与 Application 在未开多卡、非 process 时的选择一致。
        self.parallel_mode: Literal["npu_multi", "process", "thread"] = "thread"
        # npu_multi 时供日志使用的设备串，如 ["npu:0", "npu:1", ...]；绑卡用 config.parallel.device_indices。
        self.resolved_worker_devices: list[str] = []


class ICheckpointReader(ABC):
    """
    离线 checkpoint 读取协议（不依赖模型 forward）。

    职责边界：
    - 负责“如何从磁盘读取 index/header/tensor”；
    - 不负责规则匹配、路由、转换、保存等业务决策。

    典型实现位于 ``infra/io``，本接口是 convert 领域层对读盘能力的契约。
    """

    @abstractmethod
    def read_weight_map(self) -> dict[str, str]:
        """
        读取原始索引映射。

        返回:
            dict[tensor_key, shard_path]
        """
        pass

    @abstractmethod
    def read_catalog(self) -> TensorCatalog:
        """
        构建权重目录（TensorCatalog）。

        实现可选择“仅 index 快速构建”或“携带部分 header 信息”。
        """
        pass

    @abstractmethod
    def read_header(self, key: str) -> tuple[str, tuple[int, ...]]:
        """
        读取单个 tensor 的头信息（不加载真实 tensor 数据）。

        返回:
            (dtype_name, shape)
        """
        pass

    @abstractmethod
    def load_tensors(
        self,
        inverse_weight_map: dict[str, list[str] | None],
        device: str = "cpu",
    ) -> dict[str, Any]:
        """
        按任务加载 tensor 数据。

        参数:
            inverse_weight_map:
                dict[shard_path, list[tensor_key] | None]
                - list 为具体要加载的 key 列表
                - None 表示该 shard 全量加载（由实现决定是否支持）
            device:
                目标设备（通常为 "cpu"）

        返回:
            dict[tensor_key, tensor]
        """
        pass

    @abstractmethod
    def read_model_config(self) -> dict[str, Any]:
        """
        读取模型配置（如 config.json）。

        用于提供 preprocess/processor 所需的上下文参数（例如 num_experts、block_size）。
        """
        pass


class IPreprocessExecutor(ABC):
    """
    预处理执行接口：对 catalog 应用 preprocess 规则并产出后续阶段可直接消费的结果。

    输出 ``PreprocessResult``，通常包含：
    - 预处理后的逻辑目录（catalog）
    - DependencyMap（任务分组、逆向加载映射等）
    """

    @abstractmethod
    def run(
        self,
        context: ConvertContext,
        raw_catalog: TensorCatalog,
        rules: list[WeightMappingRule],
    ) -> PreprocessResult:
        pass


class IVirtualModelTreeBuilder(ABC):
    """
    虚拟模块树构建接口。

    输入:
    - preprocess 后目录
    - module_rules / policy / reader

    输出:
    - 懒加载的虚拟 ``nn.Module`` 树（ModelFreeLinear / PassthroughModule 等）
    """

    @abstractmethod
    def build(self, context: ConvertContext, catalog: TensorCatalog) -> nn.Module:
        pass


class IRTaskBuilder(ABC):
    """
    IR 任务构建接口。

    从虚拟树中枚举“需要转换的层”，并结合 convert_rules 生成 ``IRTask`` 列表。
    """

    @abstractmethod
    def build(
        self,
        context: ConvertContext,
        tree: nn.Module,
        catalog: TensorCatalog,
    ) -> list[IRTask]:
        pass


@runtime_checkable
class IIRTransformProcessor(Protocol):
    """
    IR 变换处理器协议（路由图中的“一条有向边”）。

    语义：
    - 一个 processor = 一种 ``src_ir -> dst_ir`` 的转换能力；
    - 例如 ``FP8_BLOCK -> FLOAT``、``FLOAT -> W8A8_MXFP8``。

    字段约定：
    - name: 处理器唯一名称（注册和检索使用）
    - src_ir / dst_ir: 源/目标 IR 类型标签
    - requires_forward / requires_calibration:
      convert 场景必须为 False（否则 router 拒绝注册）
    - loss_level: 转换损失等级（lossy / lossless 等）

    transform 约定：
    - 输入为可转换模块（通常已 lazy_init 完成）；
    - 返回转换后的模块（可原地改，也可返回新对象）；
    - 不应依赖运行时样本数据（forward/calibration 数据）。
    """

    name: str
    src_ir: IRKind
    dst_ir: IRKind
    requires_forward: bool
    requires_calibration: bool
    loss_level: str

    def transform(self, module: nn.Module, context: ConvertContext) -> nn.Module:
        """
        执行单步 IR 变换。

        参数:
            module: 当前层模块（通常为虚拟层或中间转换产物）
            context: 本次运行上下文（可用于读取配置/reader）

        返回:
            转换后的模块（目标 IR 形态）
        """
        pass


class IConvertExecutor(ABC):
    """
    转换执行器接口：调度并执行已完成路由的任务列表（可并行）。

    为何返回 ``Iterator[IRResult]``：
    - 支持流式消费结果（边执行边汇报/边写回）；
    - 降低一次性聚合所有结果的内存压力；
    - 出错时可更早暴露并中止流程。

    典型调用方：
    - ``ConvertApplication.run`` 中：
      ``for result in executor.run(context, routed_tasks): ...``
    """

    @abstractmethod
    def run(
        self,
        context: ConvertContext,
        routed_tasks: list[RoutedTask],
    ) -> Iterator[IRResult]:
        pass
