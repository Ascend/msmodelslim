# 量化格式接入指南

## 简介

本文档面向需要将**新量化落盘格式**接入 msModelSlim 的外部开发者。以 `compressed_tensors` 为完整 1-shot 示例，说明如何基于 `IFormat` 协议实现格式导出，并通过 YAML 配置启用。

> 格式选型请参见《[格式支持矩阵](../user_guide/quantization_formats/README.md)》。AscendV1、MindIE 等旧格式走 Legacy Saver 路径，不在本文档范围内。

## 导出生命周期

`IFormat` 协议定义了三段式导出流程：

```mermaid
flowchart LR
  prepareExport["prepare_export()"] --> traverse["process_module_tensors()"]
  traverse --> finalize["finalize_export()"]
```

## IFormat 协议接口

定义于 [`msmodelslim/format/interface.py`](https://gitcode.com/Ascend/msmodelslim/blob/master/msmodelslim/format/interface.py)：

| 方法 | 是否必须实现 | 职责 |
|------|-------------|------|
| `prepare_export()` | 否（默认空实现） | 量化前准备 |
| `process_module_tensors(prefix, module)` | **是** | 导出模块子树内的量化张量及量化描述信息 |
| `finalize_export(model)` | **是** | 收尾：关闭 writer、写入全模型元数据 |

### ExportContext

导出运行时环境，由框架在构造 `IFormat` 实例时注入：

| 字段 | 说明 |
|------|------|
| `save_directory` | 输出目录 |
| `source_model_path` | 源模型路径（用于复制 HF 辅助文件） |
| `rank` / `world_size` | 分布式 rank 信息 |

### QuantFormatBase（推荐基类）

继承 [`QuantFormatBase`](https://gitcode.com/Ascend/msmodelslim/blob/master/msmodelslim/format/base.py) 可自动获得：

- 模块树遍历（`named_modules` + `processed_modules` 去重）
- `WrapperIR` 原子/非原子处理
- handler 映射分发

子类需实现：

```python
def build_module_handler_map(self) -> Dict[Type[nn.Module], ModuleHandler]:
    """模块类型 → 落盘 handler 的映射表。"""
    ...

def on_float_module(self, prefix: str, module: nn.Module) -> None:
    """未量化模块的 fallback：将原始参数写入目标格式。"""
    ...
```

未在 handler map 中注册的模块类型，会由基类自动调用 `on_float_module()`。也可在 map 中显式注册 `nn.Module: self.on_float_module` 作为兜底 handler。

## 五步接入流程

以下以 `compressed_tensors` 为 1-shot 示例。

### 步骤 1：定义 Config 类

继承 `QuantFormatConfig`，设置唯一的 `type` Literal：

```python
from typing import Literal
from msmodelslim.format.base import QuantFormatConfig

class MyQuantFormatConfig(QuantFormatConfig):
    type: Literal["my_format"] = "my_format"
    part_file_size: int = 4
```

参考：[`CompressedTensorsQuantFormatConfig`](https://gitcode.com/Ascend/msmodelslim/blob/master/msmodelslim/format/compressed_tensors_format/compressed_tensors.py)

### 步骤 2：实现 IFormat 子类

```python
from typing import Dict, Type

import torch
from torch import nn

import msmodelslim.ir as qir
from msmodelslim.format.base import QuantFormatBase, ModuleHandler


class MyQuantFormat(QuantFormatBase):
    def prepare_export(self) -> None:
        # 创建 safetensors writer 等
        self.safetensors_writer = ...

    def build_module_handler_map(self) -> Dict[Type[nn.Module], ModuleHandler]:
        return {
            qir.W8A8StaticFakeQuantLinear: self.on_w8a8_static,
            nn.Linear: self.on_float_linear,
            nn.Module: self.on_float_module,
        }

    def finalize_export(self, model: nn.Module) -> None:
        # 写入 config.json、关闭 writer
        try:
            ...
        finally:
            if self.safetensors_writer is not None:
                self.safetensors_writer.close()
                self.safetensors_writer = None

    def on_w8a8_static(self, prefix: str, module: qir.W8A8StaticFakeQuantLinear) -> None:
        self.safetensors_writer.write(prefix + ".weight", module.weight.to(torch.int8))
        self.safetensors_writer.write(prefix + ".weight_scale", module.weight_scale.unsqueeze(1))
        if module.bias is not None:
            self.safetensors_writer.write(prefix + ".bias", module.bias)
        self.safetensors_writer.write(prefix + ".input_scale", module.input_scale.to(torch.float32))

    def on_float_linear(self, prefix: str, module: nn.Linear) -> None:
        return self.on_float_module(prefix, module)

    def on_float_module(self, prefix: str, module: nn.Module) -> None:
        for name, param in module.named_parameters(recurse=False, prefix=prefix):
            self.safetensors_writer.write(name, param.detach())
```

参考：[`CompressedTensorsQuantFormat`](https://gitcode.com/Ascend/msmodelslim/blob/master/msmodelslim/format/compressed_tensors_format/compressed_tensors.py)

### 步骤 3：注册格式绑定

在 [`msmodelslim/format/registry.py`](https://gitcode.com/Ascend/msmodelslim/blob/master/msmodelslim/format/registry.py) 中注册：

```python
class QuantFormatFactory:
    BUILTIN_BINDINGS = (
        (CompressedTensorsQuantFormatConfig, CompressedTensorsQuantFormat),
        (MyQuantFormatConfig, MyQuantFormat),  # 新增
    )
```

或运行时调用：

```python
from msmodelslim.processor.save.registry import register_quant_format
register_quant_format(MyQuantFormatConfig, MyQuantFormat)
```

### 步骤 4：加入 YAML 联合类型

在 [`QuantFormatConfigUnion`](https://gitcode.com/Ascend/msmodelslim/blob/master/msmodelslim/format/registry.py) 中加入新 Config 类，使 Pydantic 能按 `type` 字段反序列化：

```python
QuantFormatConfigUnion = Annotated[
    Union[
        CompressedTensorsQuantFormatConfig,
        MyQuantFormatConfig,  # 新增
        AscendV1QuantFormatConfig,
        MindIEQuantFormatConfig,
    ],
    Field(discriminator="type"),
]
```

`import msmodelslim.format` 时会自动执行 `QuantFormatFactory.install()` 完成注册。

### 步骤 5：YAML 配置启用

```yaml
spec:
  save:
    - type: "my_format"
      part_file_size: 4
```

## Handler 编写要点

### QIR 模块映射

每种 QIR 量化模块类型需对应一个 handler，负责将模块参数写入目标格式：

```python
def build_module_handler_map(self):
    return {
        qir.W8A8StaticFakeQuantLinear: self.on_w8a8_static,
        qir.W8A8DynamicPerChannelFakeQuantLinear: self.on_w8a8_dynamic,
        nn.Linear: self.on_float_linear,
        nn.Module: self.on_float_module,
    }
```

### WrapperIR 处理

`QuantFormatBase` 自动处理 `WrapperIR`：

- **非原子性**（`is_atomic() = False`）：先处理被包装模块，再处理包装器
- **原子性**（`is_atomic() = True`）：只处理包装器，跳过被包装模块

### 未量化层 Fallback

未在 handler map 中注册的模块类型，默认调用 `on_float_module()`，遍历 `named_parameters` 直接写入原始参数。

### 元数据反向推导

推荐在 `finalize_export()` 中扫描模型 QIR 模块，反向推导格式元数据（如 compressed-tensors 的 `config.json` → `quantization_config`），而非在 handler 中逐层累积。

## 测试与验证

参考 [`test/cases/format/compressed_tensors_format/`](https://gitcode.com/Ascend/msmodelslim/tree/master/test/cases/format/compressed_tensors_format/)：

1. 实现 `MockSafetensorsWriter` 内存 writer
2. 构造最小 QIR 模型（W8A8 Static / Dynamic）
3. 调用 `prepare_export()` → `process_module_tensors()` → `finalize_export()`
4. 断言 safetensors 张量键名、dtype、shape 与 config 元数据

## 完整参考实现

| 组件 | 路径 |
|------|------|
| IFormat 协议 | [`msmodelslim/format/interface.py`](https://gitcode.com/Ascend/msmodelslim/blob/master/msmodelslim/format/interface.py) |
| QuantFormatBase | [`msmodelslim/format/base.py`](https://gitcode.com/Ascend/msmodelslim/blob/master/msmodelslim/format/base.py) |
| 注册表 | [`msmodelslim/format/registry.py`](https://gitcode.com/Ascend/msmodelslim/blob/master/msmodelslim/format/registry.py) |
| 保存处理器 | [`msmodelslim/processor/save/processor.py`](https://gitcode.com/Ascend/msmodelslim/blob/master/msmodelslim/processor/save/processor.py) |
| compressed-tensors 实现 | [`msmodelslim/format/compressed_tensors_format/`](https://gitcode.com/Ascend/msmodelslim/tree/master/msmodelslim/format/compressed_tensors_format) |
| 单元测试 | [`test/cases/format/compressed_tensors_format/`](https://gitcode.com/Ascend/msmodelslim/tree/master/test/cases/format/compressed_tensors_format) |

## 相关文档

- 《[格式支持矩阵](../user_guide/quantization_formats/README.md)》
- 《[compressed-tensors 格式说明](../user_guide/quantization_formats/compressed_tensors.md)》 — 1-shot 参考实现的目标格式
- 《[AscendV1 格式说明](../user_guide/quantization_formats/ascendv1.md)》 — Legacy 格式对比参考
