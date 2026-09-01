# 量化格式接入指南

## 1. 适用范围

本流程面向需要将**新量化落盘格式**接入 msModelSlim 的外部开发者与维护者。以 `compressed_tensors` 为完整 1-shot 示例，说明如何基于 `IFormat` 协议实现格式导出，并通过 YAML 配置启用。

不适用：仅消费已有格式的用户（请先阅读《[量化格式](README.md)》地图并跳转对应使用指南）。

## 2. 流程关系与前置条件

**上级流程**：开源贡献 / 新格式需求评审通过后。

**前置条件**：

- 已阅读《[量化格式](README.md)》并确认需新增格式（而非扩展既有 handler）
- 已明确目标推理框架的加载约定（张量命名与元数据 schema）

**后续操作**：代码合入后补齐文档：按《[资料规范](../../contributing/development_guide/docs_standards/README.md)》与《[新建量化格式资料](../../../../skills/docs-management/scenarios/add-quantization-format.md)》编写格式词条与使用指南，并更新《[量化格式](README.md)》格式地图。

## 3. 输入和交付件

| 类型 | 名称 | 来源或保存位置 | 格式或约束 | 验收方式 |
| --- | --- | --- | --- | --- |
| 输入 | 目标格式规范 | 上游项目或内部 RFC | 张量键名、dtype、元数据字段可核对 | 与推理侧加载文档对照 |
| 输入 | QIR 模块类型列表 | `msmodelslim/ir` | 需实现 handler 的 FakeQuant 类型 | 列出模块类名 |
| 交付件 | IFormat 实现与 Config | `msmodelslim/format/<name>/` | Config 加入 `QuantFormatConfigUnion` | 单测通过 |
| 交付件 | YAML 可启用的 `type` | 一键量化 `spec.save` | 与 Config `Literal` 一致 | 配置反序列化成功 |
| 交付件 | 单元测试 | `test/cases/format/` | 覆盖张量键与元数据 | CI / 本地断言通过 |

## 4. 流程总览

`IFormat` 协议定义了三段式导出流程：

```mermaid
flowchart LR
  prepareExport["prepare_export"] --> traverse["process_module_tensors"]
  traverse --> finalize["finalize_export"]
```

## 5. 操作步骤

### 步骤 1：适配协议与基类

**目标**：理解 `IFormat` / `ExportContext` / `QuantFormatBase` 职责边界，并据此完成新格式适配所需的协议对齐与基类选型。

**操作**：阅读 [`msmodelslim/format/interface.py`](../../../../msmodelslim/format/interface.py) 与 [`msmodelslim/format/base.py`](../../../../msmodelslim/format/base.py)。

| 方法 | 是否必须实现 | 职责 |
| --- | --- | --- |
| `prepare_export()` | 否（默认空实现） | 量化前准备 |
| `process_module_tensors(prefix, module)` | **是** | 导出模块子树内的量化张量及量化描述信息 |
| `finalize_export(model)` | **是** | 收尾：关闭 writer、写入全模型元数据 |

`ExportContext` 字段：`save_directory`、`source_model_path`、`rank` / `world_size`。

继承 `QuantFormatBase` 可自动获得模块树遍历、`WrapperIR` 处理与 handler 映射分发。子类需实现：

```python
def build_module_handler_map(self) -> Dict[Type[nn.Module], ModuleHandler]:
    """模块类型 → 落盘 handler 的映射表。"""
    ...

def on_float_module(self, prefix: str, module: nn.Module) -> None:
    """未量化模块的 fallback：将原始参数写入目标格式。"""
    ...
```

**输出**：选定基类与需支持的 QIR 类型清单。

### 步骤 2：定义 Config 类

**操作**：继承 `QuantFormatConfig`，设置唯一的 `type` Literal：

```python
from typing import Literal
from msmodelslim.format.base import QuantFormatConfig

class MyQuantFormatConfig(QuantFormatConfig):
    type: Literal["my_format"] = "my_format"
    part_file_size: int = 4
```

参考：[`CompressedTensorsQuantFormatConfig`](../../../../msmodelslim/format/compressed_tensors_format/compressed_tensors.py)

**输出**：Config 类源文件。

### 步骤 3：实现 IFormat 子类

**操作**：实现 handler 与三段生命周期。最小示意：

```python
from typing import Dict, Type

import torch
from torch import nn

import msmodelslim.ir as qir
from msmodelslim.format.base import QuantFormatBase, ModuleHandler


class MyQuantFormat(QuantFormatBase):
    def prepare_export(self) -> None:
        self.safetensors_writer = ...

    def build_module_handler_map(self) -> Dict[Type[nn.Module], ModuleHandler]:
        return {
            qir.W8A8StaticFakeQuantLinear: self.on_w8a8_static,
            nn.Linear: self.on_float_linear,
            nn.Module: self.on_float_module,
        }

    def finalize_export(self, model: nn.Module) -> None:
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

参考：[`CompressedTensorsQuantFormat`](../../../../msmodelslim/format/compressed_tensors_format/compressed_tensors.py)

**Handler 编写要点**：

- 每种 QIR 量化模块类型需对应一个 handler。
- `WrapperIR`：非原子性先处理被包装模块再处理包装器；原子性只处理包装器。
- 未注册模块类型默认 `on_float_module()`。
- 推荐在 `finalize_export()` 中扫描 QIR 反向推导元数据（如 compressed-tensors 的 `quantization_config`）。

**输出**：Format 实现类。

### 步骤 4：注册格式绑定与 YAML 联合类型

**目标**：使 YAML spec.save[].type 能按 type 反序列化为新 Config。

**操作**：在 [msmodelslim/format/registry.py](../../../../msmodelslim/format/registry.py) 中 import 新 Config，并加入 QuantFormatConfigUnion：

```python
from msmodelslim.format.my_format.my_format import MyQuantFormatConfig

QuantFormatConfigUnion = Annotated[
    Union[
        CompressedTensorsQuantFormatConfig,
        AscendV1QuantFormatConfig,
        MindIEQuantFormatConfig,
        MyQuantFormatConfig,  # 新增
    ],
    Field(discriminator="type"),
]
```

加入后，parse_format_config 与一键量化 spec.save 即可识别 type: "my_format"。

**输出**：可被 Pydantic 按 `type` 反序列化的配置绑定。

### 步骤 5：配置启用、测试与资料同步

**操作**：

1. YAML 启用：

   ```yaml
   spec:
     save:
       - type: "my_format"
         part_file_size: 4
   ```

2. 参考 [`test/cases/format/compressed_tensors_format/`](../../../../test/cases/format/compressed_tensors_format)：实现 Mock writer、构造最小 QIR 模型、调用三段生命周期并断言键名 / dtype / 元数据。

3. 按资料标准在 `docs/zh/knowledge_base/quantization_format/<format_name>/` 新增词条与使用指南，并在《[量化格式](README.md)》地图登记链接。

**输出**：可通过一键量化 YAML（`spec.save.type`）启动导出、单测通过、文档地图可导航。

**通过条件**：配置可解析；单测断言通过；地图存在新格式词条与使用指南链接。

## 6. 验收条件

- 仅外部脚本导出、未注册进 msModelSlim 配置与工厂，不视为接入完成。
- 缺少对关键 QIR 类型的 handler 或元数据写入失败，不得合入。
- 文档未登记到格式地图时，资料验收不通过。

## 7. 术语

| 术语 | 简述 | 链接 |
| --- | --- | --- |
| 量化格式 | 落盘与加载协议 | 《[量化格式](README.md)》 |
| compressed-tensors | 1-shot 参考格式 | 《[compressed-tensors](compressed_tensors/term_compressed_tensors.md)》 |
| AscendV1 | 昇腾默认格式（对照） | 《[AscendV1](ascendv1/term_ascendv1.md)》 |

## 8. 接口文档列表

| 接口或能力 | 简述 | 链接 |
| --- | --- | --- |
| `IFormat` | 量化落盘格式协议，需实现三个接口：`prepare_export`、`process_module_tensors`、`finalize_export` | [接口定义](../../../../msmodelslim/format/interface.py) |
| `QuantFormatFactory` / registry | `QuantFormatConfigUnion` 反序列化与格式工厂构造 | [注册与工厂](../../../../msmodelslim/format/registry.py) |
