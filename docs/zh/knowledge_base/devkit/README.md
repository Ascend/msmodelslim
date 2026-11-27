# DevKit

DevKit 是 msModelSlim V1 量化流水线中的**编排与资源管理**能力，挂在 `spec.process[]`（或由框架注入）上，以 `type` 分派，但不属于量化算法本体。算法负责**如何量化**，DevKit 负责**如何组合执行**。

> **阅读建议**：已对外开放的 DevKit 组件提供使用指南。字段级类型、默认值与完整 YAML 参考见《[配置说明导航](../../api_reference/config/README.md)》中对应 `type` 页。源码位于 [`msmodelslim/core/devkit/`](../../../../msmodelslim/core/devkit/)。

## 1. 组件分类

按在量化流水线中的职责对DevKit组件分类：

| 分类 | 职责 | 组件 |
|------|------|------|
| **量化校准** | 编排校准前向、合并需共享校准推理的处理器，降低重复跑数 | `group` |
| **设备与资源** | 在处理前后管理模块上/下设备与内存（框架侧注入，一般不手写） | `load` |
| **结果落盘** | 将量化结果按配置的格式写出（通常由 `spec.save[]` / 框架侧驱动） | `save` |

当前对外提供使用指南的仅 **量化校准** 类中的 `group`；`load` / `save` 保留在源码中供流水线使用。

## 2. 组件索引

| 分类 | 组件 | `type` | 作用 | 使用指南 |
|------|------|--------|------|----------|
| 量化校准 | Group | `group` | 合并多个前向处理器，共享校准推理 | [组合处理器 (Group) 使用指南](group/usage_group.md) |

## 3. 与相邻领域的边界

| 领域 | 关系 |
|------|------|
| [量化算法](../quantization_algorithm/README.md) | 算法处理器可出现在 `group.configs` 或与 DevKit 组件同列于 `process`；DevKit 不实现量化公式 |
| [量化格式](../quantization_format/README.md) | 落盘格式由 `spec.save[]`（如 `ascendv1_saver`）配置，详见量化格式领域 |
| [量化模式](../quantization_schema/README.md) | 模式描述 IR / FakeQuant 结构；DevKit 不定义新模式 |

## 4. 源码位置

```text
msmodelslim/core/devkit/
├── __init__.py
├── group.py          # type: group（量化校准）
├── load.py           # type: load（设备与资源）
└── save.py           # type: saver（结果落盘）
```
