<!-- generated-by: tools.config_docgen ; class: msmodelslim.core.devkit.save.QuantSaveProcessorConfig -->
> **自动生成文档，请勿手工修改。**
>
> 本页由 `tools.config_docgen` 根据源码注解生成。
> 修改字段描述、类概述或校验逻辑后，请重新生成并提交生成稿。

# saver 配置说明

## 1. 配置概述

统一保存处理器配置。

- 该处理器位于 `spec.process[]`。`type: saver` 是内部保存处理器的分派标识，不是模型导出格式；用户可见的导出格式由 `format.type` 决定。
- 根级 `spec.save[]` 支持 `ascendv1_saver`、`compressed_tensors` 和 `mindie_format_saver`；`ascendv1_saver` 与 `mindie_format_saver` 会路由到各自的保存处理器，`compressed_tensors` 则使用该统一保存处理器。
- 直接使用 `QuantSaveProcessor` 时，`format` 当前仅支持 `compressed_tensors`。通常由框架根据根级 `spec.save[]` 自动注入 `format`，普通量化任务无需在 `spec.process[]` 中手工编写 `type: saver`。

| 项目 | 内容 |
|------|------|
| 配置类 | `QuantSaveProcessorConfig` |
| 源码 | [save.py](../../../../../msmodelslim/core/devkit/save.py) |

## 2. 参数列表

<h3 id="2-1-saver">2.1 QuantSaveProcessorConfig</h3>

| 字段路径 | 类型 | 必选/可选 | 默认值 | 取值范围或格式 | 含义 | 引用配置 |
|----------|------|-----------|--------|----------------|------|----------|
| `type` | `string` | 可选 | `saver` | `saver` | 处理器分派标识，固定为 `saver`；该值不是导出格式，导出格式由 `format.type` 指定。 | 无 |
| `format` | `object` | 必选 | 无 | — | 导出格式配置（单对象）；由 `QuantSaveProcessor` 直接使用时当前必须为 `compressed_tensors`，通常由框架从 `spec.save[]` 自动注入。 | 本页 <a href="#2-2-quantformatconfig">§2.2</a> |

**配置约束**

- 无。

<h3 id="2-2-quantformatconfig">2.2 《<a href="../format/ascendv1_saver.md">QuantFormatConfig</a>》</h3>

**派生类**

| 配置类 | `type` | 说明 | 文档 |
|--------|----------|------|------|
| `AscendV1QuantFormatConfig` | `ascendv1_saver` | AscendV1 保存格式配置，导出昇腾落盘格式的权重文件。 | 《[ascendv1_saver 配置说明](../format/ascendv1_saver.md)》 |
| `CompressedTensorsQuantFormatConfig` | `compressed_tensors` | compressed_tensors 保存格式配置，导出 safetensors 权重与 config.json。 | 《[compressed_tensors 配置说明](../format/compressed_tensors.md)》 |
| `MindIEQuantFormatConfig` | `mindie_format_saver` | MindIE 保存格式配置，导出 MindIE 落盘格式的权重文件。 | 《[mindie_format_saver 配置说明](../format/mindie_format_saver.md)》 |

## 3. 完整配置参考

```yaml
apiversion: modelslim_v1
spec:
  save:
  - type: ascendv1_saver
    part_file_size: 4
```

## 4. 注意事项

- 如需选择导出格式，请在根级 `spec.save[]` 中配置；只有直接构造统一保存处理器时，才需要在 `format` 中显式指定 `type: compressed_tensors` 及其扩展字段。
