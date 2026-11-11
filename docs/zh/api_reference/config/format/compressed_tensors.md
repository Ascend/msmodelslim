<!-- generated-by: tools.config_docgen ; class: msmodelslim.format.compressed_tensors_format.compressed_tensors.CompressedTensorsQuantFormatConfig -->
> **自动生成文档，请勿手工修改。**
>
> 本页由 `tools.config_docgen` 根据源码注解生成。
> 修改字段描述、类概述或校验逻辑后，请重新生成并提交生成稿。

# compressed_tensors 配置说明

## 1. 配置概述

compressed_tensors 保存格式配置，导出 safetensors 权重与 config.json。

| 项目 | 内容 |
|------|------|
| 配置类 | `CompressedTensorsQuantFormatConfig` |
| 源码 | [compressed_tensors.py](../../../../../msmodelslim/format/compressed_tensors_format/compressed_tensors.py) |

## 2. 参数列表

<h3 id="2-1-compressed-tensors">2.1 CompressedTensorsQuantFormatConfig</h3>

| 字段路径 | 类型 | 必选/可选 | 默认值 | 取值范围或格式 | 含义 | 引用配置 |
|----------|------|-----------|--------|----------------|------|----------|
| `type` | `string` | 可选 | `compressed_tensors` | `compressed_tensors` | 保存格式类型，固定为 `compressed_tensors`。 | 无 |
| `part_file_size` | `int` | 可选 | `4` | — | 分片文件大小，单位 GB；0 表示不分片。 | 无 |

**配置约束**

- 无。

## 3. 完整配置参考

```yaml
apiversion: modelslim_v1
spec:
  save:
  - type: compressed_tensors
    part_file_size: 4
```
