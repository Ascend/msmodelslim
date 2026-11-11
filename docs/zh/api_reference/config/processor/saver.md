<!-- generated-by: tools.config_docgen ; class: msmodelslim.processor.save.processor.QuantSaveProcessorConfig -->
> **自动生成文档，请勿手工修改。**
>
> 本页由 `tools.config_docgen` 根据源码注解生成。
> 修改字段描述、类概述或校验逻辑后，请重新生成并提交生成稿。

# saver 配置说明

## 1. 配置概述

统一保存处理器配置。

| 项目 | 内容 |
|------|------|
| 配置类 | `QuantSaveProcessorConfig` |
| 源码 | [processor.py](../../../../../msmodelslim/processor/save/processor.py) |

## 2. 参数列表

<h3 id="2-1-saver">2.1 QuantSaveProcessorConfig</h3>

| 字段路径 | 类型 | 必选/可选 | 默认值 | 取值范围或格式 | 含义 | 引用配置 |
|----------|------|-----------|--------|----------------|------|----------|
| `type` | `string` | 可选 | `saver` | `saver` | 处理器类型，固定为 `saver`。 | 无 |
| `format` | `object` | 必选 | 无 | — | 导出格式配置（单对象），由保存处理器自动注入。 | 《[QuantFormatConfig](../../../../../msmodelslim/format/base.py)》 |

**配置约束**

- 无。

## 3. 完整配置参考

```yaml
apiversion: modelslim_v1
spec:
  save:
  - type: ascendv1_saver
    part_file_size: 4
```
