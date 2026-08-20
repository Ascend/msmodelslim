<!-- generated-by: skills/docs-management/scripts/gen_quant_config_docs.py ; class: msmodelslim.processor.save.processor.QuantSaveProcessorConfig -->
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
| `format` | `object` | 必选 | 无 | — | 导出格式配置（单对象），见《QuantFormatConfig 配置说明》；由保存处理器自动注入。 | QuantFormatConfig |

**配置约束**

- 无。

## 3. 完整配置参考

```yaml
apiversion: modelslim_v1
spec:
  process:
  - type: saver
    format:
      type: _auto_save
```
