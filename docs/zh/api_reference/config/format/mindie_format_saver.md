<!-- generated-by: skills/docs-management/scripts/gen_quant_config_docs.py ; class: msmodelslim.format.mindie_format.mindie.MindIEQuantFormatConfig -->
# mindie_format_saver 配置说明

## 1. 配置概述

MindIE 保存格式配置，导出 MindIE 落盘格式的权重文件。

| 项目 | 内容 |
|------|------|
| 配置类 | `MindIEQuantFormatConfig` |
| 源码 | [mindie.py](../../../../../msmodelslim/format/mindie_format/mindie.py) |

## 2. 参数列表

<h3 id="2-1-mindie-format-saver">2.1 MindIEQuantFormatConfig</h3>

| 字段路径 | 类型 | 必选/可选 | 默认值 | 取值范围或格式 | 含义 | 引用配置 |
|----------|------|-----------|--------|----------------|------|----------|
| `type` | `string` | 可选 | `mindie_format_saver` | `mindie_format_saver` | 保存格式类型，固定为 `mindie_format_saver`。 | 无 |
| `part_file_size` | `int` | 可选 | `4` | — | 分片文件大小，单位 GB；0 表示不分片。 | 无 |
| `ext` | `object` | 可选 | `{}` | — | 保存格式扩展参数，随实现而定；空对象表示无扩展参数。 | 无 |

**配置约束**

- 无。

## 3. 完整配置参考

```yaml
apiversion: modelslim_v1
spec:
  save:
  - type: mindie_format_saver
    part_file_size: 4
    ext: {}
```
