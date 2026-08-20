<!-- generated-by: skills/docs-management/scripts/gen_quant_config_docs.py ; class: msmodelslim.format.ascendV1_format.ascendV1.AscendV1QuantFormatConfig -->
# ascendv1_saver 配置说明

## 1. 配置概述

AscendV1 保存格式配置，导出昇腾落盘格式的权重文件。

| 项目 | 内容 |
|------|------|
| 配置类 | `AscendV1QuantFormatConfig` |
| 源码 | [ascendV1.py](../../../../../msmodelslim/format/ascendV1_format/ascendV1.py) |

## 2. 参数列表

<h3 id="2-1-ascendv1-saver">2.1 AscendV1QuantFormatConfig</h3>

| 字段路径 | 类型 | 必选/可选 | 默认值 | 取值范围或格式 | 含义 | 引用配置 |
|----------|------|-----------|--------|----------------|------|----------|
| `type` | `string` | 可选 | `ascendv1_saver` | `ascendv1_saver` | 保存格式类型，固定为 `ascendv1_saver`。 | 无 |
| `part_file_size` | `int` | 可选 | `4` | — | 分片文件大小，单位 GB；0 表示不分片。 | 无 |
| `ext` | `object` | 可选 | `{}` | — | 保存格式扩展参数，随实现而定；空对象表示无扩展参数。 | 无 |

**配置约束**

- 无。

## 3. 完整配置参考

```yaml
apiversion: modelslim_v1
spec:
  save:
  - type: ascendv1_saver
    part_file_size: 4
    ext: {}
```
