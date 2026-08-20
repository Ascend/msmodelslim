<!-- generated-by: skills/docs-management/scripts/gen_quant_config_docs.py ; class: msmodelslim.processor.kv_smooth.processor.KVSmoothProcessorConfig -->
# kv_smooth 配置说明

## 1. 配置概述

KV cache 平滑处理器配置。

| 项目 | 内容 |
|------|------|
| 配置类 | `KVSmoothProcessorConfig` |
| 源码 | [processor.py](../../../../../msmodelslim/processor/kv_smooth/processor.py) |

## 2. 参数列表

<h3 id="2-1-kv-smooth">2.1 KVSmoothProcessorConfig</h3>

| 字段路径 | 类型 | 必选/可选 | 默认值 | 取值范围或格式 | 含义 | 引用配置 |
|----------|------|-----------|--------|----------------|------|----------|
| `type` | `string` | 可选 | `kv_smooth` | `kv_smooth` | 处理器类型，固定为 `kv_smooth`。 | 无 |
| `smooth_factor` | `float` | 可选 | `1.0` | — | KV 平滑因子，必须大于0；越大平滑越强。 | 无 |
| `include` | `list[string]` | 可选 | `['*']` | — | 包含的模块名称模式，默认 `*` 匹配全部模块 | 无 |
| `exclude` | `list[string]` | 可选 | `[]` | — | 排除的模块名称模式，优先级高于 `include` | 无 |

**配置约束**

- 无。

## 3. 完整配置参考

```yaml
apiversion: modelslim_v1
spec:
  process:
  - type: kv_smooth
    smooth_factor: 1.0
    include:
    - '*'
    exclude: []
```
