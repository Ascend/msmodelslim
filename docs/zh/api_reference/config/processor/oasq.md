<!-- generated-by: skills/docs-management/scripts/gen_quant_config_docs.py ; class: msmodelslim.processor.anti_outlier.oasq.processor.OASQProcessorConfig -->
# oasq 配置说明

## 1. 配置概述

OASQ（Outlier-Aware Smooth Quantization）处理器配置。

| 项目 | 内容 |
|------|------|
| 配置类 | `OASQProcessorConfig` |
| 源码 | [processor.py](../../../../../msmodelslim/processor/anti_outlier/oasq/processor.py) |

## 2. 参数列表

<h3 id="2-1-oasq">2.1 OASQProcessorConfig</h3>

| 字段路径 | 类型 | 必选/可选 | 默认值 | 取值范围或格式 | 含义 | 引用配置 |
|----------|------|-----------|--------|----------------|------|----------|
| `type` | `string` | 可选 | `oasq` | `oasq` | 处理器类型，固定为 `oasq`。 | 无 |
| `max_iters` | `int / null` | 可选 | `null` | — | 最大迭代次数；不设置时使用实现默认值，必须大于0。 | 无 |
| `symmetric` | `bool` | 可选 | `true` | — | 是否对称量化后续的权重/激活。 | 无 |
| `enable_subgraph_type` | `list[any]` | 可选 | `['norm-linear', 'linear-linear', 'ov', 'up-down']` | — | 应用 OASQ 的子图类型列表，默认 `norm-linear`、`linear-linear`、`ov`、`up-down`。 | 无 |
| `include` | `list[string] / null` | 可选 | `null` | — | 包含的模块名称模式；不设置表示全部匹配。 | 无 |
| `exclude` | `list[string] / null` | 可选 | `null` | — | 排除的模块名称模式，优先级高于 `include`。 | 无 |

**配置约束**

- 校验 max_iters：设置时必须大于 0。

## 3. 完整配置参考

```yaml
apiversion: modelslim_v1
spec:
  process:
  - type: oasq
    symmetric: true
    enable_subgraph_type:
    - norm-linear
    - linear-linear
    - ov
    - up-down
```
