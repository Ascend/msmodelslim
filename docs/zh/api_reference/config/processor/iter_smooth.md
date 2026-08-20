<!-- generated-by: skills/docs-management/scripts/gen_quant_config_docs.py ; class: msmodelslim.processor.anti_outlier.iter_smooth.processor.IterSmoothProcessorConfig -->
# iter_smooth 配置说明

## 1. 配置概述

迭代平滑（IterativeSmooth）处理器配置。

| 项目 | 内容 |
|------|------|
| 配置类 | `IterSmoothProcessorConfig` |
| 源码 | [processor.py](../../../../../msmodelslim/processor/anti_outlier/iter_smooth/processor.py) |

## 2. 参数列表

<h3 id="2-1-iter-smooth">2.1 IterSmoothProcessorConfig</h3>

| 字段路径 | 类型 | 必选/可选 | 默认值 | 取值范围或格式 | 含义 | 引用配置 |
|----------|------|-----------|--------|----------------|------|----------|
| `type` | `string` | 可选 | `iter_smooth` | `iter_smooth` | 处理器类型，固定为 `iter_smooth`。 | 无 |
| `alpha` | `float` | 可选 | `0.9` | — | 平滑迁移强度（0~1），越大迁移越多离群值到权重。 | 无 |
| `scale_min` | `float` | 可选 | `1e-05` | — | 平滑缩放因子的最小值下限，防止数值下溢。 | 无 |
| `symmetric` | `bool` | 可选 | `true` | — | 是否对称量化后续的权重/激活。 | 无 |
| `enable_subgraph_type` | `list[any]` | 可选 | `['norm-linear', 'linear-linear', 'ov', 'up-down']` | — | 应用迭代平滑的子图类型列表，默认 `norm-linear`、`linear-linear`、`ov`、`up-down`。 | 无 |
| `include` | `list[string] / null` | 可选 | `null` | — | 包含的模块名称模式；不设置表示全部匹配。 | 无 |
| `exclude` | `list[string] / null` | 可选 | `null` | — | 排除的模块名称模式，优先级高于 `include`。 | 无 |

**配置约束**

- 无。

## 3. 完整配置参考

```yaml
apiversion: modelslim_v1
spec:
  process:
  - type: iter_smooth
    alpha: 0.9
    scale_min: 1.0e-05
    symmetric: true
    enable_subgraph_type:
    - norm-linear
    - linear-linear
    - ov
    - up-down
```
