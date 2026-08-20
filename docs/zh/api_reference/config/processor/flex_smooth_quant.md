<!-- generated-by: skills/docs-management/scripts/gen_quant_config_docs.py ; class: msmodelslim.processor.anti_outlier.flex_smooth.processor.FlexSmoothQuantProcessorConfig -->
# flex_smooth_quant 配置说明

## 1. 配置概述

FlexSmoothQuant 平滑量化处理器配置。

| 项目 | 内容 |
|------|------|
| 配置类 | `FlexSmoothQuantProcessorConfig` |
| 源码 | [processor.py](../../../../../msmodelslim/processor/anti_outlier/flex_smooth/processor.py) |

## 2. 参数列表

<h3 id="2-1-flex-smooth-quant">2.1 FlexSmoothQuantProcessorConfig</h3>

| 字段路径 | 类型 | 必选/可选 | 默认值 | 取值范围或格式 | 含义 | 引用配置 |
|----------|------|-----------|--------|----------------|------|----------|
| `type` | `string` | 可选 | `flex_smooth_quant` | `flex_smooth_quant` | 处理器类型，固定为 `flex_smooth_quant`。 | 无 |
| `alpha` | `float` | 可选 | `null` | — | 激活→权重的平滑迁移强度（0~1），越大迁移越多离群值到权重。 | 无 |
| `beta` | `float` | 可选 | `null` | — | 额外的平滑调优系数（0~1），配合 alpha 使用。 | 无 |
| `enable_subgraph_type` | `list[any]` | 可选 | `['norm-linear', 'linear-linear', 'ov', 'up-down']` | — | 应用平滑的子图类型列表，默认 `norm-linear`、`linear-linear`、`ov`、`up-down`。 | 无 |
| `include` | `list[string] / null` | 可选 | `null` | — | 包含的模块名称模式；不设置表示全部匹配。 | 无 |
| `exclude` | `list[string] / null` | 可选 | `null` | — | 排除的模块名称模式，优先级高于 `include`。 | 无 |

**配置约束**

- 无。

## 3. 完整配置参考

```yaml
apiversion: modelslim_v1
spec:
  process:
  - type: flex_smooth_quant
    enable_subgraph_type:
    - norm-linear
    - linear-linear
    - ov
    - up-down
```
