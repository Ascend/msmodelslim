<!-- generated-by: tools.config_docgen ; class: msmodelslim.processor.anti_outlier.smooth_quant.processor.SmoothQuantProcessorConfig -->
> **自动生成文档，请勿手工修改。**
>
> 本页由 `tools.config_docgen` 根据源码注解生成。
> 修改字段描述、类概述或校验逻辑后，请重新生成并提交生成稿。

# smooth_quant 配置说明

## 1. 配置概述

SmoothQuant 平滑量化处理器配置。

| 项目 | 内容 |
|------|------|
| 配置类 | `SmoothQuantProcessorConfig` |
| 源码 | [processor.py](../../../../../msmodelslim/processor/anti_outlier/smooth_quant/processor.py) |

## 2. 参数列表

<h3 id="2-1-smooth-quant">2.1 SmoothQuantProcessorConfig</h3>

| 字段路径 | 类型 | 必选/可选 | 默认值 | 取值范围或格式 | 含义 | 引用配置 |
|----------|------|-----------|--------|----------------|------|----------|
| `type` | `string` | 可选 | `smooth_quant` | `smooth_quant` | 处理器类型，固定为 `smooth_quant`。 | 无 |
| `alpha` | `float` | 可选 | `0.5` | >0.0；<1.0；0 < 值 < 1 | 平滑迁移强度（0~1），越大表示把越多的激活离群值迁移到权重。 | 无 |
| `symmetric` | `bool` | 可选 | `true` | — | 是否对称量化；影响平滑后量化的对称性。 | 无 |
| `include` | `list[string] / null` | 可选 | `null` | 列表元素：字符串长度 ≤ 200；或 null | 包含的模块名称模式；不设置表示全部匹配。 | 无 |
| `exclude` | `list[string] / null` | 可选 | `null` | 列表元素：字符串长度 ≤ 200；或 null | 排除的模块名称模式，优先级高于 `include`。 | 无 |

**配置约束**

- 无。

## 3. 完整配置参考

```yaml
apiversion: modelslim_v1
spec:
  process:
  - type: smooth_quant
    alpha: 0.5
    symmetric: true
```
