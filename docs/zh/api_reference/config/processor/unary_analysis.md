<!-- generated-by: skills/docs-management/scripts/gen_quant_config_docs.py ; class: msmodelslim.processor.analysis.unary_operator.processor.UnaryAnalysisProcessorConfig -->
# unary_analysis 配置说明

## 1. 配置概述

一元（无量化）敏感度分析处理器配置。

| 项目 | 内容 |
|------|------|
| 配置类 | `UnaryAnalysisProcessorConfig` |
| 源码 | [processor.py](../../../../../msmodelslim/processor/analysis/unary_operator/processor.py) |

## 2. 参数列表

<h3 id="2-1-unary-analysis">2.1 UnaryAnalysisProcessorConfig</h3>

| 字段路径 | 类型 | 必选/可选 | 默认值 | 取值范围或格式 | 含义 | 引用配置 |
|----------|------|-----------|--------|----------------|------|----------|
| `type` | `string` | 可选 | `unary_analysis` | `unary_analysis` | 处理器类型，固定为 `unary_analysis`。 | 无 |
| `metrics` | `string` | 可选 | `kurtosis` | — | 分析指标：`quantile`（分位数）、`std`（标准差）、`kurtosis`（峰度） | 无 |
| `patterns` | `list[string]` | 可选 | `['*']` | — | 待分析的层名模式列表，默认 `*` 匹配全部 | 无 |

**配置约束**

- 无。

## 3. 完整配置参考

```yaml
apiversion: modelslim_v1
spec:
  process:
  - type: unary_analysis
    metrics: kurtosis
    patterns:
    - '*'
```
