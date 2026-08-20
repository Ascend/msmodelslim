<!-- generated-by: skills/docs-management/scripts/gen_quant_config_docs.py ; class: msmodelslim.processor.svd_residual.processor.SVDResidualProcessorConfig -->
# svd_res 配置说明

## 1. 配置概述

SVD 残差（低秩补偿）处理器配置。

| 项目 | 内容 |
|------|------|
| 配置类 | `SVDResidualProcessorConfig` |
| 源码 | [processor.py](../../../../../msmodelslim/processor/svd_residual/processor.py) |

## 2. 参数列表

<h3 id="2-1-svd-res">2.1 SVDResidualProcessorConfig</h3>

| 字段路径 | 类型 | 必选/可选 | 默认值 | 取值范围或格式 | 含义 | 引用配置 |
|----------|------|-----------|--------|----------------|------|----------|
| `type` | `string` | 可选 | `svd_res` | `svd_res` | 处理器类型，固定为 `svd_res`。 | 无 |
| `rank` | `int` | 可选 | `32` | >0 | 低秩分解的秩，必须大于0 | 无 |
| `include` | `list[string]` | 可选 | `['*']` | — | 包含的模块名称模式，默认 `*` 匹配全部模块 | 无 |
| `exclude` | `list[string]` | 可选 | `[]` | — | 排除的模块名称模式，优先级高于 `include` | 无 |

**配置约束**

- 无。

## 3. 完整配置参考

```yaml
apiversion: modelslim_v1
spec:
  process:
  - type: svd_res
    rank: 32
    include:
    - '*'
    exclude: []
```
