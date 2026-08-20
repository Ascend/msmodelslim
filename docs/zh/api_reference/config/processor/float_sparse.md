<!-- generated-by: skills/docs-management/scripts/gen_quant_config_docs.py ; class: msmodelslim.processor.sparse.float_sparse.FloatSparseProcessorConfig -->
# float_sparse 配置说明

## 1. 配置概述

浮点稀疏处理器配置。

| 项目 | 内容 |
|------|------|
| 配置类 | `FloatSparseProcessorConfig` |
| 源码 | [float_sparse.py](../../../../../msmodelslim/processor/sparse/float_sparse.py) |

## 2. 参数列表

<h3 id="2-1-float-sparse">2.1 FloatSparseProcessorConfig</h3>

| 字段路径 | 类型 | 必选/可选 | 默认值 | 取值范围或格式 | 含义 | 引用配置 |
|----------|------|-----------|--------|----------------|------|----------|
| `type` | `string` | 可选 | `float_sparse` | `float_sparse` | 处理器类型，固定为 `float_sparse`。 | 无 |
| `sparse_ratio` | `float` | 可选 | `0.3` | — | 稀疏比例（0~1），置零权重占比，越大稀疏越多。 | 无 |
| `include` | `list[string]` | 可选 | `[]` | — | 包含的模块名称模式 | 无 |
| `exclude` | `list[string]` | 可选 | `[]` | — | 排除的模块名称模式，优先级高于 `include` | 无 |

**配置约束**

- 无。

## 3. 完整配置参考

```yaml
apiversion: modelslim_v1
spec:
  process:
  - type: float_sparse
    sparse_ratio: 0.3
    include: []
    exclude: []
```
