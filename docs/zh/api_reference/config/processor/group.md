<!-- generated-by: skills/docs-management/scripts/gen_quant_config_docs.py ; class: msmodelslim.processor.container.group.GroupProcessorConfig -->
# group 配置说明

## 1. 配置概述

处理器合并器配置。

| 项目 | 内容 |
|------|------|
| 配置类 | `GroupProcessorConfig` |
| 源码 | [group.py](../../../../../msmodelslim/processor/container/group.py) |

## 2. 参数列表

<h3 id="2-1-group">2.1 GroupProcessorConfig</h3>

| 字段路径 | 类型 | 必选/可选 | 默认值 | 取值范围或格式 | 含义 | 引用配置 |
|----------|------|-----------|--------|----------------|------|----------|
| `type` | `string` | 必选 | 无 | `group` | 处理器类型，固定为 `group`。 | 无 |
| `configs` | `list[object]` | 必选 | 无 | — | 被合并的处理器配置列表，按顺序执行；每个元素是 `type` 分派的处理器配置。 | 本页 <a href="#2-2-autoprocessorconfig">§2.2</a> |

**配置约束**

- `configs` 内各子处理器必须为同一 `type`（同类合并，如均为 `linear_quant`）；不同 `type` 请拆成多个独立的 `process` 项，不要混在同一 `group` 中。

<h3 id="2-2-autoprocessorconfig">2.2 《<a href="linear_quant.md">AutoProcessorConfig</a>》</h3>

**派生类**

| 配置类 | `type` | 说明 | 文档 |
|--------|----------|------|------|
| `AdaptRotationProcessorConfig` | `adapt_rotation` | 自适应旋转（adapt_rotation）处理器配置。 | 《[adapt_rotation 配置说明](adapt_rotation.md)》 |
| `AutoroundProcessorConfig` | `autoround_quant` | autoround 量化处理器配置。 | 《[autoround_quant 配置说明](autoround_quant.md)》 |
| `AWQProcessorConfig` | `awq` | AWQ（Activation-aware Weight Quantization）处理器配置。 | 《[awq 配置说明](awq.md)》 |
| `BinaryAnalysisProcessorConfig` | `binary_analysis` | 二值（有/无量化）敏感度分析处理器配置。 | 《[binary_analysis 配置说明](binary_analysis.md)》 |
| `BinaryOperatorLayerWiseProcessorConfig` | `binary_operator_layer_wise` | 逐层敏感度分析处理器配置（对比逐块浮点与量化输出）。 | 《[binary_operator_layer_wise 配置说明](binary_operator_layer_wise.md)》 |
| `BinaryOperatorModelWiseProcessorConfig` | `binary_operator_model_wise` | 模型级敏感度分析配置（对比模型最终输出，使用 MSE 指标） | 《[binary_operator_model_wise 配置说明](binary_operator_model_wise.md)》 |
| `DynamicCacheProcessorConfig` | `dynamic_cache` | KV cache 量化处理器配置。 | 《[dynamic_cache 配置说明](dynamic_cache.md)》 |
| `FA3QuantProcessorConfig` | `fa3_quant` | FA3（FlashAttention-3）量化处理器配置。 | 《[fa3_quant 配置说明](fa3_quant.md)》 |
| `FlatQuantProcessorConfig` | `flatquant` | FlatQuant处理器配置：定义量化训练参数、策略、混合精度等 | 《[flatquant 配置说明](flatquant.md)》 |
| `FlexAWQSSZProcessorConfig` | `flex_awq_ssz` | Flex AWQ SSZ 离群值抑制处理器配置。 | 《[flex_awq_ssz 配置说明](flex_awq_ssz.md)》 |
| `FlexSmoothQuantProcessorConfig` | `flex_smooth_quant` | FlexSmoothQuant 平滑量化处理器配置。 | 《[flex_smooth_quant 配置说明](flex_smooth_quant.md)》 |
| `FloatSparseProcessorConfig` | `float_sparse` | 浮点稀疏处理器配置。 | 《[float_sparse 配置说明](float_sparse.md)》 |
| `GroupProcessorConfig` | `group` | 处理器合并器配置。 | 本页 <a href="#2-1-group">§2.1</a> |
| `IterSmoothProcessorConfig` | `iter_smooth` | 迭代平滑（IterativeSmooth）处理器配置。 | 《[iter_smooth 配置说明](iter_smooth.md)》 |
| `KVSmoothProcessorConfig` | `kv_smooth` | KV cache 平滑处理器配置。 | 《[kv_smooth 配置说明](kv_smooth.md)》 |
| `LinearProcessorConfig` | `linear_quant` | 线性层（Linear）量化处理器配置。 | 《[linear_quant 配置说明](linear_quant.md)》 |
| `LoadProcessorConfig` | `load` | 模块加载/卸载处理器配置。 | 《[load 配置说明](load.md)》 |
| `OASQProcessorConfig` | `oasq` | OASQ（Outlier-Aware Smooth Quantization）处理器配置。 | 《[oasq 配置说明](oasq.md)》 |
| `OnlineQuaRotProcessorConfig` | `online_quarot` | 在线 QuaRot 旋转处理器配置。 | 《[online_quarot 配置说明](online_quarot.md)》 |
| `QuaRotProcessorConfig` | `quarot` | QuaRot（离线旋转）处理器配置。 | 《[quarot 配置说明](quarot.md)》 |
| `QuantSaveProcessorConfig` | `saver` | 统一保存处理器配置。 | 《[saver 配置说明](saver.md)》 |
| `SmoothQuantProcessorConfig` | `smooth_quant` | SmoothQuant 平滑量化处理器配置。 | 《[smooth_quant 配置说明](smooth_quant.md)》 |
| `SVDResidualProcessorConfig` | `svd_res` | SVD 残差（低秩补偿）处理器配置。 | 《[svd_res 配置说明](svd_res.md)》 |
| `TrainableLinearQuantProcessorConfig` | `trainable_linear_quant` | 可训练线性量化（TLQ）处理器配置。 | 《[trainable_linear_quant 配置说明](trainable_linear_quant.md)》 |
| `UnaryAnalysisProcessorConfig` | `unary_analysis` | 一元（无量化）敏感度分析处理器配置。 | 《[unary_analysis 配置说明](unary_analysis.md)》 |

## 3. 完整配置参考

```yaml
apiversion: modelslim_v1
spec:
  process:
  - type: group
    configs:
    - type: linear_quant
      qconfig:
        act:
          dtype: int8
          scope: per_tensor
          symmetric: false
          method: minmax
        weight:
          dtype: int8
          scope: per_channel
          symmetric: true
          method: minmax
      include:
      - '*self_attn*'
    - type: linear_quant
      qconfig:
        act:
          dtype: int8
          scope: per_token
          symmetric: true
          method: minmax
        weight:
          dtype: int8
          scope: per_channel
          symmetric: true
          method: minmax
      include:
      - '*mlp*'
      exclude:
      - '*gate*'
```
