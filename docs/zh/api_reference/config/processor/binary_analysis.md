<!-- generated-by: skills/docs-management/scripts/gen_quant_config_docs.py ; class: msmodelslim.processor.analysis.binary_operator.processor.BinaryAnalysisProcessorConfig -->
# binary_analysis 配置说明

## 1. 配置概述

二值（有/无量化）敏感性分析处理器配置。

| 项目 | 内容 |
|------|------|
| 配置类 | `BinaryAnalysisProcessorConfig` |
| 源码 | [processor.py](../../../../../msmodelslim/processor/analysis/binary_operator/processor.py) |

## 2. 参数列表

<h3 id="2-1-binary-analysis">2.1 BinaryAnalysisProcessorConfig</h3>

| 字段路径 | 类型 | 必选/可选 | 默认值 | 取值范围或格式 | 含义 | 引用配置 |
|----------|------|-----------|--------|----------------|------|----------|
| `type` | `string` | 可选 | `binary_analysis` | `binary_analysis` | 处理器类型，固定为 `binary_analysis`。 | 无 |
| `metrics` | `string` | 可选 | `mse` | — | 分析指标：`mse`（均方误差） | 无 |
| `patterns` | `list[string]` | 可选 | `['*']` | — | 待分析的层名模式列表，默认 `*` 匹配全部 | 无 |
| `configs` | `list[object]` | 可选 | `[]` | — | 用于执行量化-反量化路径的量化子处理器配置列表 | 本页 <a href="#2-2-autoprocessorconfig">§2.2</a> |

**配置约束**

- 无。

<h3 id="2-2-autoprocessorconfig">2.2 AutoProcessorConfig</h3>

**派生类**

- `AdaptRotationProcessorConfig`（`type: adapt_rotation`） — 自适应旋转（adapt_rotation）处理器配置。 《[adapt_rotation 配置说明](adapt_rotation.md)》
- `AutoroundProcessorConfig`（`type: autoround_quant`） — autoround 量化处理器配置。 《[autoround_quant 配置说明](autoround_quant.md)》
- `AWQProcessorConfig`（`type: awq`） — AWQ（Activation-aware Weight Quantization）处理器配置。 《[awq 配置说明](awq.md)》
- `BinaryAnalysisProcessorConfig`（`type: binary_analysis`） — 二值（有/无量化）敏感性分析处理器配置。 本页 <a href="#2-1-binary-analysis">§2.1</a>
- `BinaryOperatorLayerWiseProcessorConfig`（`type: binary_operator_layer_wise`） — 逐层敏感度分析处理器配置（对比逐块浮点与量化输出）。 《[binary_operator_layer_wise 配置说明](binary_operator_layer_wise.md)》
- `BinaryOperatorModelWiseProcessorConfig`（`type: binary_operator_model_wise`） — 模型级敏感性分析配置（对比模型最终输出，使用 MSE 指标） 《[binary_operator_model_wise 配置说明](binary_operator_model_wise.md)》
- `DynamicCacheProcessorConfig`（`type: dynamic_cache`） — KV cache 量化处理器配置。 《[dynamic_cache 配置说明](dynamic_cache.md)》
- `FA3QuantProcessorConfig`（`type: fa3_quant`） — FA3（FlashAttention-3）量化处理器配置。 《[fa3_quant 配置说明](fa3_quant.md)》
- `FlatQuantProcessorConfig`（`type: flatquant`） — FlatQuant处理器配置：定义量化训练参数、策略、混合精度等 《[flatquant 配置说明](flatquant.md)》
- `FlexAWQSSZProcessorConfig`（`type: flex_awq_ssz`） — FlexAWQSSZ 平滑+AWQ 处理器配置。 《[flex_awq_ssz 配置说明](flex_awq_ssz.md)》
- `FlexSmoothQuantProcessorConfig`（`type: flex_smooth_quant`） — FlexSmoothQuant 平滑量化处理器配置。 《[flex_smooth_quant 配置说明](flex_smooth_quant.md)》
- `FloatSparseProcessorConfig`（`type: float_sparse`） — 浮点稀疏处理器配置。 《[float_sparse 配置说明](float_sparse.md)》
- `GroupProcessorConfig`（`type: group`） — 处理器合并器配置。 《[group 配置说明](group.md)》
- `IterSmoothProcessorConfig`（`type: iter_smooth`） — 迭代平滑（IterativeSmooth）处理器配置。 《[iter_smooth 配置说明](iter_smooth.md)》
- `KVSmoothProcessorConfig`（`type: kv_smooth`） — KV cache 平滑处理器配置。 《[kv_smooth 配置说明](kv_smooth.md)》
- `LinearProcessorConfig`（`type: linear_quant`） — 线性层（Linear）量化处理器配置。 《[linear_quant 配置说明](linear_quant.md)》
- `LoadProcessorConfig`（`type: load`） — 模块加载/卸载处理器配置。 《[load 配置说明](load.md)》
- `OASQProcessorConfig`（`type: oasq`） — OASQ（Outlier-Aware Smooth Quantization）处理器配置。 《[oasq 配置说明](oasq.md)》
- `OnlineQuaRotProcessorConfig`（`type: online_quarot`） — 在线 QuaRot 旋转处理器配置。 《[online_quarot 配置说明](online_quarot.md)》
- `QuaRotProcessorConfig`（`type: quarot`） — QuaRot（离线旋转）处理器配置。 《[quarot 配置说明](quarot.md)》
- `QuantSaveProcessorConfig`（`type: saver`） — 统一保存处理器配置。 《[saver 配置说明](saver.md)》
- `SmoothQuantProcessorConfig`（`type: smooth_quant`） — SmoothQuant 平滑量化处理器配置。 《[smooth_quant 配置说明](smooth_quant.md)》
- `SVDResidualProcessorConfig`（`type: svd_res`） — SVD 残差（低秩补偿）处理器配置。 《[svd_res 配置说明](svd_res.md)》
- `TrainableLinearQuantProcessorConfig`（`type: trainable_linear_quant`） — 可训练线性量化（TLQ）处理器配置。 《[trainable_linear_quant 配置说明](trainable_linear_quant.md)》
- `UnaryAnalysisProcessorConfig`（`type: unary_analysis`） — 一元（无量化）敏感性分析处理器配置。 《[unary_analysis 配置说明](unary_analysis.md)》

## 3. 完整配置参考

```yaml
apiversion: modelslim_v1
spec:
  process:
  - type: binary_analysis
    metrics: mse
    patterns:
    - '*'
    configs: []
```
