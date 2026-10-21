<!-- generated-by: skills/docs-management/scripts/gen_quant_config_docs.py ; class: msmodelslim.processor.analysis.binary_operator.processor.BinaryAnalysisProcessorConfig -->
# binary_analysis Configuration Reference

## 1. Configuration Overview

Binary (quantized/non-quantized) sensitivity analysis processor configuration.

| Item | Content |
|------|------|
| Config Class | `BinaryAnalysisProcessorConfig` |
| Source | [processor.py](../../../../../msmodelslim/processor/analysis/binary_operator/processor.py) |

## 2. Parameter List

<h3 id="2-1-binary-analysis">2.1 BinaryAnalysisProcessorConfig</h3>

| Field Path | Type | Required/Optional | Default | Value Range or Format | Description | Referenced Config |
|----------|------|-----------|--------|----------------|------|----------|
| `type` | `string` | Optional | `binary_analysis` | `binary_analysis` | Processor type, fixed to `binary_analysis`. | None |
| `metrics` | `string` | Optional | `mse` | — | Analysis metric: `mse` (mean squared error) | None |
| `patterns` | `list[string]` | Optional | `['*']` | — | List of layer name patterns to analyze; the default `*` matches all | None |
| `configs` | `list[object]` | Optional | `[]` | — | List of quantization sub-processor configurations used to run the quantize-dequantize path | This page <a href="#2-2-autoprocessorconfig">§2.2</a> |

**Configuration Constraints**

- None.

<h3 id="2-2-autoprocessorconfig">2.2 <a href="linear_quant.md">AutoProcessorConfig</a></h3>

**Derived Classes**

| Config Class | `type` | Description | Document |
|--------|----------|------|------|
| `AdaptRotationProcessorConfig` | `adapt_rotation` | Adaptive rotation (adapt_rotation) processor configuration. | [adapt_rotation Configuration Reference](adapt_rotation.md) |
| `AutoroundProcessorConfig` | `autoround_quant` | AutoRound quantization processor configuration. | [autoround_quant Configuration Reference](autoround_quant.md) |
| `AWQProcessorConfig` | `awq` | AWQ (Activation-aware Weight Quantization) processor configuration. | [awq Configuration Reference](awq.md) |
| `BinaryAnalysisProcessorConfig` | `binary_analysis` | Binary (quantized/non-quantized) sensitivity analysis processor configuration. | This page <a href="#2-1-binary-analysis">§2.1</a> |
| `BinaryOperatorLayerWiseProcessorConfig` | `binary_operator_layer_wise` | Layer-wise sensitivity analysis processor configuration (compares per-block floating-point and quantized outputs). | [binary_operator_layer_wise Configuration Reference](binary_operator_layer_wise.md) |
| `BinaryOperatorModelWiseProcessorConfig` | `binary_operator_model_wise` | Model-level sensitivity analysis configuration (compares final model outputs, using the MSE metric) | [binary_operator_model_wise Configuration Reference](binary_operator_model_wise.md) |
| `DynamicCacheProcessorConfig` | `dynamic_cache` | KV cache quantization processor configuration. | [dynamic_cache Configuration Reference](dynamic_cache.md) |
| `FA3QuantProcessorConfig` | `fa3_quant` | FA3 (FlashAttention-3) quantization processor configuration. | [fa3_quant Configuration Reference](fa3_quant.md) |
| `FlatQuantProcessorConfig` | `flatquant` | FlatQuant processor configuration: defines quantization training parameters, strategy, mixed precision, and more | [flatquant Configuration Reference](flatquant.md) |
| `FlexAWQSSZProcessorConfig` | `flex_awq_ssz` | Flex AWQ SSZ outlier suppression processor configuration. | [flex_awq_ssz Configuration Reference](flex_awq_ssz.md) |
| `FlexSmoothQuantProcessorConfig` | `flex_smooth_quant` | FlexSmoothQuant smooth quantization processor configuration. | [flex_smooth_quant Configuration Reference](flex_smooth_quant.md) |
| `FloatSparseProcessorConfig` | `float_sparse` | Floating-point sparsity processor configuration. | [float_sparse Configuration Reference](float_sparse.md) |
| `GroupProcessorConfig` | `group` | Processor combiner configuration. | [group Configuration Reference](group.md) |
| `IterSmoothProcessorConfig` | `iter_smooth` | Iterative smooth (IterativeSmooth) processor configuration. | [iter_smooth Configuration Reference](iter_smooth.md) |
| `KVSmoothProcessorConfig` | `kv_smooth` | KV cache smoothing processor configuration. | [kv_smooth Configuration Reference](kv_smooth.md) |
| `LinearProcessorConfig` | `linear_quant` | Linear layer quantization processor configuration. | [linear_quant Configuration Reference](linear_quant.md) |
| `LoadProcessorConfig` | `load` | Module load/unload processor configuration. | [load Configuration Reference](load.md) |
| `OASQProcessorConfig` | `oasq` | OASQ (Outlier-Aware Smooth Quantization) processor configuration. | [oasq Configuration Reference](oasq.md) |
| `OnlineQuaRotProcessorConfig` | `online_quarot` | Online QuaRot rotation processor configuration. | [online_quarot Configuration Reference](online_quarot.md) |
| `QuaRotProcessorConfig` | `quarot` | QuaRot (offline rotation) processor configuration. | [quarot Configuration Reference](quarot.md) |
| `QuantSaveProcessorConfig` | `saver` | Unified save processor configuration. | [saver Configuration Reference](saver.md) |
| `SmoothQuantProcessorConfig` | `smooth_quant` | SmoothQuant smooth quantization processor configuration. | [smooth_quant Configuration Reference](smooth_quant.md) |
| `SVDResidualProcessorConfig` | `svd_res` | SVD residual (low-rank compensation) processor configuration. | [svd_res Configuration Reference](svd_res.md) |
| `TrainableLinearQuantProcessorConfig` | `trainable_linear_quant` | Trainable linear quantization (TLQ) processor configuration. | [trainable_linear_quant Configuration Reference](trainable_linear_quant.md) |
| `UnaryAnalysisProcessorConfig` | `unary_analysis` | Unary (non-quantized) sensitivity analysis processor configuration. | [unary_analysis Configuration Reference](unary_analysis.md) |

## 3. Complete Configuration Reference

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
