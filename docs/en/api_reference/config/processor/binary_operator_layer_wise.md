<!-- generated-by: skills/docs-management/scripts/gen_quant_config_docs.py ; class: msmodelslim.processor.analysis.binary_operator_layer_wise.processor.BinaryOperatorLayerWiseProcessorConfig -->
# binary_operator_layer_wise Configuration Reference

## 1. Configuration Overview

Layer-wise sensitivity analysis processor configuration (compares per-block floating-point and quantized outputs).

| Item | Content |
|------|------|
| Config Class | `BinaryOperatorLayerWiseProcessorConfig` |
| Source | [processor.py](../../../../../msmodelslim/processor/analysis/binary_operator_layer_wise/processor.py) |

## 2. Parameter List

<h3 id="2-1-binary-operator-layer-wise">2.1 BinaryOperatorLayerWiseProcessorConfig</h3>

| Field Path | Type | Required/Optional | Default | Value Range or Format | Description | Referenced Config |
|----------|------|-----------|--------|----------------|------|----------|
| `type` | `string` | Optional | `binary_operator_layer_wise` | `binary_operator_layer_wise` | Processor type, fixed to `binary_operator_layer_wise`. | None |
| `metrics` | `string` | Optional | `mse_layer_wise` | — | Analysis metric: `mse_layer_wise`. | None |
| `quant_modules` | `list[string]` | Optional | `['*']` | — | Consistent with linear_quant.include and the CLI `--quant_modules` (YAML placeholder ${quant_modules}); used as the suffix of layer sensitivity result display names, such as model.layers.2 (mod1, mod2). | None |
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
| `BinaryAnalysisProcessorConfig` | `binary_analysis` | Binary (quantized/non-quantized) sensitivity analysis processor configuration. | [binary_analysis Configuration Reference](binary_analysis.md) |
| `BinaryOperatorLayerWiseProcessorConfig` | `binary_operator_layer_wise` | Layer-wise sensitivity analysis processor configuration (compares per-block floating-point and quantized outputs). | This page <a href="#2-1-binary-operator-layer-wise">§2.1</a> |
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
  - type: binary_operator_layer_wise
    metrics: mse_layer_wise
    quant_modules:
    - '*'
    configs: []
```
