# Layer-wise MSE (mse_layer_wise): Sensitive Layer Analysis Algorithm

<!-- md-trans-meta sourceCommit=3909078a95b1f3d532dd72a3a08143f4d01c1435 translatedAt=2026-08-19T08:16:55.378Z pushedAt=2026-08-19T08:17:52.377Z -->

## Introduction

- **Overview**: `mse_layer_wise` is used for **layer**-scope analysis: within each **Decoder block**, the submodules selected by `quant_modules` (mainly the Linear layers within the block in implementation) are respectively subjected to floating-point and quantized forward passes, and the Mean Squared Error (MSE) is computed on the submodule outputs and **averaged within the block** to obtain a sensitivity score. The results are output as a **Decoder block-granularity** ranking, which guides the fallback of an entire layer or an entire block (such as a whole attention/MLP segment).

- **Core idea**: The larger the aggregated quantization reconstruction error within a block, the more sensitive the block is under the current quantization subset.

## Before You Begin

Install the msModelSlim tool. For details, see [msModelSlim Tool Installation Guide](../../../install_guide/install_guide.md).

## Principle

1. For the same batch of calibration samples, traverse by Decoder layer: within a block, collect the **floating-point** and **quantized** forward outputs of the submodules that match the configuration.

2. Compute the MSE for each alignable output; **average** all valid MSEs within the block as the block's score.

3. **Interpretation**: A larger score indicates that the Decoder block is more sensitive to quantization under the current `quant_modules` parameter configuration.

## Applicable Requirements

- **Recommended scenario**: You want to compare the sensitivity of each layer from the perspective of the **submodule outputs within a Decoder block**, so as to assist in the fallback decision for an entire layer or an entire block.

- **Model adaptation**: No additional analysis interface needs to be implemented by the model adapter. The supported range of `model_type` is consistent with ModelslimV1 quantization. For details, see [Foundation Model Support Matrix](../../model_support/foundation_model_support_matrix.md).

- **Configuration impact**: The `quant_modules` parameter determines the set of submodules that participate in the comparison quantization. Different configurations produce different rankings.

## Function Description

### Command Line Example

```bash
msmodelslim analyze layer \
    --model_type Qwen3-32B \
    --model_path ${model_path} \
    --metrics mse_layer_wise \
    --quant_modules "*mlp*" \
    --calib_dataset ${calib_dataset} \
    --topk 15 \
    --device npu
```

### Command-line Parameter Description

| Name | Description |
|------|------|
| `layer` | Decoder block-level sensitivity analysis |
| `--metrics` | Analysis algorithm. The algorithm is used when the value is `mse_layer_wise`. |
| `--quant_modules` | Wildcard list that specifies the range of modules participating in the quantization comparison. |

For complete parameters, see [Parameter Description in the Sensitive Layer Analysis Tool Usage Guide](../../feature_guide/sensitive_layer_analysis/usage.md#34-parameter-description).

## FAQ

### Is it normal for layer ordering to change after modifying `quant_modules`?

**Symptom**: After adjusting the `quant_modules` wildcard, the TopK order of each layer is inconsistent with that before the adjustment.

**Solution**: This is expected. After adjusting `quant_modules`, the set of submodules participating in the quantization comparison changes, and the aggregated MSE result within the block changes accordingly, so the relative ranking of each layer also differs. Fix a configuration according to the target quantization scheme before interpreting the results. If the same command still produces unstable results, investigate the calibration order and randomness.
