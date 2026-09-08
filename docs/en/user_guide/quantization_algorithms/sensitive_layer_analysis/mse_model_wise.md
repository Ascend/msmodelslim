# Model-wise MSE (mse_model_wise): Sensitive Layer Analysis Algorithm

<!-- md-trans-meta sourceCommit=3909078a95b1f3d532dd72a3a08143f4d01c1435 translatedAt=2026-08-19T08:17:00.368Z pushedAt=2026-08-19T08:17:52.379Z -->

## Introduction

- **Overview**: `mse_model_wise` is used for **layer**-scope analysis: it compares, layer by layer, the **final model output** (usually the output of the last Decoder layer) before and after quantizing only the substructures related to that layer, computes the MSE (Mean Squared Error), and obtains a layer sensitivity ranking for whole-layer or whole-block fallback.

- **Core idea**: The final output error characterizes the cumulative impact of quantizing that layer on end-to-end behavior; it relies on layer-by-layer chained forward propagation, using the output of the previous layer as the input of the next layer to simulate the real inference path.

## Prerequisites

Install the msModelSlim tool. For details, see [*msModelSlim Tool Installation Guide*](../../../install_guide/install_guide.md).

## Principle

1. For the same batch of calibration data, evaluate each Decoder layer in sequence during chained forward: compare the structures selected by `quant_modules` within that layer before and after quantization, and collect the **final model output**.

2. Use the MSE of the final output as the sensitivity score of that layer; a larger score indicates that the layer is more sensitive to quantization (that is, quantizing that layer has a greater impact on the final model output).

3. **Chained constraint**: If inter-layer tensor shapes or semantics cannot be aligned at a certain layer, the implementation skips subsequent layers starting from that layer and prints a warning, and the ranking includes only the layers that completed successfully (commonly seen in structures such as MTP).

## Applicable Requirements

- **Recommended scenarios**: You want to evaluate the quantization sensitivity of each layer from the perspective of the **final model output**, to assist in fallback decisions for an entire layer or an entire block.

- **Resources and data**: The number of calibration batches and the sequence length significantly increase the number of forward passes and intermediate caches, which may cause **OOM** on large models. It is recommended to control the calibration scale.

- **Model adaptation**: No additional analysis interface is required. The supported range of `model_type` is consistent with ModelslimV1 quantization. For details, see [Large Model Support Matrix](../../model_support/foundation_model_support_matrix.md). Some architectures are subject to chained forward alignment restrictions. For details, see [FAQs](#faqs) below.

## Function Description

### Command Line Example

```bash
msmodelslim analyze layer \
    --model_type Qwen3-32B \
    --model_path ${model_path} \
    --metrics mse_model_wise \
    --quant_modules "*mlp*" \
    --calib_dataset ${calib_dataset} \
    --topk 15 \
    --device npu
```

### Command-line Parameter Description

| Name | Description |
|------|------|
| `layer` | Decoder block-level sensitivity analysis |
| `--metrics` | Analysis algorithm. The algorithm is used when the value is `mse_model_wise`. |
| `--quant_modules` | Wildcard list that specifies the module scope participating in the quantization comparison. |

For complete parameters, see [Sensitive Layer Analysis Tool Usage Guide Parameter Description](../../feature_guide/sensitive_layer_analysis/usage.md#34-parameter-description).

## FAQs

### Why Do Warnings Appear Mid-Analysis, and Why Are Subsequent Layers Missing from the Results?

**Symptom**: A warning appears in the log, and subsequent Decoder layers do not appear in the analysis results.

**Solution**: This is mostly because the chained forward cannot align the input and output at a certain layer. You can check the log to locate the layer number, and preferentially fall back on that layer or a special structure (such as MTP), or use `mse_layer_wise` for block-level evaluation instead.

### What Are the Main Differences from `mse_layer_wise`?

**Symptom**: You need to choose between `mse_layer_wise` and `mse_model_wise`, but the difference in their measurement perspectives is unclear.

**Solution**: `mse_layer_wise` focuses on the local reconstruction error on the **output of a single block**, reflecting the quantization impact of that block itself; `mse_model_wise` focuses on the global accumulated error on the **final model output**, which is closer to the end-to-end effect but imposes higher computation and alignment requirements.
