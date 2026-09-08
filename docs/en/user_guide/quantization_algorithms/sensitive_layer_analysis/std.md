# Std: Sensitive Layer Analysis Algorithm

<!-- md-trans-meta sourceCommit=0dcb0df3ef37737beef6b9a2fa93197f635a1366 translatedAt=2026-08-19T08:17:19.887Z pushedAt=2026-08-19T08:17:52.387Z -->

## Overview

- **Overview**: The std metric is used for **linear** scope analysis in `msmodelslim analyze`: it collects statistics on the activations of linear layers (and supported convolutional layers), uses the ratio of the numerical range to the standard deviation as the sensitivity score, and ranks layers at the **linear layer granularity**.

- **Core idea**: Quantization error is related to the dynamic range and the degree of dispersion. When the activation standard deviation is large, the relative perturbation under the same dynamic range is smaller. Therefore, a score in the form of `max(|max|,|min|)/std` is used to characterize how sensitive a layer is to quantization.

## Before You Begin

Install the msModelSlim tool. For details, see [msModelSlim Tool Installation Guide](../../../install_guide/install_guide.md).

## Principle

1. **Statistics**: During the calibration forward pass, the global maximum/minimum values and the standard deviation (after data-center shifting) of the target layer activations are collected.

2. **Score**:

   - $\text{abs\_max} = \max(|\text{max\_value}|, |\text{min\_value}|)$

   - $\text{score} = \text{abs\_max} / \text{std}$ (the implementation includes protective handling for cases such as $\text{std}=0$)

3. **Interpretation**: A larger score indicates that the layer is more sensitive to quantization, reflecting that its activations have a larger dynamic range/fluctuation ratio. The specific threshold must be determined based on the model and business accuracy requirements.

## Applicable Requirements

- **Recommended scenario**: Coarse screening of sensitive layers before conventional quantization.

- **Computational characteristics**: Relatively lightweight implementation with fast execution.

- **Model adaptation**: No additional analysis interface needs to be implemented in the model adapter; for the supported range of `model_type`, see [Foundation Model Support Matrix](../../model_support/foundation_model_support_matrix.md).

## Feature Description

### Command Line Example

```bash
msmodelslim analyze linear \
    --model_type Qwen2.5-7B-Instruct \
    --model_path ${model_path} \
    --metrics std \
    --calib_dataset ${calib_dataset} \
    --pattern "*.down_proj*" "*.o_proj*" \
    --topk 15 \
    --device npu
```

### Command-line Parameter Description

| Name | Description |
|------|------|
| `linear` | Linear layer sensitivity analysis |
| `--metrics` | Analysis algorithm. When set to `std`, this algorithm is used |
| `--pattern` | Layer name wildcard pattern for filtering the linear layers to be analyzed |

For complete parameters, see [Parameter Description in the Sensitive Layer Analysis Tool Usage Guide](../../feature_guide/sensitive_layer_analysis/usage.md#34-parameter-description).

## FAQ

### How to Choose Among Other Linear Metrics (quantile, kurtosis)?

**Symptom**: You are unsure how to choose among `std`, `quantile`, and `kurtosis`, or you want to understand the focus of each metric.

**Solution**: All three target the activation distribution of linear layers. `std` focuses on the ratio of range to dispersion. If the data contains many outliers, you can use it together with `quantile`. If you care about peaks and tails, you can use it together with `kurtosis`. You can read the corresponding algorithm descriptions and then choose based on your scenario.
