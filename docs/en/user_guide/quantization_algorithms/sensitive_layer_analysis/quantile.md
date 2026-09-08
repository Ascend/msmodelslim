# Quantile: Sensitive Layer Analysis Algorithm

<!-- md-trans-meta sourceCommit=0dcb0df3ef37737beef6b9a2fa93197f635a1366 translatedAt=2026-08-19T08:17:13.978Z pushedAt=2026-08-19T08:17:52.381Z -->

## Introduction

- **Overview**: The Quantile metric is used for **linear** range analysis: it constructs a score based on activation quantiles and the interquartile range (IQR), is relatively robust to outliers, and is used for sensitivity ranking at the **linear layer granularity**.

- **Core idea**: The lower quartile $Q_1$ (the 1/4 quantile) and the upper quartile $Q_3$ (the 3/4 quantile) are used to describe the main body width of the activation distribution, and a score is constructed by combining them with the absolute magnitude. A larger interquartile range $\text{IQR} = Q_3 - Q_1$ indicates a more dispersed main body distribution, which is relatively less sensitive to quantization under the same dynamic range.

## Prerequisites

Install the msModelSlim tool. For details, see [msModelSlim Tool Installation Guide](../../../install_guide/install_guide.md).

## Principle

1. Compute the first and third quartiles $Q_1$ and $Q_3$ of the activations, along with the statistic used for the magnitude term.

2. Compute the score: `score = 2 × max(|max_value|, |min_value|) / 254 / (Q3 - Q1)`, where the numerator `max(|max_value|, |min_value|)` is the maximum absolute value of the activations, and the denominator `Q3 - Q1` is the interquartile range (IQR).

3. **Interpretation**: A larger score indicates that the layer is more sensitive to quantization. Specifically:

   - The larger the maximum absolute value of the activations, the larger the quantization step size relative to the range, the more significant the quantization error, and the larger the score;

   - The larger the IQR, the more dispersed the main distribution of activations; under the same dynamic range, each quantization interval has more representative values, resulting in a smaller relative error and a smaller score.

## Applicable Requirements

- **Recommended scenarios**: The activation distribution has a heavy tail, and you want to reduce the dominant influence of outliers on the per-layer score to make the sensitivity ranking more robust.

- **Model adaptation**: No additional analysis interface needs to be implemented by the model adapter. For the supported range of `model_type`, see [Foundation Model Support Matrix](../../model_support/foundation_model_support_matrix.md).

## Feature Description

### Command Line Example

```bash
msmodelslim analyze linear \
    --model_type Qwen2.5-7B-Instruct \
    --model_path ${model_path} \
    --metrics quantile \
    --calib_dataset ${calib_dataset} \
    --pattern "*.down_proj*" "*.o_proj*" \
    --topk 15 \
    --device npu
```

### Command-line Parameter Description

| Name | Description |
|------|------|
| `linear` | Linear layer sensitivity analysis |
| `--metrics` | Analysis algorithm. When the value is `quantile`, this algorithm is used |
| `--pattern` | Layer name wildcard used to filter the linear layers to be analyzed |

For complete parameters, see [Parameter Description in the Sensitive Layer Analysis Tool Usage Guide](../../feature_guide/sensitive_layer_analysis/usage.md#34-parameter-description).
