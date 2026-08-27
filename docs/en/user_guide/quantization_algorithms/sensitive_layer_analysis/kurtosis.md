# Kurtosis: Sensitive Layer Analysis Algorithm

<!-- md-trans-meta sourceCommit=0dcb0df3ef37737beef6b9a2fa93197f635a1366 translatedAt=2026-08-19T08:16:52.468Z pushedAt=2026-08-19T08:17:52.374Z -->

## Introduction

- **Overview**: The Kurtosis metric is used for **linear** range analysis: after sampling activations, kurtosis is estimated to identify the impact of distribution peaks and tail extreme values on quantization, and a **linear-layer granularity** ranking is output.

- **Core idea**: Excess kurtosis characterizes the peakedness relative to a normal distribution; the more concentrated the distribution and the more prominent the extreme values, the higher the relative risk introduced by quantization truncation tends to be.

## Prerequisites

Install the msModelSlim tool. For details, see the [msModelSlim Tool Installation Guide](../../../install_guide/install_guide.md).

## Principle

1. Sort and step-sample the layer activations (to control memory and computation), and estimate the kurtosis on the sampled sequence.

2. The common form of excess kurtosis: $\text{kurtosis} = \mathbb{E}[(X-\mu)^4]/\sigma^4 - 3$.

3. **Interpretation**: A larger kurtosis generally indicates a more peaked distribution that is more sensitive to extreme values; a value close to 0 indicates a form closer to Gaussian (in the implementation, the specific `compute_score` output is used for inter-layer ranking).

## Applicable Requirements

- **Recommended scenarios**: When it is necessary to identify quantization-sensitive layers based on the peakedness of the activation distribution, so as to assist fallback or mixed-precision decisions.

- **Model adaptation**: No additional analysis interface needs to be implemented by the model adapter. For the supported range of `model_type`, see [Foundation Model Support Matrix](../../model_support/foundation_model_support_matrix.md).

## Feature Description

### Command Line Example

```bash
msmodelslim analyze linear \
    --model_type Qwen2.5-7B-Instruct \
    --model_path ${model_path} \
    --metrics kurtosis \
    --calib_dataset ${calib_dataset} \
    --pattern "*.down_proj*" "*.o_proj*" \
    --topk 15 \
    --device npu
```

### Command-line Parameter Description

| Name | Description |
|------|------|
| `linear` | Linear layer sensitivity analysis |
| `--metrics` | Analysis algorithm. When set to `kurtosis`, this algorithm is used (also the default metrics for `linear`) |
| `--pattern` | Layer name wildcard used to filter the linear layers to be analyzed |

For the complete parameters, see [Parameter Description in the Sensitive Layer Analysis Tool Usage Guide](../../feature_guide/sensitive_layer_analysis/usage.md#34-parameter-description).
