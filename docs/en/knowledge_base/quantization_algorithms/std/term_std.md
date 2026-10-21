# Std Sensitive Layer Analysis Algorithm Terminology Entry

> **Entry Category**: [Sensitive Layer Analysis Algorithms](../README.md#3-sensitive-layer-analysis-algorithms)<br>
> **English Name**: std<br>
> **Application Domain**: Quantization sensitive layer analysis, model quantization accuracy tuning<br>

---

## 1. Overview

Std is a statistical metric that uses the relationship between the activation dynamic range and the standard deviation to perform a coarse screening of sensitive layers. It compares the span of the extremes with the scale of the bulk fluctuation to judge whether a few large values could significantly widen the quantization range and reduce effective precision. Its core characteristics are low computational overhead and intuitive interpretation; it is suitable as a first-pass screen for linear layers and can be combined with [Quantile](../quantile/term_quantile.md) and [Kurtosis](../kurtosis/term_kurtosis.md).

---

## 2. Term Description

Quantization error is related to the dynamic range and the dispersion of the activations. Std observes that when the activation standard deviation is large, the relative perturbation under the same dynamic range is smaller, so a score of the form `max(|max|,|min|)/std` can characterize how sensitive a layer is to quantization and help identify layers that need fallback or reduced quantization strength.

### 2.1 Core Idea

The core idea of Std is to "measure sensitivity by the ratio of the dynamic range to the dispersion": during the calibration forward pass, collect the global maximum/minimum of the target layer's activations and the standard deviation after centering the data, then use the ratio of the absolute maximum to the standard deviation as the sensitivity score. The larger the score, the more sensitive the layer is to quantization.

The Std metric expresses "how far the tail is from the center" as a ratio at extremely low cost, so it can quickly find layers that are easily dragged down by the global range.

### 2.2 Working Mechanism

For symmetric MinMax INT8, the quantization step size is approximately proportional to $\text{abs\_max}/127$, while the standard deviation $\sigma$ describes the typical scale of the bulk fluctuation. Thus $\text{abs\_max}/\sigma$ can be seen as the ratio of the quantization step size to the natural signal scale (ignoring the constant factor $1/127$). If the extreme is far larger than the standard deviation, it means that a few tails stretch the representable range very wide and the bulk variation occupies only a few levels on the integer grid, which is usually harder to quantize.

This ratio is approximately invariant to overall multiplicative scaling: when the activations are multiplied by a constant, abs_max and std change in proportion, so the score reflects distribution shape rather than absolute units. Compared with Kurtosis, it relies only on second-order statistics and extremes, is cheap to compute and suitable for coarse screening, but it cannot distinguish skewness, multimodality, or different tail shapes that share the same max/std.

### 2.3 Mathematical Description

Computing the statistics:

$$
\text{abs\_max} = \max(|\text{max\_value}|, |\text{min\_value}|)
$$

$$
\text{score} = \frac{\text{abs\_max}}{\text{std}}
$$

- $\text{max\_value}$, $\text{min\_value}$: the global maximum and minimum of the activations
- $\text{abs\_max}$: the absolute maximum
- $\text{std}$: the standard deviation after centering the data
- $\text{score}$: sensitivity score

When $\text{std}=0$, the ratio loses its discriminative power, so the actual computation must avoid division by zero and treat this as a degenerate distribution.

If $a=\max|X|$ and the symmetric INT8 step size is $S=a/127$, then:

$$
\frac{a}{\sigma}=127\frac{S}{\sigma}.
$$

Under the high-resolution approximation, the standard deviation of the quantization noise within the interval is about $S/\sqrt{12}$, so:

$$
\frac{\sigma_q}{\sigma_x}\approx\frac{1}{127\sqrt{12}}\frac{a}{\sigma}.
$$

This gives an intuitive link between the score and the "quantization noise relative to the signal standard deviation", but the approximation ignores clipping, non-uniform distributions, and low-bit-width effects, so it cannot be used as an exact error prediction.

### 2.4 Key Properties

- **Linear-layer granularity**: statistics are collected on the input activations of linear transformations, producing a relative sensitivity ranking across layers.
- **Lightweight and efficient**: only max/min and std must be collected, so the implementation is relatively lightweight and fast to run.
- **Coarse screening**: suitable as a coarse screen for sensitive layers before conventional quantization.
- **Approximately scale-invariant**: overall multiplicative scaling changes both abs-max and std, so the ratio mainly reflects distribution shape.

In terms of error and applicability boundaries, it remains a heuristic proxy: distributions with the same abs_max/std may have completely different skewness, kurtosis, and multimodal structure, and their quantization errors will also differ; if the standard deviation is itself inflated by a broad bulk, the score may underestimate tail risk.

### 2.5 Use Cases

- Coarse screening of sensitive layers before conventional quantization.
- Scenarios that require a quick sensitivity ranking of linear layers to support quantization-layer fallback and mixed-precision decisions.

### 2.6 Usage Restrictions

- The specific threshold for the score must be determined in light of the model and the business accuracy requirements.
- Only for `linear`-scope analysis; not applicable to the `layer`/`attn` scopes.

---

## 3. Related Terms

- [Quantile](../quantile/term_quantile.md): a similar algorithm and another `linear`-scope analysis metric that is relatively robust to outliers.
- [Kurtosis](../kurtosis/term_kurtosis.md): a similar algorithm and another `linear`-scope analysis metric that focuses on spiky distributions.
- [MSE Layer Wise](../mse_layer_wise/term_mse_layer_wise.md): a companion term and a `layer`-scope analysis metric, usable for block-level evaluation.

---

## 4. Reference Documents

1. [Std Analysis Configuration Workflow Guide](./usage_std.md)
