# Kurtosis Sensitive Layer Analysis Algorithm Terminology Entry

> **Entry Category**: [Sensitive Layer Analysis Algorithms](../README.md#3-sensitive-layer-analysis-algorithms)<br>
> **English Name**: kurtosis<br>
> **Application Domain**: Quantization sensitive layer analysis, model quantization accuracy tuning<br>

---

## 1. Overview

Kurtosis is a sensitive layer analysis algorithm based on higher-order statistics of activations. It uses excess kurtosis to measure how spiky and heavy-tailed a distribution is, helping identify linear layers that are more easily dragged into a wide quantization range by a few extreme values. Its core characteristics are a focus on distribution shape and relatively low computational overhead, and it can be cross-checked against statistical metrics such as [Std](../std/term_std.md) and [Quantile](../quantile/term_quantile.md).

---

## 2. Term Description

Quantization clipping is sensitive to spiky distributions and extreme values. Kurtosis observes that the more concentrated the distribution and the more prominent the extreme values, the higher the relative risk introduced by quantization clipping; estimating the kurtosis of the activations identifies such layers and supports fallback or mixed-precision decisions.

### 2.1 Core Idea

The core idea of Kurtosis is to "characterize how spiky a distribution is through kurtosis": sort and stride-sample the layer activations (to bound memory and computation), estimate the kurtosis on the sampled sequence, and use excess kurtosis to describe how much spikier the distribution is than a normal one. A larger kurtosis usually means a spikier distribution that is more sensitive to extreme values.

High kurtosis means that most of the codes in the quantization interval may be "pulled away" by a few far-tail samples: taking MinMax as an example, the quantization scale is determined by the extremes, while most samples near the center occupy only a narrow interval, reducing the effective resolution.

### 2.2 Working Mechanism

Kurtosis measures the "tail heaviness / spikiness" of the activation distribution through the standardized fourth central moment. After subtracting the mean and dividing by the standard deviation, ordinary scale changes are removed; raising the standardized values to the fourth power then gives samples far from the center a weight far greater than that of ordinary samples. After subtracting 3, a normal distribution has an excess kurtosis of 0, and a positive value usually indicates heavier tails or a spikier center than a normal distribution.

In quantization, heavy-tailed distributions let a few extremes dominate range estimation such as MinMax, thereby widening the step size and reducing the effective resolution of the bulk data. Kurtosis can therefore serve as a quantization-free surrogate indicator of whether a layer is easily dragged down by outlier tails. Because the fourth power is very sensitive to extreme samples, the calibration sample coverage, sampling method, and occasional anomalies significantly affect the score, so it is usually better suited to relative ranking within the same model.

### 2.3 Mathematical Description

A common form of excess kurtosis:

$$
\text{kurtosis} = \frac{\mathbb{E}[(X - \mu)^4]}{\sigma^4} - 3
$$

- $X$: the sampled sequence of layer activations
- $\mu$: the activation mean
- $\sigma$: the activation standard deviation
- $\text{kurtosis}$: excess kurtosis (approximately 0 for a normal distribution)

The actual sensitivity score can be a monotonic mapping of the kurtosis; the key is to keep the same metric definition so that rankings across layers remain comparable.

Letting the standardized variable be $Z=(X-\mu)/\sigma$, the excess kurtosis is $\mathbb E[Z^4]-3$. The in-interval error of symmetric uniform quantization is often roughly approximated as $S^2/12$, and $S$ is in turn determined by the tail extremes. High kurtosis does not appear directly in the quantization error formula, but it signals that "the extreme-determined $S$ and the bulk standard deviation may be severely mismatched", making it an indirect risk indicator.

### 2.4 Key Properties

- **Linear-layer granularity**: statistics are collected on the input activations of linear transformations, producing a relative sensitivity ranking across layers.
- **Spike identification**: identifies the influence of distribution spikes and tail extremes on quantization.
- **Controllable memory**: controls memory and computational overhead through sorting and stride sampling.
- **A scale-independent shape metric**: after standardization, the fourth moment is insensitive to overall linear scaling but very sensitive to tail anomalies.

However, high kurtosis does not necessarily imply high quantization loss. Some heavy-tailed directions may happen not to affect the task, or may already be handled by subsequent methods such as per-channel and clipping; conversely, a low-kurtosis layer may still require high precision because of its functional sensitivity in the model.

### 2.5 Use Cases

- Scenarios that identify quantization-sensitive layers from how spiky the activation distribution is, and support fallback or mixed-precision decisions.
- Scenarios that focus on the influence of spiky distributions and tail extremes.

### 2.6 Usage Restrictions

- The specific threshold for the score must be determined in light of the model and the business accuracy requirements.
- Only for `linear`-scope analysis; not applicable to the `layer`/`attn` scopes.

---

## 3. Related Terms

- [Std](../std/term_std.md): a similar algorithm and another `linear`-scope analysis metric that emphasizes the ratio of range to dispersion.
- [Quantile](../quantile/term_quantile.md): a similar algorithm and another `linear`-scope analysis metric that is relatively robust to outliers.
- [MSE Layer Wise](../mse_layer_wise/term_mse_layer_wise.md): a companion term and a `layer` -scope analysis metric, usable for block-level evaluation.

---

## 4. Reference Documents

1. [Kurtosis Analysis Configuration Workflow Guide](./usage_kurtosis.md)
