# MinMax Minimum-Maximum Quantization Algorithm Terminology Entry

> **Entry Category**: [Quantization Algorithms](../README.md#2-quantization-algorithms)<br>
> **English Name**: minmax<br>
> **Application Domain**: Deep learning model compression, large language model quantization<br>

---

## 1. Overview

MinMax is a basic quantization method that determines the quantization range from the minimum and maximum values observed in a tensor. It computes the scale and an optional zero point directly from those bounds and maps floating-point values into the target integer or low-precision numerical range. Its core characteristics are a simple implementation, low statistical overhead, and broad applicability; it is commonly used as the default baseline for [linear quantization](../linear_quant/term_linear_quant.md), but it is relatively sensitive to outliers.

---

## 2. Term Description

Quantization requires determining the mapping range from floating-point values to integers. MinMax observes that using a tensor's actual minimum/maximum values as the quantization bounds fully covers the data distribution at extremely low computational cost, making it suitable as a basic quantization method. However, it is sensitive to outliers: when extreme outliers are present in a tensor, the quantization range is widened and the precision of normal values decreases.

### 2.1 Core Idea

The core idea of MinMax is to "define the range by the extremes": collect the minimum and maximum values in the tensor as the quantization bounds, compute the scale factor and zero point from them, and linearly map floating-point values into the target integer range (such as the INT8 range of $[-128, 127]$ or the FP8 representation range).

The advantage of MinMax is that no active clipping occurs within the range, so when the data distribution is close to uniform, the tails are not pronounced, or the weights are relatively stable, it is often a strong and simple baseline.

### 2.2 Working Mechanism

MinMax first collects the minimum and maximum values at the specified sharing granularity, then linearly maps that interval to the target integer codes. Symmetric quantization usually uses the absolute extreme $a=\max(|r_{min}|,|r_{max}|)$ to construct $[-a,a]$; asymmetric quantization uses $[r_{min},r_{max}]$ directly, allowing the zero point to deviate from the integer center. As long as values fall within the observed range, MinMax does not actively clip the tails, so the method is simple, deterministic, and easy to reproduce.

For static tensors such as weights, the range can be computed once; for activations, the range comes from calibration samples and may accumulate across batches. When the statistics use a globally accumulated extreme, the range can only stay the same or grow, so a single abnormal batch can permanently raise the quantization step size. Refining the statistical granularity from per-tensor to per-channel/per-group isolates some abnormal channels but increases the number of scales.

### 2.3 Mathematical Description

First determine the range:

$$
V_{\min} = \min(X), \quad V_{\max} = \max(X)
$$

Scale factor $S$:

$$
S = \frac{V_{\max} - V_{\min}}{Q_{\max} - Q_{\min}} \quad (\text{asymmetric})
$$

$$
S = \frac{\max(|V_{\min}|, |V_{\max}|)}{Q_{\max}} \quad (\text{symmetric})
$$

Zero point $Z$:

$$
Z = Q_{\min} - \operatorname{round}\left(\frac{V_{\min}}{S}\right) \quad (\text{asymmetric})
$$

$$
Z = 0 \quad (\text{symmetric})
$$

- $X$: the tensor to quantize (weights or activation values)
- $V_{\min}$, $V_{\max}$: the minimum and maximum values of the tensor
- $S$: scale factor
- $Z$: zero-point offset
- $Q_{\min}$, $Q_{\max}$: the numerical range of the target data type (for example, $Q_{\max}=127$ for INT8 symmetric quantization)

If symmetric $b$-bit signed quantization uses the maximum positive-side code $Q=2^{b-1}-1$, then $S=a/Q$ and $a=\max|X|$. For most of the bulk values lying within $[-\sigma,\sigma]$, the effective number of discrete levels is roughly proportional to $\sigma/S=Q\sigma/a$. Therefore, when $a/\sigma$ is enlarged by a few outliers, the number of codes actually available to the bulk decreases significantly. Sensitivity metrics such as Std, Quantile, and Kurtosis can all measure this risk from different angles.

### 2.4 Key Properties

- **Simple and efficient**: only the extremes must be collected, so the computational overhead is extremely low.
- **Linear mapping**: the floating-point range is mapped linearly to the integer range, which is easy to implement in hardware.
- **Symmetric and asymmetric**: both modes are supported: symmetric ($Z=0$) and asymmetric (adjustable $Z$).
- **Sensitive to outliers**: extreme outliers widen the quantization range and reduce the precision of normal values.
- **Extreme-value sensitive**: not clipping actively means that a few abnormal values can significantly widen the quantization scale.

In terms of error and applicability boundaries: for dynamic quantization, MinMax can recompute the statistics on every input, so the influence of outliers is confined to the current sample/group; for static activation quantization, the calibrated extremes must be extrapolated to future inputs, and distribution drift introduces additional saturation risk.

### 2.5 Use Cases

- Basic quantization scenarios; the preferred quantization method for conventional bit widths such as INT8.
- Scenarios that are sensitive to computational overhead and require quantization to complete quickly.

### 2.6 Usage Restrictions

- Very sensitive to outliers; accuracy drops noticeably at low bit widths (such as INT8 and INT4).
- When extreme outliers are present, combine it with outlier suppression algorithms such as [SmoothQuant](../smooth_quant/term_smooth_quant.md), or use a more advanced quantization algorithm such as [SSZ](../ssz/term_ssz.md).

---

## 3. Related Terms

- [Linear Quantization](../linear_quant/term_linear_quant.md): the application target; MinMax is the basic quantization method of linear quantization.
- [Histogram](../histogram_activation_quantization/term_histogram_activation_quantization.md): a comparison algorithm that filters outliers through histogram clipping; its accuracy is usually better than MinMax.
- [SSZ](../ssz/term_ssz.md): a comparison algorithm that optimizes quantization parameters through iterative search; suitable for low-bit weight quantization.
- [SmoothQuant](../smooth_quant/term_smooth_quant.md): a companion term that can suppress outliers to improve MinMax accuracy.

---

## 4. Reference Documents

1. Jacob B et al. Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference. CVPR 2018. <https://arxiv.org/abs/1712.05877>
2. [MinMax Parameter Configuration Workflow Guide](./usage_minmax.md)
