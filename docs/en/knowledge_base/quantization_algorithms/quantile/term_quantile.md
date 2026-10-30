# Quantile Sensitive Layer Analysis Algorithm Terminology Entry

> **Entry Category**: [Sensitive Layer Analysis Algorithms](../README.md#3-sensitive-layer-analysis-algorithms)<br>
> **English Name**: quantile<br>
> **Application Domain**: Quantization sensitive layer analysis, model quantization accuracy tuning<br>

---

## 1. Overview

Quantile is a sensitive layer analysis algorithm based on robust statistics. It uses quantiles and the interquartile range to describe the relative relationship between the bulk activation distribution and the tail, reducing the dominance of a very small number of abnormal values over the sensitivity metric, and thereby ranks linear layers; its core characteristic is relative robustness to outliers, and it is suitable for cross-validation with metrics such as [Std](../std/term_std.md) and [Kurtosis](../kurtosis/term_kurtosis.md).

---

## 2. Term Description

Metrics based on extremes and the standard deviation, such as [Std](../std/term_std.md), are easily dominated by outliers, which inflates a single layer's score. Quantile observes that describing the bulk width of the activation distribution with quartiles (the 1/4 and 3/4 quantiles) reduces the dominance of outliers over the single-layer score, making the sensitivity ranking more robust.

### 2.1 Core Idea

The core idea of Quantile is to "characterize the bulk distribution with quartiles": compute the lower quartile $Q_1$ (the 1/4 quantile) and the upper quartile $Q_3$ (the 3/4 quantile) of the activations, describe the bulk distribution width by the interquartile range $\text{IQR} = Q_3 - Q_1$, and build a score together with the absolute magnitude, which is relatively robust to outliers.

This ratio is approximately invariant under an overall linear scaling, making it suitable for comparing layers in the same model that differ in units and absolute magnitude.

### 2.2 Working Mechanism

The Quantile metric first uses $Q_1$ and $Q_3$ to describe the span of the middle 50% of the activation samples, namely the IQR. The IQR is insensitive to a single extreme value and can therefore characterize "how wide the bulk distribution is" relatively stably. At the same time the numerator still retains the absolute maximum and uses it to approximate the overall dynamic range of symmetric INT8 MinMax quantization. Dividing the two gives a dimensionless ratio expressing "how large the quantization step size determined by the extreme is relative to the bulk distribution width".

This means the metric does not entirely ignore outliers; on the contrary, it lets the extreme appear only in the numerator and the robust bulk scale in the denominator, thereby highlighting layers where "a small tail stretches the quantization range far beyond the bulk width". When this ratio is high, the bulk data may occupy only a small number of integer codes, and MinMax-style quantization is more likely to lose resolution. Since both the quantiles and the extremes come from the calibration activations, sample coverage and sampling still affect the ranking.

### 2.3 Mathematical Description

Interquartile range:

$$
\text{IQR} = Q_3 - Q_1
$$

Sensitivity score:

$$
\text{score} = \frac{2 \times \max(|\text{max\_value}|, |\text{min\_value}|)}{254 \times (Q_3 - Q_1)}
$$

- $Q_1$: the 1/4 quantile of the activations
- $Q_3$: the 3/4 quantile of the activations
- $\text{IQR}$: the interquartile range
- $\max(|\text{max\_value}|, |\text{min\_value}|)$: the maximum absolute activation value
- $\text{score}$: sensitivity score

Interpretation: the larger the maximum absolute activation value, the larger the score (a relatively larger quantization step and more significant error); the larger the IQR, the smaller the score (a more spread-out bulk distribution and a smaller relative error).

Let $a=\max(|x_{min}|,|x_{max}|)$; the full real-valued range width of symmetric INT8 is $2a$ across 254 adjacent intervals, so:

$$
S\approx\frac{2a}{254},\qquad \text{score}=\frac{S}{Q_3-Q_1}.
$$

The score is thus dimensionless: if all activations are multiplied by a constant $c$, both $S$ and the IQR are multiplied by $|c|$ and the score stays essentially unchanged. What it measures is the shape of the distribution and the size of the tail relative to the bulk, not the absolute magnitude.

### 2.4 Key Properties

- **Linear-layer granularity**: statistics are collected on the input activations of linear transformations, producing a relative sensitivity ranking across layers.
- **Robust bulk scale**: the IQR is relatively robust to a small number of outliers, while the full score still deliberately retains sensitivity to long-tail risk through the abs-max.
- **Bulk distribution characterization**: uses the IQR to describe the activation bulk distribution width.
- **Tail-relative-to-bulk metric**: the denominator IQR robustly describes the bulk, and the numerator abs-max deliberately keeps sensitivity to the outlier tail.

In terms of error and applicability boundaries, a single sporadic abnormal value still raises the score, which is part of how the metric detects tail risk, but it also means calibration noise can cause false positives. Too sparse quantile sampling may also distort $Q_1/Q_3$, so this metric is suitable for fast relative screening rather than absolute error prediction.

### 2.5 Use Cases

- Scenarios where the activation distribution has a heavy tail and the dominance of outliers over the single-layer score should be reduced.
- Scenarios that need a more robust sensitivity ranking of linear layers.

### 2.6 Usage Restrictions

- The specific threshold for the score must be determined in light of the model and the business accuracy requirements.
- Only for `linear`-scope analysis; not applicable to the `layer`/`attn` scopes.

---

## 3. Related Terms

- [Std](../std/term_std.md): a similar algorithm and another `linear`-scope analysis metric that emphasizes the ratio of range to dispersion.
- [Kurtosis](../kurtosis/term_kurtosis.md): a similar algorithm and another `linear`-scope analysis metric that focuses on spiky distributions.
- [MSE Layer Wise](../mse_layer_wise/term_mse_layer_wise.md): a companion term and a `layer`-scope analysis metric, usable for block-level evaluation.

---

## 4. Reference Documents

1. [Quantile Analysis Configuration Workflow Guide](./usage_quantile.md)
