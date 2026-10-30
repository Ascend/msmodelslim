# Histogram-Based Activation Quantization Algorithm Terminology Entry

> **Entry Category**: [Quantization Algorithms](../README.md#2-quantization-algorithms)<br>
> **English Name**: histogram_activation_quantization<br>
> **Application Domain**: Large language model quantization and compression, activation quantization<br>

---

## 1. Overview

Histogram is a method that selects the activation quantization interval based on distribution statistics. It first gathers an activation histogram and then compares the quantization error under different clipping boundaries, reducing the effective accuracy loss caused by a few outliers inflating the quantization range; its core characteristics are distribution-level statistics, automatic clipping search, and an error-criterion-driven choice, making it suitable for activation quantization with long tails or obvious outliers.

---

## 2. Term Description

The conventional [MinMax](../minmax/term_minmax.md) quantizer is easily affected by outliers, which leads to an excessively wide quantization range, low utilization of the effective bit width, and severe accuracy loss. Histogram observes that analyzing the activation value distribution histogram and searching for the optimal clipping interval filters out outliers, improving the quantization accuracy while preserving the overall distribution accuracy.

### 2.1 Core Idea

The core idea of Histogram is to "approximate the full activation distribution with a histogram and let the quantization range be determined by a distribution-level error". The algorithm divides the continuous value range into a finite number of bins to accumulate probability mass, then searches the left and right clipping boundaries; candidate boundaries are compared using objectives such as the L2 reconstruction error or KL divergence, deliberately discarding a very small number of long-tail samples in exchange for a finer quantization resolution over the bulk region.

Compared with MinMax, the histogram approach can explicitly accept a small amount of clipping error to reduce the rounding error of many ordinary samples.

### 2.2 Working Mechanism

Histogram activation quantization first compresses the calibration activations into a histogram-approximated distribution instead of storing only a global minimum and maximum. This makes it possible to estimate "where the bulk of the mass is concentrated and how much probability the tail holds", allowing the quantization range to actively cut off a very small part of the extreme values in exchange for a smaller quantization step size over the bulk region.

When a new calibration batch widens the observed range, the existing histogram must be mapped/merged into the new interval to preserve the cumulative distribution information. After the statistics are complete, the algorithm evaluates an objective function over candidate clipping boundaries: L2-type objectives are directly concerned with the numerical reconstruction error after clipping plus quantization, while KL-type objectives are more concerned with the shape difference between the original distribution and the quantized approximation. The final boundary determines the quantization scale, so it is in effect a data-driven trade-off between "tail saturation error" and "bulk rounding error".

### 2.3 Mathematical Description

After the histogram statistics, the clipping interval $[\text{start\_bin}, \text{end\_bin}]$ is searched using the L2-norm error as the objective:

$$
\text{err} = \left\| p - q \right\|_2
$$

The KL divergence error is:

$$
\text{KL} = \sum_i p_i \log \frac{p_i}{q_i}
$$

- $p$: the original (true) distribution
- $q$: the distribution after quantization
- $p_i$, $q_i$: the probability of the $i$-th bin

During the search, the candidate clipping interval is progressively narrowed and the quantized distribution or reconstruction error is estimated for each candidate boundary. When further narrowing no longer improves the objective, the better boundary is kept, and the quantization parameters are computed from that clipping interval:

$$
\text{scale} = \frac{\text{clip\_max} - \text{clip\_min}}{Q_{\max} - Q_{\min}}
$$

- $\text{clip\_min}$, $\text{clip\_max}$: the lower and upper bounds of the optimal clipping interval
- $Q_{\min}$, $Q_{\max}$: the quantization numerical range
- $\text{scale}$: the quantization scaling factor

If the symmetric clipping threshold is $T$ and the maximum positive integer code is $Q$, the quantization step size is $S=T/Q$. The total expected error can be conceptually decomposed as:

$$
E(T)=\underbrace{\mathbb E[(X-\operatorname{clip}(X,-T,T))^2]}_{\text{clipping}}+\underbrace{\mathbb E[(\operatorname{clip}(X,-T,T)-\hat X_T)^2]}_{\text{rounding}}.
$$

The histogram approximates the expectation above using the bin counts/probabilities and minimizes it over a set of candidate $T$ values. As $T$ decreases the first term grows and the second usually shrinks, so the optimum lies at a trade-off between the two error types rather than necessarily at the sample maximum.

### 2.4 Key Properties

- **Outlier filtering**: filters outliers through the clipping interval, preventing the quantization range from becoming too wide.
- **Distribution aware**: characterizes the distribution from a histogram, reflecting the overall distribution shape better than MinMax.
- **Selectable error metric**: supports both the L2 norm and KL divergence error metrics.
- **Adaptive search**: a nonlinear parameter search automatically finds the optimal clipping interval.
- **Active clipping**: allows sacrificing a very small part of the tail to reduce the bulk quantization step size, which is its key difference from pure MinMax.

In terms of error and applicability boundaries, L2 penalizes large-magnitude errors more strongly and corresponds more directly to numerical reconstruction, while KL emphasizes the probability distribution shape and may choose a different clipping boundary. The histogram is only an approximation with a finite number of bins; too few bins lose boundary detail, and cross-batch remapping also introduces approximation error. If the calibration set lacks the true long tail, the resulting clipping interval may be too narrow.

### 2.5 Use Cases

- Scenarios where activations contain outliers and MinMax quantization loses considerable accuracy.
- Static activation quantization scenarios that require higher accuracy.

### 2.6 Usage Restrictions

- The benefit of histogram clipping depends on the statistical granularity and the target numerical format; when moving to a finer granularity or a lower bit width, the corresponding distribution statistics should be rebuilt and the clipping interval re-searched rather than reusing the threshold from another configuration.
- Supports only activation quantization, not weight quantization (weights should not be configured with `method: "histogram"`).
- Only for static quantization scenarios; calibration data is required to build the histogram.

---

## 3. Related Terms

- [Linear Quantization](../linear_quant/term_linear_quant.md): the application target; Histogram is used as the activation quantization method of linear quantization.
- [MinMax](../minmax/term_minmax.md): a comparison algorithm; Histogram is the outlier-filtering optimization of MinMax.
- [SSZ](../ssz/term_ssz.md): a similar algorithm and another quantization parameter search algorithm (SSZ targets weights).
- [SmoothQuant](../smooth_quant/term_smooth_quant.md): a companion term that can be combined with outlier suppression to further improve accuracy.

---

## 4. Reference Documents

1. [Histogram Parameter Configuration Workflow Guide](./usage_histogram_activation_quantization.md)
