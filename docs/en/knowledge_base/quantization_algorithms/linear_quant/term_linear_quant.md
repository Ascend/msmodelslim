# Linear Quantization Algorithm Terminology Entry

> **Entry Category**: [Quantization Algorithms](../README.md#2-quantization-algorithms)<br>
> **English Name**: linear_quant<br>
> **Application Domain**: Deep learning model compression, large language model quantization<br>

---

## 1. Overview

Linear quantization is a class of basic quantization methods that map continuous floating-point values onto a finite discrete set using a scale factor and an optional zero point; it can be applied to the weights and activations of linear layers. The bit width, quantization granularity, symmetry, and parameter estimation method jointly control the error and the compression benefit. [MinMax](../minmax/term_minmax.md), [Histogram](../histogram_activation_quantization/term_histogram_activation_quantization.md), and others are common implementations.

---

## 2. Term Description

Model inference and deployment place extremely high demands on memory, bandwidth, and compute, making the storage and computation overhead of floating-point weights a bottleneck. Linear quantization significantly reduces model size and computational cost by mapping floating-point values to low-bit integers. According to when the quantization parameters are measured and computed, it can be divided into static quantization, dynamic quantization, and hybrid quantization (such as [PDMIX](../pdmix/term_pdmix.md)), matching different accuracy and performance requirements.

### 2.1 Core Idea

The core idea of linear quantization is to map continuous real numbers onto a finite, equally spaced set of integer grid points. The quantization scale $S$ determines the spacing between adjacent integer codes in the real domain, and the zero point $Z$ determines the integer position corresponding to real zero. During quantization, the value is first divided by $S$, then $Z$ is added, then it is rounded and clipped; during dequantization, $S(Q-Z)$ is computed. Linear quantization is therefore essentially a projection onto the uniform grid points of the real axis.

The algorithm's behavior is mainly determined by three things: how wide a range the quantization interval covers, how many elements share the same pair of $S/Z$, and whether the quantization parameters are obtained from offline statistics or from runtime data. If the interval is too wide, the grid points become sparse; if it is too narrow, saturation clipping occurs. Finer granularity usually fits the local distribution better, but it also increases the scale metadata and computational complexity.

The total error of linear quantization can be roughly divided into three classes: elements outside the range produce clipping error; elements inside the range projected onto the nearest grid point produce rounding error; asymmetric quantization is also affected by zero-point rounding and by the imperfect symmetry of the finite integer endpoints.

### 2.2 Working Mechanism

First determine the real range $[r_{min},r_{max}]$ and the integer range $[Q_{min},Q_{max}]$. Asymmetric quantization usually takes $S=(r_{max}-r_{min})/(Q_{max}-Q_{min})$ and then chooses the $Z$ that maps real 0 as precisely as possible; symmetric quantization makes the range symmetric about 0, commonly using $S=\max|x|/Q_{max}$ with $Z=0$ (the exact integer endpoints vary with the format convention). Every element is then projected onto the nearest available integer grid point and clipped when it falls outside the range.

The granularity determines the sharing scope of $r_{min}/r_{max}$ and $S/Z$. Per-tensor has the fewest parameters but is easily dominated by local outliers; per-channel isolates the scale differences of different output/input channels; per-group trades off between accuracy and metadata; per-token dynamic quantization lets each token adapt to its own magnitude. The observer estimates the range or statistics, and the quantizer performs the discrete mapping according to those parameters; together they determine the final error.

### 2.3 Mathematical Description

Let the quantization scale of a real number $x$ be $S>0$, the zero point be an integer $Z$, and the target integer range be $[Q_{\min},Q_{\max}]$. Affine linear quantization can be written as:

$$
q = \operatorname{clip}\left(\operatorname{round}\left(\frac{x}{S}\right)+Z,\;Q_{\min},Q_{\max}\right)
$$

Dequantization is:

$$
\hat{x}=S(q-Z)
$$

For asymmetric quantization, a common range mapping is:

$$
S=\frac{r_{\max}-r_{\min}}{Q_{\max}-Q_{\min}}, \qquad
Z=\operatorname{clip}\left(\operatorname{round}\left(Q_{\min}-\frac{r_{\min}}{S}\right),Q_{\min},Q_{\max}\right)
$$

Symmetric quantization usually sets $Z=0$ (or uses the center code specified by the format) and uses:

$$
S=\frac{\max(|r_{\min}|,|r_{\max}|)}{Q_{\max}^{+}}
$$

where $Q_{\max}^{+}$ denotes the maximum positive-side integer code used by the symmetric format.

- $S$: the real-number interval corresponding to one integer code step.
- $Z$: the mapping position of real zero in the integer domain.
- $r_{\min},r_{\max}$: the real range obtained from the observer or from dynamic statistics.
- $q$, $\hat x$: the integer representation and its dequantized approximation.

For values that do not undergo clipping, rounding to the nearest grid point gives an error $e=x-\hat x$ that approximately satisfies $|e|\le S/2$. Under the classical high-resolution assumption that "the in-grid error is approximately uniform and independent of the signal":

$$
\mathbb E[e]\approx0,\qquad \mathbb E[e^2]\approx\frac{S^2}{12}.
$$

This yields an important intuition: doubling the range approximately doubles $S$, while the in-interval quantization noise variance grows by roughly a factor of four. Conversely, if the range is narrowed, values beyond the bounds produce a clipping error that may be far larger than $S/2$. Choosing the quantization range is therefore essentially a trade-off between these two kinds of error.

### 2.4 Key Properties

- **Basic and general**: applicable to the quantization of weights and activations, and the foundation of most quantization schemes.
- **Adjustable granularity**: quantization granularities such as per-tensor, per-channel, and per-group can be used.
- **Static and dynamic**: supports static quantization (parameters fixed offline) and dynamic quantization (parameters computed online).
- **Extensible algorithms**: quantization parameter estimation can use different methods such as MinMax, Histogram, SSZ, and GPTQ to match different data distributions and accuracy targets.
- **Unified representation framework**: methods such as MinMax, Histogram, and dynamic quantization differ mainly in how they estimate $S/Z$, and all of them ultimately reduce to the same affine quantization form.

In terms of error and applicability boundaries: a fine-grained scale gives each group of data a smaller local range, so it is usually more accurate, but it increases the number of parameters, memory accesses, and operator constraints. Dynamic quantization adapts to input changes but introduces an online statistical cost.

### 2.5 Use Cases

- Basic quantization scenarios, where the weights and activations of a linear transformation must be represented as integers or in low precision.
- Can serve as the underlying carrier for quantization ideas such as PDMIX, Histogram, SSZ, and GPTQ.

### 2.6 Usage Restrictions

- The linear quantization discussed in this entry mainly targets two-dimensional linear transformations; other operators such as convolution and normalization require a quantization definition that matches their tensor layout.
- There are compatibility relationships among the data format, the quantization granularity, the symmetry, and the parameter estimation method; a practical scheme should choose a valid combination supported by the deployment backend.

---

## 3. Related Terms

- [MinMax](../minmax/term_minmax.md): an application target; the most basic quantization method of linear quantization.
- [Histogram](../histogram_activation_quantization/term_histogram_activation_quantization.md): an application target; the activation quantization method of linear quantization.
- [SSZ](../ssz/term_ssz.md): an application target; the weight quantization method of linear quantization.
- [GPTQ](../gptq/term_gptq.md): an application target; the high-accuracy weight quantization method of linear quantization.
- [PDMIX](../pdmix/term_pdmix.md): an application target; the hybrid activation quantization method of linear quantization.

---

## 4. Reference Documents

1. Jacob B et al. Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference. CVPR 2018. <https://arxiv.org/abs/1712.05877>
2. [Linear Quantization Parameter Configuration Workflow Guide](./usage_linear_quant.md)
