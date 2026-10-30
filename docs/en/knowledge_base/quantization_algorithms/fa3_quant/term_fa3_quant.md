# FA3 Quant Attention Activation Quantization Algorithm Terminology Entry

> **Entry Category**: [Quantization Algorithms](../README.md#2-quantization-algorithms)<br>
> **English Name**: fa3_quant<br>
> **Application Domain**: Large language model quantization and compression, inference acceleration, long-sequence inference<br>

---

## 1. Overview

FA3 Quant is an attention activation quantization algorithm that works on a per-head basis. It establishes quantization ranges for attention intermediate tensors such as Q, K, and V head by head in order to reduce the storage and computation overhead of long-sequence attention while keeping the activation quantization error under control; its core characteristics are per-head granularity, activation quantization along the attention path, and a dedicated statistical strategy for range selection.

---

## 2. Term Description

At long sequence lengths, the attention intermediate activations — the Q, K, and V tensors — account for a large share of memory, so quantizing them can effectively reduce memory usage and improve computational efficiency. However, the activations of Q, K, and V have a wide dynamic range and a highly uneven distribution, and quantizing them globally leads to severe accuracy loss. FA3 Quant quantizes each attention head independently so as to accommodate the differences in activation distribution between heads.

### 2.1 Core Idea

The core idea of FA3 Quant is "independent quantization per attention head": compute the quantization parameters (scale) for each attention head separately so as to accommodate differences in activation distribution between heads; each head uses the Recall Window algorithm to find the smallest numerical distribution interval containing a specified proportion of the data as the quantization range, preventing outliers from inflating the quantization scale.

Per-head statistics block the error coupling in which "an outlier in one head coarsens all heads"; Recall Window further solves the problem of a very small number of tail samples within a single head.

### 2.2 Working Mechanism

The Q/K/V or intermediate activation distributions of different attention heads are not necessarily the same. If all heads share one quantization scale, an extreme value in one head raises the step size for the whole group, wasting a large number of code points on the bulk data of the other heads. FA3 Quant therefore collects statistics per head independently, removing "cross-head dynamic-range differences" as a source of quantization error.

Within each head, Recall Window does not insist that the quantization interval cover all samples; instead it searches the sorted samples for the shortest interval that contains a given proportion of the data. This amounts to deliberately discarding a very small number of tail points in exchange for a narrower bulk range and a smaller quantization step size. For symmetric quantization, the larger of the absolute values at the two ends of the window then forms the $[-a,a]$ range; across calibration batches, statistics must be merged so that an accidentally narrow interval from one batch does not cause frequent saturation during real inference.

### 2.3 Mathematical Description

For an activation tensor $x$ (with shape $(B, H, S, D)$), reshape it to $(H, N)$, where $N = B \times S \times D$, and collect $N$ data points independently for each head.

Use the Recall Window algorithm on each head to find the minimum quantization range:

1. Sort: $\text{sorted\_data} = \operatorname{sort}(\text{head\_data})$
2. Target number of elements: $\text{target\_num} = \lfloor \text{ratio} \times N \rfloor$
3. Sliding-window search for the range with the smallest window length: $[\text{sorted\_data}[i], \text{sorted\_data}[i + \text{target\_num} - 1]]$

Symmetric quantization parameters:

$$
\text{abs\_max}[h] = \max(|\min\_values[h]|, |\max\_values[h]|), \quad \text{scale}[h] = \frac{\text{abs\_max}[h]}{127}
$$

- $B$: the batch size
- $H$: the number of attention heads
- $S$: the sequence length
- $D$: the head dimension
- $h$: the attention head index
- $\text{ratio}$: the proportion of samples that Recall Window is expected to keep inside the quantization interval.
- $\text{scale}[h]$: the quantization scaling factor of head $h$

When accumulating statistics across batches, the quantization range takes the union of the minimum/maximum values over all calibration batches.

Writing the sorted samples of a head as $x_{(1)}\le\cdots\le x_{(N)}$ and the number retained as $m=\lfloor rN\rfloor$, Recall Window can be written as:

$$
i^*=\arg\min_{1\le i\le N-m+1}\left(x_{(i+m-1)}-x_{(i)}\right),
$$

$$
[l,u]=[x_{(i^*)},x_{(i^*+m-1)}].
$$

Samples excluded by the window produce clipping error; the narrower the window, the smaller the quantization step size within the range. For symmetric INT8, one can further set $a=\max(|l|,|u|)$ and $s=a/127$. The retention rate $r$ therefore explicitly controls the trade-off between "how much of the tail is clipped" and "how fine the bulk resolution is".

### 2.4 Key Properties

- **Per-head quantization**: computes quantization parameters independently for each attention head to accommodate distribution differences.
- **Recall Window**: uses a sliding window to find the minimum range containing a given proportion of the data, suppressing the influence of outliers.
- **Multi-granularity support**: supports multiple quantization granularities for INT8 and FP8 (static/dynamic).
- **Trade-off between clipping and accuracy**: the retention ratio of Recall Window directly determines the balance between tail saturation and bulk resolution.

In terms of error and applicability boundaries, attention is especially sensitive to Q/K quantization because the error enters the dot product and passes through softmax. A seemingly small logit deviation can change the probability ranking when two candidate positions score similarly; the calibration data must therefore cover real sequence lengths, positional patterns, and head behavior.

### 2.5 Use Cases

- Long-sequence inference scenarios that need to reduce the memory footprint of attention intermediate activations.
- Full quantization solutions for models based on the MLA architecture (such as the DeepSeek series).

### 2.6 Usage Restrictions

- Primarily aimed at attention structures compatible with the FA3 low-precision attention path, especially MLA-like architectures; ordinary attention structures cannot be assumed to have the same quantization insertion points.
- The quantization granularity and data format of Q, K, and V must be consistent with the actual capabilities of the attention operator; lower-precision formats usually require stricter end-to-end accuracy validation.

---

## 3. Related Terms

- [Linear Quantization](../linear_quant/term_linear_quant.md): a companion term; FA3 Quant is usually combined with linear quantization to implement a full quantization solution.
- [KVCache Quant](../kvcache_quant/term_kvcache_quant.md): a companion term in the same family of cache/activation quantization for long-sequence inference.
- [KV Smooth](../kv_smooth/term_kv_smooth.md): a companion term and another cache optimization algorithm for long-sequence inference.
- [MinMax](../minmax/term_minmax.md): the application target; the quantization parameters of FA3 Quant are computed following the MinMax approach.

---

## 4. Reference Documents

1. Dao T et al. FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision. 2024. <https://arxiv.org/abs/2407.08608>
2. [FA3 Quant Parameter Configuration Workflow Guide](./usage_fa3_quant.md)
