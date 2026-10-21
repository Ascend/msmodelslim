# Ceil_X Adaptive Divisor MXFP4 Quantization Algorithm Terminology Entry

> **Entry Category**: [Quantization Algorithms](../README.md#2-quantization-algorithms)<br>
> **English Name**: ceil_x<br>
> **Application Domain**: MXFP4 weight quantization, low-bit quantization accuracy optimization<br>

---

## 1. Overview

Ceil_X is a scaling optimization algorithm for MXFP4 per-block weight quantization. It adjusts the shared exponent through a combination of `ceil` and a configurable divisor so that fewer large values within a block are clipped because of insufficient scaling; its core characteristics are block-level shared exponent, adjustable divisor, and optional error search, making it suitable for reducing the accuracy loss caused by large-value clipping in MXFP4 low-bit weight quantization.

---

## 2. Term Description

In MXFP4 per-block quantization, traditional methods compute the shared exponent as $s = \lfloor \log_2(\max(|x|)) \rfloor - e_{\text{max}}$, after which the scaled numerical range is $[4, 8)$, while the representation ceiling of MXFP4 is $6.0$. As a result, large values in the $[6, 8)$ interval are clipped, introducing a large quantization error. Ceil_X introduces a configurable divisor $c$ together with the `ceil` operation to compress the scaled numerical range to $(c/2, c]$, which falls entirely within the representable interval of MXFP4.

### 2.1 Core Idea

The core idea of Ceil_X is to "use a ceil shared exponent to guarantee the dynamic range first, then use the divisor $c$ to adjust the balance between range and resolution". For each block, a power-of-two shared scale is selected according to $\lceil\log_2(\max|x|/c)\rceil$. Compared with directly flooring the exponent, the ceil candidate reserves more representation space for large values; the parameter $c$ shifts the exponent transition boundary, letting the algorithm trade off between reducing saturation and keeping a finer quantization step size.

The advantage of ceil is that it significantly reduces the risk of overflow or clipping of the maximum value caused by a too-small shared exponent, which is especially useful for blocks where FP4 high-end code points are limited and saturation error is expensive.

### 2.2 Working Mechanism

Block floating-point formats such as MXFP4 typically let a group of elements share a power-of-two scale (the shared exponent). Given the maximum absolute value within a block, the continuous ideal scale usually falls between two adjacent powers of two; choosing the smaller exponent gives a finer step size but may push the largest element outside the representable range, while choosing the larger exponent provides more headroom at the cost of resolution. Ceil_X prioritizes guaranteeing sufficient representation headroom for large values within the block by rounding the shared exponent up.

The parameter $c$ shifts the boundary at which a jump to the next exponent is required. Dividing the maximum by $c$ before applying $\lceil\log_2\rceil$ is equivalent to controlling which target interval the scaled maximum should fall into. A larger $c$ usually tends to keep a smaller shared exponent and obtain a finer step size, but the saturation risk increases; a smaller $c$ is more conservative. If the search is enabled, real QDQ is performed on candidate $c$ values and the cumulative MSE is used to select a global compromise point.

### 2.3 Mathematical Description

Conventional floor scaling:

$$
s_{\text{floor}} = \lfloor \log_2(\max(|x|)) \rfloor - e_{\text{max}}
$$

Ceil_X scaling:

$$
s = \operatorname{ceil}\left(\log_2\left(\frac{\max(|x|)}{c} + \epsilon\right)\right) - e_{\text{max}}
$$

- $s$: shared exponent
- $\max(|x|)$: the maximum absolute weight value within the block
- $c$: a positive divisor that controls the shared-exponent transition boundary.
- $\epsilon$: a numerical stability term, $\epsilon = 9.6 \times 10^{-7}$
- $e_{\text{max}}$: the exponent bias; for the MXFP4 E4M2 format, $e_{\text{max}} = 2^{e_{\text{bits}}-1} = 2$

Magnitude of the maximum value within the block after scaling:

$$
\frac{\max(|x|)}{2^s} = \frac{\max(|x|)}{2^{\operatorname{ceil}(\log_2(\max(|x|)/c))}} \in \left(\frac{c}{2}, c\right]
$$

In general, the scaled block maximum is constrained within the magnitude interval $(c/2,c]$. When $c$ changes, the shared exponent only jumps discretely when a power-of-two boundary is crossed, so the error varies piecewise with respect to $c$.

Optional adaptive search:

$$
c^* = \arg\min_{c \in [c_{\min}, c_{\max}]} \sum_{\text{blocks}} \|x - \hat{x}(c)\|^2
$$

- $c^*$: the optimal divisor found by the search
- $c_{\min}$, $c_{\max}$: the search boundaries for candidate divisors.
- $\hat{x}(c)$: the value after quantization-dequantization using divisor $c$

If the shared exponent is $s$ and the normalized block is written as $u=x/2^s$, the subsequent MXFP4 encoding can be abstracted as $\hat x=2^s\,\mathcal F(u)$, where $\mathcal F$ denotes the low-bit floating-point lattice projection. The block error is then:

$$
E(s)=\sum_i\left(x_i-2^s\mathcal F(x_i/2^s)\right)^2.
$$

Ceil_X does not optimize $E(s)$ directly and continuously; instead it controls the integer exponent $s(c)$ through $c$. When $s(c)$ is unchanged, $E$ is completely unchanged; when $s(c)$ crosses to an adjacent exponent, the entire block switches to the new lattice scale at once.

### 2.4 Key Properties

- **ceil scaling**: uses the ceil operation to tighten the scaling range and avoid clipping large values.
- **Adjustable exponent boundary**: the divisor $c$ controls when to switch to a larger shared exponent.
- **Adaptive search**: optionally enables a global MSE search for the optimal divisor $c$.
- **Range/resolution trade-off**: ceil provides more dynamic-range headroom, but shifting the exponent up enlarges the quantization step size at the same time.

In terms of error and applicability boundaries, the $c$ obtained by the global search is optimal on average across many blocks and is not guaranteed to be optimal for every block, which is an important difference from per-block MSE_Round.

### 2.5 Use Cases

- Scenarios with higher accuracy requirements for mxFP4 quantization.
- Model layers where the weight distribution spans a wide range and floor scaling results in a too-coarse step size or clipping of large values.

### 2.6 Usage Restrictions

- Only supports per_block symmetric quantization in the mxFP4 format.
- Enabling `enable_search` adds several forward quantization evaluations and increases computational overhead.
- `search_max` must be greater than `search_min`.

---

## 3. Related Terms

- [Linear Quantization](../linear_quant/term_linear_quant.md): the application target; Ceil_X is used as the weight quantization method of linear quantization.
- [FouroverSix](../fouroversix/term_fouroversix.md): a similar algorithm and another scaling-strategy optimization algorithm for the MXFP4 format.
- [MSE_Round](../mse_round/term_mse_round.md): a similar algorithm and another shared-exponent optimization algorithm for the MXFP format (targeting MXFP8).
- [MinMax](../minmax/term_minmax.md): a comparison algorithm; Ceil_X is the clipping suppression optimization for MinMax MXFP4 quantization.

---

## 4. Reference Documents

1. [Ceil_X Parameter Configuration Workflow Guide](./usage_ceil_x.md)
