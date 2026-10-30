# FouroverSix Adaptive Block Scaling Quantization Algorithm Terminology Entry

> **Entry Category**: [Quantization Algorithms](../README.md#2-quantization-algorithms)<br>
> **English Name**: fouroversix<br>
> **Application Domain**: MXFP4 weight quantization, low-bit quantization accuracy optimization<br>

---

## 1. Overview

FouroverSix (4-over-6) is an adaptive block scaling algorithm for MXFP4 per-block weight quantization. It compares the quantization reconstruction error across different candidate scaling schemes and selects the better one for each block, alleviating the inadequate adaptation of uniform scaling to different numerical distributions; its core characteristics are block-level two-candidate evaluation, MSE-based selection, and adaptive shared-exponent choice.

---

## 2. Term Description

Conventional mxFP4 quantization methods use a fixed scaling factor uniformly when processing data blocks with different distributions (usually scaling the maximum to the FP4 ceiling of 6), which can cause a relatively large quantization error for some blocks. FouroverSix observes that when a data block contains outliers or has an uneven distribution, using a smaller scaling factor (for instance scaling to 4) can significantly reduce the quantization error, and therefore makes a per-block adaptive choice of the optimal scaling scheme.

### 2.1 Core Idea

The core idea of FouroverSix is to "adaptively select the scaling target for each data block": for each data block it tries two scaling schemes at the same time — scheme A (Scale-to-6) scales the block maximum to 6, the maximum of the FP4 format, to make full use of the dynamic range, while scheme B (Scale-to-4) scales the block maximum to 4 to provide more headroom — computes the mean squared error (MSE) of the two schemes, and chooses the scheme with the smaller error as the final quantization scheme for that block.

The method exploits the structure of the non-uniform lattice of low-precision floating point rather than an ordinary uniform integer grid.

### 2.2 Working Mechanism

The numerical lattice of MXFP4 is not uniformly spaced along the real axis; especially at the high end of the representable range, the spacing between adjacent FP4 values increases noticeably. Mapping the block maximum exactly to the format maximum therefore does not always minimize the error of the whole block. Four Over Six constructs two scale candidates for each block: one scales the maximum with a near-full-range target close to 6, and the other scales it with a more conservative target close to 4.

The two candidates are passed through the same FP4 encoding/decoding and their in-block MSE is compared directly. Scale-to-6 uses the dynamic range more fully, but the coarse high-end lattice may make a group of elements near the maximum incur a larger error; Scale-to-4 shifts where these elements land on the FP4 lattice and sometimes obtains a denser or more suitable effective representation at the cost of part of the nominal range. The algorithm ultimately just selects the better-reconstructing scale path for each block without changing the underlying FP4 data format.

### 2.3 Mathematical Description

Scaling factors of scheme A (Scale-to-6) and scheme B (Scale-to-4):

$$
\text{scale\_a} = \frac{\max\_per\_block}{6.0}, \quad \text{scale\_b} = \frac{\max\_per\_block}{4.0}
$$

- $\max\_per\_block$: the maximum absolute weight value within the block

Both real-valued scales also need to be mapped to a shared scale representable in e8m0. Denote the mapped scales $\tilde{s}_a$ and $\tilde{s}_b$; this process essentially constrains the continuous scale to near an integer power of two, so the scale itself also introduces discretization error.

Compute the MSE of the two schemes separately:

$$
\text{MSE}_a = \frac{1}{N}\sum_{i=1}^{N}(w_i - \hat{w}_i^{(a)})^2, \quad \text{MSE}_b = \frac{1}{N}\sum_{i=1}^{N}(w_i - \hat{w}_i^{(b)})^2
$$

- $w_i$: the weight values within the block
- $\hat{w}_i^{(a)}$, $\hat{w}_i^{(b)}$: the values after quantization-dequantization under scheme A/B
- $N$: number of elements within a block

Finally select the scheme with the smaller MSE:

$$
s^* = \begin{cases} \tilde{s}_a & \text{if } \text{MSE}_a \le \text{MSE}_b \\ \tilde{s}_b & \text{otherwise} \end{cases}
$$

Exponent rounding strategy (e8m0 format): when the mantissa is > 0.5, the exponent is incremented by 1; when the mantissa is < 0.5, the exponent is left unchanged; when the mantissa == 0.5, the banker's rounding rule is applied (an even exponent is incremented and an odd one is not).

Let the representable FP4 set be $\mathcal F$ and the e8m0 scale projection be $P_E(\cdot)$. The two candidates can be written more generally as:

$$
\tilde s_t=P_E\left(\frac{a}{t}\right),\qquad \hat w_i^{(t)}=\tilde s_t\,P_{\mathcal F}\left(\frac{w_i}{\tilde s_t}\right),\quad t\in\{4,6\},
$$

where $a=\max_i|w_i|$ and $P_{\mathcal F}$ denotes projection to the nearest FP4 code point. Finally the $t$ with the smaller $\sum_i(w_i-\hat w_i^{(t)})^2$ is selected. This expression distinguishes the "target value" from the "actually storable scale".

### 2.4 Key Properties

- **Adaptive scaling**: each block intelligently selects its scaling target between 6 and 4.
- **Two-path evaluation**: evaluates the quantization error of both the Scale-to-6 and Scale-to-4 schemes simultaneously.
- **Banker's rounding**: the e8m0 exponent uses nearest-neighbor rounding (including the banker's rounding rule) to ensure precision.
- **Format compatible**: keeps the mxFP4 quantization format unchanged, optimizing only the scaling strategy.
- **Format preserving**: only the block scale candidate selection changes; the FP4 code table itself need not change.

In terms of error and applicability boundaries, its advantage is a very small search space: comparing just two QDQ results per block captures a considerable part of the error differences caused by scale rounding, at a relatively controllable cost. The limitation is that there are only two candidates, so if the optimal scale lies far from both preset targets there is still room for improvement; moreover the benefit depends on the specific FP4 code table and shared-scale rules and cannot be extrapolated directly to uniform INT4.

### 2.5 Use Cases

- mxFP4 quantization scenarios with higher accuracy requirements, especially well suited to models with an uneven data distribution.
- Transformer models whose attention-layer and feed-forward-layer weights differ considerably in distribution, as well as multimodal models.

### 2.6 Usage Restrictions

- Only supports per_block symmetric quantization in the mxFP4 format.
- Weights must be 2D tensors.
- Each block requires two quantization/dequantization passes and an MSE computation, so the computational cost is slightly higher than conventional mxFP4 quantization.

---

## 3. Related Terms

- [Linear Quantization](../linear_quant/term_linear_quant.md): the application target; FouroverSix is used as the weight quantization method of linear quantization.
- [Ceil_X](../ceil_x/term_ceil_x.md): a similar algorithm and another scaling-strategy optimization algorithm for the MXFP4 format.
- [MSE_Round](../mse_round/term_mse_round.md): a similar algorithm and another per-block two-candidate selection quantization algorithm (targeting MXFP8).
- [MinMax](../minmax/term_minmax.md): a comparison algorithm; FouroverSix is the adaptive optimization of MinMax MXFP4 quantization.

---

## 4. Reference Documents

1. [FouroverSix Parameter Configuration Workflow Guide](./usage_fouroversix.md)
2. Cook J, Guo J, Xiao G, Lin Y, Han S. "Four Over Six: More Accurate NVFP4 Quantization with Adaptive Block Scaling." arXiv preprint arXiv:2512.02010, 2025. <https://arxiv.org/abs/2512.02010>
