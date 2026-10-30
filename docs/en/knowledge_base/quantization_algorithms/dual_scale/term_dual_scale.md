# DualScale Dual-Scale Quantization Algorithm Terminology Entry

> **Entry Category**: [Quantization Algorithms](../README.md#2-quantization-algorithms)<br>
> **English Name**: dual_scale<br>
> **Application Domain**: Large language model quantization and compression, low-bit quantization accuracy optimization<br>

---

## 1. Overview

DualScale is a dual-scale quantization method for low-bit scenarios such as W4A4. It introduces a finer-grained scale in addition to a coarser-grained scale to weaken the dominance of abnormal channels over the same quantization interval; its core characteristics are two cooperating scale levels, the ability to use different scaling paths for weights and activations, and improved adaptability of the low-bit representation to local dynamic-range differences while remaining hardware-friendly.

---

## 2. Term Description

Conventional group quantization usually adopts a single-scale strategy: each group of K elements shares one scaling factor. However, structured abnormal channels exist in the activations — the values of some channels are on average several orders of magnitude larger than others — and single-scale quantization cannot precisely represent such widely differing numerical ranges at the same time, causing accuracy loss. DualScale uses a two-level progressive scaling factor: it first computes an outer scale over a large block to absorb the abnormal-channel differences, then completes the low-bit quantization in the inner block.

### 2.1 Core Idea

The core idea of DualScale is to "absorb distribution differences progressively through two scaling levels": first estimate the outer scale over a larger coarse-grained block and complete magnitude normalization, then use an MXFP4 shared scale over a smaller inner block to represent local variation. The outer scale is thus responsible for the order-of-magnitude differences between blocks, and the inner scale for the details within a block, preventing a single low-bit shared exponent from having to carry an excessively large global dynamic range.

The reason two scale levels are effective is that they allocate "dynamic-range representation capability" in layers.

### 2.2 Working Mechanism

A single-level block scale requires one scale to describe both the absolute magnitude of the whole shared block and the local variation within it. When different large blocks differ greatly in magnitude, a single shared exponent often has to make a difficult trade-off between "covering the largest block" and "giving ordinary blocks enough resolution". DualScale separates these two problems: the outer scale first removes the larger-scale inter-block magnitude differences, producing a normalized tensor, and the inner MXFP4 scale then describes the local distribution of the smaller normalized blocks.

The effective dequantization scale of each element can therefore be understood as "outer scale × inner shared scale". The outer level usually uses larger blocks with less metadata, mainly capturing the coarse-grained dynamic range; the inner blocks are smaller and handle fine adaptation to local peaks. This two-level decomposition frees the inner low-bit format from having to carry a dynamic range spanning many orders of magnitude by itself, improving the utilization efficiency of the limited exponent/mantissa code points.

### 2.3 Mathematical Description

Outer scaling of activations (Dual Scale):

$$
X_{\text{dualscaled}} = \frac{X}{S_{\text{dual\_x}}}, \quad S_{\text{dual\_x}} = \frac{\max(|X_{\text{block}}|)}{\text{MXFP4\_MAX\_NORMAL}}
$$

Inner quantization-dequantization:

$$
X_{q\_dq\_inner} = \operatorname{mxfp4\_quantize\_dequantize}(X_{\text{dualscaled}}, S_{\text{inner\_x}})
$$

Outer dequantization:

$$
X_{q\_dq} = X_{q\_dq\_inner} \times S_{\text{dual\_x}}
$$

Weights are statically quantized and stored at initialization, and two-level dequantization is performed during the forward pass:

$$
W_{q\_dq} = W_{\text{dualscaled\_q\_dq}} \times S_{\text{dual\_w}}
$$

- $X$: activation values
- $W$: the weights
- $S_{\text{dual\_x}}$, $S_{\text{dual\_w}}$: the outer scales of activations / weights
- $S_{\text{inner\_x}}$, $S_{\text{inner\_w}}$: the inner scales of activations / weights
- $\text{MXFP4\_MAX\_NORMAL}$: the largest normal number of MXFP4, taken as $6.0$

Matrix multiplication:

$$
\text{Output} = X_{q\_dq} \cdot W_{q\_dq}^T + \text{bias}
$$

For element $x_i$, let the scale of the outer block it belongs to be $S_o(b(i))$ and that of the inner block be $S_i(g(i))$. The effective dequantization relationship can then be summarized as:

$$
\hat x_i=S_o(b(i))\,S_i(g(i))\,q_i.
$$

The effective step size is therefore also the product of the two scales. If the outer normalization compresses $|x|$ to a similar order of magnitude, the dynamic-range requirement on $S_i$ becomes smaller and the discrete error of the low-bit shared scale decreases accordingly. Conversely, the quantization/storage precision of the outer scale itself also becomes an additional error source.

### 2.4 Key Properties

- **Two-level scaling**: a progressive structure of outer large-block scaling plus inner block quantization.
- **Abnormal-channel absorption**: the outer scale absorbs the differences of abnormal channels, alleviating the accuracy loss of single-scale quantization.
- **Hierarchical dynamic-range decomposition**: the coarse scale describes inter-block magnitude and the fine scale describes the local distribution within a block.

The cost is extra scale metadata and more levels of scaling/de-scaling. If the dual block is too large, the outer level may still be dominated by a few extreme values; if too small, the number of scales and the memory-access overhead rise.

### 2.5 Use Cases

- Ultra-low-bit quantization scenarios such as W4A4, where relatively high model accuracy must be maintained.
- Model quantization scenarios where the activations contain structured abnormal channels.

### 2.6 Usage Restrictions

- Sufficient calibration data or training iterations are needed to optimize the parameters, so quantization takes relatively longer.
- Currently targeted mainly at Qwen3 dense-series models (such as Qwen3-8B/14B/32B); generalization to other model series is not guaranteed.
- The algorithm implementation involves a training process and has certain NPU memory requirements; only devices with NPU memory ≥64 GB are supported.

---

## 3. Related Terms

- [Linear Quantization](../linear_quant/term_linear_quant.md): the application target; DualScale is used as a quantization method of linear quantization (`scope: "dual_scale"`).
- [LAOS](../laos/term_laos.md): a similar algorithm and another W4A4 quantization solution for the Qwen3 dense series.
- [MinMax](../minmax/term_minmax.md): a comparison algorithm; DualScale alleviates MinMax's sensitivity to abnormal channels through two-level scaling.

---

## 4. Reference Documents

1. [DualScale Parameter Configuration Workflow Guide](./usage_dual_scale.md)
