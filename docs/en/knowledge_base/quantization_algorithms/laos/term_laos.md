# LAOS Low-Bit Quantization Solution Terminology Entry

> **Entry Category**: [Quantization Algorithms](../README.md#2-quantization-algorithms)<br>
> **English Name**: laos<br>
> **Application Domain**: Large language model quantization and compression, low-bit quantization accuracy optimization<br>

---

## 1. Overview

LAOS is a **combined quantization solution** (not a single processor) for ultra-low-bit **W4A4** scenarios of **large language models**. It first uses [Adapt Rotation](../adapt_rotation/term_adapt_rotation.md) to redistribute outlier information, then uses [AutoRound](../autoround/term_autoround.md) to optimize low-bit weight rounding, and protects sensitive modules with mixed precision; its core characteristics are the chaining of rotation and rounding optimization, being data-driven, and targeting accuracy recovery at extremely low bit widths.

---

## 2. Term Description

At low bit widths (such as W4A4), model accuracy loss is particularly pronounced. The core difficulty is that extreme outliers in the weights and activations significantly distort the quantization interval, causing a sharp drop in the precision of the numerical representation, which traditional methods struggle to solve. LAOS combines outlier suppression with low-bit quantization training: it first optimizes the rotation matrix in a data-driven way to suppress outliers, then uses learnable rounding to optimize the quantization parameters, achieving higher accuracy and stability under W4A4.

### 2.1 Core Idea

The core idea of LAOS is a two-stage scheme that "suppresses outliers first and then trains the quantization parameters": Stage 1 uses [Adapt Rotation](../adapt_rotation/term_adapt_rotation.md) to iteratively optimize the Hadamard rotation matrix against calibration data so that the transformed activations are more amenable to quantization; Stage 2 uses [AutoRound](../autoround/term_autoround.md) to introduce a learnable rounding offset parameter and adaptively adjust each weight's rounding direction through the SignSGD optimizer, minimizing the quantization reconstruction error.

The two-stage scheme is effective because it improves the geometry of the problem first and then performs discrete optimization.

### 2.2 Working Mechanism

LAOS handles the two main difficulties of low-bit quantization separately. The first stage reorganizes the coordinate distribution of activations/weights through an adaptive orthogonal rotation, weakening as far as possible the dominance of outlier channels and extreme values over the low-bit range; the second stage, on the already friendlier distribution, learns rounding and scale-related decisions through an AutoRound-style optimization method, further reducing the local reconstruction error caused by discretization.

These two kinds of operations are complementary: rotation mainly addresses the conditioning/dynamic-range problem of "how the same energy is distributed across coordinate axes" but cannot eliminate the rounding error caused by a finite code point set itself; learnable rounding can optimize the integer code choice on a fixed low-bit grid but handles poorly on its own the scale waste caused by severe outliers. Shaping the distribution first and then optimizing the discrete decisions is usually more robust than putting all the pressure on a single quantization step. For the small number of sensitive layers that remain unacceptable, a higher bit width can be used to form a mixed-precision boundary.

### 2.3 Mathematical Description

The LAOS solution consists of two stages. The first stage is based on rotation optimization:

$$
H_{\text{adapted}} = H \cdot R
$$

- $H$: initial Hadamard matrix
- $R$: the orthogonal rotation matrix obtained by data-driven optimization
- $H_{\text{adapted}}$: the optimized rotation matrix

The second stage is weight quantization based on learnable rounding:

$$
\hat{W} = s \times (\operatorname{clip}(\lfloor W/s + zp + V \rceil, n, m) - zp)
$$

- $\hat{W}$: the quantized weight (the dequantized reconstruction value)
- $s$: scale factor
- $zp$: zero point
- $V$: learnable offset controlling the rounding direction
- $n$, $m$: lower and upper bounds after quantization

Denote the orthogonal transform obtained in the first stage $Q^*$ and the final quantized weight of the second stage $\hat W(Q^*,V)$; the overall objective can then be written conceptually as:

$$
Q^*=\arg\min_{Q^TQ=I}\mathcal L_{\mathrm{quant}}(Q),\qquad V^*=\arg\min_V\|Y_{fp}-Y_q(Q^*,V)\|^2.
$$

This is a staged approximation rather than a global joint optimization of $Q,V$ at the same time. Its advantage is that it decomposes a high-dimensional difficult problem; the cost is that the optimal rotation of the first stage is not necessarily globally optimal once the second stage's learned rounding is taken into account.

### 2.4 Key Properties

- **Two-stage combination**: rotation optimization (Adapt Rotation) and rounding training (AutoRound) jointly improve W4A4 accuracy.
- **Data-driven rotation**: the rotation matrix is iteratively optimized against calibration data, adapting to the model distribution better than a fixed Hadamard.
- **Learnable rounding**: the rounding direction is determined by a learnable offset, reducing the quantization reconstruction error.
- **Mixed quantization support**: supports a mixed strategy using W8A8 for sensitive layers and W4A4 for the rest.
- **Divided error sources**: rotation mainly handles dynamic range/outliers, and learned rounding mainly handles the discrete error of a finite code point set.

In terms of error and applicability boundaries, errors may still come from rotation calibration bias, activation-sensitive layers at low bit width, and mismatch between the local reconstruction objective and the final task. W4A4 in particular compresses both weights and activations, so the error sources are more complex than W4 alone, and a small number of structurally sensitive layers may still need a W8A8 fallback.

### 2.5 Use Cases

- Ultra-low-bit quantization scenarios such as W4A4, where relatively high model accuracy must be maintained.
- Scenarios where the activations have significant outliers and data-driven rotation suppression is required.

### 2.6 Usage Restrictions

- Sufficient calibration data or training iterations are needed to optimize the parameters, so quantization takes relatively longer.
- Currently targeted mainly at Qwen3 dense-series models (such as Qwen3-8B/14B/32B); generalization to other model series is not guaranteed.
- The algorithm implementation involves a training process and has certain NPU memory requirements; only devices with NPU memory ≥64 GB are supported.

---

## 3. Related Terms

- [Adapt Rotation](../adapt_rotation/term_adapt_rotation.md): a prerequisite term; LAOS uses its rotation optimization as the outlier suppression step.
- [AutoRound](../autoround/term_autoround.md): a companion term; LAOS uses its rounding training to complete the low-bit quantization.
- [QuaRot](../quarot/term_quarot.md): the parent concept; Adapt Rotation is based on the QuaRot framework.
- [DualScale](../dual_scale/term_dual_scale.md): a similar algorithm and another W4A4 quantization solution for the Qwen3 dense series.
- [DAOS](../daos/term_daos.md): a similar algorithm and a combined low-bit solution for multimodal generation (OASQ + TLQ).

---

## 4. Reference Documents

1. [LAOS Parameter Configuration Workflow Guide](./usage_laos.md)
