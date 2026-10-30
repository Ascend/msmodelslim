# QuaRot Rotation Quantization Algorithm Terminology Entry

> **Entry Category**: [Outlier Suppression Algorithms](../README.md#1-outlier-suppression-algorithms)<br>
> **English Name**: quarot<br>
> **Application Domain**: Large language model quantization and compression, low-bit quantization accuracy optimization<br>
> **First Proposed**: Ashkboos et al., NeurIPS 2024

---

## 1. Overview

QuaRot (Quantization with Rotation) is a method that suppresses quantization outliers through an orthogonal rotation. It uses orthogonal matrices such as Hadamard matrices to spread the large magnitudes concentrated in a few channels across more dimensions, reducing the imbalance of the dynamic range between channels while preserving the linear-computation equivalence; its core characteristics are the orthogonal equivalent transform, outlier spreading, and decoupling from the subsequent low-bit quantization.

---

## 2. Term Description

Outliers exist in the activations of large language models; when quantized directly, an extreme outlier in a single channel forces the scaling factor to be too large and degrades the overall quantization accuracy. QuaRot observes that an orthogonal rotation can redistribute the parameters and balance the numerical range of each channel and that, because an orthogonal matrix satisfies $QQ^T = I$, the model is mathematically equivalent before and after the transform. It can therefore significantly smooth the activation distribution without changing model behavior, laying the groundwork for low-bit quantization.

### 2.1 Core Idea

The core idea of QuaRot is to "balance the channel distribution with an orthogonal rotation": apply a left rotation to the weight matrix $W$ and a right rotation to the activation matrix $X$, so that the maximum of the rotated activations is as uniform as possible across channels. Because the rotation is an invertible orthogonal transform, $X' W' = XW$ holds strictly and the model computation is equivalent before and after the transform.

The intuition for why Hadamard-type rotations work on outliers is that a spike concentrated on a single coordinate, after an approximately uniform orthogonal mixing, is spread across many coordinates with coefficients of the order of $1/\sqrt d$ (the exact peak is affected by sign cancellation and the input structure).

### 2.2 Working Mechanism

QuaRot exploits two properties of the orthogonal matrix $Q$: first, $Q^TQ=I$, so inserting $Q$ and $Q^T$ between adjacent linear transforms preserves the original function; second, an orthogonal transform preserves $L_2$ energy but can change how that energy is distributed across coordinates. A Hadamard/random-sign rotation combines a large magnitude concentrated on a few coordinates into more coordinates, so the per-channel peak usually decreases while the total energy is unchanged.

In a Transformer, some rotations can be absorbed offline into adjacent weights without producing an extra floating-point model difference; other subspaces that cannot be fully fused can be implemented at runtime with an efficient Hadamard structure. Arranging left and right rotations appropriately for the residual stream, attention projections, and MLP can weaken outlier channels at several quantization-sensitive positions simultaneously. Its difference from SmoothQuant is that SmoothQuant migrates magnitude between activations and weights, whereas QuaRot mainly redistributes energy within the same vector space.

### 2.3 Mathematical Description

Let the orthogonal rotation matrix be $Q$ (satisfying $Q^T Q = I$); transform the weights and activations respectively:

$$
W' = Q^T W, \quad X' = X Q
$$

The transform preserves computational equivalence:

$$
X' W' = (X Q)(Q^T W) = X W
$$

- $W$: weight matrix
- $X$: activation matrix
- $Q$: the orthogonal rotation matrix (such as a Hadamard matrix)
- $I$: the identity matrix

For layers containing RMSNorm, since $RMSNorm(X) = RMSNorm(X Q^T) \cdot Q$, the computational invariance still holds. When a block-diagonal rotation is used, $Q$ consists of several smaller orthogonal blocks; larger blocks mix channels more thoroughly, but the rotation computation and fusion constraints are also greater.

For a random-sign Hadamard $Q=HD$ ($D$ being a random $\pm1$ diagonal matrix), each rotated coordinate of a given vector $x$ is a sum of many signed components. Intuitively, provided the input is not highly coherent:

$$
\|xQ\|_2=\|x\|_2,\qquad \|xQ\|_\infty\ \text{is usually significantly smaller than}\ \|x\|_\infty.
$$

The quantizer often cares precisely about the second quantity, since many scales are determined by the maximum absolute value. The exact amount of reduction depends on the input structure, and $1/\sqrt d$ cannot be treated as a guaranteed bound that every vector necessarily attains.

### 2.4 Key Properties

- **Computational equivalence**: orthogonal rotation does not change the mathematical result of matrix multiplication and introduces no additional inference error.
- **Outlier spreading**: the rotation spreads outliers across multiple channels and balances the per-channel maxima.
- **Inter-layer invariance**: the scaling invariance of RMSNorm guarantees that the model output is unchanged after inter-layer rotation.
- **Block-diagonalizable**: the orthogonal block size controls the mixing range, trading off between outlier-spreading capability and computational cost.
- **Energy preserved, peaks rearranged**: the orthogonal rotation preserves the $L_2$ norm but changes the per-coordinate maxima, thereby changing the low-bit quantization error.

The limitation is that some structures cannot absorb the rotation for free and require an online transform; a block-diagonal rotation is cheaper to compute but has a smaller mixing range and may weaken outlier spreading. Incomplete model-structure adaptation also breaks equivalence.

### 2.5 Use Cases

- Large language model scenarios where the activations have significant outliers and low-bit (such as W4A4) quantization is required.
- Serves as a preceding outlier suppression step for low-bit quantization algorithms such as [AutoRound](../autoround/term_autoround.md).

### 2.6 Usage Restrictions

- When the online rotation is used together with tensor parallelism, the rotation block partitioning must be compatible with the parallel sharding; common implementations require the parallelism degree to satisfy a power-of-two structural constraint.
- The online rotation requires extra rotation operations in the inference path and may therefore increase runtime overhead; weigh the outlier suppression benefit against inference performance.

These restrictions reflect the algorithm's dependence on model structure and equivalence transformation conditions. If the target model does not satisfy the corresponding structural assumptions, applying the algorithm indiscriminately may break the original computational relationships. Therefore, when encountering incompatible structures, first narrow the scope or use a recipe already verified for the model, rather than blindly increasing the smoothing strength.

---

## 3. Related Terms

- [Adapt Rotation](../adapt_rotation/term_adapt_rotation.md): a subordinate concept that optimizes the rotation matrix in a data-driven way on top of QuaRot.
- [SmoothQuant](../smooth_quant/term_smooth_quant.md): a comparison algorithm that suppresses outliers by channel-level scaling rather than orthogonal rotation.
- [AutoRound](../autoround/term_autoround.md): a companion term; AutoRound weight quantization is often applied after this algorithm.
- [KVCache Quant](../kvcache_quant/term_kvcache_quant.md): a companion term that can be combined with this algorithm for long-sequence inference.
- [MinMax](../minmax/term_minmax.md): the application target; rotated activations are easier to quantize with MinMax.

---

## 4. Reference Documents

1. Ashkboos S et al. QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs. NeurIPS 2024. <https://arxiv.org/abs/2404.00456>
2. [QuaRot Parameter Configuration Workflow Guide](./usage_quarot.md)
