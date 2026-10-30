# Adapt Rotation Adaptive Rotation Optimization Algorithm Terminology Entry

> **Entry Category**: [Outlier Suppression Algorithms](../README.md#1-outlier-suppression-algorithms)<br>
> **English Name**: adapt_rotation<br>
> **Application Domain**: Large language model quantization and compression, low-bit quantization accuracy optimization<br>

---

## 1. Overview

Adapt Rotation (adaptive rotation optimization) is an outlier suppression algorithm for low-bit quantization and a data-driven extension of [QuaRot](../quarot/term_quarot.md). It uses calibration data to optimize the orthogonal rotation so that the activation distribution becomes more balanced and the quantization reconstruction error smaller; its core characteristics are preserving linear-transform equivalence, learnable rotation parameters, and splitting rotation learning from application into separate stages.

---

## 2. Term Description

[QuaRot](../quarot/term_quarot.md) applies a fixed Hadamard matrix to perform an orthogonal rotation on weights and activations so as to balance the numerical range of each channel, but a fixed Hadamard matrix is not necessarily the best match for the activation distribution of a particular model. Adapt Rotation observes that if the rotation matrix can be iteratively optimized against calibration data, the transformed activations yield a smaller reconstruction error after quantization-dequantization. It therefore introduces data-driven rotation optimization on top of QuaRot, offering a better answer to the mismatch between a fixed rotation matrix and the target model's activation distribution.

### 2.1 Core Idea

The core idea of Adapt Rotation is to "optimize the orthogonal rotation in a data-driven way": given an initial Hadamard matrix $H$ and calibration activation data, iteratively learn an optimizable orthogonal matrix $R$ so that the transformed rotation matrix $H_{\text{adapted}} = H \cdot R$ minimizes the quantization-dequantization reconstruction error on the given activation data. Because $R$ is orthogonal, the model computation is equivalent before and after the transformation.

Its effectiveness comes from "changing the coordinate peaks without losing vector energy".

### 2.2 Working Mechanism

A fixed Hadamard rotation mixes coordinates uniformly, but it only exploits the matrix structure and does not know in which directions the real model activations are hardest to quantize. Adapt Rotation first uses Hadamard as a stable initial orthogonal basis, then repeatedly uses the calibration activations to construct the correspondence between "the current rotated floating-point activations" and "their quantization-dequantization result". If the two are denoted $A$ and $B$, the algorithm seeks an orthogonal update that brings $A$ as close to $B$ as possible after rotation.

With the current quantization result held fixed, this step can be written as an orthogonal Procrustes-type problem whose solution is related to the orthogonal factor of the polar decomposition of $A^TB$. The algorithm obtains that orthogonal factor iteratively, accumulates it into the existing rotation, and recomputes the quantization result, alternating in this way. Throughout, the rotation stays orthogonal, so what changes is how energy is distributed across the coordinates, not the $L_2$ norm of the vectors or the information a linear transform can express.

### 2.3 Mathematical Description

Let the initial Hadamard matrix be $H$ and the learned orthogonal rotation be $R$; the transformed rotation matrix is then:

$$
H_{\text{adapted}} = H \cdot R
$$

Here $R$ is an orthogonal matrix (satisfying $R^T R = I$), obtained by applying Newton-Schulz iteration to $A^T B$ to find the orthogonal polar factor, with the accumulated rotation written as $R_{\text{acc}} = R_{\text{acc}} \cdot R_{\text{step}}$. The optimization objective is to minimize the reconstruction loss after the transformed activations undergo per-token symmetric quantization-dequantization.

- $H$: initial Hadamard matrix
- $R$: the orthogonal rotation matrix learned by iteration
- $H_{\text{adapted}}$: the optimized rotation matrix, which remains orthogonal and preserves computational equivalence
- $A$, $B$: the matrices used in the Newton-Schulz iteration to solve for the orthogonal polar factor
- $R_{\text{acc}}$: the accumulated rotation matrix

If the currently rotated floating-point activations are denoted $A=XQ$ and the corresponding quantization-dequantization result is denoted $B=\mathcal{Q}(A)$, then with $B$ fixed we may consider:

$$
Q^{*}=\arg\min_{Q^TQ=I}\|XQ-B\|_F^2.
$$

This problem is equivalent to maximizing $\operatorname{tr}(Q^TX^TB)$. If $X^TB=U\Sigma V^T$, the classical orthogonal Procrustes solution is $Q^*=UV^T$. In practice the iteration can approximate or update this solution using the orthogonal factor of the polar decomposition. After $Q$ is obtained again, $B=\mathcal{Q}(XQ)$ must be recomputed, because the quantization projection changes with the coordinate system.

### 2.4 Key Properties

- **Computational equivalence**: orthogonal rotation does not change the mathematical result of matrix multiplication and introduces no additional inference error.
- **Data-driven optimization**: the rotation matrix is iteratively optimized against calibration data rather than being a fixed Hadamard matrix.
- **Two-stage flow**: Stage 1 optimizes the rotation matrix; Stage 2 applies the optimized result to the QuaRot pipeline.
- **Calibration-distribution dependent**: the optimized rotation targets the calibration activations and quantization setting actually used; the optimal rotation may change when the data or bit width changes.

In terms of error and applicability boundaries, the main error risks are calibration overfitting and quantization-target mismatch: if the calibration data cannot cover real inputs, the learned rotation may be valid only for a local sample; if the bit width/granularity used during optimization differs from the final quantization scheme, the optimal direction may also change.

### 2.5 Use Cases

- Scenarios that need better outlier suppression than a fixed Hadamard rotation and involve low-bit quantization.
- Serves as a preceding outlier suppression step for low-bit quantization algorithms such as [AutoRound](../autoround/term_autoround.md).

### 2.6 Usage Restrictions

- The two stages must use the same learned rotation matrix consecutively: the first stage optimizes it, and the second stage applies it to an equivalent rotation transform.
- The rotation optimization objective should match the actual bit width used later for activation quantization; when the target bit width changes, the rotation matrix should be re-optimized.
- In models with many non-shared expert layers, such as MoE models, widening the rotation optimization scope significantly increases activation statistics and iterative optimization overhead.

These restrictions reflect the algorithm's dependence on model structure and equivalence transformation conditions. If the target model does not satisfy the corresponding structural assumptions, applying the algorithm indiscriminately may break the original computational relationships. Therefore, when encountering incompatible structures, first narrow the scope or use a recipe already verified for the model, rather than blindly increasing the smoothing strength.

---

## 3. Related Terms

- [QuaRot](../quarot/term_quarot.md): the parent concept; this algorithm optimizes the rotation matrix on top of QuaRot.
- [SmoothQuant](../smooth_quant/term_smooth_quant.md): a comparison algorithm that suppresses outliers by channel-level scaling rather than orthogonal rotation.
- [AutoRound](../autoround/term_autoround.md): a companion term; AutoRound weight quantization is often applied after this algorithm.
- [MinMax](../minmax/term_minmax.md): the application target; rotated activations are easier to quantize with MinMax.

---

## 4. Reference Documents

1. [Adapt Rotation Parameter Configuration Workflow Guide](./usage_adapt_rotation.md)
