# Flex AWQ SSZ Flexible Activation-Aware Weight Quantization Smoothing Algorithm Terminology Entry

> **Entry Category**: [Outlier Suppression Algorithms](../README.md#1-outlier-suppression-algorithms)<br>
> **English Name**: flex_awq_ssz<br>
> **Application Domain**: Large language model quantization and compression, low-bit quantization accuracy optimization<br>

---

## 1. Overview

Flex AWQ SSZ is an outlier suppression algorithm that combines activation-aware scaling with low-bit weight quantization error evaluation. It constructs channel scaling from activation statistics and uses a real quantization process such as [SSZ](../ssz/term_ssz.md) to compare candidate parameters, thereby searching for the scaling factor that better suits the target low-bit configuration; its core characteristics are quantizer-aware evaluation, automatic parameter search, and adaptation to low-bit scenarios such as W4.

---

## 2. Term Description

[Flex Smooth Quant](../flex_smooth_quant/term_flex_smooth_quant.md) uses simulated quantization to evaluate parameters and cannot accurately reflect the effect after real quantization. Flex AWQ SSZ instead uses the real quantizer to evaluate the quantization error under different parameters and, for low-bit quantization scenarios, uses the activation mean (rather than the maximum) to compute the activation scale, thereby obtaining more accurate parameter choices under different quantization configurations.

### 2.1 Core Idea

The core idea of Flex AWQ SSZ is to "first perform activation-aware equivalent scaling, then use the target weight quantizer to judge directly whether the scaling is worth keeping". A positive scaling vector $s$ is applied to the input channels, turning the activations into $X'=X/s$ and the weights into $W'=W\cdot s$, so the linear computation $XW^T$ is preserved in the floating-point domain. Unlike smoothing methods that rely only on analytical statistics, Flex AWQ SSZ passes each candidate scaling through the target quantizer and then selects $\alpha$ by the reconstruction error of the quantized linear output relative to the floating-point output.

The method searches only the activation-side exponent $\alpha$ and fixes $\beta=0$. In candidate scale evaluation, the activation statistic is the per-input-channel $\max(|X|)$; a larger $\alpha$ tends to suppress the activation channel peaks while migrating the corresponding magnitude to the weight side. This preserves the AWQ-family intuition that "important channels need protection" while leaving the final judgment criterion to the weight quantization scheme actually used.

It is effective because the parameter search objective is consistent with the final weight quantizer.

### 2.2 Working Mechanism

For each candidate $\alpha$, the per-input-channel peak of the calibration activations first gives $s_j=\max|X_j|^{\alpha}$, after which $X'_j=X_j/s_j$ and $W'_j=W_j s_j$ are applied. Because $\beta=0$, the scale no longer explicitly divides by a weight statistic, reducing the search space from two dimensions to one; $\alpha=0$ means no migration, and a larger $\alpha$ progressively strengthens the compression of activation peaks.

The key point is that candidate scales are not scored by a proxy formula; instead, the scaled weights are handed to the target low-bit quantizer to obtain the actual quantized linear output, which is then compared with the floating-point output by normalized RMSE. In this way, the scale optimization, rounding, and grouping characteristics of SSZ or another target weight quantizer all enter the selection criterion for $\alpha$, forming a closed loop between the smoothing parameters and the final weight quantization error.

### 2.3 Mathematical Description

Let the linear layer input be $X$ and the weight be $W$, with per-input-channel scale $s$. Flex AWQ SSZ uses reciprocal scaling:

$$
X' = X \operatorname{diag}(s)^{-1}, \qquad W' = W\operatorname{diag}(s)
$$

The floating-point output is therefore unchanged:

$$
X'W'^T = XW^T
$$

In candidate parameter evaluation, the scale is constructed from the per-input-channel activation peak:

$$
s_j = \operatorname{clamp}\left(A_j^{\alpha},\; \min=10^{-5}\right),
\qquad A_j = \max |X_j|, \qquad \beta=0
$$

- $A_j$: the maximum absolute value of the $j$-th input channel over the calibration activations.
- $\alpha$: the activation migration strength; the search interval is usually $[0,1]$.
- $\beta$: fixed to $0$, meaning the scale does not explicitly use the weight term $W_j^{-\beta}$.
- $s_j$: the final equivalent scale applied to both the activations and the weights.

The objective for a candidate $\alpha$ is not to minimize $\|W-W_q\|$ directly but to make the linear output produced by the target quantizer as close as possible to the floating-point output. Expressed as normalized RMSE:

$$
E(\alpha)=
\frac{\sqrt{\operatorname{mean}\left((Y_q(\alpha)-Y_{fp})^2\right)}}
{\sqrt{\operatorname{mean}(Y_{fp}^2)}}
$$

Finally, the candidate with the smallest $E(\alpha)$ is selected.

More completely, the search can be written as:

$$
\alpha^*=\arg\min_{\alpha\in\mathcal A}\;\frac{\|XW^T-(XD_\alpha^{-1})\,\mathcal Q_{\theta^*(\alpha)}(WD_\alpha)^T\|_F}{\|XW^T\|_F},
$$

where $D_\alpha=\operatorname{diag}(s(\alpha))$, and the inner quantization parameters satisfy:

$$
\theta^*(\alpha)=\arg\min_{\theta}\|WD_\alpha-\hat W(\theta)\|^2.
$$

These two expressions capture the logic of "first fix the scaling, then let the target quantizer reach its own favorable state, and finally compare the scaling candidates by the real output error".

### 2.4 Key Properties

- **Real-quantizer evaluation**: uses the target weight quantizer to evaluate the reconstruction error of candidate scalings directly.
- **Simplified parameter space**: fixes $\beta=0$ and searches only $\alpha$, reducing the search complexity.
- **One-dimensional parameter search**: only different $\alpha$ candidates need to be compared, a smaller search space than optimizing both exponents at once.
- **Computational equivalence**: coordinated scaling preserves overall computational equivalence and does not change the model output.
- **Coupled with the target quantizer**: the best $\alpha$ varies with the bit width, group size, and target weight quantizer.

The risk is that a one-dimensional $\alpha$ search has limited expressive power, and the optimum depends strongly on the calibration data and the target quantization scheme; after changing the bit width, group size, or quantizer, the previous optimal $\alpha$ may no longer hold.

### 2.5 Use Cases

- Low-bit quantization scenarios that require real-quantizer evaluation of smoothing parameters and pursue higher accuracy.
- Scenarios that require simultaneous outlier suppression for multiple structures such as the attention `ov`, MLP `up-down`, and consecutive linear layers.

### 2.6 Usage Restrictions

- The target subgraph must have a structural relationship that permits equivalent cooperative scaling, such as consecutive linear layers, attention O/V projections, or MLP up/down projections.
- Candidate search should use the same activation/weight quantization as the final target; the weight side is often combined with [SSZ](../ssz/term_ssz.md).
- The automatic search performs candidate quantization and error comparison many times, trading a higher calibration time for accuracy gain.

These restrictions reflect the algorithm's dependence on model structure and equivalence transformation conditions. If the target model does not satisfy the corresponding structural assumptions, applying the algorithm indiscriminately may break the original computational relationships. Therefore, when encountering incompatible structures, first narrow the scope or use a recipe already verified for the model, rather than blindly increasing the smoothing strength.

---

## 3. Related Terms

- [AWQ](../awq_smooth/term_awq_smooth.md): the parent concept; this algorithm incorporates AWQ's activation-aware idea.
- [Flex Smooth Quant](../flex_smooth_quant/term_flex_smooth_quant.md): a similar algorithm that evaluates parameters with simulated quantization.
- [SSZ](../ssz/term_ssz.md): a companion term; the weight side of this algorithm is usually combined with the SSZ weight quantization method.
- [SmoothQuant](../smooth_quant/term_smooth_quant.md): the parent concept; this algorithm belongs to the smoothing family of outlier suppression algorithms.
- [AutoRound](../autoround/term_autoround.md): a companion term, often used together with this algorithm before low-bit quantization.

---

## 4. Reference Documents

1. Lin J et al. AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration. MLSys 2024. <https://arxiv.org/abs/2306.00978>
2. [Flex AWQ SSZ Parameter Configuration Workflow Guide](./usage_flex_awq_ssz.md)
