# SmoothQuant Smooth Quantization Algorithm Terminology Entry

> **Entry Category**: [Outlier Suppression Algorithms](../README.md#1-outlier-suppression-algorithms)<br>
> **English Name**: smooth_quant<br>
> **Application Domain**: Large language model quantization and compression, inference acceleration<br>
> **First Proposed**: Xiao et al., ICML 2023

---

## 1. Overview

SmoothQuant is an activation outlier suppression algorithm applied before quantization. It redistributes channel scales between adjacent operators that permit equivalent transformation, migrating part of the activation dynamic range to the weight side so that the activations become easier to represent at low bit width while keeping the original model computation equivalent as far as possible; its core characteristics are per-channel scaling, cooperative activation-weight balancing, and adjustable migration strength, and it forms a different technical route from rotation-based methods such as [QuaRot](../quarot/term_quarot.md).

---

## 2. Term Description

Outliers are common in the activations of large language models: the values of a few channels are far larger than the others, and quantizing directly inflates the quantization range in order to cover those outliers, leaving most channels with insufficient effective bits and severely degrading accuracy. SmoothQuant observes that cooperatively scaling the activations and the weights keeps the linear layer output mathematically unchanged while migrating the activation outliers into the weights, thereby reducing the quantization difficulty of the activations without changing the model output.

### 2.1 Core Idea

The core idea of SmoothQuant is to "migrate" the persistent outliers in the activations to the weight side, which is usually easier to quantize, through a reversible per-channel reparameterization. Let the input channel scale be $s_j>0$; divide the $j$-th activation channel by $s_j$ and multiply the corresponding input channel of the weight by $s_j$. The floating-point matrix multiplication result is then unchanged, but the dynamic range is redistributed between the two sides.

This migration targets the phenomenon common in large model activations where "a few channels have persistently large magnitudes". If activations are quantized at a coarser granularity such as per-tensor or per-token, one extreme channel raises the step size for the whole group, leaving many ordinary channels with very few effective code points; SmoothQuant lowers the peaks of those activation channels so that the bulk activations obtain a finer quantization resolution, and uses the finer per-channel/per-group quantization on the weight side to absorb the increased magnitude.

The key to SmoothQuant's effectiveness is the asymmetry in the "load-bearing capacity" of activation and weight quantization.

### 2.2 Working Mechanism

Let $D=\operatorname{diag}(s)$ and apply $\hat X=XD^{-1}$ and $\hat W=DW$ to the linear layer; then $\hat X\hat W=XW$. The scale is generally constructed jointly from the per-channel activation peak $A_j$ and the peak $W_j$ of the weight's corresponding input channel, for example $s_j=A_j^{\alpha}/W_j^{1-\alpha}$. As $\alpha$ grows, more activation magnitude is migrated to the weights; as $\alpha$ shrinks, the weights change less but more activation outliers are retained.

The smoothed scale can usually be fused offline into the normalization layer parameters or the adjacent linear layer weights, so ideally no extra elementwise scaling needs to be executed in the inference graph. What genuinely needs to be traded off is the quantization error: lowering the activation peaks shrinks the activation step size, but channels whose weights were amplified may become harder to quantize. The role of $\alpha$ is to choose a balance point between the errors on the two sides, rather than "the larger the better".

### 2.3 Mathematical Description

Let the linear layer input be $X$ and the weight be $W$, with the per-input-channel positive scales forming the diagonal matrix $D=\operatorname{diag}(s)$; then:

$$
Y=XW^T=(XD^{-1})(WD)^T=\hat X\hat W^T
$$

That is:

$$
\hat X_j=\frac{X_j}{s_j}, \qquad \hat W_{:,j}=W_{:,j}s_j
$$

A common form of the SmoothQuant scale is:

$$
s_j=\operatorname{clamp}\left(
\frac{A_j^{\alpha}}{W_j^{1-\alpha}},\;\min=10^{-5}
\right)
$$

where:

- $A_j=\max |X_j|$: the calibration peak of the $j$-th activation channel.
- $W_j=\max |W_{:,j}|$: the peak of the weight's corresponding input channel.
- $\alpha\in[0,1]$: the migration balance parameter between the activation and weight sides.
- $s_j$: the per-channel scaling applied reciprocally to the activations and the weights.

The larger $\alpha$ is, the more strongly $A_j$ influences the scale, so the activation channel is divided by a larger $s_j$; at the same time the corresponding weight channel is multiplied by the same $s_j$.

If the post-smoothing quantization errors are $E_X=\mathcal Q(XD^{-1})-XD^{-1}$ and $E_W=\mathcal Q(WD)-WD$, then:

$$
Y_q-Y\approx E_X(WD)^T+(XD^{-1})E_W^T+E_XE_W^T.
$$

This shows that the ideal equivalent transform only guarantees a zero error "before quantization"; the real goal is to make the sum of the quantization errors on the two sides and their cross term as small as possible. Increasing $\alpha$ usually reduces $E_X$ but may increase $E_W$, so the optimum depends on the bit widths, granularity, and data distributions on both sides.

### 2.4 Key Properties

- **Computational equivalence**: $Y = XW$ is strictly unchanged before and after the transform, introducing no extra inference error.
- **Outlier migration**: the outliers in the activations are migrated to the weights; although the weights become harder to quantize, they can be handled by weight quantization algorithms such as [SSZ](../ssz/term_ssz.md) and [GPTQ](../gptq/term_gptq.md).
- **Per-channel granularity**: the scaling factor is computed independently per channel, adapting to dynamic-range differences between channels.
- **Adjustable migration strength**: the balance exponent controls the proportion of the dynamic range borne by the activation and weight sides.
- **Magnitude migration rather than deletion**: the outlier magnitude is transferred from the activation side to the weight side, and floating-point information is not clipped away by the smoothing itself.

In terms of error and applicability boundaries, the error does not disappear; it is only redistributed. When $\alpha$ is too large the weight channels are over-amplified and the weight quantization error rises; when it is too small the activation outlier problem remains.

### 2.5 Use Cases

- Large language model quantization scenarios where the activations have significant outliers and direct quantization loses considerable accuracy.
- Serves as a preceding outlier suppression step for quantization schemes such as W8A8 and W4A8, creating a more uniform numerical distribution for activation quantizers such as [MinMax](../minmax/term_minmax.md).

### 2.6 Usage Restrictions

- Classical SmoothQuant assumes that the per-channel scale can be fused between the normalization/pre-scaling and the following linear layer, which makes it best suited to `norm-linear`-type structures.
- For other structures such as `ov`, `up-down`, and consecutive linear layers, an extended smoothing method that preserves the corresponding computational equivalence must be used.

These restrictions reflect the algorithm's dependence on model structure and equivalence transformation conditions. If the target model does not satisfy the corresponding structural assumptions, applying the algorithm indiscriminately may break the original computational relationships. Therefore, when encountering incompatible structures, first narrow the scope or use a recipe already verified for the model, rather than blindly increasing the smoothing strength.

---

## 3. Related Terms

- [Iterative Smooth](../iterative_smooth/term_iterative_smooth.md): a similar algorithm and an iterative extension of this one that supports more subgraph types.
- [Flex Smooth Quant](../flex_smooth_quant/term_flex_smooth_quant.md): a similar algorithm that automatically finds the optimal `alpha`/`beta` through a two-stage grid search.
- [OASQ](../oasq/term_oasq.md): a similar algorithm and a smoothing extension that splits the scale by channel outlier status.
- [AWQ](../awq_smooth/term_awq_smooth.md): a similar algorithm that identifies important channels based on the activation mean and searches for the scaling factor.
- [QuaRot](../quarot/term_quarot.md): a comparison algorithm that suppresses outliers by orthogonal rotation rather than channel scaling.
- [MinMax](../minmax/term_minmax.md): the application target; the smoothed activations are easier to quantize with MinMax.

---

## 4. Reference Documents

1. Xiao G et al. SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models. ICML 2023. <https://arxiv.org/abs/2211.10438>
2. [SmoothQuant Parameter Configuration Workflow Guide](./usage_smooth_quant.md)
