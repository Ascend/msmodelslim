# Flex Smooth Quant Flexible Smooth Quantization Algorithm Terminology Entry

> **Entry Category**: [Outlier Suppression Algorithms](../README.md#1-outlier-suppression-algorithms)<br>
> **English Name**: flex_smooth_quant<br>
> **Application Domain**: Large language model quantization and compression, inference acceleration<br>

---

## 1. Overview

Flex Smooth Quant is a searchable extension of [SmoothQuant](../smooth_quant/term_smooth_quant.md) that automatically finds a better scaling balance between activations and weights. It searches parameters such as `alpha` and `beta` in stages to adapt to different subgraph structures and data distributions, reducing the quantization error after smoothing; its core characteristics are multi-subgraph support, automatic parameter search, and configurable fused/non-fused smoothing.

---

## 2. Term Description

Conventional [SmoothQuant](../smooth_quant/term_smooth_quant.md) uses a fixed `alpha` and supports only the `norm-linear` subgraph, which adapts poorly to complex structures. Flex Smooth Quant generalizes the scaling formula so that activations and weights use the two exponents `alpha` and `beta` respectively, and finds the optimal parameters automatically through a grid search, avoiding manual tuning while extending support to more subgraph types.

### 2.1 Core Idea

The core idea of Flex Smooth Quant is to extend smooth scaling from a constrained single-exponent relationship to two relatively independent exponents, $\alpha$ and $\beta$, so that the influence of activation peaks and weight peaks on the final scale can be adjusted separately. The algorithm searches candidates in two stages for the combination with the smaller joint quantization output error, thereby adapting to the differing degrees of activation/weight outlier severity across layers and computational subgraphs.

The two-exponent degrees of freedom can handle layers in which the "degree of activation outlier" and the "degree of weight outlier" do not satisfy a fixed complementary relationship.

### 2.2 Working Mechanism

Flex Smooth Quant separates the mutually bound exponent relationship of standard SmoothQuant. The scale can be written as $s_j=A_j^{\alpha}W_j^{-\beta}$: $\alpha$ controls how strongly activation peaks push the scale, and $\beta$ controls how strongly weight peaks suppress it. This allows the balance of "better activation quantization without unduly worsening the weights" to be sought over a larger two-dimensional region instead of being confined to the single curve $\beta=1-\alpha$.

The automatic search usually first follows the constrained path $\beta=1-\alpha$ to find a stable starting point, then fixes the better $\alpha$ and searches $\beta$. Each candidate first performs the equivalent scaling, then simulates quantization on the scaled activations and weights separately, and finally compares the normalized error of the linear output. The two-stage search reduces the cost of a direct two-dimensional exhaustive search and lets the second stage relax the weight-side constraint once a reasonable activation migration strength has been found.

### 2.3 Mathematical Description

Scaling factor formula:

$$
s = \operatorname{clamp}\left( \frac{A_{\text{scale}}^{\alpha}}{W_{\text{scale}}^{\beta}}, \min=10^{-5} \right)
$$

- $s$: per-channel scale factor
- $A_{\text{scale}}$: the per-channel scaling statistic of the activation values
- $W_{\text{scale}}$: the per-column maximum of the weights
- $\alpha$: the activation scaling coefficient, controlling how strongly the activations influence the scaling factor ($0$–$1$)
- $\beta$: the weight scaling coefficient, controlling how strongly the weights influence the scaling factor ($0$–$1$)
- $10^{-5}$: the minimum value of the scaling factor, preventing numerical instability

The algorithm can compare the joint post-quantization output error over the $\alpha$ and $\beta$ candidate space to determine the smoothing strength in a data-driven way.

Let the per-channel activation peak be $A_j$ and the weight peak be $W_j$; the generalized scale can be written as:

$$
s_j=\frac{A_j^{\alpha}}{W_j^{\beta}},\qquad X'_j=X_j/s_j,\qquad W'_{:,j}=W_{:,j}s_j.
$$

The floating-point domain still satisfies $X'W'^T=XW^T$. If both activations and weights are quantized, the output error approximately contains:

$$
\Delta Y\approx E_XW'^T+X'E_W^T+E_XE_W^T,
$$

where $E_X=\mathcal Q(X')-X'$ and $E_W=\mathcal Q(W')-W'$. This shows that the errors on both sides not only exist individually but also produce cross terms, so a joint search is more meaningful than looking only at the tensor error on one side.

### 2.4 Key Properties

- **Two-exponent scaling**: activation and weight statistics are controlled by $\alpha$ and $\beta$ respectively, giving the scale more degrees of freedom.
- **Two-stage search**: first search a stable starting point along the constrained relationship, then relax the other exponent to enlarge the space of possible solutions.
- **Extensible to graph structures**: as long as the scaling can be equivalently absorbed between adjacent operators, the same smoothing idea can be applied to different linear subgraphs.
- **Standalone-layer smoothing**: besides fusable adjacent subgraphs, input-side scaling can also be applied to standalone linear layers that lack a preceding fusion source.
- **Computational equivalence**: coordinated scaling preserves overall computational equivalence and does not change the model output.
- **Two-exponent degrees of freedom**: the influence of the activation side and the weight side on the scale can be adjusted independently, free from the fixed relation $\beta=1-\alpha$.

But more degrees of freedom also mean the search relies more on the data. Too large an $\alpha$ smooths the activations but amplifies the weights, while too large a $\beta$ may compress the scale so much that the activation magnitude grows instead.

### 2.5 Use Cases

- Quantization scenarios that require automatically searching the optimal smoothing parameters and reducing manual tuning.
- Scenarios that require simultaneous outlier suppression for multiple structures such as the attention `ov`, MLP `up-down`, and consecutive linear layers.

### 2.6 Usage Restrictions

- The target structure needs to be recognizable as a subgraph that preserves computational equivalence, such as `norm-linear`, `linear-linear`, `ov`, or `up-down`.
- For standalone linear layers that cannot be fused with a preceding operator, only multiplicative scaling is appropriate; a simultaneous bias shift is not suitable.

These restrictions reflect the algorithm's dependence on model structure and equivalence transformation conditions. If the target model does not satisfy the corresponding structural assumptions, applying the algorithm indiscriminately may break the original computational relationships. Therefore, when encountering incompatible structures, first narrow the scope or use a recipe already verified for the model, rather than blindly increasing the smoothing strength.

---

## 3. Related Terms

- [SmoothQuant](../smooth_quant/term_smooth_quant.md): the parent concept; this algorithm is a flexible extension of SmoothQuant.
- [Iterative Smooth](../iterative_smooth/term_iterative_smooth.md): a similar algorithm that likewise supports smoothing across multiple subgraph types.
- [OASQ](../oasq/term_oasq.md): a similar algorithm that splits the scale by channel outlier status rather than searching alpha/beta.
- [Flex AWQ SSZ](../flex_awq_ssz/term_flex_awq_ssz.md): a similar algorithm that evaluates parameters with a real quantizer and fixes `beta` to `0`.
- [AWQ](../awq_smooth/term_awq_smooth.md): a comparison algorithm that identifies important channels based on the activation mean and searches for the scaling factor.
- [AutoRound](../autoround/term_autoround.md): a companion term, often used together with this algorithm before low-bit quantization.

---

## 4. Reference Documents

1. Xiao G et al. SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models. ICML 2023. <https://arxiv.org/abs/2211.10438>
2. [Flex Smooth Quant Parameter Configuration Workflow Guide](./usage_flex_smooth_quant.md)
