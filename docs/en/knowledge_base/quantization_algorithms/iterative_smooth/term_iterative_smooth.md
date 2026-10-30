# Iterative Smooth Iterative Smoothing Algorithm Terminology Entry

> **Entry Category**: [Outlier Suppression Algorithms](../README.md#1-outlier-suppression-algorithms)<br>
> **English Name**: iterative_smooth<br>
> **Application Domain**: Large language model quantization and compression, inference acceleration<br>

---

## 1. Overview

Iterative Smooth is an outlier suppression algorithm used for distribution shaping before quantization. It repeatedly adjusts the scaling relationship between adjacent operators that can be transformed equivalently, so that the dynamic ranges of activations and weights gradually reach a balance better suited to quantization; its core characteristics are iterative scaling, adaptation to multiple subgraph types, and continuous correction of local structures, and it can be regarded as an iterative extension of the [SmoothQuant](../smooth_quant/term_smooth_quant.md) idea.

---

## 2. Term Description

SmoothQuant supports only the `norm-linear` subgraph and struggles to cover the `ov` structure inside attention, the `up-down` structure of MLP gating, and consecutive linear layers. Iterative Smooth provides a unified scaling framework for these structures and further improves outlier suppression by processing subgraphs in order of priority and redistributing the quantization error between adjacent layers.

### 2.1 Core Idea

The core idea of Iterative Smooth is to extend the SmoothQuant-style equivalent reparameterization — "divide the activation by the scale and let the adjacent weight absorb the scale" — to more computational subgraphs, and to choose, for each connection relationship, a per-channel scale that keeps the computation of all branches equivalent at the same time. It does not simply compress the activations of a single Linear independently; it treats scaling as a transform that can propagate along the computational graph and be absorbed by adjacent weights.

When a source node is consumed by several downstream linear layers at once, the scale must not make only one weight branch easy to quantize, as that may shift the difficulty to the other branches. A compromise scale must therefore be found by combining the channel statistics of the shared input activation and of the multiple target weights; for some asymmetric distributions a shift can also be introduced so that the activation is centered before scaling and compensated equivalently by the subsequent weights/biases.

Its benefit comes from turning a local extremum problem into an equivalent scale-allocation problem on a graph.

### 2.2 Working Mechanism

For a typical adjacent linear transform, one may choose a positive scale $s$ and apply $X'=X/s$, $W'=W\cdot s$; if the scale is absorbed into upstream normalization parameters or into several downstream weights, the floating-point graph can remain equivalent. The key to Iterative Smooth is to decide from the subgraph topology which activation statistics and which weight statistics should jointly compute the scale, rather than looking at an isolated layer alone.

When the same source activation flows to multiple projections, all consumers must share a compatible scale, so the weight statistics must be merged; in structures such as GQA/attention projections, head-sharing relationships must also be respected. For asymmetric activations, a shift term can first move the distribution center to a position more favorable for quantization, with the bias of the following linear layer providing the opposite compensation. In this way both scaling and shifting become reparameterizations on the graph rather than direct modifications of the model function.

### 2.3 Mathematical Description

Let an activation $X$ be used by a set of linear transforms $W^{(1)},\ldots,W^{(k)}$ at the same time. After choosing a per-channel positive scale $s$, this can be written uniformly as:

$$
X' = X\operatorname{diag}(s)^{-1}, \qquad
W'^{(j)} = W^{(j)}\operatorname{diag}(s), \quad j=1,\ldots,k
$$

Each branch then preserves:

$$
X'W'^{(j)T}=XW^{(j)T}
$$

The common scale can still be constructed from the activation statistic $A$ and the aggregated weight statistic $W_s$:

$$
s=\operatorname{clamp}\left(\frac{A^{\alpha}}{W_s^{1-\alpha}},\;\min=s_{min}\right)
$$

- $A$: the per-channel activation magnitude statistic.
- $W_s$: the aggregated magnitude statistic of one or more target weights that share the scale with this activation.
- $\alpha$: controls how strongly the magnitude migrates from the activation side to the weight side.
- $s_{min}$: the numerical lower bound that prevents the scale from degenerating to 0.

If a shift vector $z$ is introduced to handle a clearly asymmetric distribution, a compensation must also be added to the bias of the adjacent affine transform at the same time so that the floating-point function stays consistent.

If the same activation $X$ is consumed by multiple linear branches $W^{(1)},\ldots,W^{(m)}$, introducing $D=\operatorname{diag}(s)$ requires satisfying at the same time:

$$
XW^{(j)T}=(XD^{-1})(W^{(j)}D)^T,\qquad j=1,\ldots,m.
$$

One scale vector therefore couples several consumers. If there are also normalization parameters or biases that can absorb the scaling, they need corresponding algebraic compensation so that the whole subgraph, rather than a single edge, remains equivalent. This is also the main difference between graph-level smoothing and simple single-layer SmoothQuant.

### 2.4 Key Properties

- **Equivalent transform**: cooperative scaling of two adjacent layers preserves the overall computation, without changing the model output.
- **Shared-scale modeling**: multiple consumers of the same source activation can jointly contribute to the weight statistics, so that one scale is compatible with multiple branches.
- **Standalone-layer smoothing**: besides fusable adjacent subgraphs, input-side scaling can also be applied to standalone linear layers that lack a preceding fusion source.
- **Distributed friendly**: supports the aggregation of statistics in distributed training environments.
- **Graph-level scale consistency**: multiple consumers sharing the same activation source need a jointly compatible scale rather than layer-by-layer independent optimization.

In terms of error and applicability boundaries, error risk mainly arises when the equivalence conditions are incomplete or the migration is excessive. The reparameterization is equivalent in the floating-point domain, but the subsequent quantization is nonlinear, so an overly aggressive scale merely transfers the problem from the activation side to the weight side; the shift also affects the numerical range of the bias.

### 2.5 Use Cases

- Scenarios that require simultaneous outlier suppression for multiple structures such as the attention `ov`, MLP `up-down`, and consecutive linear layers.
- Serves as a preceding outlier suppression step for quantization schemes such as W8A8 and W4A8, creating a more uniform numerical distribution for activation quantizers such as [MinMax](../minmax/term_minmax.md).

### 2.6 Usage Restrictions

- The target structure must satisfy the condition that adjacent operators permit equivalent cooperative scaling.
- For standalone linear layers that cannot be fused with a preceding operator, only multiplicative scaling is appropriate; a simultaneous bias shift is not suitable.

These restrictions reflect the algorithm's dependence on model structure and equivalence transformation conditions. If the target model does not satisfy the corresponding structural assumptions, applying the algorithm indiscriminately may break the original computational relationships. Therefore, when encountering incompatible structures, first narrow the scope or use a recipe already verified for the model, rather than blindly increasing the smoothing strength.

---

## 3. Related Terms

- [SmoothQuant](../smooth_quant/term_smooth_quant.md): the parent concept; this algorithm is an iterative extension of SmoothQuant.
- [Flex Smooth Quant](../flex_smooth_quant/term_flex_smooth_quant.md): a similar algorithm that automatically finds the optimal `alpha`/`beta` through a two-stage grid search.
- [OASQ](../oasq/term_oasq.md): a similar algorithm that also supports multi-subgraph fusion but splits the scale by outlier channel.
- [SVDQuant](../svdquant/term_svdquant.md): a companion term; the SVDQuant pipeline uses this algorithm to perform outlier migration.
- [AutoRound](../autoround/term_autoround.md): a companion term, often used together with this algorithm before low-bit quantization.
- [QuaRot](../quarot/term_quarot.md): a comparison algorithm that suppresses outliers by orthogonal rotation rather than channel scaling.

---

## 4. Reference Documents

1. Xiao G et al. SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models. ICML 2023. <https://arxiv.org/abs/2211.10438>
2. [Iterative Smooth Parameter Configuration Workflow Guide](./usage_iterative_smooth.md)
