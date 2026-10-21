# AWQ Activation-Aware Weight Quantization Algorithm Terminology Entry

> **Entry Category**: [Outlier Suppression Algorithms](../README.md#1-outlier-suppression-algorithms)<br>
> **English Name**: awq_smooth<br>
> **Application Domain**: Large language model quantization and compression, low-bit quantization accuracy optimization<br>
> **First Proposed**: Lin et al., MLSys 2024

---

## 1. Overview

AWQ (Activation-aware Weight Quantization) is an activation-aware weight scaling and outlier suppression method. It uses calibration activations to estimate channel importance and searches for a scaling factor, reducing the subsequent weight quantization error while preserving the equivalent-transform relationship; its core characteristics are activation awareness, channel-level scaling, and error-driven search, making it suitable as a distribution-shaping step before low-bit weight quantization.

---

## 2. Term Description

Not all weight channels matter equally to the model output. AWQ observes that important weight channels can be identified and protected from the activation distribution: applying a smaller quantization perturbation to important channels yields better accuracy in low-bit quantization. Unlike the fixed scaling of [SmoothQuant](../smooth_quant/term_smooth_quant.md), AWQ performs a grid search guided by activation statistics to find the optimal scaling factor.

### 2.1 Core Idea

The core idea of AWQ is to "protect weight channels according to activation importance": use calibration activation statistics to estimate the importance of each input channel, then apply an equivalent per-channel scaling that allocates a more favorable quantization resolution to the weight channels that matter most to the output. The scaling strength is not a fixed empirical value but is chosen among a set of candidate ratios based on the actual output reconstruction error after real weight quantization, balancing "protecting important channels" against "avoiding worsening the scale of other weights".

From the perspective of error propagation, if the quantized weight is written as $Q(sW)=sW+E$, the output error after equivalent scaling is approximately $(X/s)E$.

### 2.2 Working Mechanism

AWQ-style methods first use calibration activations to estimate the importance of each input channel. If a channel's activations often have large magnitudes, the weight quantization error on that channel is amplified by the larger input coefficient and affects the layer output more noticeably. The algorithm therefore does not treat all weights equally, but amplifies the weights of more important channels through a per-channel scale so that those weights obtain a more favorable representation on the low-bit grid.

The scaling itself uses an equivalent reparameterization: if $X'=X/s$ and $W'=W\cdot s$, the floating-point product is unchanged. What actually needs to be searched is how much magnitude is best to migrate. The candidate ratio determines how strongly activation importance influences the scale; each candidate is passed through the target weight quantization and its reconstruction error is computed on the output of the relevant sub-layers. The search objective is thus aimed directly at the post-quantization computational error rather than merely making the weight tensor look smoother.

### 2.3 Mathematical Description

The scaling factor is computed as:

$$
s = \frac{\operatorname{act\_mean}^{\text{ratio}}}{\sqrt{\max(s) \cdot \min(s)}}, \quad s = \operatorname{clamp}(s, \min=10^{-4})
$$

- $s$: per-channel scale factor
- $\text{act\_mean}$: the per-channel mean of the absolute activation values, i.e. $mean(|act|)$, reflecting the importance of each channel
- $\text{ratio}$: the scaling ratio coefficient, grid-searched over $[0, 1)$ with a step of $1 / n\_grid$
- $10^{-4}$: the minimum value of the scaling factor

Block-level error evaluation uses MSE:

$$
\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_{\text{float}}^{(i)} - y_{\text{quant}}^{(i)})^2
$$

- $y_{\text{float}}$: the output of the ancestor module using the original floating-point weights
- $y_{\text{quant}}$: the output of the ancestor module using the quantized weights

Writing the scaling as $D=\operatorname{diag}(s)$, the floating-point equivalence is:

$$
Y=XW^T=(XD^{-1})(WD)^T.
$$

If $WD$ produces an error matrix $E=\mathcal Q(WD)-WD$ after weight quantization, the quantized output error is:

$$
\Delta Y=(XD^{-1})E^T.
$$

Increasing $s_j$ for a channel therefore explicitly reduces that channel's activation coefficient $X_j/s_j$, but it also changes $E$, because the quantizer sees the rescaled $WD$. AWQ's search is essentially about finding the point of minimum output error between these two opposing effects.

### 2.4 Key Properties

- **Activation-aware**: identifies and protects important channels based on the activation mean.
- **Candidate scale search**: compares the real post-quantization output error across different migration strengths instead of relying only on analytical heuristics.
- **Real-quantizer evaluation**: uses the real weight quantizer to evaluate the quantization error of candidate scaling factors.
- **Block-level evaluation**: evaluates the error at block level through the automatically discovered lowest common ancestor module, rather than relying on single-layer weight error.
- **Output-oriented error objective**: candidate scales are compared by the output reconstruction after real quantization rather than by merely minimizing the MSE of the weight tensor itself.

The scale therefore cannot grow without bound, and the trade-off must be searched using the real quantizer and the output error. The activation mean is a robust importance proxy, but if a task-critical behavior is triggered by a small number of rare tokens, the mean alone may underestimate those channels.

### 2.5 Use Cases

- Low-bit quantization scenarios that require automatically searching for the optimal weight scaling factor and protecting important channels.
- As a preprocessing step for weight quantization, providing a better weight distribution for weight quantization algorithms such as [MinMax](../minmax/term_minmax.md) and [SSZ](../ssz/term_ssz.md).

### 2.6 Usage Restrictions

- Requires calibration data to provide stable activation statistics; when the calibration distribution deviates from the real workload, the selected important channels and scaling factors may become invalid.
- The search stage should evaluate candidate scales using the same weight quantization as the final deployment; otherwise the minimum reconstruction error may not correspond to the final optimal accuracy.
- The grid search adds extra calibration overhead; the denser the search, the longer it takes.

These restrictions reflect the algorithm's dependence on model structure and equivalence transformation conditions. If the target model does not satisfy the corresponding structural assumptions, applying the algorithm indiscriminately may break the original computational relationships. Therefore, when encountering incompatible structures, first narrow the scope or use a recipe already verified for the model, rather than blindly increasing the smoothing strength.

---

## 3. Related Terms

- [SmoothQuant](../smooth_quant/term_smooth_quant.md): a similar algorithm in the same outlier suppression family, but it uses fixed scaling rather than activation-aware search.
- [Flex AWQ SSZ](../flex_awq_ssz/term_flex_awq_ssz.md): a similar algorithm and an extension of this one that evaluates parameters with a real quantizer.
- [MinMax](../minmax/term_minmax.md): a companion term; AWQ's search stage needs the target weight quantization method to evaluate candidate scales.
- [SSZ](../ssz/term_ssz.md): a companion term, often combined with AWQ smoothing for weight quantization.
- [QuaRot](../quarot/term_quarot.md): a comparison algorithm that suppresses outliers by orthogonal rotation rather than scaling.

---

## 4. Reference Documents

1. Lin J et al. AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration. MLSys 2024. <https://arxiv.org/abs/2306.00978>
2. [AWQ Parameter Configuration Workflow Guide](./usage_awq_smooth.md)
