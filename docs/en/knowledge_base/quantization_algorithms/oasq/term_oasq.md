# OASQ Outlier-Aware Smooth Quantization Algorithm Terminology Entry

> **Entry Category**: [Outlier Suppression Algorithms](../README.md#1-outlier-suppression-algorithms)<br>
> **English Name**: oasq<br>
> **Application Domain**: Large language model quantization and compression, inference acceleration<br>

---

## 1. Overview

OASQ (Outlier-Aware Smooth Quantization) is a smoothing preprocessing algorithm for activation outlier channels. It first identifies outlier channels by channel-wise z-score, then applies different scale formulas to normal and outlier channels, and fuses the scales into adjacent subgraphs that permit equivalent transformation; its core characteristics are outlier awareness, iterative threshold search, and multi-subgraph fusion, and it can be combined with subsequent quantization steps such as [Linear Quant](../linear_quant/term_linear_quant.md) and [Trainable Linear Quant](../trainable_linear_quant/term_trainable_linear_quant.md).

---

## 2. Term Description

Conventional [SmoothQuant](../smooth_quant/term_smooth_quant.md) uses uniform exponential scaling for all channels, making it hard to distinguish the "bulk distribution" from a few extreme channels. Before smoothing, OASQ estimates the outlier proportion from the activation channel statistics and iteratively adjusts the z threshold within the target outlier-proportion interval, so that normal channels use a scale closer to geometric-mean balancing while outlier channels use a gentler logarithmic scale, reducing the dominance of extreme channels over the overall quantization scale.

### 2.1 Core Idea

The core idea of OASQ is to "identify outlier channels first and then design the smoothing scale separately": build a z-score from the activation channel mean and standard deviation and divide the channels into a normal set and an outlier set; normal channels emphasize the balance between activation and weight magnitudes, while outlier channels use a more conservative logarithmic migration, avoiding pressing too much difficulty onto the weight side in one step. The implementation also iteratively adjusts the z threshold so that the outlier proportion falls into the preset target interval.

The benefit comes from turning "a few channels inflating the global scale" into a scale-allocation problem that is split by channel type.

### 2.2 Working Mechanism

During calibration, MinMax statistics are collected on the input of the target linear layer to obtain per-channel activation magnitudes (the absolute maximum in symmetric mode; in asymmetric mode, when the subgraph supports it, the value is first centered by shift and then the absolute maximum is taken). The `OASQScaleCalculator` then adjusts the z threshold within `max_iters` iterations: if the outlier proportion is too high the threshold is raised, and if too low it is lowered, until the proportion falls into the target interval or the iterations end.

Once the per-channel scale $s$ is obtained, it is fused equivalently according to the subgraph type: `norm-linear`, `linear-linear`, `ov`, `up-down`, and similar absorb the scale into the adjacent norm/weight (and optionally the shift); an independent linear layer that cannot be fused upstream can apply a multiplicative scaling to its input at forward time through a non-fused hook. The asymmetric shift is currently enabled mainly on the `norm-linear` path, and is disabled on other subgraphs.

### 2.3 Mathematical Description

Let the channel activation statistic be $a_j$ and the weight input-channel statistic be $w_j$, with channel mean and standard deviation $\mu$ and $\sigma$. Define:

$$
z_j=\frac{a_j-\mu}{\sigma+\varepsilon},\qquad
\mathcal O=\{\,j:\lvert z_j\rvert>z_{\mathrm{thr}}\,\}.
$$

- $z_j$: the z-score of the $j$-th channel
- $z_{\mathrm{thr}}$: the outlier decision threshold (started from a baseline value and adjusted iteratively in the implementation)
- $\mathcal O$: the set of channels judged to be outliers
- $\varepsilon$: a numerical stability term

Construct separate scales for normal and outlier channels (the implementation also mean-normalizes the scale):

$$
s_j=
\begin{cases}
\sqrt{\dfrac{w_j}{\sqrt{a_j}+\varepsilon}+\varepsilon}, & j\notin\mathcal O\\[1.2em]
\log\!\big(1+\dfrac{w_j}{\log(1+a_j)+\varepsilon}+\varepsilon\big), & j\in\mathcal O
\end{cases}
$$

- $s_j$: the smoothing scale of the $j$-th channel
- $a_j$: the activation channel magnitude statistic
- $w_j$: the magnitude statistic of the weight's corresponding input channel

In the floating-point domain, $X'=X/s$ and $W'=Ws$ still preserve $X'W'^T=XW^T$. The iteration adjusts $z_{\mathrm{thr}}$ to bring the outlier proportion into the target interval $[r_{\min},r_{\max}]$, achieving a compromise between "over-smoothing" and "barely identifying outliers".

### 2.4 Key Properties

- **Outlier aware**: the scale formula is split by whether a channel is an outlier, rather than sharing one exponent across all channels.
- **Searchable threshold**: a limited number of iterations constrains the outlier proportion to the target interval, reducing the difficulty of manually setting a fixed threshold.
- **Multi-subgraph fusion**: supports structures that can absorb the scale equivalently, such as `norm-linear`, `linear-linear`, `ov`, and `up-down`.
- **Symmetric / asymmetric paths**: `symmetric` affects the statistics and whether shift is enabled; the asymmetric shift mainly covers `norm-linear`.
- **Distributed friendly**: statistics collection supports distributed synchronization according to module sharing relationships.

In terms of error boundaries, too strict a threshold makes almost all channels follow the normal formula, bringing a benefit close to ordinary smoothing; too loose a threshold labels too many channels as outliers, and the logarithmic scale may migrate insufficiently. The scale is still nonlinearly clipped by the subsequent quantization, so the end-to-end accuracy depends on the combination with the downstream quantizer.

### 2.5 Use Cases

- Scenarios where the activations have obvious channel-level outliers and uniform exponential smoothing gives limited benefit.
- As a preprocessing outlier suppression step for schemes such as W8A8 and W4A8, creating a more uniform activation distribution for subsequent linear quantization.

### 2.6 Usage Restrictions

- The target structure needs to be recognizable as a subgraph that preserves computational equivalence; the non-fused path is suitable only for multiplicative scaling.
- The asymmetric `shift` is currently supported mainly for `norm-linear`; it is disabled on other subgraphs.
- Representative calibration data is required to collect activation statistics; some modules such as inactive MoE experts may be skipped because they have no statistics.

---

## 3. Related Workflows

- *[OASQ Parameter Configuration Workflow Guide](./usage_oasq.md)*: this workflow is how OASQ is written into a quantization task and how adaptation and parameter tuning are completed.

---

## 4. Related Terms

- [SmoothQuant](../smooth_quant/term_smooth_quant.md): the parent concept; this algorithm belongs to the same activation-weight cooperative smoothing route but splits the scale by outlier channel.
- [Iterative Smooth](../iterative_smooth/term_iterative_smooth.md): a similar algorithm that likewise supports multi-subgraph smoothing fusion.
- [Flex Smooth Quant](../flex_smooth_quant/term_flex_smooth_quant.md): a similar algorithm that determines the smoothing strength by searching alpha/beta.
- [AWQ](../awq_smooth/term_awq_smooth.md): a comparison algorithm that searches for the scaling factor based on activation importance.
- [Trainable Linear Quant](../trainable_linear_quant/term_trainable_linear_quant.md): a companion term; this algorithm can be used to suppress outliers before low-bit trainable quantization.
- [DAOS](../daos/term_daos.md): a companion term; in the multimodal generation combined low-bit solution this algorithm serves as the outlier suppression stage, followed by TLQ.

---

## 5. Reference Documents

1. Xiao G et al. SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models. ICML 2023. <https://arxiv.org/abs/2211.10438>
2. [OASQ Parameter Configuration Workflow Guide](./usage_oasq.md)
