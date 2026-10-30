# AutoRound Low-Bit Quantization Algorithm Terminology Entry

> **Entry Category**: [Quantization Algorithms](../README.md#2-quantization-algorithms)<br>
> **English Name**: autoround<br>
> **Application Domain**: Large language model quantization and compression, low-bit quantization accuracy optimization<br>

---

## 1. Overview

AutoRound is an optimization algorithm for low-bit weight quantization of large language models. It introduces a learnable rounding offset for weights and iteratively optimizes it to choose a more favorable rounding direction, reducing the local reconstruction error after quantization; its core characteristics are learnable rounding, layer-by-layer optimization, and accuracy enhancement of low-bit weights such as INT4, and it can be combined with outlier suppression methods such as [QuaRot](../quarot/term_quarot.md).

---

## 2. Term Description

Traditional quantization methods (such as round-to-nearest) are not optimal for weight quantization and often introduce a large quantization error, which is especially pronounced in low-bit scenarios (4-bit and below). AutoRound observes that by introducing a learnable rounding offset together with an optimizer, the optimal rounding direction can be selected adaptively for each weight, significantly reducing the quantization reconstruction error.

### 2.1 Core Idea

The core idea of AutoRound is to "turn the rounding direction into a learnable parameter": instead of simple round-to-nearest, it uses SignSGD (sign gradient descent) to learn a rounding offset $V$ for each weight, adaptively deciding whether the weight rounds up or down, and adjusts the scaling factor and zero point in a targeted way, thereby minimizing the quantization reconstruction error.

The key to AutoRound's effectiveness is that "low-bit errors are correlated, not independent per weight".

### 2.2 Working Mechanism

Traditional nearest rounding selects the nearest integer code independently for each scalar; it only guarantees that the current scalar's error is locally minimal, not that the output reconstruction error of a whole layer is minimal. AutoRound introduces a learnable rounding offset for weights near the quantization threshold so that weights can cross their original rounding boundary; at the same time it can jointly adjust clipping/scale-related parameters to find a better balance between "widening the representable range" and "reducing the quantization step size".

During optimization, the floating-point layer output under the calibration input serves as the teacher signal and the quantized weights produce the student output, with the loss formed by their reconstruction difference. SignSGD updates the high-dimensional rounding variables using only the gradient sign, avoiding over-reliance on gradient magnitude and enabling a large number of variables near discrete boundaries to move stably toward lower output error. After optimization, the learned variables are folded into the final integer codes and quantization parameters, with no further iteration required at inference time.

### 2.3 Mathematical Description

In traditional quantization, the quantization formula for weights $W$ is:

$$
\hat{W} = s \times \operatorname{clip}(\lfloor W/s + zp \rceil, n, m)
$$

- $\hat{W}$: the quantized weights
- $s$: scale factor
- $zp$: zero point
- $n$, $m$: lower and upper bounds after quantization

On this basis, AutoRound introduces a learnable rounding offset $V$ and optional scaling-factor adjustment parameters $\alpha$ and $\beta$:

$$
\hat{W} = s \times \operatorname{clip}(\lfloor W/s + zp + V \rceil, n, m)
$$

$$
s = \frac{\max(W) \times \alpha - \min(W) \times \beta}{2^{bit} - 1}
$$

- $V$: learnable offset controlling the rounding direction
- $\alpha$, $\beta$: parameters used to adjust the range of the scaling factor
- $bit$: the number of quantization bits

The optimization proceeds layer by layer: collect the floating-point forward output as the reference, initialize and train the scaling factor and rounding offset, obtain the quantization result through a quantize-dequantize forward pass, compute the reconstruction loss, update the parameters with SignSGD, and repeat until convergence or until the maximum number of iterations is reached.

If the normalized weight is written as $u=W/s+z$, traditional nearest rounding takes $q=\operatorname{round}(u)$. Learnable rounding can be abstracted as:

$$
q(V)=\operatorname{clip}\big(\lfloor u\rfloor+h(V),q_{\min},q_{\max}\big),\qquad h(V)\in\{0,1\},
$$

Here the training phase approximates the binary function $h$ with an optimizable continuous form, which is then hardened into a 0/1 decision. If the calibration input is $X$, a common layer reconstruction objective can be written as:

$$
\mathcal L=\|XW^T-X\hat W(V,s,z)^T\|_F^2.
$$

This accounts for the importance of input directions beyond simply minimizing $\|W-\hat W\|_F^2$: if some weight errors correspond to activations that are consistently small, their impact on the output is also weak.

### 2.4 Key Properties

- **Learnable rounding**: the rounding direction is determined by a learnable offset $V$, not by fixed round-to-nearest.
- **Layer-by-layer optimization**: each decoder layer is optimized independently to avoid error accumulation across layers.
- **Ultra-low-bit support**: designed for ultra-low-bit quantization scenarios at 4-bit and below.
- **Combinable with mixed precision**: layers that remain sensitive can keep a higher bit width while the rest use a lower one.
- **No optimization overhead at inference**: the training-style search is only used to determine the final integer codes and quantization parameters; the rounding variables need not be updated further after deployment.

In terms of error and applicability boundaries, the lower the bit width, the fewer the integer codes and the more sensitive the rounding thresholds and clipping boundaries become, so the optimization gain is usually more pronounced — but it is also easier to overfit a small number of calibration samples. Insufficient iterations may not yet have found a stable integer assignment, while too many iterations or an unsuitable learning setup may improve only the local reconstruction without improving the downstream task.

### 2.5 Use Cases

- Ultra-low-bit weight quantization scenarios such as 4-bit.
- Scenarios that still require relatively high model accuracy under low-bit conditions.

### 2.6 Usage Restrictions

- Applies only to the quantization of linear layers in LLMs.
- Requires sufficient calibration data or training iterations to optimize the parameters.
- Involves a training process and has certain NPU memory requirements; only devices with NPU memory ≥64 GB are supported.
- Low-bit quantization depends heavily on a good outlier suppression algorithm; it is recommended to use it together with [QuaRot](../quarot/term_quarot.md) or [Iterative Smooth](../iterative_smooth/term_iterative_smooth.md) rather than on its own.

---

## 3. Related Terms

- [QuaRot](../quarot/term_quarot.md): a companion term, often used as an outlier suppression step before this algorithm.
- [Adapt Rotation](../adapt_rotation/term_adapt_rotation.md): a companion term, often used together with this algorithm for W4A4 quantization.
- [Iterative Smooth](../iterative_smooth/term_iterative_smooth.md): a companion term, often used as an outlier suppression step before this algorithm.
- [Trainable Linear Quant](../trainable_linear_quant/term_trainable_linear_quant.md): a similar algorithm that is also training-based linear quantization using SignSGD, and that offers a composable OP pipeline.
- [GPTQ](../gptq/term_gptq.md): a similar algorithm and another high-accuracy weight quantization optimization algorithm.
- [MinMax](../minmax/term_minmax.md): a comparison algorithm; this algorithm is a low-bit accuracy optimization extending MinMax.

---

## 4. Reference Documents

1. Cheng W, Zhang W, Shen H, et al. "Optimize Weight Rounding via Signed Gradient Descent for the Quantization of LLMs." Findings of EMNLP 2024. <https://arxiv.org/abs/2309.05516>
2. [AutoRound Parameter Configuration Workflow Guide](./usage_autoround.md)
