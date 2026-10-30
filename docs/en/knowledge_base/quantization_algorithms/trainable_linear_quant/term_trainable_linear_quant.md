# Trainable Linear Quantization Algorithm Terminology Entry

> **Entry Category**: [Quantization Algorithms](../README.md#2-quantization-algorithms)<br>
> **English Name**: trainable_linear_quant<br>
> **Application Domain**: Large language model quantization and compression, low-bit quantization accuracy optimization<br>

---

## 1. Overview

Trainable Linear Quant (TLQ) is a block-level training-based quantization optimization algorithm for linear layers. It uses the floating-point block output as the teacher signal and learns quantization parameters and rounding with SignSGD through pluggable OPs (such as `minmax_tune`, `round_tune`, `trainable_smooth`); its core characteristics are block-level training and composable OPs, it can be regarded as an extensible implementation of [AutoRound](../autoround/term_autoround.md), and it is mainly used for W4A4 accuracy enhancement and can serve as the second stage of [DAOS](../daos/term_daos.md).

---

## 2. Term Description

A conventional one-shot-calibrated [MinMax](../minmax/term_minmax.md) or fixed rounding often introduces a relatively large reconstruction error at 4-bit and below. [AutoRound](../autoround/term_autoround.md) proved that learnable rounding can significantly improve low-bit weights, but its implementation and configuration are relatively fixed. TLQ abstracts the same idea into a "block-level Trainer + OP plugins + strategy qconfig": users can assign bit widths/granularity to different modules by strategy and select trainable OPs as needed, minimizing the block output reconstruction loss within a unified training loop.

### 2.1 Core Idea

The core idea of TLQ is to "turn the quantization parameters into trainable variables and optimize them jointly under a block-level reconstruction objective". For each decoder block: first collect the floating-point teacher output; then obtain the student output through a pseudo-quantization forward pass; update each OP parameter according to the loss specified by `train_config` (such as L1) and the best-snapshot strategy; after training, fold the learned results back into the quantization IR / weight representation, with no further training during inference.

The key to its effectiveness is that low-bit errors are correlated at the layer output, and reconstructing the block output is closer to the final accuracy objective than independent elementwise rounding.

### 2.2 Working Mechanism

The configuration side consists of three parts:

1. **`strategies`**: match `LinearQConfig` (activation and weight quantization dtype/scope/method) to linear layers by `include/exclude`.
2. **`operations`**: define the OPs enabled in the training pipeline (such as `minmax_tune`, `round_tune`, `trainable_smooth`), chosen as needed.
3. **`train_config`**: controls the number of block-level iterations, the learning rate, gradient accumulation, the loss type, and the best-checkpoint selection (ema / min_loss / last).

During execution, linear layers are wrapped block by block into trainable wrappers that bind the OP parameters; the teacher path stays in floating point, and the student path goes through the pseudo-quantization kernel (currently supporting the INT and MXFP families). The optional `train_with_act_quant` determines whether the activations are pseudo-quantized in the training forward pass; `enable_quanted_input` determines whether this layer's quantized output is used as the bypass quantized input of subsequent blocks. The optimizer uses SignSGD by default, which suits sign updates of high-dimensional rounding-type variables.

### 2.3 Mathematical Description

For block input $X$ and floating-point weight $W$, denote the floating-point teacher output $Y=f(X,W)$. The quantized student uses trainable parameters $\theta$ (range scaling, rounding offset, smoothing scale, and so on) to produce $\hat Y=f_{\mathcal Q}(X;W,\theta)$. The block-level objective can be written as:

$$
\mathcal L(\theta)=\ell\big(Y,\hat Y\big),
$$

where $\ell$ defaults to elementwise L1 (`loss_type: l1` in the implementation), and a custom form that weights outlier regions (`custom_outlier`) can also be used.

Taking weight quantization as an example, the normalized learnable rounding can be abstracted as:

$$
q(V)=\operatorname{clip}\big(\lfloor u\rfloor+h(V),q_{\min},q_{\max}\big),\qquad u=W/s+z,
$$

- $u$: the weight normalized by the scale $s$ and zero point $z$
- $V$: the learnable rounding offset (optimized by `round_tune`)
- $h(V)$: the 0/1 rounding decision, continuously relaxed during training and hardened at the end
- $q_{\min},q_{\max}$: the lower and upper bounds of the quantization codes

`minmax_tune` additionally learns symmetric/asymmetric range-related scaling, so the scale is no longer locked to the original MinMax extremes. If `trainable_smooth` is enabled, a smoothing scale $s_{\mathrm{sm}}$ is also learned on the subgraph and written back at finalize time into fused weights or a non-fused hook.

### 2.4 Key Properties

- **Pluggable OPs**: range tuning, rounding tuning, and trainable smoothing can be selected as needed, rather than a single fixed recipe.
- **Extensible OPs**: the training pipeline is pluggable; besides the built-in `minmax_tune` / `round_tune` / `trainable_smooth`, other trainable operators can be added through the same interface.
- **Block-level training**: uses the decoder block output reconstruction as the objective, reducing the misleading effect of cross-layer error accumulation on local optimization.
- **Strategy-based mixed precision**: different modules can use different `qconfig`, which facilitates fallback of sensitive layers.
- **Multiple numerical formats**: the pseudo-quantization kernel covers registered dtype/method combinations such as INT and MXFP.
- **No training overhead at inference**: the optimization happens only in the quantization phase; deployment uses the folded quantization parameters.

In terms of error and applicability boundaries, the lower the bit width, the more sensitive the rounding boundaries and clipping become, so the training gain is usually more pronounced — but it is also easier to overfit a small number of calibration samples. Insufficient iterations may not yet have converged; too many iterations or an unsuitable learning rate may improve only the local reconstruction without improving the downstream task.

### 2.5 Use Cases

- Ultra-low-bit linear layer quantization such as W4A4, where one-shot calibration is not accurate enough and higher memory usage and runtime are acceptable in exchange for reconstruction accuracy.
- Scenarios that need to combine rounding optimization, range tuning, and even trainable smoothing in the same pipeline.
- Numerical formats for which the TLQ kernel is registered and supported, such as INT and MXFP.

### 2.6 Usage Restrictions

- Aimed at linear layers only; the dtype/method combinations in `strategies` must have a corresponding TLQ kernel, otherwise configuration validation fails.
- Involves block-level training, so the quantization runtime and memory usage are significantly higher than one-shot calibration algorithms; the current implementation does not claim distributed training support.
- `trainable_smooth` depends on subgraph types recognizable by the adapter; when the subgraph does not match, `enable_subgraph_type` should not be forcibly widened.
- Low-bit scenarios still depend strongly on good upstream outlier suppression; using it together with [OASQ](../oasq/term_oasq.md), [QuaRot](../quarot/term_quarot.md), or [Iterative Smooth](../iterative_smooth/term_iterative_smooth.md) is recommended.

---

## 3. Related Workflows

- *[Trainable Linear Quant Parameter Configuration Workflow Guide](./usage_trainable_linear_quant.md)*: this workflow is how TLQ is written into a quantization task and how adaptation and parameter tuning are completed.

---

## 4. Related Terms

- [AutoRound](../autoround/term_autoround.md): a similar algorithm that is also SignSGD-based learnable rounding optimization; TLQ extends its idea into a composable OP pipeline.
- [MinMax](../minmax/term_minmax.md): a comparison algorithm; this algorithm further learns the range and rounding on top of the MinMax initial values.
- [Linear Quant](../linear_quant/term_linear_quant.md): the application target; TLQ acts on the linear quantization configuration of linear layers.
- [OASQ](../oasq/term_oasq.md): a prerequisite term that can suppress activation outliers before trainable quantization.
- [QuaRot](../quarot/term_quarot.md): a companion term; rotation-based outlier suppression is often applied before low-bit quantization.
- [DAOS](../daos/term_daos.md): a companion term; in the multimodal generation combined low-bit solution, OASQ is the preceding stage and this algorithm is the second stage.

---

## 5. Reference Documents

1. Cheng W, Zhang W, Shen H, et al. "Optimize Weight Rounding via Signed Gradient Descent for the Quantization of LLMs." Findings of EMNLP 2024. <https://arxiv.org/abs/2309.05516>
2. [Trainable Linear Quant Parameter Configuration Workflow Guide](./usage_trainable_linear_quant.md)
