# DAOS Low-Bit Quantization Solution Terminology Entry

> **Entry Category**: [Quantization Algorithms](../README.md#2-quantization-algorithms)<br>
> **English Name**: daos<br>
> **Application Domain**: Multimodal generative model quantization and compression, low-bit quantization accuracy optimization<br>

---

## 1. Overview

DAOS is a combined **W4A4 / MXFP** low-bit solution for multimodal **generative** models (not a single processor). It first uses [OASQ](../oasq/term_oasq.md) for channel-level smooth suppression of activation outliers, then uses [Trainable Linear Quant](../trainable_linear_quant/term_trainable_linear_quant.md) with block-level training to optimize the quantization range and rounding; its core characteristic is the divide-and-conquer chain of **OASQ first, TLQ second**. The solution can also pair with [FA3 Quant](../fa3_quant/term_fa3_quant.md) to perform FA quantization on attention activations for further inference acceleration.

---

## 2. Term Description

Multimodal generative models (text-to-video / image-to-video, and so on) also face the superposition of activation outliers and rounding errors at extremely low bit widths such as W4A4 and MXFP4. DAOS combines outlier suppression with trainable linear quantization: OASQ first splits and smooths outlier activations by channel-wise z-score, and TLQ then performs block-level reconstruction training on the linear layers over a friendlier distribution, recovering accuracy along the low-bit deployment path for generative models.

Positioned within the quantization flow, DAOS is not a single processor but a recommended solution composed of OASQ and TLQ in series under the `multimodal_sd_modelslim_v1` task; for the specific parameters and practical recipe, see the *[DAOS Parameter Configuration Workflow Guide](./usage_daos.md)*.

### 2.1 Core Idea

The core idea of DAOS is to **suppress outliers first and then train the quantization parameters**:

1. **OASQ**: identifies outlier channels on the calibration activations and applies a smoothing scale to fusible subgraphs (such as norm-linear), reducing the dynamic-range waste of subsequent low-bit quantization.
2. **TLQ**: after OASQ, performs trainable quantization (range / rounding and other OPs) on the linear layers, optimizing the pseudo-quantization parameters against the block output reconstruction loss and folding back into the final weight quantization representation.

The two stages are complementary: OASQ improves the activation conditioning but does not by itself produce the final low-bit representation; TLQ optimizes discretization and scale but remains sensitive to severely outlier channels. Smoothing first and then training is usually more stable than enabling only TLQ or only OASQ.

### 2.2 Working Mechanism

DAOS handles the two difficulties of low-bit multimodal generative quantization separately. In the first stage, OASQ identifies and smooths outlier activations by channel so that subsequent quantization faces a more balanced dynamic range; in the second stage, over the already smoothed distribution, TLQ performs block-level training on the linear layers, optimizing parameters such as the quantization range and rounding to reduce the reconstruction error caused by discretization.

These two kinds of operations are complementary: smoothing mainly relieves the dominance of outlier channels over the quantization step size but cannot eliminate the rounding error of a finite code point set itself; trainable quantization can optimize the scale and rounding on a fixed low-bit grid but is still affected by severe outliers. Shaping the distribution first and then optimizing the discrete decisions is usually more robust than putting all the pressure on a single step.

Thanks to the solution's reconstruction-based design, DAOS can maintain relatively high accuracy at low bit widths and can therefore be further stacked with `fa3_quant` to perform FA quantization on the attention path for inference acceleration.

Calibration relies on a `multimodal_sd_config.dump_config` floating-point dump (usually requiring `enable_dump: true`), which differs from the LLM text jsonl calibration path.

### 2.3 Mathematical Description (Conceptual)

The OASQ stage learns a smoothing scale $s$ on the subgraph so that activations and weights migrate outlier energy according to $X' = X \operatorname{diag}(s)^{-1}$ and $W' = \operatorname{diag}(s) W$, while keeping $X'W' = XW$.

- $s$: the per-channel smoothing scale
- $X$, $W$: the activations and weights before smoothing
- $X'$, $W'$: the activations and weights after smoothing

The TLQ stage minimizes a block-level reconstruction loss before pseudo-quantization, for example:

$$
\min_{\theta}\ \| Y_{\mathrm{fp}} - Y_{q}(\theta) \|_{1}
$$

- $\theta$: the trainable OP parameters such as the scale and rounding
- $Y_{\mathrm{fp}}$: the floating-point block output
- $Y_{q}(\theta)$: the pseudo-quantized block output

Overall this remains a staged approximation: the optimal OASQ scale is not necessarily globally optimal once TLQ training is taken into account, but it is decomposable and reproducible in engineering terms.

### 2.4 Key Properties

- **A combined solution rather than a single algorithm**: composed of `oasq` + `trainable_linear_quant` in series.
- **For multimodal generation**: the task API is `multimodal_sd_modelslim_v1`; calibration uses the dump path rather than the default LLM text set.
- **Can be further stacked with FA quantization**: once the low-bit accuracy is acceptable, FA3 and similar can be stacked to perform FA quantization on attention for further inference acceleration.
- **Depends on adapter subgraphs**: OASQ requires the adapter to provide a usable `get_adapter_config_for_subgraph` (implementing `OASQInterface` is recommended); otherwise it may fall back to default probing and end up with an empty scope.

### 2.5 Use Cases

- W4A4 / MXFP and other low-bit quantization of multimodal generative models.
- Scenarios where activation channels have obvious outliers and MinMax plus static linear quantization alone does not deliver enough accuracy.

### 2.6 Usage Restrictions

- Includes OASQ smoothing and TLQ block-level training, so the quantization runtime is significantly longer than a static scheme with one calibration pass; a sufficient time budget should be reserved before a formal run.
- The calibration dump and block-level training noticeably raise memory usage; when resources are tight, assess device capacity in advance and, if necessary, narrow the scope or reduce training parallelism.
- The OASQ scope must match the model subgraph naming; if the adapter does not correctly expose the subgraph, OASQ may run empty.
- The deployment format must be consistent with the saver; enabling FA3 and similar also depends on the corresponding adapter interface.

---

## 3. Related Workflows

- *[DAOS Parameter Configuration Workflow Guide](./usage_daos.md)*: this workflow is how OASQ and TLQ (and optionally FA3) are configured in series to implement DAOS.

---

## 4. Related Terms

- [OASQ](../oasq/term_oasq.md): a prerequisite term and the outlier suppression stage of DAOS.
- [Trainable Linear Quant](../trainable_linear_quant/term_trainable_linear_quant.md): a companion term and the trainable linear quantization stage of DAOS.
- [LAOS](../laos/term_laos.md): a similar algorithm and a combined low-bit solution for LLMs (Adapt Rotation + AutoRound).
- [FA3 Quant](../fa3_quant/term_fa3_quant.md): a companion term for attention FA quantization, used to further accelerate inference.

---

## 5. Reference Documents

1. [DAOS Parameter Configuration Workflow Guide](./usage_daos.md)
2. [OASQ Parameter Configuration Workflow Guide](../oasq/usage_oasq.md)
3. [Trainable Linear Quant Parameter Configuration Workflow Guide](../trainable_linear_quant/usage_trainable_linear_quant.md)
4. [FA3 Quant Parameter Configuration Workflow Guide](../fa3_quant/usage_fa3_quant.md)
