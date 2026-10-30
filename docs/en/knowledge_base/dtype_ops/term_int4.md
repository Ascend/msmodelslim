# INT4

> **Entry Category**: Quantization Data Type ([Quantization Data Type](README.md))<br>
> **English Name**: INT4<br>
> **Application Domain**: Weight quantization, ultra-low-bit compression<br>

## 1. Overview

**INT4** is a 4-bit signed integer with the value range $[-8, 7]$, for 16 uniform levels in total. Its compression ratio doubles again relative to [INT8](term_int8.md); it is usually applied to weights only, and is combined with grouping and reconstruction ([GPTQ](../quantization_algorithms/gptq/term_gptq.md), [LAOS](../quantization_algorithms/laos/term_laos.md)). The formula is the same as INT8, except that $b=4$. For function usage, see [Quantization Data Type ops Usage Workflow Guide](usage_dtype_ops.md).

---

## 2. Term Description

### 2.1 Encoding

$$q_{\min} = -2^{b-1} = -8, \qquad q_{\max} = 2^{b-1} - 1 = 7, \qquad b = 4$$

- $b$: the bit width.
- $q_{\min}$, $q_{\max}$: the endpoints of the signed integer range.

### 2.2 Affine Formula

Same origin as [INT8](term_int8.md).

**Symmetric**: $s=\max(-x_{\min},x_{\max})/7$, $q=\mathrm{clamp}(\mathrm{round}(x/s),-8,7)$, $\hat{x}=q\cdot s$.

**Asymmetric**: $s=(x_{\max}-x_{\min})/15$, $z=\mathrm{round}(-x_{\min}/s)+q_{\min}$.

- $x$: the floating-point value to be quantized.
- $s$: the scale.
- $z$: the zero-point; $0$ when symmetric.
- $x_{\min}$, $x_{\max}$: the statistical interval, which is forced to include $0$.

Rounding uses `roundTiesToEven` and saturates to $[-8, 7]$.

### 2.3 Relationship with Related Formats

- Relative to [INT8](term_int8.md): the formula is the same and the bit width is halved; the resolution drops from 256 levels to 16 levels.
- Relative to [MXFP4](term_mxfp4.md): both are 4-bit; INT4 is a uniform integer grid, while MXFP4 uses a block-level shared exponent to retain the floating-point dynamic range.

### 2.4 Applicable Scenarios and Limitations

- **Applicable**: weight quantization, for example [W4A8 Dynamic Quantization](../quantization_mode/linear_layer_quantization/term_w4a8_dynamic.md) and [W4A4 Dynamic Quantization](../quantization_mode/linear_layer_quantization/term_w4a4_dynamic.md). Combine it with GPTQ / LAOS when accuracy is sensitive.
- **Limitations**: there are only 16 levels, so it is generally not used for activations; it is more sensitive to outliers than [MXFP4](term_mxfp4.md).

---

## 3. Related Workflows

- [Quantization Data Type ops Usage Workflow Guide](usage_dtype_ops.md): the call steps of the functions for this format.
- [One-Click Quantization Usage Guide](../../user_guide/usage_one_click_quantization.md): the execution entry point for quantization tasks.
- [Quantization Accuracy Tuning Guide](../../user_guide/process_quantization_precision_tuning.md): accuracy problems caused by this format can be diagnosed through this workflow.

---

## 4. Related Terms

- [Quantization Data Type](README.md): the higher-level concept; the directory this entry belongs to.
- [Quantization and Dequantization](term_quantization.md): a prerequisite term; describes the affine formula and group granularity.
- [INT8](term_int8.md): an integer format of the same family, with the same formula but a different bit width.
- [MXFP4](term_mxfp4.md): a 4-bit format of the same family, the block-level shared exponent counterpart.
- [GEMM](../quantization_mode/term_gemm.md): a companion term; low-bit matrix multiplication.
- [W4A8 Dynamic Quantization](../quantization_mode/linear_layer_quantization/term_w4a8_dynamic.md): an application object; linear layer weights adopt this format.

---

## 5. Reference Documents

1. IEEE. "IEEE Standard for Floating-Point Arithmetic." IEEE 754-2019, 2019. <https://ieeexplore.ieee.org/document/8766229>
2. Jacob B, et al. "Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference." CVPR, 2018. <https://arxiv.org/abs/1712.05877>
3. Frantar E, et al. "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers." arXiv preprint arXiv:2210.17323, 2022. <https://arxiv.org/abs/2210.17323>
