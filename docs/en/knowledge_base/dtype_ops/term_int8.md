# INT8

> **Entry Category**: Quantization Data Type ([Quantization Data Type](README.md))<br>
> **English Name**: INT8<br>
> **Application Domain**: Weight, activation, and KVCache quantization<br>

## 1. Overview

**INT8** is an 8-bit signed integer with the value range $[-128, 127]$, for 256 uniform levels in total. It compresses a floating-point tensor to half the memory-access width and connects to integer [GEMM](../quantization_mode/term_gemm.md). The quantization method is affine quantization: floating-point values are mapped onto the integer grid through the scale and an optional zero-point, rounded by `roundTiesToEven`, and saturated to $[-128, 127]$. For function usage, see [Quantization Data Type ops Usage Workflow Guide](usage_dtype_ops.md).

---

## 2. Term Description

### 2.1 Encoding

$$q_{\min} = -2^{b-1} = -128, \qquad q_{\max} = 2^{b-1} - 1 = 127, \qquad b = 8$$

- $b$: the bit width.
- $q_{\min}$, $q_{\max}$: the endpoints of the signed integer range.

### 2.2 Affine Formula

$$q = \mathrm{clamp}\left(\mathrm{round}\left(\frac{x}{s}\right) + z, \; q_{\min}, \; q_{\max}\right), \qquad \hat{x} = (q - z) \cdot s$$

- $x$: the floating-point value to be quantized.
- $s$: the scale, the floating-point step size corresponding to one integer level.
- $z$: the zero-point; $0$ for symmetric quantization.
- $q$: the integer after quantization.
- $\hat{x}$: the value restored by dequantization.

**Symmetric** ($z=0$):

$$s = \frac{\max(-x_{\min}, x_{\max})}{q_{\max}}, \qquad q = \mathrm{clamp}\left(\mathrm{round}\left(\frac{x}{s}\right), \; q_{\min}, \; q_{\max}\right), \qquad \hat{x} = q \cdot s$$

**Asymmetric**:

$$s = \frac{x_{\max} - x_{\min}}{2^b - 1}, \qquad z = \mathrm{round}\left(\frac{-x_{\min}}{s}\right) + q_{\min}$$

- $x_{\min}$, $x_{\max}$: the lower and upper bounds of the statistical interval. The quantization interval is forced to include $0$ so that the integer zero-point falls within $[q_{\min}, q_{\max}]$.

Rounding uses `roundTiesToEven`, which is mandatory in IEEE / OCP; out-of-range values saturate, and the sign is preserved.

### 2.3 Relationship with Related Formats

- Relative to [FP16 / BF16](term_fp16_bf16.md): memory access is halved, but uniform-level error is introduced.
- Relative to [INT4](term_int4.md): the formula is the same, only $b=8$; the resolution is higher and the compression ratio is lower.
- Relative to [FP8](term_fp8.md): both are 8-bit; INT8 is a uniform integer grid, while FP8 retains the floating-point exponent and is more robust to outliers.

### 2.4 Applicable Scenarios and Limitations

- **Applicable**: weights, activations, and KVCache, for example [W8A8 Static Quantization](../quantization_mode/linear_layer_quantization/term_w8a8_static.md), [W8A8 Dynamic Quantization](../quantization_mode/linear_layer_quantization/term_w8a8_dynamic.md), and [KVCache-PerChannel Quantization](../quantization_mode/kv_cache_quantization/term_kv_cache_perchannel.md).
- **Limitations**: uniform levels are sensitive to outliers, so [SmoothQuant](../quantization_algorithms/smooth_quant/term_smooth_quant.md) is often combined. Asymmetric quantization needs to store one extra offset.

---

## 3. Related Workflows

- [Quantization Data Type ops Usage Workflow Guide](usage_dtype_ops.md): the call steps of the functions for this format.
- [One-Click Quantization Usage Guide](../../user_guide/usage_one_click_quantization.md): the execution entry point for quantization tasks.
- [Quantization Accuracy Tuning Guide](../../user_guide/process_quantization_precision_tuning.md): accuracy problems caused by this format can be diagnosed through this workflow.

---

## 4. Related Terms

- [Quantization Data Type](README.md): the higher-level concept; the directory this entry belongs to.
- [Quantization and Dequantization](term_quantization.md): a prerequisite term; describes the affine formula, scale, and zero-point.
- [FP16 / BF16](term_fp16_bf16.md): a comparison term; the unquantized 16-bit baseline.
- [INT4](term_int4.md): an integer format of the same family, with half the bit width.
- [FP8](term_fp8.md): an 8-bit format of the same family, the per-element floating-point counterpart.
- [GEMM](../quantization_mode/term_gemm.md): a companion term; integer matrix multiplication.
- [W8A8 Static Quantization](../quantization_mode/linear_layer_quantization/term_w8a8_static.md): an application object; linear layers adopt this format.

---

## 5. Reference Documents

1. IEEE. "IEEE Standard for Floating-Point Arithmetic." IEEE 754-2019, 2019. <https://ieeexplore.ieee.org/document/8766229>
2. Jacob B, et al. "Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference." CVPR, 2018. <https://arxiv.org/abs/1712.05877>
3. Open Compute Project. "OCP Microscaling Formats (MX) Specification." Version 1.0, 2023. <https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf>
