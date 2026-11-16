# INT8

> **Entry Category**: Quantization Data Type ([Quantization Data Types](README.md))<br>
> **English Name**: INT8<br>
> **Application Domain**: Weight, activation, and KVCache quantization<br>

## 1. Overview

**INT8** is an 8-bit signed integer with values in $[-128, 127]$, that is, 256 uniform levels. It compresses a floating-point tensor to half the memory access width and interfaces with the integer [GEMM](../quantization_mode/term_gemm.md). The quantization method is affine quantization: the floating-point value is mapped onto an integer grid through a scale and an optional zero-point, rounded with `roundTiesToEven` as specified in [IEEE 754-2019](https://ieeexplore.ieee.org/document/8766229), and then saturated to $[-128, 127]$. For function usage, see the [Quantization Data Type Ops Usage Guide](usage_dtype_ops.md).

---

## 2. Term Description

### 2.1 Encoding

$$q_{\min} = -2^{b-1} = -128, \qquad q_{\max} = 2^{b-1} - 1 = 127, \qquad b = 8$$

- $b$: the bit width.
- $q_{\min}$ and $q_{\max}$: the endpoints of the signed integer.

### 2.2 Affine Formula

$$q = \mathrm{clamp}\left(\mathrm{round}\left(\frac{x}{s}\right) + z, \; q_{\min}, \; q_{\max}\right), \qquad \hat{x} = (q - z) \cdot s$$

- $x$: the floating-point value to be quantized.
- $s$: the scale, that is, the floating-point step represented by one integer level.
- $z$: the zero-point; $0$ for symmetric quantization.
- $q$: the quantized integer.
- $\hat{x}$: the dequantized restored value.

**Symmetric** ($z=0$):

$$s = \frac{\max(-x_{\min}, x_{\max})}{q_{\max}}, \qquad q = \mathrm{clamp}\left(\mathrm{round}\left(\frac{x}{s}\right), \; q_{\min}, \; q_{\max}\right), \qquad \hat{x} = q \cdot s$$

**Asymmetric**:

$$s = \frac{x_{\max} - x_{\min}}{2^b - 1}, \qquad z = \mathrm{round}\left(\frac{-x_{\min}}{s}\right) + q_{\min}$$

- $x_{\min}$ and $x_{\max}$: the lower and upper bounds of the statistical interval. The quantization interval is forced to contain $0$ so that the integer zero-point falls within $[q_{\min}, q_{\max}]$.

Rounding uses `roundTiesToEven`, which is mandatory in IEEE / OCP; out-of-range values are saturated and the sign is preserved.

### 2.3 Relationship to Related Formats

- Compared with [FP16 / BF16](term_fp16_bf16.md): the memory access is halved, but uniform-level error is introduced.
- Compared with [INT4](term_int4.md): the formula is the same and only $b=8$ differs; the resolution is higher and the compression ratio is lower.
- Compared with [FP8](term_fp8.md): both are 8-bit; INT8 is a uniform integer grid, while FP8 keeps a floating-point exponent and is more robust to outliers.

### 2.4 Applicable Scenarios and Limitations

- **Applicable**: weights, activations, and the KVCache, for example [W8A8 Static Quantization](../quantization_mode/linear_layer_quantization/term_w8a8_static.md), [W8A8 Dynamic Quantization](../quantization_mode/linear_layer_quantization/term_w8a8_dynamic.md), and [KVCache-PerChannel Quantization](../quantization_mode/kv_cache_quantization/term_kv_cache_perchannel.md).
- **Limitations**: uniform levels are sensitive to outliers, so [SmoothQuant](../quantization_algorithms/smooth_quant/term_smooth_quant.md) is often used together with it. Asymmetric quantization stores one extra offset.

---

## 3. Related Workflows

- [Quantization Data Type Ops Usage Guide](usage_dtype_ops.md): the calling steps for the functions of this type.
- [One-Click Quantization Usage Guide](../../user_guide/usage_one_click_quantization.md): the execution entry point for quantization tasks.
- [Quantization Accuracy Tuning Guide](../../user_guide/process_quantization_precision_tuning.md): accuracy problems caused by this format can be diagnosed through this workflow.

---

## 4. Related Terms

- [Quantization Data Types](README.md): The parent concept and the directory this entry belongs to.
- [Quantization and Dequantization](term_quantization.md): A prerequisite term; it explains the affine formula, scale, and zero-point.
- [FP16 / BF16](term_fp16_bf16.md): A comparison term; the unquantized 16-bit baseline.
- [INT4](term_int4.md): A sibling integer format with half the bit width.
- [FP8](term_fp8.md): A sibling 8-bit format, the per-element floating-point counterpart.
- [GEMM](../quantization_mode/term_gemm.md): A companion term; integer matrix multiplication.
- [W8A8 Static Quantization](../quantization_mode/linear_layer_quantization/term_w8a8_static.md): An application target; the linear layer uses this format.

---

## 5. Reference Documents

1. IEEE. "IEEE Standard for Floating-Point Arithmetic." IEEE 754-2019, 2019. <https://ieeexplore.ieee.org/document/8766229>
2. Jacob B, et al. "Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference." CVPR, 2018. <https://arxiv.org/abs/1712.05877>
3. Open Compute Project. "OCP Microscaling Formats (MX) Specification." Version 1.0, 2023. <https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf>
