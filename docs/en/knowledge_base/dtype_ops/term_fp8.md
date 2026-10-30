# FP8

> **Entry Category**: Quantization Data Type ([Quantization Data Type](README.md))<br>
> **English Name**: FP8<br>
> **Application Domain**: Weight and activation quantization<br>

## 1. Overview

**FP8** follows OCP OFP8: each element is encoded independently, with no block-level shared exponent. Of the two element formats, E4M3 has higher precision (upper bound 448) and E5M2 has a wider dynamic range (upper bound 57344). The bit width is the same as [INT8](term_int8.md), but it is more robust to outliers. The standard defines only the symmetric path (a real-number scale, no zero-point). For function usage, see [Quantization Data Type ops Usage Workflow Guide](usage_dtype_ops.md).

---

## 2. Term Description

### 2.1 Encoding

| Format | Sign | Exponent | Mantissa | $e_{\max}$ | $\mathrm{max\_norm}$ |
|------|------|------|------|-----------|----------------------|
| E4M3 | 1 | 4 | 3 | 8 | $256\times 1.75=448$ |
| E5M2 | 1 | 5 | 2 | 15 | $2^{15}\times 1.5=57344$ |

- $e_{\max}$: the maximum exponent.
- $\mathrm{max\_norm}$: the upper bound of finite values, 448 for E4M3 and 57344 for E5M2.

### 2.2 Quantization Formula

$$s = \frac{\mathrm{amax}}{\mathrm{max\_norm}}, \qquad x' = \frac{x}{s}, \qquad q = \mathrm{roundTiesToEven\_to\_FP8}(x'), \qquad q = \mathrm{clamp}(q, \pm\mathrm{max\_norm}), \qquad \hat{x} = q \cdot s$$

- $x$: the floating-point value to be quantized.
- $\mathrm{amax}$: the maximum absolute value computed along the quantization axis.
- $s$: the real-number scale.
- $x'$: the value normalized by the scale.
- $q$: the quantized value after falling onto the FP8 grid.
- $\hat{x}$: the value restored by dequantization.

Element rounding (OCP OFP8):

$$\mathrm{min\_exp} = 2 - 2^{e_{\mathrm{bits}}-1}, \qquad a \leftarrow \mathrm{round}\left(\frac{a}{2^{e_{\mathrm{priv}}}} \cdot 2^{m-2}\right) / 2^{m-2} \cdot 2^{e_{\mathrm{priv}}}$$

- $e_{\mathrm{bits}}$: the number of exponent bits.
- $m$: the number of mantissa bits including the implicit bit.
- $e_{\mathrm{priv}}=\lfloor\log_2|a|\rfloor$, clamped to $\mathrm{min\_exp}$ first.
- No zero-point.

### 2.3 Relationship with Related Formats

- Relative to [INT8](term_int8.md): both are 8-bit; FP8 retains the exponent, is more robust to outliers, and relies on FP8 GEMM.
- Relative to [MXFP8](term_mxfp8.md): the element encoding is the same, but every 32 elements of MXFP8 share one E8M0, so the scale overhead is lower.

### 2.4 Applicable Scenarios and Limitations

- **Applicable**: weights and activations, for example [W8A8 FP8 Dynamic Quantization](../quantization_mode/linear_layer_quantization/term_w8a8_fp8_dynamic.md). Prefer E4M3 when there are many outliers, and use E5M2 when the distribution is wider.
- **Limitations**: hardware FP8 GEMM is required. The upper bound of E4M3 is 448; larger values are handled by scale scaling. Only the symmetric path is available.

---

## 3. Related Workflows

- [Quantization Data Type ops Usage Workflow Guide](usage_dtype_ops.md): the call steps of the functions for this format, including `fmt` defaulting to `e4m3`.
- [One-Click Quantization Usage Guide](../../user_guide/usage_one_click_quantization.md): the execution entry point for quantization tasks.
- [Quantization Accuracy Tuning Guide](../../user_guide/process_quantization_precision_tuning.md): accuracy problems caused by this format can be diagnosed through this workflow.

---

## 4. Related Terms

- [Quantization Data Type](README.md): the higher-level concept; the directory this entry belongs to.
- [Quantization and Dequantization](term_quantization.md): a prerequisite term; describes scale and symmetric quantization.
- [INT8](term_int8.md): an 8-bit format of the same family, the integer affine counterpart.
- [MXFP8](term_mxfp8.md): an 8-bit floating-point format of the same family, with a block-level shared exponent.
- [GEMM](../quantization_mode/term_gemm.md): a companion term; low-precision matrix multiplication.
- [W8A8 FP8 Dynamic Quantization](../quantization_mode/linear_layer_quantization/term_w8a8_fp8_dynamic.md): an application object; linear layers adopt this format.

---

## 5. Reference Documents

1. Open Compute Project. "OCP 8-bit Floating Point Specification (OFP8)." Revision 1.0, 2023. <https://www.opencompute.org/documents/ocp-8-bit-floating-point-specification-ofp8-revision-1-0-2023-12-01-pdf-1>
2. Kuzmin A, et al. "FP8 Formats for Deep Learning." arXiv preprint arXiv:2209.05433, 2022. <https://arxiv.org/abs/2209.05433>
3. IEEE. "IEEE Standard for Floating-Point Arithmetic." IEEE 754-2019, 2019. <https://ieeexplore.ieee.org/document/8766229>
