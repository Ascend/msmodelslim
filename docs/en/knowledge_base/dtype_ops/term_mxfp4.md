# MXFP4

> **Entry Category**: Quantization Data Type ([Quantization Data Types](README.md))<br>
> **English Name**: MXFP4<br>
> **Application Domain**: Ultra-low-bit compression of weights and activations<br>

## 1. Overview

**MXFP4** is the 4-bit block-level format of [OCP MX v1.0](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf): every 32 elements share one E8M0, and the elements in a block are E2M1 ($e_{\max}=2$, $\mathrm{max\_norm}=6$). It preserves the floating-point dynamic range at 4 bits and is more robust to outliers than [INT4](term_int4.md). The formula is the same as for [MXFP8](term_mxfp8.md), with different element parameters. Only the symmetric path is available. For function usage, see the [Quantization Data Type Ops Usage Guide](usage_dtype_ops.md).

---

## 2. Term Description

### 2.1 Encoding

- The block length is $k=32$; E8M0 is the same as in [MXFP8](term_mxfp8.md).
- The only element is E2M1: 1 sign + 2 exponent + 1 mantissa.

$$\mathrm{max\_norm} = 2^{2}\cdot\tfrac{3}{2}=6, \qquad \mathrm{min\_exp}=2-2^{1}=0$$

- $\mathrm{max\_norm}$: the upper bound of finite E2M1 values, which is 6.
- $\mathrm{min\_exp}$: the minimum exponent, which is 0.

| Element | Exponent Bits | Mantissa Bits (Including Implicit) | $e_{\max}$ | $\mathrm{max\_norm}$ |
|------|--------|------------------|-----------|----------------------|
| E2M1 | 2 | 3 | 2 | 6 |

### 2.2 Quantization Formula

The same as for [MXFP8](term_mxfp8.md), only with $e_{\max}=2$ and $\mathrm{max\_norm}=6$:

$$e_{\mathrm{shared}}=\mathrm{clamp}(\lfloor\log_2\max|x|\rfloor-2,-127,127)$$

$$x'=x/2^{e_{\mathrm{shared}}},\quad q=\mathrm{clamp}(\mathrm{roundTiesToEven\_to\_E2M1}(x'),\pm 6),\quad \hat{x}=q\cdot 2^{e_{\mathrm{shared}}}$$

- $x$: the floating-point value to be quantized.
- $e_{\mathrm{shared}}$: the block-level shared exponent.
- $x'$: the value normalized by the shared exponent.
- $q$: the E2M1 quantized value.
- $\hat{x}$: the dequantized restored value.

### 2.3 Relationship to Related Formats

- Compared with [MXFP8](term_mxfp8.md): the shared-exponent mechanism is the same, and the element is changed from E4M3/E5M2 to E2M1.
- Compared with [INT4](term_int4.md): both are 4-bit; MXFP4 preserves the floating-point dynamic range and is more robust to outliers.

### 2.4 Applicable Scenarios and Limitations

- **Applicable**: scenarios with a very high compression requirement, for example [W4A4 MX Dynamic Quantization](../quantization_mode/linear_layer_quantization/term_w4a4_mx_dynamic.md) and [W4A4 MX Dual-Scale Quantization](../quantization_mode/linear_layer_quantization/term_w4a4_mx_dualscale.md).
- **Limitations**: it depends on hardware MX operators. The accuracy threshold at 4 bits is high. Only the symmetric path is available.

---

## 3. Related Workflows

- [Quantization Data Type Ops Usage Guide](usage_dtype_ops.md): the calling steps for the functions of this type.
- [One-Click Quantization Usage Guide](../../user_guide/usage_one_click_quantization.md): the execution entry point for quantization tasks.
- [Quantization Accuracy Tuning Guide](../../user_guide/process_quantization_precision_tuning.md): accuracy problems caused by this format can be diagnosed through this workflow.

---

## 4. Related Terms

- [Quantization Data Types](README.md): The parent concept and the directory this entry belongs to.
- [Quantization and Dequantization](term_quantization.md): A prerequisite term; it explains the block granularity and symmetric quantization.
- [MXFP8](term_mxfp8.md): A sibling MX format with the same formula and E4M3/E5M2 elements.
- [INT4](term_int4.md): A sibling 4-bit format, the integer affine counterpart.
- [GEMM](../quantization_mode/term_gemm.md): A companion term; low-bit matrix multiplication.
- [W4A4 MX Dynamic Quantization](../quantization_mode/linear_layer_quantization/term_w4a4_mx_dynamic.md): An application target; the linear layer uses this format.

---

## 5. Reference Documents

1. Open Compute Project. "OCP Microscaling Formats (MX) Specification." Version 1.0, 2023. <https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf>
2. IEEE. "IEEE Standard for Floating-Point Arithmetic." IEEE 754-2019, 2019. <https://ieeexplore.ieee.org/document/8766229>
