# MXFP4

> **Entry Category**: Quantization Data Type ([Quantization Data Type](README.md))<br>
> **English Name**: MXFP4<br>
> **Application Domain**: Ultra-low-bit compression of weights and activations<br>

## 1. Overview

**MXFP4** is the 4-bit block-level format of OCP MX v1.0: every 32 elements share one E8M0, and the elements inside a block are E2M1 ($e_{\max}=2$, $\mathrm{max\_norm}=6$). It still retains the floating-point dynamic range at 4 bits and is more robust to outliers than [INT4](term_int4.md). The formula is the same as [MXFP8](term_mxfp8.md), only with different element parameters. Only the symmetric path is available. For function usage, see [Quantization Data Type ops Usage Workflow Guide](usage_dtype_ops.md).

---

## 2. Term Description

### 2.1 Encoding

- Block size $k=32$; E8M0 is the same as in [MXFP8](term_mxfp8.md).
- Elements are E2M1 only: 1 sign + 2 exponent + 1 mantissa.

$$\mathrm{max\_norm} = 2^{2}\cdot\tfrac{3}{2}=6, \qquad \mathrm{min\_exp}=2-2^{1}=0$$

- $\mathrm{max\_norm}$: the upper bound of E2M1 finite values, which is 6.
- $\mathrm{min\_exp}$: the minimum exponent, which is 0.

| Element | Exponent Bits | Mantissa Bits (Implicit Included) | $e_{\max}$ | $\mathrm{max\_norm}$ |
|------|--------|------------------|-----------|----------------------|
| E2M1 | 2 | 3 | 2 | 6 |

### 2.2 Quantization Formula

Same as [MXFP8](term_mxfp8.md), except that $e_{\max}=2$ and $\mathrm{max\_norm}=6$:

$$e_{\mathrm{shared}}=\mathrm{clamp}(\lfloor\log_2\max|x|\rfloor-2,-127,127)$$

$$x'=x/2^{e_{\mathrm{shared}}},\quad q=\mathrm{clamp}(\mathrm{roundTiesToEven\_to\_E2M1}(x'),\pm 6),\quad \hat{x}=q\cdot 2^{e_{\mathrm{shared}}}$$

- $x$: the floating-point value to be quantized.
- $e_{\mathrm{shared}}$: the block-level shared exponent.
- $x'$: the value normalized by the shared exponent.
- $q$: the E2M1 quantized value.
- $\hat{x}$: the value restored by dequantization.

### 2.3 Relationship with Related Formats

- Relative to [MXFP8](term_mxfp8.md): the shared-exponent mechanism is the same, and the element changes from E4M3/E5M2 to E2M1.
- Relative to [INT4](term_int4.md): both are 4-bit; MXFP4 retains the floating-point dynamic range and is more robust to outliers.

### 2.4 Applicable Scenarios and Limitations

- **Applicable**: scenarios with very high compression-ratio requirements, for example [W4A4 MX Dynamic Quantization](../quantization_mode/linear_layer_quantization/term_w4a4_mx_dynamic.md) and [W4A4 MX Dual-Scale Quantization](../quantization_mode/linear_layer_quantization/term_w4a4_mx_dualscale.md).
- **Limitations**: it depends on hardware MX operators. The accuracy bar at 4 bits is high. Only the symmetric path is available.

---

## 3. Related Workflows

- [Quantization Data Type ops Usage Workflow Guide](usage_dtype_ops.md): the call steps of the functions for this format.
- [One-Click Quantization Usage Guide](../../user_guide/usage_one_click_quantization.md): the execution entry point for quantization tasks.
- [Quantization Accuracy Tuning Guide](../../user_guide/process_quantization_precision_tuning.md): accuracy problems caused by this format can be diagnosed through this workflow.

---

## 4. Related Terms

- [Quantization Data Type](README.md): the higher-level concept; the directory this entry belongs to.
- [Quantization and Dequantization](term_quantization.md): a prerequisite term; describes block granularity and symmetric quantization.
- [MXFP8](term_mxfp8.md): an MX format of the same family, with the same formula and E4M3/E5M2 elements.
- [INT4](term_int4.md): a 4-bit format of the same family, the integer affine counterpart.
- [GEMM](../quantization_mode/term_gemm.md): a companion term; low-bit matrix multiplication.
- [W4A4 MX Dynamic Quantization](../quantization_mode/linear_layer_quantization/term_w4a4_mx_dynamic.md): an application object; linear layers adopt this format.

---

## 5. Reference Documents

1. Open Compute Project. "OCP Microscaling Formats (MX) Specification." Version 1.0, 2023. <https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf>
2. IEEE. "IEEE Standard for Floating-Point Arithmetic." IEEE 754-2019, 2019. <https://ieeexplore.ieee.org/document/8766229>
