# MXFP8

> **Entry Category**: Quantization Data Type ([Quantization Data Type](README.md))<br>
> **English Name**: MXFP8<br>
> **Application Domain**: Weight and activation quantization<br>

## 1. Overview

**MXFP8** is the 8-bit block-level format of OCP MX v1.0: every 32 elements share one E8M0 exponent (a power of two), and the elements inside a block are E4M3 or E5M2. It is more robust to outliers than [INT8](term_int8.md), and its scale overhead is lower than that of per-element [FP8](term_fp8.md). The formula follows OCP MX §6.3. Only the symmetric path is available. For function usage, see [Quantization Data Type ops Usage Workflow Guide](usage_dtype_ops.md).

---

## 2. Term Description

### 2.1 Encoding

- Block size $k=32$.
- E8M0: $e\in[-127,127]$; when encoded as unsigned 8-bit it is written as $e+127$, and $0\mathrm{xFF}$ is reserved for NaN.
- Elements: E4M3 by default ($e_{\max}=8$, $\mathrm{max\_norm}=448$), E5M2 optionally ($e_{\max}=15$, $\mathrm{max\_norm}=57344$).

| Element | Exponent Bits | Mantissa Bits (Implicit Included) | $e_{\max}$ | $\mathrm{max\_norm}$ | $\mathrm{min\_exp}$ |
|------|--------|------------------|-----------|----------------------|---------------------|
| E4M3 | 4 | 5 | 8 | 448 | $-6$ |
| E5M2 | 5 | 4 | 15 | 57344 | $-14$ |

### 2.2 Quantization Formula

$$e_{\mathrm{shared}} = \mathrm{clamp}\left(\lfloor \log_2 \max_{i} |x_i| \rfloor - e_{\max}, \; -127, \; 127\right)$$

$$x'_i = x_i / 2^{e_{\mathrm{shared}}}, \qquad e_{\mathrm{priv},i} = \lfloor \log_2 |x'_i| \rfloor$$

$$q_i = \mathrm{clamp}(\mathrm{roundTiesToEven\_to\_FP8}(x'_i), \pm\mathrm{max\_norm}), \qquad \hat{x}_i = q_i \cdot 2^{e_{\mathrm{shared}}}$$

- $x_i$: the $i$-th floating-point element in the block.
- $e_{\mathrm{shared}}$: the block-level shared exponent.
- $e_{\max}$: the maximum exponent of the element format, 8 for E4M3 and 15 for E5M2.
- $x'_i$: the value normalized by the shared exponent.
- $q_i$: the FP8 quantized value in the block.
- $\hat{x}_i$: the value restored by dequantization.

Subtracting $e_{\max}$ makes the largest element in the block fall on the upper edge of the element format. $e_{\mathrm{shared}}=-127$ for an all-zero block is a legal value. Element rounding is the same as [FP8](term_fp8.md). There is no zero-point.

### 2.3 Relationship with Related Formats

- Relative to [FP8](term_fp8.md): the element grid is the same, but MXFP8 replaces the per-element real-number scale with a block-level E8M0.
- Relative to [INT8](term_int8.md): both are 8-bit; MX retains the floating-point dynamic range and is more robust to outliers.
- Relative to [MXFP4](term_mxfp4.md): the shared-exponent mechanism is the same, and the element changes to E2M1.

### 2.4 Applicable Scenarios and Limitations

- **Applicable**: 8-bit deployment where there are many outliers and the scale overhead is to be amortized, for example [W8A8 MX Dynamic Quantization](../quantization_mode/linear_layer_quantization/term_w8a8_mx_dynamic.md).
- **Limitations**: the benefit depends on hardware MX operators. Only the symmetric path is available.

---

## 3. Related Workflows

- [Quantization Data Type ops Usage Workflow Guide](usage_dtype_ops.md): the call steps of the functions for this format, including `fmt` defaulting to `e4m3`.
- [One-Click Quantization Usage Guide](../../user_guide/usage_one_click_quantization.md): the execution entry point for quantization tasks.
- [Quantization Accuracy Tuning Guide](../../user_guide/process_quantization_precision_tuning.md): accuracy problems caused by this format can be diagnosed through this workflow.

---

## 4. Related Terms

- [Quantization Data Type](README.md): the higher-level concept; the directory this entry belongs to.
- [Quantization and Dequantization](term_quantization.md): a prerequisite term; describes scale and block granularity.
- [FP8](term_fp8.md): an 8-bit floating-point format of the same family, the per-element counterpart.
- [MXFP4](term_mxfp4.md): an MX format of the same family, with E2M1 elements.
- [INT8](term_int8.md): an 8-bit format of the same family, the integer affine counterpart.
- [GEMM](../quantization_mode/term_gemm.md): a companion term; low-precision matrix multiplication.
- [W8A8 MX Dynamic Quantization](../quantization_mode/linear_layer_quantization/term_w8a8_mx_dynamic.md): an application object; linear layers adopt this format.

---

## 5. Reference Documents

1. Open Compute Project. "OCP Microscaling Formats (MX) Specification." Version 1.0, 2023. <https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf>
2. Open Compute Project. "OCP 8-bit Floating Point Specification (OFP8)." Revision 1.0, 2023. <https://www.opencompute.org/documents/ocp-8-bit-floating-point-specification-ofp8-revision-1-0-2023-12-01-pdf-1>
3. IEEE. "IEEE Standard for Floating-Point Arithmetic." IEEE 754-2019, 2019. <https://ieeexplore.ieee.org/document/8766229>
