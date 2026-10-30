# Quantization Data Type

> **Entry Category**: Quantization Data Type<br>
> **English Name**: Quantization Data Type<br>
> **Application Domain**: Large language model quantization and compression, inference acceleration, KVCache compression<br>

## 1. Overview

A quantization data type describes the quantized value itself: how many bits it occupies, how it is encoded, what range it can represent, and how it is rounded and truncated. This directory collects the concept entries for each format, together with a function usage guide. The underlying atomic capability quantizes and dequantizes any tensor by format, with the granularity, static or dynamic, and symmetry attached, without distinguishing whether a tensor comes from weights, activations, or KV.

[Quantization Modes](../quantization_mode/README.md) first distinguishes the quantization object (linear layer / KVCache / FA), and then selects the capability above on that object. Different objects, and different quantization methods and data types on the same object, correspond to different modes. For example, an 8-bit linear layer can be [W8A8 Static Quantization](../quantization_mode/linear_layer_quantization/term_w8a8_static.md) ([INT8](term_int8.md)), [W8A8 FP8 Dynamic Quantization](../quantization_mode/linear_layer_quantization/term_w8a8_fp8_dynamic.md), or [W8A8 MX Dynamic Quantization](../quantization_mode/linear_layer_quantization/term_w8a8_mx_dynamic.md).

Before reading the mode entries, you are advised to first read the formulas, scale, and granularity in [Quantization and Dequantization](term_quantization.md), and the unquantized baseline [FP16 / BF16](term_fp16_bf16.md). For how the specific functions are called, see [Quantization Data Type ops Usage Workflow Guide](usage_dtype_ops.md).

---

## 2. Term Description

### 2.1 Document Index

| Document | Type | Description |
|------|------|------|
| [Quantization and Dequantization](term_quantization.md) | Entry | scale, zero_point, static vs. dynamic, granularity, symmetry |
| [FP16 / BF16](term_fp16_bf16.md) | Entry | Unquantized 16-bit baseline |
| [INT8](term_int8.md) | Entry | Signed integer affine, value range $[-128, 127]$ |
| [INT4](term_int4.md) | Entry | Signed integer affine, value range $[-8, 7]$ |
| [FP8](term_fp8.md) | Entry | OCP OFP8, E4M3 / E5M2 |
| [MXFP8](term_mxfp8.md) | Entry | OCP MX, E8M0 + E4M3/E5M2, block size 32 |
| [MXFP4](term_mxfp4.md) | Entry | OCP MX, E8M0 + E2M1, block size 32 |
| [Quantization Data Type ops Usage Workflow Guide](usage_dtype_ops.md) | Workflow guide | Function naming, granularity, `fmt`, `dtype`, and call steps |

### 2.2 Three Encoding Families

- **Integer affine**: [INT8](term_int8.md), [INT4](term_int4.md). Uniform levels, available in both symmetric and asymmetric forms.
- **Per-element floating-point**: [FP8](term_fp8.md). Symmetric only; the scale is a real number.
- **Block-level shared exponent**: [MXFP8](term_mxfp8.md), [MXFP4](term_mxfp4.md). Every 32 elements share one E8M0. Symmetric only.

---

## 3. Related Workflows

- [Quantization Data Type ops Usage Workflow Guide](usage_dtype_ops.md): the call entry point for the functions in this directory.
- [One-Click Quantization Usage Guide](../../user_guide/usage_one_click_quantization.md): the execution entry point for quantization tasks.
- [Quantization Accuracy Tuning Guide](../../user_guide/process_quantization_precision_tuning.md): accuracy problems caused by a data type can be diagnosed through this workflow.

---

## 4. Related Terms

- [Quantization Modes](../quantization_mode/README.md): the upper-layer concept; the formats in this directory are first selected by quantization object.
- [Linear Layer Quantization](../quantization_mode/linear_layer_quantization/README.md): an application object.
- [KVCache Quantization](../quantization_mode/kv_cache_quantization/README.md): an application object.
- [FA Quantization](../quantization_mode/fa_quantization/README.md): an application object.
- [GEMM](../quantization_mode/term_gemm.md): a companion term; low-precision matrix multiplication.

---

## 5. Reference Documents

1. Open Compute Project. "OCP Microscaling Formats (MX) Specification." Version 1.0, 2023. <https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf>
2. Open Compute Project. "OCP 8-bit Floating Point Specification (OFP8)." Revision 1.0, 2023. <https://www.opencompute.org/documents/ocp-8-bit-floating-point-specification-ofp8-revision-1-0-2023-12-01-pdf-1>
3. IEEE. "IEEE Standard for Floating-Point Arithmetic." IEEE 754-2019, 2019. <https://ieeexplore.ieee.org/document/8766229>
