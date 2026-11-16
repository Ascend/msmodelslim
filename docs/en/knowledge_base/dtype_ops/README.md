# Quantization Data Types

> **Entry Category**: Quantization Data Type<br>
> **English Name**: Quantization Data Type<br>
> **Application Domain**: Large language model quantization and compression, inference acceleration, KVCache compression<br>

## 1. Overview

A quantization data type describes the quantized value itself: how many bits it occupies, how it is encoded, what range it can represent, and how it rounds and saturates. This directory collects the concept entries of each type, plus one function usage guide. The underlying atomic capability quantizes and dequantizes an arbitrary tensor per type, together with the granularity, static or dynamic mode, and symmetry, without distinguishing whether the tensor comes from weights, activations, or the KV cache. INT8 / INT4 use affine quantization and round according to [IEEE 754-2019](https://ieeexplore.ieee.org/document/8766229); FP8 follows [OCP OFP8](https://www.opencompute.org/documents/ocp-8-bit-floating-point-specification-ofp8-revision-1-0-2023-12-01-pdf-1); MXFP8 / MXFP4 follow [OCP MX v1.0](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf); and the unquantized baselines FP16 / BF16 follow [IEEE 754-2019](https://ieeexplore.ieee.org/document/8766229).

[Quantization Modes](../quantization_mode/README.md) first distinguishes the quantization object (linear layer / KVCache / FA) and then selects the capabilities above on that object. Different objects, and different quantization schemes and data types on the same object, correspond to different modes. For example, an 8-bit linear layer can be [W8A8 Static Quantization](../quantization_mode/linear_layer_quantization/term_w8a8_static.md) ([INT8](term_int8.md)), [W8A8 FP8 Dynamic Quantization](../quantization_mode/linear_layer_quantization/term_w8a8_fp8_dynamic.md), or [W8A8 MX Dynamic Quantization](../quantization_mode/linear_layer_quantization/term_w8a8_mx_dynamic.md).

Before reading the mode entries, it is recommended to first read the formulas, scale, and granularity in [Quantization and Dequantization](term_quantization.md), as well as the unquantized baseline [FP16 / BF16](term_fp16_bf16.md). For how to call the concrete functions, see the [Quantization Data Type Ops Usage Guide](usage_dtype_ops.md); for the interface list, see the [Quantization Data Type Ops Python API](../../api_reference/python_api/dtype_ops.md).

---

## 2. Term Description

### 2.1 Document Index

| Document | Type | Description |
|------|------|------|
| [Quantization and Dequantization](term_quantization.md) | Entry | scale, zero_point, static and dynamic, granularity, symmetry |
| [FP16 / BF16](term_fp16_bf16.md) | Entry | Unquantized 16-bit baseline |
| [INT8](term_int8.md) | Entry | Signed integer affine, values $[-128, 127]$ |
| [INT4](term_int4.md) | Entry | Signed integer affine, values $[-8, 7]$ |
| [FP8](term_fp8.md) | Entry | OCP OFP8, E4M3 / E5M2 |
| [MXFP8](term_mxfp8.md) | Entry | OCP MX, E8M0 + E4M3/E5M2, block length 32 |
| [MXFP4](term_mxfp4.md) | Entry | OCP MX, E8M0 + E2M1, block length 32 |
| [Quantization Data Type Ops Usage Guide](usage_dtype_ops.md) | Process guide | Function naming, granularity, `fmt`, `dtype`, and calling steps |
| [Quantization Data Type Ops Python API](../../api_reference/python_api/dtype_ops.md) | Python API | The public interface list of `dtype_ops` |

### 2.2 The Three Encoding Families

- **Integer affine**: [INT8](term_int8.md), [INT4](term_int4.md). Uniform levels, with symmetric and asymmetric variants.
- **Per-element floating point**: [FP8](term_fp8.md). Symmetric only; the scale is a real number.
- **Block-level shared exponent**: [MXFP8](term_mxfp8.md), [MXFP4](term_mxfp4.md). Every 32 elements share one E8M0. Symmetric only.

---

## 3. Related Workflows

- [Quantization Data Type Ops Usage Guide](usage_dtype_ops.md): the calling entry point for the functions in this directory.
- [Quantization Data Type Ops Python API](../../api_reference/python_api/dtype_ops.md): the public interface list of this directory.
- [One-Click Quantization Usage Guide](../../user_guide/usage_one_click_quantization.md): the execution entry point for quantization tasks.
- [Quantization Accuracy Tuning Guide](../../user_guide/process_quantization_precision_tuning.md): accuracy problems caused by the data type can be diagnosed through this workflow.

---

## 4. Related Terms

- [Quantization Modes](../quantization_mode/README.md): A higher-level concept; select the types in this directory according to the quantization object first.
- [Linear Layer Quantization](../quantization_mode/linear_layer_quantization/README.md): An application target.
- [KVCache Quantization](../quantization_mode/kv_cache_quantization/README.md): An application target.
- [FA Quantization](../quantization_mode/fa_quantization/README.md): An application target.
- [GEMM](../quantization_mode/term_gemm.md): A companion term; low-precision matrix multiplication.

---

## 5. Reference Documents

1. Open Compute Project. "OCP Microscaling Formats (MX) Specification." Version 1.0, 2023. <https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf>
2. Open Compute Project. "OCP 8-bit Floating Point Specification (OFP8)." Revision 1.0, 2023. <https://www.opencompute.org/documents/ocp-8-bit-floating-point-specification-ofp8-revision-1-0-2023-12-01-pdf-1>
3. IEEE. "IEEE Standard for Floating-Point Arithmetic." IEEE 754-2019, 2019. <https://ieeexplore.ieee.org/document/8766229>
