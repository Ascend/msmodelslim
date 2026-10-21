# MXFP8/MXFP4

> **Entry Category**: Quantization Basic Concept ([Quantization Basics](README.md))<br>
> **English Name**: MXFP8 / MXFP4 (Microscaling Formats)<br>
> **Application Domain**: Weight and activation quantization, ultra-low-bit compression<br>

## 1. Overview

**MX** (OCP Microscaling Formats) is a class of **block-level shared exponent** formats defined by the Open Compute Project (OCP): every **32 elements** form a block, and the whole block shares one **E8M0 exponent** (8-bit exponent, 0-bit mantissa), with the elements storing only low-order data relative to that exponent. The elements within a block share the same order of magnitude, so a large dynamic range is covered with a small bit width, and the scaling-parameter overhead is amortized to one per 32 elements.

---

## 2. Term Description

### 2.1 MXFP8

- **Structure**: a block-shared E8M0 exponent + elements in **E4M3** (1 sign bit + 4 exponent bits + 3 mantissa bits).
- **Characteristics**: an 8-bit width with half the memory traffic of [FP16/BF16](term_fp16_bf16.md), while retaining the floating-point dynamic range; compared with the uniform leveling of INT8 it tolerates outliers better.
- **Application**: such as [W8A8 MX Dynamic Quantization](../quantization_mode/linear_layer_quantization/term_w8a8_mx_dynamic.md).

### 2.2 MXFP4

- **Structure**: a block-shared E8M0 exponent + elements in **E2M1** (1 sign bit + 2 exponent bits + 1 mantissa bit, emax=2, max 6).
- **Characteristics**: a 4-bit width with a high compression ratio; the block-level shared exponent retains the floating-point dynamic range, and it tolerates outliers better than the uniform leveling of INT4.
- **Application**: such as [W4A4 MX Dynamic Quantization](../quantization_mode/linear_layer_quantization/term_w4a4_mx_dynamic.md) and [W4A4 MX Dual-Scale Quantization](../quantization_mode/linear_layer_quantization/term_w4a4_mx_dualscale.md).

### 2.3 Comparison with Related Types

- Compared with [FP8 (E4M3)](term_fp8.md): MXFP8 adopts a block-level shared exponent, amortizing the scaling-parameter overhead from one per element to one per 32 elements; the cost is a coarser block granularity and a need for hardware MX support.
- Compared with [INT4](term_int4.md): MXFP4 uses a block-level shared exponent to retain the dynamic range and tolerates outliers better; the cost is a dependence on MX hardware and a requirement that the length of the last dimension be divisible by 32.

---

## 3. Related Workflows

- [One-Click Quantization Usage Guide](../../user_guide/usage_one_click_quantization.md): the execution entry point for quantization tasks.
- [Quantization Accuracy Tuning Guide](../../user_guide/process_quantization_precision_tuning.md): accuracy problems caused by the data type and the quantization scheme can be diagnosed and tuned through this workflow.

---

## 4. Related Terms

- [Quantization Basics](README.md): The parent concept and the directory this entry belongs to.
- [Quantization and Dequantization](term_quantization.md): A companion term; the mathematical mechanism of MX quantization.
- [GEMM](../quantization_mode/term_gemm.md): A companion term; MX inputs go through a low-precision floating-point GEMM.
- [W8A8 MX Dynamic Quantization](../quantization_mode/linear_layer_quantization/term_w8a8_mx_dynamic.md): A companion mode; the typical implementation of MXFP8.
- [W4A4 MX Dynamic Quantization](../quantization_mode/linear_layer_quantization/term_w4a4_mx_dynamic.md): A companion mode; the typical implementation of MXFP4.

---

## 5. Reference Documents

1. [Quantization Modes](../quantization_mode/README.md)
