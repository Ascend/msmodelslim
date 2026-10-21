# Quantization Basics

> **Entry Category**: Quantization Basic Concept<br>
> **English Name**: Quantization Basics<br>
> **Application Domain**: Large language model quantization and compression, inference acceleration, KVCache compression<br>

## 1. Overview

This directory collects the basic terminology of the quantization field and is the prerequisite knowledge for reading knowledge entries such as [Quantization Modes](../quantization_mode/README.md). Reading [Quantization and Dequantization](term_quantization.md) first to understand the quantization mechanism is recommended, and the data type entries can then be consulted as needed; the computational benefit brought by quantization (integer GEMM) is elaborated in [Quantization Modes](../quantization_mode/README.md).

## 2. Term Description

The entries are split by terminology as follows:

| Term | Entry | Content |
|------|------|------|
| Quantization / Dequantization | [Quantization and Dequantization](term_quantization.md) | Quantization/dequantization formulas, scale/zero_point, static/dynamic, quantization granularity, symmetry |
| Data type FP16 / BF16 | [FP16/BF16](term_fp16_bf16.md) | 16-bit floating-point format, the unquantized baseline |
| Data type INT8 | [INT8](term_int8.md) | 8-bit integer format |
| Data type INT4 | [INT4](term_int4.md) | 4-bit integer format |
| Data type FP8 | [FP8 (E4M3)](term_fp8.md) | 8-bit floating-point format |
| Data type MXFP8 / MXFP4 | [MXFP8/MXFP4](term_mxfp.md) | Block-level shared exponent format |

### 2.1 WxAy Notation

Mode names commonly use the `WxAy` notation to express the **bit-width combination of the weights (W) and activations (A)** in [Linear Layer Quantization](../quantization_mode/linear_layer_quantization/README.md). The digits after `W` and `A` are the bit widths, and the suffix is optional: either the bit width alone (such as `W8A8` or `W4A4`), or additionally the data type or how the parameters are obtained (such as `W8A8 static` or `W4A4 MX dynamic`). For the complete description, see the [How to Understand a Quantization Mode](../quantization_mode/README.md#how-to-read) section in [Quantization Modes](../quantization_mode/README.md).

## 3. Related Workflows

- [One-Click Quantization Usage Guide](../../user_guide/usage_one_click_quantization.md): the execution entry point for quantization tasks.
- [Quantization Accuracy Tuning Guide](../../user_guide/process_quantization_precision_tuning.md): accuracy problems caused by the data type and the quantization scheme can be diagnosed and tuned through this workflow.

## 4. Related Terms

- [Quantization Modes](../quantization_mode/README.md): The target reading material; the entries in this directory are its prerequisite fundamentals.
- [Linear Layer Quantization](../quantization_mode/linear_layer_quantization/README.md): A subordinate concept; the concrete application of the WxAy notation and GEMM.
- [KVCache Quantization](../quantization_mode/kv_cache_quantization/README.md): A subordinate concept; the application of quantization to cached K/V tensors.
- [FA Quantization](../quantization_mode/fa_quantization/README.md): A subordinate concept; the application of quantization to attention matrix computation.

## 5. Reference Documents

1. [Quantization Modes](../quantization_mode/README.md)
