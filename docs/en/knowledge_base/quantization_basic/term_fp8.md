# FP8 (E4M3)

> **Entry Category**: Quantization Basic Concept ([Quantization Basics](README.md))<br>
> **English Name**: FP8 (8-bit Floating Point, E4M3)<br>
> **Application Domain**: Weight and activation quantization<br>

## 1. Overview

**FP8** is an 8-bit floating-point format; this entry describes the **E4M3** variant: **1 sign + 4 exponent + 3 mantissa**, with a representable range of about $\pm448$ (emax=8). Like an integer it is only 8 bits wide, taking half the memory traffic of [FP16/BF16](term_fp16_bf16.md), while retaining a floating-point dynamic range, so it suits scenarios with a wide distribution span that are sensitive to integer rounding.

---

## 2. Term Description

### 2.1 Definition

The bit allocation of E4M3 is: 1 sign bit + 4 exponent bits + 3 mantissa bits, 8 bits in total. The 4-bit exponent provides a relatively large dynamic range (maximum about 448), and the 3-bit mantissa ensures basic precision.

### 2.2 Use Cases

- **Weights and activations**: such as [W8A8 FP8 Dynamic Quantization](../quantization_mode/linear_layer_quantization/term_w8a8_fp8_dynamic.md).
- **Outlier-sensitive scenarios**: compared with the uniform leveling of INT8, the E4M3 floating-point format tolerates a wide dynamic range and outliers better, with no need for prior outlier suppression.
- The cost is a dependence on hardware operators that support FP8 GEMM, so the ecosystem is narrower than that of INT8.

### 2.3 Comparison with Related Types

- Compared with [INT8](term_int8.md): both are 8-bit with the same memory traffic benefit, but E4M3 retains the floating-point dynamic range and tolerates outliers better; the cost is that FP8 hardware operator support is narrower.
- Compared with [MXFP8](term_mxfp.md): FP8 represents every element independently and completely; MXFP8 shares one E8M0 exponent per 32 elements, lowering the scaling-parameter overhead.

---

## 3. Related Workflows

- [One-Click Quantization Usage Guide](../../user_guide/usage_one_click_quantization.md): the execution entry point for quantization tasks.
- [Quantization Accuracy Tuning Guide](../../user_guide/process_quantization_precision_tuning.md): accuracy problems caused by the data type and the quantization scheme can be diagnosed and tuned through this workflow.

---

## 4. Related Terms

- [Quantization Basics](README.md): The parent concept and the directory this entry belongs to.
- [Quantization and Dequantization](term_quantization.md): A companion term; the mathematical mechanism of FP8 quantization.
- [MXFP8/MXFP4](term_mxfp.md): A similar algorithm and likewise an 8-bit floating-point format.
- [GEMM](../quantization_mode/term_gemm.md): A companion term; FP8 inputs go through a low-precision floating-point GEMM.
- [W8A8 FP8 Dynamic Quantization](../quantization_mode/linear_layer_quantization/term_w8a8_fp8_dynamic.md): A companion mode; the typical implementation of FP8 E4M3.

---

## 5. Reference Documents

1. Kuzmin A et al. FP8 Formats for Deep Learning. arXiv:2209.05433. <https://arxiv.org/abs/2209.05433>
2. [Quantization Modes](../quantization_mode/README.md)
