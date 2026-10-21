# INT4

> **Entry Category**: Quantization Basic Concept ([Quantization Basics](README.md))<br>
> **English Name**: INT4 (4-bit Integer)<br>
> **Application Domain**: Weight quantization, ultra-low-bit compression<br>

## 1. Overview

**INT4** is a 4-bit signed integer, with a range of $-8\sim7$ and 0.5 bytes per element. Compared with [INT8](term_int8.md) it compresses more (the weights are 1/4 of [FP16/BF16](term_fp16_bf16.md)), but its resolution is low, so it is usually combined with group quantization and dedicated reconstruction algorithms to control the accuracy loss.

---

## 2. Term Description

### 2.1 Definition

INT4 takes 4 bits and can represent 16 integer levels from $-8\sim7$. Mapping a floating-point tensor onto these levels by the [quantization formula](term_quantization.md) and storing it is INT4 quantization.

### 2.2 Use Cases

- **Weights**: such as the INT4 weights of [W4A8 Dynamic Quantization](../quantization_mode/linear_layer_quantization/term_w4a8_dynamic.md) and [W4A4 Dynamic Quantization](../quantization_mode/linear_layer_quantization/term_w4a4_dynamic.md).
- **Combined with reconstruction algorithms**: INT4 carries a high accuracy risk, so it is often combined with group quantization and weight reconstruction (such as [GPTQ](../quantization_algorithms/gptq/term_gptq.md) and [LAOS](../quantization_algorithms/laos/term_laos.md)) to reduce the loss.

### 2.3 Comparison with Related Types

- Compared with [INT8](term_int8.md): half the bit width and greater compression, but a low resolution and a high accuracy risk, so it is generally used only for weights and not for activations.
- Compared with [MXFP4](term_mxfp.md): INT4 uses uniform leveling with no floating-point dynamic range and is more sensitive to outliers; MXFP4 uses a block-level shared exponent to retain the dynamic range.

---

## 3. Related Workflows

- [One-Click Quantization Usage Guide](../../user_guide/usage_one_click_quantization.md): the execution entry point for quantization tasks.
- [Quantization Accuracy Tuning Guide](../../user_guide/process_quantization_precision_tuning.md): accuracy problems caused by the data type and the quantization scheme can be diagnosed and tuned through this workflow.

---

## 4. Related Terms

- [Quantization Basics](README.md): The parent concept and the directory this entry belongs to.
- [Quantization and Dequantization](term_quantization.md): A companion term; the mathematical mechanism of INT4 quantization.
- [GEMM](../quantization_mode/term_gemm.md): A companion term; INT4 inputs can go through an integer GEMM.
- [W4A8 Dynamic Quantization](../quantization_mode/linear_layer_quantization/term_w4a8_dynamic.md): A companion mode; the typical implementation of INT4 weights with INT8 activations.

---

## 5. Reference Documents

1. Frantar E et al. GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers. arXiv:2210.17323. <https://arxiv.org/abs/2210.17323>
2. [Quantization Modes](../quantization_mode/README.md)
