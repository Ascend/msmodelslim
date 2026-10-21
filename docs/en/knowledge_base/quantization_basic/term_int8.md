# INT8

> **Entry Category**: Quantization Basic Concept ([Quantization Basics](README.md))<br>
> **English Name**: INT8 (8-bit Integer)<br>
> **Application Domain**: Weight, activation, and KVCache quantization<br>

## 1. Overview

**INT8** is an 8-bit signed integer with a range of $-128\sim127$ and 1 byte per element. The storage and bandwidth of INT8 are half those of [FP16/BF16](term_fp16_bf16.md), and the integer format can use [GEMM](../quantization_mode/term_gemm.md) acceleration; it can be used for weights, activations, and the KVCache, and is the mainstream quantization format.

---

## 2. Term Description

### 2.1 Definition

INT8 takes 8 bits and can represent 256 integer levels from $-128\sim127$. Mapping a floating-point tensor onto these levels by the [quantization formula](term_quantization.md) and storing it is INT8 quantization.

### 2.2 Use Cases

- **Weights**: such as the weights of [W8A8 Static Quantization](../quantization_mode/linear_layer_quantization/term_w8a8_static.md) and [W8A8 Dynamic Quantization](../quantization_mode/linear_layer_quantization/term_w8a8_dynamic.md).
- **Activations**: combined with dynamic quantization (per-token), such as the activations of [W8A8 Dynamic Quantization](../quantization_mode/linear_layer_quantization/term_w8a8_dynamic.md).
- **KVCache**: such as [KVCache-PerChannel Quantization](../quantization_mode/kv_cache_quantization/term_kv_cache_perchannel.md).

### 2.3 Comparison with Related Types

- Compared with [FP8 (E4M3)](term_fp8.md): INT8 uses uniform leveling with no floating-point dynamic range and is more sensitive to outliers, but its hardware operator ecosystem is more mature.
- Compared with [INT4](term_int4.md): the resolution is higher and the precision more stable, but the bit width is doubled and the compression is smaller.

---

## 3. Related Workflows

- [One-Click Quantization Usage Guide](../../user_guide/usage_one_click_quantization.md): the execution entry point for quantization tasks.
- [Quantization Accuracy Tuning Guide](../../user_guide/process_quantization_precision_tuning.md): accuracy problems caused by the data type and the quantization scheme can be diagnosed and tuned through this workflow.

---

## 4. Related Terms

- [Quantization Basics](README.md): The parent concept and the directory this entry belongs to.
- [Quantization and Dequantization](term_quantization.md): A companion term; the mathematical mechanism of INT8 quantization.
- [FP16/BF16](term_fp16_bf16.md): A companion term; the baseline format for the compression benefit of INT8.
- [FP8](term_fp8.md): A similar algorithm and likewise an 8-bit data format.
- [INT4](term_int4.md): A similar algorithm and a lower-bit integer format.
- [GEMM](../quantization_mode/term_gemm.md): A companion term; INT8 inputs can go through an integer GEMM.
- [W8A8 Static Quantization](../quantization_mode/linear_layer_quantization/term_w8a8_static.md): A companion mode; the typical implementation of INT8 static quantization.

---

## 5. Reference Documents

1. [Quantization Modes](../quantization_mode/README.md)
