# FP16/BF16

> **Entry Category**: Quantization Basic Concept ([Quantization Basics](README.md))<br>
> **English Name**: FP16 / BF16 (16-bit Floating Point)<br>
> **Application Domain**: Unquantized baseline for weights/activations, high-precision retention scenarios<br>

## 1. Overview

**FP16** and **BF16** are both 16-bit floating-point formats, each taking 2 bytes per element. They are the **common unquantized formats** for model weights and activations and the **baseline** for the various "halved / 1/4" compression comparisons of quantization—the "halved memory traffic" of [INT8](term_int8.md) is measured relative to them. FP16 has a high mantissa precision, but its representable range is far smaller than FP32 (maximum finite value about 65504); BF16 has exactly the same exponent range as FP32, but a lower mantissa precision. Quantization modes often use them as the high-precision baseline for "not quantizing", for example the activations of [W8A16 Static Quantization](../quantization_mode/linear_layer_quantization/term_w8a16_static.md) remain FP16.

---

## 2. Term Description

### 2.1 Definition

- **FP16**: 1 sign + 5 exponent + 10 mantissa. The representable range is far smaller than FP32 (maximum finite value about 65504), and the 10-bit mantissa ensures relatively high numerical precision.
- **BF16**: 1 sign + 8 exponent + 7 mantissa. The exponent range is exactly the same as FP32 (representable range about $3.4\times10^{38}$), but the mantissa is only 7 bits, so the local precision is lower than FP16; it suits training/inference scenarios that need a large dynamic range and can sacrifice local precision.

Both have the same bit width (2 bytes each), so the footprint and memory traffic for a given tensor are identical; the only difference is the trade-off between "dynamic range" and "local precision". Compared with 8-bit formats ([INT8](term_int8.md), [FP8](term_fp8.md)), they take 2 bytes—twice as much—so they normally serve as the original storage and bandwidth baseline before quantization.

### 2.2 Use Cases

- **Unquantized baseline**: weights/activations before quantization default to 16-bit floating point (FP16 or BF16, both 2 bytes), and the compression ratio and bandwidth benefit of each quantization type are measured against it.
- **High-precision retention layers**: activations that are particularly sensitive to quantization error (such as the activations of [W8A16 Static Quantization](../quantization_mode/linear_layer_quantization/term_w8a16_static.md)) remain FP16 and do not participate in quantization.
- **Choosing between FP16 and BF16**: use BF16 when a large dynamic range is needed and FP16 when local precision is needed. The numerical format of training and inference is usually kept consistent—training generally uses BF16 (the same dynamic range as FP32), so inference also defaults to BF16.

### 2.3 Comparison with Related Types

- Compared with [INT8](term_int8.md): FP16/BF16 are unquantized floating point with zero quantization error, but they take 2 bytes, doubling the memory traffic and computation overhead; INT8 trades half the bit width for half the storage/bandwidth.
- Compared with [FP8 (E4M3)](term_fp8.md): FP16/BF16 have a larger dynamic range (BF16 the same as FP32) and more stable precision, but double the bit width and offer a smaller compression benefit.
- Compared with [MXFP8](term_mxfp.md): FP16/BF16 represent every element independently and completely, with no block-level shared parameters; MXFP8 uses a block-level shared exponent to retain the floating-point dynamic range within 8 bits.

---

## 3. Related Workflows

- [One-Click Quantization Usage Guide](../../user_guide/usage_one_click_quantization.md): the execution entry point for quantization tasks.
- [Quantization Accuracy Tuning Guide](../../user_guide/process_quantization_precision_tuning.md): accuracy problems caused by the data type and the quantization scheme can be diagnosed and tuned through this workflow.

---

## 4. Related Terms

- [Quantization Basics](README.md): The parent concept and the directory this entry belongs to.
- [Quantization and Dequantization](term_quantization.md): A companion term; the quantization formula takes an FP16/BF16 tensor as input.
- [INT8](term_int8.md): A companion term; the compression benefit of INT8 is measured against FP16/BF16.
- [W8A16 Static Quantization](../quantization_mode/linear_layer_quantization/term_w8a16_static.md): A companion mode in which the activations remain FP16 and are not quantized.

---

## 5. Reference Documents

1. Kalamkar D et al. A Study of BFLOAT16 for Deep Learning Training. arXiv:1905.12322. <https://arxiv.org/abs/1905.12322>
2. [Quantization Modes](../quantization_mode/README.md)
