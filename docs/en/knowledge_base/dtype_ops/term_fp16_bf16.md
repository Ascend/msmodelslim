# FP16 / BF16

> **Entry Category**: Quantization Data Type ([Quantization Data Types](README.md))<br>
> **English Name**: FP16 / BF16<br>
> **Application Domain**: Unquantized baseline for weights and activations, high-precision retention scenarios<br>

## 1. Overview

**FP16** and **BF16** are both 16-bit floating-point formats, each element occupying 2 bytes, and their encoding follows [IEEE 754-2019](https://ieeexplore.ieee.org/document/8766229). They are the common unquantized formats for weights and activations, and they are also the baseline for comparisons such as "halved / quartered": for example, the halved memory access of [INT8](term_int8.md) is measured against them. FP16 has a finer mantissa, but its representable range is far smaller than that of FP32 (the maximum finite value is about 65504). BF16 has the same exponent range as FP32 and a coarser mantissa. Quantization modes often treat them as the high-precision baseline that is left unquantized; for example, the activation of [W8A16 Static Quantization](../quantization_mode/linear_layer_quantization/term_w8a16_static.md) stays FP16.

---

## 2. Term Description

### 2.1 Definition

- **FP16**: 1 sign + 5 exponent + 10 mantissa. The representable range is far smaller than that of FP32 (the maximum finite value is about 65504), and the 10-bit mantissa guarantees high numerical precision.
- **BF16**: 1 sign + 8 exponent + 7 mantissa. The exponent range is exactly the same as FP32 (the representable range is about $3.4\times10^{38}$), but the mantissa is only 7 bits, so the local precision is lower than FP16; it suits training/inference scenarios that need a large dynamic range and can trade local precision.

The two have the same bit width, both 2 bytes, so the footprint and memory access are identical; they differ only in dynamic range and local precision. Compared with 8-bit formats ([INT8](term_int8.md), [FP8](term_fp8.md)), they occupy 2 bytes and usually serve as the storage and bandwidth baseline before quantization.

### 2.2 Usage Scenarios

- **Unquantized baseline**: the weights/activations before quantization default to 16-bit floating point (FP16 or BF16, both 2 bytes), and the compression ratio and bandwidth benefit of every kind of quantization are measured against it.
- **High-precision retention layers**: activations that are especially sensitive to quantization error (such as the activation of [W8A16 Static Quantization](../quantization_mode/linear_layer_quantization/term_w8a16_static.md)) stay FP16 and do not participate in quantization.
- **Choosing between FP16 and BF16**: use BF16 when a large dynamic range is needed, and FP16 when local precision is needed. The numerical format of training and inference is usually kept consistent: training generally uses BF16 (the same dynamic range as FP32), so inference defaults to BF16 as well.

### 2.3 Comparison with Related Types

- Compared with [INT8](term_int8.md): FP16/BF16 are unquantized floating point with zero quantization error, but they occupy 2 bytes, doubling the memory access and computation overhead; INT8 trades half the bit width for half the storage/bandwidth.
- Compared with [FP8 (E4M3)](term_fp8.md): FP16/BF16 have a larger dynamic range (BF16 is the same as FP32) and more stable accuracy, but the bit width is doubled, so the compression benefit is smaller.
- Compared with [MXFP8](term_mxfp8.md): each FP16/BF16 element is represented independently and completely, with no block-level shared parameter; MXFP8 uses a block-level shared exponent to preserve the floating-point dynamic range within 8 bits.

---

## 3. Related Workflows

- [Quantization Data Type Ops Usage Guide](usage_dtype_ops.md): the 16-bit tensor before quantization serves as the input of the functions in this directory.
- [One-Click Quantization Usage Guide](../../user_guide/usage_one_click_quantization.md): the execution entry point for quantization tasks.
- [Quantization Accuracy Tuning Guide](../../user_guide/process_quantization_precision_tuning.md): accuracy problems caused by the data type and the quantization scheme can be diagnosed and tuned through this workflow.

---

## 4. Related Terms

- [Quantization Data Types](README.md): The parent concept and the directory this entry belongs to.
- [Quantization and Dequantization](term_quantization.md): A companion term; the quantization formulas take FP16/BF16 tensors as input.
- [INT8](term_int8.md): A companion term; the compression benefit of INT8 is measured against this format.
- [FP8](term_fp8.md): A sibling 8-bit floating-point format.
- [MXFP8](term_mxfp8.md): An 8-bit format with a block-level shared exponent.
- [W8A16 Static Quantization](../quantization_mode/linear_layer_quantization/term_w8a16_static.md): A companion mode; the activation stays FP16 and is not quantized.

---

## 5. Reference Documents

1. IEEE. "IEEE Standard for Floating-Point Arithmetic." IEEE 754-2019, 2019. <https://ieeexplore.ieee.org/document/8766229>
2. Kalamkar D, et al. "A Study of BFLOAT16 for Deep Learning Training." arXiv preprint arXiv:1905.12322, 2019. <https://arxiv.org/abs/1905.12322>
3. [Quantization Modes](../quantization_mode/README.md)
