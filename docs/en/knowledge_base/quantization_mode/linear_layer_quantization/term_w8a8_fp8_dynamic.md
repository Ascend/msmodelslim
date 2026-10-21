# W8A8 FP8 Dynamic Quantization

> **Entry Category**: Quantization Data Format ([Linear Layer Quantization](README.md))<br>
> **English Name**: W8A8 FP8 Dynamic Quantization<br>
> **Application Domain**: Large language model quantization and compression, inference acceleration<br>
> **Hosting IR Class**: `WFP8AFP8DynamicPerChannelFakeQuantLinear` ([`msmodelslim/ir/w8a8_fp_dynamic.py`](../../../../../msmodelslim/ir/w8a8_fp_dynamic.py))<br>

## 1. Overview

W8A8 FP8 dynamic quantization is the **[FP8 (E4M3)](../../quantization_basic/term_fp8.md) version** of W8A8 dynamic quantization: both the weights and the activations are stored as [FP8 (E4M3)](../../quantization_basic/term_fp8.md), and the activation quantization parameters are computed online per token. Compared with [INT8](../../quantization_basic/term_int8.md), the 4bit exponent of E4M3 preserves the floating-point dynamic range, making it more tolerant of activations with a wide numerical span and many outliers, with no need for outlier suppression first; the bit width is still 8bit, so the memory-traffic benefit is the same as [INT8](../../quantization_basic/term_int8.md). The cost is the dependence on hardware operators that support [FP8](../../quantization_basic/term_fp8.md) [GEMM](../term_gemm.md). The mode name follows the family notation `W{bit}A{bit} + data type`, where `W8A8 FP8` means that both W and A are 8bit and the data type is [FP8 (E4M3)](../../quantization_basic/term_fp8.md).

---

## 2. Term Description

### 2.1 Mode Specification

| Dimension | Value | Description |
|------|------|------|
| Quantization Target | Weight W, Activation A | The tensors on both sides of the linear layer matrix multiplication |
| Bit Width | W 8bit, A 8bit | Both 8bit; memory traffic is halved |
| Data Type | FP8 (E4M3) | Floating-point format, 1bit sign + 4bit exponent + 3bit mantissa, with a max value of 448 |
| Parameter Acquisition | Activation dynamic / weight static | The activation computes the scale online per token; the weight scale is fixed during quantization |
| Quantization Granularity | Weight per-channel; activation per-token | The weight shares a scale per output channel; the activation shares a scale per token |
| Symmetry | Symmetric | Scale only, no offset |

### 2.2 Quantization Formula

$$q = \mathrm{round}_{FP8}(x / s), \qquad \hat{x} = q \cdot s, \qquad s = \frac{\max(|x|)}{448}$$

Here $x$ is the tensor to quantize, $q$ is the quantized FP8 value ($\mathrm{round}_{FP8}$ denotes rounding to the nearest representable E4M3 value, whose maximum finite value is 448), $\hat{x}$ is the dequantized reconstruction, and $s$ is the scale. The FP8 (E4M3) format can represent a maximum value of 448, so $s$ is determined according to the representable upper limit of FP8. For how the quantization parameters are obtained and their granularity, see the mode specification table above.

### 2.3 Relationship to Other Modes

- **Relationship to [W8A8 Dynamic Quantization](term_w8a8_dynamic.md) (both are in the 8bit dynamic family, with different data types)**:
  - This mode (FP8 E4M3): the advantage is the floating-point format, whose 4bit exponent preserves the dynamic range, making it more tolerant of a wide dynamic range and outliers than uniform INT8 leveling; the disadvantage is that hardware support for FP8 GEMM is narrower than for INT8 and the operator ecosystem is less mature.
  - INT8: the advantage is a mature integer GEMM ecosystem and broad hardware support; the disadvantage is that uniform leveling is sensitive to outliers, so wide-dynamic-range activations often require suppression such as [SmoothQuant](../../quantization_algorithms/smooth_quant/term_smooth_quant.md) first.

- **Relationship to [W8A8 MX Dynamic Quantization](term_w8a8_mx_dynamic.md) (both are in the FP8 family, with different formats and granularity)**:
  - This mode: the advantage is per-channel/per-token granularity with per-element E4M3, a finer granularity that needs no block alignment; the disadvantage is that every channel/token must record an extra scale.
  - MXFP8: the advantage is the block-level shared E8M0 exponent, which lowers the scale storage overhead and expresses values more compactly; the disadvantage is the need for hardware MX support and for the block size (32 elements) to be aligned.

- **Relationship to [W8A16 Static Quantization](term_w8a16_static.md) (high-accuracy baseline)**:
  - This mode: the advantage is that the activation is also quantized and can use low-precision GEMM, giving thorough compression; the disadvantage is that the activation introduces FP8 quantization error.
  - W8A16: the advantage is a zero-error FP16 activation; the disadvantage is that the weight is dequantized back to floating point, with no low-precision GEMM speedup.

### 2.4 Applicable Scenarios and Limitations

#### 2.4.1 Applicable Scenarios

- **FP8-capable hardware deployment**: scenarios where the target hardware natively supports FP8 GEMM.
- **Wide-dynamic-range activations**: for models whose activations have a wide numerical span and many outliers, the FP8 floating-point format is more tolerant than uniform INT8 leveling.

#### 2.4.2 Usage Limitations

- **Hardware operator dependency**: FP8 GEMM requires support from the target hardware operator library; otherwise it can only be emulated in software and the benefit is discounted.
- **Overflow boundary**: the E4M3 representable range is limited (max 448), so out-of-range values rely on scale scaling to compensate, and extreme distributions still need verification.
- **Online reduction overhead**: per-token dynamic quantization incurs a small min/max reduction overhead.

---

## 3. Related Workflows

- *[One-Click Quantization Usage Guide](../../../user_guide/usage_one_click_quantization.md)*: select this mode via `--quant_type` or a YAML configuration and run quantization.
- *[Quantization Accuracy Tuning Guide](../../../user_guide/process_quantization_precision_tuning.md)*: accuracy degradation caused by this mode can be rolled back layer by layer and tuned through that workflow.

---

## 4. Related Terms

- [Quantization Modes](../README.md): the parent concept; this entry belongs to the [Linear Layer Quantization](README.md) category and is one concrete mode under it.
- [W8A8 Dynamic Quantization](term_w8a8_dynamic.md): a comparison mode, the isomorphic scheme with the INT8 data type.
- [W8A8 MX Dynamic Quantization](term_w8a8_mx_dynamic.md): a peer mode, the MXFP8 scheme with a block-level shared exponent.
- [W8A16 Static Quantization](term_w8a16_static.md): a comparison mode, the high-accuracy baseline that keeps the activation at 16bit.
- [FA FP8 Dynamic Quantization](../fa_quantization/term_fa_fp8_dynamic.md): a companion mode, attention activation quantization that likewise uses the FP8 data type.
- [Linear Quantization Algorithm](../../quantization_algorithms/linear_quant/term_linear_quant.md): a companion entry; the code implementation of this mode.

---

## 5. Reference Documents

1. Kuzmin A et al. FP8 Formats for Deep Learning. arXiv:2209.05433. <https://arxiv.org/abs/2209.05433>
2. *[Linear Quantization Parameter Configuration Workflow Guide](../../quantization_algorithms/linear_quant/usage_linear_quant.md)*
3. *[Quantization Modes](../README.md)*
