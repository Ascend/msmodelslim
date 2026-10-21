# W4A4 MX Dynamic Quantization

> **Entry Category**: Quantization Data Format ([Linear Layer Quantization](README.md))<br>
> **English Name**: W4A4 MX Dynamic Quantization<br>
> **Application Domain**: Large language model quantization and compression, inference acceleration<br>
> **Hosting IR Class**: `W4A4MXDynamicPerBlockFakeQuantLinear` ([`msmodelslim/ir/w4a4_mx_dynamic.py`](../../../../../msmodelslim/ir/w4a4_mx_dynamic.py))<br>

## 1. Overview

W4A4 MX dynamic quantization is a dual-4bit MX ultra-low-bit scheme: the weights and activations both share an E8M0 exponent per 32-element block, with [MXFP4](../../quantization_basic/term_mxfp.md) elements. At 4bit it uses the block-level shared exponent to preserve the floating-point dynamic range, and both the weight and activation memory traffic drop to 1/4 of [FP16/BF16](../../quantization_basic/term_fp16_bf16.md), making it the linear layer scheme with the highest compression ratio; it is more tolerant of outliers than uniform [INT4](../../quantization_basic/term_int4.md) leveling, but the inherent accuracy risk of a 4bit activation remains.

---

## 2. Term Description

### 2.1 Mode Specification

| Dimension | Value | Description |
|------|------|------|
| Quantization Target | Weight W, Activation A | The tensors on both sides of the linear layer matrix multiplication |
| Bit Width | W 4bit, A 4bit | Both weight and activation memory traffic drop to 1/4 of FP16 |
| Data Type | MXFP4 | One E8M0 exponent shared per 32 elements; the element is 1 sign bit + 2 exponent bits + 1 mantissa bit (emax=2, max 6) |
| Parameter Acquisition | Activation dynamic / weight static | Activation block-level exponents are counted online in the forward pass; weight block-level exponents are fixed during quantization |
| Quantization Granularity | Weight/activation both per-block (32 elements) | The last dimension is divided into blocks of 32 elements, and each block shares one exponent |
| Symmetry | Symmetric | Exponent/scale only, no offset |

### 2.2 Quantization Formula

The MX format shares an E8M0 exponent per block; each element in the block is scaled by the block-level exponent and then quantized to a mantissa:
$$e = \lfloor \log_2(\max_{i \in \mathrm{block}}|x_i|) \rfloor, \qquad q_i = \mathrm{round}_{MX}\left(\frac{x_i}{2^e}\right), \qquad \hat{x}_i = q_i \cdot 2^e$$

Here the block size is 32 elements, $e$ is the E8M0 exponent shared within the block, $q_i$ is the MX-format quantized value of the $i$-th element in the block (an FP8 value for MXFP8 and an FP4 value for MXFP4), and $\hat{x}_i$ is the dequantized reconstruction. Each block shares one $e$; the quantization parameters are computed online (activation) or fixed during the quantization stage (weight); see the mode specification table above.

### 2.3 Relationship to Other Modes

- **Relationship to [W4A4 Dynamic Quantization](term_w4a4_dynamic.md) (both are W4A4, with different data types)**:
  - This mode (MXFP4): the advantage is that a block-level shared exponent preserves the floating-point dynamic range, making it more tolerant of outliers than uniform [INT4](../../quantization_basic/term_int4.md) leveling; the disadvantage is the dependence on hardware MX support and the requirement that the last dimension (such as in_dim / head_dim) be a multiple of 32.
  - INT4: the advantage is a mature integer GEMM ecosystem and a simple implementation; the disadvantage is that uniform leveling is sensitive to outliers and needs finer granularity and outlier suppression to compensate.

- **Relationship to [W4A4 MX Dual-Scale Quantization](term_w4a4_mx_dualscale.md) (both are MXFP4, with different scaling levels)**:
  - This mode (single layer): the advantage is a simple implementation with the lowest parameter overhead; the disadvantage is that when the magnitude varies greatly within a large block or strong outliers are present, a single exponent can hardly cover both the overall magnitude and the local detail.
  - Dual scale: the advantage is that the outer high-accuracy scale prevents overflow while the inner exponent preserves detail, making the accuracy clearly more stable; the disadvantage is one extra layer of parameters and configuration (`dual_block_size`), making the implementation more complex.

- **Relationship to [W4A8 MX Dynamic Quantization](term_w4a8_mx_dynamic.md) (different activation bit width)**:
  - This mode: the advantage is that the activation is also compressed to 4bit, maximizing the compression ratio; the disadvantage is that the 4bit activation accuracy risk is far higher than at 8bit.
  - W4A8 MX: the advantage is that the activation stays at 8bit for more stable accuracy, making it a compromise between compression and accuracy.

### 2.4 Applicable Scenarios and Limitations

#### 2.4.1 Applicable Scenarios

- **Extreme compression deployment**: scenarios limited by memory or bandwidth and pursuing the maximum compression ratio, such as large-scale server-side multi-model deployment.
- **Outlier-sensitive but needing 4bit**: scenarios that need higher accuracy than INT4 at 4bit.
- **Ascend MXFP inference path**: the hardware/operator library natively supports deployment in the MX format.

#### 2.4.2 Usage Limitations

- **Depends on hardware MX support**: MXFP4 computation requires support from the target hardware operator library.
- **Block granularity alignment**: the length of the tensor's last dimension (such as in_dim / head_dim) must be a multiple of 32; otherwise padding is required.
- **Low-bit accuracy bar**: 4bit quantization carries a large accuracy risk and usually needs outlier suppression together with accuracy tuning.
- **4bit activation risk**: a 4bit activation is extremely sensitive to the distribution, a higher risk than the W4A8 MX scheme.

---

## 3. Related Workflows

- *[One-Click Quantization Usage Guide](../../../user_guide/usage_one_click_quantization.md)*: select this mode via `--quant_type` or a YAML configuration and run quantization.
- *[Quantization Accuracy Tuning Guide](../../../user_guide/process_quantization_precision_tuning.md)*: accuracy degradation caused by this mode can be rolled back layer by layer and tuned through that workflow.

---

## 4. Related Terms

- [Quantization Modes](../README.md): the parent concept; this entry belongs to the [Linear Layer Quantization](README.md) category and is one concrete mode under it.
- [W4A4 Dynamic Quantization](term_w4a4_dynamic.md): a contrasting mode; the INT4 integer approach.
- [W4A4 MX Dual-Scale Quantization](term_w4a4_mx_dualscale.md): a peer mode, the accuracy-enhanced version with dual-level scaling.
- [W4A8 MX Dynamic Quantization](term_w4a8_mx_dynamic.md): a peer mode; activations use MXFP8 to preserve accuracy.
- [Linear Quantization Algorithm](../../quantization_algorithms/linear_quant/term_linear_quant.md): a companion entry; the code implementation of this mode.

---

## 5. Reference Documents

1. *[Linear Quantization Parameter Configuration Workflow Guide](../../quantization_algorithms/linear_quant/usage_linear_quant.md)*
2. *[Quantization Modes](../README.md)*
