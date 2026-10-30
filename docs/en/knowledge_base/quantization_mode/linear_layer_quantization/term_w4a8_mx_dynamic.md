# W4A8 MX Dynamic Quantization

> **Entry Category**: Quantization Data Format ([Linear Layer Quantization](README.md))<br>
> **English Name**: W4A8 MX Dynamic Quantization<br>
> **Application Domain**: Large language model quantization and compression, inference acceleration<br>
> **Hosting IR Class**: `W4A8MXDynamicPerBlockFakeQuantLinear` ([`msmodelslim/ir/w4a8_mx_dynamic.py`](../../../../../msmodelslim/ir/w4a8_mx_dynamic.py))<br>

## 1. Overview

W4A8 MX dynamic quantization swaps the W4A8 bit-width pairing for the MX format: the weight uses [MXFP4](../../dtype_ops/term_mxfp4.md) (1 sign + 2 exponent + 1 mantissa) for 4bit compression, while the activation uses [MXFP8](../../dtype_ops/term_mxfp8.md) (E4M3) to preserve 8bit accuracy, with both sides sharing an E8M0 exponent per 32-element block. Compared with the [INT4](../../dtype_ops/term_int4.md)/[INT8](../../dtype_ops/term_int8.md) combination, the MX floating-point format is more tolerant of outliers; compared with an all-out 4bit scheme, keeping the activation at 8bit markedly lowers the accuracy risk—it is the MX-format compromise of extreme weight compression plus preserved activation accuracy.

---

## 2. Term Description

### 2.1 Mode Specification

| Dimension | Value | Description |
|------|------|------|
| Quantization Target | Weight W, Activation A | The tensors on both sides of the linear layer matrix multiplication |
| Bit Width | W 4bit, A 8bit | The weight memory traffic drops to 1/4 of FP16 and the activation to 1/2 |
| Data Type | Weight MXFP4; activation MXFP8 | MXFP4 emax=2, max 6; MXFP8 emax=8, max 448; both share an E8M0 exponent per 32 elements |
| Parameter Acquisition | Activation dynamic / weight static | Activation block-level exponents are counted online in the forward pass; weight block-level exponents are fixed during quantization |
| Quantization Granularity | Weight/activation both per-block (32 elements) | The last dimension is divided into blocks of 32 elements, and each block shares one exponent |
| Symmetry | Symmetric | Exponent/scale only, no offset |

### 2.2 Quantization Formula

The MX format shares an E8M0 exponent per block; each element in the block is scaled by the block-level exponent and then quantized to a mantissa:
$$e = \lfloor \log_2(\max_{i \in \mathrm{block}}|x_i|) \rfloor, \qquad q_i = \mathrm{round}_{MX}\left(\frac{x_i}{2^e}\right), \qquad \hat{x}_i = q_i \cdot 2^e$$

Here the block size is 32 elements, $e$ is the E8M0 exponent shared within the block, $q_i$ is the quantized mantissa of the $i$-th element in the block (an FP8 mantissa for MXFP8 and an FP4 mantissa for MXFP4), and $\hat{x}_i$ is the dequantized reconstruction. Each block shares one $e$; quantization parameters are computed online (activation) or fixed during the quantization stage (weight). See the mode specification table above.

### 2.3 Relationship to Other Modes

- **Relationship to [W4A8 Dynamic Quantization](term_w4a8_dynamic.md) (both are W4A8, with different data types and granularity)**:
  - This mode (MX format): the advantage is that the MXFP4 weight shares an exponent per block, making it more tolerant of outliers, and the MXFP8 activation preserves a wider floating-point dynamic range; the disadvantage is the dependence on hardware MX support and the requirement that the last dimension (such as in_dim / head_dim) be a multiple of 32.
  - INT4/INT8: the advantage is the integer format with a mature hardware operator ecosystem; the disadvantage is that uniform leveling is sensitive to outliers.

- **Relationship to [W8A8 MX Dynamic Quantization](term_w8a8_mx_dynamic.md) (different weight bit width)**:
  - This mode: the advantage is that the weight is compressed to 4bit and the memory traffic drops to 1/4 of FP16; the disadvantage is that the weight has only a few representable values, so the accuracy risk is higher than at 8bit.
  - W8A8 MX: the advantage is more stable 8bit weight accuracy; the disadvantage is that the weight memory traffic is only halved.

- **Relationship to [W4A4 MX Dynamic Quantization](term_w4a4_mx_dynamic.md) (different activation bit width)**:
  - This mode: the advantage is that the activation stays at 8bit, well away from the most sensitive activation quantization error, making it a compromise between compression and accuracy; the disadvantage is that the activation memory-traffic benefit is smaller than with all-out 4bit.
  - W4A4 MX: the advantage is the maximum dual-4bit compression; the disadvantage is the high 4bit activation accuracy risk.

### 2.4 Applicable Scenarios and Limitations

#### 2.4.1 Applicable Scenarios

- **Weight-memory-limited with a wide activation distribution range**: scenarios that need 4bit weight compression while worrying that INT8 activation accuracy is insufficient.
- **Outlier-sensitive low-bit deployment**: the MX floating-point format is more tolerant of outliers than integer formats.
- **Ascend MXFP inference path**: the hardware/operator library natively supports deployment in the MX format.

#### 2.4.2 Usage Limitations

- **Depends on hardware MX support**: MXFP4/MXFP8 computation requires support from the target hardware operator library.
- **Block granularity alignment**: the length of the tensor's last dimension (such as in_dim / head_dim) must be a multiple of 32; otherwise padding is required.
- **4bit weight accuracy risk**: MXFP4 has only 16 representable values, so accuracy still needs to be verified under extreme weight distributions.

---

## 3. Related Workflows

- *[One-Click Quantization Usage Guide](../../../user_guide/usage_one_click_quantization.md)*: select this mode via `--quant_type` or a YAML configuration and run quantization.
- *[Quantization Accuracy Tuning Guide](../../../user_guide/process_quantization_precision_tuning.md)*: accuracy degradation caused by this mode can be rolled back layer by layer and tuned through that workflow.

---

## 4. Related Terms

- [Quantization Modes](../README.md): the parent concept; this entry belongs to the [Linear Layer Quantization](README.md) category and is one concrete mode under it.
- [W4A8 Dynamic Quantization](term_w4a8_dynamic.md): a comparison mode, the integer scheme with an INT4 weight + INT8 activation.
- [W8A8 MX Dynamic Quantization](term_w8a8_mx_dynamic.md): a peer mode, with the weight also MXFP8.
- [W4A4 MX Dynamic Quantization](term_w4a4_mx_dynamic.md): a peer mode, with the activation also lowered to MXFP4.
- [Linear Quantization Algorithm](../../quantization_algorithms/linear_quant/term_linear_quant.md): a companion entry; the code implementation of this mode.

---

## 5. Reference Documents

1. *[Linear Quantization Parameter Configuration Workflow Guide](../../quantization_algorithms/linear_quant/usage_linear_quant.md)*
2. *[Quantization Modes](../README.md)*
