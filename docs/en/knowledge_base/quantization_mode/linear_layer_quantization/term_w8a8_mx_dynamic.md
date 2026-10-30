# W8A8 MX Dynamic Quantization

> **Entry Category**: Quantization Data Format ([Linear Layer Quantization](README.md))<br>
> **English Name**: W8A8 MX Dynamic Quantization<br>
> **Application Domain**: Large language model quantization and compression, inference acceleration<br>
> **Hosting IR Class**: `W8A8MXDynamicPerBlockFakeQuantLinear` ([`msmodelslim/ir/w8a8_mx_dynamic.py`](../../../../../msmodelslim/ir/w8a8_mx_dynamic.py))<br>

## 1. Overview

W8A8 MX dynamic quantization is the **[MXFP8](../../dtype_ops/term_mxfp8.md) version** of W8A8: both the weights and the activations are divided into 32-element blocks, each sharing an 8bit exponent (E8M0), with 8bit floating-point (E4M3) elements. The block-level shared exponent amortizes the scaling-parameter overhead from one per element to one per 32 elements while preserving the floating-point dynamic range—it is more tolerant of outliers than uniform [INT8](../../dtype_ops/term_int8.md) leveling, and has a lower per-element scale parameter overhead than ordinary [FP8 (E4M3)](../../dtype_ops/term_fp8.md).

---

## 2. Term Description

### 2.1 Mode Specification

| Dimension | Value | Description |
|------|------|------|
| Quantization Target | Weight W, Activation A | The tensors on both sides of the linear layer matrix multiplication |
| Bit Width | W 8bit, A 8bit | Element bit width 8bit; the memory traffic is halved |
| Data Type | MXFP8 | One E8M0 8bit exponent shared per 32 elements, with E4M3 elements (emax=8, max 448) |
| Parameter Acquisition | Activation dynamic / weight static | Activation block-level exponents are counted online in the forward pass; weight block-level exponents are fixed during quantization |
| Quantization Granularity | Weight/activation both per-block (32 elements) | The last dimension is divided into blocks of 32 elements, and each block shares one exponent |
| Symmetry | Symmetric | Exponent/scale only, no offset |

### 2.2 Quantization Formula

The MX format shares an E8M0 exponent per block; each element in the block is scaled by the block-level exponent and then quantized to a mantissa:
$$e = \lfloor \log_2(\max_{i \in \mathrm{block}}|x_i|) \rfloor, \qquad q_i = \mathrm{round}_{MX}\left(\frac{x_i}{2^e}\right), \qquad \hat{x}_i = q_i \cdot 2^e$$

Here the block size is 32 elements, $e$ is the E8M0 exponent shared within the block, $q_i$ is the quantized mantissa of the $i$-th element in the block (an FP8 mantissa for MXFP8 and an FP4 mantissa for MXFP4), and $\hat{x}_i$ is the dequantized reconstruction. Each block shares one $e$; quantization parameters are computed online (activation) or fixed during the quantization stage (weight). See the mode specification table above.

### 2.3 Relationship to Other Modes

- **Relationship to [W8A8 Dynamic Quantization](term_w8a8_dynamic.md) (both are in the 8bit dynamic family, with different quantization granularity)**:
  - A token is the basic unit of inference and usually consists of N channels. This mode quantizes per-block: every 32 channels of each token share one set of quantization parameters, characterizing the numerical differences between channels more finely.
  - W8A8 dynamic quantization quantizes per-token: each token (that is, all N channels) shares one set of quantization parameters, giving a smaller parameter overhead but a coarser characterization of the differences between channels.

- **Relationship to [W8A8 FP8 Dynamic Quantization](term_w8a8_fp8_dynamic.md) (both are in the 8bit dynamic family, with different quantization granularity)**:
  - This mode is still per-block: every 32 channels of each token share one set of quantization parameters.
  - W8A8 FP8 dynamic quantization quantizes per-token or per-channel: each token or each channel shares one set of parameters, a granularity different from this mode.

- **Relationship to [W4A4 MX Dynamic Quantization](term_w4a4_mx_dynamic.md) (different bit width)**:
  - This mode: the advantage is more stable 8bit accuracy; the disadvantage is that the weight memory traffic is only halved.
  - W4A4 MX: the advantage is that both the weight and the activation are compressed to 1/4 of FP16, maximizing the compression ratio; the disadvantage is the high 4bit activation accuracy risk.

### 2.4 Applicable Scenarios and Limitations

#### 2.4.1 Applicable Scenarios

- **Low-bit deployment with outliers**: compared with INT8, MXFP8 has a larger floating-point dynamic range and is more tolerant of outliers.
- **8bit with a need for low parameter overhead**: the block-level shared exponent is better than per-element scale schemes in parameter overhead.
- **Ascend MXFP inference path**: the hardware/operator library natively supports deployment in the MX format.

#### 2.4.2 Usage Limitations

- **Depends on hardware MX support**: MXFP8 computation requires support from the target hardware operator library; otherwise the benefit cannot be realized.
- **Block granularity alignment**: the length of the tensor's last dimension (such as in_dim / head_dim) must be a multiple of 32; otherwise padding is required.
- **Dynamic reduction overhead**: the online per-block statistics of the activation add a small amount of latency.

---

## 3. Related Workflows

- *[One-Click Quantization Usage Guide](../../../user_guide/usage_one_click_quantization.md)*: select this mode via `--quant_type` or a YAML configuration and run quantization.
- *[Quantization Accuracy Tuning Guide](../../../user_guide/process_quantization_precision_tuning.md)*: accuracy degradation caused by this mode can be rolled back layer by layer and tuned through that workflow.

---

## 4. Related Terms

- [Quantization Modes](../README.md): the parent concept; this entry belongs to the [Linear Layer Quantization](README.md) category and is one concrete mode under it.
- [W8A8 Dynamic Quantization](term_w8a8_dynamic.md): a comparison mode, the INT8 integer scheme.
- [W8A8 FP8 Dynamic Quantization](term_w8a8_fp8_dynamic.md): a comparison mode, the ordinary per-element FP8 scaling scheme.
- [W4A4 MX Dynamic Quantization](term_w4a4_mx_dynamic.md): a peer mode, the lower-bit MXFP4 scheme.
- [Linear Quantization Algorithm](../../quantization_algorithms/linear_quant/term_linear_quant.md): a companion entry; the code implementation of this mode.

---

## 5. Reference Documents

1. *[Linear Quantization Parameter Configuration Workflow Guide](../../quantization_algorithms/linear_quant/usage_linear_quant.md)*
2. *[Quantization Modes](../README.md)*
