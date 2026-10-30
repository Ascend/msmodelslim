# W4A8 Dynamic Quantization

> **Entry Category**: Quantization Data Format ([Linear Layer Quantization](README.md))<br>
> **English Name**: W4A8 Dynamic Quantization<br>
> **Application Domain**: Large language model quantization and compression, inference acceleration<br>
> **Hosting IR Class**: `W4A8DynamicFakeQuantLinear` ([`msmodelslim/ir/w4a8_dynamic.py`](../../../../../msmodelslim/ir/w4a8_dynamic.py))<br>

## 1. Overview

W4A8 dynamic quantization is a **mixed bit-width** linear layer scheme in which the weight is compressed to [INT4](../../dtype_ops/term_int4.md) while the activation stays [INT8](../../dtype_ops/term_int8.md), with the activation quantization parameters computed online per token. It strikes a compromise between compression and accuracy: the weight is static data that must be read in full for every token, so compressing it to 4bit drops the weight memory traffic to 1/4 of [FP16/BF16](../../dtype_ops/term_fp16_bf16.md); the activation is the side most sensitive to quantization error, so it keeps 8bit resolution and is dynamically quantized per token—a typical configuration of compressing the weight hard while preserving activation accuracy.

---

## 2. Term Description

### 2.1 Mode Specification

| Dimension | Value | Description |
|------|------|------|
| Quantization Target | Weight W, Activation A | The tensors on both sides of the linear layer matrix multiplication |
| Bit Width | W 4bit, A 8bit | Mixed bit width; the weight memory traffic drops to 1/4 of [FP16/BF16](../../dtype_ops/term_fp16_bf16.md) and the activation to 1/2 |
| Data Type | Weight INT4; activation INT8 | Integer format; the weight is packed and stored in an INT8 container (two INT4 values in one byte) |
| Parameter Acquisition | Activation dynamic / weight static | The activation computes min/max online per token to derive the scale; the weight scale is fixed during quantization |
| Quantization Granularity | Weight per-channel; activation per-token | The weight shares a scale per output channel; the activation shares a scale per token |
| Symmetry | Symmetric | Scale only, no offset |

### 2.2 Quantization Formula

Quantization and dequantization use the standard linear mapping. Symmetric quantization (scale only, no zero point):
$$q = \mathrm{round}(x / s), \qquad \hat{x} = q \cdot s, \qquad s = \frac{\max(|x|)}{2^{b-1}-1}$$

Asymmetric quantization (with zero-point offset):
$$q = \mathrm{round}(x / s) + z, \qquad \hat{x} = (q - z) \cdot s, \qquad s = \frac{\max(x) - \min(x)}{2^b - 1}$$

Here $x$ is the tensor to quantize, $q$ is the quantized integer value, $\hat{x}$ is the dequantized reconstruction, $s$ is the scale, $z$ is the zero-point offset, and $b$ is the bit width ($b=8$ for INT8 and $b=4$ for INT4), with $z = \mathrm{round}(-\min(x)/s)$. For how the quantization parameters $s$ and $z$ are obtained (static calibration or online dynamic) and their granularity (per-channel, per-token, per-tensor, and so on), see the mode specification table above.

### 2.3 Relationship to Other Modes

- **Relationship to [W8A8 Dynamic Quantization](term_w8a8_dynamic.md) (the activation is also dynamic, with a different weight bit width)**:
  - This mode: the advantage is that the weight is compressed to 4bit and the memory traffic drops to 1/4 of FP16, giving more thorough compression on the weight side; the disadvantage is that signed INT4 can represent only 16 integer levels from -8 to 7, so there is an accuracy risk, and operators must handle the mixed weight/activation bit widths.
  - W8A8 dynamic: the advantage is the higher 8bit weight resolution and more stable accuracy with a simple operator implementation; the disadvantage is that the weight memory traffic is only halved.

- **Relationship to [W4A4 Dynamic Quantization](term_w4a4_dynamic.md) (both have a 4bit weight, with different activation bit widths)**:
  - This mode: the advantage is that the activation stays at 8bit, well away from the most sensitive activation quantization error, so the accuracy risk is far lower than W4A4; the disadvantage is that the activation memory traffic is only halved, so the compression margin is smaller than with all-out 4bit.
  - W4A4: the advantage is the maximum compression ratio with the activation bit width lowered as well; the disadvantage is the extremely high 4bit activation accuracy risk, requiring more compensating measures.

- **Relationship to [W8A16 Static Quantization](term_w8a16_static.md) (high-accuracy baseline)**:
  - This mode: the advantage is that both the weight and the activation are quantized, giving far more thorough compression than W8A16 (4bit weight + 8bit activation); the disadvantage is that the activation introduces quantization error, and when the weights/activations are dequantized back to floating point there is limited compute speedup, so the benefit comes mainly from memory traffic.
  - W8A16: the advantage is a zero-error FP16 activation and the most stable accuracy; the disadvantage is that the weight is only compressed to 8bit and must be dequantized back to floating point, with no low-precision GEMM speedup.

- **Relationship to [W4A8 MX Dynamic Quantization](term_w4a8_mx_dynamic.md) (both are W4A8, with different data types and granularity)**:
  - This mode (INT4 per-channel): the advantage is the integer format with a mature hardware operator ecosystem; the disadvantage is that uniform leveling is sensitive to weight outliers, and the per-channel granularity requires recording a scale per output channel.
  - MXFP4: the advantage is that a block-level shared exponent preserves the floating-point dynamic range, making it more tolerant of outliers; the disadvantage is the dependence on hardware MX support and the need to align the block size.

### 2.4 Applicable Scenarios and Limitations

#### 2.4.1 Applicable Scenarios

- **Weight-memory/bandwidth-dominated deployment**: scenarios where the model weights are the main memory consumer and weight reads are the throughput bottleneck during decode (the inference phase that generates tokens one by one).
- **Accuracy/compression compromise**: scenarios that need stronger weight compression than W8A8 but dare not compress the activation to 4bit.

#### 2.4.2 Usage Limitations

- **Weight accuracy risk**: signed INT4 can represent only 16 integer levels from -8 to 7 and is sensitive to outliers, so outlier suppression or finer grouping is needed.
- **Mixed bit-width operator dependency**: INT4 x INT8 mixed bit-width GEMM requires support from the target hardware operator library; otherwise the benefit is discounted.

---

## 3. Related Workflows

- *[One-Click Quantization Usage Guide](../../../user_guide/usage_one_click_quantization.md)*: select this mode via `--quant_type` or a YAML configuration and run quantization.
- *[Quantization Accuracy Tuning Guide](../../../user_guide/process_quantization_precision_tuning.md)*: accuracy degradation caused by this mode can be rolled back layer by layer and tuned through that workflow.

---

## 4. Related Terms

- [Quantization Modes](../README.md): the parent concept; this entry belongs to the [Linear Layer Quantization](README.md) category and is one concrete mode under it.
- [W8A8 Dynamic Quantization](term_w8a8_dynamic.md): a comparison mode with the weight raised to 8bit and more stable accuracy.
- [W4A4 Dynamic Quantization](term_w4a4_dynamic.md): a comparison mode with the activation compressed to 4bit for heavier compression.
- [W8A16 Static Quantization](term_w8a16_static.md): a comparison mode, the high-accuracy baseline that does not quantize the activation.
- [W4A8 MX Dynamic Quantization](term_w4a8_mx_dynamic.md): a peer mode, using the MX floating-point format to carry the same bit widths.
- [Linear Quantization Algorithm](../../quantization_algorithms/linear_quant/term_linear_quant.md): a companion entry; the code implementation of this mode.

---

## 5. Reference Documents

1. *[Linear Quantization Parameter Configuration Workflow Guide](../../quantization_algorithms/linear_quant/usage_linear_quant.md)*
2. *[Quantization Modes](../README.md)*
