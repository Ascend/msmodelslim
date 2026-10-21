# W4A4 MX Dual-Scale Quantization

> **Entry Category**: Quantization Data Format ([Linear Layer Quantization](README.md))<br>
> **English Name**: W4A4 MX Dual-Scale Quantization<br>
> **Application Domain**: Large language model quantization and compression, inference acceleration<br>
> **Hosting IR Class**: `W4A4MXDynamicDualScaleFakeQuantLinear` ([`msmodelslim/ir/w4a4_mx_dynamic_dualscale.py`](../../../../../msmodelslim/ir/w4a4_mx_dynamic_dualscale.py))<br>

## 1. Overview

W4A4 MX dual-scale quantization is the **accuracy-enhanced version** of [W4A4 MX Dynamic Quantization](term_w4a4_mx_dynamic.md): on top of the single-layer block-level exponent, it stacks another layer of large-block (such as 512-element) high-accuracy floating-point scale, forming a two-level scaling in which the outer level ensures the magnitude and the inner level preserves detail. It solves the insufficient accuracy of single-layer [MXFP4](../../quantization_basic/term_mxfp.md) when the magnitude varies greatly within a large block or strong outliers are present, trading a minimal amount of extra parameters for a clear accuracy improvement at 4bit.

---

## 2. Term Description

### 2.1 Mode Specification

| Dimension | Value | Description |
|------|------|------|
| Quantization Target | Weight W, Activation A | The tensors on both sides of the linear layer matrix multiplication |
| Bit Width | W 4bit, A 4bit | Both weight and activation memory traffic drop to 1/4 of FP16 |
| Data Type | MXFP4 dual scale | An outer large-block high-accuracy floating-point scale + an inner 32-element E8M0 exponent; the element is 1 sign + 2 exponent + 1 mantissa |
| Parameter Acquisition | Activation dynamic / weight static | The two levels of activation parameters are computed online in the forward pass; the two levels of weight parameters are fixed during quantization |
| Quantization Granularity | Dual-level block granularity | The outer `dual_block_size` (such as 512) shares a high-accuracy scale; the inner 32 elements share an E8M0 exponent |
| Symmetry | Symmetric | Scale/exponent only, no offset |

### 2.2 Quantization Formula

The MX format shares an E8M0 exponent per block; each element in the block is scaled by the block-level exponent and then quantized to a mantissa:
$$e = \lfloor \log_2(\max_{i \in \mathrm{block}}|x_i|) \rfloor, \qquad q_i = \mathrm{round}_{MX}\left(\frac{x_i}{2^e}\right), \qquad \hat{x}_i = q_i \cdot 2^e$$

Here the block size is 32 elements, $e$ is the E8M0 exponent shared within the block, $q_i$ is the quantized mantissa of the $i$-th element in the block (an FP8 mantissa for MXFP8 and an FP4 mantissa for MXFP4), and $\hat{x}_i$ is the dequantized reconstruction. Each block shares one $e$; quantization parameters are computed online (activation) or fixed during the quantization stage (weight). See the mode specification table above.

### 2.3 Relationship to Other Modes

- **Relationship to [W4A4 MX Dynamic Quantization](term_w4a4_mx_dynamic.md) (both are MXFP4, with different scaling levels)**:
  - This mode (dual scale): the advantage is that the two scaling levels handle the overall magnitude and the local dynamic range separately, making the accuracy clearly more stable for strong outliers or large blocks with large magnitude differences; the disadvantage is one extra layer of parameters and configuration (`dual_block_size`), making the implementation more complex.
  - Single layer: the advantage is a simple implementation with the lowest parameter overhead; the disadvantage is that a single block exponent can hardly cover both magnitude and detail, so the accuracy is limited under extreme distributions.

- **Relationship to [W4A4 Dynamic Quantization](term_w4a4_dynamic.md) (different data type and scaling)**:
  - This mode (MXFP4 dual scale): the advantage is the floating-point dynamic range plus two-level scaling, giving markedly better accuracy than uniform INT4 leveling; the disadvantage is the dependence on MX hardware support and the block-alignment constraint.
  - INT4: the advantage is a mature integer GEMM ecosystem and a simple implementation; the disadvantage is sensitivity to outliers and a limited dynamic range.

- **Relationship to [W4A8 MX Dynamic Quantization](term_w4a8_mx_dynamic.md) (different activation bit width)**:
  - This mode: the advantage is that the activation is also compressed to 4bit, maximizing the compression ratio, with the dual scale used to compensate for activation accuracy; the disadvantage is that the inherent 4bit accuracy risk is still higher than at 8bit.
  - W4A8 MX: the advantage is that the activation stays at 8bit and is naturally more stable; the disadvantage is that the activation memory-traffic benefit is smaller than with all-out 4bit.

### 2.4 Applicable Scenarios and Limitations

#### 2.4.1 Applicable Scenarios

- **W4A4 accuracy-sensitive deployment**: scenarios that need to approach FP16 accuracy as closely as possible at 4bit.
- **Ultra-low-bit quantization with outliers**: single-layer MXFP4 accuracy is insufficient and two-level scaling is needed to mitigate the effect of outliers.
- **Ascend MXFP inference path**: deployments where the hardware / operator library natively supports the MX format and dual-scale computation.

#### 2.4.2 Usage Limitations

- **Depends on hardware / operator support**: dual-scale computation requires support from the target hardware operator library.
- **Block granularity alignment**: both the outer large block and the inner small block sizes must divide the last dimension of the tensor.
- **Configuration constraint**: `dual_block_size` must be chosen reasonably according to the hardware and the tensor shape; an inappropriate value introduces extra overhead.
- **Inherent 4bit accuracy bar**: even with dual scale, the accuracy risk of 4bit activations/weights is still higher than that of 8bit schemes.

---

## 3. Related Workflows

- *[One-Click Quantization Usage Guide](../../../user_guide/usage_one_click_quantization.md)*: select this mode via `--quant_type` or a YAML configuration and run quantization.
- *[Quantization Accuracy Tuning Guide](../../../user_guide/process_quantization_precision_tuning.md)*: accuracy degradation caused by this mode can be rolled back layer by layer and tuned through that workflow.

---

## 4. Related Terms

- [Quantization Modes](../README.md): the parent concept; this entry belongs to the [Linear Layer Quantization](README.md) category and is one concrete mode under it.
- [W4A4 MX Dynamic Quantization](term_w4a4_mx_dynamic.md): a peer mode, the single-level-scaling MXFP4 base version.
- [W4A4 Dynamic Quantization](term_w4a4_dynamic.md): a contrasting mode; the INT4 integer approach.
- [W4A8 MX Dynamic Quantization](term_w4a8_mx_dynamic.md): a peer mode; activations use MXFP8 to preserve accuracy.
- [DualScale Dual-Scale Quantization Algorithm](../../quantization_algorithms/dual_scale/term_dual_scale.md): a companion term, the algorithm processor documentation corresponding to this mode.

---

## 5. Reference Documents

1. *[Linear Quantization Parameter Configuration Workflow Guide](../../quantization_algorithms/linear_quant/usage_linear_quant.md)*
2. *[DualScale Parameter Configuration Workflow Guide](../../quantization_algorithms/dual_scale/usage_dual_scale.md)*
3. *[Quantization Modes](../README.md)*
