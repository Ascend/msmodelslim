# W8A8 Dynamic Quantization

> **Entry Category**: Quantization Data Format ([Linear Layer Quantization](README.md))<br>
> **English Name**: W8A8 Dynamic Quantization<br>
> **Application Domain**: Large language model quantization and compression, inference acceleration<br>
> **Hosting IR Class**: `W8A8DynamicPerChannelFakeQuantLinear` / `W8A8DynamicPerGroupFakeQuantLinear` ([`msmodelslim/ir/w8a8_dynamic.py`](../../../../../msmodelslim/ir/w8a8_dynamic.py))<br>

## 1. Overview

W8A8 dynamic quantization applies [INT8](../../dtype_ops/term_int8.md) quantization to both the weights and the activations of linear layers, where **the activation quantization parameters are computed online per token at inference**. Its bit width is exactly the same as [W8A8 Static Quantization](term_w8a8_static.md); the only difference is the parameter acquisition: static fixes them by offline calibration, while dynamic counts them online. The dynamic scheme needs no calibration and fits each token's own numerical range better, at the cost of one extra min/max reduction on every forward pass.

---

## 2. Term Description

### 2.1 Mode Specification

| Dimension | Value | Description |
|------|------|------|
| Quantization Target | Weight W, Activation A | The tensors on both sides of the linear layer matrix multiplication |
| Bit Width | W 8bit, A 8bit | Both 8bit; memory traffic is halved |
| Data Type | INT8 | Integer format; can use INT8 integer GEMM |
| Parameter Acquisition | Activation dynamic / weight static | The activation computes min/max online for each token to derive the scale, with no calibration and adapting to the input; the weight scale is fixed during quantization |
| Quantization Granularity | Weight per-channel/per-group; activation per-token | The weight shares a scale per output channel or per fixed group (such as 128 elements); the activation shares a scale per token (one row) |
| Symmetry | Activation symmetric; weight symmetric (per-group supports asymmetric) | The activation is symmetric with only a scale; the weight can be per-group with an offset |

### 2.2 Quantization Formula

Quantization and dequantization use the standard linear mapping. Symmetric quantization (scale only, no zero point):
$$q = \mathrm{round}(x / s), \qquad \hat{x} = q \cdot s, \qquad s = \frac{\max(|x|)}{2^{b-1}-1}$$

Asymmetric quantization (with zero-point offset):
$$q = \mathrm{round}(x / s) + z, \qquad \hat{x} = (q - z) \cdot s, \qquad s = \frac{\max(x) - \min(x)}{2^b - 1}$$

Here $x$ is the tensor to quantize, $q$ is the quantized integer value, $\hat{x}$ is the dequantized reconstruction, $s$ is the scale, $z$ is the zero-point offset, and $b$ is the bit width ($b=8$ for INT8 and $b=4$ for INT4), with $z = \mathrm{round}(-\min(x)/s)$. For how the quantization parameters $s$ and $z$ are obtained (static calibration or online dynamic) and their granularity (per-channel, per-token, per-tensor, and so on), see the mode specification table above.

### 2.3 Relationship to Other Modes

- **Relationship to [W8A8 Static Quantization](term_w8a8_static.md) (same bit width, different parameter acquisition)**:
  - This mode (dynamic): the advantage is that the scale is computed online per token with no calibration, so the activation quantization fits each token's numerical range, giving higher accuracy than static per-tensor and robustness to input distribution drift; the disadvantage is one extra min/max reduction on every forward pass, which adds a small latency overhead.
  - Static: the advantage is that the activation scale is fixed offline with zero online overhead at inference; the disadvantage is that the coarse per-tensor granularity depends on calibration data and has a lower accuracy ceiling.

- **Relationship to [W8A8 PD-Mix Quantization](term_w8a8_pdmix.md) (both are W8A8, with the activation strategy differing by phase)**:
  - This mode: per-token dynamic in all phases, with a uniform activation accuracy; the disadvantage is that in the decode phase each step has only one token—the activation is the whole tensor, so the per-token and per-tensor granularities are equivalent, and the per-token reduction is both statistically meaningless and an unnecessary overhead.
  - PD-Mix: dynamic in prefill to preserve accuracy and static in decode for zero overhead, each phase being optimal; the disadvantage is that it depends on the inference framework being phase-aware and switching, and it is currently limited to MindIE deployment.

- **Relationship to [W4A4 Dynamic Quantization](term_w4a4_dynamic.md) (both are in the dynamic family, with different bit widths)**:
  - This mode: the advantage is that 8bit has higher resolution and more stable accuracy, making it the general choice; the disadvantage is that the weight memory traffic is only halved (1/2 of FP16).
  - W4A4: the advantage is that the weight memory traffic drops to 1/4 of FP16, giving more thorough compression; the disadvantage is that signed INT4 can represent only 16 integer levels from -8 to 7, so the accuracy risk is high and finer granularity and outlier suppression are needed to compensate.

### 2.4 Applicable Scenarios and Limitations

#### 2.4.1 Applicable Scenarios

- **Insufficient calibration data or an uncertain distribution**: calibration-free, suitable for scenarios without a representative calibration set or where the input distribution varies greatly at deployment.
- **Accuracy requirements higher than static**: needs better activation accuracy than static W8A8 while accepting a small amount of online reduction overhead.

#### 2.4.2 Usage Limitations

- **Online reduction overhead**: the per-token min/max reduction adds a small amount of latency, and the overhead share is more noticeable for layers with very little computation per token.
- **Hardware operator dependency**: per-token dynamic quantization requires the inference framework / operator library to support per-token parameter computation; otherwise it can only be emulated in software and the benefit is discounted.

---

## 3. Related Workflows

- *[One-Click Quantization Usage Guide](../../../user_guide/usage_one_click_quantization.md)*: select this mode via `--quant_type` or a YAML configuration and run quantization.
- *[Quantization Accuracy Tuning Guide](../../../user_guide/process_quantization_precision_tuning.md)*: accuracy degradation caused by this mode can be rolled back layer by layer and tuned through that workflow.

---

## 4. Related Terms

- [Quantization Modes](../README.md): the parent concept; this entry belongs to the [Linear Layer Quantization](README.md) category and is one concrete mode under it.
- [W8A8 Static Quantization](term_w8a8_static.md): a comparison mode, with the activation parameters fixed offline.
- [W8A8 PD-Mix Quantization](term_w8a8_pdmix.md): a peer mode, with the activation strategy switching with the Prefill/Decode phase.
- [W4A4 Dynamic Quantization](term_w4a4_dynamic.md): a peer mode; lower-bit weight + activation quantization.
- [FA INT8 Dynamic Quantization](../fa_quantization/term_fa_int8_dynamic.md): a peer mode that applies per-token dynamic quantization to the attention Q/K/V activations.
- [Linear Quantization Algorithm](../../quantization_algorithms/linear_quant/term_linear_quant.md): a companion entry; the code implementation of this mode.

---

## 5. Reference Documents

1. *[Linear Quantization Parameter Configuration Workflow Guide](../../quantization_algorithms/linear_quant/usage_linear_quant.md)*
2. *[Quantization Modes](../README.md)*
