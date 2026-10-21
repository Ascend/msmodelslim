# FA FP8 Dynamic Quantization

> **Entry Category**: Quantization Data Format ([FA Quantization](README.md))<br>
> **English Name**: FA FP8 Dynamic Quantization<br>
> **Application Domain**: Long-context inference acceleration, low-precision attention computation<br>
> **Hosting IR Class**: `FakeQuantActivationPerToken` ([`msmodelslim/ir/activation_dynamic.py`](../../../../../msmodelslim/ir/activation_dynamic.py))<br>

---

## 1. Overview

FA FP8 dynamic quantization is a holistic composite mode in which the Q/K/V activations fed into Flash Attention are uniformly dynamically quantized to [FP8 (E4M3)](../../quantization_basic/term_fp8.md) row by row per token. The quantization parameters are computed online with no calibration, so the activation resolution of each token fits its own numerical range. It is a common mode for calibration-free attention quantization in scenarios such as multimodal generation and long-sequence decoding; K/V carry cache compression ([KVCache Quantization](../kv_cache_quantization/README.md)) and Q enables low-precision matrix computation.

---

## 2. Term Description

### 2.1 Mode Specification

| Dimension | Value | Description |
| --- | --- | --- |
| Quantization Target | Attention Q/K/V activations | The activation tensors fed into Flash Attention (shape (B, H, S, D)); all three branches `fa_q`/`fa_k`/`fa_v` are quantized. K/V carry cache compression, and Q enables low-precision score computation |
| Bit Width | 8bit | Q/K/V are all 8bit |
| Data Type | FP8 (E4M3) | 8bit floating point, 4 exponent bits and 3 mantissa bits, maximum representable value 448 |
| Parameter Acquisition | Dynamic | Computes min/max online per token in the forward pass and derives the scale; no calibration needed |
| Quantization Granularity | per-token | After reshaping to (B\*H\*S, D), the scale is shared per row (token) |
| Symmetry | Symmetric | Scale only, no offset |

### 2.2 Quantization Formula

Symmetric quantization (scale only, no zero point):

$$q = \mathrm{round}(x / s), \qquad \hat{x} = q \cdot s, \qquad s = \frac{\max(|x|)}{448}$$

In the formula above, $x$ is the tensor to quantize, $q$ is the quantized value of that tensor, $\hat{x}$ is the dequantized reconstruction, and $s$ is the scale. The FP8 (E4M3) format can represent a maximum value of 448, so the scale must be determined according to the representable range of FP8. At per-token granularity, the input is first reshaped to (B\*H\*S, D), after which min/max is computed row by row and the scale of each token is calculated dynamically.

### 2.3 Relationship to Other Modes

- **Relationship to [FA INT8 Dynamic Quantization](term_fa_int8_dynamic.md) (same uniform per-token dynamic, different data type)**:
  - This mode uses the FP8 (E4M3) floating-point format; INT8 dynamic uses the INT8 integer format. FP8 floating point has a larger dynamic range and is relatively more tolerant of outliers, but the E4M3 mantissa is only 3 bits; the choice between the two depends on the deployment operator and the accuracy trade-off.
- **Relationship to [FA Q-FP8 Dynamic K/V-FP8 Static Quantization](term_fa_q_fp8_dynamic_kv_fp8.md) (same FP8 family, different branch quantization)**:
  - This mode uses unified per-token dynamic quantization for all three branches; the mixed mode changes K/V to per-head static, trading K/V calibration for lower online overhead.
- **Relationship to [W8A8 FP8 Dynamic Quantization](../linear_layer_quantization/term_w8a8_fp8_dynamic.md) (same FP8 per-token dynamic idea)**:
  - This mode applies the same idea to the attention Q/K/V activations; W8A8 FP8 dynamic applies it to linear layer weights and activations.

### 2.4 Applicable Scenarios and Limitations

#### 2.4.1 Applicable Scenarios

- **Long-context / multimodal generation**: the recommended default for models such as Wan2.2, Flux1, HunYuanVideo, and QwenImageEdit; dynamic calibration-free quantization is more robust to changes in the multimodal input distribution.
- **Calibration data unreliable or missing**: when no representative calibration set is available, the dynamic per-token mode is more robust.
- **High attention accuracy requirements**: a finer-grained activation quantization than static quantization is needed, and the online reduction overhead is acceptable.

#### 2.4.2 Usage Limitations

- **Online reduction overhead**: the per-token min/max reduction adds a small amount of latency, which is more noticeable for short sequences.
- **Limited FP8 representable range**: E4M3 has a max of 448, so large outliers need attention; the quantization error directly affects attention scores, and accuracy under extremely long sequences must be verified.
- **Hardware operator dependency**: FP8 per-token dynamic quantization requires the inference framework / operator library to support FP8 attention computation.

---

## 3. Related Workflows

- *[One-Click Quantization Usage Guide](../../../user_guide/usage_one_click_quantization.md)*: select this mode via a YAML configuration and run quantization.
- *[Quantization Accuracy Tuning Guide](../../../user_guide/process_quantization_precision_tuning.md)*: accuracy degradation caused by this mode can be rolled back layer by layer and tuned through that workflow.

---

## 4. Related Terms

- [Quantization Modes](../README.md): the parent concept; this entry belongs to the [FA Quantization](README.md) category and is one concrete mode under it.
- [FA INT8 Dynamic Quantization](term_fa_int8_dynamic.md): a peer mode; the unified INT8 mode with per-token dynamic quantization.
- [FA Q-FP8 Dynamic K/V-FP8 Static Quantization](term_fa_q_fp8_dynamic_kv_fp8.md): a peer mode, an FP8 mixed mode with Q dynamic and K/V static.
- [FA MXFP4 Dynamic Quantization](term_fa_mxfp4_dynamic.md): a peer mode, a per-block dynamic low-bit uniform mode.
- [W8A8 FP8 Dynamic Quantization](../linear_layer_quantization/term_w8a8_fp8_dynamic.md): a companion mode, the application of the FP8 per-token dynamic idea to linear layers.
- [KVCache Quantization](../kv_cache_quantization/README.md): the base category; this mode adds Q quantization on top of it.

---

## 5. Reference Documents

1. *[FA3 Quant Parameter Configuration Workflow Guide](../../quantization_algorithms/fa3_quant/usage_fa3_quant.md)*
2. *[Quantization Modes](../README.md)*
