# FA INT8 Dynamic Quantization

> **Entry Category**: Quantization Data Format ([FA Quantization](README.md))<br>
> **English Name**: FA INT8 Dynamic Quantization<br>
> **Application Domain**: Long-context inference acceleration, low-precision attention computation<br>
> **Hosting IR Class**: `FakeQuantActivationPerToken` ([`msmodelslim/ir/activation_dynamic.py`](../../../../../msmodelslim/ir/activation_dynamic.py))<br>

---

## 1. Overview

FA INT8 dynamic quantization is a holistic composite mode in which the Q/K/V activations fed into Flash Attention are uniformly dynamically quantized to [INT8](../../quantization_basic/term_int8.md) row by row per token. The quantization parameters are computed online with no calibration, so the activation resolution of each token fits its own numerical range. It is the INT8 mode of the "dynamic, calibration-free" tier in FA quantization; K/V carry cache compression ([KVCache Quantization](../kv_cache_quantization/README.md)), and after Q is quantized as well, the attention score computation can use low-precision matrix computation.

---

## 2. Term Description

### 2.1 Mode Specification

| Dimension | Value | Description |
| --- | --- | --- |
| Quantization Target | Attention Q/K/V activations | The activation tensors fed into Flash Attention (shape (B, H, S, D)); all three branches `fa_q`/`fa_k`/`fa_v` are quantized. K/V carry cache compression, and Q enables low-precision score computation |
| Bit Width | 8bit | Q/K/V are all 8bit |
| Data Type | INT8 | Symmetric integer format |
| Parameter Acquisition | Dynamic | Computes min/max online per token in the forward pass and derives the scale; no calibration needed |
| Quantization Granularity | per-token | After reshaping to (B\*H\*S, D), the scale is shared per row (token) |
| Symmetry | Symmetric | Scale only, no offset |

### 2.2 Quantization Formula

Symmetric quantization (scale only, no zero point):

$$q = \mathrm{round}(x / s), \qquad \hat{x} = q \cdot s, \qquad s = \frac{\max(|x|)}{2^{b-1}}$$

In the formula above, $x$ is the tensor to quantize, $q$ is the quantized value of that tensor, $\hat{x}$ is the dequantized reconstruction, $s$ is the scale, and $b=8$ is the bit width. This mode uses symmetric quantization, that is, scale only and no offset. At per-token granularity, the input is first reshaped to (B\*H\*S, D), after which min/max is computed row by row and the scale of each token is calculated dynamically.

### 2.3 Relationship to Other Modes

- **Relationship to [FA INT8 PerHead Quantization](term_fa_int8_perhead.md) (same INT8 uniform mode, different parameter acquisition and granularity)**:
  - This mode (per-token dynamic): the advantage is that the scale is computed online per token with no calibration, so the quantization fits the numerical range of each token; the disadvantage is one extra per-token min/max reduction on every forward pass, with overhead on the same order as the number of tokens.
  - PerHead (per-head static): the advantage is that the scale is fixed offline with zero online overhead at inference, and the parameters are only the number of heads; the disadvantage is the dependence on calibration data and the coarse granularity, making it insensitive to distribution changes between tokens.
- **Relationship to [FA FP8 Dynamic Quantization](term_fa_fp8_dynamic.md) (same uniform per-token dynamic, different data type)**:
  - This mode uses the INT8 integer format, while FP8 dynamic uses the FP8 (E4M3) floating-point format; both can be used as calibration-free uniform dynamic modes, and the choice is made according to the deployment operator and accuracy.
- **Relationship to [FA Q-INT8 Dynamic K/V-INT8 Static Quantization](term_fa_q_int8_dynamic_kv_int8.md) (same INT8 family, different branch quantization)**:
  - This mode uses unified per-token dynamic quantization for all three branches; the mixed mode changes K/V to per-head static, trading K/V calibration for lower online overhead.
- **Relationship to [W8A8 Dynamic Quantization](../linear_layer_quantization/term_w8a8_dynamic.md) (same per-token dynamic idea)**:
  - This mode applies the same idea to the attention Q/K/V activations; W8A8 dynamic applies it to linear layer weights and activations. The two are orthogonal and can be combined.

### 2.4 Applicable Scenarios and Limitations

#### 2.4.1 Applicable Scenarios

- **Calibration data unreliable or missing**: when no representative calibration set is available, the dynamic per-token mode is more robust.
- **High attention accuracy requirements**: a finer-grained activation quantization than static quantization is needed, and the online reduction overhead is acceptable.
- **Long-sequence attention acceleration**: combined with the integer/low-precision computation of Flash Attention, it reduces bandwidth and compute overhead.

#### 2.4.2 Usage Limitations

- **Online reduction overhead**: the per-token min/max reduction adds a small amount of latency, which is more noticeable for short sequences.
- **Attention accuracy sensitive**: quantization errors in Q/K directly affect the attention score, so accuracy must be validated on extremely long sequences.
- **Hardware operator dependency**: per-token dynamic quantization requires the inference framework / operator library to support per-token parameter computation.

---

## 3. Related Workflows

- *[One-Click Quantization Usage Guide](../../../user_guide/usage_one_click_quantization.md)*: select this mode via a YAML configuration and run quantization.
- *[Quantization Accuracy Tuning Guide](../../../user_guide/process_quantization_precision_tuning.md)*: accuracy degradation caused by this mode can be rolled back layer by layer and tuned through that workflow.

---

## 4. Related Terms

- [Quantization Modes](../README.md): the parent concept; this entry belongs to the [FA Quantization](README.md) category and is one concrete mode under it.
- [FA INT8 PerHead Quantization](term_fa_int8_perhead.md): a peer mode, the per-head static INT8 uniform mode.
- [FA Q-INT8 Dynamic K/V-INT8 Static Quantization](term_fa_q_int8_dynamic_kv_int8.md): a peer mode; the INT8 mixed mode with dynamic Q and static K/V.
- [FA FP8 Dynamic Quantization](term_fa_fp8_dynamic.md): a peer mode; the unified FP8 mode with per-token dynamic quantization.
- [W8A8 Dynamic Quantization](../linear_layer_quantization/term_w8a8_dynamic.md): a companion mode, the application of the per-token dynamic idea to linear layers.
- [KVCache Quantization](../kv_cache_quantization/README.md): the base category; this mode adds Q quantization on top of it.

---

## 5. Reference Documents

1. *[FA3 Quant Parameter Configuration Workflow Guide](../../quantization_algorithms/fa3_quant/usage_fa3_quant.md)*
2. *[Quantization Modes](../README.md)*
