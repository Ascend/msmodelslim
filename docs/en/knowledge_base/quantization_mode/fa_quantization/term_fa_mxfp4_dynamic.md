# FA MXFP4 Dynamic Quantization

> **Entry Category**: Quantization Data Format ([FA Quantization](README.md))<br>
> **English Name**: FA MXFP4 Dynamic Quantization<br>
> **Application Domain**: Long-context inference acceleration, low-precision attention computation<br>
> **Hosting IR Class**: `FakeQuantActivationPerBlock` ([`msmodelslim/ir/activation_dynamic.py`](../../../../../msmodelslim/ir/activation_dynamic.py))<br>

---

## 1. Overview

FA MXFP4 dynamic quantization is a holistic composite mode in which the Q/K/V activations fed into Flash Attention are uniformly dynamically quantized to [MXFP4](../../quantization_basic/term_mxfp.md) by splitting them into 32-element blocks along head_dim with an E8M0 exponent shared at block level. The quantization parameters are computed online with no calibration, and the block-level shared exponent preserves the floating-point dynamic range. Its granularity is finer than per-token (it can capture distribution differences within head_dim), making it the tier of FA quantization with the **largest bandwidth and compute benefit**.

---

## 2. Term Description

### 2.1 Mode Specification

| Dimension | Value | Description |
| --- | --- | --- |
| Quantization Target | Attention Q/K/V activations | The activation tensors fed into Flash Attention (shape (B, H, S, D)); all three branches `fa_q`/`fa_k`/`fa_v` are quantized. K/V carry cache compression, and Q enables low-precision score computation |
| Bit Width | 4bit | Q/K/V are all 4bit |
| Data Type | MXFP4 | One E8M0 exponent shared per 32 elements; MXFP4 emax=2, max 6 |
| Parameter Acquisition | Dynamic | The block-level exponent is counted online in the forward pass, with no calibration |
| Quantization Granularity | per-block | Split into 32-element blocks along the last dimension (head_dim), each block sharing one exponent |
| Symmetry | Symmetric | Exponent/scale only, no offset |

### 2.2 Quantization Formula

The MX format shares an E8M0 exponent per block; each element in the block is scaled by the block-level exponent and then quantized to a mantissa:

$$e = \lfloor \log_2(\max_{i \in \mathrm{block}}|x_i|) \rfloor, \qquad q_i = \mathrm{round}_{MX}\left(\frac{x_i}{2^e}\right), \qquad \hat{x}_i = q_i \cdot 2^e$$

Here each block contains 32 elements, $e$ is the E8M0 exponent shared within the block, $q_i$ is the quantization mantissa of the $i$-th element in the block (a 1bit mantissa for MXFP4), and $\hat{x}_i$ is the value of that element restored by dequantization. The quantization parameters of this mode are obtained by online statistics during quantization, so no calibration data is needed; for the specific parameter acquisition, see the "Mode Specification" table above.

### 2.3 Relationship to Other Modes

- **Relationship to [FA FP8 Dynamic Quantization](term_fa_fp8_dynamic.md) / [FA INT8 Dynamic Quantization](term_fa_int8_dynamic.md) (same uniform dynamic, different format and granularity)**:
  - This mode (MXFP4 per-block): the advantage is the largest 4bit bandwidth benefit, a block-level shared exponent that is more tolerant of outliers, and a finer granularity along head_dim; the disadvantage is the dependence on hardware MX support, the requirement that head_dim be divisible by 32, and an MXFP4 mantissa of only 1bit (emax=2, max 6).
  - FP8/INT8 dynamic: the advantage is an 8bit per-element independent numerical format and a more mature hardware ecosystem; the disadvantage is the higher bit width and the per-token granularity.
- **Relationship to [W8A8 MX Dynamic Quantization](../linear_layer_quantization/term_w8a8_mx_dynamic.md) (same MX per-block dynamic idea)**:
  - This mode applies the same idea to the attention Q/K/V activations; W8A8 MX applies it to linear layer weights and activations.
- **Relationship to [FA QK-MXFP8 Dynamic / V-MXFP8 PerChannel Static Quantization](term_fa_qk_mxfp8_dynamic_v_mxfp8_perchannel.md) (same MX family, different bit width and branch quantization)**:
  - This mode is 4bit, with the three branches Q/K/V uniformly per-block dynamic and calibration-free; the MXFP8 mix is 8bit, with Q/K per-block dynamic and V switched to per-channel static. Making V static avoids the tail-block problem that may arise from grouping by 32 and gives higher accuracy, at the cost of a larger bit width and the need for calibration data for V.

### 2.4 Applicable Scenarios and Limitations

#### 2.4.1 Applicable Scenarios

- **Low-bit attention acceleration**: scenarios that require 4bit attention activation quantization while expecting more stable accuracy than integer formats.
- **Outlier-sensitive attention**: the MX floating-point dynamic range is more tolerant of Q/K/V outliers.
- **Ascend MXFP deployment environments**: deployments where the hardware / operator library natively supports MX attention computation.

#### 2.4.2 Usage Limitations

- **Depends on hardware MX support**: MXFP4 attention computation requires support from the target hardware operator library.
- **Block granularity alignment**: head_dim must be divisible by 32, otherwise padding is required.
- **Online reduction overhead**: per-block statistics add a small amount of latency, which is more noticeable for short sequences.
- **Limited MXFP4 accuracy**: the mantissa is only 1bit with emax=2 and max 6, requiring a trade-off in accuracy-sensitive scenarios.
- **Attention accuracy sensitive**: quantization errors in Q/K directly affect the attention score, so accuracy must be validated on extremely long sequences.

---

## 3. Related Workflows

- *[One-Click Quantization Usage Guide](../../../user_guide/usage_one_click_quantization.md)*: select this mode via a YAML configuration and run quantization.
- *[Quantization Accuracy Tuning Guide](../../../user_guide/process_quantization_precision_tuning.md)*: accuracy degradation caused by this mode can be rolled back layer by layer and tuned through that workflow.

---

## 4. Related Terms

- [Quantization Modes](../README.md): the parent concept; this entry belongs to the [FA Quantization](README.md) category and is one concrete mode under it.
- [FA FP8 Dynamic Quantization](term_fa_fp8_dynamic.md): a peer mode; the unified FP8 mode with per-token dynamic quantization.
- [FA INT8 Dynamic Quantization](term_fa_int8_dynamic.md): a peer mode; the unified INT8 mode with per-token dynamic quantization.
- [FA QK-MXFP8 Dynamic / V-MXFP8 PerChannel Static Quantization](term_fa_qk_mxfp8_dynamic_v_mxfp8_perchannel.md): a peer mode, the 8bit mixed mode with MXFP8 per-block/per-channel.
- [W8A8 MX Dynamic Quantization](../linear_layer_quantization/term_w8a8_mx_dynamic.md): a companion mode, the application of the MX per-block idea to linear layers.
- [KVCache Quantization](../kv_cache_quantization/README.md): the base category; this mode adds Q quantization on top of it.

---

## 5. Reference Documents

1. *[FA3 Quant Parameter Configuration Workflow Guide](../../quantization_algorithms/fa3_quant/usage_fa3_quant.md)*
2. *[Quantization Modes](../README.md)*
