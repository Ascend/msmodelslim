# FA QK-MXFP8 Dynamic / V-MXFP8 PerChannel Static Quantization

> **Entry Category**: Quantization Data Format ([FA Quantization](README.md))<br>
> **English Name**: FA QK-MXFP8 Dynamic V-MXFP8 PerChannel Static Quantization<br>
> **Application Domain**: Long-context inference acceleration, low-precision attention computation<br>
> **Hosting IR Class**: `FakeQuantActivationPerBlock` (Q/K, [`msmodelslim/ir/activation_dynamic.py`](../../../../../msmodelslim/ir/activation_dynamic.py)), `MXFP8FakeQuantActivationPerChannel` (V, [`msmodelslim/ir/mxfp8_activation_static.py`](../../../../../msmodelslim/ir/mxfp8_activation_static.py))<br>

---

## 1. Overview

FA QK-MXFP8 dynamic / V-MXFP8 per-channel static quantization is a mixed holistic mode: the Q/K branches are dynamically quantized to [MXFP8](../../quantization_basic/term_mxfp.md) by splitting them into 32-element blocks along head_dim with an E8M0 exponent shared at block level, while the V branch is statically quantized to MXFP8 per-channel by sharing an E8M0 exponent along the `(head, dim)` channel (see [FA Quantization](README.md)). This mode uses MXFP8 to carry the quantization of all three branches Q/K/V: Q/K are dynamic and calibration-free, V is static and calibrated offline per channel; K/V carry cache compression ([KVCache Quantization](../kv_cache_quantization/README.md)), and Q enables low-precision attention matrix computation.

---

## 2. Term Description

### 2.1 Mode Specification

| Dimension | fa_q | fa_k | fa_v |
| --- | --- | --- | --- |
| Bit Width | 8bit | 8bit | 8bit |
| Data Type | MXFP8 | MXFP8 | MXFP8 |
| Parameter Acquisition | Dynamic (online statistics, no calibration) | Dynamic (online statistics, no calibration) | Static (offline calibration) |
| Quantization Granularity | per-block (32-element blocks along head_dim) | per-block (32-element blocks along head_dim) | per-channel (along `(head, dim)` channels) |
| Symmetry | Symmetric | Symmetric | Symmetric |

### 2.2 Quantization Formula

The Q/K branches use dynamic quantization with a block-shared E8M0 exponent:

$$e = \lfloor \log_2(\max_{i \in \mathrm{block}}|x_i|) \rfloor, \qquad q_i = \mathrm{round}_{MX}\left(\frac{x_i}{2^e}\right), \qquad \hat{x}_i = q_i \cdot 2^e$$

The V branch uses static quantization with an E8M0 exponent shared along the `(head, dim)` channel (the scale is fixed by offline calibration):

$$e_c = \lfloor \log_2(\max_{i \in \mathrm{channel}}|x_i|) \rfloor, \qquad q_i = \mathrm{round}_{MX}\left(\frac{x_i}{2^{e_c}}\right), \qquad \hat{x}_i = q_i \cdot 2^{e_c}$$

In the formula above, $x_i$ is the activation element to quantize, $q_i$ is the MX mantissa of that element after quantization (an FP8 mantissa for MXFP8), $\hat{x}_i$ is the value of that element restored by dequantization, $e$ is the E8M0 exponent shared within each 32-element block of the Q/K branches, and $e_c$ is the E8M0 exponent shared by each `(head, dim)` channel of the V branch. The only difference between Q/K and V lies in the statistical range over which the exponent is shared: Q/K compute a block-level exponent over each block of 32 elements, whereas V merges the head dimension into the channel dimension and computes a channel-level exponent over `(head, dim)`, with each channel covering the elements of that `(head, dim)` across all sequence positions.

### 2.3 Design Motivation and Division of Labor

- **Q/K uses per-block dynamic**: Q/K participate in the score computation and vary with the input distribution, so dynamic calibration-free quantization fits the actual distribution and avoids the dependence on calibrating Q/K; per-block captures distribution differences within channels along head_dim.
- **V uses per-channel static**: V is the static branch among the three that carries cache compression, and the core motivation for per-channel is to avoid the tail-block problem caused by splitting into blocks of 32 elements. MXFP8 block quantization splits along head_dim with a group size of 32; if head_dim is not a multiple of 32, a tail block of fewer than 32 elements is left at the end of each row. The V cache is appended token by token during decode and spans the entire inference lifecycle, so repeatedly padding or special-casing the tail block is costly. V therefore no longer groups by 32; instead it merges the head dimension into the channel dimension and shares the E8M0 exponent across the entire sequence with `(head, dim)` as the channel, eliminating the tail block at the root. The quantization parameters of V are determined statically offline and remain fixed throughout the cache lifecycle, which is also consistent with the channel semantics of KVCache quantization.

### 2.4 Relationship to Other Modes

- **Relationship to [FA MXFP4 Dynamic Quantization](term_fa_mxfp4_dynamic.md) (same MX family, different bit width and branch quantization)**:
  - This mode is 8bit MXFP8 with the V branch switched to per-channel static, avoiding the tail-block problem that may arise from grouping by 32; MXFP4 dynamic is 4bit with the three branches uniformly per-block dynamic. This mode has higher accuracy and V cache parameters fixed offline, at the cost of a larger bit width and the need for calibration data for V.
- **Relationship to [FA Q-INT8 Dynamic K/V-INT8 Static Quantization](term_fa_q_int8_dynamic_kv_int8.md) (same mixed structure of Q dynamic + K/V static, different data type)**:
  - On top of the mixed structure, this mode further distinguishes Q/K (dynamic) from V (static per-channel) and adopts the MXFP8 block-level exponent floating-point data type throughout; the Q-INT8 mixed mode is an INT8 integer mode with Q dynamic + K/V per-head static.
- **Relationship to [KVCache-PerChannel Quantization](../kv_cache_quantization/term_kv_cache_perchannel.md) (base)**:
  - This mode builds on K/V quantization, and the per-channel static quantization of V aligns directly with the channel semantics of cache compression.

### 2.5 Applicable Scenarios and Limitations

#### 2.5.1 Applicable Scenarios

- **MXFP8 low-precision attention acceleration**: deployments whose target backend natively supports MXFP8 attention computation.
- **Q/K calibration-free + V cache static**: scenarios where Q/K are hard to cover with a fixed calibration while the V cache quantization parameters should be determined offline.
- **Full quantization of large MoE models**: used as the attention activation part of the holistic quantization together with linear layer W4A4/W8A8 and FA attention.

#### 2.5.2 Usage Limitations

- **Depends on hardware MX support**: MXFP8 attention computation requires support from the target hardware operator library.
- **32-alignment of the Q/K branches**: Q/K use per-block, so head_dim must be divisible by 32, otherwise padding is required; the V branch is per-channel and does not group by 32, so it has no such restriction.
- **The V branch depends on calibration**: the per-channel scale of V is determined by the calibration set; without calibration data the static parameters cannot be determined.
- **Both online and offline overhead coexist**: Q/K have dynamic reduction overhead, while static V depends on a calibration workflow.
- **Attention accuracy sensitive**: quantization errors in Q/K directly affect the attention score, so accuracy must be validated on extremely long sequences.

---

## 3. Related Workflows

- *[One-Click Quantization Usage Guide](../../../user_guide/usage_one_click_quantization.md)*: select this mode via a YAML configuration and run quantization.
- *[Quantization Accuracy Tuning Guide](../../../user_guide/process_quantization_precision_tuning.md)*: accuracy degradation caused by this mode can be rolled back layer by layer and tuned through that workflow.

---

## 4. Related Terms

- [Quantization Modes](../README.md): the parent concept; this entry belongs to the [FA Quantization](README.md) category and is one concrete mode under it.
- [FA Quantization](README.md): the parent concept, the category this entry belongs to.
- [FA MXFP4 Dynamic Quantization](term_fa_mxfp4_dynamic.md): a peer mode, the low-bit mode with uniform MXFP4 per-block dynamic quantization across the three branches.
- [FA Q-INT8 Dynamic K/V-INT8 Static Quantization](term_fa_q_int8_dynamic_kv_int8.md): a peer mixed-structure mode, the INT8 mode with Q dynamic + K/V per-head static.
- [KVCache Quantization](../kv_cache_quantization/README.md): the base category; this mode adds Q quantization on top of it.
- [KVCache-PerChannel Quantization](../kv_cache_quantization/term_kv_cache_perchannel.md): a prerequisite term, the specific mode of K/V quantization (with which the per-channel semantics of the V branch align).
- [AscendV1 Quantization Format](../../quantization_format/ascendv1/term_ascendv1.md#desc-faquant): a companion format, the AscendV1 export convention of this mode (`{prefix}.quant_type`, such as `QK_MXFP8_DYNAMIC_V_MXFP8_PER_CHANNEL`).

---

## 5. Reference Documents

1. *[FA3 Quant Parameter Configuration Workflow Guide](../../quantization_algorithms/fa3_quant/usage_fa3_quant.md)*
2. *[Quantization Modes](../README.md)*
