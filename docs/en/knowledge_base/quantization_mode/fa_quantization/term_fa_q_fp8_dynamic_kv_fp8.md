# FA Q-FP8 Dynamic K/V-FP8 Static Quantization

> **Entry Category**: Quantization Data Format ([FA Quantization](README.md))<br>
> **English Name**: FA Q-FP8 Dynamic K/V-FP8 Static Quantization<br>
> **Application Domain**: Long-context inference acceleration, low-precision attention computation<br>
> **Hosting IR Class**: `FakeQuantActivationPerToken` (Q, [`msmodelslim/ir/activation_dynamic.py`](../../../../../msmodelslim/ir/activation_dynamic.py)), `FP8FakeQuantActivationPerHead` (K/V, [`msmodelslim/ir/activation_static.py`](../../../../../msmodelslim/ir/activation_static.py))<br>

---

## 1. Overview

FA Q-FP8 dynamic K/V-FP8 static quantization is a mixed holistic mode: the Q branch is dynamically quantized per token, while the K/V branches are statically quantized per attention head to [FP8 (E4M3)](../../quantization_basic/term_fp8.md) (see [FA Quantization](README.md)). Q computes its scale online per token with no calibration, ensuring that the input to the score computation fits the actual distribution; K/V use per-head static quantization (carrying cache compression, [KVCache Quantization](../kv_cache_quantization/README.md)). Compared with a uniform FP8 dynamic scheme across the three branches, making K/V static determines the cache parameters offline and lowers online overhead; compared with a uniform FP8 per-head static scheme across the three branches, making Q dynamic removes the dependence on calibrating Q.

---

## 2. Term Description

### 2.1 Mode Specification

| Dimension | fa_q | fa_k | fa_v |
| --- | --- | --- | --- |
| Bit Width | 8bit | 8bit | 8bit |
| Data Type | FP8 (E4M3) | FP8 (E4M3) | FP8 (E4M3) |
| Parameter Acquisition | Dynamic (online min/max, no calibration) | Static (offline calibration) | Static (offline calibration) |
| Quantization Granularity | per-token | per-head | per-head |
| Symmetry | Symmetric | Symmetric | Symmetric |

### 2.2 Quantization Formula

The Q branch uses symmetric dynamic per-token quantization:

$$q = \mathrm{round}(x / s), \qquad \hat{x} = q \cdot s, \qquad s = \frac{\max(|x|)}{448}$$

The K/V branches use symmetric static quantization per head:

$$q = \mathrm{round}(x / s_h), \qquad \hat{x} = q \cdot s_h, \qquad s_h = \frac{\max(|x_h|)}{448}$$

In the formula above, $x$ is the tensor to quantize, $q$ is the quantized value of that tensor, $\hat{x}$ is the dequantized reconstruction, $s$ is the dynamic scale computed online per token by the Q branch, $s_h$ is the static scale of the $h$-th attention head, and $x_h$ is the activation subset covered by the $h$-th head. The FP8 (E4M3) format can represent a maximum value of 448, so both $s$ and $s_h$ are determined according to the representable upper limit of FP8. Unlike the dynamic values of the Q branch, the scale of the K/V branches is determined and fixed during the offline calibration stage.

### 2.3 Design Motivation and Division of Labor

- **Q uses per-token dynamic**: Q is the direct input to score computation and is sensitive to changes in the input distribution; dynamic quantization needs no calibration, avoids the Q branch's dependence on the calibration set, and better matches each token's actual range.
- **K/V uses per-head static**: K/V carries cache compression, and per-head static quantization determines cache parameters offline with zero online overhead at inference, consistent with the cache semantics of KVCache quantization; without calibration data the K/V branch cannot determine static parameters.

### 2.4 Relationship to Other Modes

- **Relationship to [FA FP8 Dynamic Quantization](term_fa_fp8_dynamic.md) (same FP8 family, different branch quantization)**:
  - This mode changes the K/V branch from per-token dynamic to per-head static: K/V cache parameters are fixed offline, reducing online overhead, at the cost of K/V depending on calibration data.
- **Relationship to [FA Q-INT8 Dynamic K/V-INT8 Static Quantization](term_fa_q_int8_dynamic_kv_int8.md) (same mixed structure, different data type)**:
  - This mode uses FP8 (E4M3); the Q-INT8 mixed mode uses INT8. FP8 floating point has a larger dynamic range and is relatively more tolerant of outliers.
- **Relationship to [KVCache-PerChannel Quantization](../kv_cache_quantization/term_kv_cache_perchannel.md) (base)**:
  - This mode builds on K/V quantization: the static K/V quantization captures the cache compression benefit, while the dynamic Q quantization enables low-precision score computation.

### 2.5 Applicable Scenarios and Limitations

#### 2.5.1 Applicable Scenarios

- **Insufficient calibration data for the Q branch**: when Q's distribution varies greatly with the input and is hard to cover with fixed calibration, Q is released separately as dynamic.
- **Static cache parameters wanted for the K/V branch**: the K/V cache quantization parameters must be determined offline, avoiding online statistics in the decode stage.
- **FP8 long-sequence attention**: scenarios that expect the benefit of the FP8 floating-point dynamic range while wanting the K/V cache parameters to be determined statically offline.

#### 2.5.2 Usage Limitations

- **The static branch depends on calibration**: the per-head scale of K/V is determined by the calibration set, so without K/V calibration data the static parameters cannot be determined.
- **Limited FP8 representable range**: E4M3 has a max of 448, so large outliers need attention.
- **Attention accuracy sensitive**: quantization errors in Q/K directly affect the attention score, so accuracy must be validated on extremely long sequences.

---

## 3. Related Workflows

- *[One-Click Quantization Usage Guide](../../../user_guide/usage_one_click_quantization.md)*: select this mode via a YAML configuration and run quantization.
- *[Quantization Accuracy Tuning Guide](../../../user_guide/process_quantization_precision_tuning.md)*: accuracy degradation caused by this mode can be rolled back layer by layer and tuned through that workflow.

---

## 4. Related Terms

- [Quantization Modes](../README.md): the parent concept; this entry belongs to the [FA Quantization](README.md) category and is one concrete mode under it.
- [FA FP8 Dynamic Quantization](term_fa_fp8_dynamic.md): a peer mode, the FP8 mode with uniform per-token dynamic quantization across the three branches.
- [FA Q-INT8 Dynamic K/V-INT8 Static Quantization](term_fa_q_int8_dynamic_kv_int8.md): a peer mixed-structure mode, with Q dynamic and K/V static but using the INT8 data type.
- [KVCache Quantization](../kv_cache_quantization/README.md): the base category; this mode adds Q quantization on top of it.
- [KVCache-PerChannel Quantization](../kv_cache_quantization/term_kv_cache_perchannel.md): a prerequisite entry; the concrete mode of K/V quantization.

---

## 5. Reference Documents

1. *[FA3 Quant Parameter Configuration Workflow Guide](../../quantization_algorithms/fa3_quant/usage_fa3_quant.md)*
2. *[Quantization Modes](../README.md)*
