# FA INT8 PerHead Quantization

> **Entry Category**: Quantization Data Format ([FA Quantization](README.md))<br>
> **English Name**: FA INT8 PerHead Quantization<br>
> **Application Domain**: Long-context inference acceleration, low-precision attention computation<br>
> **Hosting IR Class**: `INT8FakeQuantActivationPerHead` ([`msmodelslim/ir/activation_static.py`](../../../../../msmodelslim/ir/activation_static.py))<br>

---

## 1. Overview

FA INT8 PerHead quantization is a holistic composite mode in which the Q/K/V activations fed into Flash Attention are uniformly statically quantized to [INT8](../../quantization_basic/term_int8.md) per attention head (per-head). It is the typical mode in FA quantization where all three branches uniformly use per-head static quantization. Building on quantized K/V ([KVCache Quantization](../kv_cache_quantization/README.md), which compresses the cache memory), it further quantizes Q, so that the tensors participating in the attention score computation (the dot product of Q and K) and in the weighted sum (the dot product with V after softmax) are all low-bit, thereby enabling low-precision matrix computation in addition to the KV memory benefit.

---

## 2. Term Description

### 2.1 Mode Specification

| Dimension | Value | Description |
| --- | --- | --- |
| Quantization Target | Attention Q/K/V activations | The activation tensors fed into Flash Attention (shape (B, H, S, D)); all three branches `fa_q`/`fa_k`/`fa_v` are quantized. K/V carry cache compression, and Q enables low-precision score computation |
| Bit Width | 8bit | Q/K/V are all 8bit |
| Data Type | INT8 | Symmetric integer format |
| Parameter Acquisition | Static | The scale is determined by offline calibration and then fixed, with zero online overhead at inference; only the scale is used, with no offset |
| Quantization Granularity | per-head | Each attention head shares one scale; the parameter overhead is only the number of heads, adapting to the independent numerical distribution of each head |
| Symmetry | Symmetric | Scale only, no offset |

### 2.2 Quantization Formula

Quantization and dequantization use the standard linear mapping. Symmetric quantization (scale only, no zero point):

$$q = \mathrm{round}(x / s), \qquad \hat{x} = q \cdot s, \qquad s = \frac{\max(|x|)}{2^{b-1}}$$

In the formula above, $x$ is the tensor to quantize, $q$ is the quantized value of that tensor, $\hat{x}$ is the dequantized reconstruction, $s$ is the scale, and $b=8$ is the bit width. This mode uses symmetric quantization, that is, scale only and no offset. At per-head granularity, the tensor is split along the first dimension (head), and the elements within each head share the same scale.

### 2.3 Relationship to Other Modes

- **Relationship to [FA INT8 Dynamic Quantization](term_fa_int8_dynamic.md) (same QKV-uniform INT8 mode, different parameter acquisition and granularity)**:
  - This mode (per-head static): the advantage is that the scale is fixed offline with zero online overhead at inference, and the parameter overhead is only the number of heads; the disadvantage is the relatively coarse granularity and the dependence on calibration data, making it insensitive to per-token distribution changes.
  - INT8 dynamic (per-token dynamic): the advantage is that the scale is computed online per token with no calibration, so the quantization fits the numerical range of each token better; the disadvantage is one extra per-token min/max reduction on every forward pass, with overhead on the same order as the number of tokens.
- **Relationship to [FA Q-INT8 Dynamic K/V-INT8 Static Quantization](term_fa_q_int8_dynamic_kv_int8.md) (same INT8 family, different branch quantization)**:
  - This mode uniformly uses per-head static quantization across the three branches; the mixed mode changes Q to per-token dynamic, making the Q branch calibration-free.
- **Relationship to [KVCache-PerChannel Quantization](../kv_cache_quantization/term_kv_cache_perchannel.md) (base)**:
  - This mode builds on K/V quantization: it inherits the cache memory benefit and adds Q quantization to enable low-precision matrix computation.

### 2.4 Applicable Scenarios and Limitations

#### 2.4.1 Applicable Scenarios

- **Long-sequence attention acceleration**: in long-context decoding scenarios with intensive attention computation, integer/low-precision attention reduces bandwidth and compute overhead.
- **Combination with linear layer quantization**: it can be stacked with linear layer quantization such as W8A8; the quantization targets differ and do not overlap, jointly reducing the overhead of weights/activations and of attention computation.
- **Baseline with identical accuracy requirements across the three branches**: suitable as the baseline for a first attempt at FA quantization.

#### 2.4.2 Usage Limitations

- **Static quantization depends on calibration**: the per-head scale is determined by the calibration set, and a distribution bias will affect attention score accuracy.
- **Attention accuracy is sensitive**: the quantization error of Q/K directly affects attention scores and must be verified under extremely long sequences.
- **Data type selection**: per-head static is commonly INT8; when a lower bit width is needed, MX dynamic modes can be used instead.

---

## 3. Related Workflows

- *[One-Click Quantization Usage Guide](../../../user_guide/usage_one_click_quantization.md)*: select this mode via a YAML configuration and run quantization.
- *[Quantization Accuracy Tuning Guide](../../../user_guide/process_quantization_precision_tuning.md)*: accuracy degradation caused by this mode can be rolled back layer by layer and tuned through that workflow.

---

## 4. Related Terms

- [Quantization Modes](../README.md): the parent concept; this entry belongs to the [FA Quantization](README.md) category and is one concrete mode under it.
- [FA INT8 Dynamic Quantization](term_fa_int8_dynamic.md): a peer mode; the unified INT8 mode with per-token dynamic quantization.
- [FA Q-INT8 Dynamic K/V-INT8 Static Quantization](term_fa_q_int8_dynamic_kv_int8.md): a peer mode; the INT8 mixed mode with dynamic Q and static K/V.
- [KVCache Quantization](../kv_cache_quantization/README.md): the base category; this mode adds Q quantization on top of it.
- [FA3 Quant Attention Activation Quantization Algorithm](../../quantization_algorithms/fa3_quant/term_fa3_quant.md): a companion term, the algorithm processor documentation corresponding to this mode.

---

## 5. Reference Documents

1. *[FA3 Quant Parameter Configuration Workflow Guide](../../quantization_algorithms/fa3_quant/usage_fa3_quant.md)*
2. *[Quantization Modes](../README.md)*
