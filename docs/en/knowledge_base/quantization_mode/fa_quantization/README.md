# FA Quantization

> **Entry Category**: Quantization Data Format ([Quantization Modes](../README.md))<br>
> **English Name**: Flash Attention Quantization<br>
> **Application Domain**: Attention computation acceleration, long-context inference<br>
> **msModelSlim Implementation**: see the "Hosting IR Class" field of each mode entry<br>

<a id="overview"></a>

## 1. Overview

FA (Flash Attention) quantization is a category of quantization modes that quantizes the **Q, K, V activation tensors fed into Flash Attention**. It is one of the categories under [Quantization Modes](../README.md), and is an advanced form of [KVCache Quantization](../kv_cache_quantization/README.md): building on quantized K/V (which compresses the cache memory), it further quantizes Q.

K/V quantization mainly solves the storage problem (halving the cache memory, supporting longer contexts). Once Q is also quantized, the tensors participating in the attention score computation (the dot product of Q and K) and in the weighted sum (the dot product with V after softmax) are all low-bit, so integer or low-precision matrix-multiply operators can be called directly, enabling low-precision matrix computation while inheriting the KVCache memory benefit. A typical example is [FA INT8 PerHead Quantization](term_fa_int8_perhead.md): the Q, K, V fed into Flash Attention are statically quantized to INT8 per attention head (per-head).

FA quantization **is essentially a composite quantization mode**: it acts on the three attention branches Q/K/V, with each branch choosing one activation quantization mode; the combination of the three branches constitutes **one holistic mode**. See [Holistic Modes](#holistic-mode).

---

<a id="mode-options"></a>

## 2. Term Description

### 2.1 General Quantization Choices of This Category

- **Quantization target**: the activation tensors of the three branches Q/K/V fed into Flash Attention (K/V carry cache compression, Q enables low-precision score computation).
- **Quantization granularity available per branch**:
  - **per-head**: the scale is shared per attention head, **static** (requires calibration), such as [FA INT8 PerHead Quantization](term_fa_int8_perhead.md).
  - **per-token**: the scale is computed online per token, **dynamic** (no calibration), such as [FA INT8 Dynamic Quantization](term_fa_int8_dynamic.md) and [FA FP8 Dynamic Quantization](term_fa_fp8_dynamic.md).
  - **per-block**: split into 32-element blocks along head_dim with a block-level shared exponent, **dynamic** (no calibration), such as [FA MXFP4 Dynamic Quantization](term_fa_mxfp4_dynamic.md) and [FA QK-MXFP8 Dynamic / V-MXFP8 PerChannel Static Quantization](term_fa_qk_mxfp8_dynamic_v_mxfp8_perchannel.md).
  - **per-channel (V branch only)**: the exponent is shared along `(head, dim)` channels, **static** (requires calibration), such as [FA QK-MXFP8 Dynamic / V-MXFP8 PerChannel Static Quantization](term_fa_qk_mxfp8_dynamic_v_mxfp8_perchannel.md).
- **Data types**: INT8, FP8 (E4M3), MXFP8, MXFP4.
- **Relationship with KV**: this category builds on [KVCache Quantization](../kv_cache_quantization/README.md) (inheriting its memory benefit) and adds Q quantization to enable low-precision attention matrix computation.

---

<a id="holistic-mode"></a>

### 2.2 Holistic Modes

One "mode" of FA quantization refers to **the holistic combination of the quantization schemes of the three branches Q/K/V**, rather than the granularity of a single branch. The three branches may use the same activation quantization mode, or choose a different activation quantization mode per branch, with each branch playing a different role in accuracy and overhead; for example, Q uses per-token dynamic calibration-free quantization while K/V use per-head static quantization to carry cache compression, forming a mixed holistic mode such as [FA Q-INT8 Dynamic K/V-INT8 Static Quantization](term_fa_q_int8_dynamic_kv_int8.md).

---

<a id="branch-combination"></a>

### 2.3 Validated Holistic Modes

The table below lists FA holistic modes that have been validated in practice, each corresponding to one entry.

| Mode Entry | Q Branch | K Branch | V Branch | Parameter Acquisition |
| --- | --- | --- | --- | --- |
| [FA INT8 PerHead Quantization](term_fa_int8_perhead.md) | INT8 per-head | INT8 per-head | INT8 per-head | Static (requires calibration) |
| [FA INT8 Dynamic Quantization](term_fa_int8_dynamic.md) | INT8 per-token | INT8 per-token | INT8 per-token | Dynamic |
| [FA FP8 Dynamic Quantization](term_fa_fp8_dynamic.md) | FP8 per-token | FP8 per-token | FP8 per-token | Dynamic |
| [FA MXFP4 Dynamic Quantization](term_fa_mxfp4_dynamic.md) | MXFP4 per-block | MXFP4 per-block | MXFP4 per-block | Dynamic |
| [FA Q-INT8 Dynamic K/V-INT8 Static Quantization](term_fa_q_int8_dynamic_kv_int8.md) | INT8 per-token | INT8 per-head | INT8 per-head | Q dynamic; K/V static |
| [FA Q-FP8 Dynamic K/V-FP8 Static Quantization](term_fa_q_fp8_dynamic_kv_fp8.md) | FP8 per-token | FP8 per-head | FP8 per-head | Q dynamic; K/V static |
| [FA QK-MXFP8 Dynamic / V-MXFP8 PerChannel Static Quantization](term_fa_qk_mxfp8_dynamic_v_mxfp8_perchannel.md) | MXFP8 per-block | MXFP8 per-block | MXFP8 per-channel | Q/K dynamic; V static |

The three branches may use the same activation quantization mode, or different activation quantization modes respectively; unquantized branches keep their original precision. Each row in the table above represents one selectable FA quantization mode.

---

<a id="modes"></a>

### 2.4 Mode List of This Category

The following is the list of FA quantization modes (for complete specifications including the hosting IR class, see the [Mode Index](../README.md#modes)):

| Mode | One-liner |
| --- | --- |
| [FA INT8 PerHead Quantization](term_fa_int8_perhead.md) | Q/K/V uniformly statically quantized per attention head, INT8, requires calibration |
| [FA INT8 Dynamic Quantization](term_fa_int8_dynamic.md) | Q/K/V uniformly dynamically quantized per token, INT8, no calibration |
| [FA FP8 Dynamic Quantization](term_fa_fp8_dynamic.md) | Q/K/V uniformly dynamically quantized per token, FP8 (E4M3), no calibration |
| [FA MXFP4 Dynamic Quantization](term_fa_mxfp4_dynamic.md) | Q/K/V uniformly block-level shared-exponent quantized, MXFP4, no calibration |
| [FA Q-INT8 Dynamic K/V-INT8 Static Quantization](term_fa_q_int8_dynamic_kv_int8.md) | Q dynamic per token, K/V static per head, INT8 |
| [FA Q-FP8 Dynamic K/V-FP8 Static Quantization](term_fa_q_fp8_dynamic_kv_fp8.md) | Q dynamic per token, K/V static per head, FP8 |
| [FA QK-MXFP8 Dynamic / V-MXFP8 PerChannel Static Quantization](term_fa_qk_mxfp8_dynamic_v_mxfp8_perchannel.md) | Q/K dynamic per block, V static per `(head, dim)` channel, MXFP8 |

---

## 3. Related Workflows

- [One-Click Quantization Usage Guide](../../../user_guide/usage_one_click_quantization.md): enable the modes of this category via the `fa3_quant` processor.
- [Quantization Accuracy Tuning Guide](../../../user_guide/process_quantization_precision_tuning.md): accuracy verification for attention activation quantization.

---

<a id="related-terms"></a>

## 4. Related Terms

- [Quantization Modes](../README.md): the parent concept; this entry belongs to the "FA quantization" category.
- [KVCache Quantization](../kv_cache_quantization/README.md): the parent concept (the basis); this category adds Q quantization on top of it.
- [Linear Layer Quantization](../linear_layer_quantization/README.md): a sibling concept; the quantization category acting on weights and activations, which can be combined with this category.
- [FA INT8 PerHead Quantization](term_fa_int8_perhead.md): a subordinate concept, the holistic mode in this category where the three branches uniformly use per-head static quantization.
- [FA3 Quant Attention Activation Quantization Algorithm](../../quantization_algorithms/fa3_quant/term_fa3_quant.md): a companion term that describes the FA quantization algorithm.

---

## 5. Reference Documents

1. Dao T et al. FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness. NeurIPS 2022. <https://arxiv.org/abs/2205.14135>
2. *[FA3 Quant Parameter Configuration Workflow Guide](../../quantization_algorithms/fa3_quant/usage_fa3_quant.md)*
3. *[Quantization Modes](../README.md)*
