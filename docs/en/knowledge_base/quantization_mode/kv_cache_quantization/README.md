# KVCache Quantization

> **Entry Category**: Quantization Data Format ([Quantization Modes](../README.md))<br>
> **English Name**: KV Cache Quantization<br>
> **Application Domain**: KVCache compression, long-context inference acceleration<br>
> **msModelSlim Implementation**: [`msmodelslim/ir/attention.py`](../../../../../msmodelslim/ir/attention.py)<br>

<a id="overview"></a>

## 1. Overview

KVCache quantization is a category of quantization modes that quantizes the **KVCache (the cached Key/Value tensors)** of the attention mechanism. It is one of the categories under [Quantization Modes](../README.md). The KVCache is essentially a **special kind of activation**—the Key/Value projections of historical tokens (the outputs of the K/V linear layers) are cached for reuse by the attention computation of subsequent tokens, rather than being model weights; it does not participate in weight matrix multiplication and only affects the attention computation. Its memory grows linearly with sequence length and is the main memory cost of long-context inference.

Quantizing K/V does not change the quantization scheme of the weights and activations and can be stacked with linear layer quantization; after quantization the cache memory is roughly halved (INT8 is about half of FP16), thereby supporting longer contexts or greater concurrency; the impact on inference latency depends on the operator and hardware implementation and must be evaluated by measurement. A typical example is [KVCache-PerChannel Quantization](term_kv_cache_perchannel.md)—the cached K/V shares quantization parameters along the hidden dimension (channel) and is quantized to INT8.

---

<a id="mode-options"></a>

## 2. Term Description

### 2.1 General Quantization Choices of This Category

- **Quantization target**: the cached K and V tensors (Q is not included; quantizing Q belongs to [FA Quantization](../fa_quantization/README.md)).
- **Quantization granularity**: currently only per-channel is supported in msModelSlim—the scale/offset is shared along the hidden dimension (head_dim), symmetric or asymmetric, that is, [KVCache-PerChannel Quantization](term_kv_cache_perchannel.md).
- **Data types**: INT8 (symmetric/asymmetric). Finer granularity (such as per-head or per-token) or a lower bit width (INT4/FP8) are research directions that msModelSlim does not yet support.
- **Combination**: it can be stacked with linear layer quantization (the two have different quantization targets and do not overlap). Going one step further and also quantizing **the Q activation tensor fed into the attention computation** upgrades it to [FA Quantization](../fa_quantization/README.md)—FA quantization quantizes the Q activation (the output tensor of QProj), rather than doing weight/matrix-multiply quantization on the QProj linear layer itself.

---

<a id="modes"></a>

### 2.2 Mode List of This Category

The following is the list of KVCache quantization modes supported by msModelSlim (for complete specifications including the hosting IR class, see the [Mode Index](../README.md#modes)):

| Mode | One-liner |
|------|--------|
| [KVCache-PerChannel Quantization](term_kv_cache_perchannel.md) | The cached K/V shares parameters along the hidden dimension, INT8 per-channel |

---

## 3. Related Workflows

- [One-Click Quantization Usage Guide](../../../user_guide/usage_one_click_quantization.md): enable the modes of this category via the `dynamic_cache` processor.
- [Quantization Accuracy Tuning Guide](../../../user_guide/process_quantization_precision_tuning.md): long-sequence accuracy verification and KV fallback.

---

<a id="related-terms"></a>

## 4. Related Terms

- [Quantization Modes](../README.md): the parent concept; this entry is the "KVCache quantization" category.
- [Linear Layer Quantization](../linear_layer_quantization/README.md): a sibling concept; the quantization category acting on weights and activations, which can be combined with this category.
- [FA Quantization](../fa_quantization/README.md): an advanced concept; it further quantizes Q on top of this category.
- [KVCache-PerChannel Quantization](term_kv_cache_perchannel.md): a subordinate concept, the only quantization mode currently in this category.
- [KVCache Quant Cache Quantization Algorithm](../../quantization_algorithms/kvcache_quant/term_kvcache_quant.md): a companion term that describes the KVCache quantization algorithm.
- [KV Smooth Cache Smoothing Algorithm](../../quantization_algorithms/kv_smooth/term_kv_smooth.md): a prerequisite entry; smooths KV before quantization to reduce quantization error.

---

## 5. Reference Documents

1. *[KVCache Quant Parameter Configuration Workflow Guide](../../quantization_algorithms/kvcache_quant/usage_kvcache_quant.md)*
2. *[KV Smooth Parameter Configuration Workflow Guide](../../quantization_algorithms/kv_smooth/usage_kv_smooth.md)*
3. *[Quantization Modes](../README.md)*
