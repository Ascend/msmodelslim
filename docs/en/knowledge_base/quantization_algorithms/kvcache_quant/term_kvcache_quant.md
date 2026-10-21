# KVCache Quant Cache Quantization Algorithm Terminology Entry

> **Entry Category**: [Quantization Algorithms](../README.md#2-quantization-algorithms)<br>
> **English Name**: kvcache_quant<br>
> **Application Domain**: KV cache compression, long-sequence inference acceleration<br>

---

## 1. Overview

KVCache Quant is a low-bit quantization algorithm for the inference KV cache. It builds a quantized representation of the Key/Value states that must be cached for a long time, reducing the cache footprint that grows with context length during long-sequence generation; its core characteristics are that it targets cache tensors rather than model weights, that it usually computes quantization parameters per channel or at another local granularity, and that it must account for the different effects of Key and Value errors on the attention computation.

---

## 2. Term Description

In large model inference, the Key/Value states stored in the KV cache consume a large amount of memory and grow linearly with sequence length, becoming the bottleneck of long-sequence inference. Quantizing the states written into the KV cache can significantly reduce cache memory usage while maintaining generation quality, improving long-sequence inference efficiency.

### 2.1 Core Idea

The core idea of KVCache Quant is to compress the Key/Value states that must be preserved across tokens for a long time in autoregressive attention from a high-precision representation into a low-bit representation, and to restore them by the corresponding scale or consume them directly with low-bit operators when the subsequent attention computation needs those historical states. Because the KV cache capacity grows approximately linearly with the number of layers, sequence length, number of KV heads, and head dimension, reducing the bit width per element directly reduces the cache overhead of long-context inference.

Unlike ordinary weight quantization, the KV cache is a dynamic state that is continuously appended during decoding, and its distribution varies with layer, head, channel, and token position. The algorithm must therefore decide not only the bit width but also the scale sharing granularity and the scale update method: too wide a sharing range is easily dominated by local outliers, while too fine a range increases scale metadata, statistics, and memory-access overhead.

The main benefit of cache quantization is almost proportional to the cache bit width, so the longer the context, the more pronounced the memory benefit.

### 2.2 Working Mechanism

Taking standard attention as an example, the historical Key/Value are continuously appended as $K_{1:t},V_{1:t}$, and the current Query reads them through $\operatorname{softmax}(Q_tK_{1:t}^T/\sqrt d)V_{1:t}$. Quantization can be written as $K_q=Q(K;S_K,Z_K)$ and $V_q=Q(V;S_V,Z_V)$, after which the attention is completed using the dequantized values or operators that support low-bit inputs. When compressing from FP16/BF16 to INT8, the theoretical element storage is roughly halved if scale metadata is ignored; lower bit widths reduce it further.

The scale granularity determines which tokens/heads/channels share the quantization interval. per-channel/per-head can reduce outlier crosstalk between different channels; a dynamic scale adapts to the current data but requires online statistics and extra metadata, whereas a static scale costs less but depends on the calibration distribution. The error propagation of K and V also differs: a K error first changes the logits and may be nonlinearly amplified after softmax; a V error mainly propagates through the linear weighted sum. The two may therefore exhibit different sensitivities even at the same bit width.

### 2.3 Mathematical Description

For any tensor to be cached, $X\in\{K,V\}$, linear low-bit quantization can be written as:

$$
q_X=\operatorname{clip}\left(\operatorname{round}\left(\frac{X}{S_X}\right)+Z_X,\;Q_{min},Q_{max}\right)
$$

Dequantization is approximately:

$$
\hat X=S_X(q_X-Z_X)
$$

where $S_X,Z_X$ can be shared per tensor, head, channel, or a finer granularity. Standard attention with a quantized cache is approximately:

$$
\hat Y_t=\operatorname{softmax}\left(\frac{Q_t\hat K_{1:t}^{T}}{\sqrt d}\right)\hat V_{1:t}
$$

If the original cache element bit width is $b_{fp}$ and the quantized data bit width is $b_q$, the theoretical storage ratio of the data region, ignoring scale, zero-point, and alignment overhead, is approximately:

$$
\frac{M_q}{M_{fp}}\approx\frac{b_q}{b_{fp}}
$$

The actual saving also depends on the scale granularity, the metadata, and the backend storage format.

Let the quantized cache be $\hat K=K+E_K$ and $\hat V=V+E_V$, and the attention output be $O=PV$, where $P=\operatorname{softmax}(QK^T/\sqrt d)$. To first order:

$$
\Delta L=\frac{QE_K^T}{\sqrt d},\qquad\Delta P\approx J_{\mathrm{sm}}\Delta L,
$$

$$
\Delta O\approx \Delta P\,V+P E_V.
$$

The first term reflects the nonlinear propagation of the K error through softmax, and the second reflects the probability-weighted propagation of the V error. If the original cache uses $b_{fp}$ bits and the quantized cache uses $b_q$ bits, ignoring scale metadata, the ideal storage ratio is about $b_q/b_{fp}$; the actual ratio must also account for scale/zero-point and other metadata.

### 2.4 Key Properties

- **State quantization**: the quantization target is the Key/Value historical state that is preserved and repeatedly read across decoding steps.
- **per_channel quantization**: computes quantization parameters along the hidden dimension, balancing accuracy and efficiency.
- **Memory compression**: FP16→INT8 can theoretically reduce KV cache memory usage by about 50%.
- **Incremental calibration**: supports incremental calibration for dynamically changing sequence lengths.
- **Temporally reused error**: the quantization error of historical K/V is read repeatedly across many subsequent decoding steps.

In terms of error and applicability boundaries, too coarse a scale is dominated by head/channel outliers, while too fine a scale increases storage and computation overhead; a dynamic scale adapts well but may add reduction latency to the decoding path. A low bit width also distorts the representation of a very small number of important historical tokens, so in practice the algorithm is often combined with KV Smooth, head selection, or a mixed-precision strategy.

### 2.5 Use Cases

- Long-sequence inference scenarios that need to reduce the KV cache memory footprint.
- Combined with [Linear Quantization](../linear_quant/term_linear_quant.md) to implement a full quantization solution and improve long-sequence inference efficiency.

### 2.6 Usage Restrictions

- The attention module's `forward` function must accept a `DynamicCache` object and call `cache.update()`.
- Currently only INT8 quantization is supported, and only the KV cache states are quantized.
- The pseudo-quantization stage still requires full-precision memory; real memory savings require backend operator support.
- Based on module class name pattern matching (`*self_attn*`); custom naming requires adaptation.

---

## 3. Related Terms

- [Linear Quantization](../linear_quant/term_linear_quant.md): a companion term; KVCache Quant is usually combined with linear quantization to implement a full quantization solution.
- [KV Smooth](../kv_smooth/term_kv_smooth.md): a companion term; this algorithm can work with KV Smooth to suppress Key outliers.
- [FA3 Quant](../fa3_quant/term_fa3_quant.md): a companion term in the same family of cache/activation quantization for long-sequence inference.
- [MinMax](../minmax/term_minmax.md): the application target; cache quantization uses the MinMax method to compute quantization parameters.

---

## 4. Reference Documents

1. [KVCache Quant Parameter Configuration Workflow Guide](./usage_kvcache_quant.md)
