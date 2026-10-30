# RA Compress Long-Sequence Compression Algorithm Terminology Entry

> **Entry Category**: [Attention Head Analysis Algorithms](../README.md#4-attention-head-analysis-algorithms)<br>
> **English Name**: RazorAttention Compress (`ra_compress`)<br>
> **Application Domain**: Long-sequence inference, KV cache compression, attention head selection<br>

---

## 1. Overview

RA Compress (`ra_compress`) is a long-sequence compression and selection algorithm at the granularity of attention heads (KV heads). Based on the segment-spanning attention behavior of attention heads in a Transformer, it identifies the critical KV heads capable of prefix matching (induction heads) and copying matching (echo heads), serving as the retention criterion for KV cache compression during long-sequence inference; its core characteristic is that it constructs controlled inputs from equal-length repeated segments and turns "segment-spanning retrieval capability" itself into a measurable score. It can be used together with attn-scope sensitive layer analysis metrics such as [Attention MSE](../attention_mse/term_attention_mse.md): the latter measures quantization sensitivity to support fallback decisions, whereas the former selects the critical heads to retain for long-sequence inference; the two are not the same kind of metric.

---

## 2. Term Description

During long-text (long-sequence) inference, the KV cache grows linearly with sequence length and is the main memory bottleneck for on-device and large model inference. RA Compress observes that the contribution of KV heads to long-context quality is not uniform — attention heads with strong induction and copying ability are the critical anchors for segment-spanning information transfer and should be retained preferentially when compressing the KV cache. It therefore does not infer importance from task data; instead it directly constructs repeated-segment inputs and measures the attention strength at fixed offset positions between segments, using "whether segment-spanning retrieval capability exists" as the selection criterion.

### 2.1 Core Idea

The core idea of RA Compress is to "locate the critical segment-spanning heads using the repeated-segment structure". The input is designed as several **completely identical token sequences**; under this structure, the heads that genuinely carry segment-spanning information transfer exhibit predictable attention landing points: an induction head at some position in the current segment attends to the token immediately preceding that position in the previous segment (prefix matching), while an echo head directly copies the information at the same position in the previous segment (copying matching).

On ordinary text these landing points are disturbed by semantics and the position distribution and are hard to observe stably, whereas under the repeated-segment structure they appear as concentrated attention peaks, so "whether a head has segment-spanning retrieval capability" becomes a reproducible score under controlled conditions.

### 2.2 Working Mechanism

The working mechanism of RA Compress can be divided into three steps:

1. **Construct the repeated-segment calibration input**: it consists of a first token plus several equal-length random token segments, with the segment length and count built into the algorithm — the first token provides a reusable starting condition, the segment length is `2500` and it repeats for `4` segments, giving a total length of `1 + 2500 × 4`; the content within a segment is completely identical, and the whole input depends on no external data.
2. **Compute the inter-segment offset attention**: collect the outputs on the self-attention Q / K projections, compute $QK^\top$ for each layer and apply softmax to obtain the attention matrix; then read two fixed offset positions for each segment pair — `-SEG + 1` corresponds to prefix matching and `-SEG` to copying matching — and average that offset position over the repeated segment pairs to obtain the two scores of each head in that layer.
3. **Group and select**: if the model uses GQA / MQA, first take the maximum score within the query head group corresponding to each KV head as that KV head's representative score; then sort the scores of all layers and all KV heads together and select the top KV heads by ratio as the retention targets, outputting a list of KV head indices.

Throughout the process, what is evaluated is not "how the model performs on some dataset" but whether the attention structure itself has segment-spanning retrieval capability under controlled repetition conditions, so the user does not need to prepare a calibration set.

### 2.3 Mathematical Description

Let $A \in \mathbb{R}^{L \times L}$ be the attention matrix of a layer on the repeated-segment calibration data (after applying softmax to the outer product of Q and K). For all positions $i$ satisfying $i \in [(k+1) \cdot SEG, (k+2) \cdot SEG]$ (the $(k+1)$-th segment), examine the following two offset positions respectively:

**Prefix matching score (induction heads)**:

$$
\text{prefix\_score}_h = \text{mean}_{k, i} A_h[i, i - k \cdot SEG + 1]
$$

That is, at position $i$ of segment $k+1$, the probability that attention lands at position $i - SEG + 1$ of segment $k$ (the previous segment's prefix position + 1), corresponding to the typical behavior of induction heads.

**Copying matching score (echo heads)**:

$$
\text{copying\_score}_h = \text{mean}_{k, i} A_h[i, i - (k+1) \cdot SEG]
$$

That is, at position $i$ of segment $k+1$, the probability that attention lands at the same offset $i - SEG$ in segment $k$, corresponding to the echo-head behavior of "copying the corresponding position of the previous segment".

Here $SEG = 2500$ (the segment length), $k \in \{0, 1, 2\}$ (3 segment pairs in total), and head $h$ is the attention head before GQA grouping.

**GQA group merging**: if the model uses GQA / MQA ($num\_kv\_heads < num\_attention\_heads$), take the maximum score within the query head group corresponding to each KV head as that KV head's representative score:

$$
\text{grouped\_score}[g] = \max_{h \in \text{group}[g]} \text{score}_h
$$

**Top ratio selection**: sort the scores of all layers and all KV heads together and take the top `induction_head_ratio` (default 14%) of KV heads as the prefix matching selection (induction heads), and the top `echo_head_ratio` (default 1%) of KV heads as the copying matching selection (echo heads). The valid value range of both ratios is $[0, 1]$.

### 2.4 Key Properties

- **attn_head granularity**: outputs a KV-head-granularity selection list (several head indices per layer) rather than a layer-granularity ranking.
- **Two-channel selection**: induction heads (prefix matching) and echo heads (copying matching) are selected independently by ratio, corresponding to two different segment-spanning information transfer mechanisms.
- **Controlled synthetic input**: the calibration input is constructed internally by the algorithm from a fixed recipe and reads no external calibration set, so the result is independent of the user's business corpus and highly reproducible.
- **The ratio determines the selection size**: how many heads to retain is determined not by a score threshold but by the top ratio, and the ratio setting directly determines the size of the selected set.
- **Positional encoding consistency**: positional encoding is applied to the injected Q / K before computing the scores, so that the measurement conditions stay consistent with the attention pattern during real inference.
- **Output is a discrete set**: it outputs only the selected head indices, not a sensitivity score ranking, so it cannot be used for fallback decisions with continuous thresholds.
- **The result is used for compression, not fallback**: the output is `head.pt` (a `.pt` file holding a serialized dict), delivered to the MindIE inference framework for long-sequence KV cache compression. Users can complete long-sequence compression from `head.pt`, and the compressed model can be used for subsequent inference deployment; for the specific list of adapted quantization models, see the "Models Supported by MindIE" section of *[What Is MindIE](https://www.hiascend.com/document/detail/en/mindie/10RC3/whatismindie/mindie_what_0001.html)*. This artifact is not used for the exclude / include configuration of a quantization YAML, nor for layer fallback decisions.

### 2.5 Use Cases

- Long-sequence inference scenarios where it is necessary to know which KV heads carry segment-spanning information transfer (induction / copying heads) in the model.
- Scenarios that need to determine the retention set of "which KV heads should be kept" for KV cache compression.
- Analysis scenarios that require no external calibration set and want to screen the model's attention head capabilities quickly.

### 2.6 Usage Restrictions

- Currently only large language models (LLMs) are supported; multimodal understanding models (VLMs) and multimodal generative models (text-to-image / text-to-video, and so on) are not supported.
- Only the Q / K projection layers (or fused QKV layer) of Transformer self-attention are analyzed; other structures are out of scope. For models whose projection layer naming does not match the default pattern, the model adapter must provide the naming information through `RaCompressAnalysisInterface.get_proj_names`.
- The result depends on the controlled repeated-segment input structure and reflects the segment-spanning retrieval capability of attention heads; it cannot be equated directly with the model's quality on real long-text tasks.
- The output is a discrete selection set rather than a continuous score; a change in the selection ratio changes the result as a whole, and results at different ratios are not comparable in the sense of "score magnitude".
- For the supported range of `model_type`, see the *[Foundation Model Support Matrix](../../model/README.md)* and the corresponding adapter implementations.

---

## 3. Related Workflows

- *[Attention Head Screening and Analysis Usage Guide](../../../user_guide/usage_sensitive_attn_head_analysis.md)*: this algorithm is used as the metric of `attn_head` scope analysis.
- On the MindIE inference framework side (the consumer of the deliverable): the `head.pt` produced by the analysis is delivered to the MindIE inference framework, from which users can complete long-sequence compression and use the compressed model for subsequent inference deployment; for the specific list of adapted quantization models, see the "Models Supported by MindIE" section of *[What Is MindIE](https://www.hiascend.com/document/detail/en/mindie/10RC3/whatismindie/mindie_what_0001.html)*.

---

## 4. Related Terms

- [KVCache Quant](../kvcache_quant/term_kvcache_quant.md): a companion term and the quantization solution for the KV cache, in the same long-sequence memory optimization category as RA Compress.
- [KV Smooth](../kv_smooth/term_kv_smooth.md): a companion term and an outlier suppression algorithm for the KV cache.
- [Attention MSE](../attention_mse/term_attention_mse.md): a related term and an attn-scope sensitive layer analysis metric (used for FA fallback decisions, which differs in purpose from RA Compress's head selection).
- [MSE Layer Wise](../mse_layer_wise/term_mse_layer_wise.md): a companion term and a layer-wise sensitive layer analysis metric.

---

## 5. Reference Documents

1. Paper: *RazorAttention: Efficient KV Cache Compression Through Retrieval Heads* (background on induction / copying heads)
