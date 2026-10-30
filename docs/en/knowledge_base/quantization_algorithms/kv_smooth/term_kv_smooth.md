# KV Smooth Cache Smoothing Algorithm Terminology Entry

> **Entry Category**: [Outlier Suppression Algorithms](../README.md#1-outlier-suppression-algorithms)<br>
> **English Name**: kv_smooth<br>
> **Application Domain**: KV cache compression, large language model quantization and compression<br>

---

## 1. Overview

KV Smooth is an outlier suppression algorithm for KV cache quantization. It exploits the invariance of the Q/K dot product under reciprocal scaling to migrate large magnitudes in the Key channels to the Query side, compressing the Key dynamic range while keeping the attention score as equivalent as possible; its core characteristics are cache-side range shaping, paired Q/K compensation, and a constraint on the RoPE pairing structure, and it can provide upstream smoothing for [KVCache Quant](../kvcache_quant/term_kvcache_quant.md).

---

## 2. Term Description

KV Cache quantization aims to reduce the memory footprint of long-sequence inference, but outliers in the Key tensor inflate the quantization scale and severely degrade the quantization accuracy of most channels. Because inference quantizes only the `key_states` written into the KV cache and not the `query_states`, the outliers can be migrated from `key_states` to `query_states`, obtaining more robust KV cache quantization while keeping the attention score unchanged.

### 2.1 Core Idea

The core idea of KV Smooth is to exploit the "reciprocal scaling invariance" of the attention dot product and migrate the channel peaks on the Key side that are unfavorable to cache quantization over to the Query side. Applying $K'=K/s$ and $Q'=Q\cdot s$ to the same feature channel ideally gives $Q'K'^T=QK^T$, so the attention logits do not change because of this reparameterization, while the dynamic range of the K that enters the KV cache is compressed.

This idea is especially suited to scenarios where "only the KV cache is stored at low bit width": K resides for a long time and grows with the sequence length, whereas Q only participates in the current step's computation and does not enter the cache. Under the condition of preserving the dot-product equivalence, migrating part of the magnitude pressure from K to Q is therefore often more economical than letting a small number of outlier channels in K dominate the cache quantization scale for a long time. For models that use RoPE, the pairing structure of the rotated two-dimensional subspace must also be respected so that scaling and rotation remain commutable or equivalently compensated.

Its effectiveness comes from the cost asymmetry between the cache side and the immediate-computation side: in a typical KV-cache-only quantization scheme, the scale error of K is stored for a long time and reused across a large number of subsequent tokens, whereas Q does not enter the cache.

### 2.2 Working Mechanism

Let the Query/Key of one attention head before RoPE be $Q,K$; introduce a diagonal scale matrix $D=\operatorname{diag}(s)$ over the feature dimension and apply $Q'=QD$ and $K'=KD^{-1}$. If no subsequent operation breaks this correspondence, then $Q'K'^T=QDD^{-1}K^T=QK^T$, so the attention logits before softmax stay unchanged. The scale is usually constructed from the per-channel peaks of the calibration K, so as to compress first the channels most likely to dominate the cache quantization range.

RoPE rotates two coordinates that are adjacent or paired by design as a two-dimensional vector. A general non-uniform diagonal scaling does not commute with a two-dimensional rotation: if a pair of RoPE coordinates uses different scales, scaling before rotation and rotating before scaling give different results, breaking the exact dot-product equivalence. The solution is to share a compatible scale across the RoPE-paired dimensions so that $D$ is equivalent to a scalar multiple within each two-dimensional rotation subspace; in GQA scenarios the K scale must also be aligned with the multiple Q heads that share it.

### 2.3 Mathematical Description

Let the scaling vector be $s$ and the outputs of the K and Q branches before RoPE be $K$ and $Q$; the smoothed values are then:

$$
K' = \frac{K}{s}, \quad Q' = Q \times s
$$

The attention score remains unchanged:

$$
Q'K'^T = Q \times s \cdot \left(\frac{K}{s}\right)^T = QK^T
$$

- $K$: the Key representation before being written into the KV cache.
- $Q$: the Query representation that participates in the dot product with the Key.
- $s$: the positive scaling vector constructed from the per-channel magnitude statistics of the calibration Key.

The smoothing statistics are usually based on the per-channel peak of each layer's Key, and the reciprocal scale is absorbed into a projection or normalization parameter before RoPE that can be transformed equivalently.

The RoPE constraint can be seen directly from the two-dimensional rotation. For a given RoPE-pair dimension, the rotation matrix is:

$$
R(\theta)=\begin{bmatrix}\cos\theta&-\sin\theta\\\sin\theta&\cos\theta\end{bmatrix}.
$$

If the scaling of that two-dimensional subspace is $D=\operatorname{diag}(s_1,s_2)$, then in general $R(\theta)D\ne DR(\theta)$; only when $s_1=s_2$, giving $D=sI$, does it commute with the rotation for all $\theta$. The RoPE-paired coordinates must therefore share a scale so that the scaling can be moved safely across RoPE while preserving equivalence.

### 2.4 Key Properties

- **Attention score unchanged**: the scaling migration keeps $QK^T$ unchanged and does not alter the attention score.
- **Cache-side outlier migration**: Key peaks are reduced and the corresponding magnitude is migrated to the Query; in the typical KV-cache-only quantization scenario this lowers the cache-side quantization pressure.
- **RoPE pairing handling**: exploits the paired-rotation property of RoPE channels and takes the maximum across paired channels before restoring.
- **Adjustable smoothing strength**: the scaling strength determines how much dynamic range is migrated from Key to Query.
- **RoPE constraint**: the rotation-paired dimensions must share a compatible scale, otherwise the reciprocal scaling and RoPE cannot remain exactly equivalent.

But the scale cannot grow without bound. Excessive smoothing makes the Q magnitude too large, which may affect the Q's own low-precision operators, numerical stability, or other coupled quantization; and if the RoPE pairing constraint is handled incorrectly it directly breaks the $QK^T$ equivalence.

### 2.5 Use Cases

- Scenarios where Key outliers must be suppressed before KV cache quantization (such as [KVCache Quant](../kvcache_quant/term_kvcache_quant.md)) to improve cache quantization accuracy.
- Long-sequence inference scenarios that need to compress the KV cache dynamic range to reduce memory usage.

### 2.6 Usage Restrictions

- The algorithm requires that an equivalent scaling relationship between the corresponding Q/K channels can be established across RoPE; if the model structure inserts a nonlinear transform between them that breaks this relationship, the reciprocal scaling cannot be applied directly.
- For structures using GQA/MQA, the Query heads sharing the same Key head must be aligned with its scale, otherwise the original attention dot-product relationship is broken.
- Based on the assumption of paired-channel reduction in RoPE; non-RoPE structures require careful evaluation.
- Based on the assumption that only the KV cache's `key_states`/`value_states` are quantized and `query_states` are not.

These restrictions reflect the algorithm's dependence on model structure and equivalence transformation conditions. If the target model does not satisfy the corresponding structural assumptions, applying the algorithm indiscriminately may break the original computational relationships. Therefore, when encountering incompatible structures, first narrow the scope or use a recipe already verified for the model, rather than blindly increasing the smoothing strength.

---

## 3. Related Terms

- [KVCache Quant](../kvcache_quant/term_kvcache_quant.md): a companion term; this algorithm provides outlier suppression for KV cache quantization.
- [SmoothQuant](../smooth_quant/term_smooth_quant.md): a similar algorithm in the same scaling family of outlier suppression, but aimed at activations and weights rather than the KV cache.
- [QuaRot](../quarot/term_quarot.md): a comparison algorithm that suppresses outliers by orthogonal rotation rather than scaling.
- [FA3 Quant](../fa3_quant/term_fa3_quant.md): a companion term that likewise addresses the quantization of attention activations under long sequences.

---

## 4. Reference Documents

1. [KV Smooth Parameter Configuration Workflow Guide](./usage_kv_smooth.md)
