# KVCache-PerChannel Quantization

> **Entry Category**: Quantization Data Format ([KVCache Quantization](README.md))<br>
> **English Name**: KV Cache Per-Channel Quantization<br>
> **Application Domain**: KVCache compression, long-context inference acceleration<br>
> **Hosting IR Class**: `FakeQuantDynamicCache` ([`msmodelslim/ir/attention.py`](../../../../../msmodelslim/ir/attention.py))<br>

## 1. Overview

KVCache-PerChannel quantization applies [INT8](../../dtype_ops/term_int8.md) per-channel quantization to the K/V tensors cached by the attention mechanism. The KVCache is a **special activation** in which the Key/Value projections of historical tokens are cached across time steps; its memory grows linearly with sequence length and is the main memory bottleneck of long-context inference, so quantizing K/V halves the cache memory. It is currently the only mode in the [KVCache Quantization](README.md) category, and "PerChannel" in the mode name means that the quantization parameters are shared along the hidden dimension (channel).

---

## 2. Term Description

### 2.1 Mode Specification

| Dimension | Value | Description |
|------|------|------|
| Quantization Target | The cached K/V tensors | The Key/Value cache of the attention mechanism; it does not participate in weight matrix multiplication and only participates in the attention computation. It does not change the quantization scheme of the weights and activations and can be stacked with linear layer quantization |
| Bit Width | 8bit | The cache drops from FP16 to INT8, halving both the memory and the KV read/write bandwidth |
| Data Type | INT8 | Integer format; the cache uses 1 byte per element |
| Parameter Acquisition | Static | The scale/offset is determined during quantization and then fixed as a parameter (`requires_grad=False`) and used directly at inference; "Dynamic" in the name refers to quantizing the dynamically growing cache in real time at inference, not to computing the scale online |
| Quantization Granularity | per-channel | The scale/offset is shared along the hidden dimension (head_dim); the number of parameters is on the same order as the hidden size, so the overhead is negligible. It adapts to the numerical differences between hidden dimensions |
| Symmetry | Symmetric or asymmetric | Chosen according to whether the numerical distribution is symmetric; symmetric uses only the scale, while asymmetric adds an offset |

### 2.2 Quantization Formula

Quantization and dequantization use the standard linear mapping. Symmetric quantization (scale only, no zero point):
$$q = \mathrm{round}(x / s), \qquad \hat{x} = q \cdot s, \qquad s = \frac{\max(|x|)}{2^{b-1}-1}$$

Asymmetric quantization (with zero-point offset):
$$q = \mathrm{round}(x / s) + z, \qquad \hat{x} = (q - z) \cdot s, \qquad s = \frac{\max(x) - \min(x)}{2^b - 1}$$

Here $x$ is the tensor to quantize, $q$ is the quantized integer value, $\hat{x}$ is the dequantized reconstruction, $s$ is the scale, $z$ is the zero-point offset, and $b$ is the bit width ($b=8$ for INT8 here), with $z = \mathrm{round}(-\min(x)/s)$. The target is the cached K/V tensors, and the quantization parameters are computed per-channel along the hidden dimension (head_dim) (symmetric or asymmetric); see the mode specification table above.

### 2.3 Relationship to Other Modes

- **Relationship to [FA Quantization](../fa_quantization/README.md) (advanced relationship)**:
  - This mode quantizes only K/V and solves the "storage" problem—halving the cache memory and supporting longer contexts.
  - FA quantization further quantizes Q on top of it, additionally enabling low-precision attention matrix computation. This mode is the necessary basis of FA quantization: if K is not quantized, quantizing only Q cannot use integer/low-precision attention score computation.

- **Relationship to [W8A8 Static Quantization](../linear_layer_quantization/term_w8a8_static.md) (orthogonal, can be stacked)**:
  - The quantization targets differ: this mode acts on the attention cache (a special activation), while W8A8 acts on the weight and activation matrix multiplication.
  - The two do not interfere and can be combined: the weights and activations are quantized with W8A8, and the KVCache is compressed separately with this mode, jointly forming the overall quantization plan.

### 2.4 Applicable Scenarios and Limitations

#### 2.4.1 Applicable Scenarios

- **Long-context inference**: scenarios where the context length is large and the KV memory becomes the bottleneck, such as long-document question answering and code-repository understanding.
- **High-concurrency serving**: compresses the per-request cache memory, raising the concurrent batch size under the same memory budget.
- **As the basis of FA quantization**: after K/V quantization, Q quantization is stacked on top, enabling low-precision attention matrix computation in addition to saving KV memory.

#### 2.4.2 Usage Limitations

- **Currently only per-channel + INT8 is supported**: in msModelSlim, `FakeQuantDynamicCache` registers only INT8 per-channel (symmetric/asymmetric); finer granularity or a lower bit width is not yet supported.
- **Accuracy-sensitive scenarios need evaluation**: KV participates in the attention score computation, so the quantization error may be accumulated and amplified, and accuracy must be verified under extremely long sequences.

---

## 3. Related Workflows

- *[One-Click Quantization Usage Guide](../../../user_guide/usage_one_click_quantization.md)*: select this mode via a YAML configuration and run quantization.
- *[Quantization Accuracy Tuning Guide](../../../user_guide/process_quantization_precision_tuning.md)*: accuracy degradation caused by this mode can be rolled back layer by layer and tuned through that workflow.

---

## 4. Related Terms

- [Quantization Modes](../README.md): the parent concept; this entry belongs to the [KVCache Quantization](README.md) category and is one concrete mode under it.
- [FA Quantization](../fa_quantization/README.md): an advanced concept; it adds Q quantization on top of K/V quantization.
- [W8A8 Static Quantization](../linear_layer_quantization/term_w8a8_static.md): a companion mode that can be stacked with this mode.
- [KVCache Quant Cache Quantization Algorithm](../../quantization_algorithms/kvcache_quant/term_kvcache_quant.md): a companion term, the algorithm processor documentation corresponding to this mode.
- [KV Smooth Cache Smoothing Algorithm](../../quantization_algorithms/kv_smooth/term_kv_smooth.md): a prerequisite entry; smooths KV before quantization to reduce quantization error.

---

## 5. Reference Documents

1. *[KVCache Quant Parameter Configuration Workflow Guide](../../quantization_algorithms/kvcache_quant/usage_kvcache_quant.md)*
2. *[KV Smooth Parameter Configuration Workflow Guide](../../quantization_algorithms/kv_smooth/usage_kv_smooth.md)*
3. *[Quantization Modes](../README.md)*
