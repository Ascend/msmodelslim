# Quantization Modes

> **Entry Category**: Quantization Basic Concept ([Quantization Data Type](../dtype_ops/README.md))<br>
> **English Name**: Quantization Mode<br>
> **Application Domain**: Large language model quantization and compression, inference acceleration, KVCache compression<br>

<a id="overview"></a>

## 1. Overview

Before reading this entry, it is recommended to first understand the basic concepts in [Quantization Data Type](../dtype_ops/README.md), such as [Quantization and Dequantization](../dtype_ops/term_quantization.md), scale, and common data types.

A quantization mode is a foundational concept that describes how quantization is applied to a particular class of model structure. The basic object of quantization is the **tensor**—weights, activations, and cached K/V are all tensors; a quantization mode is the combination formed by applying some quantization (bit width, data type, parameter acquisition, quantization granularity, symmetry) to one or several such tensors respectively. When a quantization mode is mentioned, it usually states at the same time which structure it acts on, which tensors within it are quantized, and at what bit width and with what parameter acquisition. Take W8A8 static quantization as an example—it is **quantization of the linear layer**: both weights and activations are 8bit (INT8), and both use **static quantization** ([W8A8 Static Quantization](linear_layer_quantization/term_w8a8_static.md)).

By quantization target, common quantization modes fall into three categories: **linear layer quantization** acts on the weights and activations of matrix multiplication; **KVCache quantization** acts on the K/V cache of the attention mechanism; **FA quantization** is an advanced form of KVCache quantization—building on quantized K/V it further quantizes Q, enabling low-precision attention matrix computation. The three are covered in [Linear Layer Quantization](linear_layer_quantization/README.md), [KVCache Quantization](kv_cache_quantization/README.md), and [FA Quantization](fa_quantization/README.md) respectively.

---

<a id="how-to-read"></a>

## 2. Term Description

### 2.1 How to Understand a Quantization Mode

A quantization mode name is usually composed of "acting structure + tensor bit width + parameter acquisition + quantization granularity + data type"; once these elements are mastered, any mode name can be decoded:

- **Acting structure**: linear layers (Linear/MatMul) use the `WxAy` notation for the bit-width combination of weights and activations. The suffix is optional: you may write only the bit width (such as `W8A8`, `W4A4`), or append the data type or the parameter acquisition (such as `W8A8 static`, `W4A4 MX dynamic`). KVCache acts on the cached K/V tensors; FA quantization acts on the attention inputs Q/K/V.
- **Tensor bit width**: W and A denote the bit widths of the weights and activations respectively. The lower the bit width, the greater the memory-traffic and compute benefit, but the lower the numerical resolution and the higher the risk of accuracy loss.
- **Parameter acquisition**: static (the scale/offset is obtained by offline calibration and fixed at inference time, such as [W8A8 Static Quantization](linear_layer_quantization/term_w8a8_static.md)) or dynamic (computed online from statistics at inference time, such as [W8A8 Dynamic Quantization](linear_layer_quantization/term_w8a8_dynamic.md) and [W4A4 Dynamic Quantization](linear_layer_quantization/term_w4a4_dynamic.md)).
- **Quantization granularity**: the range over which a group of elements shares one set of quantization parameters. Common granularities include per-tensor, per-channel, per-group, per-token, per-head, per-block, and so on; how coarse or fine it is depends on **how many elements each set of quantization parameters covers**—the fewer the elements covered (the smaller the block), the finer the granularity, the more accurately the numerical distribution is characterized and the better the accuracy, but the greater the number of quantization parameters and the compute overhead (for example, per-token is covered by [W8A8 Dynamic Quantization](linear_layer_quantization/term_w8a8_dynamic.md), and per-block by [W8A8 MX Dynamic Quantization](linear_layer_quantization/term_w8a8_mx_dynamic.md)).
- **Data type**: the numerical format after quantization, such as INT integers (INT8/INT4), FP8 floating point (E4M3), and MXFP8/MXFP4 with a block-level shared exponent. For the correspondence between the standard definition of each data type and the [`qal/dtype_ops`](../dtype_ops/README.md) implementation in this repository, see [Quantization Data Type](../dtype_ops/README.md).

---

<a id="categories"></a>

### 2.2 Classification of Quantization Modes

By quantization target, common quantization modes are divided into three categories:

- **[Linear Layer Quantization](linear_layer_quantization/README.md)**: quantizes the weights and activations of the linear layers (Linear / MatMul), which account for the largest share of large language models. It is the main body of quantization modes and has the largest number of modes, such as [W8A8 Static Quantization](linear_layer_quantization/term_w8a8_static.md) and [W4A4 Dynamic Quantization](linear_layer_quantization/term_w4a4_dynamic.md).
- **[KVCache Quantization](kv_cache_quantization/README.md)**: quantizes the KVCache of the attention mechanism. The KVCache is essentially a special kind of activation—the Key/Value projections of historical tokens are cached for reuse by subsequent tokens, and its memory grows linearly with sequence length; quantizing K/V halves the cache memory, such as [KVCache-PerChannel Quantization](kv_cache_quantization/term_kv_cache_perchannel.md).
- **[FA Quantization](fa_quantization/README.md)**: an advanced form of KVCache quantization—building on quantized K/V it further quantizes the Q fed into Flash Attention, enabling low-precision attention matrix computation, such as [FA INT8 PerHead Quantization](fa_quantization/term_fa_int8_perhead.md).

The three categories act at different positions in the computation graph and can be used independently or in combination, together forming the optimization space for inference acceleration; among them, FA quantization builds on KVCache quantization and is its extension at the compute level.

---

<a id="modes"></a>

### 2.3 Mode Index

Below is the complete list of quantization modes supported by msModelSlim (categorized by quantization target), which also serves as the overview entry point for each specific quantization mode entry. The "Hosting IR Class" column lists class names in [`msmodelslim/ir/`](../../../../msmodelslim/ir/); the **FakeQuant** (fake quantization) in the class names refers to a placeholder layer that dequantizes immediately after quantization, used to simulate the numerical effect of real quantization in floating-point computation:

#### 2.3.1 Linear Layer Quantization

| Mode | Hosting IR Class | Weight/Activation Quantization |
|------|-----------|---------------|
| [W8A8 Static Quantization](linear_layer_quantization/term_w8a8_static.md) | `W8A8StaticFakeQuantLinear` | INT8 per-channel / per-tensor static |
| [W8A8 Dynamic Quantization](linear_layer_quantization/term_w8a8_dynamic.md) | `W8A8DynamicPerChannel/PerGroupFakeQuantLinear` | INT8 per-channel/per-group / per-token dynamic |
| [W4A4 Dynamic Quantization](linear_layer_quantization/term_w4a4_dynamic.md) | `W4A4DynamicPerChannel/PerGroupFakeQuantLinear` | INT4 per-channel/per-group / per-token dynamic |
| [W8A16 Static Quantization](linear_layer_quantization/term_w8a16_static.md) | `W8A16StaticPerChannel/PerGroupFakeQuantLinear` | INT8 / FP16 activation |
| [W4A8 Dynamic Quantization](linear_layer_quantization/term_w4a8_dynamic.md) | `W4A8DynamicFakeQuantLinear` | INT4 / INT8 per-token dynamic |
| [W8A8 PD-Mix Quantization](linear_layer_quantization/term_w8a8_pdmix.md) | `W8A8PDMixFakeQuantLinear` | INT8 per-token (prefill) / per-tensor (decode) |
| [W8A8 FP8 Dynamic Quantization](linear_layer_quantization/term_w8a8_fp8_dynamic.md) | `WFP8AFP8DynamicPerChannelFakeQuantLinear` | FP8(E4M3) per-channel / per-token dynamic |
| [W8A8 MX Dynamic Quantization](linear_layer_quantization/term_w8a8_mx_dynamic.md) | `W8A8MXDynamicPerBlockFakeQuantLinear` | MXFP8 per-block / per-block dynamic |
| [W4A8 MX Dynamic Quantization](linear_layer_quantization/term_w4a8_mx_dynamic.md) | `W4A8MXDynamicPerBlockFakeQuantLinear` | MXFP4 / MXFP8 per-block dynamic |
| [W4A4 MX Dynamic Quantization](linear_layer_quantization/term_w4a4_mx_dynamic.md) | `W4A4MXDynamicPerBlockFakeQuantLinear` | MXFP4 per-block / per-block dynamic |
| [W4A4 MX Dual-Scale Quantization](linear_layer_quantization/term_w4a4_mx_dualscale.md) | `W4A4MXDynamicDualScaleFakeQuantLinear` | MXFP4 dual scale / per-block dynamic |
| [W8A8 Sparse Quantization](linear_layer_quantization/term_w8a8_sparse.md) | `W8A8StaticFakeQuantLinear` (export type `W8A8S`) | INT8 using only a subset of the codebook; `W8A8SC` is the secondary-compression form |
| [Float Sparse Quantization](linear_layer_quantization/term_float_sparse.md) | `W16A16sLinear` | 16bit weights/activations, weights zeroed for sparsity; `W16A16SC` is the secondary-compression form |
| [SVDQuant Quantization](linear_layer_quantization/term_svdquant.md) | `SVDResidualWrapper` ([`svd_residual.py`](../../../../msmodelslim/ir/svd_residual.py), used together with linear_quant) | Low-rank decomposition + low-bit quantization of the residual |

> Note: prefill and decode are the two phases of LLM inference—prefill processes the entire input prompt at once (compute-intensive), while decode generates tokens one by one (memory-traffic-intensive); see [Linear Layer Quantization](linear_layer_quantization/README.md) for details.

#### 2.3.2 KVCache Quantization

| Mode | Hosting IR Class | Quantization Parameters |
|------|-----------|----------|
| [KVCache-PerChannel Quantization](kv_cache_quantization/term_kv_cache_perchannel.md) | `FakeQuantDynamicCache` | INT8 per-channel (symmetric/asymmetric) |

#### 2.3.3 FA Quantization (Advanced Form of KVCache Quantization)

FA quantization **is essentially a composite quantization mode**, acting on the three attention branches **Q/K/V**, with each branch choosing one activation quantization mode; the combination of the three branches constitutes the **holistic mode** (see [FA Quantization](fa_quantization/README.md)). The table below lists holistic modes that have been validated in practice, each corresponding to one entry.

| Mode | Hosting IR Class | Q Branch | K Branch | V Branch | Parameter Acquisition |
|------|-----------|-----------|----------|----------|---------|
| [FA INT8 PerHead Quantization](fa_quantization/term_fa_int8_perhead.md) | `INT8FakeQuantActivationPerHead` | INT8 per-head | INT8 per-head | INT8 per-head | Static |
| [FA INT8 Dynamic Quantization](fa_quantization/term_fa_int8_dynamic.md) | `FakeQuantActivationPerToken` | INT8 per-token | INT8 per-token | INT8 per-token | Dynamic |
| [FA FP8 Dynamic Quantization](fa_quantization/term_fa_fp8_dynamic.md) | `FakeQuantActivationPerToken` | FP8 per-token | FP8 per-token | FP8 per-token | Dynamic |
| [FA MXFP4 Dynamic Quantization](fa_quantization/term_fa_mxfp4_dynamic.md) | `FakeQuantActivationPerBlock` | MXFP4 per-block | MXFP4 per-block | MXFP4 per-block | Dynamic |
| [FA Q-INT8 Dynamic K/V-INT8 Static Quantization](fa_quantization/term_fa_q_int8_dynamic_kv_int8.md) | Q: `FakeQuantActivationPerToken`; K/V: `INT8FakeQuantActivationPerHead` | INT8 per-token | INT8 per-head | INT8 per-head | Q dynamic; K/V static |
| [FA Q-FP8 Dynamic K/V-FP8 Static Quantization](fa_quantization/term_fa_q_fp8_dynamic_kv_fp8.md) | Q: `FakeQuantActivationPerToken`; K/V: `FP8FakeQuantActivationPerHead` | FP8 per-token | FP8 per-head | FP8 per-head | Q dynamic; K/V static |
| [FA QK-MXFP8 Dynamic / V-MXFP8 PerChannel Static Quantization](fa_quantization/term_fa_qk_mxfp8_dynamic_v_mxfp8_perchannel.md) | Q/K: `FakeQuantActivationPerBlock`; V: `MXFP8FakeQuantActivationPerChannel` | MXFP8 per-block | MXFP8 per-block | MXFP8 per-channel | Q/K dynamic; V static |

The three branches may use the same activation quantization mode or different ones respectively; unquantized branches keep their original precision.

---

## 3. Related Workflows

- [One-Click Quantization Usage Guide](../../user_guide/usage_one_click_quantization.md): select and execute a quantization mode via `--quant_type` or a YAML configuration.
- [Quantization Accuracy Tuning Guide](../../user_guide/process_quantization_precision_tuning.md): accuracy degradation caused by the quantization mode can be rolled back and tuned layer by layer through this workflow.

---

<a id="related-terms"></a>

## 4. Related Terms

- [Linear Layer Quantization](linear_layer_quantization/README.md): a subordinate concept, the main category of quantization modes under this entry.
- [KVCache Quantization](kv_cache_quantization/README.md): a subordinate concept, the quantization category targeting the KVCache.
- [FA Quantization](fa_quantization/README.md): a subordinate concept, an advanced category of KVCache quantization.
- [W8A8 Static Quantization](linear_layer_quantization/term_w8a8_static.md): a subordinate concept, the typical "static weight/activation" mode in linear layer quantization.
- [Linear Quantization Algorithm](../quantization_algorithms/linear_quant/term_linear_quant.md): a companion term that describes the code implementation of linear layer quantization modes.
- [KVCache Quant Cache Quantization Algorithm](../quantization_algorithms/kvcache_quant/term_kvcache_quant.md): a companion term that describes the KVCache quantization algorithm.
- [FA3 Quant Attention Activation Quantization Algorithm](../quantization_algorithms/fa3_quant/term_fa3_quant.md): a companion term that describes the FA quantization algorithm.
- [Quantization Data Type](../dtype_ops/README.md): a companion concept; the standard definitions of the five data types used by quantization modes ([INT8](../dtype_ops/term_int8.md)/[INT4](../dtype_ops/term_int4.md)/[FP8](../dtype_ops/term_fp8.md)/[MXFP8](../dtype_ops/term_mxfp8.md)/[MXFP4](../dtype_ops/term_mxfp4.md)) correspond to the [`qal/dtype_ops`](../../../../msmodelslim/ir/qal/dtype_ops/) implementation.

---

## 5. Reference Documents

1. Jacob B et al. Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference. CVPR 2018. <https://arxiv.org/abs/1712.05877>
2. Yao Z et al. ZeroQuant: Efficient and Affordable Post-Training Quantization for Large-Scale Transformers. NeurIPS 2022. <https://arxiv.org/abs/2206.01861>
3. Liu Z et al. KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache. ICML 2024. <https://arxiv.org/abs/2402.02750>
4. [Linear Quantization Parameter Configuration Workflow Guide](../quantization_algorithms/linear_quant/usage_linear_quant.md)
5. [AscendV1 Format Specification](../quantization_format/ascendv1/ascendv1_usage.md)
