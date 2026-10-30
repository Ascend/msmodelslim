# Linear Layer Quantization

> **Entry Category**: Quantization Data Format ([Quantization Modes](../README.md))<br>
> **English Name**: Linear Layer Quantization<br>
> **Application Domain**: Large language model quantization and compression, inference acceleration<br>
> **msModelSlim Implementation**: [`msmodelslim/ir/w8a8_static.py`](../../../../../msmodelslim/ir/w8a8_static.py) (the `*FakeQuantLinear` family)<br>

<a id="overview"></a>

## 1. Overview

Linear layer quantization is a category of quantization modes that quantizes the <b>weights (W) and activations (A) of the linear layers (Linear / MatMul)</b>, which account for the largest share of large language models. It is the main category under [Quantization Modes](../README.md). It quantizes the tensors on both sides of the linear layer matrix multiplication: the weight is a static parameter that must be read in full for every token; the activation is an intermediate tensor that changes in real time with the input. The bit-width combination of these two tensors is usually written with the `WxAy` notation, where W8A8 means that both the weights and the activations are 8bit (INT8).

Why focus on linear layers: every token's forward pass must read all the weights, so the weight bit width directly determines the memory-bandwidth bottleneck; the activation bit width in turn determines whether integer GEMM can be applied. This is why this category has the largest number of modes and is the main body of quantization modes. A typical example is [W8A8 Static Quantization](term_w8a8_static.md)—INT8 static quantization applied to both the weights and the activations of linear layers.

---

<a id="linear-compute"></a>

## 2. Term Description

### 2.1 How a Linear Layer Computes and What Can Be Quantized

The linear layer (Linear / MatMul) is the main computation of a Transformer—the Q/K/V projections, the attention output projection, and the up/gate/down layers of the MLP are all linear layers. One linear layer computation is one matrix multiplication:

$$Y = X \cdot W + b$$

Here $X$ is the input activation (shape $(batch, seq, in\_dim)$), $W$ is the weight (shape $(in\_dim, out\_dim)$), $b$ is the bias, and $Y$ is the output activation. LLM inference has two phases: **prefill** processes the entire input prompt at once (compute-intensive), and **decode** generates tokens one by one (memory-traffic-intensive).

A matrix multiplication has only two input tensors, so a linear layer also has only two quantizable targets:

- **Weight $W$**: a static tensor, fixed after training and read in full once on every inference step. In the decode phase, weight memory traffic is the main throughput bottleneck, so weight quantization focuses on **compression**—lowering the bit width to reduce memory traffic, such as the INT8 weights of [W8A8 Static Quantization](term_w8a8_static.md) and the INT4 weights of [W4A8 Dynamic Quantization](term_w4a8_dynamic.md).
- **Input activation $X$**: a dynamic tensor, produced in real time by the previous layer with a distribution that varies with the input, and sensitive to quantization error. Activation quantization focuses on preserving accuracy, so finer-grained schemes such as dynamic quantization are usually preferred, such as the per-token dynamic activation quantization of [W8A8 Dynamic Quantization](term_w8a8_dynamic.md).

A quantization mode is essentially **the combination formed by applying some quantization (bit width, data type, parameter acquisition, quantization granularity, symmetry) to these two tensors respectively**: for example, [W8A8 Static Quantization](term_w8a8_static.md) applies 8bit INT8 static quantization to both W and A, [W8A16 Static Quantization](term_w8a16_static.md) quantizes only W while keeping A at 16bit, and [W4A8 Dynamic Quantization](term_w4a8_dynamic.md) uses W 4bit and A 8bit with the activation dynamic. The choice of dimensions for each of the two tensors constitutes the "general quantization choices of this category" described below.

---

<a id="mode-options"></a>

### 2.2 General Quantization Choices of This Category

The same class of linear layer structure can be quantized in different ways, differing along the following dimensions:

- **Bit-width combination**: the weights and activations may use the same or different bit widths. The same bit width (W8A8, W4A4) balances compression and accuracy; an activation bit width higher than the weight (W8A16, W4A8) trades some compression for accuracy. See [W8A16 Static Quantization](term_w8a16_static.md) and [W4A8 Dynamic Quantization](term_w4a8_dynamic.md).
- **Parameter acquisition**: static (offline calibration, fixed at inference, such as [W8A8 Static Quantization](term_w8a8_static.md)) or dynamic (counted online at inference, no calibration, such as [W8A8 Dynamic Quantization](term_w8a8_dynamic.md) and [W4A4 Dynamic Quantization](term_w4a4_dynamic.md)).
- **Quantization granularity**: weights are usually statically quantized per-channel or per-group; activations are usually dynamically quantized per-token or statically quantized per-tensor (the whole tensor); the MX family uses per-block quantization with a block-level shared exponent.
- **Data type**: INT integers (INT8/INT4), FP8 floating point (such as [W8A8 FP8 Dynamic Quantization](term_w8a8_fp8_dynamic.md)), and MX block-level shared exponents (such as [W4A4 MX Dynamic Quantization](term_w4a4_mx_dynamic.md) and [W4A4 MX Dual-Scale Quantization](term_w4a4_mx_dualscale.md)).
- **Special variants**: [W8A8 PD-Mix Quantization](term_w8a8_pdmix.md) switches the activation strategy by Prefill/Decode phase; [W8A8 Sparse Quantization](term_w8a8_sparse.md) does not use all 256 INT8 levels and is then compressed a second time to raise the compression ratio; [Float Sparse Quantization](term_float_sparse.md) does not lower the bit width but zeroes out part of the floating-point weights.

Each mode trades off between **accuracy loss** (the degree of degradation relative to the floating-point baseline) and **performance** (the weight/activation memory-traffic and compute benefit). Accuracy loss is graded as **low / medium / high**; the modes are not compared on accuracy directly.

| Mode | Accuracy Loss | Performance (Qualitative) | Trade-off Point |
|------|---------|------------|---------|
| [W8A8 Dynamic Quantization](term_w8a8_dynamic.md) | Low | High | Activation per-token dynamic; among the 8bit schemes the loss is relatively small |
| [W8A8 FP8 Dynamic Quantization](term_w8a8_fp8_dynamic.md) | Low | High | The same tier as W8A8 dynamic; depends on FP8 operators |
| [W8A8 PD-Mix Quantization](term_w8a8_pdmix.md) | Low | High | Prefill per-token, Decode static; overall the same tier as W8A8 dynamic |
| [W8A8 Static Quantization](term_w8a8_static.md) | Low | High | Full-range INT8 static; the loss is still in the low tier |
| [W8A8 MX Dynamic Quantization](term_w8a8_mx_dynamic.md) | Low | High | 8bit MX per-block, the same tier as the other W8A8 modes |
| [W8A16 Static Quantization](term_w8a16_static.md) | Low | Medium | The activation is not quantized, so the loss is low; without integer GEMM the performance benefit is halved |
| [Float Sparse Quantization](term_float_sparse.md) | Medium | Medium | The bit width is not lowered, but zeroing introduces a medium accuracy loss |
| [W8A8 Sparse Quantization](term_w8a8_sparse.md) | Medium | High | Uses only a subset of the 256 levels, so the loss is higher than full-range W8A8 in exchange for a higher compression ratio |
| [W4A8 Dynamic Quantization](term_w4a8_dynamic.md) | Medium | High | Weight 4bit, activation 8bit dynamic; the loss is medium |
| [W4A8 MX Dynamic Quantization](term_w4a8_mx_dynamic.md) | Medium | High | The same tier as W4A8 dynamic |
| [W4A4 Dynamic Quantization](term_w4a4_dynamic.md) | High | Highest | Dual 4bit; the accuracy loss is high and the memory-traffic benefit is the largest |
| [W4A4 MX Dynamic Quantization](term_w4a4_mx_dynamic.md) | High | Highest | W4A4 class, dual 4bit MX |
| [W4A4 MX Dual-Scale Quantization](term_w4a4_mx_dualscale.md) | High | High | Still W4A4 class; the dual-layer scale only mitigates the loss and does not change the tier |

> Note: "Performance" in the table refers specifically to the weight/activation memory-traffic and compute benefit after quantization; the actual loss and benefit depend on the target hardware operator support and the model distribution. See each mode entry and the [Mode Index](../README.md#modes) for details.

---

<a id="modes"></a>

### 2.3 Mode List of This Category

The following is the list of linear layer quantization modes supported by msModelSlim (for complete specifications including the hosting IR class and the quantization parameters, see the [Mode Index](../README.md#modes)):

| Mode | One-liner |
|------|--------|
| [W8A8 Static Quantization](term_w8a8_static.md) | Both weights and activations INT8 static |
| [W8A8 Dynamic Quantization](term_w8a8_dynamic.md) | Weight static, activation per-token dynamic |
| [W4A4 Dynamic Quantization](term_w4a4_dynamic.md) | Both weights and activations INT4, activation dynamic |
| [W8A16 Static Quantization](term_w8a16_static.md) | Only the weight is INT8; the activation stays FP16 and is not quantized |
| [W4A8 Dynamic Quantization](term_w4a8_dynamic.md) | Weight INT4, activation INT8 dynamic |
| [W8A8 PD-Mix Quantization](term_w8a8_pdmix.md) | The activation strategy switches with the Prefill/Decode phase |
| [W8A8 FP8 Dynamic Quantization](term_w8a8_fp8_dynamic.md) | Both weights and activations FP8(E4M3) dynamic |
| [W8A8 MX Dynamic Quantization](term_w8a8_mx_dynamic.md) | Both weights and activations MXFP8 per-block dynamic |
| [W4A8 MX Dynamic Quantization](term_w4a8_mx_dynamic.md) | Weight MXFP4, activation MXFP8 per-block dynamic |
| [W4A4 MX Dynamic Quantization](term_w4a4_mx_dynamic.md) | Both weights and activations MXFP4 per-block dynamic |
| [W4A4 MX Dual-Scale Quantization](term_w4a4_mx_dualscale.md) | MXFP4 dual scale, per-block dynamic |
| [W8A8 Sparse Quantization](term_w8a8_sparse.md) | INT8 using only a codebook subset, then `W8A8SC` secondary compression; a higher compression ratio than W8A8 |
| [Float Sparse Quantization](term_float_sparse.md) | 16bit weights/activations with the weights zeroed for sparsity; `W16A16SC` is the secondary-compression form |
| [SVDQuant Quantization](term_svdquant.md) | Low-rank decomposition + low-bit quantization of the residual, used together with linear layer quantization |

---

## 3. Related Workflows

- [One-Click Quantization Usage Guide](../../../user_guide/usage_one_click_quantization.md): select a linear layer quantization mode via `--quant_type` or a YAML configuration.
- [Quantization Accuracy Tuning Guide](../../../user_guide/process_quantization_precision_tuning.md): accuracy degradation caused by the quantization mode can be rolled back and tuned layer by layer through this workflow.

---

<a id="related-terms"></a>

## 4. Related Terms

- [Quantization Modes](../README.md): the parent concept; this entry is the "linear layer quantization" category.
- [KVCache Quantization](../kv_cache_quantization/README.md): a sibling concept, the quantization category acting on the attention K/V cache, which can be stacked with this category.
- [FA Quantization](../fa_quantization/README.md): a sibling concept, an advanced form of KVCache quantization.
- [W8A8 Static Quantization](term_w8a8_static.md): a subordinate concept, the most basic static mode in this category.
- [W8A8 Sparse Quantization](term_w8a8_sparse.md): a subordinate concept, in which the INT8 codebook is narrowed and then compressed a second time.
- [Float Sparse Quantization](term_float_sparse.md): a subordinate concept, the zeroing-sparsity baseline that does not lower the bit width.
- [Linear Quantization Algorithm](../../quantization_algorithms/linear_quant/term_linear_quant.md): a companion term that describes the code implementation of linear layer quantization modes.

---

## 5. Reference Documents

1. *[Linear Quantization Parameter Configuration Workflow Guide](../../quantization_algorithms/linear_quant/usage_linear_quant.md)*
2. *[Quantization Modes](../README.md)*
