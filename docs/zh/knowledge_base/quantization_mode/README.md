# 量化模式

<!-- waiver: G01 原因：本文件为量化模式目录的索引文档，不适用 term_<english_name>.md 词条命名 -->

> **词条类别**：量化基础概念（[量化基础](../quantization_basic/README.md)）<br>
> **英文名称**：Quantization Mode<br>
> **应用领域**：大语言模型量化压缩、推理加速、KVCache 压缩<br>

<a id="overview"></a>

## 1. 概述

阅读本词条前，建议先了解[量化基础](../quantization_basic/README.md)中的量化/反量化、scale 与常用数据类型等基本概念。

量化模式（Quantization Mode）是用于描述针对某一类模型结构如何实施量化的基础概念。量化操作的基本对象是**张量**——权重、激活、缓存的 K/V 等都是张量；量化模式即针对某个或某几个张量分别应用某种量化（位宽、数据类型、参数获取方式、量化粒度、对称性）后形成的组合。提及某个量化模式时，它通常同时指出：作用于哪类结构、量化其中的哪些张量、以什么位宽和参数获取方式量化。以 W8A8 静态量化为例——它就是**对线性层进行量化**：权重与激活均为 8bit（INT8），且权重与激活均采用**静态量化**（[W8A8 静态量化](linear_layer_quantization/term_w8a8_static.md)）。

按量化对象，常用量化模式可归为三类：**线性层量化**作用于矩阵乘法的权重与激活；**KVCache 量化**作用于注意力机制的 K/V 缓存；**FA 量化**是 KVCache 量化的进阶——在量化 K/V 的基础上进一步量化 Q，使能低精度注意力矩阵运算。三者分别见 [线性层量化](linear_layer_quantization/README.md)、[KVCache 量化](kv_cache_quantization/README.md)、[FA 量化](fa_quantization/README.md)。

---

<a id="how-to-read"></a>

## 2. 词条介绍

### 2.1 如何理解一个量化模式

一个量化模式名通常由"作用结构 + 张量位宽 + 参数获取方式 + 量化粒度 + 数据类型"几部分构成，掌握了这些要素，就能读懂任意一个模式名：

- **作用结构**：线性层（Linear/MatMul）用 `WxAy` 记法表示权重与激活的位宽组合。后缀可选：可以只写位宽（如 `W8A8`、`W4A4`），也可以追加数据类型或参数获取方式（如 `W8A8 静态`、`W4A4 MX 动态`）。KVCache 作用于缓存的 K/V 张量；FA 量化作用于注意力输入 Q/K/V。
- **张量位宽**：W 与 A 分别表示权重与激活的位宽。位宽越低，访存与计算收益越大，但数值分辨率越低、精度损失风险越高。
- **参数获取方式**：静态（离线校准得到 scale/offset，推理时固化，如 [W8A8 静态量化](linear_layer_quantization/term_w8a8_static.md)）或动态（推理时在线统计并计算，如 [W8A8 动态量化](linear_layer_quantization/term_w8a8_dynamic.md)、[W4A4 动态量化](linear_layer_quantization/term_w4a4_dynamic.md)）。
- **量化粒度**：一组元素共享一个量化参数的范围。常用粒度有 per-tensor、per-channel、per-group、per-token、per-head、per-block 等，粗细取决于**每份量化参数覆盖的元素数**——覆盖越少（块越小）越细，数值分布刻画越准、精度越好，但量化参数数量与计算开销越大（如 per-token 见 [W8A8 动态量化](linear_layer_quantization/term_w8a8_dynamic.md)，per-block 见 [W8A8 MX 动态量化](linear_layer_quantization/term_w8a8_mx_dynamic.md)）。
- **数据类型**：量化后的数值格式，如 INT 整数（INT8/INT4）、FP8 浮点（E4M3）、带块级共享指数的 MXFP8/MXFP4。

---

<a id="categories"></a>

### 2.2 量化模式的分类

按量化对象，常用量化模式分为三类：

- **[线性层量化](linear_layer_quantization/README.md)**：对大语言模型中占比最高的线性层（Linear / MatMul）权重与激活进行量化，是量化模式的主体、模式数量最多，如 [W8A8 静态量化](linear_layer_quantization/term_w8a8_static.md)、[W4A4 动态量化](linear_layer_quantization/term_w4a4_dynamic.md)。
- **[KVCache 量化](kv_cache_quantization/README.md)**：对注意力机制的 KVCache 进行量化。KVCache 本质上是一种特殊的激活——历史 token 的 Key/Value 投影被缓存下来供后续 token 复用，其显存随序列长度线性增长；量化 K/V 使缓存显存减半，如 [KVCache-PerChannel 量化](kv_cache_quantization/term_kv_cache_perchannel.md)。
- **[FA 量化](fa_quantization/README.md)**：KVCache 量化的进阶——在量化 K/V 的基础上进一步量化送入 Flash Attention 的 Q，使能低精度注意力矩阵运算，如 [FA INT8 PerHead 量化](fa_quantization/term_fa_int8_perhead.md)。

三类量化作用于计算图的不同位置，可以独立或组合使用，共同构成推理加速的优化空间；其中 FA 量化以 KVCache 量化为基础，是其在计算层面的延伸。

---

<a id="modes"></a>

## 3. 模式索引

以下为 msModelSlim 支持的量化模式全量清单（按量化对象分类），也是各具体量化模式词条的总览入口。表中"承载 IR 类"为 [`msmodelslim/ir/`](../../../../msmodelslim/ir/) 中的类名，类名里的 **FakeQuant**（伪量化）指量化后立即反量化的占位层，用于在浮点计算中模拟真实量化的数值效果：

### 3.1 线性层量化

| 模式 | 承载 IR 类 | 权重/激活量化 |
|------|-----------|---------------|
| [W8A8 静态量化](linear_layer_quantization/term_w8a8_static.md) | `W8A8StaticFakeQuantLinear` | INT8 per-channel / per-tensor 静态 |
| [W8A8 动态量化](linear_layer_quantization/term_w8a8_dynamic.md) | `W8A8DynamicPerChannel/PerGroupFakeQuantLinear` | INT8 per-channel/per-group / per-token 动态 |
| [W4A4 动态量化](linear_layer_quantization/term_w4a4_dynamic.md) | `W4A4DynamicPerChannel/PerGroupFakeQuantLinear` | INT4 per-channel/per-group / per-token 动态 |
| [W8A16 静态量化](linear_layer_quantization/term_w8a16_static.md) | `W8A16StaticPerChannel/PerGroupFakeQuantLinear` | INT8 / FP16 激活 |
| [W4A8 动态量化](linear_layer_quantization/term_w4a8_dynamic.md) | `W4A8DynamicFakeQuantLinear` | INT4 / INT8 per-token 动态 |
| [W8A8 PD-Mix 量化](linear_layer_quantization/term_w8a8_pdmix.md) | `W8A8PDMixFakeQuantLinear` | INT8 per-token（prefill）/ per-tensor（decode） |
| [W8A8 FP8 动态量化](linear_layer_quantization/term_w8a8_fp8_dynamic.md) | `WFP8AFP8DynamicPerChannelFakeQuantLinear` | FP8(E4M3) per-channel / per-token 动态 |
| [W8A8 MX 动态量化](linear_layer_quantization/term_w8a8_mx_dynamic.md) | `W8A8MXDynamicPerBlockFakeQuantLinear` | MXFP8 per-block / per-block 动态 |
| [W4A8 MX 动态量化](linear_layer_quantization/term_w4a8_mx_dynamic.md) | `W4A8MXDynamicPerBlockFakeQuantLinear` | MXFP4 / MXFP8 per-block 动态 |
| [W4A4 MX 动态量化](linear_layer_quantization/term_w4a4_mx_dynamic.md) | `W4A4MXDynamicPerBlockFakeQuantLinear` | MXFP4 per-block / per-block 动态 |
| [W4A4 MX 双 Scale 量化](linear_layer_quantization/term_w4a4_mx_dualscale.md) | `W4A4MXDynamicDualScaleFakeQuantLinear` | MXFP4 双 scale / per-block 动态 |
| [W8A8 稀疏量化](linear_layer_quantization/term_w8a8_sparse.md) | `W8A8StaticFakeQuantLinear`（导出类型 `W8A8S`） | INT8 只用码本子集；`W8A8SC` 为二次压缩形态 |
| [浮点稀疏量化](linear_layer_quantization/term_float_sparse.md) | `W16A16sLinear` | 16bit 权重/激活，权重置零稀疏；`W16A16SC` 为二次压缩形态 |
| [SVDQuant 量化](linear_layer_quantization/term_svdquant.md) | `SVDResidualWrapper`（[`svd_residual.py`](../../../../msmodelslim/ir/svd_residual.py)，配合 linear_quant） | 低秩分解 + 残差低比特量化 |

> 注：prefill 与 decode 是 LLM 推理的两个阶段——prefill 一次处理整个输入 prompt（计算密集），decode 逐个生成 token（访存密集），详见[线性层量化](linear_layer_quantization/README.md)。

### 3.2 KVCache 量化

| 模式 | 承载 IR 类 | 量化参数 |
|------|-----------|----------|
| [KVCache-PerChannel 量化](kv_cache_quantization/term_kv_cache_perchannel.md) | `FakeQuantDynamicCache` | INT8 per-channel（对称/非对称） |

### 3.3 FA 量化（KVCache 量化的进阶）

FA 量化**本质上是一个组合量化模式**，作用于注意力 **Q/K/V 三分支**，各分支选择一种激活值量化模式，三分支的组合构成**整体方案**（见 [FA 量化](fa_quantization/README.md)）。下表为已实践验证的整体方案，每个方案对应一个词条。

| 模式 | 承载 IR 类 | Q 分支 | K 分支 | V 分支 | 参数获取 |
|------|-----------|-----------|----------|----------|---------|
| [FA INT8 PerHead 量化](fa_quantization/term_fa_int8_perhead.md) | `INT8FakeQuantActivationPerHead` | INT8 per-head | INT8 per-head | INT8 per-head | 静态 |
| [FA INT8 动态量化](fa_quantization/term_fa_int8_dynamic.md) | `FakeQuantActivationPerToken` | INT8 per-token | INT8 per-token | INT8 per-token | 动态 |
| [FA FP8 动态量化](fa_quantization/term_fa_fp8_dynamic.md) | `FakeQuantActivationPerToken` | FP8 per-token | FP8 per-token | FP8 per-token | 动态 |
| [FA MXFP4 动态量化](fa_quantization/term_fa_mxfp4_dynamic.md) | `FakeQuantActivationPerBlock` | MXFP4 per-block | MXFP4 per-block | MXFP4 per-block | 动态 |
| [FA Q-INT8 动态 K/V-INT8 静态量化](fa_quantization/term_fa_q_int8_dynamic_kv_int8.md) | Q：`FakeQuantActivationPerToken`；K/V：`INT8FakeQuantActivationPerHead` | INT8 per-token | INT8 per-head | INT8 per-head | Q 动态；K/V 静态 |
| [FA Q-FP8 动态 K/V-FP8 静态量化](fa_quantization/term_fa_q_fp8_dynamic_kv_fp8.md) | Q：`FakeQuantActivationPerToken`；K/V：`FP8FakeQuantActivationPerHead` | FP8 per-token | FP8 per-head | FP8 per-head | Q 动态；K/V 静态 |

三分支可采用同一种激活值量化模式，也可分别选用不同的激活值量化模式；未量化的分支保持原精度。

---

## 4. 关联流程

- 《[一键量化完整指南](../../user_guide/usage_quick_quantization.md)》：通过 `--quant_type` 或 YAML 配置选择并执行量化模式。
- 《[量化精度调优指南](../../user_guide/process_quantization_precision_tuning.md)》：量化模式导致的精度劣化可通过该流程逐层回退与调优。

---

<a id="related-terms"></a>

## 5. 关联词条

- [线性层量化](linear_layer_quantization/README.md)：下位概念，本词条下量化模式的主体类别。
- [KVCache 量化](kv_cache_quantization/README.md)：下位概念，针对 KVCache 的量化类别。
- [FA 量化](fa_quantization/README.md)：下位概念，KVCache 量化的进阶类别。
- [W8A8 静态量化](linear_layer_quantization/term_w8a8_static.md)：下位概念，线性层量化中"权重/激活静态"的典型模式。
- [线性量化算法](../quantization_algorithms/linear_quant/term_linear_quant.md)：配套术语，描述线性层量化模式的处理器实现。
- [KVCache Quant 缓存量化算法](../quantization_algorithms/kvcache_quant/term_kvcache_quant.md)：配套术语，描述 KVCache 量化算法。
- [FA3 Quant 注意力激活量化算法](../quantization_algorithms/fa3_quant/term_fa3_quant.md)：配套术语，描述 FA 量化算法。

---

## 6. 参考文档

1. Jacob B et al. Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference. CVPR 2018. https://arxiv.org/abs/1712.05877
2. Yao Z et al. ZeroQuant: Efficient and Affordable Post-Training Quantization for Large-Scale Transformers. NeurIPS 2022. https://arxiv.org/abs/2206.01861
3. Liu Z et al. KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache. ICML 2024. https://arxiv.org/abs/2402.02750
4. 《[线性量化参数配置流程指南](../quantization_algorithms/linear_quant/usage_linear_quant.md)》
5. 《[AscendV1 格式说明](../quantization_format/ascendv1/ascendv1_usage.md)》
