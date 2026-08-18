# 量化模式 量化术语百科词条

<!-- waiver: G01 原因：本文件为量化模式目录的索引文档，不适用 term_<english_name>.md 词条命名 -->
<!-- waiver: S01 原因：索引/类别文档采用目录结构（术语列表 / 模式清单），不适用词条模板的「2. 词条介绍」必填章节，依 01 清单 S03 组织自身结构 -->

> **英文名称**：Quantization Mode
> **应用领域**：大语言模型量化压缩、推理加速、KVCache 压缩

---

<a id="overview"></a>

## 概述

阅读本词条前，建议先了解[量化基础](../quantization_basic/README.md)中的量化/反量化、scale 与常用数据类型等基本概念。

量化模式（Quantization Mode）是 msModelSlim 中用于描述**针对某一类特定的模型结构，如何实施一种特定的量化方式**的基础概念。量化操作的基本对象是**张量**——权重、激活、缓存的 K/V 等都是张量；量化模式即针对某个或某几个张量分别应用某种量化（位宽、数据类型、参数获取方式、量化粒度、对称性）后形成的组合。提及某个量化模式时，它通常同时指出：作用于哪类结构、量化其中的哪些张量、以什么位宽和参数获取方式量化。以 W8A8 静态量化为例——它就是**对 Linear 层进行量化**：权重与激活均为 8bit（INT8），且权重与激活均采用**静态量化**（[W8A8 静态量化](linear_layer_quantization/term_w8a8_static.md)）。

按量化对象，常用量化模式可归为三类：**线性层量化**作用于矩阵乘法的权重与激活；**KVCache 量化**作用于注意力机制的 K/V 缓存；**FA 量化**是 KVCache 量化的进阶——在量化 K/V 的基础上进一步量化 Q，使能低精度注意力矩阵运算。三者分别见 [线性层量化](linear_layer_quantization/README.md)、[KVCache 量化](kv_cache_quantization/README.md)、[FA 量化](fa_quantization/README.md)。

---

<a id="how-to-read"></a>

## 如何理解一个量化模式

一个量化模式名通常由"作用结构 + 张量位宽 + 参数获取方式 + 量化粒度 + 数据类型"几部分构成，掌握了这些要素，就能读懂任意一个模式名：

- **作用结构**：线性层（Linear/MatMul）用 `WxAy` 记法表示权重与激活的位宽组合（`y` 为可选后缀，表示数据类型与参数获取方式，如 W8A8 静态、W4A4 MX 动态），如 W8A8、W4A4；KVCache 作用于缓存的 K/V 张量；FA 量化作用于注意力输入 Q。
- **张量位宽**：W 与 A 分别表示权重与激活的位宽。位宽越低，访存与计算收益越大，但数值分辨率越低、精度损失风险越高。
- **参数获取方式**：静态（离线校准得到 scale/offset，推理时固化，如 [W8A8 静态量化](linear_layer_quantization/term_w8a8_static.md)）或动态（推理时在线统计并计算，如 [W8A8 动态量化](linear_layer_quantization/term_w8a8_dynamic.md)、[W4A4 动态量化](linear_layer_quantization/term_w4a4_dynamic.md)）。
- **量化粒度**：一组元素共享一个量化参数的范围。常用粒度有 per-tensor、per-channel、per-group、per-token、per-head、per-block 等，粗细取决于**每份量化参数覆盖的元素数**——覆盖越少（块越小）越细，数值分布刻画越准、精度越好，但量化参数数量与计算开销越大（如 per-token 见 [W8A8 动态量化](linear_layer_quantization/term_w8a8_dynamic.md)，per-block 见 [W8A8 MX 动态量化](linear_layer_quantization/term_w8a8_mx_dynamic.md)）。
- **数据类型**：量化后的数值格式，如 INT 整数（INT8/INT4）、FP8 浮点（E4M3）、带块级共享指数的 MXFP8/MXFP4。

---

<a id="categories"></a>

## 量化模式的分类

按量化对象，常用量化模式分为三类：

- **[线性层量化](linear_layer_quantization/README.md)**：对大语言模型中占比最高的线性层（Linear / MatMul）权重与激活进行量化，是量化模式的主体、模式数量最多，如 [W8A8 静态量化](linear_layer_quantization/term_w8a8_static.md)、[W4A4 动态量化](linear_layer_quantization/term_w4a4_dynamic.md)。
- **[KVCache 量化](kv_cache_quantization/README.md)**：对注意力机制的 KVCache 进行量化。KVCache 本质上是一种特殊的激活——历史 token 的 Key/Value 投影被缓存下来供后续 token 复用，其显存随序列长度线性增长；量化 K/V 使缓存显存减半，如 [KVCache-PerChannel 量化](kv_cache_quantization/term_kv_cache_perchannel.md)。
- **[FA 量化](fa_quantization/README.md)**：KVCache 量化的进阶——在量化 K/V 的基础上进一步量化送入 Flash Attention 的 Q，使能低精度注意力矩阵运算，如 [FA PerHead 量化](fa_quantization/term_fa_perhead.md)。

三类量化作用于计算图的不同位置，可以独立或组合使用，共同构成推理加速的优化空间；其中 FA 量化以 KVCache 量化为基础，是其在计算层面的延伸。

---

<a id="modes"></a>

## 模式索引

以下为 msModelSlim 支持的量化模式全量清单（按量化对象分类），也是各具体量化模式词条的总览入口。表中"承载 IR 类"为 `msmodelslim/ir/` 中的类名，类名里的 **FakeQuant**（伪量化）指量化后立即反量化的占位层，用于在浮点计算中模拟真实量化的数值效果：

### 线性层量化

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
| [W16A16S 量化](linear_layer_quantization/term_w16a16s.md) | `W16A16sLinear` | 16bit 权重/激活（含稀疏） |
| [SVDQuant](../quantization_algorithms/svdquant/usage_svdquant.md) | `SVDResidualWrapper`（[`svd_residual.py`](../../../../msmodelslim/ir/svd_residual.py)，配合 linear_quant） | 低秩分解 + 残差低比特量化 |

> 注：prefill 与 decode 是 LLM 推理的两个阶段——prefill 一次处理整个输入 prompt（计算密集），decode 逐个生成 token（访存密集），详见[线性层量化](linear_layer_quantization/README.md)。

### KVCache 量化

| 模式 | 承载 IR 类 | 量化参数 |
|------|-----------|----------|
| [KVCache-PerChannel 量化](kv_cache_quantization/term_kv_cache_perchannel.md) | `FakeQuantDynamicCache` | INT8 per-channel（对称/非对称） |

### FA 量化（KVCache 量化的进阶）

| 模式 | 承载 IR 类 | 量化参数 |
|------|-----------|----------|
| [FA PerHead 量化](fa_quantization/term_fa_perhead.md) | INT8：`INT8FakeQuantActivationPerHead`；FP8：`FP8FakeQuantActivationPerHead` | INT8/FP8 per-head 静态 |
| [FA PerToken 量化](fa_quantization/term_fa_pertoken.md) | `FakeQuantActivationPerToken` | INT8/FP8 per-token 动态 |
| [FA PerBlock 量化](fa_quantization/term_fa_perblock.md) | `FakeQuantActivationPerBlock` | MXFP8/MXFP4 per-block 动态 |

FA 量化**本质上是一个组合量化模式**，实际作用于注意力 **Q/K/V 三分支**（`fa_q` / `fa_k` / `fa_v`），三分支各选一个量化模式组合成整体方案。当前最佳实践实际应用过的组合（均通过 `fa3_quant.qconfig` 统一配置三分支）：

| Q/K/V 组合 | 数据类型 / 粒度 | 参数获取 | 配置来源 |
|-----------|----------------|---------|---------|
| Q/K/V 统一 [INT8 per-head](fa_quantization/term_fa_perhead.md) | INT8 per-head | 静态（minmax） | `fa3_quant` 默认配置 |
| Q/K/V 统一 [FP8（E4M3）per-token](fa_quantization/term_fa_pertoken.md) | FP8 per-token | 动态 | Wan2_2、HunYuanVideo 示例 |
| Q/K/V 统一 [MXFP4 per-block](fa_quantization/term_fa_perblock.md) | MXFP4 per-block | 动态 | QwenImageEdit 示例 |

框架亦支持通过 `fa3_quant.details`（`fa_q`/`fa_k`/`fa_v`）为各分支分别指定配置（未配置的分支不量化），当前示例均使用统一配置。

---

## 关联流程

- 《[一键量化完整指南](../../user_guide/usage_quick_quantization.md)》：通过 `--quant_type` 或 YAML 配置选择并执行量化模式。
- 《[量化精度调优指南](../../user_guide/process_quantization_precision_tuning.md)》：量化模式导致的精度劣化可通过该流程逐层回退与调优。

---

<a id="related-terms"></a>

## 关联词条

- [线性层量化](linear_layer_quantization/README.md)：下位概念，本词条下量化模式的主体类别。
- [KVCache 量化](kv_cache_quantization/README.md)：下位概念，针对 KVCache 的量化类别。
- [FA 量化](fa_quantization/README.md)：下位概念，KVCache 量化的进阶类别。
- [W8A8 静态量化](linear_layer_quantization/term_w8a8_static.md)：下位概念，线性层量化中"权重/激活静态"的典型模式。
- 《[线性量化算法说明](../quantization_algorithms/linear_quant/usage_linear_quant.md)》：配套术语，描述线性层量化模式的处理器实现。
- 《[KVCache量化：缓存量化算法说明](../quantization_algorithms/kvcache_quant/usage_kvcache_quant.md)》：配套术语，描述 KVCache 量化算法。
- 《[FA3量化：Flash Attention 3激活量化算法说明](../quantization_algorithms/fa3_quant/usage_fa3_quant.md)》：配套术语，描述 FA 量化算法。

---

## 参考资料

1. Jacob B et al. Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference. CVPR 2018. https://arxiv.org/abs/1712.05877
2. Yao Z et al. ZeroQuant: Efficient and Affordable Post-Training Quantization for Large-Scale Transformers. NeurIPS 2022. https://arxiv.org/abs/2206.01861
3. Liu Z et al. KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache. ICML 2024. https://arxiv.org/abs/2402.02750
4. 《[线性量化算法说明](../quantization_algorithms/linear_quant/usage_linear_quant.md)》
5. 《[AscendV1 格式说明](../quantization_format/ascendv1/ascendv1_usage.md)》
