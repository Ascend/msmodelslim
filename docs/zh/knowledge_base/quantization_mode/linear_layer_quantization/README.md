# 线性层量化

<!-- waiver: G01 原因：本文件为量化模式目录下的类别文档，不适用 term_<english_name>.md 词条命名 -->

> **词条类别**：量化数据格式（[量化模式](../README.md)）<br>
> **英文名称**：Linear Layer Quantization<br>
> **应用领域**：大语言模型量化压缩、推理加速<br>
> **msModelSlim 实现**：[`msmodelslim/ir/w8a8_static.py`](../../../../../msmodelslim/ir/w8a8_static.py)（`*FakeQuantLinear` 系列）<br>

<a id="overview"></a>

## 1. 概述

线性层量化是对大语言模型中占比最高的<b>线性层（Linear / MatMul）的权重（W）与激活（A）</b>进行量化的一类量化模式，属于[量化模式](../README.md)中的主体类别。它量化的对象是线性层矩阵乘法两侧的张量：权重是静态参数、每个 token 都必须整体读取；激活是随输入实时变化的中间张量。对这两类张量的位宽组合，通常用 `WxAy` 记法表示，如 W8A8 表示权重与激活均为 8bit（INT8）。

为何聚焦线性层：每个 token 前向都要读取全部权重，权重位宽直接决定访存带宽瓶颈；激活位宽则决定整数 GEMM 能否落地。因此此类模式数量最多，是量化模式的主体。典型如 [W8A8 静态量化](term_w8a8_static.md)——对线性层权重与激活均做 INT8 静态量化。

---

<a id="linear-compute"></a>

## 2. 词条介绍

### 2.1 线性层如何运算与可量化对象

线性层（Linear / MatMul）是 Transformer 的主体计算——Q/K/V 投影、注意力输出投影、MLP 的上/门/下三层都是线性层。一次线性层计算就是一次矩阵乘法：

$$Y = X \cdot W + b$$

其中 $X$ 是输入激活（形状 $(batch, seq, in\_dim)$），$W$ 是权重（形状 $(in\_dim, out\_dim)$），$b$ 是偏置，$Y$ 是输出激活。LLM 推理分两个阶段：**prefill** 一次处理整个输入 prompt（计算密集），**decode** 逐个生成 token（访存密集）。

矩阵乘法只有两个输入张量，因此线性层中可量化对象也只有两个：

- **权重 $W$**：静态张量，训练后固定、每步推理都要整体读取一次。decode 阶段权重访存是主要吞吐瓶颈，因此权重量化侧重**压缩**——压低位宽以减小访存，如 [W8A8 静态量化](term_w8a8_static.md) 的 INT8 权重、[W4A8 动态量化](term_w4a8_dynamic.md) 的 INT4 权重。
- **输入激活 $X$**：动态张量，由前一层实时产生、分布随输入变化，对量化误差敏感。激活量化侧重保精度，因此通常优选动态量化等细粒度的量化方式，如 [W8A8 动态量化](term_w8a8_dynamic.md) 的激活 per-token 动态量化。

量化模式本质上是**对这两个张量分别应用某种量化（位宽、数据类型、参数获取方式、量化粒度、对称性）后的组合**：如 [W8A8 静态量化](term_w8a8_static.md) 对 W 与 A 均做 8bit INT8 静态量化、[W8A16 静态量化](term_w8a16_static.md) 只量化 W 而 A 保持 16bit、[W4A8 动态量化](term_w4a8_dynamic.md) 则 W 4bit、A 8bit 且激活走动态。两个张量各自的维度选择，就构成下述"该类一般的量化选择"。

---

<a id="mode-options"></a>

### 2.2 该类一般的量化选择

同一类线性层结构可实施不同的量化方式，差异体现在以下几个维度：

- **位宽组合**：权重与激活可取相同或不同位宽。相同位宽（W8A8、W4A4）压缩与精度较均衡；激活位宽高于权重（W8A16、W4A8）牺牲部分压缩换取精度。见 [W8A16 静态量化](term_w8a16_static.md)、[W4A8 动态量化](term_w4a8_dynamic.md)。
- **参数获取方式**：静态（离线校准、推理时固化，如 [W8A8 静态量化](term_w8a8_static.md)）或动态（推理在线统计、免校准，如 [W8A8 动态量化](term_w8a8_dynamic.md)、[W4A4 动态量化](term_w4a4_dynamic.md)）。
- **量化粒度**：权重通常按通道（per-channel）或分组（per-group）静态量化；激活通常按 token（per-token）动态量化或按整个张量（per-tensor）静态量化；MX 系列采用按块（per-block）块级共享指数。
- **数据类型**：INT 整数（INT8/INT4）、FP8 浮点（如 [W8A8 FP8 动态量化](term_w8a8_fp8_dynamic.md)）、MX 块级共享指数（如 [W4A4 MX 动态量化](term_w4a4_mx_dynamic.md)、[W4A4 MX 双 Scale 量化](term_w4a4_mx_dualscale.md)）。
- **特殊变体**：[W8A8 PD-Mix 量化](term_w8a8_pdmix.md) 按 Prefill/Decode 阶段切换激活策略；[W8A8 稀疏量化](term_w8a8_sparse.md) 不铺满 INT8 的 256 个档位，再经二次压缩提高压缩率；[浮点稀疏量化](term_float_sparse.md) 不降位宽、对浮点权重做置零稀疏。

各类模式在**精度损失**（相对浮点基线的劣化程度）与**性能**（权重/激活访存与计算收益）之间取舍。精度损失分 **低 / 中 / 高** 三档，不直接比较精度高低。

| 模式 | 精度损失 | 性能（定性） | 取舍要点 |
|------|---------|------------|---------|
| [W8A8 动态量化](term_w8a8_dynamic.md) | 低 | 高 | 激活 per-token 动态，8bit 方案里损失较小 |
| [W8A8 FP8 动态量化](term_w8a8_fp8_dynamic.md) | 低 | 高 | 与 W8A8 动态同档，依赖 FP8 算子 |
| [W8A8 PD-Mix 量化](term_w8a8_pdmix.md) | 低 | 高 | Prefill per-token、Decode 静态，整体与 W8A8 动态同档 |
| [W8A8 静态量化](term_w8a8_static.md) | 低 | 高 | 满量程 INT8 静态，损失仍属低档 |
| [W8A8 MX 动态量化](term_w8a8_mx_dynamic.md) | 低 | 高 | 8bit MX per-block，与其他 W8A8 同档 |
| [W8A16 静态量化](term_w8a16_static.md) | 低 | 中 | 激活不量化，损失低；无整数 GEMM，性能收益减半 |
| [浮点稀疏量化](term_float_sparse.md) | 中 | 中 | 不降位宽，但置零会引入中等精度损失 |
| [W8A8 稀疏量化](term_w8a8_sparse.md) | 中 | 高 | 只用 256 档中的子集，损失高于满量程 W8A8，换更高压缩率 |
| [W4A8 动态量化](term_w4a8_dynamic.md) | 中 | 高 | 权重 4bit、激活 8bit 动态，损失中等 |
| [W4A8 MX 动态量化](term_w4a8_mx_dynamic.md) | 中 | 高 | 与 W4A8 动态同档 |
| [W4A4 动态量化](term_w4a4_dynamic.md) | 高 | 最高 | 双 4bit，精度损失高，访存收益最大 |
| [W4A4 MX 动态量化](term_w4a4_mx_dynamic.md) | 高 | 最高 | W4A4 类，双 4bit MX |
| [W4A4 MX 双 Scale 量化](term_w4a4_mx_dualscale.md) | 高 | 高 | 仍属 W4A4 类；双层 scale 只减轻损失，不改变档位 |

> 注：表中"性能"特指量化后的权重/激活访存与计算收益；实际损失与收益取决于目标硬件算子支持与模型分布，详见各模式词条与[量化模式](../README.md)模式索引。

---

<a id="modes"></a>

## 3. 该类下的模式清单

以下为 msModelSlim 支持的线性层量化模式清单（完整规格含承载 IR 类与量化参数，见[量化模式](../README.md)模式索引）：

| 模式 | 一句话 |
|------|--------|
| [W8A8 静态量化](term_w8a8_static.md) | 权重与激活均 INT8 静态 |
| [W8A8 动态量化](term_w8a8_dynamic.md) | 权重静态、激活 per-token 动态 |
| [W4A4 动态量化](term_w4a4_dynamic.md) | 权重与激活均 INT4、激活动态 |
| [W8A16 静态量化](term_w8a16_static.md) | 仅权重 INT8，激活 FP16 不量化 |
| [W4A8 动态量化](term_w4a8_dynamic.md) | 权重 INT4、激活 INT8 动态 |
| [W8A8 PD-Mix 量化](term_w8a8_pdmix.md) | 激活策略随 Prefill/Decode 阶段切换 |
| [W8A8 FP8 动态量化](term_w8a8_fp8_dynamic.md) | 权重与激活均 FP8(E4M3) 动态 |
| [W8A8 MX 动态量化](term_w8a8_mx_dynamic.md) | 权重与激活均 MXFP8 per-block 动态 |
| [W4A8 MX 动态量化](term_w4a8_mx_dynamic.md) | 权重 MXFP4、激活 MXFP8 per-block 动态 |
| [W4A4 MX 动态量化](term_w4a4_mx_dynamic.md) | 权重与激活均 MXFP4 per-block 动态 |
| [W4A4 MX 双 Scale 量化](term_w4a4_mx_dualscale.md) | MXFP4 双 scale、per-block 动态 |
| [W8A8 稀疏量化](term_w8a8_sparse.md) | INT8 只用码本子集，再经 `W8A8SC` 二次压缩，压缩率高于 W8A8 |
| [浮点稀疏量化](term_float_sparse.md) | 16bit 权重/激活，权重置零稀疏；`W16A16SC` 为二次压缩形态 |
| [SVDQuant 量化](term_svdquant.md) | 低秩分解 + 残差低比特量化，配合线性层量化使用 |

---

## 4. 关联流程

- 《[一键量化完整指南](../../../user_guide/usage_quick_quantization.md)》：通过 `--quant_type` 或 YAML 配置选择线性层量化模式。
- 《[量化精度调优指南](../../../user_guide/process_quantization_precision_tuning.md)》：量化模式导致的精度劣化可通过该流程逐层回退与调优。

---

<a id="related-terms"></a>

## 5. 关联词条

- [量化模式](../README.md)：上位概念，本词条是"线性层量化"类别。
- [KVCache 量化](../kv_cache_quantization/README.md)：同位概念，作用于注意力 K/V 缓存的量化类别，可与本类叠加。
- [FA 量化](../fa_quantization/README.md)：同位概念，KVCache 量化的进阶。
- [W8A8 静态量化](term_w8a8_static.md)：下位概念，本类别下最基础的静态模式。
- [W8A8 稀疏量化](term_w8a8_sparse.md)：下位概念，INT8 收窄码本后再二次压缩。
- [浮点稀疏量化](term_float_sparse.md)：下位概念，不降位宽的置零稀疏基线。
- [线性量化算法](../../quantization_algorithms/linear_quant/term_linear_quant.md)：配套术语，描述线性层量化模式的处理器实现。

---

## 6. 参考文档

1. 《[线性量化参数配置流程指南](../../quantization_algorithms/linear_quant/usage_linear_quant.md)》
2. 《[量化模式](../README.md)》
