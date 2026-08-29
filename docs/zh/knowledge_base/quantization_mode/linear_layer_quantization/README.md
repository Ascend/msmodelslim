# 线性层量化 量化术语百科词条

<!-- waiver: G01 原因：本文件为量化模式目录下的类别文档，不适用 term_<english_name>.md 词条命名 -->
<!-- waiver: S01 原因：索引/类别文档采用目录结构（术语列表 / 模式清单），不适用词条模板的「2. 词条介绍」必填章节，依 01 清单 S03 组织自身结构 -->

> **词条类别**：量化数据格式（[量化模式](../README.md)）
> **英文名称**：Linear Layer Quantization
> **应用领域**：大语言模型量化压缩、推理加速
> **msModelSlim 实现**：[`msmodelslim/ir/w8a8_static.py`](../../../../../msmodelslim/ir/w8a8_static.py)（`*FakeQuantLinear` 系列）

---

<a id="overview"></a>

## 概述

线性层量化是对大语言模型中占比最高的**线性层（Linear / MatMul）的权重（W）与激活（A）**进行量化的一类量化模式，属于[量化模式](../README.md)中的主体类别。它量化的对象是线性层矩阵乘法两侧的张量：权重是静态参数、每个 token 都必须整体读取；激活是随输入实时变化的中间张量。对这两类张量的位宽组合，通常用 `WxAy` 记法表示，如 W8A8 表示权重与激活均为 8bit（INT8）。

为何聚焦线性层：每个 token 前向都要读取全部权重，权重位宽直接决定访存带宽瓶颈；激活位宽则决定整数 GEMM 能否落地。因此此类模式数量最多，是量化模式的主体。典型如 [W8A8 静态量化](term_w8a8_static.md)——对 Linear 层权重与激活均做 INT8 静态量化。

---

<a id="linear-compute"></a>

## 线性层如何运算与可量化对象

线性层（Linear / MatMul）是 Transformer 的主体计算——Q/K/V 投影、注意力输出投影、MLP 的上/门/下三层都是线性层。一次线性层计算就是一次矩阵乘法：

$$Y = X \cdot W + b$$

其中 $X$ 是输入激活（形状 $(batch, seq, in\_dim)$），$W$ 是权重（形状 $(in\_dim, out\_dim)$），$b$ 是偏置，$Y$ 是输出激活。LLM 推理分两个阶段：**prefill** 一次处理整个输入 prompt（计算密集），**decode** 逐个生成 token（访存密集）。

矩阵乘法只有两个输入张量，因此线性层中可量化对象也只有两个：

- **权重 $W$**：静态张量，训练后固定、每步推理都要整体读取一次。decode 阶段权重访存是主要吞吐瓶颈，因此权重量化侧重**压缩**——压低位宽以减小访存，如 [W8A8 静态量化](term_w8a8_static.md) 的 INT8 权重、[W4A8 动态量化](term_w4a8_dynamic.md) 的 INT4 权重。
- **输入激活 $X$**：动态张量，由前一层实时产生、分布随输入变化，对量化误差敏感。激活量化侧重**保精度**——多配合动态量化或细粒度（如 per-token），如 [W8A8 动态量化](term_w8a8_dynamic.md) 的激活 per-token 动态量化。

量化模式本质上是**对这两个张量分别应用某种量化（位宽、数据类型、参数获取方式、量化粒度、对称性）后的组合**：如 [W8A8 静态量化](term_w8a8_static.md) 对 W 与 A 均做 8bit INT8 静态量化、[W8A16 静态量化](term_w8a16_static.md) 只量化 W 而 A 保持 16bit、[W4A8 动态量化](term_w4a8_dynamic.md) 则 W 4bit、A 8bit 且激活走动态。两个张量各自的维度选择，就构成下述"该类一般的量化选择"。

---

<a id="mode-options"></a>

## 该类一般的量化选择

同一类线性层结构可实施不同的量化方式，差异体现在以下几个维度：

- **位宽组合**：权重与激活可取相同或不同位宽。相同位宽（W8A8、W4A4）压缩与精度较均衡；激活位宽高于权重（W8A16、W4A8）牺牲部分压缩换取精度。见 [W8A16 静态量化](term_w8a16_static.md)、[W4A8 动态量化](term_w4a8_dynamic.md)。
- **参数获取方式**：静态（离线校准、推理时固化，如 [W8A8 静态量化](term_w8a8_static.md)）或动态（推理在线统计、免校准，如 [W8A8 动态量化](term_w8a8_dynamic.md)、[W4A4 动态量化](term_w4a4_dynamic.md)）。
- **量化粒度**：权重通常按通道（per-channel）或分组（per-group）静态量化；激活通常按 token（per-token）动态量化或按整个张量（per-tensor）静态量化；MX 系列采用按块（per-block）块级共享指数。
- **数据类型**：INT 整数（INT8/INT4）、FP8 浮点（如 [W8A8 FP8 动态量化](term_w8a8_fp8_dynamic.md)）、MX 块级共享指数（如 [W4A4 MX 动态量化](term_w4a4_mx_dynamic.md)、[W4A4 MX 双 Scale 量化](term_w4a4_mx_dualscale.md)）。
- **特殊变体**：[W8A8 PD-Mix 量化](term_w8a8_pdmix.md) 按 Prefill/Decode 阶段切换激活策略；[W16A16S 量化](term_w16a16s.md) 不降位宽、承载上游稀疏权重，作为稀疏量化基线。

各类模式在**精度**（量化误差对模型输出的影响）与**性能**（权重/激活访存与计算收益）之间取舍，可定性对比如下：

| 模式 | 精度（定性） | 性能（定性） | 取舍要点 |
|------|------------|------------|---------|
| [W8A8 静态量化](term_w8a8_static.md) | 高 | 高 | INT8 主流方案，需离线校准，静态 scale 对分布变化略敏感 |
| [W8A8 动态量化](term_w8a8_dynamic.md) | 更高 | 高 | 激活 per-token 在线统计 scale，免校准，但增加少量在线开销 |
| [W4A4 动态量化](term_w4a4_dynamic.md) | 中 | 最高 | 双 4bit 访存收益最大，激活 4bit 精度风险高 |
| [W8A16 静态量化](term_w8a16_static.md) | 最高 | 中 | 仅压缩权重，激活保 16bit 精度，性能收益减半 |
| [W4A8 动态量化](term_w4a8_dynamic.md) | 中高 | 高 | 权重 4bit 极致压缩，激活 8bit 保精度，压缩与精度的折中 |
| [W8A8 PD-Mix 量化](term_w8a8_pdmix.md) | 高 | 高 | 激活策略随 prefill/decode 切换，兼顾两阶段特性 |
| [W8A8 FP8 动态量化](term_w8a8_fp8_dynamic.md) | 高 | 高 | FP8 保留浮点动态范围、对离群值耐受，依赖 FP8 算子 |
| [W8A8 MX 动态量化](term_w8a8_mx_dynamic.md) | 中高 | 高 | MXFP8 块级共享指数，缩放参数开销低，依赖 MX 硬件 |
| [W4A8 MX 动态量化](term_w4a8_mx_dynamic.md) | 中高 | 高 | MXFP4 权重 + MXFP8 激活，访存收益与精度折中 |
| [W4A4 MX 动态量化](term_w4a4_mx_dynamic.md) | 中 | 最高 | 双 4bit MX 压缩最大化，依赖 MX 硬件、需块对齐 |
| [W4A4 MX 双 Scale 量化](term_w4a4_mx_dualscale.md) | 中高 | 高 | 双层 scale 兜底 4bit 精度，实现与配置更复杂 |
| [W16A16S 量化](term_w16a16s.md) | 最高 | 中 | 不降位宽，承载上游稀疏权重，作为稀疏基线 |

> 注：表中"性能"特指量化后的权重/激活访存与计算收益，均为定性排序；实际收益取决于目标硬件算子支持与模型分布，详见各模式词条与[量化模式](../README.md)「模式索引」。

---

<a id="modes"></a>

## 该类下的模式清单

以下为 msModelSlim 支持的线性层量化模式清单（完整规格含承载 IR 类与量化参数，见[量化模式](../README.md)「模式索引」）：

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
| [W16A16S 量化](term_w16a16s.md) | 16bit 权重/激活（含稀疏） |
| [SVDQuant 量化](term_svdquant.md) | 低秩分解 + 残差低比特量化，配合线性层量化使用 |

---

## 关联流程

- 《[一键量化完整指南](../../../user_guide/usage_quick_quantization.md)》：通过 `--quant_type` 或 YAML 配置选择线性层量化模式。
- 《[量化精度调优指南](../../../user_guide/process_quantization_precision_tuning.md)》：量化模式导致的精度劣化可通过该流程逐层回退与调优。

---

<a id="related-terms"></a>

## 关联词条

- [量化模式](../README.md)：上位概念，本词条是"线性层量化"类别。
- [KVCache 量化](../kv_cache_quantization/README.md)：同位概念，作用于注意力 K/V 缓存的量化类别，可与本类叠加。
- [FA 量化](../fa_quantization/README.md)：同位概念，KVCache 量化的进阶。
- [W8A8 静态量化](term_w8a8_static.md)：下位概念，本类别下最基础的静态模式。
- 《[线性量化算法说明](../../quantization_algorithms/linear_quant/term_linear_quant.md)》：配套术语，描述线性层量化模式的处理器实现。

---

## 参考资料

1. 《[线性量化算法说明](../../quantization_algorithms/linear_quant/term_linear_quant.md)》
2. 《[量化模式](../README.md)》
