# FA INT8 PerHead 量化

> **词条类别**：量化数据格式（[FA 量化](README.md)）<br>
> **英文名称**：FA INT8 PerHead Quantization<br>
> **应用领域**：长上下文推理加速、低精度注意力计算<br>
> **承载 IR 类**：`INT8FakeQuantActivationPerHead`（[`msmodelslim/ir/activation_static.py`](../../../../../msmodelslim/ir/activation_static.py)）<br>

---

## 1. 概述

FA INT8 PerHead 量化是对送入 Flash Attention 的 Q/K/V 激活统一按注意力头（per-head）静态量化到 [INT8](../../quantization_basic/term_int8.md) 的整体组合模式，是 FA 量化中三分支统一 per-head 静态的典型模式。它在量化 K/V（[KVCache 量化](../kv_cache_quantization/README.md)，压缩缓存显存）的基础上进一步量化 Q，使注意力得分计算（Q 与 K 的点乘）与加权求和（softmax 后与 V 点乘）的参与张量均为低比特，从而在继承 KV 显存收益之外使能低精度矩阵运算。

---

## 2. 词条介绍

### 2.1 模式规格

| 维度 | 取值 | 说明 |
| --- | --- | --- |
| 量化对象 | 注意力 Q/K/V 激活 | 送入 Flash Attention 的激活张量（形状 (B, H, S, D)），`fa_q`/`fa_k`/`fa_v` 三分支均被量化；K/V 承接缓存压缩，Q 使能低精度得分计算 |
| 位宽 | 8bit | Q/K/V 均为 8bit |
| 数据类型 | INT8 | 对称整数格式 |
| 参数获取方式 | 静态 | scale 离线校准确定并固化，推理零在线开销；仅使用 scale，无 offset |
| 量化粒度 | per-head | 每个注意力头共享一个 scale，参数开销仅 head 数，适配各头独立的数值分布 |
| 对称性 | 对称 | 仅 scale，无 offset |

### 2.2 量化公式

量化与反量化采用标准线性映射。对称量化（仅 scale、无零点）：

$$q = \mathrm{round}(x / s), \qquad \hat{x} = q \cdot s, \qquad s = \frac{\max(|x|)}{2^{b-1}}$$

上式中，$x$ 表示待量化的张量，$q$ 表示该张量量化后的数值，$\hat{x}$ 表示该张量经反量化还原后的数值，$s$ 表示 scale，$b=8$ 表示位宽。该模式采用对称量化，即只有 scale、没有 offset。在 per-head 粒度下，张量按第 1 维（head）切分，每个 head 内的元素共享同一个 scale。

### 2.3 与其他模式的关系

- **与 [FA INT8 动态量化](term_fa_int8_dynamic.md)（同为 QKV 统一的 INT8 模式，参数获取方式与粒度不同）**
  - 本模式（per-head 静态）：优势是 scale 离线固化、推理零在线开销，参数开销仅 head 数；劣势是粒度较粗、需依赖校准数据，对逐 token 分布变化不敏感。
  - INT8 动态（per-token 动态）：优势是逐 token 在线算 scale、免校准，量化更贴合每个 token 的数值范围；劣势是每次前向多一次按 token 的 min/max 归约，开销与 token 数同量级。
- **与 [FA Q-INT8 动态 K/V-INT8 静态量化](term_fa_q_int8_dynamic_kv_int8.md)（同为 INT8 家族，分支量化方式不同）**
  - 本模式三分支统一 per-head 静态；混合模式将 Q 改为 per-token 动态，使 Q 分支免校准。
- **与 [KVCache-PerChannel 量化](../kv_cache_quantization/term_kv_cache_perchannel.md)（基础关系）**
  - 本模式以 K/V 量化为基础：继承其缓存显存收益，追加 Q 量化使能低精度矩阵运算。

### 2.4 适用场景与限制

#### 2.4.1 适用场景

- **长序列注意力加速**：注意力计算密集的长上下文解码场景，整数/低精度注意力降低带宽与计算开销。
- **与线性层量化组合**：可与 W8A8 等线性层量化叠加使用，量化对象不同、互不重叠，共同降低权重/激活与注意力计算的开销。
- **三路精度需求一致的基线**：适合作为 FA 量化首次尝试的基线。

#### 2.4.2 使用限制

- **静态依赖校准**：per-head scale 由校准集确定，分布偏差会影响注意力得分精度。
- **注意力精度敏感**：Q/K 的量化误差直接影响注意力得分，极端长序列下需验证。
- **数据类型选择**：per-head 静态常见为 INT8；需要更低比特时可换用 MX 动态模式。

---

## 3. 关联流程

- 《[一键量化完整指南](../../../user_guide/usage_quick_quantization.md)》：通过 YAML 配置选择本模式并执行量化。
- 《[量化精度调优指南](../../../user_guide/process_quantization_precision_tuning.md)》：本模式导致的精度劣化可通过该流程逐层回退与调优。

---

## 4. 关联词条

- [量化模式](../README.md)：上位概念，本词条属于[FA 量化](README.md)类别，是该类别下的一种具体模式。
- [FA INT8 动态量化](term_fa_int8_dynamic.md)：同类模式，per-token 动态的 INT8 统一模式。
- [FA Q-INT8 动态 K/V-INT8 静态量化](term_fa_q_int8_dynamic_kv_int8.md)：同类模式，Q 走动态、K/V 走静态的 INT8 混合模式。
- [KVCache 量化](../kv_cache_quantization/README.md)：基础类别，本模式在其基础上追加 Q 量化。
- [FA3 Quant 注意力激活量化算法](../../quantization_algorithms/fa3_quant/term_fa3_quant.md)：配套术语，本模式对应的算法处理器文档。

---

## 5. 参考文档

1. 《[FA3 Quant 参数配置流程指南](../../quantization_algorithms/fa3_quant/usage_fa3_quant.md)》
2. 《[量化模式](../README.md)》
