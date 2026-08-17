# FA PerHead 量化

> **词条类别**：量化数据格式（[FA 量化](README.md)）
> **英文名称**：FA Per-Head Quantization
> **应用领域**：长上下文推理加速、低精度注意力计算
> **承载 IR 类**：`FakeQuantActivationPerHead` 及 INT8/FP8 变体（[`msmodelslim/ir/activation_static.py`](../../../../../msmodelslim/ir/activation_static.py)）

---

## 1. 概述

FA PerHead 量化 = 对送入 Flash Attention 的 Q/K/V 激活按注意力头（per-head）做静态量化。它是 [KVCache 量化](../kv_cache_quantization/README.md)的**进阶**：在量化 K/V（压缩缓存显存）的基础上进一步量化 Q，使注意力得分计算（Q 与 K 的点乘）与加权求和（softmax 后与 V 点乘）的参与张量均为低比特，从而在继承 KV 显存收益之外使能低精度矩阵运算。模式名中「PerHead」表示量化粒度为按注意力头。

---

## 2. 词条介绍

### 模式规格

| 维度 | 取值 | 说明 |
|------|------|------|
| 量化对象 | 注意力 Q/K/V 激活 | 送入 Flash Attention 的激活张量（形状 (B, H, S, D)），Q/K/V 均被量化；K/V 承接缓存压缩，Q 使能低精度得分计算 |
| 位宽 | 8bit | Q/K/V 均为 8bit |
| 数据类型 | INT8 / FP8（E4M3） | 两种实现按精度/吞吐需求选择 |
| 参数获取方式 | 静态 | scale 离线校准确定并固化，推理零在线开销；要求提供 `ext['scale']`，忽略 offset |
| 量化粒度 | per-head | 每个注意力头共享一个 scale，参数开销仅 head 数，适配各头独立的数值分布 |
| 对称性 | 对称 | 仅 scale，无 offset |

### 量化公式

量化与反量化采用标准线性映射。对称量化（仅 scale、无零点）：
$$q = \mathrm{round}(x / s), \qquad \hat{x} = q \cdot s, \qquad s = \frac{\max(|x|)}{2^{b-1}}$$

非对称量化（含零点 offset）：
$$q = \mathrm{round}(x / s) + z, \qquad \hat{x} = (q - z) \cdot s, \qquad s = \frac{\max(x) - \min(x)}{2^b - 1}$$

其中 $x$ 为待量化张量，$q$ 为量化后的数值，$\hat{x}$ 为反量化还原值，$s$ 为 scale，$z$ 为零点 offset，$b$ 为位宽，且 $z = \mathrm{round}(-\min(x)/s)$。本模式为对称量化（无 offset）。FP8（E4M3）变体的 scale 按 FP8 可表示最大值 448 确定。量化对象为送入 Flash Attention 的 Q/K/V 激活，参数获取方式与作用粒度（per-head / per-token）见「模式规格」表。

### 与其他模式的关系

- **与 [FA PerToken 量化](term_fa_pertoken.md)（同为 FA 激活量化，粒度与参数获取方式不同）**
  - 本模式（per-head 静态）：优势是 scale 离线固化、推理零在线开销，参数开销仅 head 数；劣势是粒度较粗、需依赖校准数据，对逐 token 分布变化不敏感。
  - PerToken（per-token 动态）：优势是逐 token 在线算 scale、免校准，量化更贴合每个 token 的数值范围；劣势是每次前向多一次按 token 的 min/max 归约，开销与 token 数同量级。

- **与 [FA PerBlock 量化](term_fa_perblock.md)（同为 FA 激活量化，格式与粒度不同）**
  - 本模式：INT8/FP8 数据格式 + 静态 per-head 粒度，无需特殊硬件格式支持；劣势是粒度最粗（整头共享参数），对 head_dim 内部的分布差异不敏感。
  - PerBlock：MX 格式块级共享指数（32元素一块），粒度最细、对离群值比整数格式更耐受；劣势是依赖硬件 MX 支持、head_dim 需能被32整除、量化参数在线计算。

- **与 [KVCache-PerChannel 量化](../kv_cache_quantization/term_kv_cache_perchannel.md)（基础关系）**
  - 本模式以 K/V 量化为基础：继承其缓存显存减半的收益，追加 Q 量化使能低精度矩阵运算。K/V 不量化则 Q 量化无法单独兑现整数/低精度注意力得分计算的收益。

### 适用场景与限制

#### 1. 适用场景

- **长序列注意力加速**：注意力计算密集的长上下文解码场景，整数/低精度注意力降低带宽与计算开销。
- **与线性层量化组合**：作为整体量化方案的一部分，与 W8A8 等线性层模式叠加进一步压缩。

#### 2. 使用限制

- **静态依赖校准**：per-head scale 由校准集确定，分布偏差会影响注意力得分精度。
- **数据类型有限**：msModelSlim 中 per-head 量化仅提供 INT8 与 FP8 两种数据类型。
- **注意力精度敏感**：Q/K 的量化误差直接影响注意力得分，极端长序列下需验证。

---

## 3. 关联流程

- 《[一键量化完整指南](../../../user_guide/usage_quick_quantization.md)》：通过 YAML 配置选择本模式并执行量化。
- 《[量化精度调优指南](../../../user_guide/process_quantization_precision_tuning.md)》：本模式导致的精度劣化可通过该流程逐层回退与调优。

---

## 4. 关联词条

- [量化模式](../README.md)：上位概念，本词条属于[FA 量化](README.md)类别，是该类别下的一种具体模式。
- [FA PerToken 量化](term_fa_pertoken.md)：同类模式，per-token 动态激活量化。
- [FA PerBlock 量化](term_fa_perblock.md)：同类模式，per-block MX 动态激活量化。
- [KVCache 量化](../kv_cache_quantization/README.md)：基础类别，本模式在其基础上追加 Q 量化。
- [KVCache-PerChannel 量化](../kv_cache_quantization/term_kv_cache_perchannel.md)：前置术语，K/V 量化的具体模式。
- [W8A8 静态量化](../linear_layer_quantization/term_w8a8_static.md)：配套模式，可与本模式组合使用。
- 《[FA3量化：Flash Attention 3激活量化算法说明](../../quantization_algorithms/fa3_quant/usage_fa3_quant.md)》：配套术语，本模式对应的算法处理器文档。

---

## 5. 参考文档

1. 《[FA3量化：Flash Attention 3激活量化算法说明](../../quantization_algorithms/fa3_quant/usage_fa3_quant.md)》
2. 《[量化模式](../README.md)》
