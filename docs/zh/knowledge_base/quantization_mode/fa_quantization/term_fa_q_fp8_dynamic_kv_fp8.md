# FA Q-FP8 动态 K/V-FP8 静态量化

> **词条类别**：量化数据格式（[FA 量化](README.md)）<br>
> **英文名称**：FA Q-FP8 Dynamic K/V-FP8 Static Quantization<br>
> **应用领域**：长上下文推理加速、低精度注意力计算<br>
> **承载 IR 类**：`FakeQuantActivationPerToken`（Q，[`msmodelslim/ir/activation_dynamic.py`](../../../../../msmodelslim/ir/activation_dynamic.py)）、`FP8FakeQuantActivationPerHead`（K/V，[`msmodelslim/ir/activation_static.py`](../../../../../msmodelslim/ir/activation_static.py)）<br>

---

## 1. 概述

FA Q-FP8 动态 K/V-FP8 静态量化是混合整体模式：Q 分支按 token 动态量化、K/V 分支按注意力头静态量化到 [FP8（E4M3）](../../quantization_basic/term_fp8.md)（见 [FA 量化](README.md)）。Q 逐 token 在线算 scale、免校准，保证得分计算输入贴合实际分布；K/V 走 per-head 静态（承接缓存压缩，[KVCache 量化](../kv_cache_quantization/README.md)）。相比三分支统一 FP8 动态，K/V 静态化使缓存参数离线确定、降低在线开销；相比三分支统一 FP8 per-head 静态，Q 动态化免去对 Q 的校准依赖。

---

## 2. 词条介绍

### 2.1 模式规格

| 维度 | fa_q | fa_k | fa_v |
| --- | --- | --- | --- |
| 位宽 | 8bit | 8bit | 8bit |
| 数据类型 | FP8（E4M3） | FP8（E4M3） | FP8（E4M3） |
| 参数获取方式 | 动态（在线 min/max，免校准） | 静态（离线校准） | 静态（离线校准） |
| 量化粒度 | per-token（逐 token） | per-head（按注意力头） | per-head（按注意力头） |
| 对称性 | 对称 | 对称 | 对称 |

### 2.2 量化公式

Q 分支按 token 对称动态量化：

$$q = \mathrm{round}(x / s), \qquad \hat{x} = q \cdot s, \qquad s = \frac{\max(|x|)}{448}$$

K/V 分支按 head 对称静态量化（scale 离线校准固化，FP8 表示上限 448）：

$$q = \mathrm{round}(x / s_h), \qquad \hat{x} = q \cdot s_h, \qquad s_h = \frac{\max(|x_h|)}{448}$$

上式中，$x$ 表示待量化的张量，$q$ 表示该张量量化后的数值，$\hat{x}$ 表示该张量经反量化还原后的数值，$s$ 表示 Q 分支按 token 在线统计的动态 scale，$s_h$ 表示第 $h$ 个注意力头的静态 scale，$x_h$ 表示第 $h$ 个头所覆盖的激活子集。与 Q 分支的动态取值不同，K/V 分支的 scale 在离线校准阶段确定并固化。

### 2.3 设计动机与分工

- **Q 走 per-token 动态**：Q 是得分计算的直接输入，随输入分布变化敏感；动态免校准可避免 Q 分支对校准集的依赖，也更贴合每个 token 实际范围。
- **K/V 走 per-head 静态**：K/V 承接缓存压缩，per-head 静态使缓存参数离线确定、推理零在线开销，与 KVCache 量化的缓存语义一致；K/V 分支缺少校准数据则无法确定静态参数。

### 2.4 与其他模式的关系

- **与 [FA FP8 动态量化](term_fa_fp8_dynamic.md)（同为 FP8 家族，分支量化方式不同）**
  - 本模式将 K/V 分支从 per-token 动态改为 per-head 静态：K/V 缓存参数离线固化、降低在线开销，代价是 K/V 依赖校准数据。
- **与 [FA Q-INT8 动态 K/V-INT8 静态量化](term_fa_q_int8_dynamic_kv_int8.md)（同类混合结构，数据类型不同）**
  - 本模式为 FP8（E4M3）；Q-INT8 混合模式为 INT8。FP8 浮点动态范围更大、对离群值相对更耐受。
- **与 [KVCache-PerChannel 量化](../kv_cache_quantization/term_kv_cache_perchannel.md)（基础关系）**
  - 本模式以 K/V 量化为基础：K/V 的静态量化承接缓存压缩收益，Q 的动态量化使能低精度得分计算。

### 2.5 适用场景与限制

#### 2.5.1 适用场景

- **Q 分支校准数据不充分**：Q 的分布随输入变化大、难以用固定校准覆盖时，将 Q 单独放开为动态。
- **K/V 分支希望静态缓存参数**：需要 K/V 的缓存量化参数离线确定、避免 decode 阶段在线统计时。
- **FP8 长序列注意力**：期望 FP8 浮点动态范围收益、同时希望 K/V 缓存参数离线静态确定的场景。

#### 2.5.2 使用限制

- **静态分支依赖校准**：K/V 的 per-head scale 由校准集确定，K/V 缺少校准数据则无法确定静态参数。
- **FP8 表示范围有限**：E4M3 max 448，大离群值需留意。
- **注意力精度敏感**：Q/K 的量化误差直接影响注意力得分，需验证极端长序列下的精度。

---

## 3. 关联流程

- 《[一键量化完整指南](../../../user_guide/usage_quick_quantization.md)》：通过 YAML 配置选择本模式并执行量化。
- 《[量化精度调优指南](../../../user_guide/process_quantization_precision_tuning.md)》：本模式导致的精度劣化可通过该流程逐层回退与调优。

---

## 4. 关联词条

- [量化模式](../README.md)：上位概念，本词条属于[FA 量化](README.md)类别，是该类别下的一种具体模式。
- [FA FP8 动态量化](term_fa_fp8_dynamic.md)：同类模式，三分支统一 per-token 动态的 FP8 模式。
- [FA Q-INT8 动态 K/V-INT8 静态量化](term_fa_q_int8_dynamic_kv_int8.md)：同类混合结构模式，Q 动态、K/V 静态但数据类型为 INT8。
- [KVCache 量化](../kv_cache_quantization/README.md)：基础类别，本模式在其基础上追加 Q 量化。
- [KVCache-PerChannel 量化](../kv_cache_quantization/term_kv_cache_perchannel.md)：前置术语，K/V 量化的具体模式。

---

## 5. 参考文档

1. 《[FA3 Quant 参数配置流程指南](../../quantization_algorithms/fa3_quant/usage_fa3_quant.md)》
2. 《[量化模式](../README.md)》
