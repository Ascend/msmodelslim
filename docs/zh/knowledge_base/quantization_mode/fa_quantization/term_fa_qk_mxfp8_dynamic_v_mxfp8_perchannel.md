# FA QK-MXFP8 动态 / V-MXFP8 PerChannel 静态量化

> **词条类别**：量化数据格式（[FA 量化](README.md)）<br>
> **英文名称**：FA QK-MXFP8 Dynamic V-MXFP8 PerChannel Static Quantization<br>
> **应用领域**：长上下文推理加速、低精度注意力计算<br>
> **承载 IR 类**：`FakeQuantActivationPerBlock`（Q/K，[`msmodelslim/ir/activation_dynamic.py`](../../../../../msmodelslim/ir/activation_dynamic.py)）、`MXFP8FakeQuantActivationPerChannel`（V，[`msmodelslim/ir/mxfp8_activation_static.py`](../../../../../msmodelslim/ir/mxfp8_activation_static.py)）<br>

---

## 1. 概述

FA QK-MXFP8 动态 / V-MXFP8 PerChannel 静态量化是混合整体模式：Q/K 分支沿 head_dim 按 32 元素分块、块级共享 E8M0 指数做 [MXFP8](../../quantization_basic/term_mxfp.md) 动态量化，V 分支沿 `(head, dim)` 通道共享 E8M0 指数做 MXFP8 per-channel 静态量化（见 [FA 量化](README.md)）。该模式以 MXFP8 承接 Q/K/V 三分支的量化：Q/K 动态免校准，V 静态按通道离线校准；K/V 承接缓存压缩（[KVCache 量化](../kv_cache_quantization/README.md)），Q 使能低精度注意力矩阵运算。

---

## 2. 词条介绍

### 2.1 模式规格

| 维度 | fa_q | fa_k | fa_v |
| --- | --- | --- | --- |
| 位宽 | 8bit | 8bit | 8bit |
| 数据类型 | MXFP8 | MXFP8 | MXFP8 |
| 参数获取方式 | 动态（在线统计，免校准） | 动态（在线统计，免校准） | 静态（离线校准） |
| 量化粒度 | per-block（沿 head_dim 按 32 元素分块） | per-block（沿 head_dim 按 32 元素分块） | per-channel（沿 `(head, dim)` 通道） |
| 对称性 | 对称 | 对称 | 对称 |

### 2.2 量化公式

Q/K 分支按块共享 E8M0 指数动态量化：

$$e = \lfloor \log_2(\max_{i \in \mathrm{block}}|x_i|) \rfloor, \qquad q_i = \mathrm{round}_{MX}\left(\frac{x_i}{2^e}\right), \qquad \hat{x}_i = q_i \cdot 2^e$$

V 分支按 `(head, dim)` 通道共享 E8M0 指数静态量化（scale 离线校准固化）：

$$e_c = \lfloor \log_2(\max_{i \in \mathrm{channel}}|x_i|) \rfloor, \qquad q_i = \mathrm{round}_{MX}\left(\frac{x_i}{2^{e_c}}\right), \qquad \hat{x}_i = q_i \cdot 2^{e_c}$$

上式中，$x_i$ 表示待量化的激活元素，$q_i$ 表示该元素量化后的 MX 尾数（MXFP8 中为 FP8 尾数），$\hat{x}_i$ 表示该元素经反量化还原后的数值，$e$ 表示 Q/K 分支每个 32 元素块内共享的 E8M0 指数，$e_c$ 表示 V 分支每个 `(head, dim)` 通道共享的 E8M0 指数。Q/K 与 V 的区别仅在于共享指数的统计范围：Q/K 以 32 个元素为一块计算块级指数，V 把 head 维并入通道维、以 `(head, dim)` 为通道计算通道级指数，每个通道覆盖该 `(head, dim)` 在全部序列位置上的元素。

### 2.3 设计动机与分工

- **Q/K 走 per-block 动态**：Q/K 参与得分计算、随输入分布变化，动态免校准贴合实际分布，避免对 Q/K 的校准依赖；per-block 沿 head_dim 捕捉通道内部分布差异。
- **V 走 per-channel 静态**：V 是三分支中承接缓存压缩的静态分支，per-channel 的核心动机是规避按 32 元素分块产生的尾块问题。MXFP8 块量化按 group size 32 沿 head_dim 分块，若 head_dim 不是 32 的倍数，每行末尾会留下不足 32 个元素的尾块。V 缓存随 decode 逐 token 追加、贯穿整个推理生命周期，对尾块反复 padding 或特判的代价高。因此 V 不再按 32 分组，而是将 head 并入通道维、以 `(head, dim)` 为通道跨整条序列共享 E8M0 指数，从根上消除尾块。V 的量化参数离线静态确定，在缓存生命周期内固定不变，也与 KVCache 量化的通道语义一致。

### 2.4 与其他模式的关系

- **与 [FA MXFP4 动态量化](term_fa_mxfp4_dynamic.md)（同为 MX 家族，位宽与分支量化方式不同）**
  - 本模式为 8bit MXFP8，V 分支改为 per-channel 静态，规避了按 32 分组可能出现的尾块问题；MXFP4 动态为 4bit 三分支统一 per-block 动态。本模式精度更高、V 缓存参数离线固化，代价是位宽更高，且 V 需校准数据。
- **与 [FA Q-INT8 动态 K/V-INT8 静态量化](term_fa_q_int8_dynamic_kv_int8.md)（同为 Q 动态 + K/V 静态的混合结构，数据类型不同）**
  - 本模式在混合结构基础上进一步区分 Q/K（动态）与 V（静态 per-channel），并整体采用 MXFP8 块级指数浮点数据类型；Q-INT8 混合模式为 Q 动态 + K/V per-head 静态的 INT8 整数模式。
- **与 [KVCache-PerChannel 量化](../kv_cache_quantization/term_kv_cache_perchannel.md)（基础关系）**
  - 本模式以 K/V 量化为基础，V 的 per-channel 静态量化与缓存压缩的通道语义直接对齐。

### 2.5 适用场景与限制

#### 2.5.1 适用场景

- **MXFP8 低精度注意力加速**：目标后端原生支持 MXFP8 注意力计算的部署。
- **Q/K 免校准 + V 缓存静态化**：Q/K 难以用固定校准覆盖、又希望 V 的缓存量化参数离线确定的场景。
- **MoE 大模型全量化**：作为线性层 W4A4/W8A8 与 FA 注意力整体量化中的注意力激活部分。

#### 2.5.2 使用限制

- **依赖硬件 MX 支持**：MXFP8 注意力计算需目标硬件算子库支持。
- **Q/K 分支 32 对齐**：Q/K 走 per-block，head_dim 需能被 32 整除，否则需 padding；V 分支为 per-channel、不按 32 分组，无此限制。
- **V 分支依赖校准**：V 的 per-channel scale 由校准集确定，无校准数据则无法确定静态参数。
- **在线与离线开销并存**：Q/K 有动态归约开销，同时 V 静态依赖校准流程。
- **注意力精度敏感**：Q/K 的量化误差直接影响注意力得分，需验证极端长序列下的精度。

---

## 3. 关联流程

- 《[一键量化完整指南](../../../user_guide/usage_quick_quantization.md)》：通过 YAML 配置选择本模式并执行量化。
- 《[量化精度调优指南](../../../user_guide/process_quantization_precision_tuning.md)》：本模式导致的精度劣化可通过该流程逐层回退与调优。

---

## 4. 关联词条

- [量化模式](../README.md)：上位概念，本词条属于[FA 量化](README.md)类别，是该类别下的一种具体模式。
- [FA 量化](README.md)：上位概念，本词条所属类别。
- [FA MXFP4 动态量化](term_fa_mxfp4_dynamic.md)：同类模式，三分支统一 MXFP4 per-block 动态的低比特模式。
- [FA Q-INT8 动态 K/V-INT8 静态量化](term_fa_q_int8_dynamic_kv_int8.md)：同类混合结构模式，Q 动态 + K/V per-head 静态的 INT8 模式。
- [KVCache 量化](../kv_cache_quantization/README.md)：基础类别，本模式在其基础上追加 Q 量化。
- [KVCache-PerChannel 量化](../kv_cache_quantization/term_kv_cache_perchannel.md)：前置术语，K/V 量化的具体模式（V 分支 per-channel 语义与之对齐）。
- 《[AscendV1 量化格式](../../quantization_format/ascendv1/term_ascendv1.md#desc-faquant)》：配套格式，本模式 AscendV1 导出约定（`{prefix}.quant_type` 如 `QK_MXFP8_DYNAMIC_V_MXFP8_PER_CHANNEL`）。

---

## 5. 参考文档

1. 《[FA3 Quant 参数配置流程指南](../../quantization_algorithms/fa3_quant/usage_fa3_quant.md)》
2. 《[量化模式](../README.md)》
