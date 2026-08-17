# FA PerToken 量化

> **词条类别**：量化数据格式（[FA 量化](README.md)）
> **英文名称**：FA Per-Token Quantization
> **应用领域**：长上下文推理加速、低精度注意力计算
> **承载 IR 类**：`FakeQuantActivationPerToken`（[`msmodelslim/ir/activation_dynamic.py`](../../../../../msmodelslim/ir/activation_dynamic.py)）

---

## 1. 概述

FA PerToken 量化 = 对送入 Flash Attention 的 Q/K/V 激活按 token 逐行动态量化（[INT8](../../quantization_basic/term_int8.md) 或 [FP8（E4M3）](../../quantization_basic/term_fp8.md)），量化参数在线计算、免校准。它是 [KVCache 量化](../kv_cache_quantization/README.md)的**进阶**：K/V 量化省缓存显存，Q 一并量化后使注意力得分计算可走低精度矩阵运算；per-token 粒度让每个 token 的激活分辨率贴合自身数值范围。

---

## 2. 词条介绍

### 模式规格

| 维度 | 取值 | 说明 |
|------|------|------|
| 量化对象 | 注意力 Q/K/V 激活 | 送入 Flash Attention 的激活张量（形状 (B, H, S, D)），Q/K/V 均被量化；K/V 承接缓存压缩，Q 使能低精度得分计算 |
| 位宽 | 8bit | Q/K/V 均为 8bit |
| 数据类型 | INT8 / FP8（E4M3） | 两种实现按精度/吞吐需求选择 |
| 参数获取方式 | 动态 | 每个 token 前向在线求 min/max 并计算 scale，免校准 |
| 量化粒度 | per-token | reshape 为 (B\*H\*S, D) 后按行（token）共享 scale |
| 对称性 | 对称 | 仅 scale，无 offset |

### 量化公式

量化与反量化采用标准线性映射。对称量化（仅 scale、无零点）：
$$q = \mathrm{round}(x / s), \qquad \hat{x} = q \cdot s, \qquad s = \frac{\max(|x|)}{2^{b-1}}$$

非对称量化（含零点 offset）：
$$q = \mathrm{round}(x / s) + z, \qquad \hat{x} = (q - z) \cdot s, \qquad s = \frac{\max(x) - \min(x)}{2^b - 1}$$

其中 $x$ 为待量化张量，$q$ 为量化后的数值，$\hat{x}$ 为反量化还原值，$s$ 为 scale，$z$ 为零点 offset，$b$ 为位宽，且 $z = \mathrm{round}(-\min(x)/s)$。本模式为对称量化（无 offset）。FP8（E4M3）变体的 scale 按 FP8 可表示最大值 448 确定。量化对象为送入 Flash Attention 的 Q/K/V 激活，参数获取方式与作用粒度（per-head / per-token）见「模式规格」表。

### 与其他模式的关系

- **与 [FA PerHead 量化](term_fa_perhead.md)（同为 FA 激活量化，粒度与参数获取方式不同）**
  - 本模式（per-token 动态）：优势是逐 token 在线算 scale、免校准，量化贴合每个 token 的数值范围，精度更高；劣势是每次前向多一次按 token 的 min/max 归约，开销与 token 数同量级。
  - PerHead（per-head 静态）：优势是 scale 离线固化、推理零在线开销，参数仅 head 数；劣势是需依赖校准数据、粒度粗，对 token 间分布变化不敏感。

- **与 [FA PerBlock 量化](term_fa_perblock.md)（同为 FA 激活量化，格式与粒度不同）**
  - 本模式：INT8/FP8 每元素格式 + per-token 粒度，无需特殊硬件格式支持；劣势是 scale 数与 token 数同量级、粒度行级。
  - PerBlock：MX 格式块级共享指数（32元素一块），粒度更细（沿 head_dim）、对离群值更耐受；劣势是依赖硬件 MX 支持、head_dim 需32对齐。

- **与 [W8A8 动态量化](../linear_layer_quantization/term_w8a8_dynamic.md)（同类 per-token 动态思路）**
  - 本模式把同一思路应用于注意力 Q/K/V 激活；W8A8 动态应用于线性层权重与激活。二者正交、可组合：线性层走 W8A8，注意力路径走本模式，共同构成整体量化方案。

### 适用场景与限制

#### 1. 适用场景

- **校准数据不可靠或缺失**：无法获得代表性校准集时，动态 per-token 方案更稳。
- **注意力精度要求高**：需要比静态方案更细粒度的激活量化，且接受在线归约开销。
- **长序列注意力加速**：与 Flash Attention 整数/低精度路径结合，降低带宽与计算开销。

#### 2. 使用限制

- **在线归约开销**：per-token min/max 归约增加少量延迟，短序列下占比更明显。
- **注意力精度敏感**：Q/K 的量化误差直接影响注意力得分，需验证极端长序列下的精度。
- **硬件算子依赖**：per-token 动态量化需推理框架/算子库支持逐 token 参数计算。

---

## 3. 关联流程

- 《[一键量化完整指南](../../../user_guide/usage_quick_quantization.md)》：通过 YAML 配置选择本模式并执行量化。
- 《[量化精度调优指南](../../../user_guide/process_quantization_precision_tuning.md)》：本模式导致的精度劣化可通过该流程逐层回退与调优。

---

## 4. 关联词条

- [量化模式](../README.md)：上位概念，本词条属于[FA 量化](README.md)类别，是该类别下的一种具体模式。
- [FA PerHead 量化](term_fa_perhead.md)：同类模式，per-head 静态激活量化。
- [FA PerBlock 量化](term_fa_perblock.md)：同类模式，per-block MX 动态激活量化。
- [KVCache 量化](../kv_cache_quantization/README.md)：基础类别，本模式在其基础上追加 Q 量化。
- [W8A8 动态量化](../linear_layer_quantization/term_w8a8_dynamic.md)：配套模式，per-token 动态思路在线性层上的应用。
- 《[FA3量化：Flash Attention 3激活量化算法说明](../../quantization_algorithms/fa3_quant/usage_fa3_quant.md)》：配套术语，本模式对应的算法处理器文档。

---

## 5. 参考文档

1. 《[FA3量化：Flash Attention 3激活量化算法说明](../../quantization_algorithms/fa3_quant/usage_fa3_quant.md)》
2. 《[量化模式](../README.md)》
