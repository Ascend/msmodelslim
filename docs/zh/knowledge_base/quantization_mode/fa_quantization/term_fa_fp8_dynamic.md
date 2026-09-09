# FA FP8 动态量化

> **词条类别**：量化数据格式（[FA 量化](README.md)）<br>
> **英文名称**：FA FP8 Dynamic Quantization<br>
> **应用领域**：长上下文推理加速、低精度注意力计算<br>
> **承载 IR 类**：`FakeQuantActivationPerToken`（[`msmodelslim/ir/activation_dynamic.py`](../../../../../msmodelslim/ir/activation_dynamic.py)）<br>

---

## 1. 概述

FA FP8 动态量化是对送入 Flash Attention 的 Q/K/V 激活统一按 token 逐行动态量化到 [FP8（E4M3）](../../quantization_basic/term_fp8.md) 的整体组合模式。量化参数在线计算、免校准，每个 token 的激活分辨率贴合自身数值范围。它是多模态生成、长序列解码等场景中免校准注意力量化的常用模式，K/V 承接缓存压缩（[KVCache 量化](../kv_cache_quantization/README.md)），Q 使能低精度矩阵运算。

---

## 2. 词条介绍

### 2.1 模式规格

| 维度 | 取值 | 说明 |
| --- | --- | --- |
| 量化对象 | 注意力 Q/K/V 激活 | 送入 Flash Attention 的激活张量（形状 (B, H, S, D)），`fa_q`/`fa_k`/`fa_v` 三分支均被量化；K/V 承接缓存压缩，Q 使能低精度得分计算 |
| 位宽 | 8bit | Q/K/V 均为 8bit |
| 数据类型 | FP8（E4M3） | 8bit 浮点，指数4位、尾数3位，可表示最大值448 |
| 参数获取方式 | 动态 | 每个 token 前向在线求 min/max 并计算 scale，免校准 |
| 量化粒度 | per-token | reshape 为 (B\*H\*S, D) 后按行（token）共享 scale |
| 对称性 | 对称 | 仅 scale，无 offset |

### 2.2 量化公式

对称量化（仅 scale、无零点）：

$$q = \mathrm{round}(x / s), \qquad \hat{x} = q \cdot s, \qquad s = \frac{\max(|x|)}{448}$$

上式中，$x$ 表示待量化的张量，$q$ 表示该张量量化后的数值，$\hat{x}$ 表示该张量经反量化还原后的数值，$s$ 表示 scale。FP8（E4M3）格式可表示的最大值为 448，因此 scale 需要按 FP8 的表示范围确定。在 per-token 粒度下，输入先被 reshape 为 (B\*H\*S, D)，随后逐行统计 min/max 并动态计算每个 token 的 scale。

### 2.3 与其他模式的关系

- **与 [FA INT8 动态量化](term_fa_int8_dynamic.md)（同为统一 per-token 动态，数据类型不同）**
  - 本模式为 FP8（E4M3）浮点格式；INT8 动态为 INT8 整数格式。FP8 浮点动态范围更大、对离群值相对更耐受，但 E4M3 精度尾数仅3bit；两者按部署算子与精度取舍。
- **与 [FA Q-FP8 动态 K/V-FP8 静态量化](term_fa_q_fp8_dynamic_kv_fp8.md)（同为 FP8 家族，分支量化方式不同）**
  - 本模式三分支统一 per-token 动态；混合模式将 K/V 改为 per-head 静态，用 K/V 的校准换取更低的在线开销。
- **与 [W8A8 FP8 动态量化](../linear_layer_quantization/term_w8a8_fp8_dynamic.md)（同类 FP8 per-token 动态思路）**
  - 本模式把同一思路应用于注意力 Q/K/V 激活；W8A8 FP8 动态应用于线性层权重与激活。

### 2.4 适用场景与限制

#### 2.4.1 适用场景

- **长上下文/多模态生成**：Wan2.2、Flux1、HunYuanVideo、QwenImageEdit 等模型的推荐默认，动态免校准对多模态输入分布变化更鲁棒。
- **校准数据不可靠或缺失**：无法获得代表性校准集时，动态 per-token 模式更稳。
- **注意力精度要求高**：需要比静态量化更细粒度的激活量化，且接受在线归约开销。

#### 2.4.2 使用限制

- **在线归约开销**：per-token min/max 归约增加少量延迟，短序列下占比更明显。
- **FP8 表示范围有限**：E4M3 max 448，大离群值需留意；量化误差直接影响注意力得分，需验证极端长序列下的精度。
- **硬件算子依赖**：FP8 per-token 动态量化需推理框架/算子库支持 FP8 注意力计算。

---

## 3. 关联流程

- 《[一键量化完整指南](../../../user_guide/usage_quick_quantization.md)》：通过 YAML 配置选择本模式并执行量化。
- 《[量化精度调优指南](../../../user_guide/process_quantization_precision_tuning.md)》：本模式导致的精度劣化可通过该流程逐层回退与调优。

---

## 4. 关联词条

- [量化模式](../README.md)：上位概念，本词条属于[FA 量化](README.md)类别，是该类别下的一种具体模式。
- [FA INT8 动态量化](term_fa_int8_dynamic.md)：同类模式，per-token 动态的 INT8 统一模式。
- [FA Q-FP8 动态 K/V-FP8 静态量化](term_fa_q_fp8_dynamic_kv_fp8.md)：同类模式，Q 走动态、K/V 走静态的 FP8 混合模式。
- [FA MXFP4 动态量化](term_fa_mxfp4_dynamic.md)：同类模式，per-block 动态的低比特统一模式。
- [W8A8 FP8 动态量化](../linear_layer_quantization/term_w8a8_fp8_dynamic.md)：配套模式，FP8 per-token 动态思路在线性层上的应用。
- [KVCache 量化](../kv_cache_quantization/README.md)：基础类别，本模式在其基础上追加 Q 量化。

---

## 5. 参考文档

1. 《[FA3 Quant 参数配置流程指南](../../quantization_algorithms/fa3_quant/usage_fa3_quant.md)》
2. 《[量化模式](../README.md)》
