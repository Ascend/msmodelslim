# FA MXFP4 动态量化

> **词条类别**：量化数据格式（[FA 量化](README.md)）<br>
> **英文名称**：FA MXFP4 Dynamic Quantization<br>
> **应用领域**：长上下文推理加速、低精度注意力计算<br>
> **承载 IR 类**：`FakeQuantActivationPerBlock`（[`msmodelslim/ir/activation_dynamic.py`](../../../../../msmodelslim/ir/activation_dynamic.py)）<br>

---

## 1. 概述

FA MXFP4 动态量化是对送入 Flash Attention 的 Q/K/V 激活统一沿 head_dim 按 32 元素分块、块级共享 E8M0 指数做 [MXFP4](../../quantization_basic/term_mxfp.md) 动态量化的整体组合模式。量化参数在线计算、免校准，块级共享指数保留浮点动态范围。它比 per-token 粒度更细（能捕捉 head_dim 内部分布差异），是 FA 量化中“带宽、计算收益最大”的一档。

---

## 2. 词条介绍

### 2.1 模式规格

| 维度 | 取值 | 说明 |
| --- | --- | --- |
| 量化对象 | 注意力 Q/K/V 激活 | 送入 Flash Attention 的激活张量（形状 (B, H, S, D)），`fa_q`/`fa_k`/`fa_v` 三分支均被量化；K/V 承接缓存压缩，Q 使能低精度得分计算 |
| 位宽 | 4bit | Q/K/V 均为 4bit |
| 数据类型 | MXFP4 | 每 32 元素共享一个 E8M0 指数；MXFP4 emax=2、max 6 |
| 参数获取方式 | 动态 | 块级指数前向在线统计，免校准 |
| 量化粒度 | per-block | 沿最后维（head_dim）按 32 元素分块，每块共享一个指数 |
| 对称性 | 对称 | 仅指数/scale，无 offset |

### 2.2 量化公式

MX 格式按块共享 E8M0 指数，块内每个元素用块级指数缩放后量化为主尾数：

$$e = \lfloor \log_2(\max_{i \in \mathrm{block}}|x_i|) \rfloor, \qquad q_i = \mathrm{round}_{MX}\left(\frac{x_i}{2^e}\right), \qquad \hat{x}_i = q_i \cdot 2^e$$

其中，每个块包含 32 个元素，$e$ 表示块内共享的 E8M0 指数，$q_i$ 表示块内第 $i$ 个元素的量化尾数（MXFP4 为 1bit 尾数），$\hat{x}_i$ 表示该元素经反量化还原后的数值。该模式的量化参数在量化过程中在线统计得到，因此无需校准数据，具体参数获取方式可参见上文“模式规格”表。

### 2.3 与其他模式的关系

- **与 [FA FP8 动态量化](term_fa_fp8_dynamic.md) / [FA INT8 动态量化](term_fa_int8_dynamic.md)（同为统一动态，格式与粒度不同）**
  - 本模式（MXFP4 per-block）：优势是 4bit 带宽收益最大，块级共享指数对离群值更耐受、粒度沿 head_dim 更细；劣势是依赖硬件 MX 支持、head_dim 需32对齐、MXFP4 尾数仅1bit（emax=2、max 6）。
  - FP8/INT8 动态：优势是 8bit 每元素格式、硬件生态更成熟；劣势是位宽更高、粒度行级。
- **与 [W8A8 MX 动态量化](../linear_layer_quantization/term_w8a8_mx_dynamic.md)（同类 MX per-block 动态思路）**
  - 本模式把同一思路应用于注意力 Q/K/V 激活；W8A8 MX 应用于线性层权重与激活。
- **与 [FA QK-MXFP8 动态 / V-MXFP8 PerChannel 静态量化](term_fa_qk_mxfp8_dynamic_v_mxfp8_perchannel.md)（同为 MX 家族，位宽与分支量化方式不同）**
  - 本模式为 4bit，Q/K/V 三分支统一 per-block 动态、免校准；MXFP8 混合为 8bit，Q/K 走 per-block 动态、V 改走 per-channel 静态。V 静态化规避按 32 分组可能出现的尾块问题，精度更高，但位宽更大、V 需校准数据。

### 2.4 适用场景与限制

#### 2.4.1 适用场景

- **低比特注意力加速**：需要 4bit 注意力激活量化、又希望比整数格式精度更稳的场景。
- **离群值敏感的注意力**：MX 浮点动态范围对 Q/K/V 离群值更耐受。
- **昇腾 MXFP 部署环境**：硬件/算子库原生支持 MX 注意力计算的部署。

#### 2.4.2 使用限制

- **依赖硬件 MX 支持**：MXFP4 注意力计算需目标硬件算子库支持。
- **块粒度对齐**：head_dim 需能被32整除，否则需 padding。
- **在线归约开销**：per-block 统计增加少量延迟，短序列下占比更明显。
- **MXFP4 精度有限**：尾数仅1bit、emax=2、max 6，精度敏感场景需权衡。
- **注意力精度敏感**：Q/K 的量化误差直接影响注意力得分，需验证极端长序列下的精度。

---

## 3. 关联流程

- 《[一键量化完整指南](../../../user_guide/usage_quick_quantization.md)》：通过 YAML 配置选择本模式并执行量化。
- 《[量化精度调优指南](../../../user_guide/process_quantization_precision_tuning.md)》：本模式导致的精度劣化可通过该流程逐层回退与调优。

---

## 4. 关联词条

- [量化模式](../README.md)：上位概念，本词条属于[FA 量化](README.md)类别，是该类别下的一种具体模式。
- [FA FP8 动态量化](term_fa_fp8_dynamic.md)：同类模式，per-token 动态的 FP8 统一模式。
- [FA INT8 动态量化](term_fa_int8_dynamic.md)：同类模式，per-token 动态的 INT8 统一模式。
- [FA QK-MXFP8 动态 / V-MXFP8 PerChannel 静态量化](term_fa_qk_mxfp8_dynamic_v_mxfp8_perchannel.md)：同类模式，MXFP8 per-block/per-channel 的 8bit 混合模式。
- [W8A8 MX 动态量化](../linear_layer_quantization/term_w8a8_mx_dynamic.md)：配套模式，MX per-block 思路在线性层上的应用。
- [KVCache 量化](../kv_cache_quantization/README.md)：基础类别，本模式在其基础上追加 Q 量化。

---

## 5. 参考文档

1. 《[FA3 Quant 参数配置流程指南](../../quantization_algorithms/fa3_quant/usage_fa3_quant.md)》
2. 《[量化模式](../README.md)》
