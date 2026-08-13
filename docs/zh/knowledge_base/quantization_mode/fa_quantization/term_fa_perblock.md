# FA PerBlock 量化

> **词条类别**：量化数据格式（[FA 量化](../README.md)）
> **英文名称**：FA Per-Block Quantization
> **应用领域**：长上下文推理加速、低精度注意力计算
> **承载 IR 类**：`FakeQuantActivationPerBlock`（[`msmodelslim/ir/activation_dynamic.py`](../../../../../msmodelslim/ir/activation_dynamic.py)）

---

## 1. 概述

FA PerBlock 量化 = 对送入 Flash Attention 的 Q/K/V 激活沿 head_dim 按32元素分块，每块共享一个 E8M0 指数做 MX 动态量化（[MXFP8](../../quantization_basic/term_mxfp.md) 或 [MXFP4](../../quantization_basic/term_mxfp.md)），量化参数在线计算、免校准。它比 per-token 粒度更细（能捕捉 head_dim 内部分布差异），又以块级共享指数保留浮点动态范围、对离群值比整数格式更耐受，是注意力激活量化中「粒度/开销」折中最细的一档。

---

## 2. 词条介绍

### 模式规格

| 维度 | 取值 | 说明 |
|------|------|------|
| 量化对象 | 注意力 Q/K/V 激活 | 送入 Flash Attention 的激活张量（形状 (B, H, S, D)），Q/K/V 均被量化；K/V 承接缓存压缩，Q 使能低精度得分计算 |
| 位宽 | 4bit 或 8bit | MXFP4 为 4bit、MXFP8 为 8bit，按精度/吞吐需求选择 |
| 数据类型 | MXFP4 / MXFP8 | 每32元素共享一个 E8M0 指数；MXFP4 emax=2、max 6，MXFP8 emax=8、max 448 |
| 参数获取方式 | 动态 | 块级指数前向在线统计，免校准 |
| 量化粒度 | per-block | 沿最后维（head_dim）按32元素分块，每块共享一个指数 |
| 对称性 | 对称 | 仅指数/scale，无 offset |

### 量化公式

MX 格式按块共享 E8M0 指数，块内每个元素用块级指数缩放后量化为主尾数：
$$e = \lfloor \log_2(\max_{i \in \mathrm{block}}|x_i|) \rfloor, \qquad q_i = \mathrm{round}_{MX}\left(\frac{x_i}{2^e}\right), \qquad \hat{x}_i = q_i \cdot 2^e$$

其中块大小32元素，$e$ 为块内共享的 E8M0 指数，$q_i$ 为块内第 $i$ 个元素的量化尾数（MXFP8 为 FP8 尾数、MXFP4 为 FP4 尾数），$\hat{x}_i$ 为反量化还原值。量化对象为送入 Flash Attention 的 Q/K/V 激活，量化参数在线计算，见「模式规格」表。

### 与其他模式的关系

- **与 [FA PerToken 量化](term_fa_pertoken.md)（同为 FA 激活量化，格式与粒度不同）**
  - 本模式（MX per-block）：优势是块级共享指数保留浮点动态范围、对离群值比 INT8/FP8 每元素格式更耐受，粒度沿 head_dim 更细；劣势是依赖硬件 MX 支持、head_dim 需32对齐，块统计归约开销与块数相关。
  - PerToken：优势是 INT8/FP8 每元素格式、硬件生态成熟、实现简单；劣势是 scale 数与 token 数同量级、粒度行级。

- **与 [FA PerHead 量化](term_fa_perhead.md)（同为 FA 激活量化，粒度与参数获取方式不同）**
  - 本模式：per-block 动态、免校准、粒度最细；劣势是在线归约开销、依赖 MX 硬件。
  - PerHead：per-head 静态、需校准、粒度最粗；优势是零在线开销、参数仅 head 数。

- **与 [W8A8 MX 动态量化](../linear_layer_quantization/term_w8a8_mx_dynamic.md)（同类 MX per-block 动态思路）**
  - 本模式把同一思路应用于注意力 Q/K/V 激活；W8A8 MX 应用于线性层权重与激活。二者正交、可组合：线性层走 MX，注意力路径走本模式。

### 适用场景与限制

#### 1. 适用场景

- **低比特注意力加速**：需要 4bit/8bit 注意力激活量化、又希望比整数格式精度更稳的场景。
- **离群值敏感的注意力**：MX 浮点动态范围对 Q/K/V 离群值更耐受。
- **昇腾 MXFP 推理路径**：硬件/算子库原生支持 MX 注意力计算的部署。

#### 2. 使用限制

- **依赖硬件 MX 支持**：MXFP4/MXFP8 注意力计算需目标硬件算子库支持。
- **块粒度对齐**：head_dim 需能被32整除，否则需 padding。
- **在线归约开销**：per-block 统计增加少量延迟，短序列下占比更明显。
- **注意力精度敏感**：Q/K 的量化误差直接影响注意力得分，需验证极端长序列下的精度。

---

## 3. 关联流程

- 《[一键量化完整指南](../../../user_guide/usage_quick_quantization.md)》：通过 YAML 配置选择本模式并执行量化。
- 《[量化精度调优指南](../../../user_guide/process_quantization_precision_tuning.md)》：本模式导致的精度劣化可通过该流程逐层回退与调优。

---

## 4. 关联词条

- [量化模式](../README.md)：上位概念，本词条属于[FA 量化](README.md)类别，是该类别下的一种具体模式。
- [FA PerToken 量化](term_fa_pertoken.md)：同类模式，per-token 动态激活量化。
- [FA PerHead 量化](term_fa_perhead.md)：同类模式，per-head 静态激活量化。
- [KVCache 量化](../kv_cache_quantization/README.md)：基础类别，本模式在其基础上追加 Q 量化。
- [W8A8 MX 动态量化](../linear_layer_quantization/term_w8a8_mx_dynamic.md)：配套模式，MX per-block 思路在线性层上的应用。
- 《[FA3量化：Flash Attention 3激活量化算法说明](../../quantization_algorithms/fa3_quant/fa3_quant.md)》：配套术语，本模式对应的算法处理器文档。

---

## 5. 参考文档

1. 《[FA3量化：Flash Attention 3激活量化算法说明](../../quantization_algorithms/fa3_quant/fa3_quant.md)》
2. 《[量化模式](../README.md)》
