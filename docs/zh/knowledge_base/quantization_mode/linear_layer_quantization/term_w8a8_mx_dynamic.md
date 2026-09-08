# W8A8 MX 动态量化

> **词条类别**：量化数据格式（[线性层量化](README.md)）<br>
> **英文名称**：W8A8 MX Dynamic Quantization<br>
> **应用领域**：大语言模型量化压缩、推理加速<br>
> **承载 IR 类**：`W8A8MXDynamicPerBlockFakeQuantLinear`（[`msmodelslim/ir/w8a8_mx_dynamic.py`](../../../../../msmodelslim/ir/w8a8_mx_dynamic.py)）<br>

## 1. 概述

W8A8 MX 动态量化是**[MXFP8](../../quantization_basic/term_mxfp.md) 版本**的 W8A8：权重与激活都按 32元素分块，每块共享一个 8bit 指数（E8M0），元素为 8bit 浮点（E4M3）。块级共享指数把缩放参数开销从每元素一份摊薄到每32元素一份，同时保留浮点动态范围——相比 [INT8](../../quantization_basic/term_int8.md) 均匀分档对离群值更耐受，相比普通 [FP8（E4M3）](../../quantization_basic/term_fp8.md) 每元素 scale 参数开销更低。

---

## 2. 词条介绍

### 2.1 模式规格

| 维度 | 取值 | 说明 |
|------|------|------|
| 量化对象 | 权重 W、激活 A | 线性层矩阵乘两侧的张量 |
| 位宽 | W 8bit，A 8bit | 元素位宽 8bit，访存减半 |
| 数据类型 | MXFP8 | 每32元素共享一个 E8M0 8bit 指数，元素为 E4M3（emax=8，max 448） |
| 参数获取方式 | 激活动态 / 权重静态 | 激活块级指数前向在线统计；权重块级指数量化阶段固化 |
| 量化粒度 | 权重/激活均 per-block（32元素） | 最后一维按 32 元素分块，每块共享一个指数 |
| 对称性 | 对称 | 仅指数/scale，无 offset |

### 2.2 量化公式

MX 格式按块共享 E8M0 指数，块内每个元素用块级指数缩放后量化为尾数：
$$e = \lfloor \log_2(\max_{i \in \mathrm{block}}|x_i|) \rfloor, \qquad q_i = \mathrm{round}_{MX}\left(\frac{x_i}{2^e}\right), \qquad \hat{x}_i = q_i \cdot 2^e$$

其中块大小32元素，$e$ 为块内共享的 E8M0 指数，$q_i$ 为块内第 $i$ 个元素的量化尾数（MXFP8 为 FP8 尾数、MXFP4 为 FP4 尾数），$\hat{x}_i$ 为反量化还原值。每块共享一个 $e$；量化参数在线计算（激活）或量化阶段固化（权重），见上文模式规格表。

### 2.3 与其他模式的关系

- **与 [W8A8 动态量化](term_w8a8_dynamic.md)（同为 8bit 动态家族，量化粒度不同）**
  - 一个 token 是推理的基本单元，通常由 N 个通道构成。本模式按 per-block 量化：每个 token 的每 32 个通道共享一份量化参数，对通道间数值差异的刻画更细。
  - W8A8 动态量化按 per-token 量化：每个 token（也即全部 N 个通道）共享一份量化参数，参数开销更小，但对通道间差异的刻画更粗。

- **与 [W8A8 FP8 动态量化](term_w8a8_fp8_dynamic.md)（同为 8bit 动态家族，量化粒度不同）**
  - 本模式仍是 per-block：每个 token 的每 32 个通道共享一份量化参数。
  - W8A8 FP8 动态量化按 per-token 或 per-channel 量化：每个 token 或每个通道共享一份参数，粒度与本模式不同。

- **与 [W4A4 MX 动态量化](term_w4a4_mx_dynamic.md)（位宽不同）**
  - 本模式：优势是 8bit 精度更稳；劣势是权重访存仅减半。
  - W4A4 MX：优势是权重与激活均压缩至 FP16 的 1/4，压缩比最大化；劣势是 4bit 激活精度风险高。

### 2.4 适用场景与限制

#### 2.4.1 适用场景

- **存在离群值的低比特部署**：相比 INT8，MXFP8 的浮点动态范围更大，对离群值更耐受。
- **需要低参数开销的 8bit**：块级共享指数在参数开销上优于每元素 scale 方案。
- **昇腾 MXFP 推理路径**：硬件/算子库原生支持 MX 格式的部署。

#### 2.4.2 使用限制

- **依赖硬件 MX 支持**：MXFP8 计算需目标硬件算子库支持，否则无法兑现收益。
- **块粒度对齐**：张量最后一维（如 in_dim / head_dim）长度须为 32 的倍数，否则需 padding。
- **动态归约开销**：激活 per-block 在线统计增加少量延迟。

---

## 3. 关联流程

- 《[一键量化完整指南](../../../user_guide/usage_quick_quantization.md)》：通过 `--quant_type` 或 YAML 配置选择本模式并执行量化。
- 《[量化精度调优指南](../../../user_guide/process_quantization_precision_tuning.md)》：本模式导致的精度劣化可通过该流程逐层回退与调优。

---

## 4. 关联词条

- [量化模式](../README.md)：上位概念，本词条属于[线性层量化](README.md)类别，是该类别下的一种具体模式。
- [W8A8 动态量化](term_w8a8_dynamic.md)：对比模式，INT8 整数方案。
- [W8A8 FP8 动态量化](term_w8a8_fp8_dynamic.md)：对比模式，普通 FP8 每元素缩放方案。
- [W4A4 MX 动态量化](term_w4a4_mx_dynamic.md)：同类模式，MXFP4 更低比特方案。
- [线性量化算法](../../quantization_algorithms/linear_quant/term_linear_quant.md)：配套术语，本模式的处理器实现。

---

## 5. 参考文档

1. 《[线性量化参数配置流程指南](../../quantization_algorithms/linear_quant/usage_linear_quant.md)》
2. 《[量化模式](../README.md)》
