# W4A8 MX 动态量化

> **词条类别**：量化数据格式（[线性层量化](README.md)）<br>
> **英文名称**：W4A8 MX Dynamic Quantization<br>
> **应用领域**：大语言模型量化压缩、推理加速<br>
> **承载 IR 类**：`W4A8MXDynamicPerBlockFakeQuantLinear`（[`msmodelslim/ir/w4a8_mx_dynamic.py`](../../../../../msmodelslim/ir/w4a8_mx_dynamic.py)）<br>

## 1. 概述

W4A8 MX 动态量化是把 W4A8 位宽搭配换成 MX 格式：权重用 [MXFP4](../../quantization_basic/term_mxfp.md)（1符号 + 2指数 + 1尾数）拿 4bit 压缩、激活用 [MXFP8](../../quantization_basic/term_mxfp.md)（E4M3）保8bit精度，两侧都按 32元素块共享 E8M0 指数。相比 [INT4](../../quantization_basic/term_int4.md)/[INT8](../../quantization_basic/term_int8.md) 组合，MX 浮点格式对离群值更耐受；相比全 4bit 方案，激活保持 8bit 显著降低精度风险，是权重极致压缩 + 激活保精度的 MX 版折中。

---

## 2. 词条介绍

### 2.1 模式规格

| 维度 | 取值 | 说明 |
|------|------|------|
| 量化对象 | 权重 W、激活 A | 线性层矩阵乘两侧的张量 |
| 位宽 | W 4bit，A 8bit | 权重访存降至 FP16 的 1/4，激活降至 1/2 |
| 数据类型 | 权重 MXFP4；激活 MXFP8 | MXFP4 emax=2、max 6；MXFP8 emax=8、max 448；均为每32元素共享 E8M0 指数 |
| 参数获取方式 | 激活动态 / 权重静态 | 激活块级指数前向在线统计；权重块级指数量化阶段固化 |
| 量化粒度 | 权重/激活均 per-block（32元素） | 最后一维按 32 元素分块，每块共享一个指数 |
| 对称性 | 对称 | 仅指数/scale，无 offset |

### 2.2 量化公式

MX 格式按块共享 E8M0 指数，块内每个元素用块级指数缩放后量化为尾数：
$$e = \lfloor \log_2(\max_{i \in \mathrm{block}}|x_i|) \rfloor, \qquad q_i = \mathrm{round}_{MX}\left(\frac{x_i}{2^e}\right), \qquad \hat{x}_i = q_i \cdot 2^e$$

其中块大小32元素，$e$ 为块内共享的 E8M0 指数，$q_i$ 为块内第 $i$ 个元素的量化尾数（MXFP8 为 FP8 尾数、MXFP4 为 FP4 尾数），$\hat{x}_i$ 为反量化还原值。每块共享一个 $e$；量化参数在线计算（激活）或量化阶段固化（权重），见上文模式规格表。

### 2.3 与其他模式的关系

- **与 [W4A8 动态量化](term_w4a8_dynamic.md)（同为 W4A8，数据类型与粒度不同）**
  - 本模式（MX 格式）：优势是 MXFP4 权重按块共享指数，对离群值更耐受，MXFP8 激活保留更宽的浮点动态范围；劣势是依赖硬件 MX 支持、最后一维（如 in_dim / head_dim）长度须为 32 的倍数。
  - INT4/INT8：优势是整数格式、硬件算子生态成熟；劣势是均匀分档对离群值敏感。

- **与 [W8A8 MX 动态量化](term_w8a8_mx_dynamic.md)（权重位宽不同）**
  - 本模式：优势是权重压缩至 4bit、访存降至 FP16 的 1/4；劣势是权重仅有少量可表示值、精度风险高于 8bit。
  - W8A8 MX：优势是权重 8bit 精度更稳；劣势是权重访存仅减半。

- **与 [W4A4 MX 动态量化](term_w4a4_mx_dynamic.md)（激活位宽不同）**
  - 本模式：优势是激活保持 8bit、远离最敏感的激活量化误差，是压缩与精度的折中；劣势是激活访存收益小于全 4bit。
  - W4A4 MX：优势是双 4bit 压缩最大化；劣势是激活 4bit 精度风险高。

### 2.4 适用场景与限制

#### 2.4.1 适用场景

- **权重访存受限 + 激活分布动态范围大**：需要权重 4bit 压缩、又担心 INT8 激活精度不足的场景。
- **离群值敏感的低比特部署**：MX 浮点格式比整数格式更耐受离群值。
- **昇腾 MXFP 推理路径**：硬件/算子库原生支持 MX 格式的部署。

#### 2.4.2 使用限制

- **依赖硬件 MX 支持**：MXFP4/MXFP8 计算需目标硬件算子库支持。
- **块粒度对齐**：张量最后一维（如 in_dim / head_dim）长度须为 32 的倍数，否则需 padding。
- **4bit 权重精度风险**：MXFP4 仅16个可表示值，极端权重分布下仍需验证精度。

---

## 3. 关联流程

- 《[一键量化完整指南](../../../user_guide/usage_quick_quantization.md)》：通过 `--quant_type` 或 YAML 配置选择本模式并执行量化。
- 《[量化精度调优指南](../../../user_guide/process_quantization_precision_tuning.md)》：本模式导致的精度劣化可通过该流程逐层回退与调优。

---

## 4. 关联词条

- [量化模式](../README.md)：上位概念，本词条属于[线性层量化](README.md)类别，是该类别下的一种具体模式。
- [W4A8 动态量化](term_w4a8_dynamic.md)：对比模式，INT4 权重 + INT8 激活的整数方案。
- [W8A8 MX 动态量化](term_w8a8_mx_dynamic.md)：同类模式，权重亦为 MXFP8。
- [W4A4 MX 动态量化](term_w4a4_mx_dynamic.md)：同类模式，激活亦降为 MXFP4。
- [线性量化算法](../../quantization_algorithms/linear_quant/term_linear_quant.md)：配套术语，本模式的处理器实现。

---

## 5. 参考文档

1. 《[线性量化参数配置流程指南](../../quantization_algorithms/linear_quant/usage_linear_quant.md)》
2. 《[量化模式](../README.md)》
