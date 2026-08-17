# W4A4 MX 双 Scale 量化

> **词条类别**：量化数据格式（[线性层量化](README.md)）
> **英文名称**：W4A4 MX Dual-Scale Quantization
> **应用领域**：大语言模型量化压缩、推理加速
> **承载 IR 类**：`W4A4MXDynamicDualScaleFakeQuantLinear`（[`msmodelslim/ir/w4a4_mx_dynamic_dualscale.py`](../../../../../msmodelslim/ir/w4a4_mx_dynamic_dualscale.py)）

---

## 1. 概述

W4A4 MX 双 Scale 量化 = [W4A4 MX 动态量化](term_w4a4_mx_dynamic.md) 的**精度增强版**：在单层块级指数之外，再叠一层大块（如 512元素）高精度浮点 scale，构成「外层保量级、内层保细节」的两级缩放。它解决单层 [MXFP4](../../quantization_basic/term_mxfp.md) 在大块内部量级差异大或存在强离群值时的精度不足问题，以极少的额外参数换取 4bit 位宽下的明显精度提升。

---

## 2. 词条介绍

### 模式规格

| 维度 | 取值 | 说明 |
|------|------|------|
| 量化对象 | 权重 W、激活 A | 线性层矩阵乘两侧的张量 |
| 位宽 | W 4bit，A 4bit | 权重与激活访存均降至 FP16 的 1/4 |
| 数据类型 | MXFP4 双 scale | 外层大块高精度浮点 scale + 内层 32元素 E8M0 指数，元素为 1符号 + 2指数 + 1尾数 |
| 参数获取方式 | 激活动态 / 权重静态 | 激活两级参数前向在线计算；权重两级参数量化阶段固化 |
| 量化粒度 | 双层块粒度 | 外层 `dual_block_size`（如 512）共享高精度 scale；内层 32元素共享 E8M0 指数 |
| 对称性 | 对称 | 仅 scale/指数，无 offset |

### 量化公式

MX 格式按块共享 E8M0 指数，块内每个元素用块级指数缩放后量化为主尾数：
$$e = \lfloor \log_2(\max_{i \in \mathrm{block}}|x_i|) \rfloor, \qquad q_i = \mathrm{round}_{MX}\left(\frac{x_i}{2^e}\right), \qquad \hat{x}_i = q_i \cdot 2^e$$

其中块大小32元素，$e$ 为块内共享的 E8M0 指数，$q_i$ 为块内第 $i$ 个元素的量化尾数（MXFP8 为 FP8 尾数、MXFP4 为 FP4 尾数），$\hat{x}_i$ 为反量化还原值。每块共享一个 $e$；量化参数在线计算（激活）或量化阶段固化（权重），见「模式规格」表。

### 与其他模式的关系

- **与 [W4A4 MX 动态量化](term_w4a4_mx_dynamic.md)（同为 MXFP4，缩放层级不同）**
  - 本模式（双 scale）：优势是两级缩放解耦「整体量级」与「局部动态范围」，强离群值或量级差异大的大块下精度明显更稳；劣势是多一层参数与配置（`dual_block_size`）、实现更复杂。
  - 单层：优势是实现简单、参数开销最低；劣势是单个块指数难以同时覆盖量级与细节，极端分布下精度受限。

- **与 [W4A4 动态量化](term_w4a4_dynamic.md)（数据类型与缩放不同）**
  - 本模式（MXFP4 双 scale）：优势是浮点动态范围 + 两级缩放，精度显著优于 INT4 均匀分档；劣势是依赖 MX 硬件支持、块对齐约束。
  - INT4：优势是整数 GEMM 生态成熟、实现简单；劣势是对离群值敏感、动态范围受限。

- **与 [W4A8 MX 动态量化](term_w4a8_mx_dynamic.md)（激活位宽不同）**
  - 本模式：优势是激活亦压到 4bit、压缩比最大化，双 scale 用于兜底激活精度；劣势是 4bit 固有精度风险仍高于 8bit。
  - W4A8 MX：优势是激活保持 8bit、天然更稳；劣势是激活访存收益小于全 4bit。

### 适用场景与限制

#### 1. 适用场景

- **W4A4 精度敏感部署**：需要在 4bit 位宽下尽量逼近 FP16 精度的场景。
- **存在离群值的超低比特量化**：单层 MXFP4 精度不足、需要两级缩放缓解离群值影响。
- **昇腾 MXFP 推理路径**：硬件/算子库原生支持 MX 格式及双 scale 计算的部署。

#### 2. 使用限制

- **依赖硬件/算子支持**：双 scale 计算需目标硬件算子库支持。
- **块粒度对齐**：外层大块与内层小块尺寸均需能整除张量最后维。
- **配置约束**：`dual_block_size` 需按硬件与张量形状合理选取，取值不当会引入额外开销。
- **4bit 固有精度门槛**：即使双 scale，激活/权重 4bit 的精度风险仍高于 8bit 方案。

---

## 3. 关联流程

- 《[一键量化完整指南](../../../user_guide/usage_quick_quantization.md)》：通过 `--quant_type` 或 YAML 配置选择本模式并执行量化。
- 《[量化精度调优指南](../../../user_guide/process_quantization_precision_tuning.md)》：本模式导致的精度劣化可通过该流程逐层回退与调优。

---

## 4. 关联词条

- [量化模式](../README.md)：上位概念，本词条属于[线性层量化](README.md)类别，是该类别下的一种具体模式。
- [W4A4 MX 动态量化](term_w4a4_mx_dynamic.md)：同类模式，单层缩放的 MXFP4 基础版。
- [W4A4 动态量化](term_w4a4_dynamic.md)：对比模式，INT4 整数方案。
- [W4A8 MX 动态量化](term_w4a8_mx_dynamic.md)：同类模式，激活改用 MXFP8 保精度。
- 《[DualScale：w4a4量化方案说明](../../quantization_algorithms/dual_scale/usage_dual_scale.md)》：配套术语，本模式对应的算法处理器文档。

---

## 5. 参考文档

1. 《[线性量化算法说明](../../quantization_algorithms/linear_quant/usage_linear_quant.md)》
2. 《[量化模式](../README.md)》
