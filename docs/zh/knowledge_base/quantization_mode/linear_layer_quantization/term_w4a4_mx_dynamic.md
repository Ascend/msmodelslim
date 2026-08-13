# W4A4 MX 动态量化

> **词条类别**：量化数据格式（[线性层量化](../README.md)）
> **英文名称**：W4A4 MX Dynamic Quantization
> **应用领域**：大语言模型量化压缩、推理加速
> **承载 IR 类**：`W4A4MXDynamicPerBlockFakeQuantLinear`（[`msmodelslim/ir/w4a4_mx_dynamic.py`](../../../../../msmodelslim/ir/w4a4_mx_dynamic.py)）

---

## 1. 概述

W4A4 MX 动态量化 = 双 4bit 的 MX 超低比特方案：权重与激活都按 32元素块共享 E8M0 指数、元素为 [MXFP4](../../quantization_basic/term_mxfp.md)。它在 4bit 位宽下用块级共享指数保留浮点动态范围，权重与激活访存都降至 [FP16/BF16](../../quantization_basic/term_fp16_bf16.md) 的 1/4，是压缩比最高的一类线性层方案；对离群值比 [INT4](../../quantization_basic/term_int4.md) 均匀分档更耐受，但激活 4bit 的固有精度风险依然存在。

---

## 2. 词条介绍

### 模式规格

| 维度 | 取值 | 说明 |
|------|------|------|
| 量化对象 | 权重 W、激活 A | 线性层矩阵乘两侧的张量 |
| 位宽 | W 4bit，A 4bit | 权重与激活访存均降至 FP16 的 1/4 |
| 数据类型 | MXFP4 | 每32元素共享一个 E8M0 指数，元素为 1符号 + 2指数 + 1尾数（emax=2，max 6） |
| 参数获取方式 | 激活动态 / 权重静态 | 激活块级指数前向在线统计；权重块级指数量化阶段固化 |
| 量化粒度 | 权重/激活均 per-block（32元素） | 最后维按 32元素分块，每块共享一个指数 |
| 对称性 | 对称 | 仅指数/scale，无 offset |

### 量化公式

MX 格式按块共享 E8M0 指数，块内每个元素用块级指数缩放后量化为主尾数：
$$e = \lfloor \log_2(\max_{i \in \mathrm{block}}|x_i|) \rfloor, \qquad q_i = \mathrm{round}_{MX}\left(\frac{x_i}{2^e}\right), \qquad \hat{x}_i = q_i \cdot 2^e$$

其中块大小32元素，$e$ 为块内共享的 E8M0 指数，$q_i$ 为块内第 $i$ 个元素的量化尾数（MXFP8 为 FP8 尾数、MXFP4 为 FP4 尾数），$\hat{x}_i$ 为反量化还原值。每块共享一个 $e$；量化参数在线计算（激活）或量化阶段固化（权重），见「模式规格」表。

### 与其他模式的关系

- **与 [W4A4 动态量化](term_w4a4_dynamic.md)（同为 W4A4，数据类型不同）**
  - 本模式（MXFP4）：优势是块级共享指数保留浮点动态范围、对离群值比 [INT4](../../quantization_basic/term_int4.md) 均匀分档更耐受；劣势是依赖硬件 MX 支持、最后维需 32 对齐。
  - INT4：优势是整数 GEMM 生态成熟、实现简单；劣势是均匀分档对离群值敏感，需更细粒度与离群抑制兜底。

- **与 [W4A4 MX 双 Scale 量化](term_w4a4_mx_dualscale.md)（同为 MXFP4，缩放层级不同）**
  - 本模式（单层）：优势是实现简单、参数开销最低；劣势是大块内部量级差异大或存在强离群值时，单个指数难以同时覆盖「整体量级」与「局部细节」。
  - 双 Scale：优势是外层高精度 scale 保不溢出、内层指数保细节，精度明显更稳；劣势是多一层参数与配置（`dual_block_size`），实现更复杂。

- **与 [W4A8 MX 动态量化](term_w4a8_mx_dynamic.md)（激活位宽不同）**
  - 本模式：优势是激活亦压到 4bit、压缩比最大化；劣势是激活 4bit 精度风险远高于 8bit。
  - W4A8 MX：优势是激活保持 8bit、精度更稳，是压缩与精度的折中。

### 适用场景与限制

#### 1. 适用场景

- **极致压缩部署**：显存/带宽受限、追求最大压缩比的场景，如大规模服务端多模型部署。
- **离群值敏感但需 4bit**：在 4bit 位宽下需要比 INT4 更好精度的场景。
- **昇腾 MXFP 推理路径**：硬件/算子库原生支持 MX 格式的部署。

#### 2. 使用限制

- **依赖硬件 MX 支持**：MXFP4 计算需目标硬件算子库支持。
- **块粒度对齐**：张量最后维需能被 32 整除，否则需 padding。
- **低比特精度门槛**：4bit 量化精度风险大，通常需配合离群值抑制与精度调优。
- **激活 4bit 风险**：激活 4bit 对分布极其敏感，比 W4A8 MX 方案风险更高。

---

## 3. 关联流程

- 《[一键量化完整指南](../../../user_guide/usage_quick_quantization.md)》：通过 `--quant_type` 或 YAML 配置选择本模式并执行量化。
- 《[量化精度调优指南](../../../user_guide/process_quantization_precision_tuning.md)》：本模式导致的精度劣化可通过该流程逐层回退与调优。

---

## 4. 关联词条

- [量化模式](../README.md)：上位概念，本词条属于[线性层量化](README.md)类别，是该类别下的一种具体模式。
- [W4A4 动态量化](term_w4a4_dynamic.md)：对比模式，INT4 整数方案。
- [W4A4 MX 双 Scale 量化](term_w4a4_mx_dualscale.md)：同类模式，双层缩放的精度增强版。
- [W4A8 MX 动态量化](term_w4a8_mx_dynamic.md)：同类模式，激活改用 MXFP8 保精度。
- 《[线性量化算法说明](../../quantization_algorithms/linear_quant/linear_quant.md)》：配套术语，本模式的处理器实现。

---

## 5. 参考文档

1. 《[线性量化算法说明](../../quantization_algorithms/linear_quant/linear_quant.md)》
2. 《[量化模式](../README.md)》
