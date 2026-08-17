# W8A8 PD-Mix 量化

> **词条类别**：量化数据格式（[线性层量化](README.md)）
> **英文名称**：W8A8 PD-Mix Quantization
> **应用领域**：大语言模型量化压缩、推理加速
> **承载 IR 类**：`W8A8PDMixFakeQuantLinear`（[`msmodelslim/ir/w8a8_pdmix.py`](../../../../../msmodelslim/ir/w8a8_pdmix.py)）

---

## 1. 概述

W8A8 PD-Mix 量化 = 以 **Prefill / Decode 推理阶段**为界混合两种激活量化策略的 W8A8 方案：Prefill 阶段激活逐 token 动态量化（保精度），Decode 阶段激活整体静态量化（零在线开销）。它是 [W8A8 动态](term_w8a8_dynamic.md) 与 [W8A8 静态](term_w8a8_static.md) 的分阶段融合，「PD-Mix」即 Prefill-Decode Mix。

---

## 2. 词条介绍

### 模式规格

| 维度 | 取值 | 说明 |
|------|------|------|
| 量化对象 | 权重 W、激活 A | 线性层矩阵乘两侧的张量 |
| 位宽 | W 8bit，A 8bit | 同为 8bit，访存减半 |
| 数据类型 | INT8 | 整数格式，可走 INT8 整数 GEMM |
| 参数获取方式 | 混合（激活动态/静态随阶段切换；权重静态） | Prefill 阶段激活逐 token 在线求 min/max 算 scale；Decode 阶段激活整体静态（复用预置参数，此时单 token 激活即整张量，粒度与 per-token 等价）；权重 scale 固化 |
| 量化粒度 | 权重 per-channel；激活 Prefill per-token / Decode per-tensor | 权重逐输出通道共享 scale；激活按阶段取 per-token 或整体粒度 |
| 对称性 | 权重对称；激活非对称 | 激活非对称加 offset 适配各 token 分布 |

### 量化公式

量化与反量化采用标准线性映射。对称量化（仅 scale、无零点）：
$$q = \mathrm{round}(x / s), \qquad \hat{x} = q \cdot s, \qquad s = \frac{\max(|x|)}{2^{b-1}}$$

非对称量化（含零点 offset）：
$$q = \mathrm{round}(x / s) + z, \qquad \hat{x} = (q - z) \cdot s, \qquad s = \frac{\max(x) - \min(x)}{2^b - 1}$$

其中 $x$ 为待量化张量，$q$ 为量化后的整数值，$\hat{x}$ 为反量化还原值，$s$ 为 scale，$z$ 为零点 offset，$b$ 为位宽（INT8 取 $b=8$，INT4 取 $b=4$），且 $z = \mathrm{round}(-\min(x)/s)$。量化参数 $s$、$z$ 的获取方式（静态校准或在线动态）与作用粒度（per-channel、per-token、per-tensor 等）见「模式规格」表。

### 与其他模式的关系

- **与 [W8A8 动态量化](term_w8a8_dynamic.md)（同为 W8A8，激活策略不同）**
  - 本模式：优势是 decode 阶段退回静态、去掉每步 per-token 归约的无谓开销，而 prefill 阶段仍保住动态精度；劣势是需推理框架识别阶段并切换量化策略，复杂度与部署约束更高。
  - W8A8 动态：优势是全程 per-token 动态、激活精度统一，实现简单；劣势是 decode 阶段单 token 归约既无统计意义又增加延迟。

- **与 [W8A8 静态量化](term_w8a8_static.md)（同为 W8A8，激活策略不同）**
  - 本模式：优势是 prefill 阶段逐 token 动态、激活精度高于静态 per-tensor；劣势是复杂度更高、有阶段依赖。
  - 静态：优势是激活 scale 全程固化、零在线归约、实现与部署最简单；劣势是 prefill 阶段 per-tensor 粗粒度、精度上限低。

- **与 [W4A8 动态量化](term_w4a8_dynamic.md)（位宽不同）**
  - 本模式：优势是权重 8bit 精度更稳；劣势是权重访存仅减半。
  - W4A8：优势是权重压缩至 FP16 的 1/4；劣势是权重仅16档、有精度风险。

### 适用场景与限制

#### 1. 适用场景

- **Prefill 与 Decode 并重的服务端推理**：prefill 阶段需要激活精度、decode 阶段需要低延迟低开销的场景，如长上下文服务的增量生成。
- **激活分布随阶段显著变化的模型**：prefill 激活范围大而离散、decode 激活相对稳定的模型。

#### 2. 使用限制

- **依赖阶段感知**：量化策略切换需推理框架识别当前是 Prefill 还是 Decode，且两阶段共享同一套 INT8 数据流。
- **当前仅限 MindIE 部署**：阶段感知与切换当前仅在 MindIE 推理框架下生效。
- **仅支持 INT8**：本模式不提供 INT4/FP8 等数据类型变体。

---

## 3. 关联流程

- 《[一键量化完整指南](../../../user_guide/usage_quick_quantization.md)》：通过 `--quant_type` 或 YAML 配置选择本模式并执行量化。
- 《[量化精度调优指南](../../../user_guide/process_quantization_precision_tuning.md)》：本模式导致的精度劣化可通过该流程逐层回退与调优。

---

## 4. 关联词条

- [量化模式](../README.md)：上位概念，本词条属于[线性层量化](README.md)类别，是该类别下的一种具体模式。
- [W8A8 动态量化](term_w8a8_dynamic.md)：对比模式，全程 per-token 动态。
- [W8A8 静态量化](term_w8a8_static.md)：对比模式，全程静态。
- [W4A8 动态量化](term_w4a8_dynamic.md)：对比模式，权重 4bit 的混合位宽方案。
- 《[PDMIX：激活值阶段间混合量化算法说明](../../quantization_algorithms/pdmix/usage_pdmix.md)》：配套术语，本模式对应的算法处理器文档。
- 《[线性量化算法说明](../../quantization_algorithms/linear_quant/usage_linear_quant.md)》：配套术语，线性层量化的处理器实现。

---

## 5. 参考文档

1. 《[线性量化算法说明](../../quantization_algorithms/linear_quant/usage_linear_quant.md)》
2. 《[量化模式](../README.md)》
