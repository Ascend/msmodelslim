# W8A16 静态量化

> **词条类别**：量化数据格式（[线性层量化](../README.md)）
> **英文名称**：W8A16 Static Quantization
> **应用领域**：大语言模型量化压缩、推理加速
> **承载 IR 类**：`W8A16StaticPerChannelFakeQuantLinear` / `W8A16StaticPerGroupFakeQuantLinear`（[`msmodelslim/ir/w8a16_static.py`](../../../../../msmodelslim/ir/w8a16_static.py)）

---

## 1. 概述

W8A16 静态量化 = 只把线性层的权重压到 [INT8](../../quantization_basic/term_int8.md)，激活保持 [FP16/BF16](../../quantization_basic/term_fp16_bf16.md) 不量化。它面向"激活精度敏感、不敢量化"的场景：权重是每 token 都要整体读取的静态数据，压到 [INT8](../../quantization_basic/term_int8.md) 使权重访存减半；激活不量化则完全规避激活量化误差，是精度最稳的线性层方案之一。代价是激活不量化意味着矩阵乘需把 [INT8](../../quantization_basic/term_int8.md) 权重反量化回浮点执行，没有[整数 GEMM](../term_gemm.md) 加速。

---

## 2. 词条介绍

### 模式规格

| 维度 | 取值 | 说明 |
|------|------|------|
| 量化对象 | 仅权重 W | 激活保持 FP16 原样参与计算，不引入任何量化误差 |
| 位宽 | W 8bit，A 16bit | 权重 1字节/元素、访存减半；激活 2字节/元素 |
| 数据类型 | 权重 INT8；激活 FP16 | 混合位宽，压缩收益全部来自权重侧 |
| 参数获取方式 | 静态 | 权重 scale/offset 量化阶段固化，推理零在线开销；激活无参数 |
| 量化粒度 | 权重 per-channel/per-group | 逐输出通道或按固定分组（如 128元素）共享 scale，适配通道/分组间数值差异 |
| 对称性 | 权重对称或非对称 | 依据权重分布选择，非对称加 offset |

### 量化公式

量化与反量化采用标准线性映射。对称量化（仅 scale、无零点）：
$$q = \mathrm{round}(x / s), \qquad \hat{x} = q \cdot s, \qquad s = \frac{\max(|x|)}{2^{b-1}}$$

非对称量化（含零点 offset）：
$$q = \mathrm{round}(x / s) + z, \qquad \hat{x} = (q - z) \cdot s, \qquad s = \frac{\max(x) - \min(x)}{2^b - 1}$$

其中 $x$ 为待量化张量，$q$ 为量化后的整数值，$\hat{x}$ 为反量化还原值，$s$ 为 scale，$z$ 为零点 offset，$b$ 为位宽（INT8 取 $b=8$，INT4 取 $b=4$），且 $z = \mathrm{round}(-\min(x)/s)$。量化参数 $s$、$z$ 的获取方式（静态校准或在线动态）与作用粒度（per-channel、per-token、per-tensor 等）见「模式规格」表。本模式仅权重走上述公式（$b=8$），激活保持 FP16 不量化。

### 与其他模式的关系

- **与 [W8A8 静态量化](term_w8a8_static.md)（激活是否量化不同）**
  - 本模式：优势是激活保持 FP16、零量化误差，是精度最稳的基线之一，适合激活存在离群值或对量化敏感的层；劣势是激活不量化，矩阵乘需把 [INT8](../../quantization_basic/term_int8.md) 权重反量化回浮点执行，无整数 GEMM 带来的计算加速，性能较差，整体压缩低于 W8A8。
  - W8A8：优势是激活亦量化为 INT8、可走整数 GEMM，计算吞吐更高、压缩更彻底；劣势是激活引入量化误差。

- **与 [W4A8 动态量化](term_w4a8_dynamic.md)（位宽搭配不同）**
  - 本模式：优势是激活完全无量化误差，且无在线归约开销；劣势是权重仅压到 8bit、访存收益有限，无计算加速。
  - W4A8：优势是权重压到 4bit、访存降至 FP16 的 1/4；劣势是权重仅16档有精度风险，激活动态引入归约开销。

- **与 [W16A16S 量化](term_w16a16s.md)（同为高精度基线，收益来源不同）**
  - 本模式：收益来自权重位宽压缩（访存减半）。
  - W16A16S：权重与激活均不量化，收益来自承载的稀疏权重（跳过零值元素），是去除位宽量化后的稀疏对照基线。

### 适用场景与限制

#### 1. 适用场景

- **激活精度敏感层**：激活分布离群值多、量化后精度劣化明显的层，可单独回退为 W8A16。
- **权重访存主导的 decode 场景（逐个生成 token 的推理阶段）**：权重压缩减半访存，同时保住激活精度。
- **精度优先的部署**：需要确定性精度的场景，W8A16 是介于不量化与 W8A8 之间的稳妥选择。

#### 2. 使用限制

- **无整数 GEMM 加速**：激活保持 FP16，矩阵乘仍为浮点，计算吞吐提升有限，只省权重访存。
- **压缩比低于 W8A8**：激活未压缩，整体位宽收益低于权重/激活同时量化的方案。

---

## 3. 关联流程

- 《[一键量化完整指南](../../../user_guide/usage_quick_quantization.md)》：通过 `--quant_type` 或 YAML 配置选择本模式并执行量化。
- 《[量化精度调优指南](../../../user_guide/process_quantization_precision_tuning.md)》：本模式导致的精度劣化可通过该流程逐层回退与调优。

---

## 4. 关联词条

- [量化模式](../README.md)：上位概念，本词条属于[线性层量化](README.md)类别，是该类别下的一种具体模式。
- [W8A8 静态量化](term_w8a8_static.md)：对比模式，激活亦量化为 INT8、换取整数 GEMM 加速。
- [W4A8 动态量化](term_w4a8_dynamic.md)：对比模式，权重压到 4bit 的混合位宽方案。
- [W16A16S 量化](term_w16a16s.md)：同类模式，完全不量化的稀疏基线。
- 《[线性量化算法说明](../../quantization_algorithms/linear_quant/linear_quant.md)》：配套术语，本模式的处理器实现。
- 《[MinMax：最小最大值量化算法说明](../../quantization_algorithms/minmax/minmax.md)》：配套术语，静态量化常用的量化参数统计方法。

---

## 5. 参考文档

1. 《[线性量化算法说明](../../quantization_algorithms/linear_quant/linear_quant.md)》
2. 《[量化模式](../README.md)》
