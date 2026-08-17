# W8A8 静态量化

> **词条类别**：量化数据格式（[线性层量化](README.md)）
> **英文名称**：W8A8 Static Quantization
> **应用领域**：大语言模型量化压缩、推理加速
> **承载 IR 类**：`W8A8StaticFakeQuantLinear`（[`msmodelslim/ir/w8a8_static.py`](../../../../../msmodelslim/ir/w8a8_static.py)）

---

## 1. 概述

W8A8 静态量化 = 对线性层的权重与激活都做 [INT8](../../quantization_basic/term_int8.md) 静态量化。模式名按[量化模式](../README.md)的命名法则解码：W8A8 表示权重（W）与激活（A）均为 8bit（[INT8](../../quantization_basic/term_int8.md)）；「静态」指量化参数（scale/offset）在离线校准阶段确定并固化，推理时直接使用、零在线开销。它是[线性层量化](README.md)中最基础、应用最广的线性层量化模式。

---

## 2. 词条介绍

### 模式规格

| 维度 | 取值 | 说明 |
|------|------|------|
| 量化对象 | 权重 W、激活 A | 线性层矩阵乘两侧的张量；权重是静态参数、每个 token 前向都整体读取，激活是逐 token 生成的中间张量 |
| 位宽 | W 8bit，A 8bit | 两者同为 8bit，访存减半，可走 INT8 整数 GEMM |
| 数据类型 | INT8 | 整数格式，权重 1字节/元素；decode 阶段（逐个生成 token 的推理阶段）权重访存占用减半 |
| 参数获取方式 | 静态 | scale/offset 由校准集统计（如 minmax）后固化，推理零在线统计开销；精度依赖校准集与真实分布的匹配 |
| 量化粒度 | 权重 per-channel；激活 per-tensor | 权重每输出通道共享一个 scale，适配通道间数值差异；激活整张量共享 scale/offset，参数开销最小 |
| 对称性 | 权重对称；激活对称或非对称 | 权重分布天然对称免 offset；激活分布常非对称，需 scale + 零点 offset |

### 量化公式

量化与反量化采用标准线性映射。对称量化（仅 scale、无零点）：
$$q = \mathrm{round}(x / s), \qquad \hat{x} = q \cdot s, \qquad s = \frac{\max(|x|)}{2^{b-1}}$$

非对称量化（含零点 offset）：
$$q = \mathrm{round}(x / s) + z, \qquad \hat{x} = (q - z) \cdot s, \qquad s = \frac{\max(x) - \min(x)}{2^b - 1}$$

其中 $x$ 为待量化张量，$q$ 为量化后的整数值，$\hat{x}$ 为反量化还原值，$s$ 为 scale，$z$ 为零点 offset，$b$ 为位宽（INT8 取 $b=8$，INT4 取 $b=4$），且 $z = \mathrm{round}(-\min(x)/s)$。量化参数 $s$、$z$ 的获取方式（静态校准或在线动态）与作用粒度（per-channel、per-token、per-tensor 等）见「模式规格」表。

### 与其他模式的关系

- **与 [W8A8 动态量化](term_w8a8_dynamic.md)（位宽相同，参数获取方式不同）**
  - 本模式（静态）：优势是激活 scale 离线固化、推理零在线开销；劣势是激活为整张量共享参数的粗粒度，对逐 token 分布变化不敏感，精度上限低于动态方案。
  - 动态：优势是逐 token 在线算 scale、免校准，激活精度更高；劣势是每次前向多一次 min/max 归约，带来少量延迟开销。

- **与 [W8A16 静态量化](term_w8a16_static.md)（激活是否量化不同）**
  - 本模式：优势是激活亦量化为 INT8，可走 INT8 整数 GEMM，计算吞吐更高、压缩更彻底；劣势是激活引入量化误差，对存在离群值或量化敏感的层精度劣化。
  - W8A16：优势是激活保持 FP16、零量化误差，是激活精度最稳的基线之一；劣势是激活不量化意味着矩阵乘仍需将 INT8 权重反量化回浮点执行，无整数 GEMM 加速，性能较差，整体压缩低于 W8A8。

- **与 [W4A4 动态量化](term_w4a4_dynamic.md)（位宽不同）**
  - 本模式：优势是 8bit 分辨率高、精度更稳，是通用部署基线；劣势是权重访存仅减半（FP16 的 1/2）。
  - W4A4：优势是权重访存降至 FP16 的 1/4，压缩最彻底；劣势是仅16个量化档位、精度风险高，需动态量化 + per-group 粒度 + 离群值抑制兜底。

### 适用场景与限制

#### 1. 适用场景

- **通用部署基线**：精度与性能最均衡的成熟方案，适用于大多数大语言模型的默认量化。
- **带宽受限的 decode 阶段（逐个生成 token）**：每 token 全量读取权重，INT8 权重访存减半的收益最明显。

#### 2. 使用限制

- **依赖校准数据**：校准集分布偏差过大会导致 scale 失真、精度漂移。
- **激活离群值敏感**：per-tensor 静态量化在激活存在显著离群值时精度劣化，需先做离群值抑制（如 [SmoothQuant](../../quantization_algorithms/smooth_quant/term_smooth_quant.md)）或改用[动态量化](term_w8a8_dynamic.md)。

---

## 3. 关联流程

- 《[一键量化完整指南](../../../user_guide/usage_quick_quantization.md)》：通过 `--quant_type` 或 YAML 配置选择本模式并执行量化。
- 《[量化精度调优指南](../../../user_guide/process_quantization_precision_tuning.md)》：本模式导致的精度劣化可通过该流程逐层回退与调优。

---

## 4. 关联词条

- [量化模式](../README.md)：上位概念，本词条属于[线性层量化](README.md)类别，是该类别下的一种具体模式。
- [W8A8 动态量化](term_w8a8_dynamic.md)：对比模式，激活量化参数在线计算。
- [W8A16 静态量化](term_w8a16_static.md)：同类模式，激活保持 16bit。
- [W4A4 动态量化](term_w4a4_dynamic.md)：同类模式，更低比特的权重+激活量化。
- 《[线性量化算法说明](../../quantization_algorithms/linear_quant/usage_linear_quant.md)》：配套术语，本模式的处理器实现。
- 《[MinMax：最小最大值量化算法说明](../../quantization_algorithms/minmax/usage_minmax.md)》：配套术语，静态量化常用的量化参数统计方法。

---

## 5. 参考文档

1. 《[线性量化算法说明](../../quantization_algorithms/linear_quant/usage_linear_quant.md)》
2. 《[量化模式](../README.md)》
