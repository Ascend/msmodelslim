# W8A8 动态量化

> **词条类别**：量化数据格式（[线性层量化](../README.md)）
> **英文名称**：W8A8 Dynamic Quantization
> **应用领域**：大语言模型量化压缩、推理加速
> **承载 IR 类**：`W8A8DynamicPerChannelFakeQuantLinear` / `W8A8DynamicPerGroupFakeQuantLinear`（[`msmodelslim/ir/w8a8_dynamic.py`](../../../../../msmodelslim/ir/w8a8_dynamic.py)）

---

## 1. 概述

W8A8 动态量化 = 对线性层的权重与激活都做 [INT8](../../quantization_basic/term_int8.md) 量化，其中**激活的量化参数在推理时逐 token 在线计算**。它与 [W8A8 静态量化](term_w8a8_static.md) 位宽完全相同，唯一区别是参数获取方式：「静态」离线校准固化、「动态」在线统计。动态方案免校准、逐 token 更贴合自身数值范围，代价是每次前向多一次 min/max 归约。

---

## 2. 词条介绍

### 模式规格

| 维度 | 取值 | 说明 |
|------|------|------|
| 量化对象 | 权重 W、激活 A | 线性层矩阵乘两侧的张量 |
| 位宽 | W 8bit，A 8bit | 同为 8bit，访存减半 |
| 数据类型 | INT8 | 整数格式，可走 INT8 整数 GEMM |
| 参数获取方式 | 激活动态 / 权重静态 | 激活对每个 token 在线求 min/max 并计算 scale，免校准、随输入自适应；权重 scale 在量化阶段固化 |
| 量化粒度 | 权重 per-channel/per-group；激活 per-token | 权重逐输出通道或按固定分组（如 128元素）共享 scale；激活逐 token（一行）共享 scale |
| 对称性 | 激活对称；权重对称（per-group 支持非对称） | 激活对称仅 scale；权重 per-group 可加 offset |

### 量化公式

量化与反量化采用标准线性映射。对称量化（仅 scale、无零点）：
$$q = \mathrm{round}(x / s), \qquad \hat{x} = q \cdot s, \qquad s = \frac{\max(|x|)}{2^{b-1}}$$

非对称量化（含零点 offset）：
$$q = \mathrm{round}(x / s) + z, \qquad \hat{x} = (q - z) \cdot s, \qquad s = \frac{\max(x) - \min(x)}{2^b - 1}$$

其中 $x$ 为待量化张量，$q$ 为量化后的整数值，$\hat{x}$ 为反量化还原值，$s$ 为 scale，$z$ 为零点 offset，$b$ 为位宽（INT8 取 $b=8$，INT4 取 $b=4$），且 $z = \mathrm{round}(-\min(x)/s)$。量化参数 $s$、$z$ 的获取方式（静态校准或在线动态）与作用粒度（per-channel、per-token、per-tensor 等）见「模式规格」表。

### 与其他模式的关系

- **与 [W8A8 静态量化](term_w8a8_static.md)（位宽相同，参数获取方式不同）**
  - 本模式（动态）：优势是逐 token 在线算 scale、免校准，激活量化贴合每个 token 的数值范围，精度高于静态 per-tensor，对输入分布漂移鲁棒；劣势是每次前向多一次 min/max 归约，带来少量延迟开销。
  - 静态：优势是激活 scale 离线固化、推理零在线开销；劣势是 per-tensor 粗粒度依赖校准数据，精度上限较低。

- **与 [W8A8 PD-Mix 量化](term_w8a8_pdmix.md)（同为 W8A8，激活策略随阶段不同）**
  - 本模式：全阶段 per-token 动态，激活精度统一；劣势是 decode 阶段每步只有一个 token——激活即整张量，per-token 与 per-tensor 粒度等价，per-token 归约既无统计意义、开销又属多余。
  - PD-Mix：prefill 动态保精度、decode 退回静态零开销，两阶段各自最优；劣势是依赖推理框架的阶段感知与切换，当前仅限 MindIE 部署。

- **与 [W4A4 动态量化](term_w4a4_dynamic.md)（同为动态家族，位宽不同）**
  - 本模式：优势是 8bit 分辨率高、精度更稳，是通用选择；劣势是权重访存仅减半（FP16 的 1/2）。
  - W4A4：优势是权重访存降至 FP16 的 1/4、压缩更彻底；劣势是仅16档、精度风险高，需更细粒度与离群抑制兜底。

### 适用场景与限制

#### 1. 适用场景

- **校准数据不足或分布不确定**：免校准，适合无代表性校准集、或部署时输入分布变化大的场景。
- **精度要求高于静态**：需要比静态 W8A8 更好的激活精度，且可接受少量在线归约开销。

#### 2. 使用限制

- **在线归约开销**：per-token min/max 归约增加少量延迟，对每 token 计算量极小的层占比更明显。
- **硬件算子依赖**：per-token 动态量化需推理框架/算子库支持逐 token 参数计算，否则只能软件模拟、收益打折。

---

## 3. 关联流程

- 《[一键量化完整指南](../../../user_guide/usage_quick_quantization.md)》：通过 `--quant_type` 或 YAML 配置选择本模式并执行量化。
- 《[量化精度调优指南](../../../user_guide/process_quantization_precision_tuning.md)》：本模式导致的精度劣化可通过该流程逐层回退与调优。

---

## 4. 关联词条

- [量化模式](../README.md)：上位概念，本词条属于[线性层量化](README.md)类别，是该类别下的一种具体模式。
- [W8A8 静态量化](term_w8a8_static.md)：对比模式，激活参数离线固化。
- [W8A8 PD-Mix 量化](term_w8a8_pdmix.md)：同类模式，激活策略随 Prefill/Decode 阶段切换。
- [W4A4 动态量化](term_w4a4_dynamic.md)：同类模式，更低比特的权重+激活量化。
- [FA PerToken 量化](../fa_quantization/term_fa_pertoken.md)：同类模式，把 per-token 动态量化应用于注意力 Q/K/V 激活。
- 《[线性量化算法说明](../../quantization_algorithms/linear_quant/linear_quant.md)》：配套术语，本模式的处理器实现。

---

## 5. 参考文档

1. 《[线性量化算法说明](../../quantization_algorithms/linear_quant/linear_quant.md)》
2. 《[量化模式](../README.md)》
