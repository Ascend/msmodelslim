# W4A8 动态量化

> **词条类别**：量化数据格式（[线性层量化](README.md)）
> **英文名称**：W4A8 Dynamic Quantization
> **应用领域**：大语言模型量化压缩、推理加速
> **承载 IR 类**：`W4A8DynamicFakeQuantLinear`（[`msmodelslim/ir/w4a8_dynamic.py`](../../../../../msmodelslim/ir/w4a8_dynamic.py)）

---

## 1. 概述

W4A8 动态量化 = 权重压到 [INT4](../../quantization_basic/term_int4.md)、激活保持 [INT8](../../quantization_basic/term_int8.md) 的**混合位宽**线性层方案，激活量化参数逐 token 在线计算。它在压缩与精度之间取折中：权重是每 token 都要整体读取的静态数据，压到 4bit 把权重访存降至 [FP16/BF16](../../quantization_basic/term_fp16_bf16.md) 的 1/4；激活是量化误差最敏感的一方，保留 8bit 分辨率并逐 token 动态量化，是「权重使劲压、激活保精度」的典型配置。

---

## 2. 词条介绍

### 模式规格

| 维度 | 取值 | 说明 |
|------|------|------|
| 量化对象 | 权重 W、激活 A | 线性层矩阵乘两侧的张量 |
| 位宽 | W 4bit，A 8bit | 混合位宽，权重访存降至 [FP16/BF16](../../quantization_basic/term_fp16_bf16.md) 的 1/4、激活降至 1/2 |
| 数据类型 | 权重 INT4；激活 INT8 | 整数格式，权重以 INT8 容器打包存储（2个 INT4 装入 1字节） |
| 参数获取方式 | 激活动态 / 权重静态 | 激活逐 token 在线求 min/max 并计算 scale；权重 scale 在量化阶段固化 |
| 量化粒度 | 权重 per-channel；激活 per-token | 权重逐输出通道共享 scale；激活逐 token 共享 scale |
| 对称性 | 对称 | 仅 scale，无 offset |

### 量化公式

量化与反量化采用标准线性映射。对称量化（仅 scale、无零点）：
$$q = \mathrm{round}(x / s), \qquad \hat{x} = q \cdot s, \qquad s = \frac{\max(|x|)}{2^{b-1}}$$

非对称量化（含零点 offset）：
$$q = \mathrm{round}(x / s) + z, \qquad \hat{x} = (q - z) \cdot s, \qquad s = \frac{\max(x) - \min(x)}{2^b - 1}$$

其中 $x$ 为待量化张量，$q$ 为量化后的整数值，$\hat{x}$ 为反量化还原值，$s$ 为 scale，$z$ 为零点 offset，$b$ 为位宽（INT8 取 $b=8$，INT4 取 $b=4$），且 $z = \mathrm{round}(-\min(x)/s)$。量化参数 $s$、$z$ 的获取方式（静态校准或在线动态）与作用粒度（per-channel、per-token、per-tensor 等）见「模式规格」表。

### 与其他模式的关系

- **与 [W8A8 动态量化](term_w8a8_dynamic.md)（激活同为动态，权重位宽不同）**
  - 本模式：优势是权重压到 4bit、访存降至 FP16 的 1/4，权重侧压缩更彻底；劣势是权重仅16档、有精度风险，且权重/激活位宽不一致时算子需混合处理。
  - W8A8 动态：优势是权重 8bit 分辨率高、精度更稳，算子实现简单；劣势是权重访存仅减半。

- **与 [W4A4 动态量化](term_w4a4_dynamic.md)（权重同为 4bit，激活位宽不同）**
  - 本模式：优势是激活保持 8bit、远离最敏感的激活量化误差，精度风险远低于 W4A4；劣势是激活访存只减半，压缩幅度小于全 4bit。
  - W4A4：优势是压缩比最大化、激活位宽同降；劣势是 4bit 激活精度风险极高，需要更多兜底手段。

- **与 [W8A16 静态量化](term_w8a16_static.md)（高精度基线）**
  - 本模式：优势是权重、激活均量化，压缩远比 W8A16 彻底（权重 4bit + 激活 8bit）；劣势是激活引入量化误差，权重/激活反量化回浮点执行时计算加速有限，收益主要来自访存。
  - W8A16：优势是激活 FP16 零误差、精度最稳；劣势是权重仅压到 8bit、且需反量化回浮点执行，无低精度 GEMM 加速。

- **与 [W4A8 MX 动态量化](term_w4a8_mx_dynamic.md)（同为 W4A8，数据类型与粒度不同）**
  - 本模式（INT4 per-channel）：优势是整数格式、硬件算子生态成熟；劣势是均匀分档对权重离群值敏感，per-channel 粒度需逐输出通道记 scale。
  - MXFP4：优势是块级共享指数保留浮点动态范围、对离群值更耐受；劣势是依赖硬件 MX 支持、块大小需对齐。

### 适用场景与限制

#### 1. 适用场景

- **权重内存/带宽主导的部署**：模型权重是主要显存占用，decode 时（逐个生成 token 的推理阶段）权重读取是吞吐瓶颈的场景。
- **精度与压缩折中**：需要比 W8A8 更强的权重压缩，又不敢把激活压到 4bit 的场景。

#### 2. 使用限制

- **权重精度风险**：4bit 权重仅16档，对离群值敏感，需配合离群值抑制或更细分组。
- **混合位宽算子依赖**：INT4×INT8 混合位宽 GEMM 需要目标硬件算子库支持，否则收益打折。

---

## 3. 关联流程

- 《[一键量化完整指南](../../../user_guide/usage_quick_quantization.md)》：通过 `--quant_type` 或 YAML 配置选择本模式并执行量化。
- 《[量化精度调优指南](../../../user_guide/process_quantization_precision_tuning.md)》：本模式导致的精度劣化可通过该流程逐层回退与调优。

---

## 4. 关联词条

- [量化模式](../README.md)：上位概念，本词条属于[线性层量化](README.md)类别，是该类别下的一种具体模式。
- [W8A8 动态量化](term_w8a8_dynamic.md)：对比模式，权重升至 8bit、精度更稳。
- [W4A4 动态量化](term_w4a4_dynamic.md)：对比模式，激活压到 4bit、压缩更狠。
- [W8A16 静态量化](term_w8a16_static.md)：对比模式，激活不量化的高精度基线。
- [W4A8 MX 动态量化](term_w4a8_mx_dynamic.md)：同类模式，改用 MX 浮点格式承载同一位宽。
- 《[线性量化算法说明](../../quantization_algorithms/linear_quant/usage_linear_quant.md)》：配套术语，本模式的处理器实现。

---

## 5. 参考文档

1. 《[线性量化算法说明](../../quantization_algorithms/linear_quant/usage_linear_quant.md)》
2. 《[量化模式](../README.md)》
