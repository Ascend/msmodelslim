# W4A4 动态量化

> **词条类别**：量化数据格式（[线性层量化](../README.md)）
> **英文名称**：W4A4 Dynamic Quantization
> **应用领域**：大语言模型量化压缩、推理加速
> **承载 IR 类**：`W4A4DynamicPerChannelFakeQuantLinear` / `W4A4DynamicPerGroupFakeQuantLinear`（[`msmodelslim/ir/w4a4_dynamic.py`](../../../../../msmodelslim/ir/w4a4_dynamic.py)）

---

## 1. 概述

W4A4 动态量化 = 对线性层的权重与激活都做 [INT4](../../quantization_basic/term_int4.md) 量化，激活量化参数逐 token 在线计算。它是压缩比最高的一类线性层方案：权重访存降至 [FP16/BF16](../../quantization_basic/term_fp16_bf16.md) 的 1/4；但 4bit 仅16个量化档位，必须依赖 per-token 动态量化与 per-channel/per-group 细粒度兜底，才能把精度损失压到可接受范围。

---

## 2. 词条介绍

### 模式规格

| 维度 | 取值 | 说明 |
|------|------|------|
| 量化对象 | 权重 W、激活 A | 线性层矩阵乘两侧的张量 |
| 位宽 | W 4bit，A 4bit | 同为 4bit，权重访存降至 [FP16/BF16](../../quantization_basic/term_fp16_bf16.md) 的 1/4、INT8 的 1/2 |
| 数据类型 | INT4 | 整数格式（内部以 INT8 承载存储） |
| 参数获取方式 | 激活动态 / 权重静态 | 激活逐 token 在线求 min/max 并计算 scale；权重 scale/offset 在量化阶段固化 |
| 量化粒度 | 权重 per-channel/per-group；激活 per-token | 权重逐输出通道或按固定分组（如 128元素）共享 scale/offset；激活逐 token 共享 scale |
| 对称性 | 激活对称；权重对称或非对称 | 激活对称仅 scale；权重可加 offset 适配非对称分布 |

### 量化公式

量化与反量化采用标准线性映射。对称量化（仅 scale、无零点）：
$$q = \mathrm{round}(x / s), \qquad \hat{x} = q \cdot s, \qquad s = \frac{\max(|x|)}{2^{b-1}}$$

非对称量化（含零点 offset）：
$$q = \mathrm{round}(x / s) + z, \qquad \hat{x} = (q - z) \cdot s, \qquad s = \frac{\max(x) - \min(x)}{2^b - 1}$$

其中 $x$ 为待量化张量，$q$ 为量化后的整数值，$\hat{x}$ 为反量化还原值，$s$ 为 scale，$z$ 为零点 offset，$b$ 为位宽（INT8 取 $b=8$，INT4 取 $b=4$），且 $z = \mathrm{round}(-\min(x)/s)$。量化参数 $s$、$z$ 的获取方式（静态校准或在线动态）与作用粒度（per-channel、per-token、per-tensor 等）见「模式规格」表。

### 与其他模式的关系

- **与 [W8A8 静态量化](term_w8a8_static.md)（位宽不同）**
  - 本模式：优势是权重访存降至 [FP16/BF16](../../quantization_basic/term_fp16_bf16.md) 的 1/4，压缩与带宽收益最大化；劣势是仅16个量化档位、精度风险高，需动态量化 + per-group 粒度 + 离群值抑制兜底。
  - W8A8：优势是 8bit 分辨率高、精度更稳，是通用部署基线；劣势是权重访存仅减半。

- **与 [W8A8 动态量化](term_w8a8_dynamic.md)（同为动态家族，位宽不同）**
  - 本模式：优势是压缩更狠、计算位宽更低；劣势是激活 4bit 对分布极其敏感，精度风险显著高于 8bit。
  - W8A8 动态：优势是 8bit 精度更稳、在线归约与整体开销较低；劣势是压缩幅度小。

- **与 [W4A4 MX 动态量化](term_w4a4_mx_dynamic.md)（同为 W4A4，数据类型不同）**
  - 本模式（INT4）：优势是整数格式、硬件算子生态成熟；劣势是均匀分档对离群值敏感，动态范围受限。
  - MXFP4：优势是块级共享指数保留浮点动态范围，对离群值更耐受；劣势是依赖硬件 MX 支持、块粒度需对齐。

### 适用场景与限制

#### 1. 适用场景

- **高压缩比部署**：显存/带宽受限、追求极致压缩的场景，如大规模服务端多模型部署。
- **对精度不敏感的任务**：任务本身对量化噪声容忍度高的场景。

#### 2. 使用限制

- **精度门槛高**：4bit 激活量化精度风险大，通常需要配合离群值抑制（如 [SmoothQuant](../../quantization_algorithms/smooth_quant/smooth_quant.md)）与精度调优方可上线。
- **硬件算子依赖**：INT4 整数 GEMM 需目标硬件算子库支持，否则无法兑现计算收益。

---

## 3. 关联流程

- 《[一键量化完整指南](../../../user_guide/usage_quick_quantization.md)》：通过 `--quant_type` 或 YAML 配置选择本模式并执行量化。
- 《[量化精度调优指南](../../../user_guide/process_quantization_precision_tuning.md)》：本模式导致的精度劣化可通过该流程逐层回退与调优。

---

## 4. 关联词条

- [量化模式](../README.md)：上位概念，本词条属于[线性层量化](README.md)类别，是该类别下的一种具体模式。
- [W8A8 动态量化](term_w8a8_dynamic.md)：对比模式，INT8 位宽、精度更稳。
- [W8A8 静态量化](term_w8a8_static.md)：对比模式，INT8 静态方案、精度基线。
- [W4A4 MX 动态量化](term_w4a4_mx_dynamic.md)：同类模式，改用 MXFP4 浮点格式。
- [W8A16 静态量化](term_w8a16_static.md)：对比模式，激活保持 16bit 的高精度方案。
- 《[线性量化算法说明](../../quantization_algorithms/linear_quant/linear_quant.md)》：配套术语，本模式的处理器实现。
- 《[LAOS：w4a4量化方案说明](../../quantization_algorithms/laos/laos.md)》：配套术语，面向 W4A4 的精度优化算法。

---

## 5. 参考文档

1. 《[线性量化算法说明](../../quantization_algorithms/linear_quant/linear_quant.md)》
2. 《[量化模式](../README.md)》
