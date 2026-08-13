# W8A8 FP8 动态量化

> **词条类别**：量化数据格式（[线性层量化](../README.md)）
> **英文名称**：W8A8 FP8 Dynamic Quantization
> **应用领域**：大语言模型量化压缩、推理加速
> **承载 IR 类**：`WFP8AFP8DynamicPerChannelFakeQuantLinear`（[`msmodelslim/ir/w8a8_fp_dynamic.py`](../../../../../msmodelslim/ir/w8a8_fp_dynamic.py)）

---

## 1. 概述

W8A8 FP8 动态量化 = **[FP8（E4M3）](../../quantization_basic/term_fp8.md) 版本**的 W8A8 动态量化：权重与激活都以 [FP8（E4M3）](../../quantization_basic/term_fp8.md) 存储，激活量化参数逐 token 在线计算。与 [INT8](../../quantization_basic/term_int8.md) 相比，E4M3 的 4bit 指数保留了浮点动态范围，对「数值跨度大、离群值多」的激活更耐受，无需先做离群值抑制；位宽同为 8bit，访存收益与 [INT8](../../quantization_basic/term_int8.md) 一致。代价是必须依赖支持 [FP8](../../quantization_basic/term_fp8.md) [GEMM](../term_gemm.md) 的硬件算子。模式名按家族 `W{位宽}A{位宽} + 数据类型` 记法，`W8A8 FP8` 表示 W 与 A 均为 8bit、数据类型为 [FP8（E4M3）](../../quantization_basic/term_fp8.md)。

---

## 2. 词条介绍

### 模式规格

| 维度 | 取值 | 说明 |
|------|------|------|
| 量化对象 | 权重 W、激活 A | 线性层矩阵乘两侧的张量 |
| 位宽 | W 8bit，A 8bit | 同为 8bit，访存减半 |
| 数据类型 | FP8（E4M3） | 浮点格式，1bit 符号 + 4bit 指数 + 3bit 尾数，max 值 448 |
| 参数获取方式 | 激活动态 / 权重静态 | 激活逐 token 在线求 scale；权重 scale 在量化阶段固化 |
| 量化粒度 | 权重 per-channel；激活 per-token | 权重逐输出通道共享 scale；激活逐 token 共享 scale |
| 对称性 | 对称 | 仅 scale，无 offset |

### 量化公式

FP8（E4M3）量化的 scale 按 FP8 可表示最大值确定：
$$q = \mathrm{round}_{FP8}(x / s), \qquad \hat{x} = q \cdot s, \qquad s = \frac{\max(|x|)}{448}$$

其中 $x$ 为待量化张量，$q$ 为量化后的 FP8 值（$\mathrm{round}_{FP8}$ 表示舍入到 E4M3 可表示的最近值，最大有限值 448），$\hat{x}$ 为反量化还原值，$s$ 为 scale。量化参数的获取方式与作用粒度见「模式规格」表。

### 与其他模式的关系

- **与 [W8A8 动态量化](term_w8a8_dynamic.md)（同为 8bit 动态家族，数据类型不同）**
  - 本模式（FP8 E4M3）：优势是浮点格式、4bit 指数保留动态范围，对宽动态范围与离群值比 INT8 均匀分档更耐受；劣势是 FP8 GEMM 的硬件支持面窄于 INT8，算子生态不成熟。
  - INT8：优势是整数 GEMM 生态成熟、硬件支持广泛；劣势是均匀分档对离群值敏感，宽动态范围激活常需先做 [SmoothQuant](../../quantization_algorithms/smooth_quant/smooth_quant.md) 等抑制。

- **与 [W8A8 MX 动态量化](term_w8a8_mx_dynamic.md)（同为 FP8 家族，格式与粒度不同）**
  - 本模式：优势是 per-channel/per-token 粒度、逐元素 E4M3，粒度更细、无需块对齐；劣势是每个通道/token 都需额外记录 scale。
  - MXFP8：优势是块级共享 E8M0 指数，scale 存储开销低、数值表达更紧凑；劣势是需硬件 MX 支持、块大小（32元素）需对齐。

- **与 [W8A16 静态量化](term_w8a16_static.md)（高精度基线）**
  - 本模式：优势是激活亦量化、可走低精度 GEMM，压缩彻底；劣势是激活引入 FP8 量化误差。
  - W8A16：优势是激活 FP16 零误差；劣势是权重反量化回浮点执行、无低精度 GEMM 加速。

### 适用场景与限制

#### 1. 适用场景

- **支持 FP8 的硬件部署**：目标硬件原生支持 FP8 GEMM 的场景。
- **宽动态范围激活**：激活数值跨度大、离群值多的模型，FP8 浮点格式比 INT8 均匀分档更耐受。

#### 2. 使用限制

- **硬件算子依赖**：FP8 GEMM 需目标硬件算子库支持，否则只能软件模拟、收益打折。
- **溢出边界**：E4M3 表示范围有限（max 448），超出范围的值需依赖 scale 缩放兜底，极端分布仍需验证。
- **在线归约开销**：per-token 动态量化有少量 min/max 归约开销。

---

## 3. 关联流程

- 《[一键量化完整指南](../../../user_guide/usage_quick_quantization.md)》：通过 `--quant_type` 或 YAML 配置选择本模式并执行量化。
- 《[量化精度调优指南](../../../user_guide/process_quantization_precision_tuning.md)》：本模式导致的精度劣化可通过该流程逐层回退与调优。

---

## 4. 关联词条

- [量化模式](../README.md)：上位概念，本词条属于[线性层量化](README.md)类别，是该类别下的一种具体模式。
- [W8A8 动态量化](term_w8a8_dynamic.md)：对比模式，INT8 数据类型的同构方案。
- [W8A8 MX 动态量化](term_w8a8_mx_dynamic.md)：同类模式，块级共享指数的 MXFP8 方案。
- [W8A16 静态量化](term_w8a16_static.md)：对比模式，激活保持 16bit 的高精度基线。
- [FA PerHead 量化](../fa_quantization/term_fa_perhead.md)：配套模式，同样使用 FP8 数据类型的注意力激活量化。
- 《[线性量化算法说明](../../quantization_algorithms/linear_quant/linear_quant.md)》：配套术语，本模式的处理器实现。

---

## 5. 参考文档

1. Kuzmin A et al. FP8 Formats for Deep Learning. arXiv:2209.05433. https://arxiv.org/abs/2209.05433
2. 《[线性量化算法说明](../../quantization_algorithms/linear_quant/linear_quant.md)》
3. 《[量化模式](../README.md)》
