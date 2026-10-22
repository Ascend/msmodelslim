# 量化与反量化

> **词条类别**：量化数据格式（[量化数据格式](README.md)）<br>
> **英文名称**：Quantization<br>
> **应用领域**：大语言模型量化压缩、推理加速、KVCache 压缩<br>

## 1. 概述

**量化**把高精度浮点张量映射到低精度张量，用更少的比特保存同一批数据；**反量化**是它的逆过程。本词条说明量化公式、scale / zero_point、静态与动态、粒度和对称性。各数据格式如何落到这些公式上，见 [INT8](term_int8.md)、[INT4](term_int4.md)、[FP8](term_fp8.md)、[MXFP8](term_mxfp8.md)、[MXFP4](term_mxfp4.md)。

---

## 2. 词条介绍

### 2.1 量化公式

对浮点值 $x$，量化过程为：

$$q = \mathrm{round}\left(\frac{x}{scale}\right) + zero\_point$$

- $x$：待量化浮点值。
- $scale$：缩放系数，一个整数档位对应的浮点步长。
- $zero\_point$：零点，非对称量化引入的可选偏移。
- $q$：量化后的整数。

**例子**：某张量的值都在 $[0, 12]$，用非对称 INT8（整数范围 $-128\sim127$）量化，取 $scale=12/255\approx0.047$、$zero\_point=-128$：

- $x=12 \rightarrow q=\mathrm{round}(12/0.047)+(-128)=255-128=127$
- $x=6 \rightarrow q=\mathrm{round}(6/0.047)+(-128)\approx128-128=0$
- $x=0 \rightarrow q=\mathrm{round}(0)+(-128)=-128$

### 2.2 反量化公式

$$x \approx (q - zero\_point) \times scale$$

- $q$：量化整数。
- $zero\_point$：零点。
- $scale$：缩放系数。
- 还原值是对原浮点 $x$ 的近似。

如上例 $q=0$ 还原为 $(0-(-128))\times0.047\approx6.02$，接近原值 6。量化必然带来信息损失，误差来自 `round` 的舍入与有限位宽对数值范围的截断。

### 2.3 为什么要量化

- **减小内存与访存带宽**：权重从 [FP16 / BF16](term_fp16_bf16.md)（2字节/元素）降为 [INT8](term_int8.md)（1字节/元素），权重占用与读取带宽减半；KVCache 同理。
- **使能低精度矩阵运算**：权重与激活都量化成整数后，矩阵乘可走整数/低精度算子（见 [GEMM](../quantization_mode/term_gemm.md)），在专用硬件上有计算收益。
- **压缩 KVCache 显存**：量化缓存的历史 K/V，显著降低长上下文推理的显存占用。

### 2.4 量化参数：scale 与 zero_point

量化参数决定浮点值到整数档位的映射：

- **scale（缩放系数）**：1个整数档位代表的浮点步长，$scale=\frac{数值范围}{整数档位数}$。数值范围由被量化张量或其局部片段（如某个通道、某一组元素）的实际 min/max 决定。
- **zero_point（零点）**：非对称量化中对应浮点 0 的整数偏移；对称量化中为 0，可省去。

### 2.5 参数获取方式：静态与动态

量化参数的**获取方式**分两类：

- **静态量化**：离线用一批校准数据统计出 scale/zero_point，推理时直接使用（参数固化）。在线开销为零，但依赖校准集对真实分布的刻画。
- **动态量化**：推理时在线统计每个张量（或每行）的实际范围并实时计算 scale。免校准、对分布变化适应好，但每次多一次统计归约的开销。

### 2.6 量化粒度

量化粒度指**一份量化参数覆盖的元素范围**。粒度越细，数值分布刻画越准、精度越好，但量化参数数量与计算开销越大。常见的量化粒度和特点如下：

- **per-tensor**：整个张量共享一份 scale。
- **per-channel**：按输出通道（如权重 W 的 out_dim 列）各一份参数，常用于权重。
- **per-group**：按固定大小分组（如 128个元素）各一份参数，如 [GPTQ](../quantization_algorithms/gptq/term_gptq.md)/[LAOS](../quantization_algorithms/laos/term_laos.md) 的分组量化。
- **per-token**：按输入行（每个 token）各一份参数，常用于激活的动态量化。
- **per-head**：按注意力头各一份参数，用于注意力相关张量。
- **per-block**：按固定大小分块（如 [MXFP8](term_mxfp8.md)/[MXFP4](term_mxfp4.md) 的 32元素）共享一个块级指数，是 MX 格式的专用粒度；与 per-group 逐组记 scale 不同，per-block 共享的是指数。

### 2.7 对称性与非对称性

- **对称量化**：数值以 0 为中心，只记录 scale、无 zero_point。实现简单，适合分布近似对称的权重。
- **非对称量化**：记录 scale 与 zero_point，可覆盖整体偏移的数值，动态范围利用率更高。

---

## 3. 关联流程

- 《[量化数据格式 ops 使用流程指南](usage_dtype_ops.md)》：把本词条公式落到函数调用。
- 《[一键量化使用指南](../../user_guide/usage_one_click_quantization.md)》：量化任务的执行入口。
- 《[量化精度调优指南](../../user_guide/process_quantization_precision_tuning.md)》：数据类型与量化方案导致的精度问题可通过该流程排查与调优。

---

## 4. 关联词条

- [量化数据格式](README.md)：上位概念，本词条所属目录。
- [FP16 / BF16](term_fp16_bf16.md)：配套术语，量化前的 16bit 基准。
- [GEMM](../quantization_mode/term_gemm.md)：配套术语，量化是整数 GEMM 的前提。
- [INT8](term_int8.md)：配套术语，量化最常用的目标格式。
- [量化模式](../quantization_mode/README.md)：上层概念，量化/反量化在各模式中的具体实施。

---

## 5. 参考文档

1. IEEE. "IEEE Standard for Floating-Point Arithmetic." IEEE 754-2019, 2019. <https://ieeexplore.ieee.org/document/8766229>
2. Jacob B, et al. "Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference." CVPR, 2018. <https://arxiv.org/abs/1712.05877>
