# 数据类型：FP8（E4M3） 量化术语百科词条

> **词条类别**：量化基础概念（[量化基础](../README.md)）
> **英文名称**：FP8（8-bit Floating Point，E4M3）
> **应用领域**：权重、激活量化

---

## 1. 概述

**FP8** 是 8位浮点格式，本词条介绍其中的 **E4M3** 变体：**1符号 + 4指数 + 3尾数**，可表示范围约 $\pm448$（emax=8）。它像整数一样只有 8位、访存为 [FP16/BF16](term_fp16_bf16.md) 的一半，同时保留浮点动态范围，适合分布跨度大、对整数舍入敏感的场景。

---

## 2. 词条介绍

### 定义

E4M3 的位分配为：1位符号 + 4位指数 + 3位尾数，共 8位。4位指数提供较大的动态范围（最大约 448），3位尾数保证基本精度。

### 使用场景

- **权重与激活**：如 [W8A8 FP8 动态量化](../quantization_mode/linear_layer_quantization/term_w8a8_fp8_dynamic.md)。
- **离群值敏感场景**：相比 INT8 均匀分档，E4M3 浮点格式对宽动态范围与离群值更耐受，无需先做离群值抑制。
- 代价是依赖支持 FP8 GEMM 的硬件算子，生态面窄于 INT8。

### 与相关类型对比

- 相比 [INT8](term_int8.md)：同为 8位、访存收益一致，但 E4M3 保留浮点动态范围、对离群值更耐受；代价是 FP8 硬件算子支持面窄。
- 相比 [MXFP8](term_mxfp.md)：FP8 每个元素独立完整表示；MXFP8 每32元素共享一个 E8M0 指数、缩放参数开销更低。

---

## 3. 关联流程

- 《[一键量化完整指南](../../user_guide/usage_quick_quantization.md)》：量化任务的执行入口。
- 《[量化精度调优指南](../../user_guide/process_quantization_precision_tuning.md)》：数据类型与量化方案导致的精度问题可通过该流程排查与调优。

---

## 4. 关联词条

- [量化基础](README.md)：上位概念，本词条所属目录。
- [量化与反量化](term_quantization.md)：配套术语，FP8 量化的数学机制。
- [数据类型 MXFP8/MXFP4](term_mxfp.md)：同类算法，同为 8bit 浮点格式。
- [GEMM](../quantization_mode/term_gemm.md)：配套术语，FP8 输入走低精度浮点 GEMM。
- [W8A8 FP8 动态量化](../quantization_mode/linear_layer_quantization/term_w8a8_fp8_dynamic.md)：配套模式，FP8 E4M3 的典型实现。

---

## 5. 参考文档

1. Kuzmin A et al. FP8 Formats for Deep Learning. arXiv:2209.05433. https://arxiv.org/abs/2209.05433
2. 《[量化模式](../quantization_mode/README.md)》
