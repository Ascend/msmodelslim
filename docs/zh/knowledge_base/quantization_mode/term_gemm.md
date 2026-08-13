# GEMM 量化术语百科词条

> **词条类别**：量化基础概念（[量化模式](../README.md)）
> **英文名称**：GEMM（General Matrix Multiply）
> **应用领域**：线性层计算、推理加速

---

## 1. 概述

**GEMM**（General Matrix Multiply，通用矩阵乘法）即矩阵乘 $C=A\times B$，是 Transformer 中[线性层](linear_layer_quantization/README.md)计算 $Y=X\cdot W$ 的数学形式。量化使 GEMM 能以整数/低精度输入运行，是量化的主要计算收益来源。

---

## 2. 词条介绍

### 定义

GEMM 是 $C=A\times B$ 形式的矩阵乘法。Transformer 中占比最高的计算就是 GEMM：Q/K/V 投影、注意力输出投影、MLP 三层都是线性层，各自做一次矩阵乘。

### 整数 GEMM

**整数 GEMM**（Integer GEMM）指输入为 [INT8](../quantization_basic/term_int8.md)/[INT4](../quantization_basic/term_int4.md) 等低精度整数、乘累加在整数域进行的矩阵乘：权重与激活都量化成整数后，全程整数计算，结果再乘回 scale 还原。整数 GEMM 可调用专用低精度算子，是量化的主要计算收益来源。

若只有一侧量化（如 [W8A16 静态量化](linear_layer_quantization/term_w8a16_static.md)），则需把整数反量化回浮点再做浮点 GEMM，没有整数 GEMM 的加速。

### 与量化的关系

- 前提是[量化与反量化](../quantization_basic/term_quantization.md)：权重与激活都量化成整数，才能进入整数域计算。
- 数据类型的选取（[INT8](../quantization_basic/term_int8.md)、[FP8](../quantization_basic/term_fp8.md)、[MXFP8/MXFP4](../quantization_basic/term_mxfp.md)）决定整数/低精度 GEMM 的硬件算子与精度表现。

---

## 3. 关联流程

- 《[一键量化完整指南](../../user_guide/usage_quick_quantization.md)》：量化任务的执行入口。
- 《[量化精度调优指南](../../user_guide/process_quantization_precision_tuning.md)》：数据类型与量化方案导致的精度问题可通过该流程排查与调优。

---

## 4. 关联词条

- [量化模式](README.md)：上位概念，本词条所属目录。
- [量化与反量化](../quantization_basic/term_quantization.md)：配套术语，整数 GEMM 的前提。
- [线性层量化](linear_layer_quantization/README.md)：下位概念，GEMM 的量化应用。
- [W8A8 静态量化](linear_layer_quantization/term_w8a8_static.md)：配套模式，整数 GEMM 的典型实现。

---

## 5. 参考文档

1. 《[量化模式](README.md)》
