# 数据类型：MXFP8 / MXFP4 量化术语百科词条

> **词条类别**：量化基础概念（[量化基础](../README.md)）
> **英文名称**：MXFP8 / MXFP4（Microscaling Formats）
> **应用领域**：权重、激活量化，超低比特压缩

---

## 1. 概述

**MX**（Microscaling，微缩放）是 OCP（Open Compute Project，开放计算项目）定义的一类**块级共享指数**格式：每 **32个元素**为一块，整块共享一个 **E8M0 指数**（8位指数、0位尾数），元素只保存相对该指数的低位数据。块内元素共享同一数量级，从而以少量位宽覆盖大动态范围，缩放参数开销也摊薄到每32元素一份。

---

## 2. 词条介绍

### MXFP8

- **结构**：块共享 E8M0 指数 + 元素为 **E4M3**（1符号 + 4指数 + 3尾数）。
- **特点**：8位位宽、访存为 [FP16/BF16](term_fp16_bf16.md) 一半，同时保留浮点动态范围；相比 INT8 均匀分档对离群值更耐受。
- **应用**：如 [W8A8 MX 动态量化](../quantization_mode/linear_layer_quantization/term_w8a8_mx_dynamic.md)。

### MXFP4

- **结构**：块共享 E8M0 指数 + 元素为 **E2M1**（1符号 + 2指数 + 1尾数，emax=2、max 6）。
- **特点**：4位位宽、压缩比高，块级共享指数保留了浮点动态范围；对离群值比 INT4 均匀分档更耐受。
- **应用**：如 [W4A4 MX 动态量化](../quantization_mode/linear_layer_quantization/term_w4a4_mx_dynamic.md)、[W4A4 MX 双 Scale 量化](../quantization_mode/linear_layer_quantization/term_w4a4_mx_dualscale.md)。

### 与相关类型对比

- 相比 [FP8（E4M3）](term_fp8.md)：MXFP8 块级共享指数，缩放参数开销从每元素一份摊薄到每32元素一份；代价是块粒度较粗、需硬件 MX 支持。
- 相比 [INT4](term_int4.md)：MXFP4 用块级共享指数保留动态范围、对离群值更耐受；代价是依赖 MX 硬件、最后维需 32 对齐。

---

## 3. 关联流程

- 《[一键量化完整指南](../../user_guide/usage_quick_quantization.md)》：量化任务的执行入口。
- 《[量化精度调优指南](../../user_guide/process_quantization_precision_tuning.md)》：数据类型与量化方案导致的精度问题可通过该流程排查与调优。

---

## 4. 关联词条

- [量化基础](README.md)：上位概念，本词条所属目录。
- [量化与反量化](term_quantization.md)：配套术语，MX 量化的数学机制。
- [GEMM](../quantization_mode/term_gemm.md)：配套术语，MX 输入走低精度浮点 GEMM。
- [W8A8 MX 动态量化](../quantization_mode/linear_layer_quantization/term_w8a8_mx_dynamic.md)：配套模式，MXFP8 的典型实现。
- [W4A4 MX 动态量化](../quantization_mode/linear_layer_quantization/term_w4a4_mx_dynamic.md)：配套模式，MXFP4 的典型实现。

---

## 5. 参考文档

1. 《[量化模式](../quantization_mode/README.md)》
