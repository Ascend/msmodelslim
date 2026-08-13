# 数据类型：INT8 量化术语百科词条

> **词条类别**：量化基础概念（[量化基础](../README.md)）
> **英文名称**：INT8（8-bit Integer）
> **应用领域**：权重、激活、KVCache 量化

---

## 1. 概述

**INT8** 是 8位有符号整数，范围 $-128\sim127$，1字节/元素。它是量化的**主流格式**：存储与带宽为 [FP16/BF16](term_fp16_bf16.md) 的一半，且整数格式可走 [GEMM](../quantization_mode/term_gemm.md) 加速，权重、激活、KVCache 均可使用。

---

## 2. 词条介绍

### 定义

INT8 占 8位，可表示 $-128\sim127$ 共 256个整数档位。对浮点张量按[量化公式](term_quantization.md)映射到这些档位后存储，即为 INT8 量化。

### 使用场景

- **权重**：如 [W8A8 静态量化](../quantization_mode/linear_layer_quantization/term_w8a8_static.md)、[W8A8 动态量化](../quantization_mode/linear_layer_quantization/term_w8a8_dynamic.md) 的权重。
- **激活**：配合动态量化（per-token），如 [W8A8 动态量化](../quantization_mode/linear_layer_quantization/term_w8a8_dynamic.md) 的激活。
- **KVCache**：如 [KVCache-PerChannel 量化](../quantization_mode/kv_cache_quantization/term_kv_cache_perchannel.md)。

### 与相关类型对比

- 相比 [FP8（E4M3）](term_fp8.md)：INT8 是均匀分档、无浮点动态范围，对离群值更敏感；但硬件算子生态更成熟。
- 相比 [INT4](term_int4.md)：分辨率更高、精度更稳；但位宽翻倍、压缩幅度较小。

---

## 3. 关联流程

- 《[一键量化完整指南](../../user_guide/usage_quick_quantization.md)》：量化任务的执行入口。
- 《[量化精度调优指南](../../user_guide/process_quantization_precision_tuning.md)》：数据类型与量化方案导致的精度问题可通过该流程排查与调优。

---

## 4. 关联词条

- [量化基础](README.md)：上位概念，本词条所属目录。
- [量化与反量化](term_quantization.md)：配套术语，INT8 量化的数学机制。
- [数据类型 FP16/BF16](term_fp16_bf16.md)：配套术语，INT8 压缩收益的基准格式。
- [数据类型 FP8](term_fp8.md)：同类算法，同为 8bit 数据格式。
- [数据类型 INT4](term_int4.md)：同类算法，更低比特的整数格式。
- [GEMM](../quantization_mode/term_gemm.md)：配套术语，INT8 输入可走整数 GEMM。
- [W8A8 静态量化](../quantization_mode/linear_layer_quantization/term_w8a8_static.md)：配套模式，INT8 静态量化的典型实现。

---

## 5. 参考文档

1. 《[量化模式](../quantization_mode/README.md)》
