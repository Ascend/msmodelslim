# 数据类型：INT4 量化术语百科词条

> **词条类别**：量化基础概念（[量化基础](../README.md)）
> **英文名称**：INT4（4-bit Integer）
> **应用领域**：权重量化、超低比特压缩

---

## 1. 概述

**INT4** 是 4位有符号整数，范围 $-8\sim7$，0.5字节/元素。相比 [INT8](term_int8.md) 压缩更大（权重为 [FP16/BF16](term_fp16_bf16.md) 的 1/4），但分辨率低，通常配合分组量化与专门的重构算法控制精度损失。

---

## 2. 词条介绍

### 定义

INT4 占 4位，可表示 $-8\sim7$ 共 16个整数档位。按[量化公式](term_quantization.md)映射到这些档位后存储，即为 INT4 量化。

### 使用场景

- **权重**：如 [W4A8 动态量化](../quantization_mode/linear_layer_quantization/term_w4a8_dynamic.md) 的 INT4 权重、[W4A4 动态量化](../quantization_mode/linear_layer_quantization/term_w4a4_dynamic.md)。
- **配合重构算法**：INT4 精度风险高，常配合分组量化与权重重构（如 [GPTQ](../quantization_algorithms/gptq/gptq.md)、[LAOS](../quantization_algorithms/laos/laos.md)）降低损失。

### 与相关类型对比

- 相比 [INT8](term_int8.md)：位宽减半、压缩更大；但分辨率低、精度风险高，一般只用于权重、不用于激活。
- 相比 [MXFP4](term_mxfp.md)：INT4 均匀分档无浮点动态范围，对离群值更敏感；MXFP4 用块级共享指数保留动态范围。

---

## 3. 关联流程

- 《[一键量化完整指南](../../user_guide/usage_quick_quantization.md)》：量化任务的执行入口。
- 《[量化精度调优指南](../../user_guide/process_quantization_precision_tuning.md)》：数据类型与量化方案导致的精度问题可通过该流程排查与调优。

---

## 4. 关联词条

- [量化基础](README.md)：上位概念，本词条所属目录。
- [量化与反量化](term_quantization.md)：配套术语，INT4 量化的数学机制。
- [GEMM](../quantization_mode/term_gemm.md)：配套术语，INT4 输入可走整数 GEMM。
- [W4A8 动态量化](../quantization_mode/linear_layer_quantization/term_w4a8_dynamic.md)：配套模式，INT4 权重 + INT8 激活的典型实现。

---

## 5. 参考文档

1. Frantar E et al. GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers. arXiv:2210.17323. https://arxiv.org/abs/2210.17323
2. 《[量化模式](../quantization_mode/README.md)》
