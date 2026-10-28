# 量化数据类型

> **词条类别**：量化数据类型<br>
> **英文名称**：Quantization Data Type<br>
> **应用领域**：大语言模型量化压缩、推理加速、KVCache 压缩<br>

## 1. 概述

量化数据类型描述量化之后的数值本身：占多少比特、按什么编码、能表示多大范围、如何舍入和截断。本目录收录各类型的概念词条，以及一份函数用法指南。底层原子能力按类型对任意张量做量化与反量化，并带上粒度、静态或动态、对称性，不区分张量来自权重、激活还是 KV。INT8 / INT4 走仿射量化，舍入遵循 [IEEE 754-2019](https://ieeexplore.ieee.org/document/8766229)；FP8 遵循 [OCP OFP8](https://www.opencompute.org/documents/ocp-8-bit-floating-point-specification-ofp8-revision-1-0-2023-12-01-pdf-1)；MXFP8 / MXFP4 遵循 [OCP MX v1.0](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)；未量化基准 FP16 / BF16 遵循 [IEEE 754-2019](https://ieeexplore.ieee.org/document/8766229)。

[量化模式](../quantization_mode/README.md)首先区分量化对象（线性层 / KVCache / FA），再在对象上选用上述能力。不同对象、以及同一对象上不同的量化方式和数据类型，对应不同模式。例如线性层 8bit 可以是 [W8A8 静态量化](../quantization_mode/linear_layer_quantization/term_w8a8_static.md)（[INT8](term_int8.md)）、[W8A8 FP8 动态量化](../quantization_mode/linear_layer_quantization/term_w8a8_fp8_dynamic.md) 或 [W8A8 MX 动态量化](../quantization_mode/linear_layer_quantization/term_w8a8_mx_dynamic.md)。

阅读模式词条前，建议先看[量化与反量化](term_quantization.md)里的公式、scale 和粒度，以及未量化基准 [FP16 / BF16](term_fp16_bf16.md)。具体函数怎么调用见《[量化数据类型 ops 使用流程指南](usage_dtype_ops.md)》；接口清单见《[量化数据类型 ops Python API](../../api_reference/python_api/README.md)》。

---

## 2. 词条介绍

### 2.1 文档索引

| 文档 | 类型 | 说明 |
|------|------|------|
| [量化与反量化](term_quantization.md) | 词条 | scale、zero_point、静态与动态、粒度、对称性 |
| [FP16 / BF16](term_fp16_bf16.md) | 词条 | 未量化 16bit 基准 |
| [INT8](term_int8.md) | 词条 | 有符号整数仿射，取值 $[-128, 127]$ |
| [INT4](term_int4.md) | 词条 | 有符号整数仿射，取值 $[-8, 7]$ |
| [FP8](term_fp8.md) | 词条 | OCP OFP8，E4M3 / E5M2 |
| [MXFP8](term_mxfp8.md) | 词条 | OCP MX，E8M0 + E4M3/E5M2，块长 32 |
| [MXFP4](term_mxfp4.md) | 词条 | OCP MX，E8M0 + E2M1，块长 32 |
| 《[量化数据类型 ops 使用流程指南](usage_dtype_ops.md)》 | 流程指南 | 函数命名、粒度、`fmt`、`dtype` 与调用步骤 |
| 《[量化数据类型 ops Python API](../../api_reference/python_api/README.md)》 | Python API | `dtype_ops` 对外接口清单 |

### 2.2 三种编码家族

- **整数仿射**：[INT8](term_int8.md)、[INT4](term_int4.md)。均匀分档，有对称和非对称。
- **逐元素浮点**：[FP8](term_fp8.md)。只有对称，scale 是实数。
- **块级共享指数**：[MXFP8](term_mxfp8.md)、[MXFP4](term_mxfp4.md)。每32个元素共享一个 E8M0。只有对称。

---

## 3. 关联流程

- 《[量化数据类型 ops 使用流程指南](usage_dtype_ops.md)》：本目录函数的调用入口。
- 《[量化数据类型 ops Python API](../../api_reference/python_api/README.md)》：本目录对外接口清单。
- 《[一键量化使用指南](../../user_guide/usage_one_click_quantization.md)》：量化任务的执行入口。
- 《[量化精度调优指南](../../user_guide/process_quantization_precision_tuning.md)》：数据类型导致的精度问题可通过该流程排查。

---

## 4. 关联词条

- [量化模式](../quantization_mode/README.md)：上层概念，先按量化对象选用本目录类型。
- [线性层量化](../quantization_mode/linear_layer_quantization/README.md)：应用对象。
- [KVCache 量化](../quantization_mode/kv_cache_quantization/README.md)：应用对象。
- [FA 量化](../quantization_mode/fa_quantization/README.md)：应用对象。
- [GEMM](../quantization_mode/term_gemm.md)：配套术语，低精度矩阵乘。

---

## 5. 参考文档

1. Open Compute Project. "OCP Microscaling Formats (MX) Specification." Version 1.0, 2023. <https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf>
2. Open Compute Project. "OCP 8-bit Floating Point Specification (OFP8)." Revision 1.0, 2023. <https://www.opencompute.org/documents/ocp-8-bit-floating-point-specification-ofp8-revision-1-0-2023-12-01-pdf-1>
3. IEEE. "IEEE Standard for Floating-Point Arithmetic." IEEE 754-2019, 2019. <https://ieeexplore.ieee.org/document/8766229>
