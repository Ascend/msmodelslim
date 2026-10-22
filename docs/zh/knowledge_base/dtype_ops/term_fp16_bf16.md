# FP16 / BF16

> **词条类别**：量化数据格式（[量化数据格式](README.md)）<br>
> **英文名称**：FP16 / BF16<br>
> **应用领域**：权重与激活的未量化基准、高精度保留场景<br>

## 1. 概述

**FP16** 与 **BF16** 都是16位浮点格式，每个元素占2字节。它们是权重和激活常见的未量化格式，也是量化时「减半 / 1/4」这类对比的基准：例如 [INT8](term_int8.md) 的访存减半，就是相对它们而言。FP16 尾数更细，但可表示范围远小于 FP32（最大有限值约 65504）。BF16 的指数范围与 FP32 相同，尾数更粗。量化模式常把它们当作不量化的高精度基线，例如 [W8A16 静态量化](../quantization_mode/linear_layer_quantization/term_w8a16_static.md) 的激活保持 FP16。

---

## 2. 词条介绍

### 2.1 定义

- **FP16**：1符号 + 5指数 + 10尾数。可表示范围远小于 FP32（最大有限值约 65504），10位尾数保证较高的数值精度。
- **BF16**：1符号 + 8指数 + 7尾数。指数范围与 FP32 完全相同（可表示范围约 $3.4\times10^{38}$），但尾数仅 7位，局部精度低于 FP16；适合需要大动态范围、可牺牲局部精度的训练/推理场景。

两者位宽相同，都是2字节，占用和访存一致，差别只在动态范围和局部精度。相比8位格式（[INT8](term_int8.md)、[FP8](term_fp8.md)），它们占2字节，通常作为量化前的存储和带宽基线。

### 2.2 使用场景

- **未量化基线**：量化前的权重/激活默认为 16位浮点（FP16 或 BF16，均 2字节），各类量化的压缩比、带宽收益都以它为基准衡量。
- **高精度保留层**：对量化误差特别敏感的激活（如 [W8A16 静态量化](../quantization_mode/linear_layer_quantization/term_w8a16_static.md) 的激活）保持 FP16，不参与量化。
- **FP16 与 BF16 的选择**：需要大动态范围用 BF16，需要局部精度用 FP16。训练与推理的数值格式通常保持一致——训练一般用 BF16（动态范围与 FP32 相同），推理因此也默认 BF16。

### 2.3 与相关类型对比

- 相比 [INT8](term_int8.md)：FP16/BF16 是未量化浮点，零量化误差，但占 2字节，访存与计算开销翻倍；INT8 以一半位宽换取一半存储/带宽。
- 相比 [FP8（E4M3）](term_fp8.md)：FP16/BF16 动态范围更大（BF16 与 FP32 相同），精度更稳，但位宽翻倍，压缩收益更小。
- 相比 [MXFP8](term_mxfp8.md)：FP16/BF16 每个元素独立完整表示，无块级共享参数；MXFP8 用块级共享指数在 8位内保留浮点动态范围。

---

## 3. 关联流程

- 《[量化数据格式 ops 使用流程指南](usage_dtype_ops.md)》：量化前的 16bit 张量作为本目录函数的输入。
- 《[一键量化使用指南](../../user_guide/usage_one_click_quantization.md)》：量化任务的执行入口。
- 《[量化精度调优指南](../../user_guide/process_quantization_precision_tuning.md)》：数据类型与量化方案导致的精度问题可通过该流程排查与调优。

---

## 4. 关联词条

- [量化数据格式](README.md)：上位概念，本词条所属目录。
- [量化与反量化](term_quantization.md)：配套术语，量化公式以 FP16/BF16 张量为输入。
- [INT8](term_int8.md)：配套术语，INT8 的压缩收益以本格式为基准衡量。
- [FP8](term_fp8.md)：同类 8bit 浮点格式。
- [MXFP8](term_mxfp8.md)：块级共享指数的 8bit 格式。
- [W8A16 静态量化](../quantization_mode/linear_layer_quantization/term_w8a16_static.md)：配套模式，激活保持 FP16 不量化。

---

## 5. 参考文档

1. IEEE. "IEEE Standard for Floating-Point Arithmetic." IEEE 754-2019, 2019. <https://ieeexplore.ieee.org/document/8766229>
2. Kalamkar D, et al. "A Study of BFLOAT16 for Deep Learning Training." arXiv preprint arXiv:1905.12322, 2019. <https://arxiv.org/abs/1905.12322>
3. 《[量化模式](../quantization_mode/README.md)》
