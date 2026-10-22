# INT4

> **词条类别**：量化数据格式（[量化数据格式](README.md)）<br>
> **英文名称**：INT4<br>
> **应用领域**：权重量化、超低比特压缩<br>

## 1. 概述

**INT4** 是4位有符号整数，取值 $[-8, 7]$，共16个均匀档位。压缩比相对 [INT8](term_int8.md) 再翻一倍，通常只用于权重，并配合分组与重构（[GPTQ](../quantization_algorithms/gptq/term_gptq.md)、[LAOS](../quantization_algorithms/laos/term_laos.md)）。公式与 INT8 相同，只是 $b=4$。函数用法见《[量化数据格式 ops 使用流程指南](usage_dtype_ops.md)》。

---

## 2. 词条介绍

### 2.1 编码

$$q_{\min} = -2^{b-1} = -8, \qquad q_{\max} = 2^{b-1} - 1 = 7, \qquad b = 4$$

- $b$：位宽。
- $q_{\min}$、$q_{\max}$：有符号整数的端点。

### 2.2 仿射公式

与 [INT8](term_int8.md) 同源。

**对称**：$s=\max(-x_{\min},x_{\max})/7$，$q=\mathrm{clamp}(\mathrm{round}(x/s),-8,7)$，$\hat{x}=q\cdot s$。

**非对称**：$s=(x_{\max}-x_{\min})/15$，$z=\mathrm{round}(-x_{\min}/s)+q_{\min}$。

- $x$：待量化浮点值。
- $s$：scale。
- $z$：zero-point；对称时为 $0$。
- $x_{\min}$、$x_{\max}$：统计区间，强制包含 $0$。

舍入采用 `roundTiesToEven`，饱和到 $[-8, 7]$。

### 2.3 与相关格式的关系

- 相对 [INT8](term_int8.md)：公式相同、位宽减半，分辨率从 256 档降到 16 档。
- 相对 [MXFP4](term_mxfp4.md)：同为 4bit；INT4 是均匀整数网格，MXFP4 用块级共享指数保留浮点动态范围。

### 2.4 适用场景与限制

- **适用**：权重量化，例如 [W4A8 动态量化](../quantization_mode/linear_layer_quantization/term_w4a8_dynamic.md)、[W4A4 动态量化](../quantization_mode/linear_layer_quantization/term_w4a4_dynamic.md)。精度敏感时配合 GPTQ / LAOS。
- **限制**：只有16个电平，一般不用于激活；对离群值比 [MXFP4](term_mxfp4.md) 更敏感。

---

## 3. 关联流程

- 《[量化数据格式 ops 使用流程指南](usage_dtype_ops.md)》：本格式函数的调用步骤。
- 《[一键量化使用指南](../../user_guide/usage_one_click_quantization.md)》：量化任务的执行入口。
- 《[量化精度调优指南](../../user_guide/process_quantization_precision_tuning.md)》：本格式导致的精度问题可通过该流程排查。

---

## 4. 关联词条

- [量化数据格式](README.md)：上位概念，本词条所属目录。
- [量化与反量化](term_quantization.md)：前置术语，说明仿射公式与分组粒度。
- [INT8](term_int8.md)：同类整数格式，公式相同、位宽不同。
- [MXFP4](term_mxfp4.md)：同类 4bit 格式，块级共享指数对照。
- [GEMM](../quantization_mode/term_gemm.md)：配套术语，低比特矩阵乘。
- [W4A8 动态量化](../quantization_mode/linear_layer_quantization/term_w4a8_dynamic.md)：应用对象，线性层权重采用本格式。

---

## 5. 参考文档

1. IEEE. "IEEE Standard for Floating-Point Arithmetic." IEEE 754-2019, 2019. <https://ieeexplore.ieee.org/document/8766229>
2. Jacob B, et al. "Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference." CVPR, 2018. <https://arxiv.org/abs/1712.05877>
3. Frantar E, et al. "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers." arXiv preprint arXiv:2210.17323, 2022. <https://arxiv.org/abs/2210.17323>
