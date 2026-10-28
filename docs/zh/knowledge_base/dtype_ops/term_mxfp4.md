# MXFP4

> **词条类别**：量化数据类型（[量化数据类型](README.md)）<br>
> **英文名称**：MXFP4<br>
> **应用领域**：权重、激活超低比特压缩<br>

## 1. 概述

**MXFP4** 是 [OCP MX v1.0](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf) 的4位块级格式：每32个元素共享一个 E8M0，块内元素为 E2M1（$e_{\max}=2$，$\mathrm{max\_norm}=6$）。在 4bit 下仍保留浮点动态范围，对离群值比 [INT4](term_int4.md) 更稳。公式与 [MXFP8](term_mxfp8.md) 相同，只是元素参数不同。只有对称路径。函数用法见《[量化数据类型 ops 使用流程指南](usage_dtype_ops.md)》。

---

## 2. 词条介绍

### 2.1 编码

- 块长 $k=32$；E8M0 同 [MXFP8](term_mxfp8.md)。
- 元素仅 E2M1：1 符号 + 2 指数 + 1 尾数。

$$\mathrm{max\_norm} = 2^{2}\cdot\tfrac{3}{2}=6, \qquad \mathrm{min\_exp}=2-2^{1}=0$$

- $\mathrm{max\_norm}$：E2M1 有限值上界，为 6。
- $\mathrm{min\_exp}$：最小指数，为 0。

| 元素 | 指数位 | 尾数位（含隐含） | $e_{\max}$ | $\mathrm{max\_norm}$ |
|------|--------|------------------|-----------|----------------------|
| E2M1 | 2 | 3 | 2 | 6 |

### 2.2 量化公式

与 [MXFP8](term_mxfp8.md) 相同，仅 $e_{\max}=2$、$\mathrm{max\_norm}=6$：

$$e_{\mathrm{shared}}=\mathrm{clamp}(\lfloor\log_2\max|x|\rfloor-2,-127,127)$$

$$x'=x/2^{e_{\mathrm{shared}}},\quad q=\mathrm{clamp}(\mathrm{roundTiesToEven\_to\_E2M1}(x'),\pm 6),\quad \hat{x}=q\cdot 2^{e_{\mathrm{shared}}}$$

- $x$：待量化浮点值。
- $e_{\mathrm{shared}}$：块级共享指数。
- $x'$：按共享指数归一化后的值。
- $q$：E2M1 量化值。
- $\hat{x}$：反量化还原值。

### 2.3 与相关格式的关系

- 相对 [MXFP8](term_mxfp8.md)：共享指数机制相同，元素从 E4M3/E5M2 换成 E2M1。
- 相对 [INT4](term_int4.md)：同为 4bit；MXFP4 保留浮点动态范围，对离群值更稳。

### 2.4 适用场景与限制

- **适用**：压缩比要求很高的场景，例如 [W4A4 MX 动态量化](../quantization_mode/linear_layer_quantization/term_w4a4_mx_dynamic.md)、[W4A4 MX 双 Scale 量化](../quantization_mode/linear_layer_quantization/term_w4a4_mx_dualscale.md)。
- **限制**：依赖硬件 MX 算子。4bit 精度门槛高。只有对称路径。

---

## 3. 关联流程

- 《[量化数据类型 ops 使用流程指南](usage_dtype_ops.md)》：本类型函数的调用步骤。
- 《[一键量化使用指南](../../user_guide/usage_one_click_quantization.md)》：量化任务的执行入口。
- 《[量化精度调优指南](../../user_guide/process_quantization_precision_tuning.md)》：本格式导致的精度问题可通过该流程排查。

---

## 4. 关联词条

- [量化数据类型](README.md)：上位概念，本词条所属目录。
- [量化与反量化](term_quantization.md)：前置术语，说明块粒度与对称量化。
- [MXFP8](term_mxfp8.md)：同类 MX 格式，公式相同、元素为 E4M3/E5M2。
- [INT4](term_int4.md)：同类 4bit 格式，整数仿射对照。
- [GEMM](../quantization_mode/term_gemm.md)：配套术语，低比特矩阵乘。
- [W4A4 MX 动态量化](../quantization_mode/linear_layer_quantization/term_w4a4_mx_dynamic.md)：应用对象，线性层采用本格式。

---

## 5. 参考文档

1. Open Compute Project. "OCP Microscaling Formats (MX) Specification." Version 1.0, 2023. <https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf>
2. IEEE. "IEEE Standard for Floating-Point Arithmetic." IEEE 754-2019, 2019. <https://ieeexplore.ieee.org/document/8766229>
