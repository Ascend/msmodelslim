# MXFP8

> **词条类别**：量化数据格式（[量化数据格式](README.md)）<br>
> **英文名称**：MXFP8<br>
> **应用领域**：权重、激活量化<br>

## 1. 概述

**MXFP8** 是 OCP MX v1.0 的8位块级格式：每32个元素共享一个 E8M0 指数（2 的幂），块内元素为 E4M3 或 E5M2。对离群值比 [INT8](term_int8.md) 更稳，scale 开销比逐元素 [FP8](term_fp8.md) 更低。公式按 OCP MX §6.3。只有对称路径。函数用法见《[量化数据格式 ops 使用流程指南](usage_dtype_ops.md)》。

---

## 2. 词条介绍

### 2.1 编码

- 块长 $k=32$。
- E8M0：$e\in[-127,127]$；编码为无符号 8bit 时写作 $e+127$，$0\mathrm{xFF}$ 保留为 NaN。
- 元素：默认 E4M3（$e_{\max}=8$，$\mathrm{max\_norm}=448$），可选 E5M2（$e_{\max}=15$，$\mathrm{max\_norm}=57344$）。

| 元素 | 指数位 | 尾数位（含隐含） | $e_{\max}$ | $\mathrm{max\_norm}$ | $\mathrm{min\_exp}$ |
|------|--------|------------------|-----------|----------------------|---------------------|
| E4M3 | 4 | 5 | 8 | 448 | $-6$ |
| E5M2 | 5 | 4 | 15 | 57344 | $-14$ |

### 2.2 量化公式

$$e_{\mathrm{shared}} = \mathrm{clamp}\left(\lfloor \log_2 \max_{i} |x_i| \rfloor - e_{\max}, \; -127, \; 127\right)$$

$$x'_i = x_i / 2^{e_{\mathrm{shared}}}, \qquad e_{\mathrm{priv},i} = \lfloor \log_2 |x'_i| \rfloor$$

$$q_i = \mathrm{clamp}(\mathrm{roundTiesToEven\_to\_FP8}(x'_i), \pm\mathrm{max\_norm}), \qquad \hat{x}_i = q_i \cdot 2^{e_{\mathrm{shared}}}$$

- $x_i$：块内第 $i$ 个浮点元素。
- $e_{\mathrm{shared}}$：块级共享指数。
- $e_{\max}$：元素格式的最大指数，E4M3 为 8，E5M2 为 15。
- $x'_i$：按共享指数归一化后的值。
- $q_i$：块内 FP8 量化值。
- $\hat{x}_i$：反量化还原值。

减去 $e_{\max}$，是为了让块内最大元素落到元素格式的上沿。全零块的 $e_{\mathrm{shared}}=-127$ 是合法值。元素舍入与 [FP8](term_fp8.md) 相同。没有 zero-point。

### 2.3 与相关格式的关系

- 相对 [FP8](term_fp8.md)：元素网格相同，但 MXFP8 用块级 E8M0 代替逐元素实数 scale。
- 相对 [INT8](term_int8.md)：同为 8bit；MX 保留浮点动态范围，对离群值更稳。
- 相对 [MXFP4](term_mxfp4.md)：共享指数机制相同，元素改为 E2M1。

### 2.4 适用场景与限制

- **适用**：离群值较多、又希望摊薄 scale 开销的 8bit 部署，例如 [W8A8 MX 动态量化](../quantization_mode/linear_layer_quantization/term_w8a8_mx_dynamic.md)。
- **限制**：收益依赖硬件 MX 算子。只有对称路径。

---

## 3. 关联流程

- 《[量化数据格式 ops 使用流程指南](usage_dtype_ops.md)》：本格式函数的调用步骤，含 `fmt` 默认 `e4m3`。
- 《[一键量化使用指南](../../user_guide/usage_one_click_quantization.md)》：量化任务的执行入口。
- 《[量化精度调优指南](../../user_guide/process_quantization_precision_tuning.md)》：本格式导致的精度问题可通过该流程排查。

---

## 4. 关联词条

- [量化数据格式](README.md)：上位概念，本词条所属目录。
- [量化与反量化](term_quantization.md)：前置术语，说明 scale 与块粒度。
- [FP8](term_fp8.md)：同类 8bit 浮点格式，逐元素对照。
- [MXFP4](term_mxfp4.md)：同类 MX 格式，元素为 E2M1。
- [INT8](term_int8.md)：同类 8bit 格式，整数仿射对照。
- [GEMM](../quantization_mode/term_gemm.md)：配套术语，低精度矩阵乘。
- [W8A8 MX 动态量化](../quantization_mode/linear_layer_quantization/term_w8a8_mx_dynamic.md)：应用对象，线性层采用本格式。

---

## 5. 参考文档

1. Open Compute Project. "OCP Microscaling Formats (MX) Specification." Version 1.0, 2023. <https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf>
2. Open Compute Project. "OCP 8-bit Floating Point Specification (OFP8)." Revision 1.0, 2023. <https://www.opencompute.org/documents/ocp-8-bit-floating-point-specification-ofp8-revision-1-0-2023-12-01-pdf-1>
3. IEEE. "IEEE Standard for Floating-Point Arithmetic." IEEE 754-2019, 2019. <https://ieeexplore.ieee.org/document/8766229>
