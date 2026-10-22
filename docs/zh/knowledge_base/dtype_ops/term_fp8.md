# FP8

> **词条类别**：量化数据格式（[量化数据格式](README.md)）<br>
> **英文名称**：FP8<br>
> **应用领域**：权重、激活量化<br>

## 1. 概述

**FP8** 遵循 OCP OFP8：每个元素独立编码，没有块级共享指数。两种元素格式里，E4M3 精度更高（上界 448），E5M2 动态范围更宽（上界 57344）。位宽与 [INT8](term_int8.md) 相同，但对离群值更稳。标准只定义对称路径（实数 scale、没有 zero-point）。函数用法见《[量化数据格式 ops 使用流程指南](usage_dtype_ops.md)》。

---

## 2. 词条介绍

### 2.1 编码

| 格式 | 符号 | 指数 | 尾数 | $e_{\max}$ | $\mathrm{max\_norm}$ |
|------|------|------|------|-----------|----------------------|
| E4M3 | 1 | 4 | 3 | 8 | $256\times 1.75=448$ |
| E5M2 | 1 | 5 | 2 | 15 | $2^{15}\times 1.5=57344$ |

- $e_{\max}$：最大指数。
- $\mathrm{max\_norm}$：有限值上界，E4M3 为 448，E5M2 为 57344。

### 2.2 量化公式

$$s = \frac{\mathrm{amax}}{\mathrm{max\_norm}}, \qquad x' = \frac{x}{s}, \qquad q = \mathrm{roundTiesToEven\_to\_FP8}(x'), \qquad q = \mathrm{clamp}(q, \pm\mathrm{max\_norm}), \qquad \hat{x} = q \cdot s$$

- $x$：待量化浮点值。
- $\mathrm{amax}$：沿量化轴统计的绝对值最大值。
- $s$：实数 scale。
- $x'$：按 scale 归一化后的值。
- $q$：落到 FP8 网格后的量化值。
- $\hat{x}$：反量化还原值。

元素舍入（OCP OFP8）：

$$\mathrm{min\_exp} = 2 - 2^{e_{\mathrm{bits}}-1}, \qquad a \leftarrow \mathrm{round}\left(\frac{a}{2^{e_{\mathrm{priv}}}} \cdot 2^{m-2}\right) / 2^{m-2} \cdot 2^{e_{\mathrm{priv}}}$$

- $e_{\mathrm{bits}}$：指数位数。
- $m$：含隐含位的尾数位数。
- $e_{\mathrm{priv}}=\lfloor\log_2|a|\rfloor$，先 clamp 到 $\mathrm{min\_exp}$。
- 无 zero-point。

### 2.3 与相关格式的关系

- 相对 [INT8](term_int8.md)：同为 8bit；FP8 保留指数，对离群值更稳，依赖 FP8 GEMM。
- 相对 [MXFP8](term_mxfp8.md)：元素编码相同，但 MXFP8 每32个元素共享一个 E8M0，scale 开销更低。

### 2.4 适用场景与限制

- **适用**：权重与激活，例如 [W8A8 FP8 动态量化](../quantization_mode/linear_layer_quantization/term_w8a8_fp8_dynamic.md)。离群值较多时优先 E4M3，分布更宽时用 E5M2。
- **限制**：需要硬件 FP8 GEMM。E4M3 上界是 448，更大的值靠 scale 缩放。只有对称路径。

---

## 3. 关联流程

- 《[量化数据格式 ops 使用流程指南](usage_dtype_ops.md)》：本格式函数的调用步骤，含 `fmt` 默认 `e4m3`。
- 《[一键量化使用指南](../../user_guide/usage_one_click_quantization.md)》：量化任务的执行入口。
- 《[量化精度调优指南](../../user_guide/process_quantization_precision_tuning.md)》：本格式导致的精度问题可通过该流程排查。

---

## 4. 关联词条

- [量化数据格式](README.md)：上位概念，本词条所属目录。
- [量化与反量化](term_quantization.md)：前置术语，说明 scale 与对称量化。
- [INT8](term_int8.md)：同类 8bit 格式，整数仿射对照。
- [MXFP8](term_mxfp8.md)：同类 8bit 浮点格式，块级共享指数。
- [GEMM](../quantization_mode/term_gemm.md)：配套术语，低精度矩阵乘。
- [W8A8 FP8 动态量化](../quantization_mode/linear_layer_quantization/term_w8a8_fp8_dynamic.md)：应用对象，线性层采用本格式。

---

## 5. 参考文档

1. Open Compute Project. "OCP 8-bit Floating Point Specification (OFP8)." Revision 1.0, 2023. <https://www.opencompute.org/documents/ocp-8-bit-floating-point-specification-ofp8-revision-1-0-2023-12-01-pdf-1>
2. Kuzmin A, et al. "FP8 Formats for Deep Learning." arXiv preprint arXiv:2209.05433, 2022. <https://arxiv.org/abs/2209.05433>
3. IEEE. "IEEE Standard for Floating-Point Arithmetic." IEEE 754-2019, 2019. <https://ieeexplore.ieee.org/document/8766229>
