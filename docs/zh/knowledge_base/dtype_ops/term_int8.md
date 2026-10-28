# INT8

> **词条类别**：量化数据类型（[量化数据类型](README.md)）<br>
> **英文名称**：INT8<br>
> **应用领域**：权重、激活、KVCache 量化<br>

## 1. 概述

**INT8** 是8位有符号整数，取值 $[-128, 127]$，共256个均匀档位。它把浮点张量压到一半访存宽度，并对接整数 [GEMM](../quantization_mode/term_gemm.md)。量化方式是仿射量化：浮点经 scale 以及可选 zero-point 映射到整数网格，按 [IEEE 754-2019](https://ieeexplore.ieee.org/document/8766229) 的 `roundTiesToEven` 舍入后饱和到 $[-128, 127]$。函数用法见《[量化数据类型 ops 使用流程指南](usage_dtype_ops.md)》。

---

## 2. 词条介绍

### 2.1 编码

$$q_{\min} = -2^{b-1} = -128, \qquad q_{\max} = 2^{b-1} - 1 = 127, \qquad b = 8$$

- $b$：位宽。
- $q_{\min}$、$q_{\max}$：有符号整数的端点。

### 2.2 仿射公式

$$q = \mathrm{clamp}\left(\mathrm{round}\left(\frac{x}{s}\right) + z, \; q_{\min}, \; q_{\max}\right), \qquad \hat{x} = (q - z) \cdot s$$

- $x$：待量化浮点值。
- $s$：scale，一个整数档位对应的浮点步长。
- $z$：zero-point；对称量化时为 $0$。
- $q$：量化后的整数。
- $\hat{x}$：反量化还原值。

**对称**（$z=0$）：

$$s = \frac{\max(-x_{\min}, x_{\max})}{q_{\max}}, \qquad q = \mathrm{clamp}\left(\mathrm{round}\left(\frac{x}{s}\right), \; q_{\min}, \; q_{\max}\right), \qquad \hat{x} = q \cdot s$$

**非对称**：

$$s = \frac{x_{\max} - x_{\min}}{2^b - 1}, \qquad z = \mathrm{round}\left(\frac{-x_{\min}}{s}\right) + q_{\min}$$

- $x_{\min}$、$x_{\max}$：统计区间的下界与上界。量化区间强制包含 $0$，以保证整数 zero-point 落在 $[q_{\min}, q_{\max}]$ 内。

舍入采用 IEEE / OCP 必选的 `roundTiesToEven`；超范围饱和，并保留符号。

### 2.3 与相关格式的关系

- 相对 [FP16 / BF16](term_fp16_bf16.md)：访存减半，但引入均匀分档误差。
- 相对 [INT4](term_int4.md)：公式相同，只是 $b=8$；分辨率更高，压缩比更低。
- 相对 [FP8](term_fp8.md)：同为 8bit，INT8 是均匀整数网格，FP8 保留浮点指数，对离群值更稳。

### 2.4 适用场景与限制

- **适用**：权重、激活、KVCache，例如 [W8A8 静态量化](../quantization_mode/linear_layer_quantization/term_w8a8_static.md)、[W8A8 动态量化](../quantization_mode/linear_layer_quantization/term_w8a8_dynamic.md)、[KVCache-PerChannel 量化](../quantization_mode/kv_cache_quantization/term_kv_cache_perchannel.md)。
- **限制**：均匀分档对离群值敏感，常配合 [SmoothQuant](../quantization_algorithms/smooth_quant/term_smooth_quant.md)。非对称要多存一份 offset。

---

## 3. 关联流程

- 《[量化数据类型 ops 使用流程指南](usage_dtype_ops.md)》：本类型函数的调用步骤。
- 《[一键量化使用指南](../../user_guide/usage_one_click_quantization.md)》：量化任务的执行入口。
- 《[量化精度调优指南](../../user_guide/process_quantization_precision_tuning.md)》：本格式导致的精度问题可通过该流程排查。

---

## 4. 关联词条

- [量化数据类型](README.md)：上位概念，本词条所属目录。
- [量化与反量化](term_quantization.md)：前置术语，说明仿射公式、scale 与 zero-point。
- [FP16 / BF16](term_fp16_bf16.md)：对比术语，未量化 16bit 基准。
- [INT4](term_int4.md)：同类整数格式，位宽减半。
- [FP8](term_fp8.md)：同类 8bit 格式，逐元素浮点对照。
- [GEMM](../quantization_mode/term_gemm.md)：配套术语，整数矩阵乘。
- [W8A8 静态量化](../quantization_mode/linear_layer_quantization/term_w8a8_static.md)：应用对象，线性层采用本格式。

---

## 5. 参考文档

1. IEEE. "IEEE Standard for Floating-Point Arithmetic." IEEE 754-2019, 2019. <https://ieeexplore.ieee.org/document/8766229>
2. Jacob B, et al. "Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference." CVPR, 2018. <https://arxiv.org/abs/1712.05877>
3. Open Compute Project. "OCP Microscaling Formats (MX) Specification." Version 1.0, 2023. <https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf>
