# FA 量化

<!-- waiver: G01 原因：本文件为量化模式目录下的类别文档，不适用 term_<english_name>.md 词条命名 -->

> **词条类别**：量化数据格式（[量化模式](../README.md)）<br>
> **英文名称**：Flash Attention Quantization<br>
> **应用领域**：注意力计算加速、长上下文推理<br>
> **msModelSlim 实现**：见各模式词条的“承载 IR 类”字段<br>

<a id="overview"></a>

## 1. 概述

FA（Flash Attention）量化是对**送入 Flash Attention 的 Q、K、V 激活张量**进行量化的一类量化模式，属于[量化模式](../README.md)中的类别之一，是 [KVCache 量化](../kv_cache_quantization/README.md)的进阶：在量化 K/V（压缩缓存显存）的基础上，进一步量化 Q。

K/V 量化主要解决存储问题（缓存显存减半，支持更长上下文）。Q 加入量化后，注意力得分计算（Q 与 K 的点乘）与加权求和（softmax 后与 V 点乘）的参与张量均为低比特，可直接调用整数或低精度矩阵乘算子，从而在继承 KVCache 显存收益的同时使能低精度矩阵运算。典型如 [FA INT8 PerHead 量化](term_fa_int8_perhead.md)：对送入 Flash Attention 的 Q、K、V 按注意力头（per-head）静态量化到 INT8。

FA 量化**本质上是一个组合量化模式**：作用于注意力 Q/K/V 三分支，各分支选择一种激活值量化模式，三分支的组合构成**一个整体方案**，见[整体方案](#holistic-mode)。

---

<a id="mode-options"></a>

## 2. 词条介绍

### 2.1 该类一般的量化选择

- **量化对象**：送入 Flash Attention 的 Q/K/V 三个分支的激活张量（K/V 承接缓存压缩，Q 使能低精度得分计算）。
- **分支可选的量化粒度**：
  - **per-head**：按注意力头共享 scale，**静态**（需校准），如 [FA INT8 PerHead 量化](term_fa_int8_perhead.md)。
  - **per-token**：逐 token 在线计算 scale，**动态**（免校准），如 [FA INT8 动态量化](term_fa_int8_dynamic.md)、[FA FP8 动态量化](term_fa_fp8_dynamic.md)。
  - **per-block**：沿 head_dim 按 32 元素分块、块级共享指数，**动态**（免校准），如 [FA MXFP4 动态量化](term_fa_mxfp4_dynamic.md)。
- **数据类型**：INT8、FP8（E4M3）、MXFP8、MXFP4。
- **与 KV 的关系**：本类以 [KVCache 量化](../kv_cache_quantization/README.md)为基础（继承其显存收益），追加 Q 量化使能低精度注意力矩阵运算。

---

<a id="holistic-mode"></a>

### 2.2 整体方案

FA 量化的一个"模式"是指 **Q/K/V 三分支量化方式的整体组合**，而不是单看某一分支的粒度。三分支可采用同一种激活值量化模式，也可按分支选用不同的激活值量化模式，各分支分别承担不同的精度与开销角色，例如 Q 采用 per-token 动态免校准、K/V 采用 per-head 静态承接缓存压缩，构成 [FA Q-INT8 动态 K/V-INT8 静态量化](term_fa_q_int8_dynamic_kv_int8.md) 这类混合整体方案。

---

<a id="branch-combination"></a>

### 2.3 已验证整体方案

下表列出已实践验证的 FA 整体方案，每个方案对应一个词条。

| 模式词条 | Q 分支 | K 分支 | V 分支 | 参数获取 |
| --- | --- | --- | --- | --- |
| [FA INT8 PerHead 量化](term_fa_int8_perhead.md) | INT8 per-head | INT8 per-head | INT8 per-head | 静态（需校准） |
| [FA INT8 动态量化](term_fa_int8_dynamic.md) | INT8 per-token | INT8 per-token | INT8 per-token | 动态 |
| [FA FP8 动态量化](term_fa_fp8_dynamic.md) | FP8 per-token | FP8 per-token | FP8 per-token | 动态 |
| [FA MXFP4 动态量化](term_fa_mxfp4_dynamic.md) | MXFP4 per-block | MXFP4 per-block | MXFP4 per-block | 动态 |
| [FA Q-INT8 动态 K/V-INT8 静态量化](term_fa_q_int8_dynamic_kv_int8.md) | INT8 per-token | INT8 per-head | INT8 per-head | Q 动态；K/V 静态 |
| [FA Q-FP8 动态 K/V-FP8 静态量化](term_fa_q_fp8_dynamic_kv_fp8.md) | FP8 per-token | FP8 per-head | FP8 per-head | Q 动态；K/V 静态 |

三分支可采用同一种激活值量化模式，也可分别选用不同的激活值量化模式；未量化的分支保持原精度。上表中每行代表一个可选的 FA 量化模式。

---

<a id="modes"></a>

## 3. 该类下的模式清单

以下为 FA 量化模式清单（完整规格含承载 IR 类，见[量化模式](../README.md)模式索引）：

| 模式 | 一句话 |
| --- | --- |
| [FA INT8 PerHead 量化](term_fa_int8_perhead.md) | Q/K/V 统一按注意力头静态量化，INT8，需校准 |
| [FA INT8 动态量化](term_fa_int8_dynamic.md) | Q/K/V 统一逐 token 动态量化，INT8，免校准 |
| [FA FP8 动态量化](term_fa_fp8_dynamic.md) | Q/K/V 统一逐 token 动态量化，FP8（E4M3），免校准 |
| [FA MXFP4 动态量化](term_fa_mxfp4_dynamic.md) | Q/K/V 统一块级共享指数量化，MXFP4，免校准 |
| [FA Q-INT8 动态 K/V-INT8 静态量化](term_fa_q_int8_dynamic_kv_int8.md) | Q 逐 token 动态、K/V 按头静态，INT8 |
| [FA Q-FP8 动态 K/V-FP8 静态量化](term_fa_q_fp8_dynamic_kv_fp8.md) | Q 逐 token 动态、K/V 按头静态，FP8 |

---

## 4. 关联流程

- 《[一键量化完整指南](../../../user_guide/usage_quick_quantization.md)》：通过 `fa3_quant` 处理器启用本类模式。
- 《[量化精度调优指南](../../../user_guide/process_quantization_precision_tuning.md)》：注意力激活量化精度验证。

---

<a id="related-terms"></a>

## 5. 关联词条

- [量化模式](../README.md)：上位概念，本词条属于"FA 量化"类别。
- [KVCache 量化](../kv_cache_quantization/README.md)：上位概念（基础），本类在其基础上追加 Q 量化。
- [线性层量化](../linear_layer_quantization/README.md)：同位概念，作用于权重与激活的量化类别，可与本类叠加。
- [FA INT8 PerHead 量化](term_fa_int8_perhead.md)：下位概念，本类别下三分支统一 per-head 静态的整体组合模式。
- [FA3 Quant 注意力激活量化算法](../../quantization_algorithms/fa3_quant/term_fa3_quant.md)：配套术语，描述 FA 量化算法。

---

## 6. 参考文档

1. Dao T et al. FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness. NeurIPS 2022. https://arxiv.org/abs/2205.14135
2. 《[FA3 Quant 参数配置流程指南](../../quantization_algorithms/fa3_quant/usage_fa3_quant.md)》
3. 《[量化模式](../README.md)》
