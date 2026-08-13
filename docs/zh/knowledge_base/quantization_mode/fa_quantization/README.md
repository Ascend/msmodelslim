# FA 量化 量化术语百科词条

<!-- waiver: G01 原因：本文件为量化模式目录下的类别文档，不适用 term_<english_name>.md 词条命名 -->
<!-- waiver: S01 原因：索引/类别文档采用目录结构（术语列表 / 模式清单），不适用词条模板的「2. 词条介绍」必填章节，依 01 清单 S03 组织自身结构 -->

> **词条类别**：量化数据格式（[量化模式](../README.md)）
> **英文名称**：Flash Attention Quantization
> **应用领域**：注意力计算加速、长上下文推理
> **msModelSlim 实现**：[`msmodelslim/ir/activation_static.py`](../../../../../msmodelslim/ir/activation_static.py)、[`msmodelslim/ir/activation_dynamic.py`](../../../../../msmodelslim/ir/activation_dynamic.py)（`FakeQuantActivation*` 系列）

---

<a id="overview"></a>

## 概述

FA（Flash Attention，一种高效的注意力计算实现）量化是对**送入 Flash Attention 的 Q/K/V 激活张量**进行量化的一类量化模式，属于[量化模式](../README.md)中的类别之一，是 [KVCache 量化](../kv_cache_quantization/README.md)的**进阶**。它在量化 K/V（压缩缓存显存）的基础上，进一步量化 Q。

K/V 量化主要解决"存储"问题——缓存显存减半、支持更长上下文；Q 加入量化后，注意力得分计算（Q 与 K 的点乘）与加权求和（softmax 后与 V 点乘）的参与张量均为低比特，可直接调用整数/低精度矩阵乘算子，从而在继承 KVCache 显存收益的同时使能低精度的矩阵运算。它是长序列解码场景在"省显存"基础上进一步使能低精度注意力矩阵运算的手段。典型如 [FA PerHead 量化](term_fa_perhead.md)——对 Q 张量按注意力头（per-head）静态量化，INT8 或 FP8。

FA 量化**本质上是一个组合量化模式**：Q/K/V 三分支各自独立选择量化模式（粒度 × 数据类型）。当前最佳实践均通过 `fa3_quant.qconfig` 统一配置三分支，见[分支组合](#branch-combination)。

---

<a id="mode-options"></a>

## 该类一般的量化选择

- **量化对象**：送入 Flash Attention 的 Q/K/V 三个分支的激活张量（K/V 承接缓存压缩，Q 使能低精度得分计算）。
- **量化粒度**：per-head（按注意力头共享 scale，静态，如 [FA PerHead 量化](term_fa_perhead.md)）、per-token（逐 token 在线计算 scale，动态，如 [FA PerToken 量化](term_fa_pertoken.md)）、per-block（块级共享指数，如 [FA PerBlock 量化](term_fa_perblock.md)）。
- **数据类型**：INT8、FP8（E4M3）、MXFP8/MXFP4。
- **与 KV 的关系**：本类以 [KVCache 量化](../kv_cache_quantization/README.md)为基础（继承其显存收益），追加 Q 量化使能低精度注意力矩阵运算。

---

<a id="branch-combination"></a>

## 分支组合

FA 量化作用于 Q/K/V 三个分支（`fa_q` / `fa_k` / `fa_v`），三分支各选一个量化模式组合成整体方案。当前最佳实践实际应用过的组合（均通过 `fa3_quant.qconfig` 统一配置三分支）：

| 组合 | 数据类型 / 粒度 | 参数获取 | 应用示例 |
|------|----------------|---------|---------|
| Q/K/V 统一 INT8 per-head | INT8 per-head | 静态（minmax） | `fa3_quant` 默认配置 |
| Q/K/V 统一 FP8（E4M3）per-token | FP8 per-token | 动态 | Wan2_2、HunYuanVideo 示例 |
| Q/K/V 统一 MXFP4 per-block | MXFP4 per-block | 动态 | QwenImageEdit 示例 |

框架亦支持通过 `fa3_quant.details`（`fa_q`/`fa_k`/`fa_v`）为各分支分别指定配置（未配置的分支不量化），当前示例均使用统一配置。

### 组合 1：Q/K/V 统一 INT8 per-head 静态

- **是什么**：Q/K/V 三分支均按注意力头（per-head）静态量化到 INT8。每个注意力头共享一个 scale（参数开销仅头数），scale 离线校准（minmax）确定并固化，推理零在线开销。
- **配置**：`fa3_quant` 默认（不配 `qconfig` 即启用）：`QConfig(dtype=INT8, scope=PER_HEAD, symmetric=True, method="minmax")`。
- **效果**：三分支均为 8bit 对称量化，K/V 承接缓存压缩，Q 使能低精度得分计算；静态固化使推理无在线统计开销。
- **规格与限制**：需校准数据（无数据则报错）；对称、无 offset。详见 [FA PerHead 量化](term_fa_perhead.md)。

### 组合 2：Q/K/V 统一 FP8（E4M3）per-token 动态

- **是什么**：Q/K/V 三分支均按 token 逐行动态量化到 FP8（E4M3），量化参数在线统计、免校准，每个 token 的激活分辨率贴合自身数值范围。
- **配置**：`fa3_quant.qconfig`：`dtype: fp8_e4m3, scope: per_token, symmetric: True, method: minmax`。
- **效果**：三分支均为 8bit FP8，动态 per-token 免校准且对 token 间分布差异更鲁棒，是长上下文、多模态生成场景的推荐默认。
- **规格与限制**：动态统计带来少量在线开销；FP8 可表示范围有限（E4M3 max 448），大离群值需留意。详见 [FA PerToken 量化](term_fa_pertoken.md)。

### 组合 3：Q/K/V 统一 MXFP4 per-block 动态

- **是什么**：Q/K/V 三分支均沿 head_dim 按32元素分块，每块共享一个 E8M0 指数做 MXFP4 动态量化。粒度比 per-token 更细（捕捉 head_dim 内分布差异），块级共享指数保留浮点动态范围、对离群值更耐受。
- **配置**：`fa3_quant.qconfig`：`dtype: mxfp4, scope: per_block, symmetric: True, method: minmax`（QwenImageEdit 示例）。
- **效果**：三分支均为 4bit（MXFP4），带宽/计算位宽收益最大，是极端压缩场景的选择。
- **规格与限制**：块级指数前向在线统计（免校准）；MXFP4 尾数仅 2bit（emax=2、max 6），精度敏感场景需权衡。详见 [FA PerBlock 量化](term_fa_perblock.md)。

---

<a id="modes"></a>

## 该类下的模式清单

以下为 msModelSlim 支持的 FA 量化模式清单（完整规格含承载 IR 类，见[量化模式](../README.md)「模式索引」）：

| 模式 | 一句话 |
|------|--------|
| [FA PerHead 量化](term_fa_perhead.md) | 按注意力头静态量化 Q，INT8/FP8 |
| [FA PerToken 量化](term_fa_pertoken.md) | 逐 token 动态量化 Q，INT8/FP8 |
| [FA PerBlock 量化](term_fa_perblock.md) | 块级共享指数量化 Q，MXFP8/MXFP4 |

---

## 关联流程

- 《[一键量化完整指南](../../../user_guide/usage_quick_quantization.md)》：通过 `fa3_quant` 处理器启用本类模式。
- 《[量化精度调优指南](../../../user_guide/process_quantization_precision_tuning.md)》：注意力激活量化精度验证。

---

<a id="related-terms"></a>

## 关联词条

- [量化模式](../README.md)：上位概念，本词条是"FA 量化"类别。
- [KVCache 量化](../kv_cache_quantization/README.md)：上位概念（基础），本类在其基础上追加 Q 量化。
- [线性层量化](../linear_layer_quantization/README.md)：同位概念，作用于权重与激活的量化类别，可与本类叠加。
- [FA PerHead 量化](term_fa_perhead.md)：下位概念，本类别下 per-head 静态激活量化。
- 《[FA3量化：Flash Attention 3激活量化算法说明](../../quantization_algorithms/fa3_quant/fa3_quant.md)》：配套术语，描述 FA 量化算法。

---

## 参考资料

1. Dao T et al. FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness. NeurIPS 2022. https://arxiv.org/abs/2205.14135
2. 《[FA3量化：Flash Attention 3激活量化算法说明](../../quantization_algorithms/fa3_quant/fa3_quant.md)》
3. 《[量化模式](../README.md)》
