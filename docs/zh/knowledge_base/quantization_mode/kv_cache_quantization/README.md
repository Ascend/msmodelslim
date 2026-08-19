# KVCache 量化 量化术语百科词条

<!-- waiver: G01 原因：本文件为量化模式目录下的类别文档，不适用 term_<english_name>.md 词条命名 -->
<!-- waiver: S01 原因：索引/类别文档采用目录结构（术语列表 / 模式清单），不适用词条模板的「2. 词条介绍」必填章节，依 01 清单 S03 组织自身结构 -->

> **词条类别**：量化数据格式（[量化模式](../README.md)）
> **英文名称**：KV Cache Quantization
> **应用领域**：KVCache 压缩、长上下文推理加速
> **msModelSlim 实现**：[`msmodelslim/ir/attention.py`](../../../../../msmodelslim/ir/attention.py)

---

<a id="overview"></a>

## 概述

KVCache 量化是对注意力机制中的 **KVCache（缓存的 Key/Value 张量）**进行量化的一类量化模式，属于[量化模式](../README.md)中的类别之一。KVCache 本质上是一种**特殊的激活**——历史 token 的 Key/Value 投影（K/V 线性层输出）被缓存下来，供后续 token 的注意力计算复用，而非模型权重；它不参与权重矩阵乘法，只影响注意力计算。其显存随序列长度线性增长，是长上下文推理的主要显存开销。

量化 K/V 不改变权重与激活的量化方案，可与线性层量化叠加使用；量化后缓存显存约减半（INT8 约为 FP16 的一半），从而支持更长上下文或更大并发；对推理延迟的影响取决于算子与硬件实现，需实测评估。典型如 [KVCache-PerChannel 量化](term_kv_cache_perchannel.md)——对缓存的 K/V 按隐藏维度（通道）共享量化参数，INT8 量化。

---

<a id="mode-options"></a>

## 该类一般的量化选择

- **量化对象**：缓存的 K 与 V 张量（Q 不在此列，量化 Q 属于 [FA 量化](../fa_quantization/README.md)）。
- **量化粒度**：目前 msModelSlim 中仅支持 per-channel——按隐藏维度（head_dim）共享 scale/offset，对称或非对称，即 [KVCache-PerChannel 量化](term_kv_cache_perchannel.md)。
- **数据类型**：INT8（对称/非对称）。更细粒度（如 per-head、per-token）或更低比特（INT4/FP8）属于研究中的方向，msModelSlim 暂不支持。
- **组合方式**：可与线性层量化叠加使用（两者量化对象不同、互不重叠）。在此基础上进一步量化**送入注意力计算的 Q 激活张量**，即升级为 [FA 量化](../fa_quantization/README.md)——FA 量化量化的是 Q 激活（QProj 的输出张量），而非对 QProj 线性层本身做权重/矩阵乘量化。

---

<a id="modes"></a>

## 该类下的模式清单

以下为 msModelSlim 支持的 KVCache 量化模式清单（完整规格含承载 IR 类，见[量化模式](../README.md)「模式索引」）：

| 模式 | 一句话 |
|------|--------|
| [KVCache-PerChannel 量化](term_kv_cache_perchannel.md) | 缓存 K/V 按隐藏维度共享参数，INT8 per-channel |

---

## 关联流程

- 《[一键量化完整指南](../../../user_guide/usage_quick_quantization.md)》：通过 `dynamic_cache` 处理器启用本类模式。
- 《[量化精度调优指南](../../../user_guide/process_quantization_precision_tuning.md)》：长序列精度验证与 KV 回退。

---

<a id="related-terms"></a>

## 关联词条

- [量化模式](../README.md)：上位概念，本词条是"KVCache 量化"类别。
- [线性层量化](../linear_layer_quantization/README.md)：同位概念，作用于权重与激活的量化类别，可与本类叠加。
- [FA 量化](../fa_quantization/README.md)：进阶概念，在本类基础上进一步量化 Q。
- [KVCache-PerChannel 量化](term_kv_cache_perchannel.md)：下位概念，本类别当前唯一的量化模式。
- 《[KVCache量化：缓存量化算法说明](../../quantization_algorithms/kvcache_quant/term_kvcache_quant.md)》：配套术语，描述 KVCache 量化算法。
- 《[KVSmooth：KVCache量化离群值抑制算法说明](../../quantization_algorithms/kv_smooth/term_kv_smooth.md)》：前置术语，量化前对 KV 做平滑以降低量化误差。

---

## 参考资料

1. 《[KVCache量化：缓存量化算法说明](../../quantization_algorithms/kvcache_quant/term_kvcache_quant.md)》
2. 《[量化模式](../README.md)》
