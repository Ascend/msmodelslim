# KVCache-PerChannel 量化

> **词条类别**：量化数据格式（[KVCache 量化](../README.md)）
> **英文名称**：KV Cache Per-Channel Quantization
> **应用领域**：KVCache 压缩、长上下文推理加速
> **承载 IR 类**：`FakeQuantDynamicCache`（[`msmodelslim/ir/attention.py`](../../../../../msmodelslim/ir/attention.py)）

---

## 1. 概述

KVCache-PerChannel 量化 = 对注意力机制缓存的 K/V 张量做 [INT8](../../quantization_basic/term_int8.md) per-channel 量化。KVCache 是历史 token 的 Key/Value 投影跨时间步缓存下来的**特殊激活**，显存随序列长度线性增长，是长上下文推理的主要显存瓶颈；量化 K/V 使缓存显存减半。它是[KVCache 量化](README.md)类别当前唯一的模式，模式名中「PerChannel」表示按隐藏维度（通道）共享量化参数。

---

## 2. 词条介绍

### 模式规格

| 维度 | 取值 | 说明 |
|------|------|------|
| 量化对象 | 缓存的 K/V 张量 | 注意力机制的 Key/Value 缓存，不参与权重矩阵乘法、只参与注意力计算；不改变权重与激活的量化方案，可与线性层量化叠加 |
| 位宽 | 8bit | 缓存从 FP16 降为 INT8，显存与 KV 读写带宽均减半 |
| 数据类型 | INT8 | 整数格式，缓存 1字节/元素 |
| 参数获取方式 | 静态 | scale/offset 在量化阶段确定并作为参数固化（`requires_grad=False`），推理时直接使用；名字中的 "Dynamic" 指推理时对动态增长的缓存实时量化，而非在线计算 scale |
| 量化粒度 | per-channel | 按隐藏维（head_dim）共享 scale/offset，参数数量与 hidden 同量级、开销可忽略；适配不同隐藏维的数值差异 |
| 对称性 | 对称或非对称 | 依据数值分布是否对称选择，对称仅 scale、非对称加 offset |

### 量化公式

量化与反量化采用标准线性映射。对称量化（仅 scale、无零点）：
$$q = \mathrm{round}(x / s), \qquad \hat{x} = q \cdot s, \qquad s = \frac{\max(|x|)}{2^{b-1}}$$

非对称量化（含零点 offset）：
$$q = \mathrm{round}(x / s) + z, \qquad \hat{x} = (q - z) \cdot s, \qquad s = \frac{\max(x) - \min(x)}{2^b - 1}$$

其中 $x$ 为待量化张量，$q$ 为量化后的整数值，$\hat{x}$ 为反量化还原值，$s$ 为 scale，$z$ 为零点 offset，$b$ 为位宽（此处 INT8 取 $b=8$），且 $z = \mathrm{round}(-\min(x)/s)$。作用对象为缓存的 K/V 张量，量化参数按隐藏维度（head_dim）per-channel 计算（对称或非对称），见「模式规格」表。

### 与其他模式的关系

- **与 [FA PerHead 量化](../fa_quantization/term_fa_perhead.md)（进阶关系）**
  - 本模式只量化 K/V，解决"存储"问题——缓存显存减半、支持更长上下文。
  - FA 量化在其基础上进一步量化 Q，额外使能低精度注意力矩阵运算。本模式是 FA 量化的必要基础：若 K 不量化，仅量化 Q 无法走整数/低精度注意力得分计算。

- **与 [W8A8 静态量化](../linear_layer_quantization/term_w8a8_static.md)（正交关系，可叠加）**
  - 量化对象不同：本模式作用于注意力缓存（特殊激活），W8A8 作用于权重与激活矩阵乘。
  - 二者互不干扰、可组合：权重与激活按 W8A8 量化，KVCache 另行按本模式压缩，共同构成整体量化方案。

### 适用场景与限制

#### 1. 适用场景

- **长上下文推理**：上下文长度大、KV 显存成为瓶颈的场景，如长文档问答、代码仓库理解。
- **高并发服务**：压缩每请求缓存显存，提升同显存下的并发 batch。
- **作为 FA 量化的基础**：K/V 量化后叠加 Q 量化，在省 KV 显存之外进一步使能低精度注意力矩阵运算。

#### 2. 使用限制

- **当前仅支持 per-channel + INT8**：msModelSlim 中 `FakeQuantDynamicCache` 仅注册 INT8 per-channel（对称/非对称），更细粒度或更低比特暂不支持。
- **精度敏感场景需评估**：KV 参与注意力得分计算，量化误差可能被累加放大，极端长序列下需验证精度。

---

## 3. 关联流程

- 《[一键量化完整指南](../../../user_guide/usage_quick_quantization.md)》：通过 YAML 配置选择本模式并执行量化。
- 《[量化精度调优指南](../../../user_guide/process_quantization_precision_tuning.md)》：本模式导致的精度劣化可通过该流程逐层回退与调优。

---

## 4. 关联词条

- [量化模式](../README.md)：上位概念，本词条属于[KVCache 量化](README.md)类别，是该类别下的一种具体模式。
- [FA PerHead 量化](../fa_quantization/term_fa_perhead.md)：进阶模式，在 K/V 量化基础上追加 Q 量化。
- [W8A8 静态量化](../linear_layer_quantization/term_w8a8_static.md)：配套模式，可与本模式叠加使用。
- 《[KVCache量化：缓存量化算法说明](../../quantization_algorithms/kvcache_quant/kvcache_quant.md)》：配套术语，本模式对应的算法处理器文档。
- 《[KVSmooth：KVCache量化离群值抑制算法说明](../../quantization_algorithms/kv_smooth/kv_smooth.md)》：前置术语，量化前对 KV 做平滑以降低量化误差。

---

## 5. 参考文档

1. 《[KVCache量化：缓存量化算法说明](../../quantization_algorithms/kvcache_quant/kvcache_quant.md)》
2. 《[量化模式](../README.md)》
