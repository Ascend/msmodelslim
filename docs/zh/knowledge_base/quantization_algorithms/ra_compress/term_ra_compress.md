# RA Compress 长序列压缩算法 量化术语百科词条

> **词条类别**：[敏感层分析算法](../README.md#3-敏感层分析算法)<br>
> **英文名称**：RazorAttention Compress（`ra_compress`）<br>
> **应用领域**：长序列推理、KV cache 压缩、注意力头筛选<br>

---

## 1. 概述

RA Compress（`ra_compress`）是一种注意力头（KV head）粒度的长序列压缩筛选算法。它基于 Transformer 中注意力头的跨段（segment-spanning）注意力行为，识别具备 prefix matching（归纳头）与 copying matching（回声头）能力的关键 KV head，作为长序列推理时 KV cache 压缩的保留依据；核心特征是用等长重复段构造受控输入，把“跨段检索能力”本身变成可度量的分数，可与同样位于 attn 范围的 [Attention MSE](../attention_mse/term_attention_mse.md) 等敏感层分析指标互补。

---

## 2. 词条介绍

长文本（长序列）推理时，KV cache 随序列长度线性增长，是端侧和大模型推理时显存的主要瓶颈。RA Compress 观察到：并非所有 KV head 对长上下文质量的贡献都是均等的——具有强归纳（induction）和复制（copying）能力的注意力头是跨段信息传递的关键锚点，在 KV cache 压缩时应优先保留。因此它不从任务数据反推重要性，而是直接构造重复段输入，在段间固定偏移位置度量注意力强度，把“是否具备跨段检索能力”作为筛选依据。

### 2.1 核心思想

RA Compress 的核心思想是“利用重复段结构定位跨段关键头”。输入被设计为若干段**完全相同的 token 序列**；在这种结构下，真正承担跨段信息传递的头会表现出可预测的注意力落点：归纳头在当前段某位置会关注上一段中紧邻其前一位的 token（prefix matching），回声头则直接复制上一段相同位置的信息（copying matching）。

这些落点在普通文本上受语义与位置分布干扰、难以稳定观测；而在重复段结构下会以集中的注意力峰值出现，于是“某头是否具备跨段检索能力”就转化为一个在受控条件下可复现的分数。

### 2.2 工作机制

RA Compress 的工作机制可以分为三步：

1. **构造重复段校准输入**：由首 token 与若干段等长随机 token 组成，段长与段数为算法内建常量——首 token 提供可复用的起始条件，段长 `2500`、重复 `4` 段，总长 `1 + 2500 × 4`；段内内容完全相同，整体不依赖任何外部数据。
2. **计算段间偏移注意力**：在自注意力的 Q / K 投影上采集输出，对每一层计算 $QK^\top$ 并 softmax 得到注意力矩阵；再在每个段对上读取两个固定偏移位置——`-SEG + 1` 对应 prefix matching、`-SEG` 对应 copying matching，对该偏移位置在重复段对上取平均，即得到该层每个 head 的两类得分。
3. **分组与选择**：若模型使用 GQA / MQA，先在每个 KV head 对应的 query head 组内取最大得分作为该 KV head 的代表分；再把所有层、所有 KV head 的得分整体排序，按比例选出 top KV head 作为保留对象，输出为 KV head 索引列表。

整个过程中被评估的不是“模型在某份数据上的表现”，而是注意力结构本身在受控重复条件下是否具备跨段检索能力，因此不需要用户准备校准集。

### 2.3 数学描述

设 $A \in \mathbb{R}^{L \times L}$ 为某一层在重复段校准数据上的注意力矩阵（对 Q 与 K 的 outer product 做 softmax 后）。对所有满足 $i \in [(k+1) \cdot SEG, (k+2) \cdot SEG]$ 的位置 $i$（第 $k+1$ 段），分别考察以下两个偏移量位置：

**Prefix matching 得分（归纳头）**：

$$
\text{prefix\_score}_h = \text{mean}_{k, i} A_h[i, i - k \cdot SEG + 1]
$$

即第 $k+1$ 段的位置 $i$ 上，注意力落在第 $k$ 段的 $i - SEG + 1$ 处（前一段 prefix 位置 + 1）的概率，对应归纳头的典型行为。

**Copying matching 得分（回声头）**：

$$
\text{copying\_score}_h = \text{mean}_{k, i} A_h[i, i - (k+1) \cdot SEG]
$$

即第 $k+1$ 段的位置 $i$ 上，注意力落在第 $k$ 段相同偏移 $i - SEG$ 处的概率，对应“复制前一段对应位置”的回声头行为。

其中 $SEG = 2500$（段长），$k \in \{0, 1, 2\}$（共 3 个段对），head $h$ 为 GQA 分组前的 attention head。

**GQA 分组合并**：若模型使用 GQA / MQA（$num\_kv\_heads < num\_attention\_heads$），则在每个 KV head 对应的 query head 组内取最大得分作为该 KV head 的代表分：

$$
\text{grouped\_score}[g] = \max_{h \in \text{group}[g]} \text{score}_h
$$

**Top ratio 选择**：对所有层、所有 KV head 的得分整体排序，取前 `induction_head_ratio`（默认 14%）的 KV head 作为 prefix matching 入选（induction heads），取前 `echo_head_ratio`（默认 1%）的 KV head 作为 copying matching 入选（echo heads）。两个比例合法取值范围为 $[0, 1]$。

### 2.4 关键性质

- **attn_head 粒度**：输出 KV head 粒度的入选列表（每层若干 head 索引），而非层粒度排序。
- **双通道筛选**：归纳头（prefix matching）与回声头（copying matching）分别按比例独立选出，二者对应不同的跨段信息传递机制。
- **受控合成输入**：校准输入由算法内部按固定配方构造，不读取外部校准集，因此结果与用户业务语料无关，可复现性强。
- **比例决定入选规模**：保留多少个 head 不由分数阈值决定，而由 top ratio 决定，比例设置直接决定入选集合大小。
- **位置编码一致性**：计算得分前对注入的 Q / K 施加位置编码，使度量条件与真实推理时的注意力形态保持一致。
- **输出为离散集合**：只输出入选 head 索引，不输出敏感度分数排序，因此不能用于连续阈值的回退决策。
- **结果用于压缩而非回退**：输出为 `head.pt`（dict 序列化的 `.pt` 文件），交付给 MindIE 推理框架用于长序列 KV cache 压缩。用户可根据 `head.pt` 完成长序列压缩，压缩后的模型可用于后续推理部署；具体已适配的量化模型列表，请参见《[MindIE 是什么](https://www.hiascend.com/document/detail/zh/mindie/10RC3/whatismindie/mindie_what_0001.html)》的“MindIE 支持模型列表”章节。该产物不用于量化 YAML 的 exclude / include 配置，也不用于层回退决策。

### 2.5 适用场景

- 长序列推理场景下，需要了解模型中哪些 KV head 承担了跨段信息传递（归纳 / 复制头）的场景。
- 需要为 KV cache 压缩确定“应保留哪些 KV head”的保留集合的场景。
- 无需准备外部校准集，希望快速对模型注意力头能力做筛选的分析场景。

### 2.6 使用限制

- 目前仅支持大语言模型（LLM）；不支持多模态理解模型（VLM）与多模态生成模型（文生图 / 文生视频等）。
- 仅分析 Transformer 自注意力的 Q / K 投影层（或 QKV 融合层），其他结构不在分析范围内；投影层命名不符合默认模式的模型，需由模型适配器通过 `RaCompressAnalysisInterface.get_proj_names` 提供命名信息。
- 结果依赖重复段这一受控输入结构，反映的是注意力头的跨段检索能力，不能直接等同于模型在真实长文本任务上的质量表现。
- 输出为离散入选集合而非连续分数，入选比例的变化会整体改变结果，不同比例下的结果之间不具备“分数大小”意义上的可比性。
- `model_type` 支持范围参见《[大模型支持矩阵](../../model/README.md)》及对应的适配器实现情况。

---

## 3. 关联流程

- 《[Attention Head 筛选分析使用指南](../../../user_guide/usage_sensitive_attn_head_analysis.md)》：本算法作为 `attn_head` 范围分析的 metrics 使用。
- MindIE 推理框架侧（交付件消费方）：分析产出的 `head.pt` 交付给 MindIE 推理框架，用户可据此完成长序列压缩，压缩后的模型可用于后续推理部署；具体已适配的量化模型列表，请参见《[MindIE 是什么](https://www.hiascend.com/document/detail/zh/mindie/10RC3/whatismindie/mindie_what_0001.html)》的“MindIE 支持模型列表”章节。

---

## 4. 关联词条

- [KVCache Quant](../kvcache_quant/term_kvcache_quant.md)：配套术语，KV cache 的量化方案，与 RA Compress 同属长序列显存优化范畴。
- [KV Smooth](../kv_smooth/term_kv_smooth.md)：配套术语，针对 KV cache 的离群值抑制算法。
- [Attention MSE](../attention_mse/term_attention_mse.md)：同类术语，同为 attn 范围敏感层分析指标（但用于 FA 回退决策而非 head 选择）。
- [MSE Layer Wise](../mse_layer_wise/term_mse_layer_wise.md)：配套术语，层级敏感层分析指标。

---

## 5. 参考文档

1. 论文：*RazorAttention: Efficient KV Cache Compression Through Retrieval Heads*（induction / copying head 相关背景）
