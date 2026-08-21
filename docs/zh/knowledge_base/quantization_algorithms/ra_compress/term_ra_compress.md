# RA Compress 长序列压缩算法词条

> **词条类别**：敏感层分析算法（长序列 KV cache 压缩）
> **英文名称**：RazorAttention Compress
> **英文缩写**：ra_compress
> **应用领域**：长序列推理、KV cache 压缩、注意力头筛选
> **msModelSlim 实现**：`msmodelslim/processor/analysis/unary_operator/metrics/ra_compress/`

---

## 1. 概述

RA Compress（`ra_compress`）是 `msmodelslim analyze` 中 `attn_head` 范围分析的一种度量算法。它基于 Transformer 中注意力头的跨段（segment-spanning）注意力行为，识别具有 prefix matching（归纳头）和 copying matching（回声头）能力的关键 KV head，用于后续 KV cache 压缩配置。其核心特征是：依赖合成的重复段校准数据、分析 Q@K^T 注意力分数在段间偏移位置的聚合统计、按固定 ratio 选择 top heads、输出 `head.pt` 文件供后续压缩流程使用。

---

## 2. 词条介绍

长文本（长序列）推理时，KV cache 随序列长度线性增长，是端侧和大模型推理时显存的主要瓶颈。RA Compress 观察到：并非所有 KV head 对长上下文质量的贡献都是均等的——具有强归纳（induction）和复制（copying）能力的注意力头是跨段信息传递的关键锚点，在 KV cache 压缩时应优先保留。RA Compress 通过构造**等长重复段**（`DUMMY_INPUT_LENGTH=2500` × `REPET_TIMES=4`，总计 ≥ 10000 tokens）的合成校准数据，在段间特定偏移位置度量注意力强度，从而为每层、每个 head 计算 prefix matching / copying matching 得分，再按预设 ratio 选 top heads 作为压缩保留目标。

---

## 3. 原理

### 3.1 核心思想

RA Compress 的核心思想是“利用重复段结构定位跨段关键头”：构造 4 段完全相同的 2500 tokens 重复序列（共 10000 tokens）。在段边界处，具有归纳能力的头会在当前段的 prefix 位置关注前一段对应位置 + 1（prefix matching，`offset = -SEG + 1`），而具有复制能力的头会在当前段位置关注前一段对应位置（copying matching，`offset = -SEG`）。对每一层，在重复的段对上取该偏移位置的注意力分数作逐段平均，得到每个 head 的 prefix / copying 得分；得分 top ratio 的 head 入选为 KV cache 压缩时需要保留的 KV head 索引。

### 3.2 数学描述

设 $A \in \mathbb{R}^{L \times L}$ 为某一层在重复段校准数据上的注意力矩阵（对 Q 与 K 的 out product 做 softmax 后）。对所有满足 $i \in [(k+1) \cdot SEG, (k+2) \cdot SEG]$ 的位置 $i$（第 $k+1$ 段），分别考察以下两个偏移量位置：

**Prefix matching 得分（归纳头）**：
$$
\text{prefix\_score}_h = \text{mean}_{k, i} A_h[i, i - k \cdot SEG + 1]
$$
（在第 $k+1$ 段的位置 $i$ 上，注意力落在第 $k$ 段的 $i - SEG + 1$ 处的概率，即“前一段 prefix 位置 + 1”，对应归纳头的典型行为。）

**Copying matching 得分（回声头）**：
$$
\text{copying\_score}_h = \text{mean}_{k, i} A_h[i, i - (k+1) \cdot SEG]
$$
（在第 $k+1$ 段的位置 $i$ 上，注意力落在第 $k$ 段相同偏移 $i - SEG$ 处的概率，对应“复制前一段对应位置”的回声头行为。）

其中 $SEG = \text{DUMMY\_INPUT\_LENGTH} = 2500$，$k \in \{0, 1, 2\}$（共 3 个段对），head $h$ 为 GQA 分组前的 attention head。

**GQA 分组合并**：若模型使用 GQA / MQA（`num_kv_heads < num_attention_heads`），则在每个 KV head 对应的 query head 组内取最大得分作为该 KV head 的代表分：
$$
\text{grouped\_score}[g] = \max_{h \in \text{group}[g]} \text{score}_h
$$

**Top ratio 选择**：对所有层、所有 KV head 的得分整体排序，取前 `induction_head_ratio`（默认 14%）的 KV head 作为 prefix matching 入选（induction heads），取前 `echo_head_ratio`（默认 1%）的 KV head 作为 copying matching 入选（echo heads）。

### 3.3 关键性质

- **attn_head 粒度**：输出 KV head 粒度的入选列表（每层若干 head 索引），而非层粒度排序。
- **数据依赖强**：要求校准集 tokenize 后总长度 ≥ 10000 tokens，否则段位置偏移无意义、得分被置空。
- **合成段数据优先**：默认使用工具内置的 `calib_dummy.jsonl`，该文件经 tokenizer 后严格对齐到 2500×N 段边界。
- **适配器接口可选依赖**：默认匹配 `q_proj` / `k_proj` / `qkv_proj` 命名，若模型命名不同，需在适配器实现 `RaCompressAnalysisInterface`。
- **结果用于压缩而非回退**：输出为 `head.pt`（dict 序列化的 `.pt` 文件），供 RA Compress KV cache 压缩流程读取，而非用于量化 YAML 的 exclude / include 配置。

---

## 4. 流程示意

> 以下为本算法在 msModelSlim 中的简化流程概览。

```mermaid
flowchart LR
    A[calib_dummy.jsonl 合成校准集] --> B[Q_proj / K_proj hook 采集输出]
    B --> C[逐段计算 Q @ K^T]
    C --> D[段间偏移位置取注意力]
    D --> E[逐段平均得到 head 得分]
    E --> F[GQA 分组取 max]
    F --> G[按 ratio 选 top heads]
    G --> H[写入 layer_scores (enrich)]
    H --> I[保存 head.pt]
```

---

## 5. 在 msModelSlim 中的实现

### 5.1 实现位置

RA Compress 作为 `msmodelslim analyze` 命令的 `attn_head` 范围分析指标实现，位于 `msmodelslim/processor/analysis/unary_operator/metrics/ra_compress/`。

模块结构：

- `interface.py`：定义 `RaCompressAnalysisInterface`，在模型适配器侧提供 Q / K / QKV 投影层名称模式。
- `__init__.py`：`RaCompressAnalysisMethod` 主实现（hook、得分计算、分组、head 选择、layer_scores enrich）。

### 5.2 处理流程

通过 `msmodelslim analyze attn_head --metrics ra_compress` 命令执行（`attn_head` 默认 metrics）：

1. 在 Q / K（或 QKV 融合）投影层上注册 forward hook，在 `outputs` 位置保存该层的输出张量。
2. 每层计算得分时，若为 Q 层则取出该层对应 K 层的输出（反之亦然），计算 `Q @ K^T` 并按 head 拆分。
3. 对每个 head 的注意力矩阵，分别计算段间 `offset = -SEG + 1`（prefix）与 `offset = -SEG`（copying）位置的逐段平均得分。
4. 经过 `_max_every_group`（GQA 分组 max）后，按 `_select_top_heads` 取 14% / 1% top heads。
5. `enrich_layer_scores` 将入选的 `induction_heads` / `echo_heads`（KV head 索引列表）挂到 `layer_scores` 条目里。
6. 结果展示器遍历 `layer_scores` 重建 head_dict，打印入选列表并保存 `head.pt`（结构为 `{"prefix_matching": {layer_idx: [kv_head_idx, ...]}, "copying": {...}}`）。

### 5.3 命令行示例

```bash
msmodelslim analyze attn_head \
    --model_type Qwen2.5-7B-Instruct \
    --model_path ${model_path} \
    --metrics ra_compress \
    --device npu \
    --trust_remote_code True \
    --save_path ./head_result
```

### 5.4 模型适配接口

RA Compress 分析需要定位 Transformer 自注意力的 Q / K（或 QKV 融合）投影层以挂载 hook 采集输出。不同模型的投影层命名可能不一致，因此通过模型适配器接口 `RaCompressAnalysisInterface` 暴露名称模式，由分析方法据此匹配目标层。当适配器未实现该接口时，方法回退到默认名称（`q_proj` / `k_proj` / `qkv_proj`），仅适用于遵循该命名约定的模型。

**接口定义**：[`RaCompressAnalysisInterface`](../../../../../msmodelslim/processor/analysis/unary_operator/metrics/ra_compress/interface.py)

**接口方法**：

| 方法 | 返回类型 | 说明 |
| --- | --- | --- |
| `get_ra_compress_proj_patterns()` | `Dict[str, str]` | 返回 Q / K / QKV 投影层名称模式字典 |

返回值格式：

```python
{
    "q": "q_proj",      # Q 投影层名称模式
    "k": "k_proj",      # K 投影层名称模式
    "qkv": "qkv_proj",  # QKV 融合投影层名称模式（无融合时留空字符串）
}
```

**名称匹配规则**（见 [`RaCompressAnalysisMethod._is_target_layer`](../../../../../msmodelslim/processor/analysis/unary_operator/metrics/ra_compress/__init__.py)）：分析方法对 `model.named_modules()` 中每个 `nn.Linear`，判断其层名是否包含上述任一模式字符串，命中即作为目标层挂载 hook。

**默认回退**：适配器未实现 `RaCompressAnalysisInterface` 时，使用以下默认模式（见 [`__init__.py`](../../../../../msmodelslim/processor/analysis/unary_operator/metrics/ra_compress/__init__.py) 顶部常量）：

| 键 | 默认值 | 常量 |
| --- | --- | --- |
| `q` | `q_proj` | `_DEFAULT_Q_NAME_PATTERN` |
| `k` | `k_proj` | `_DEFAULT_K_NAME_PATTERN` |
| `qkv` | `qkv_proj` | `_DEFAULT_QKV_NAME_PATTERN` |

**适配器实现示例**：

当模型投影层命名与默认值不一致时（例如使用 `query_proj` / `key_proj`），在模型适配器中实现接口并覆盖名称模式：

```python
from msmodelslim.processor.analysis.unary_operator.metrics.ra_compress.interface import (
    RaCompressAnalysisInterface,
)

class XxxAdapter(RaCompressAnalysisInterface):
    def get_ra_compress_proj_patterns(self):
        return {
            "q": "query_proj",
            "k": "key_proj",
            "qkv": "",  # 该模型无 QKV 融合层
        }
```

---

## 6. 适用场景与限制

### 6.1 适用场景

- 需要为 KV cache 压缩（RA Compress）方案生成 `head.pt` 的场景。
- 长序列推理场景下，需要了解模型中哪些 KV head 承担了跨段信息传递（归纳 / 复制头）。
- 使用重复段合成校准数据的注意力头能力分析。

### 6.2 使用限制

- 校准集 tokenize 后总长度必须 ≥ `DUMMY_INPUT_LENGTH * REPET_TIMES`（2500 × 4 = 10000 tokens），不足时得分被置空并告警。
- 目前仅分析 Transformer 自注意力的 Q / K 投影层（或 QKV 融合层）；其他结构不在范围内。
- 结果不输出敏感度 Score 排序，只输出入选的 head 索引；ratio 目前使用默认常量（14% / 1%），不通过 YAML 配置。
- `model_type` 支持范围参见《[大模型支持矩阵](../../model/README.md)》及适配器对 `RaCompressAnalysisInterface` 的实现情况。

---

## 7. 关联流程

- 《[Attention Head 筛选分析使用指南](../../../user_guide/usage_sensitive_attn_head_analysis.md)》：本算法作为 `attn_head` 分析的 metrics 使用。
- 《[一键量化 (V1)](../../../user_guide/usage_quick_quantization.md)》：分析产出的 `head.pt` 可在 RA Compress 相关量化流程中使用。

---

## 8. 关联词条

- [KVCache Quant](../kvcache_quant/term_kvcache_quant.md)：配套术语，KV cache 的量化方案，与 RA Compress 同属长序列显存优化范畴。
- [KV Smooth](../kv_smooth/term_kv_smooth.md)：配套术语，针对 KV cache 的离群值抑制算法。
- [Attention MSE](../attention_mse/term_attention_mse.md)：同类术语，同为 attn 范围敏感层分析指标（但用于 FA 回退决策而非 head 选择）。
- [MSE Layer Wise](../mse_layer_wise/term_mse_layer_wise.md)：配套术语，层级敏感层分析指标。

---

## 9. 参考资料

1. 《RA Compress 使用指南》([./usage_ra_compress.md](./usage_ra_compress.md))
2. 论文：*RazorAttention: Efficient KV Cache Compression Through Retrieval Heads*（induction / copying head 相关背景）
