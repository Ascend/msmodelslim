# FA3 Quant 注意力激活量化算法词条

> **词条类别**：量化算法
> **英文名称**：FA3 Quant
> **英文缩写**：FA3
> **应用领域**：大语言模型量化压缩、推理加速、长序列推理
> **msModelSlim 实现**：`msmodelslim/processor/quant/fa3/`

---

## 1. 概述

FA3 Quant（Flash Attention 3 激活量化）是一种针对注意力机制激活的 per-head（逐注意力头）量化算法。它对注意力机制中的 Q、K、V 激活进行多种粒度的量化（INT8、FP8），在保持模型精度的前提下提升推理性能和降低显存占用。其核心特征是：per-head 静态量化、Recall Window 算法寻找最小量化范围、支持 MLA 架构，通常与 [线性量化](../linear_quant/term_linear_quant.md) 配合实现全量化方案。

---

## 2. 词条介绍

在长序列下，Attention 的中间激活 Q、K、V 张量在显存中占比高，对其进行量化可以有效降低显存占用并提升计算效率。但 Q、K、V 的激活动态范围大且分布高度不均，直接进行全局量化会导致精度损失严重。FA3 Quant 通过逐注意力头独立量化，适应不同 head 的激活分布差异。

---

## 3. 原理

### 1. 核心思想

FA3 Quant 的核心思想是“逐注意力头独立量化”：对每个注意力头独立计算量化参数（scale），适应不同 head 的激活分布差异；每个 head 使用 Recall Window 算法找到包含指定比例数据的最小数值分布区间作为量化范围，避免离群值拉大量化尺度。

### 2. 数学描述

对激活张量 $x$（shape 为 $(B, H, S, D)$），将其 reshape 为 $(H, N)$，其中 $N = B \times S \times D$，每个 head 独立收集 $N$ 个数据点。

对每个 head 使用 Recall Window 算法寻找最小量化范围：

1. 排序：$\text{sorted\_data} = \operatorname{sort}(\text{head\_data})$
2. 目标元素数量：$\text{target\_num} = \lfloor \text{ratio} \times N \rfloor$（默认 $\text{ratio} = 0.9999$）
3. 滑动窗口搜索窗口长度最小的范围 $[\text{sorted\_data}[i], \text{sorted\_data}[i + \text{target\_num} - 1]]$

对称量化参数：

$$
\text{abs\_max}[h] = \max(|\min\_values[h]|, |\max\_values[h]|), \quad \text{scale}[h] = \frac{\text{abs\_max}[h]}{127}
$$

- $B$：batch 大小
- $H$：注意力头数量
- $S$：序列长度
- $D$：head 维度
- $h$：注意力头索引
- $\text{ratio}$：Recall Window 保留比例，默认 $0.9999$
- $\text{scale}[h]$：第 $h$ 个头的量化缩放因子

跨批次累积统计时，量化范围取所有校准批次的最小值/最大值并集。

### 3. 关键性质

- **per-head 量化**：对每个注意力头独立计算量化参数，适应分布差异。
- **Recall Window**：滑动窗口寻找包含指定比例数据的最小范围，抑制离群值影响。
- **多粒度支持**：支持 INT8 与 FP8（静态/动态）多种量化粒度。
- **MLA 适配**：针对 Multi-head Latent Attention 计算的关键位置插入量化节点。

---

## 4. 流程示意

> 以下为本算法在 msModelSlim 中的简化流程概览。

```mermaid
flowchart LR
    A[注入占位器] --> B[校准收集统计]
    B --> C[Recall Window 搜索]
    C --> D[计算量化参数]
    D --> E[部署伪量化]
```

---

## 5. 在 msModelSlim 中的实现

### 1. 实现位置

算法在 `msmodelslim/processor/quant/fa3/processor.py` 中实现，通过 `type: "fa3_quant"` 处理器使用。

### 2. 处理流程

- **注入阶段**（`preprocess`）：调用模型适配器的 `inject_fa3_placeholders()`，在 MLA 计算流程的关键位置插入占位器 `FA3QuantPlaceHolder`，支持 `include/exclude` 选择性注入。
- **校准阶段**（`process`）：占位符替换为监听器 `_FA3PerheadObserver`，校准数据流经注意力层时收集每个 head 的激活统计信息。
- **伪量化部署阶段**（`postprocess`）：从监听器提取 min/max，调用 `calculate_qparam()` 计算对称量化参数，创建 IR 替换监听器。

### 3. 配置示例

> 以下为最小可用的 YAML 配置片段。各字段的详细含义如下表所示。

```yaml
spec:
  process:
    - type: "fa3_quant"
      qconfig:
          dtype: "fp8_e4m3"
          scope: "per_token"
          symmetric: True
          method: "minmax"
      include: [ "*" ]
      exclude: [ "model.layers.0.self_attn" ]
```

**字段说明**：

| 字段名 | 作用 | 说明 |
| --- | --- | --- |
| type | 处理器类型标识 | 固定为 `"fa3_quant"`。 |
| qconfig | 量化统一配置 | Q/K/V 的统一量化配置，与 `details` 不可同时配置。 |
| details | 量化详细配置 | 按 `fa_q`/`fa_k`/`fa_v` 分别配置各激活值的量化方式。 |
| include | 包含的注意力层 | 字符串列表，支持通配符匹配，指定执行 FA3 量化的注意力层。 |
| exclude | 排除的注意力层 | 字符串列表，支持通配符匹配，优先级高于 `include`。 |

### 4. 模型适配接口

模型适配需实现 `FA3QuantAdapterInterface` 接口的 `inject_fa3_placeholders()` 方法，在 MLA 计算流程关键位置插入占位器 `FA3QuantPlaceHolder`。参考实现：`msmodelslim/model/deepseek_v3/model_adapter.py`。

---

## 6. 适用场景与限制

### 1. 适用场景

- 长序列推理场景下需要降低 Attention 中间激活显存占用的场景。
- 基于 MLA 架构的模型（如 DeepSeek 系列）的全量化方案。

### 2. 使用限制

- 必须有支持 FA3 的模型适配器实现 `FA3QuantAdapterInterface`，适用于基于 MLA 的注意力机制。
- 当前支持 INT8/FP8 静态对称量化与 FP8 动态量化。
- `qconfig` 与 `details` 字段不支持同时配置。

---

## 7. 关联流程

- 《[一键量化 (V1)](../../../user_guide/usage_quick_quantization.md)》：可集成 FA3 Quant 作为注意力激活量化步骤。
- 《[量化精度调优指南](../../../user_guide/process_quantization_precision_tuning.md)》：精度不达标时可调整 FA3 Quant 配置。

---

## 8. 关联词条

- [线性量化](../linear_quant/term_linear_quant.md)：配套术语，FA3 Quant 通常与线性量化配合实现全量化方案。
- [KVCache Quant](../kvcache_quant/term_kvcache_quant.md)：配套术语，同属长序列推理的缓存/激活量化方案。
- [KV Smooth](../kv_smooth/term_kv_smooth.md)：配套术语，同为长序列推理的缓存优化算法。
- [MinMax](../minmax/term_minmax.md)：应用对象，FA3 Quant 的量化参数基于 MinMax 思路计算。

---

## 9. 参考资料

1. Dao T et al. FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision. 2024. https://arxiv.org/abs/2407.08608
2. 《FA3 Quant 使用指南》([./usage_fa3_quant.md](./usage_fa3_quant.md))
