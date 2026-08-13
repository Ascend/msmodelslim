# 层级 MSE 敏感层分析算法词条

> **词条类别**：敏感层分析算法
> **英文名称**：MSE Layer Wise
> **英文缩写**：mse_layer_wise
> **应用领域**：量化敏感层分析、模型量化精度调优
> **msModelSlim 实现**：`msmodelslim/processor/analysis/`

---

## 1. 概述

层级 MSE（`mse_layer_wise`）是 `msmodelslim analyze` 中 `layer` 范围分析的一种度量算法。它在每个 Decoder 块内，对由 `quant_modules` 选中的子模块分别做浮点与量化前向，在子模块输出上计算 MSE 并在块内取均值得到敏感度分数，输出 Decoder 块粒度的排序，指导整层或整块回退。其核心特征是：块内聚合 MSE、Decoder 块粒度排序、配置驱动，与 [MSE Model Wise](../mse_model_wise/term_mse_model_wise.md) 同为 `layer` 范围分析指标。

---

## 2. 词条介绍

[Std](../std/term_std.md) 等统计类指标无法直接反映量化重构误差。层级 MSE 观察到，通过在每个 Decoder 块内对比浮点与量化前向的输出差异（MSE），可以直接度量块内聚合的量化重构误差，从而辅助整层或整块（如整段 attention/MLP）回退决策。

---

## 3. 原理

### 1. 核心思想

层级 MSE 的核心思想是“块内聚合的量化重构误差衡量敏感度”：对同一批校准样本，按 Decoder 层遍历，在块内对命中 `quant_modules` 配置的子模块分别采集浮点与量化前向输出，计算 MSE 并在块内取均值作为该块的 score；score 越大表示该 Decoder 块在当前量化子集下越敏感。

### 2. 数学描述

对每一路可对齐的输出计算 MSE：

$$
\text{MSE}_j = \frac{1}{n} \sum_{i=1}^{n} (y_{\text{float}}^{(i)} - y_{\text{quant}}^{(i)})^2
$$

块内聚合取均值：

$$
\text{score} = \frac{1}{m} \sum_{j=1}^{m} \text{MSE}_j
$$

- $y_{\text{float}}$：子模块的浮点前向输出
- $y_{\text{quant}}$：子模块的量化前向输出
- $n$：输出元素个数
- $m$：块内有效子模块（量化对比）数量
- $\text{score}$：该 Decoder 块的敏感度分数

### 3. 关键性质

- **layer 范围分析**：输出 Decoder 块粒度的敏感度排序。
- **块内聚合**：将块内所有有效 MSE 取均值作为该块的 score。
- **配置驱动**：参数 `quant_modules` 决定参与量化对比的子模块集合，不同配置得到不同排序。
- **无适配器依赖**：无需模型适配器额外实现分析接口。

---

## 4. 流程示意

> 以下为本算法在 msModelSlim 中的简化流程概览。

```mermaid
flowchart LR
    A[校准前向] --> B[块内浮点/量化对比]
    B --> C[计算块内 MSE 均值]
    C --> D[Decoder 块敏感度排序]
```

---

## 5. 在 msModelSlim 中的实现

### 1. 实现位置

层级 MSE 作为 `msmodelslim analyze` 命令的 `layer` 范围分析指标实现，位于 `msmodelslim/processor/analysis/`。

### 2. 处理流程

通过 `msmodelslim analyze layer --metrics mse_layer_wise` 命令执行：对同一批校准样本，按 Decoder 层遍历，在块内对命中 `quant_modules` 配置的子模块采集浮点与量化前向输出，计算 MSE 并在块内取均值，输出 Decoder 块粒度的敏感度排序。

### 3. 命令行示例

```bash
msmodelslim analyze layer \
    --model_type Qwen3-32B \
    --model_path ${model_path} \
    --metrics mse_layer_wise \
    --quant_modules "*mlp*" \
    --calib_dataset ${calib_dataset} \
    --topk 15 \
    --device npu
```

---

## 6. 适用场景与限制

### 1. 适用场景

- 希望通过 Decoder 块内子模块输出角度比较各层敏感度、辅助整层或整块回退决策的场景。
- 需要块级量化重构误差评估的场景。

### 2. 使用限制

- 参数 `quant_modules` 决定参与对比量化的子模块集合，不同配置会得到不同排序。
- 仅用于 `layer` 范围分析，不适用于 `linear`/`attn` 范围。
- `model_type` 支持范围与 ModelslimV1 量化一致，参见《[大模型支持矩阵](../../model/README.md)》。

---

## 7. 关联流程

- 《[敏感层分析使用指南](../../../user_guide/usage_sensitive_layer_wise_analysis.md)》：本算法作为 `layer` 分析的 metrics 使用。
- 《[一键量化 (V1)](../../../user_guide/usage_quick_quantization.md)》：分析结果可用于辅助量化配置调优。

---

## 8. 关联词条

- [MSE Model Wise](../mse_model_wise/term_mse_model_wise.md)：同类算法，同为 `layer` 范围分析指标，但关注模型最终输出的全局误差。
- [Attention MSE](../attention_mse/term_attention_mse.md)：同类算法，同为基于 MSE 的敏感层分析指标，但用于 `attn` 范围。
- [Kurtosis](../kurtosis/term_kurtosis.md)：对比算法，同为敏感层分析指标，但用于 `linear` 范围。

---

## 9. 参考资料

1. 《层级 MSE 使用指南》([./usage_mse_layer_wise.md](./usage_mse_layer_wise.md))
