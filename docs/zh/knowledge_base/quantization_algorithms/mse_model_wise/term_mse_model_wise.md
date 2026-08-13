# 模型级 MSE 敏感层分析算法词条

> **词条类别**：敏感层分析算法
> **英文名称**：MSE Model Wise
> **英文缩写**：mse_model_wise
> **应用领域**：量化敏感层分析、模型量化精度调优
> **msModelSlim 实现**：`msmodelslim/processor/analysis/`

---

## 1. 概述

模型级 MSE（`mse_model_wise`）是 `msmodelslim analyze` 中 `layer` 范围分析的一种度量算法。它逐层对比仅该层相关子结构量化前后的模型最终输出（通常为最后一层 Decoder 输出），计算 MSE 得到层敏感度排序，用于整层或整块回退。其核心特征是：端到端视角、逐层链式前向、全局累积误差度量，与 [MSE Layer Wise](../mse_layer_wise/term_mse_layer_wise.md) 同为基于 MSE 的敏感层分析指标。

---

## 2. 词条介绍

[层级 MSE](../mse_layer_wise/term_mse_layer_wise.md) 关注单块输出上的局部重构误差，无法直接反映该层量化对端到端行为的影响。模型级 MSE 观察到，通过逐层链式前向（将上一层输出作为下一层输入）模拟真实推理路径，以模型最终输出的 MSE 作为敏感度分数，可以更贴近端到端的量化影响。

---

## 3. 原理

### 1. 核心思想

模型级 MSE 的核心思想是“以端到端输出误差衡量层敏感度”：对同一批校准数据，在链式前向中依次评估各 Decoder 层，对该层内由 `quant_modules` 选中的结构做量化前后对比，采集模型最终输出，以最终输出的 MSE 作为该层的敏感度分数。

### 2. 数学描述

对每一层，最终输出的 MSE：

$$
\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_{\text{float, final}}^{(i)} - y_{\text{quant, final}}^{(i)})^2
$$

- $y_{\text{float, final}}$：仅该层相关子结构保持浮点时模型的最终输出
- $y_{\text{quant, final}}$：仅该层相关子结构量化后模型的最终输出
- $n$：输出元素个数
- $\text{MSE}$：该层的敏感度分数

分数越大表示该层对量化越敏感（即该层量化对模型最终输出影响越大）。

### 3. 关键性质

- **layer 范围分析**：输出 Decoder 块粒度的敏感度排序。
- **端到端视角**：以模型最终输出度量量化影响，更贴近端到端效果。
- **链式前向**：逐层链式前向模拟真实推理路径。
- **对齐约束**：层间张量形状或语义无法对齐时，会从该层起跳过后续层并打印 warning。

---

## 4. 流程示意

> 以下为本算法在 msModelSlim 中的简化流程概览。

```mermaid
flowchart LR
    A[校准数据] --> B[链式前向]
    B --> C[逐层量化对比]
    C --> D[采集最终输出 MSE]
    D --> E[层敏感度排序]
```

---

## 5. 在 msModelSlim 中的实现

### 1. 实现位置

模型级 MSE 作为 `msmodelslim analyze` 命令的 `layer` 范围分析指标实现，位于 `msmodelslim/processor/analysis/`。

### 2. 处理流程

通过 `msmodelslim analyze layer --metrics mse_model_wise` 命令执行：对同一批校准数据，在链式前向中依次评估各 Decoder 层，对该层内由 `quant_modules` 选中的结构做量化前后对比，采集模型最终输出并计算 MSE，输出层敏感度排序。

### 3. 命令行示例

```bash
msmodelslim analyze layer \
    --model_type Qwen3-32B \
    --model_path ${model_path} \
    --metrics mse_model_wise \
    --quant_modules "*mlp*" \
    --calib_dataset ${calib_dataset} \
    --topk 15 \
    --device npu
```

---

## 6. 适用场景与限制

### 1. 适用场景

- 希望通过模型最终输出视角评估各层量化敏感度、辅助整层或整块回退决策的场景。
- 需要端到端量化影响评估的场景。

### 2. 使用限制

- 校准批次数与序列长度会显著增加前向次数与中间缓存，大模型上可能 OOM，建议控制校准规模。
- 部分架构受链式前向对齐限制，层间张量形状或语义无法对齐时该层及后续层会被跳过。
- 仅用于 `layer` 范围分析，不适用于 `linear`/`attn` 范围。

---

## 7. 关联流程

- 《[敏感层分析使用指南](../../../user_guide/usage_sensitive_layer_wise_analysis.md)》：本算法作为 `layer` 分析的 metrics 使用。
- 《[一键量化 (V1)](../../../user_guide/usage_quick_quantization.md)》：分析结果可用于辅助量化配置调优。

---

## 8. 关联词条

- [MSE Layer Wise](../mse_layer_wise/term_mse_layer_wise.md)：同类算法，同为 `layer` 范围分析指标，但关注单块输出的局部重构误差。
- [Attention MSE](../attention_mse/term_attention_mse.md)：同类算法，同为基于 MSE 的敏感层分析指标，但用于 `attn` 范围。
- [Std](../std/term_std.md)：对比算法，同为敏感层分析指标，但用于 `linear` 范围。

---

## 9. 参考资料

1. 《模型级 MSE 使用指南》([./usage_mse_model_wise.md](./usage_mse_model_wise.md))
