# Attention MSE 敏感层分析算法词条

> **词条类别**：敏感层分析算法
> **英文名称**：Attention MSE
> **英文缩写**：mse
> **应用领域**：量化敏感层分析、Attention 权重量化
> **msModelSlim 实现**：`msmodelslim/processor/analysis/`

---

## 1. 概述

Attention MSE（`mse`）是 `msmodelslim analyze` 中 `attn` 范围分析的一种度量算法。它分别使用浮点权重与量化权重执行前向推理，对同一 attention 模块的输出计算均方误差（MSE），输出注意力模块粒度的敏感度排序，与 [MSE Layer Wise](../mse_layer_wise/term_mse_layer_wise.md) 等同为基于 MSE 的敏感层分析指标。其核心特征是：直接度量注意力子系统在量化权重下的输出漂移、依赖适配器接口。

---

## 2. 词条介绍

对 Attention 结构做权重量化或评估其敏感度时，需要直接度量注意力子系统在量化权重下的输出漂移。Attention MSE 通过分别用浮点与量化权重执行前向，在 attention 模块输出处对比两路张量，用 MSE 刻画该模块对权重量化的敏感程度。

---

## 3. 原理

### 1. 核心思想

Attention MSE 的核心思想是“直接度量输出漂移”：对同一校准样本，分别使用浮点权重与量化权重执行前向，在 attention 模块输出处采集张量，计算两路输出的均方误差；MSE 越大，表示该 attention 模块对当前量化配置越敏感。

### 2. 数学描述

对同一层、同一样本的浮点与量化输出计算 MSE：

$$
\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_{\text{float}}^{(i)} - y_{\text{quant}}^{(i)})^2
$$

- $y_{\text{float}}$：使用浮点权重的 attention 输出
- $y_{\text{quant}}$：使用量化权重的 attention 输出
- $n$：输出元素个数
- $\text{MSE}$：均方误差，用于敏感度排序

### 3. 关键性质

- **attn 范围分析**：输出注意力模块粒度的敏感度排序。
- **直接度量**：直接度量注意力子系统在量化权重下的输出漂移。
- **适配器依赖**：需要模型适配器实现 `AttentionMSEAnalysisInterface`。
- **量化配置感知**：MSE 大小与当前量化配置相关。

---

## 4. 流程示意

> 以下为本算法在 msModelSlim 中的简化流程概览。

```mermaid
flowchart LR
    A[校准前向] --> B[浮点/量化双路]
    B --> C[计算输出 MSE]
    C --> D[attn 敏感度排序]
```

---

## 5. 在 msModelSlim 中的实现

### 1. 实现位置

Attention MSE 作为 `msmodelslim analyze` 命令的 `attn` 范围分析指标实现，位于 `msmodelslim/processor/analysis/`。

### 2. 处理流程

通过 `msmodelslim analyze attn --metrics mse` 命令执行：在 attention 子模块上挂 hook 并读取其前向输出，分别用浮点与量化权重执行前向，在 attention 模块输出处计算 MSE 并排序。不同模型的 attention 类名与 `forward` 返回值形态不一致，须在模型适配器中实现 `AttentionMSEAnalysisInterface`。

### 3. 命令行示例

```bash
msmodelslim analyze attn \
    --model_type DeepSeek-V3 \
    --model_path ${model_path} \
    --metrics mse \
    --calib_dataset ${calib_dataset} \
    --topk 15 \
    --device npu
```

### 4. 模型适配接口

模型适配需实现 `AttentionMSEAnalysisInterface` 接口，提供以下方法：

- `get_attention_module_cls()`：返回待挂 hook 的 attention 模块类名。
- `get_attention_output_extractor()`：从 `forward` 返回值中取出用于计算 MSE 的张量。

---

## 6. 适用场景与限制

### 1. 适用场景

- 需要对 Attention 结构做权重量化或评估其敏感度的场景。
- 需要注意力模块粒度敏感度排序以辅助回退决策的场景。

### 2. 使用限制

- 对应 `model_type` 的模型适配器必须实现 `AttentionMSEAnalysisInterface`，提供模块类名与输出提取函数；未实现会在分析阶段报错。
- 工具当前仅实现 DeepSeek 系列模型的接口适配，其他 `model_type` 会报错或需自行实现。

---

## 7. 关联流程

- 《[敏感层分析使用指南](../../../user_guide/usage_sensitive_layer_wise_analysis.md)》：本算法作为 `attn` 分析的 metrics 使用。
- 《[一键量化 (V1)](../../../user_guide/usage_quick_quantization.md)》：分析结果可用于辅助量化配置调优。

---

## 8. 关联词条

- [Std](../std/term_std.md)：对比算法，同为敏感层分析指标，但用于 `linear` 范围。
- [MSE Layer Wise](../mse_layer_wise/term_mse_layer_wise.md)：同类算法，同为基于 MSE 的敏感层分析指标，但用于 `layer` 范围。
- [MSE Model Wise](../mse_model_wise/term_mse_model_wise.md)：同类算法，基于模型最终输出的 MSE 分析指标。

---

## 9. 参考资料

1. 《Attention MSE 使用指南》([./usage_attention_mse.md](./usage_attention_mse.md))
