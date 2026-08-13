# Quantile 敏感层分析算法词条

> **词条类别**：敏感层分析算法
> **英文名称**：Quantile
> **英文缩写**：quantile
> **应用领域**：量化敏感层分析、模型量化精度调优
> **msModelSlim 实现**：`msmodelslim/processor/analysis/`

---

## 1. 概述

Quantile（分位数）是 `msmodelslim analyze` 中 `linear` 范围分析的一种度量算法。它基于激活的分位数与四分位距（IQR）构造 score，对离群点相对稳健，用于线性层粒度的敏感度排序。其核心特征是：基于分位数统计、对离群点稳健、适合激活尾部较重的分布，与 [Std](../std/term_std.md) 等同为 `linear` 范围分析指标。

---

## 2. 词条介绍

[Std](../std/term_std.md) 等基于极值与标准差的指标容易受离群点主导，导致单层分数被极端值放大。Quantile 观察到，用四分位数（第 1/4、第 3/4 分位数）描述激活分布的主体宽度，可以降低离群点对单层分数的主导影响，使敏感度排序更稳健。

---

## 3. 原理

### 1. 核心思想

Quantile 的核心思想是“用四分位数刻画主体分布”：计算激活的下四分位数 $Q_1$（第 1/4 分位数）与上四分位数 $Q_3$（第 3/4 分位数），用四分位距 $\text{IQR} = Q_3 - Q_1$ 描述主体分布宽度，结合绝对幅度构造分数，对离群点相对稳健。

### 2. 数学描述

四分位距：

$$
\text{IQR} = Q_3 - Q_1
$$

敏感度分数：

$$
\text{score} = \frac{2 \times \max(|\text{max\_value}|, |\text{min\_value}|)}{254 \times (Q_3 - Q_1)}
$$

- $Q_1$：激活的第 1/4 分位数
- $Q_3$：激活的第 3/4 分位数
- $\text{IQR}$：四分位距
- $\max(|\text{max\_value}|, |\text{min\_value}|)$：激活绝对值的最大值
- $\text{score}$：敏感度分数

解读：激活绝对值的最大值越大，score 越大（量化步长相对越大、误差越显著）；IQR 越大，score 越小（主体分布越分散、相对误差越小）。

### 3. 关键性质

- **线性层粒度**：用于 `linear` 范围分析，输出线性层粒度的敏感度排序。
- **对离群点稳健**：基于分位数而非极值，降低离群点对单层分数的主导影响。
- **主体分布刻画**：用 IQR 描述激活主体分布宽度。
- **无适配器依赖**：无需模型适配器额外实现分析接口。

---

## 4. 流程示意

> 以下为本算法在 msModelSlim 中的简化流程概览。

```mermaid
flowchart LR
    A[校准前向] --> B[统计分位数]
    B --> C[计算 IQR 与 score]
    C --> D[层敏感度排序]
```

---

## 5. 在 msModelSlim 中的实现

### 1. 实现位置

Quantile 作为 `msmodelslim analyze` 命令的 `linear` 范围分析指标实现，位于 `msmodelslim/processor/analysis/`。

### 2. 处理流程

通过 `msmodelslim analyze linear --metrics quantile` 命令执行：在校准前向过程中计算激活的第 1/4、第 3/4 分位数与幅度统计量，构造 score 并对线性层粒度排序。

### 3. 命令行示例

```bash
msmodelslim analyze linear \
    --model_type Qwen2.5-7B-Instruct \
    --model_path ${model_path} \
    --metrics quantile \
    --calib_dataset ${calib_dataset} \
    --pattern "*.down_proj*" "*.o_proj*" \
    --topk 15 \
    --device npu
```

---

## 6. 适用场景与限制

### 1. 适用场景

- 激活分布尾部较重、希望降低离群点对单层分数主导影响的场景。
- 需要更稳健的线性层敏感度排序的场景。

### 2. 使用限制

- score 的具体阈值需结合模型与业务精度要求判断。
- 仅用于 `linear` 范围分析，不适用于 `layer`/`attn` 范围。
- `model_type` 支持范围参见《[大模型支持矩阵](../../model/README.md)》。

---

## 7. 关联流程

- 《[敏感层分析使用指南](../../../user_guide/usage_sensitive_layer_wise_analysis.md)》：本算法作为 `linear` 分析的 metrics 使用。
- 《[一键量化 (V1)](../../../user_guide/usage_quick_quantization.md)》：分析结果可用于辅助量化配置调优。

---

## 8. 关联词条

- [Std](../std/term_std.md)：同类算法，同为 `linear` 范围分析指标，侧重范围与离散度比值。
- [Kurtosis](../kurtosis/term_kurtosis.md)：同类算法，同为 `linear` 范围分析指标，关注尖峰分布。
- [MSE Layer Wise](../mse_layer_wise/term_mse_layer_wise.md)：配套术语，`layer` 范围分析指标，可用于块级评估。

---

## 9. 参考资料

1. 《Quantile 使用指南》([./usage_quantile.md](./usage_quantile.md))
