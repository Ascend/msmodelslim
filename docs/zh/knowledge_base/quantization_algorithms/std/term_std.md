# Std 敏感层分析算法词条

> **词条类别**：敏感层分析算法
> **英文名称**：Std
> **英文缩写**：std
> **应用领域**：量化敏感层分析、模型量化精度调优
> **msModelSlim 实现**：`msmodelslim/processor/analysis/`

---

## 1. 概述

Std 是 `msmodelslim analyze` 中 `linear` 范围分析的一种度量算法。它在在线性层（及支持的卷积层）激活上采集统计量，用数值范围与标准差的比值作为敏感度分数，对线性层粒度进行排序。其核心特征是：实现轻量、运行速度快，用于量化前敏感层的粗筛，与 [Quantile](../quantile/term_quantile.md) 等同为 `linear` 范围分析指标。

---

## 2. 词条介绍

量化误差与激活的动态范围、离散程度相关。Std 观察到，在激活标准差较大时，相同动态范围下的相对扰动更小，因此用 `max(|max|,|min|)/std` 形式的 score 可以刻画层对量化的敏感程度，帮助识别需要回退或降低量化强度的层。

---

## 3. 原理

### 1. 核心思想

Std 的核心思想是“用动态范围与离散度的比值衡量敏感度”：在校准前向过程中，对目标层激活统计全局最大/最小值及基于数据中心平移后的标准差，用绝对最大值与标准差的比值作为敏感度分数，分数越大表示该层对量化越敏感。

### 2. 数学描述

统计量计算：

$$
\text{abs\_max} = \max(|\text{max\_value}|, |\text{min\_value}|)
$$

$$
\text{score} = \frac{\text{abs\_max}}{\text{std}}
$$

- $\text{max\_value}$、$\text{min\_value}$：激活的全局最大值与最小值
- $\text{abs\_max}$：绝对最大值
- $\text{std}$：数据中心平移后的标准差
- $\text{score}$：敏感度分数

实现中对 $\text{std} = 0$ 等异常情况有防护处理。

### 3. 关键性质

- **线性层粒度**：用于 `linear` 范围分析，输出线性层粒度的敏感度排序。
- **轻量高效**：仅需统计 max/min 与 std，实现相对轻量、运行速度快。
- **无适配器依赖**：无需模型适配器额外实现分析接口。
- **粗筛定位**：适合作为常规量化前的敏感层粗筛。

---

## 4. 流程示意

> 以下为本算法在 msModelSlim 中的简化流程概览。

```mermaid
flowchart LR
    A[校准前向] --> B[统计 max/min/std]
    B --> C[计算 score]
    C --> D[层敏感度排序]
```

---

## 5. 在 msModelSlim 中的实现

### 1. 实现位置

Std 作为 `msmodelslim analyze` 命令的 `linear` 范围分析指标实现，位于 `msmodelslim/processor/analysis/`。

### 2. 处理流程

通过 `msmodelslim analyze linear --metrics std` 命令执行：在校准前向过程中对目标层激活统计全局最大/最小值及标准差，计算 score 并对线性层粒度排序。

### 3. 命令行示例

```bash
msmodelslim analyze linear \
    --model_type Qwen2.5-7B-Instruct \
    --model_path ${model_path} \
    --metrics std \
    --calib_dataset ${calib_dataset} \
    --pattern "*.down_proj*" "*.o_proj*" \
    --topk 15 \
    --device npu
```

---

## 6. 适用场景与限制

### 1. 适用场景

- 常规量化前的敏感层粗筛。
- 需要快速得到线性层敏感度排序、辅助回退与 YAML 调参的场景。

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

- [Quantile](../quantile/term_quantile.md)：同类算法，同为 `linear` 范围分析指标，对离群点相对稳健。
- [Kurtosis](../kurtosis/term_kurtosis.md)：同类算法，同为 `linear` 范围分析指标，关注尖峰分布。
- [MSE Layer Wise](../mse_layer_wise/term_mse_layer_wise.md)：配套术语，`layer` 范围分析指标，可用于块级评估。

---

## 9. 参考资料

1. 《Std 使用指南》([./usage_std.md](./usage_std.md))
