# Kurtosis 敏感层分析算法词条

> **词条类别**：敏感层分析算法
> **英文名称**：Kurtosis
> **英文缩写**：kurtosis
> **应用领域**：量化敏感层分析、模型量化精度调优
> **msModelSlim 实现**：`msmodelslim/processor/analysis/`

---

## 1. 概述

Kurtosis（峰度）是 `msmodelslim analyze` 中 `linear` 范围分析的一种度量算法。它对激活采样后估计峰度，用于识别分布尖峰与尾部极端值对量化的影响，输出线性层粒度的敏感度排序。其核心特征是：估计超额峰度、关注尖峰与尾部、作为 `linear` 分析的默认 metrics，与 [Std](../std/term_std.md) 等同为 `linear` 范围分析指标。

---

## 2. 词条介绍

量化截断对分布尖峰和极端值敏感。Kurtosis 观察到，分布越集中、极端值越突出，量化截断带来的相对风险往往越高；通过估计激活的峰度可以识别这类层，辅助回退或混合精度决策。

---

## 3. 原理

### 1. 核心思想

Kurtosis 的核心思想是“用峰度刻画分布的尖峭程度”：对层激活做排序与步进采样（控制内存与计算），在采样序列上估计峰度，用超额峰度刻画相对正态的尖峭程度；峰度越大，通常表示分布越尖、对极端值越敏感。

### 2. 数学描述

超额峰度的常用形式：

$$
\text{kurtosis} = \frac{\mathbb{E}[(X - \mu)^4]}{\sigma^4} - 3
$$

- $X$：层激活采样序列
- $\mu$：激活均值
- $\sigma$：激活标准差
- $\text{kurtosis}$：超额峰度（正态分布约为 0）

实现中以具体 `compute_score` 输出为准，用于层间排序。

### 3. 关键性质

- **线性层粒度**：用于 `linear` 范围分析，输出线性层粒度的敏感度排序。
- **尖峰识别**：识别分布尖峰与尾部极端值对量化的影响。
- **内存可控**：通过排序与步进采样控制内存与计算开销。
- **默认指标**：作为 `linear` 分析的默认 metrics。

---

## 4. 流程示意

> 以下为本算法在 msModelSlim 中的简化流程概览。

```mermaid
flowchart LR
    A[校准前向] --> B[激活采样]
    B --> C[估计峰度]
    C --> D[层敏感度排序]
```

---

## 5. 在 msModelSlim 中的实现

### 1. 实现位置

Kurtosis 作为 `msmodelslim analyze` 命令的 `linear` 范围分析指标实现，位于 `msmodelslim/processor/analysis/`，同时是 `linear` 分析的默认 metrics。

### 2. 处理流程

通过 `msmodelslim analyze linear --metrics kurtosis` 命令执行：对层激活做排序与步进采样，在采样序列上估计峰度，输出线性层粒度的敏感度排序。

### 3. 命令行示例

```bash
msmodelslim analyze linear \
    --model_type Qwen2.5-7B-Instruct \
    --model_path ${model_path} \
    --metrics kurtosis \
    --calib_dataset ${calib_dataset} \
    --pattern "*.down_proj*" "*.o_proj*" \
    --topk 15 \
    --device npu
```

---

## 6. 适用场景与限制

### 1. 适用场景

- 需要根据激活分布的尖峰程度识别量化敏感层、辅助回退或混合精度决策的场景。
- 关注尖峰分布与尾部极端值影响的场景。

### 2. 使用限制

- score 的具体阈值需结合模型与业务精度要求判断。
- 仅用于 `linear` 范围分析，不适用于 `layer`/`attn` 范围。
- `model_type` 支持范围参见《[大模型支持矩阵](../../model/README.md)》。

---

## 7. 关联流程

- 《[敏感层分析使用指南](../../../user_guide/usage_sensitive_layer_wise_analysis.md)》：本算法作为 `linear` 分析的默认 metrics 使用。
- 《[一键量化 (V1)](../../../user_guide/usage_quick_quantization.md)》：分析结果可用于辅助量化配置调优。

---

## 8. 关联词条

- [Std](../std/term_std.md)：同类算法，同为 `linear` 范围分析指标，侧重范围与离散度比值。
- [Quantile](../quantile/term_quantile.md)：同类算法，同为 `linear` 范围分析指标，对离群点相对稳健。
- [MSE Layer Wise](../mse_layer_wise/term_mse_layer_wise.md)：配套术语，`layer` 范围分析指标，可用于块级评估。

---

## 9. 参考资料

1. 《Kurtosis 使用指南》([./usage_kurtosis.md](./usage_kurtosis.md))
