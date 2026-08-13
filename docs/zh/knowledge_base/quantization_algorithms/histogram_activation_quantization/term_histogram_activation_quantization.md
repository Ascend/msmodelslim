# Histogram 直方图激活值量化算法词条

> **词条类别**：量化算法
> **英文名称**：Histogram Activation Quantization
> **英文缩写**：Histogram
> **应用领域**：大语言模型量化压缩、激活值量化
> **msModelSlim 实现**：`msmodelslim/core/quantizer/impl/histogram.py`、`msmodelslim/core/observer/histogram.py`

---

## 1. 概述

Histogram（直方图）是一种激活值量化算法。它通过分析输入张量的直方图分布，自动搜索最优的截断区间（`clip_min`、`clip_max`），过滤离群值，避免量化范围过大导致的有效比特位利用率低。其核心特征是：直方图统计、非线性参数搜索最优截断区间、L2 范数或 KL 散度误差度量，通常作为 [线性量化](../linear_quant/term_linear_quant.md) 的激活值量化方法使用。

---

## 2. 词条介绍

传统 [MinMax](../minmax/term_minmax.md) 量化器容易受到离群值影响，导致量化范围过大、有效比特位利用率低、量化精度损失严重。Histogram 观察到，通过分析激活值的分布直方图并搜索最优截断区间，可以过滤离群值，在保持整体分布精度的同时提高量化精度。

---

## 3. 原理

### 1. 核心思想

Histogram 的核心思想是“用直方图刻画分布并搜索最优截断”：将输入张量的值域划分为固定数量的 bins（默认 2048）统计频次构建直方图，然后通过非线性参数搜索逐步移动截断区间，以量化误差（L2 范数或 KL 散度）作为度量，找到误差最小的截断区间作为量化范围。

### 2. 数学描述

直方图统计后，对截断区间 $[\text{start\_bin}, \text{end\_bin}]$ 进行搜索，以 L2 范数误差为目标：

$$
\text{err} = \left\| p - q \right\|_2
$$

KL 散度误差为：

$$
\text{KL} = \sum_i p_i \log \frac{p_i}{q_i}
$$

- $p$：原始分布（真实分布）
- $q$：量化后的分布
- $p_i$、$q_i$：第 $i$ 个 bin 的概率

搜索策略采用二分/迭代方式：初始化 $\alpha=0.0$（下界）、$\beta=1.0$（上界），每次移动固定百分位数（`stepsize=1e-5`），选择单次移动长度更长（分布更稀疏）的一边移动，在量化误差不再改善或边界越界时停止。得到最优截断值后计算量化参数：

$$
\text{scale} = \frac{\text{clip\_max} - \text{clip\_min}}{Q_{\max} - Q_{\min}}
$$

- $\text{clip\_min}$、$\text{clip\_max}$：最优截断区间的上下界
- $Q_{\min}$、$Q_{\max}$：量化数值范围
- $\text{scale}$：量化缩放因子

### 3. 关键性质

- **离群值过滤**：通过截断区间过滤离群值，避免量化范围过大。
- **分布感知**：基于直方图刻画分布，比 MinMax 更能反映整体分布形态。
- **误差度量可选**：支持 L2 范数与 KL 散度两种误差度量。
- **自适应搜索**：非线性参数搜索自动寻找最优截断区间。

---

## 4. 流程示意

> 以下为本算法在 msModelSlim 中的简化流程概览。

```mermaid
flowchart LR
    A[输入张量] --> B[直方图统计]
    B --> C[搜索截断区间]
    C --> D[计算量化参数]
    D --> E[伪量化输出]
```

---

## 5. 在 msModelSlim 中的实现

### 1. 实现位置

算法在 `msmodelslim/core/quantizer/impl/histogram.py` 与 `msmodelslim/core/observer/histogram.py` 中实现，核心组件为 `HistogramObserver` 与 `ActPerTensorHistogram`，作为 `linear_quant` 处理器的激活值量化方法（`method: "histogram"`）使用。

### 2. 处理流程

1. **直方图统计**：将输入张量值域划分为固定数量 bins（默认 2048），统计频次构建直方图，支持上采样（`upsample_rate=16`）。
2. **截断值搜索**：每次移动固定百分位数（`stepsize=1e-5`），通过计算量化误差评估候选区间。
3. **量化误差度量**：默认使用 L2 范数误差，也可使用 KL 散度误差。
4. **量化参数计算**：以最优截断区间上下界为 max/min，计算并保存 scale 与 zero_point，执行伪量化。

### 3. 配置示例

> 以下为最小可用的 YAML 配置片段。各字段的详细含义如下表所示。

```yaml
spec:
  process:
    - type: "linear_quant"
      qconfig:
        act:
          scope: "per_tensor"
          dtype: "int8"
          symmetric: false
          method: "histogram"
        weight:
          scope: "per_channel"
          dtype: "int8"
          symmetric: true
          method: "minmax"
```

**字段说明**：

| 字段名 | 作用 | 说明 |
| --- | --- | --- |
| qconfig.act.scope | 激活量化范围 | 仅支持 `"per_tensor"`。 |
| qconfig.act.dtype | 激活量化数据类型 | 仅支持 `"int8"`。 |
| qconfig.act.symmetric | 是否对称量化 | `true` 为对称，`false` 为非对称。 |
| qconfig.act.method | 量化方法 | 固定为 `"histogram"`，启用直方图激活值量化。 |

---

## 6. 适用场景与限制

### 1. 适用场景

- 激活值存在离群值、MinMax 量化精度损失较大的场景。
- 需要更高精度的激活值静态量化场景。

### 2. 使用限制

- 目前仅支持 per_tensor 量化范围与 int8 数据类型。
- 仅支持激活值量化，不支持权重量化（权重不应配置为 `method: "histogram"`）。
- 仅用于静态量化场景，需校准数据构建直方图。

---

## 7. 关联流程

- 《[一键量化 (V1)](../../../user_guide/usage_quick_quantization.md)》：可集成 Histogram 作为激活值量化步骤。
- 《[量化精度调优指南](../../../user_guide/process_quantization_precision_tuning.md)》：精度不达标时可考虑使用 Histogram 替代 MinMax。

---

## 8. 关联词条

- [线性量化](../linear_quant/term_linear_quant.md)：应用对象，Histogram 作为线性量化的激活值量化方法使用。
- [MinMax](../minmax/term_minmax.md)：对比算法，Histogram 是 MinMax 的离群值过滤优化。
- [SSZ](../ssz/term_ssz.md)：同类算法，同为量化参数搜索类算法（SSZ 针对权重）。
- [SmoothQuant](../smooth_quant/term_smooth_quant.md)：配套术语，可配合离群值抑制进一步提升精度。

---

## 9. 参考资料

1. 《Histogram 使用指南》([./usage_histogram_activation_quantization.md](./usage_histogram_activation_quantization.md))
