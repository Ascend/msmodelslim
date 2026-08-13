# FouroverSix 自适应块缩放量化算法词条

> **词条类别**：量化算法
> **英文名称**：FouroverSix
> **英文缩写**：FouroverSix
> **应用领域**：MXFP4 权重量化、低比特量化精度优化
> **msModelSlim 实现**：`msmodelslim/core/quantizer/impl/fouroversix.py`

---

## 1. 概述

FouroverSix（4-over-6）是一种针对 mxFP4 per-block 权重量化的自适应块缩放算法。它通过对每个数据块在 Scale-to-6（缩放到 FP4 最大值 6）与 Scale-to-4（缩放到 4）两种方案间进行 MSE 择优，自适应选择最优的缩放方案，提升量化精度。其核心特征是：双路径评估、per-block 智能择优、e8m0 指数舍入，是 [MinMax](../minmax/term_minmax.md) MXFP4 量化的自适应优化。

---

## 2. 词条介绍

传统的 mxFP4 量化方法在处理不同分布的数据块时，统一使用固定的缩放因子（通常将最大值缩放到 FP4 的上限 6），可能导致部分块的量化误差较大。FouroverSix 观察到，当数据块内存在离群值或分布不均匀时，使用较小的缩放因子（如缩放到 4）可以显著降低量化误差，因此通过 per-block 自适应选择最优缩放方案。

---

## 3. 原理

### 1. 核心思想

FouroverSix 的核心思想是“每个数据块自适应选择缩放目标”：对每个数据块同时尝试两种缩放方案——方案 A（Scale-to-6）将块内最大值缩放到 FP4 格式的最大值 6 以充分利用动态范围，方案 B（Scale-to-4）将块内最大值缩放到 4 以提供更多余量——计算两种方案的均方误差（MSE），选择误差较小的方案作为该块的最终量化方案。

### 2. 数学描述

方案 A（Scale-to-6）与方案 B（Scale-to-4）的缩放因子：

$$
\text{scale\_a} = \frac{\max\_per\_block}{6.0}, \quad \text{scale\_b} = \frac{\max\_per\_block}{4.0}
$$

- $\max\_per\_block$：block 内权重绝对值的最大值

缩放因子使用 e8m0 格式存储，通过最近邻舍入（含银行家舍入规则）转换为 `scale_E_a`、`scale_E_b`。

分别计算两种方案的 MSE：

$$
\text{MSE}_a = \frac{1}{N}\sum_{i=1}^{N}(w_i - \hat{w}_i^{(a)})^2, \quad \text{MSE}_b = \frac{1}{N}\sum_{i=1}^{N}(w_i - \hat{w}_i^{(b)})^2
$$

- $w_i$：block 内权重值
- $\hat{w}_i^{(a)}$、$\hat{w}_i^{(b)}$：方案 A/B 量化-反量化后的值
- $N$：block 内元素个数

最终选择 MSE 较小的方案：

$$
\text{scale} = \begin{cases} \text{scale\_E\_a} & \text{if } \text{MSE}_a \le \text{MSE}_b \\ \text{scale\_E\_b} & \text{otherwise} \end{cases}
$$

指数舍入策略（e8m0 格式）：尾数 > 0.5 时指数加 1，尾数 < 0.5 时指数保持不变，尾数 == 0.5 时采用银行家舍入规则（偶数进 1、奇数不进）。

### 3. 关键性质

- **自适应缩放**：每个 block 在 6 与 4 之间智能选择缩放目标。
- **双路径评估**：同时评估 Scale-to-6 与 Scale-to-4 两种方案的量化误差。
- **银行家舍入**：e8m0 指数采用最近邻舍入（含银行家舍入规则），确保精度。
- **格式兼容**：保持 mxFP4 量化格式不变，仅优化缩放策略。

---

## 4. 流程示意

> 以下为本算法在 msModelSlim 中的简化流程概览。

```mermaid
flowchart LR
    A[分块处理] --> B[计算两种缩放]
    B --> C[两方案量化]
    C --> D[计算 MSE]
    D --> E[选择更优方案]
```

---

## 5. 在 msModelSlim 中的实现

### 1. 实现位置

算法在 `msmodelslim/core/quantizer/impl/fouroversix.py` 中实现，实现类为 `WeightFouroverSixQuantizer`，注册的量化类型为 `mxfp4_per_block_sym`，作为 `linear_quant` 处理器的权重量化方法（`method: "fouroversix"`）使用。

### 2. 处理流程

1. 计算 block 内最大绝对值。
2. 计算 Scale-to-6 与 Scale-to-4 两种方案的缩放因子，并通过最近邻舍入转换为 e8m0 格式。
3. 分别量化-反量化并计算 MSE。
4. 选择 MSE 较小的方案作为该 block 的最终量化方案。

### 3. 配置示例

> 以下为最小可用的 YAML 配置片段。各字段的详细含义如下表所示。

```yaml
spec:
  process:
    - type: "linear_quant"
      qconfig:
        weight:
          scope: "per_block"
          dtype: "mxfp4"
          symmetric: true
          method: "fouroversix"
```

**字段说明**：

| 字段名 | 作用 | 说明 |
| --- | --- | --- |
| scope | 量化范围 | 固定为 `"per_block"`。 |
| dtype | 量化数据类型 | 固定为 `"mxfp4"`。 |
| symmetric | 是否对称量化 | `true` 为对称，`false` 为非对称。 |
| method | 量化方法 | 固定为 `"fouroversix"`。 |

---

## 6. 适用场景与限制

### 1. 适用场景

- 对精度要求较高的 mxFP4 量化场景，特别适合处理数据分布不均匀的模型。
- 注意力层和前馈网络层权重分布差异较大的 Transformer 模型，以及多模态模型。

### 2. 使用限制

- 仅支持 mxFP4 格式的 per_block 对称量化。
- 权重必须为 2D 张量。
- 需要对每个块执行两次量化/反量化并计算 MSE，计算量略高于传统 mxFP4 量化。

---

## 7. 关联流程

- 《[一键量化 (V1)](../../../user_guide/usage_quick_quantization.md)》：可集成 FouroverSix 作为 MXFP4 权重量化步骤。
- 《[量化精度调优指南](../../../user_guide/process_quantization_precision_tuning.md)》：精度不达标时可考虑使用 FouroverSix。

---

## 8. 关联词条

- [线性量化](../linear_quant/term_linear_quant.md)：应用对象，FouroverSix 作为线性量化的权重量化方法使用。
- [Ceil_X](../ceil_x/term_ceil_x.md)：同类算法，同为 MXFP4 格式的缩放策略优化算法。
- [MSE_Round](../mse_round/term_mse_round.md)：同类算法，同为 per-block 双候选择优的量化算法（针对 MXFP8）。
- [MinMax](../minmax/term_minmax.md)：对比算法，FouroverSix 是 MinMax MXFP4 量化的自适应优化。

---

## 9. 参考资料

1. 《FouroverSix 使用指南》([./usage_fouroversix.md](./usage_fouroversix.md))
2. 自适应块缩放相关论文：https://arxiv.org/abs/2512.02010
