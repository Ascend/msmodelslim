# Ceil_X 自适应除数 MXFP4 量化算法词条

> **词条类别**：量化算法
> **英文名称**：Ceil_X
> **英文缩写**：Ceil_X
> **应用领域**：MXFP4 权重量化、低比特量化精度优化
> **msModelSlim 实现**：`msmodelslim/core/quantizer/impl/ceil_x.py`

---

## 1. 概述

Ceil_X 是一种针对 MXFP4 per-block 权重量化的精度优化算法。它通过引入可配置除数 $c$ 配合 `ceil` 操作重新设计 shared exponent 的计算方式，将缩放后数值范围压缩至 MXFP4 的可表示区间内，减少大值截断，提升量化精度。其核心特征是：ceil + 可配置除数、可选全局 MSE 搜索最优除数、零配置开箱即用，是 [MinMax](../minmax/term_minmax.md) MXFP4 量化的截断抑制优化。

---

## 2. 词条介绍

MXFP4 per-block 量化中，传统方法使用 $s = \lfloor \log_2(\max(|x|)) \rfloor - e_{\text{max}}$ 计算 shared exponent，缩放后数值范围为 $[4, 8)$，而 MXFP4 的表示上限为 $6.0$，导致 $[6, 8)$ 区间内的大值被截断，引入较大量化误差。Ceil_X 通过引入可配置除数 $c$ 配合 `ceil` 操作，将缩放后数值范围压缩至 $(c/2, c]$，完全落在 MXFP4 的可表示区间内。

---

## 3. 原理

### 1. 核心思想

Ceil_X 的核心思想是“用 ceil + 可配置除数收紧缩放范围”：对每个 block，使用 `ceil(log2(max(|x|) / c))` 计算 shared exponent，其中 $c$ 是可配置的除数（`ceil_x_value`）。相比传统 floor 缩放，ceil 操作将缩放后数值范围从 $[4, 8)$ 压缩至 $(c/2, c]$，消除大值截断；当启用搜索模式时，在搜索范围内寻找使整体 MSE 最小的全局除数 $c$。

### 2. 数学描述

传统 floor 缩放：

$$
s_{\text{floor}} = \lfloor \log_2(\max(|x|)) \rfloor - e_{\text{max}}
$$

Ceil_X 缩放：

$$
s = \operatorname{ceil}\left(\log_2\left(\frac{\max(|x|)}{c} + \epsilon\right)\right) - e_{\text{max}}
$$

- $s$：shared exponent
- $\max(|x|)$：block 内权重绝对值的最大值
- $c$：可配置除数（`ceil_x_value`，默认 $7.25$）
- $\epsilon$：数值稳定项，$\epsilon = 9.6 \times 10^{-7}$
- $e_{\text{max}}$：指数偏置，MXFP4 E4M2 格式下 $e_{\text{max}} = 2^{e_{\text{bits}}-1} = 2$

缩放后 block 内最大值的量级：

$$
\frac{\max(|x|)}{2^s} = \frac{\max(|x|)}{2^{\operatorname{ceil}(\log_2(\max(|x|)/c))}} \in \left(\frac{c}{2}, c\right]
$$

默认 $c = 7.25$ 时，缩放后数值范围为 $(3.625, 7.25]$，相比传统方法的 $[4, 8)$：下界降低、截断比例显著减小。

可选的自适应搜索：

$$
c^* = \arg\min_{c \in [c_{\min}, c_{\max}]} \sum_{\text{blocks}} \|x - \hat{x}(c)\|^2
$$

- $c^*$：搜索得到的最优除数
- $c_{\min}$、$c_{\max}$：搜索范围（默认 $[6.0, 12.0]$）
- $\hat{x}(c)$：使用除数 $c$ 量化-反量化后的值

### 3. 关键性质

- **ceil 缩放**：使用 ceil 操作收紧缩放范围，避免大值截断。
- **可配置除数**：`ceil_x_value` 控制缩放收紧程度，默认 $7.25$。
- **自适应搜索**：可选启用全局 MSE 搜索最优除数 $c$。
- **零配置可用**：不启用搜索时无额外超参，开箱即用。

---

## 4. 流程示意

> 以下为本算法在 msModelSlim 中的简化流程概览。

```mermaid
flowchart LR
    A[分块处理] --> B[计算 ceil_x 指数]
    B --> C[量化-反量化]
    C --> D{启用搜索?}
    D -- 是 --> E[搜索最优除数]
    D -- 否 --> F[输出量化结果]
    E --> F
```

---

## 5. 在 msModelSlim 中的实现

### 1. 实现位置

算法在 `msmodelslim/core/quantizer/impl/ceil_x.py` 中实现，实现类为 `MXWeightPerBlockCeilX`，注册的量化类型为 `mxfp4_per_block_sym`，配置模型为 `CeilXExtConfig`，作为 `linear_quant` 处理器的权重量化方法（`method: "ceil_x"`）使用。

### 2. 处理流程

1. 使用 `minmax_block_observer` 计算 per-block min/max。
2. 计算 ceil_x shared exponent：`shared_exp = ceil(log2(max_val / ceil_x_value + 9.6e-7))`，并进行 clip 限制。
3. 量化权重。
4. 启用 `enable_search` 时，在 `[search_min, search_max]` 内以 `search_step` 步长搜索使整体 MSE 最小的 `ceil_x_value`。

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
          method: "ceil_x"
          ext:
            ceil_x_value: 7.25
            enable_search: false
```

**字段说明**：

| 字段名 | 作用 | 说明 |
| --- | --- | --- |
| scope | 量化范围 | 固定为 `"per_block"`。 |
| dtype | 量化数据类型 | 固定为 `"mxfp4"`。 |
| symmetric | 是否对称量化 | `true`。 |
| method | 量化方法 | 固定为 `"ceil_x"`。 |
| ext.ceil_x_value | 除数 | 取值范围 `[6.0, 12.0]`，默认 `7.25`。 |
| ext.enable_search | 是否启用 MSE 搜索 | `true`/`false`，默认 `false`。 |
| ext.search_min | 搜索范围下限 | 默认 `6.0`。 |
| ext.search_max | 搜索范围上限 | 须大于 `search_min`，默认 `12.0`。 |
| ext.search_step | 搜索步长 | 大于 0，默认 `0.25`。 |

---

## 6. 适用场景与限制

### 1. 适用场景

- 对 mxFP4 量化精度有更高要求的场景。
- 权重分布范围较大、floor 缩放导致步长过粗或大值截断的模型层。

### 2. 使用限制

- 仅支持 mxFP4 格式的 per_block 对称量化。
- 启用 `enable_search` 时增加若干次前向量化评估，计算开销增大。
- `search_max` 须大于 `search_min`。

---

## 7. 关联流程

- 《[一键量化 (V1)](../../../user_guide/usage_quick_quantization.md)》：可集成 Ceil_X 作为 MXFP4 权重量化步骤。
- 《[量化精度调优指南](../../../user_guide/process_quantization_precision_tuning.md)》：精度不达标时可考虑使用 Ceil_X。

---

## 8. 关联词条

- [线性量化](../linear_quant/term_linear_quant.md)：应用对象，Ceil_X 作为线性量化的权重量化方法使用。
- [FouroverSix](../fouroversix/term_fouroversix.md)：同类算法，同为 MXFP4 格式的缩放策略优化算法。
- [MSE_Round](../mse_round/term_mse_round.md)：同类算法，同为 MXFP 格式的 shared exponent 优化算法（针对 MXFP8）。
- [MinMax](../minmax/term_minmax.md)：对比算法，Ceil_X 是 MinMax MXFP4 量化的截断抑制优化。

---

## 9. 参考资料

1. 《Ceil_X 使用指南》([./usage_ceil_x.md](./usage_ceil_x.md))
