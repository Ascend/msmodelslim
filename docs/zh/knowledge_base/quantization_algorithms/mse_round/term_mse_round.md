# MSE_Round 权重量化算法词条

> **词条类别**：量化算法
> **英文名称**：MSE_Round
> **英文缩写**：MSE_Round
> **应用领域**：MXFP8 权重量化、低比特量化精度优化
> **msModelSlim 实现**：`msmodelslim/core/quantizer/impl/mse_round.py`

---

## 1. 概述

MSE_Round 是一种针对 MXFP8 per-block 权重量化的精度优化算法。它通过在每个 block 内对 ceil 与 floor 两档 shared exponent 进行实际量化-反量化 MSE 比较，自适应选择误差更小的缩放方案，避免大值截断。其核心特征是：per-block 双候选比较、MSE 择优、零配置开箱即用，作为 [线性量化](../linear_quant/term_linear_quant.md) 的权重量化方法使用。

---

## 2. 词条介绍

MXFP8 per-block 量化中，传统 minmax 方法固定使用 $s = \lfloor \log_2(\max(|x|)) \rfloor - e_{\text{max}}$ 计算 shared exponent，缩放后 block 内最大值的量级落在 $[8, 16)$。当 block 内最大值接近上界时，缩放结果可能超出 MXFP8 的表示上限（$448$），引发截断并引入较大量化误差。MSE_Round 通过 per-block 比较 ceil 与 floor 两档的实际量化误差，在避免截断与保持精度之间取得更优平衡。

---

## 3. 原理

### 1. 核心思想

MSE_Round 的核心思想是“per-block 两档候选 MSE 择优”：对每个 block 同时计算 ceil 与 floor 两档候选 shared exponent，分别完成量化-反量化并计算 block 内 MSE，选择 MSE 更小的 shared exponent 作为最终量化参数。ceil 档可避免大值截断，floor 档在分布均匀时可能获得更优的整体 MSE。

### 2. 数学描述

传统 floor 缩放：

$$
s_{\text{floor}} = \lfloor \log_2(\max(|x|)) \rfloor - e_{\text{max}}
$$

MSE_Round 同时计算 ceil 与 floor 两档候选：

$$
s_{\text{ceil}} = \lceil \log_2(\max(|x|)) \rceil - e_{\text{max}}
$$

$$
s_{\text{floor}} = \lfloor \log_2(\max(|x|)) \rfloor - e_{\text{max}}
$$

分别计算 block 内 MSE：

$$
\text{MSE}_{\text{ceil}} = \frac{1}{N}\sum_{i=1}^{N}(x_i - \hat{x}_i(s_{\text{ceil}}))^2, \quad \text{MSE}_{\text{floor}} = \frac{1}{N}\sum_{i=1}^{N}(x_i - \hat{x}_i(s_{\text{floor}}))^2
$$

最终 per-block 选择：

$$
s^* = \begin{cases} s_{\text{ceil}} & \text{if } \text{MSE}_{\text{ceil}} < \text{MSE}_{\text{floor}} \\ s_{\text{floor}} & \text{otherwise} \end{cases}
$$

- $x$：block 内的权重值
- $\hat{x}(s)$：使用 shared exponent $s$ 量化-反量化后的值
- $N$：block 内元素个数
- $e_{\text{max}}$：指数偏置，MXFP8 E4M3 格式下 $e_{\text{max}} = 2^{e_{\text{bits}}-1} = 8$

当某一候选的 shared exponent 超出 E8M0 表示范围（被标记为 NaN）时，自动回退至另一有效候选。

### 3. 关键性质

- **双候选缩放**：per-block 在 ceil/floor 两档间 MSE 择优。
- **避免截断**：ceil 档将缩放后范围压缩至 $(4, 8]$，避免大值截断。
- **零配置**：无需额外超参搜索，开箱即用。
- **计算可控**：每个 block 执行两次量化-反量化评估，计算量约为标准 minmax 的2倍。

---

## 4. 流程示意

> 以下为本算法在 msModelSlim 中的简化流程概览。

```mermaid
flowchart LR
    A[分块处理] --> B[计算 ceil/floor]
    B --> C[两档量化-反量化]
    C --> D[计算 MSE]
    D --> E[选择更优缩放]
```

---

## 5. 在 msModelSlim 中的实现

### 1. 实现位置

算法在 `msmodelslim/core/quantizer/impl/mse_round.py` 中实现，实现类为 `MXWeightPerBlockMseRound`，注册的量化类型为 `mxfp8_per_block_sym`，作为 `linear_quant` 处理器的权重量化方法（`method: "mse_round"`）使用。

### 2. 处理流程

1. 使用 `minmax_block_observer` 计算 per-block max。
2. 计算 ceil 与 floor 两档候选 shared exponent。
3. 分别量化-反量化，计算 block 内 MSE。
4. 通过 `select_by_mse` 选取 MSE 更小的 shared exponent。

### 3. 配置示例

> 以下为最小可用的 YAML 配置片段。各字段的详细含义如下表所示。

```yaml
spec:
  process:
    - type: "linear_quant"
      qconfig:
        act:
          scope: "per_block"
          dtype: "mxfp8"
          symmetric: true
          method: "minmax"
        weight:
          scope: "per_block"
          dtype: "mxfp8"
          symmetric: true
          method: "mse_round"
```

**字段说明**：

| 字段名 | 作用 | 说明 |
| --- | --- | --- |
| scope | 量化范围 | 固定为 `"per_block"`。 |
| dtype | 量化数据类型 | 固定为 `"mxfp8"`。 |
| symmetric | 是否对称量化 | `true`。 |
| method | 量化方法 | 固定为 `"mse_round"`。 |

---

## 6. 适用场景与限制

### 1. 适用场景

- 对 MXFP8 权重量化精度有更高要求的场景。
- block 内最大值分布不均、floor 缩放导致大值截断的模型层。

### 2. 使用限制

- 当前仅注册于 `mxfp8_per_block_sym` 权重量化方案，仅支持权重量化。
- 激活值量化请继续使用 `minmax` 等已有方法。

---

## 7. 关联流程

- 《[一键量化 (V1)](../../../user_guide/usage_quick_quantization.md)》：可集成 MSE_Round 作为 MXFP8 权重量化步骤。
- 《[量化精度调优指南](../../../user_guide/process_quantization_precision_tuning.md)》：精度不达标时可考虑使用 MSE_Round。

---

## 8. 关联词条

- [线性量化](../linear_quant/term_linear_quant.md)：应用对象，MSE_Round 作为线性量化的权重量化方法使用。
- [Ceil_X](../ceil_x/term_ceil_x.md)：同类算法，同为 MXFP 格式的 shared exponent 优化算法（针对 MXFP4）。
- [FouroverSix](../fouroversix/term_fouroversix.md)：同类算法，同为 per-block 双候选择优的量化算法（针对 MXFP4）。
- [MinMax](../minmax/term_minmax.md)：对比算法，MSE_Round 是 MinMax MXFP8 量化的精度优化。

---

## 9. 参考资料

1. 《MSE_Round 使用指南》([./usage_mse_round.md](./usage_mse_round.md))
