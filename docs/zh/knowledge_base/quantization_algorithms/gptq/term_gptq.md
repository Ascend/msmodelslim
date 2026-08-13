# GPTQ 权重量化算法词条

> **词条类别**：量化算法
> **英文名称**：GPTQ
> **英文缩写**：GPTQ
> **首次提出**：Frantar et al., ICLR 2023
> **应用领域**：大语言模型量化压缩、高精度权重量化
> **msModelSlim 实现**：`msmodelslim/core/quantizer/impl/gptq.py`

---

## 1. 概述

GPTQ 是一种高精度权重量化算法。它通过逐层逐列对权重进行量化，根据量化误差和激活值的 Hessian 矩阵对未量化的权重进行补偿，以此达到整体量化权重误差最小的目的。其核心特征是：逐层独立量化避免误差积累、基于二阶信息的误差修正、分块量化降低计算复杂度，常用于对精度要求较高的权重量化场景，与 [AutoRound](../autoround/term_autoround.md) 同属高精度权重量化优化算法。

---

## 2. 词条介绍

传统量化方法（如 [MinMax](../minmax/term_minmax.md)）在权重分布不均匀时，量化误差较大，影响模型精度。GPTQ 观察到，量化当前列的误差可以通过调整后续未量化列的权重进行补偿，从而在逐列量化的过程中持续修正累积误差，达到最小化整体量化误差的目的。

---

## 3. 原理

### 1. 核心思想

GPTQ 的核心思想是“用二阶信息补偿量化误差”：逐层独立量化以避免误差跨层积累；利用激活值的 Hessian 矩阵评估权重量化对输出的影响，动态调整未量化权重以补偿误差；将权重划分成多个块以减少计算复杂度，并采用惰性批量更新合并更新操作。

### 2. 数学描述

GPTQ 以最小化权重量化引入的层输出误差为目标：

$$
\min_{\hat{W}} \| WX - \hat{W}X \|_2^2
$$

其中对未量化权重 $w$ 的更新公式为：

$$
\delta = -\frac{w_q - w}{[H^{-1}]_{qq}} \cdot (H^{-1})_{:,q}
$$

- $W$：原始权重矩阵
- $\hat{W}$：量化后的权重矩阵
- $X$：该层的输入激活
- $H$：Hessian 矩阵，$H = X^T X$（近似）
- $q$：当前被量化的列索引
- $w_q$：第 $q$ 列的量化值
- $[H^{-1}]_{qq}$：$H^{-1}$ 的第 $q$ 个对角元素

在实际实现中，Hessian 逆矩阵计算中加入阻尼项 `percdamp`（对角添加 $\max(\operatorname{diag}(H)) \times \text{percdamp}$）以改善数值稳定性。

### 3. 关键性质

- **逐层量化**：对模型每一层独立量化，避免误差跨层积累。
- **二阶误差修正**：利用 Hessian 矩阵动态调整未量化权重以补偿量化误差。
- **分块量化**：`block_size` 控制每次迭代处理的列块大小，平衡计算效率与显存。
- **分组量化**：支持 `group_size` 分组共享量化参数，平衡精度与模型体积。

---

## 4. 流程示意

> 以下为本算法在 msModelSlim 中的简化流程概览。

```mermaid
flowchart LR
    A[收集激活] --> B[构建 Hessian]
    B --> C[逐列量化]
    C --> D[误差补偿]
    D --> E[分组更新]
    E --> F[输出量化权重]
```

---

## 5. 在 msModelSlim 中的实现

### 1. 实现位置

算法在 `msmodelslim/core/quantizer/impl/gptq.py` 中实现，包括 `WeightPerChannelGPTQ`（per_channel）与 `WeightPerGroupGPTQ`（per_group）两个实现类，作为 `linear_quant` 处理器的权重量化方法（`method: "gptq"`）使用。

### 2. 处理流程

GPTQ 作为 `linear_quant` 处理器 `qconfig.weight` 中的量化方法使用。量化前收集层激活构建 Hessian 矩阵，随后逐列量化权重并通过误差补偿更新未量化权重，支持 `per_channel` 与 `per_group` 两种量化粒度。

### 3. 配置示例

> 以下为最小可用的 YAML 配置片段。各字段的详细含义如下表所示。

```yaml
spec:
  process:
    - type: "linear_quant"
      qconfig:
        weight:
          scope: "per_channel"
          dtype: "int8"
          symmetric: true
          method: "gptq"
          ext:
            percdamp: 0.01
            block_size: 128
```

**字段说明**：

| 字段名 | 作用 | 说明 |
| --- | --- | --- |
| scope | 量化范围 | `"per_channel"` 或 `"per_group"`。 |
| dtype | 量化数据类型 | `"int8"`（当前仅支持 int8）。 |
| symmetric | 是否对称量化 | `true` 为对称，`false` 为非对称。 |
| method | 量化方法 | 固定为 `"gptq"`。 |
| ext.percdamp | 阻尼系数 | 逆 Hessian 计算中的阻尼百分比，默认 `0.01`。 |
| ext.block_size | 迭代分块大小 | 每次迭代处理的列块大小，默认 `128`。 |
| ext.group_size | 量化分组大小 | 分组量化的大小，默认 `128`。 |

---

## 6. 适用场景与限制

### 1. 适用场景

- 对精度要求较高的权重量化场景。
- 权重分布不均匀、传统 MinMax 量化误差较大的场景。

### 2. 使用限制

- 目前支持 int8 场景的 per_channel/per_group 对称与非对称量化，暂不支持 int4。
- per_tensor 量化粒度暂不支持。
- 权重必须为 2D 张量。
- 依赖模型激活值，MoE 模型量化要求校准集覆盖所有专家，此场景不推荐使用。

---

## 7. 关联流程

- 《[一键量化 (V1)](../../../user_guide/usage_quick_quantization.md)》：可集成 GPTQ 作为高精度权重量化步骤。
- 《[量化精度调优指南](../../../user_guide/process_quantization_precision_tuning.md)》：精度不达标时可考虑使用 GPTQ 提升精度。

---

## 8. 关联词条

- [MinMax](../minmax/term_minmax.md)：对比算法，GPTQ 是 MinMax 的高精度优化扩展。
- [AutoRound](../autoround/term_autoround.md)：同类算法，同为高精度权重量化优化算法。
- [SSZ](../ssz/term_ssz.md)：同类算法，同为低比特权重量化优化算法。
- [线性量化](../linear_quant/term_linear_quant.md)：应用对象，GPTQ 作为线性量化处理器的权重量化方法使用。

---

## 9. 参考资料

1. Frantar E et al. GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers. ICLR 2023. https://arxiv.org/abs/2210.17323
2. 《GPTQ 使用指南》([./usage_gptq.md](./usage_gptq.md))
