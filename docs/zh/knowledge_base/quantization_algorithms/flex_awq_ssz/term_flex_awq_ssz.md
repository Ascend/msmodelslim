# Flex AWQ SSZ 灵活激活感知权重量化平滑算法词条

> **词条类别**：离群值抑制算法
> **英文名称**：Flex AWQ SSZ
> **应用领域**：大语言模型量化压缩、低比特量化精度优化
> **msModelSlim 实现**：`msmodelslim/processor/anti_outlier/flex_smooth/`

---

## 1. 概述

Flex AWQ SSZ（灵活激活感知权重量化平滑）是一种用于大语言模型量化过程中抑制激活离群值的算法。它结合了 [AWQ](../awq_smooth/term_awq_smooth.md) 与 [SSZ](../ssz/term_ssz.md) 的思想，使用真实量化器（LinearQuantizer）评估不同 `alpha` 参数下的量化误差，自动搜索最优缩放因子，并固定 `beta` 为 `0`、以激活均值而非最大值计算激活尺度。其核心特征是：真实量化器评估、激活均值尺度、自动参数搜索。

---

## 2. 词条介绍

[Flex Smooth Quant](../flex_smooth_quant/term_flex_smooth_quant.md) 使用模拟量化评估参数，无法精确反映真实量化后的效果。Flex AWQ SSZ 改用真实的量化器评估不同参数下的量化误差，并针对低比特量化场景使用激活均值（而非最大值）计算激活尺度，从而在不同量化配置下获得更准确的参数选择。

---

## 3. 原理

### 1. 核心思想

Flex AWQ SSZ 的核心思想是“以真实量化器评估参数有效性”：对每个候选 `alpha` 值，计算缩放因子、应用缩放、创建真实量化器（LinearQuantizer）完成量化-反量化，并计算与浮点结果的归一化 MSE，选择使量化误差最小的 `alpha`。`beta` 固定为 `0`，激活尺度使用均值绝对值 `mean(abs(act))` 计算。

### 2. 数学描述

缩放因子计算公式：

$$
s = \left( \frac{\text{Act\_Mean\_Abs}^{\alpha}}{\text{Weight\_Max\_Abs}^{\beta}} \right) \cdot \operatorname{clamp}(\min=10^{-5}), \quad \beta = 0
$$

- $\text{Act\_Mean\_Abs}$：激活值的均值绝对值，即 $mean(|act|)$
- $\text{Weight\_Max\_Abs}$：权重的最大绝对值（取每列的最大值）
- $\alpha$：激活缩放系数，$0$~$1$，可自动搜索或手动配置
- $\beta$：权重缩放系数，固定为 $0$
- $10^{-5}$：缩放因子的最小值

当未配置 `alpha` 时，算法在 $[0.0, 1.0]$ 范围内以 $0.05$ 为步长搜索使量化 MSE 最小的 `alpha`。

### 3. 关键性质

- **真实量化器评估**：使用 LinearQuantizer 评估量化误差，结果更接近真实部署效果。
- **激活均值尺度**：使用 `mean(abs(act))` 而非最大值，更适合低比特量化场景。
- **参数空间精简**：固定 `beta=0`，仅搜索 `alpha`，降低搜索复杂度。
- **自动参数搜索**：未配置 `alpha` 时自动搜索最优值，减少人工调参。
- **计算等价性**：协同缩放保持整体计算等价，不改变模型输出。

---

## 4. 流程示意

> 以下为本算法在 msModelSlim 中的简化流程概览。

```mermaid
flowchart LR
    A[校准数据] --> B[子图发现]
    B --> C[收集激活均值]
    C --> D[搜索最优 alpha]
    D --> E[融合缩放]
    E --> F[交付量化]
```

---

## 5. 在 msModelSlim 中的实现

### 1. 实现位置

算法在 `msmodelslim/processor/anti_outlier/flex_smooth/processor.py` 中实现，通过 `type: "flex_awq_ssz"` 处理器使用，依赖 `qconfig` 中的真实量化器配置。

### 2. 处理流程

- **预处理阶段**：通过 `SubgraphProcessor` 获取子图信息，按 `include/exclude` 过滤，为线性模块安装前向钩子收集激活统计信息（使用子图 `targets` 中第一个线性层的激活统计）。
- **后处理阶段**：按优先级顺序处理各子图，创建 `FlexAWQSSZAlphaBetaSearcher` 搜索最优 `alpha`，应用缩放并融合；最后清理钩子并恢复模型。

### 3. 配置示例

> 以下为最小可用的 YAML 配置片段。各字段的详细含义如下表所示。

```yaml
spec:
  process:
    - type: "flex_awq_ssz"
      alpha: 0.8
      qconfig:
        act:
          scope: "per_token"
          dtype: "int8"
          symmetric: True
          method: "minmax"
        weight:
          scope: "per_channel"
          dtype: "int4"
          symmetric: True
          method: "ssz"
      enable_subgraph_type:
        - 'norm-linear'
        - 'linear-linear'
        - 'ov'
        - 'up-down'
      include: ["*"]
      exclude: ["*self_attn*"]
```

**字段说明**：

| 字段名 | 作用 | 说明 |
| --- | --- | --- |
| type | 处理器类型标识 | 固定为 `"flex_awq_ssz"`。 |
| alpha | 激活缩放权重系数 | 0~1 之间的浮点数，默认 `None`（自动搜索）。 |
| qconfig | 量化配置 | 必填参数，包含激活值（`act`）和权重（`weight`）的量化配置，用于真实量化器评估。 |
| enable_subgraph_type | 开启的子图类型 | 支持 `norm-linear`、`linear-linear`、`ov`、`up-down`。 |
| include | 包含的层 | 字符串列表，支持通配符匹配。 |
| exclude | 排除的层 | 字符串列表，支持通配符匹配。 |

### 4. 模型适配接口

模型适配需实现 `FlexSmoothQuantInterface` 接口（与 Flex Smooth Quant 相同）的 `get_adapter_config_for_subgraph()` 方法。参考实现：`msmodelslim/model/qwen3/model_adapter.py`。

---

## 6. 适用场景与限制

### 1. 适用场景

- 需要对平滑参数进行真实量化器评估、追求更高精度的低比特量化场景。
- 需要同时对注意力 `ov`、MLP `up-down`、连续线性层等多种结构做离群值抑制的场景。

### 2. 使用限制

- 模型必须实现 `FlexSmoothQuantInterface` 接口并正确配置子图映射。
- `qconfig` 为必填参数，需提供激活与权重的量化配置（权重通常使用 SSZ 方法）。
- 目标模块必须存在且具备可写的 `weight`，模块名须与 `named_modules()` 返回的完整路径一致。

---

## 7. 关联流程

- 《[一键量化 (V1)](../../../user_guide/usage_quick_quantization.md)》：默认集成本算法作为离群值抑制前置步骤。
- 《[量化精度调优指南](../../../user_guide/process_quantization_precision_tuning.md)》：精度不达标时可考虑启用本算法。

---

## 8. 关联词条

- [AWQ](../awq_smooth/term_awq_smooth.md)：上位概念，本算法结合 AWQ 的激活感知思想。
- [Flex Smooth Quant](../flex_smooth_quant/term_flex_smooth_quant.md)：同类算法，使用模拟量化评估参数。
- [SSZ](../ssz/term_ssz.md)：配套术语，本算法的 `qconfig.weight` 通常使用 SSZ 权重量化方法。
- [SmoothQuant](../smooth_quant/term_smooth_quant.md)：上位概念，本算法属于平滑类离群值抑制算法族。
- [AutoRound](../autoround/term_autoround.md)：配套术语，低比特量化前常配合本算法使用。

---

## 9. 参考资料

1. Lin J et al. AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration. MLSys 2024. https://arxiv.org/abs/2306.00978
2. 《Flex AWQ SSZ 使用指南》([./usage_flex_awq_ssz.md](./usage_flex_awq_ssz.md))
