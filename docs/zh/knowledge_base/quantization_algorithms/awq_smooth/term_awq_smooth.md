# AWQ 激活感知权重量化算法词条

> **词条类别**：离群值抑制算法
> **英文名称**：AWQ Smooth
> **英文缩写**：AWQ
> **中文别名**：激活感知权重量化
> **首次提出**：Lin et al., MLSys 2024
> **应用领域**：大语言模型量化压缩、低比特量化精度优化
> **msModelSlim 实现**：`msmodelslim/processor/anti_outlier/awq/`

---

## 1. 概述

AWQ（Activation-aware Weight Quantization，激活感知权重量化）是一种用于大语言模型量化过程中抑制激活离群值的算法。它通过观察激活值的统计特征，用激活均值度量各权重通道的重要性，并通过网格搜索找到使量化结果与浮点基准之间均方误差最小的缩放因子。其核心特征是：激活感知的重要性评估、网格搜索最优缩放、块级误差评估，常作为 [MinMax](../minmax/term_minmax.md) 等权重量化的前置步骤。

---

## 2. 词条介绍

并非所有权重通道对模型输出同等重要。AWQ 观察到，通过激活值分布可以识别重要权重通道并给予保护：对重要通道施加更小的量化扰动，可以在低比特量化场景下获得更优的精度表现。与 [SmoothQuant](../smooth_quant/term_smooth_quant.md) 的固定缩放不同，AWQ 通过网格搜索在激活统计的指导下寻找最优缩放因子。

---

## 3. 原理

### 1. 核心思想

AWQ 的核心思想是“按激活重要性保护权重通道”：使用激活值绝对值的逐通道均值度量通道重要性，在 $[0, 1)$ 范围内以网格搜索遍历 `ratio` 参数，用真实的权重量化器评估缩放后的量化误差，选择均方误差最小的缩放因子，并通过最低公共祖先（LCA）在块级别评估误差。

### 2. 数学描述

缩放因子的计算公式为：

$$
s = \frac{\operatorname{act\_mean}^{\text{ratio}}}{\sqrt{\max(s) \cdot \min(s)}}, \quad s = \operatorname{clamp}(s, \min=10^{-4})
$$

- $s$：逐通道缩放因子
- $\text{act\_mean}$：激活值绝对值的逐通道均值，即 $mean(|act|)$，反映各通道重要性
- $\text{ratio}$：缩放比例系数，在 $[0, 1)$ 范围内以 $1 / n\_grid$ 为步长网格搜索
- $n\_grid$：网格搜索步数，默认 $20$
- $10^{-4}$：缩放因子的最小值

块级误差评估使用 MSE：

$$
\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_{\text{float}}^{(i)} - y_{\text{quant}}^{(i)})^2
$$

- $y_{\text{float}}$：使用原始浮点权重在祖先模块上的输出
- $y_{\text{quant}}$：使用量化权重在祖先模块上的输出

### 3. 关键性质

- **激活感知**：基于激活均值识别重要通道并给予保护。
- **网格搜索**：在 $[0, 1)$ 范围内搜索最优 `ratio`，步长 $1/n\_grid$。
- **真实量化器评估**：使用真实权重量化器评估候选缩放因子的量化误差。
- **块级评估**：通过自动发现的最低公共祖先模块在块级别评估误差，而非单层权重误差。

---

## 4. 流程示意

> 以下为本算法在 msModelSlim 中的简化流程概览。

```mermaid
flowchart LR
    A[校准数据] --> B[收集激活均值]
    B --> C[搜索最优缩放]
    C --> D[块级误差评估]
    D --> E[融合缩放]
    E --> F[交付量化]
```

---

## 5. 在 msModelSlim 中的实现

### 1. 实现位置

算法在 `msmodelslim/processor/anti_outlier/awq/` 目录下实现，核心类包括 `AWQProcessor`、`AWQBestScalesSearcher`、`AWQStatsCollector`，通过 `type: "awq"` 处理器使用。

### 2. 处理流程

- **预处理阶段**：通过 `AWQInterface.get_adapter_config_for_subgraph()` 获取子图配置，按 `include/exclude` 过滤，为目标线性层安装 forward hook 收集激活均值，并通过 LCA 自动发现祖先模块、安装 forward pre-hook 缓存输入参数。
- **后处理阶段**：按优先级处理子图（`up-down`、`ov`、`norm-linear`、`linear-linear`），调用 `AWQBestScalesSearcher.search()` 搜索最优缩放因子，通过 `SubgraphFusionFactory` 融合到子图，最后清理 hook。

### 3. 配置示例

> 以下为最小可用的 YAML 配置片段。各字段的详细含义如下表所示。

```yaml
spec:
  process:
    - type: "awq"
      weight_qconfig:
        scope: "per_channel"
        dtype: "int8"
        symmetric: true
        method: "minmax"
      n_grid: 20
      enable_subgraph_type:
        - "norm-linear"
        - "linear-linear"
        - "ov"
        - "up-down"
      include: ["*"]
      exclude: []
```

**字段说明**：

| 字段名 | 作用 | 说明 |
| --- | --- | --- |
| type | 处理器类型标识 | 固定为 `"awq"`。 |
| weight_qconfig | 权重量化配置 | AWQ 搜索阶段使用的权重量化配置，字段定义与 `linear_quant` 的 `qconfig.weight` 一致。 |
| n_grid | 网格搜索步数 | 正整数，默认 `20`，数值越大搜索越细致但耗时增加。 |
| enable_subgraph_type | 启用的子图类型 | 支持 `norm-linear`、`linear-linear`、`ov`、`up-down`。 |
| include | 包含的层 | 字符串列表，支持通配符匹配。 |
| exclude | 排除的层 | 字符串列表，支持通配符匹配，优先级高于 `include`。 |

### 4. 模型适配接口

模型适配需实现 `AWQInterface` 接口的 `get_adapter_config_for_subgraph()` 方法，返回 `List[AdapterConfig]`（含 `subgraph_type` 与 `mapping`）。参考实现：`msmodelslim/model/qwen2/model_adapter.py`。

---

## 6. 适用场景与限制

### 1. 适用场景

- 需要自动搜索最优权重缩放因子、保护重要通道的低比特量化场景。
- 作为权重量化的前置步骤，为 [MinMax](../minmax/term_minmax.md)、[SSZ](../ssz/term_ssz.md) 等权重量化算法提供更优的权重分布。

### 2. 使用限制

- 模型适配器需要实现 `AWQInterface` 接口。
- 配置中的模块名称必须与 `named_modules()` 返回的完整路径一致。
- 目标模块必须存在且具备可写的 `weight`。
- 依赖 `ContextManager` 提供全局上下文。

---

## 7. 关联流程

- 《[一键量化 (V1)](../../../user_guide/usage_quick_quantization.md)》：默认集成本算法作为离群值抑制前置步骤。
- 《[量化精度调优指南](../../../user_guide/process_quantization_precision_tuning.md)》：精度不达标时可考虑启用本算法。

---

## 8. 关联词条

- [SmoothQuant](../smooth_quant/term_smooth_quant.md)：同类算法，同属离群值抑制算法族，但采用固定缩放而非激活感知搜索。
- [Flex AWQ SSZ](../flex_awq_ssz/term_flex_awq_ssz.md)：同类算法，本算法的扩展，使用真实量化器评估参数。
- [MinMax](../minmax/term_minmax.md)：配套术语，AWQ 搜索阶段使用 `weight_qconfig` 指定的量化配置。
- [SSZ](../ssz/term_ssz.md)：配套术语，常与 AWQ 的平滑配合用于权重量化。
- [QuaRot](../quarot/term_quarot.md)：对比算法，采用正交旋转而非缩放抑制离群值。

---

## 9. 参考资料

1. Lin J et al. AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration. MLSys 2024. https://arxiv.org/abs/2306.00978
2. 《AWQ 使用指南》([./usage_awq_smooth.md](./usage_awq_smooth.md))
