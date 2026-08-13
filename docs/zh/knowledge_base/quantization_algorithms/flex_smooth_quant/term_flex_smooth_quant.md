# Flex Smooth Quant 灵活平滑量化算法词条

> **词条类别**：离群值抑制算法
> **英文名称**：Flex Smooth Quant
> **应用领域**：大语言模型量化压缩、推理加速
> **msModelSlim 实现**：`msmodelslim/processor/anti_outlier/flex_smooth/`

---

## 1. 概述

Flex Smooth Quant（灵活平滑量化）是一种用于大语言模型量化过程中抑制激活离群值的算法。它通过二阶段网格搜索自动寻找最优的 `alpha` 与 `beta` 参数，在激活与权重之间实现更精细的缩放平衡，从而适配不同模型架构与量化需求。其核心特征是：参数可自动搜索、支持 `norm-linear`、`linear-linear`、`ov`、`up-down` 多子图类型，并支持对独立线性层做非融合平滑，是 [SmoothQuant](../smooth_quant/term_smooth_quant.md) 的灵活扩展。

---

## 2. 词条介绍

传统 [SmoothQuant](../smooth_quant/term_smooth_quant.md) 使用固定的 `alpha` 且仅支持 `norm-linear` 子图，对复杂结构适配不足。Flex Smooth Quant 将缩放公式推广为激活与权重分别使用 `alpha` 与 `beta` 两个指数，并通过网格搜索自动寻找最优参数，避免了人工调参，同时扩展了对多子图类型的支持。

---

## 3. 原理

### 1. 核心思想

Flex Smooth Quant 的核心思想是把平滑缩放公式从“单一 `alpha`”扩展为“`alpha` + `beta` 双指数”：当用户不指定参数时，通过二阶段网格搜索在激活与权重之间寻找最优的缩放平衡，使不同子图结构都能获得接近最优的离群值抑制效果。

### 2. 数学描述

缩放因子计算公式：

$$
s = \left( \frac{A_{\text{scale}}^{\alpha}}{W_{\text{scale}}^{\beta}} \right) \cdot \operatorname{clamp}(\min=10^{-5})
$$

- $s$：逐通道缩放因子
- $A_{\text{scale}}$：激活值的每通道缩放统计量
- $W_{\text{scale}}$：权重的每列最大值
- $\alpha$：激活缩放系数，控制激活对缩放因子的影响程度（$0$~$1$）
- $\beta$：权重缩放系数，控制权重对缩放因子的影响程度（$0$~$1$）
- $10^{-5}$：缩放因子的最小值，防止数值不稳定

当 `alpha`/`beta` 未配置时，算法在参数空间内以网格搜索方式评估候选组合，选择量化误差最小的参数。

### 3. 关键性质

- **双指数缩放**：激活与权重分别使用 `alpha` 与 `beta`，缩放更精细。
- **自动搜索**：未配置参数时自动搜索最优 `alpha`/`beta`，减少人工调参。
- **子图类型多样**：支持 `norm-linear`、`linear-linear`、`ov`、`up-down` 四种子图。
- **非融合能力**：`source=None` 时对独立线性层做输入侧 pre-hook 缩放。
- **计算等价性**：协同缩放保持整体计算等价，不改变模型输出。

---

## 4. 流程示意

> 以下为本算法在 msModelSlim 中的简化流程概览。

```mermaid
flowchart LR
    A[校准数据] --> B[子图发现]
    B --> C[统计激活与权重]
    C --> D[搜索 alpha/beta]
    D --> E[融合缩放]
    E --> F[交付量化]
```

---

## 5. 在 msModelSlim 中的实现

### 1. 实现位置

算法在 `msmodelslim/processor/anti_outlier/flex_smooth/processor.py` 中实现，通过 `type: "flex_smooth_quant"` 处理器使用。

### 2. 处理流程

- **预处理阶段**：通过 `SubgraphProcessor` 获取子图信息，按 `include/exclude` 过滤，为线性模块安装前向钩子收集激活张量与每通道绝对最大值。
- **后处理阶段**：按优先级顺序处理各子图；对非融合子图做 `alpha`/`beta` 搜索（或使用配置值）后对权重做缩放并注册输入侧 pre-hook；最后清理钩子并恢复模型。

### 3. 配置示例

> 以下为最小可用的 YAML 配置片段。各字段的详细含义如下表所示。

```yaml
spec:
  process:
    - type: "flex_smooth_quant"
      alpha: 0.8
      beta: 0.7
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
| type | 处理器类型标识 | 固定为 `"flex_smooth_quant"`。 |
| alpha | 激活缩放权重系数 | 0~1 之间的浮点数，控制激活对缩放因子的影响程度，默认 `None`（自动搜索）。 |
| beta | 权重缩放权重系数 | 0~1 之间的浮点数，控制权重对缩放因子的影响程度，默认 `None`（自动搜索）。 |
| enable_subgraph_type | 开启的子图类型 | 支持 `norm-linear`、`linear-linear`、`ov`、`up-down`。 |
| include | 包含的层 | 字符串列表，支持通配符匹配。 |
| exclude | 排除的层 | 字符串列表，支持通配符匹配。 |

### 4. 模型适配接口

模型适配需实现 `FlexSmoothQuantInterface` 接口的 `get_adapter_config_for_subgraph()` 方法，返回 `List[AdapterConfig]`（含 `norm-linear`、`linear-linear`、`ov`、`up-down` 等子图映射）。参考实现：`msmodelslim/model/qwen3/model_adapter.py`。

---

## 6. 适用场景与限制

### 1. 适用场景

- 需要自动搜索最优平滑参数、减少人工调参的量化场景。
- 需要同时对注意力 `ov`、MLP `up-down`、连续线性层等多种结构做离群值抑制的场景。

### 2. 使用限制

- 模型必须实现 `FlexSmoothQuantInterface` 接口并正确配置子图映射。
- 目标模块必须存在且具备可写的 `weight`，模块名须与 `named_modules()` 返回的完整路径一致。
- 非融合子图不支持 `shift`（偏置平移）。

---

## 7. 关联流程

- 《[一键量化 (V1)](../../../user_guide/usage_quick_quantization.md)》：默认集成本算法作为离群值抑制前置步骤。
- 《[量化精度调优指南](../../../user_guide/process_quantization_precision_tuning.md)》：精度不达标时可考虑启用本算法。

---

## 8. 关联词条

- [SmoothQuant](../smooth_quant/term_smooth_quant.md)：上位概念，本算法是 SmoothQuant 的灵活扩展。
- [Iterative Smooth](../iterative_smooth/term_iterative_smooth.md)：同类算法，同样支持多子图类型的平滑。
- [Flex AWQ SSZ](../flex_awq_ssz/term_flex_awq_ssz.md)：同类算法，使用真实量化器评估参数，`beta` 固定为 `0`。
- [AWQ](../awq_smooth/term_awq_smooth.md)：对比算法，基于激活均值识别重要通道并搜索缩放因子。
- [AutoRound](../autoround/term_autoround.md)：配套术语，低比特量化前常配合本算法使用。

---

## 9. 参考资料

1. Xiao G et al. SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models. ICML 2023. https://arxiv.org/abs/2211.10438
2. 《Flex Smooth Quant 使用指南》([./usage_flex_smooth_quant.md](./usage_flex_smooth_quant.md))
