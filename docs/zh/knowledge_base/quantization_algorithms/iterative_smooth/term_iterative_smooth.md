# Iterative Smooth 迭代平滑算法词条

> **词条类别**：离群值抑制算法
> **英文名称**：Iterative Smooth
> **应用领域**：大语言模型量化压缩、推理加速
> **msModelSlim 实现**：`msmodelslim/processor/anti_outlier/iter_smooth/`

---

## 1. 概述

Iterative Smooth（迭代平滑）是一种用于大语言模型量化过程中抑制激活离群值的算法。它在 [SmoothQuant](../smooth_quant/term_smooth_quant.md) 的基础上扩展，通过相邻层之间重新分配量化误差、动态调整权重与激活的缩放因子，使激活分布更加均匀。其核心特征是：支持 `norm-linear`、`linear-linear`、`ov`、`up-down` 四种子图类型，并可对无前置融合层的独立线性层做非融合平滑。

---

## 2. 词条介绍

SmoothQuant 仅支持 `norm-linear` 子图，难以覆盖注意力内部的 `ov` 结构、MLP 门控的 `up-down` 结构以及连续线性层等场景。Iterative Smooth 针对这些结构提供统一的缩放框架，并通过按优先级处理子图、在相邻层之间重分配量化误差，进一步提升离群值抑制效果。

---

## 3. 原理

### 1. 核心思想

Iterative Smooth 的核心思想是“按子图类型分别对相邻两层做协同缩放”：对前置层做反向缩放（除以缩放因子）、对后续层做正向缩放（乘以缩放因子），从而把前置层输出的离群值迁移到后续层权重中。与 SmoothQuant 相比，它支持更多子图结构，并可在 `source` 为 `None` 时对若干独立线性层做非融合平滑。

### 2. 数学描述

缩放因子计算公式：

$$
s = \left( \frac{A_{\text{scale}}^{\alpha}}{W_{\text{scale}}^{1-\alpha}} \right) \cdot \operatorname{clamp}(\min=\text{scale\_min})
$$

- $s$：逐通道缩放因子
- $A_{\text{scale}}$：激活值的每通道缩放统计量
- $W_{\text{scale}}$：权重的每列最大值
- $\alpha$：平衡参数，控制激活与权重的相对重要性（默认 $0.9$）
- $\text{scale\_min}$：缩放因子最小值（默认 $10^{-5}$）

对于 `linear-linear`、`ov`、`up-down` 子图，以目标层（`linear2`/`o_proj`/`down_proj`）的权重计算缩放因子，对其做正向缩放，对前置层（`linear1`/`v_proj`/`up_proj`）做反向缩放。

### 3. 关键性质

- **等价变换**：对相邻两层的协同缩放保持整体计算等价，不改变模型输出。
- **子图类型多样**：支持 `norm-linear`、`linear-linear`、`ov`、`up-down` 四种子图。
- **非融合能力**：`source=None` 时对独立线性层做输入侧 pre-hook 缩放，不依赖前置层。
- **优先级处理**：按 `up-down` → `ov` → `norm-linear` → `linear-linear` 的优先级顺序处理子图。
- **分布式友好**：支持分布式训练环境下的统计信息聚合。

---

## 4. 流程示意

> 以下为本算法在 msModelSlim 中的简化流程概览。

```mermaid
flowchart LR
    A[校准数据] --> B[子图发现]
    B --> C[统计激活]
    C --> D[按优先级平滑]
    D --> E[融合缩放]
    E --> F[交付量化]
```

---

## 5. 在 msModelSlim 中的实现

### 1. 实现位置

算法在 `msmodelslim/processor/anti_outlier/iter_smooth/processor.py` 中实现，通过 `type: "iter_smooth"` 处理器使用。

### 2. 处理流程

- **预处理阶段**：通过 `SubgraphProcessor` 获取四种类型的子图，按 `include/exclude` 过滤，为线性模块安装前向钩子收集激活统计信息。
- **后处理阶段**：按优先级顺序处理各子图，对每种子图应用相应的平滑方法；非融合子图（`source=None`）对权重做缩放并在每层注册输入侧 pre-hook；最后清理钩子并恢复模型。

### 3. 配置示例

> 以下为最小可用的 YAML 配置片段。各字段的详细含义如下表所示。

```yaml
spec:
  process:
    - type: "iter_smooth"
      alpha: 0.9
      scale_min: 1e-5
      symmetric: True
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
| type | 处理器类型标识 | 固定为 `"iter_smooth"`。 |
| alpha | 平衡参数 | 大于 0 的浮点数，控制激活和权重的相对重要性，默认 `0.9`。 |
| scale_min | 缩放因子最小值 | 大于 0 的浮点数，防止数值不稳定，默认 `1e-5`。 |
| symmetric | 是否对称量化 | 布尔值，`True` 为对称，`False` 为非对称，默认 `True`。 |
| enable_subgraph_type | 开启的子图类型 | 支持 `norm-linear`、`linear-linear`、`ov`、`up-down`。 |
| include | 包含的层 | 字符串列表，支持通配符匹配。 |
| exclude | 排除的层 | 字符串列表，支持通配符匹配。 |

### 4. 模型适配接口

模型适配需实现 `IterSmoothInterface` 接口的 `get_adapter_config_for_subgraph()` 方法，返回 `List[AdapterConfig]`（含 `norm-linear`、`linear-linear`、`ov`、`up-down` 等子图映射，`source` 可为 `None` 表示非融合子图）。参考实现：`msmodelslim/model/qwen3/model_adapter.py`。

---

## 6. 适用场景与限制

### 1. 适用场景

- 需要同时对注意力 `ov`、MLP `up-down`、连续线性层等多种结构做离群值抑制的场景。
- 作为 W8A8、W4A8 等量化方案的前置离群值抑制步骤，为 [MinMax](../minmax/term_minmax.md) 等激活量化器创造更均匀的数值分布。

### 2. 使用限制

- 模型必须实现 `IterSmoothInterface` 接口并正确配置子图映射。
- 目标模块必须存在且具备可写的 `weight`（及可选 `bias`），模块名须与 `named_modules()` 返回的完整路径一致。
- 非融合子图不支持 `shift`（偏置平移），若配置了 shift 会被忽略并打日志提示。

---

## 7. 关联流程

- 《[一键量化 (V1)](../../../user_guide/usage_quick_quantization.md)》：默认集成本算法作为离群值抑制前置步骤。
- 《[量化精度调优指南](../../../user_guide/process_quantization_precision_tuning.md)》：精度不达标时可考虑启用本算法。

---

## 8. 关联词条

- [SmoothQuant](../smooth_quant/term_smooth_quant.md)：上位概念，本算法是 SmoothQuant 的迭代扩展。
- [Flex Smooth Quant](../flex_smooth_quant/term_flex_smooth_quant.md)：同类算法，通过二阶段网格搜索自动寻找最优 `alpha`/`beta`。
- [SVDQuant](../svdquant/term_svdquant.md)：配套术语，SVDQuant 流水线使用本算法完成离群值迁移。
- [AutoRound](../autoround/term_autoround.md)：配套术语，低比特量化前常配合本算法使用。
- [QuaRot](../quarot/term_quarot.md)：对比算法，采用正交旋转而非通道缩放抑制离群值。

---

## 9. 参考资料

1. Xiao G et al. SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models. ICML 2023. https://arxiv.org/abs/2211.10438
2. 《Iterative Smooth 使用指南》([./usage_iterative_smooth.md](./usage_iterative_smooth.md))
