# SmoothQuant 平滑量化算法词条

> **词条类别**：离群值抑制算法
> **英文名称**：SmoothQuant
> **首次提出**：Xiao et al., ICML 2023
> **应用领域**：大语言模型量化压缩、推理加速
> **msModelSlim 实现**：`msmodelslim/processor/anti_outlier/smooth_quant/`

---

## 1. 概述

SmoothQuant 是一种用于大语言模型量化过程中抑制激活离群值的算法。它通过数学等价变换，在归一化层与线性层之间协同缩放激活与权重，将激活值中的离群值“平滑”到权重中，使激活分布更均匀、更易于量化。其核心特征是：利用等价变换不改变模型输出、逐通道缩放以适配各通道差异、缩放因子可配置（`alpha`、`symmetric`）。与 [QuaRot](../quarot/term_quarot.md) 等旋转类算法不同，SmoothQuant 采用通道级缩放而非正交旋转。

---

## 2. 词条介绍

大语言模型的激活值中普遍存在离群值：少数通道的数值远大于其他通道，若直接量化，为覆盖这些离群值会拉大量化范围，导致大部分通道的有效比特不足、精度严重下降。SmoothQuant 观察到，若对激活和权重做协同缩放，可以在数学上保持线性层输出不变，同时把激活中的离群值迁移到权重中，从而在不改变模型输出的前提下降低激活的量化难度。

---

## 3. 原理

### 1. 核心思想

SmoothQuant 的核心思想是“以等价变换换取均匀分布”：把激活值除以一个逐通道的平滑缩放因子 $s$，同时把权重乘以 $s$，使得变换前后 $XW$ 的输出不变，但激活值的分布被拉平。缩放因子的取值由激活与权重的统计信息联合决定，并通过 $alpha$ 平衡参数控制激活与权重两侧的相对“牺牲”程度。

### 2. 数学描述

设激活矩阵为 $X$、权重矩阵为 $W$、逐通道平滑缩放因子为 $s$，则：

$$
Y = XW = (X \cdot \operatorname{diag}(s)^{-1}) \cdot (\operatorname{diag}(s) \cdot W) = \hat{X} \cdot \hat{W}
$$

其中 $X \cdot \operatorname{diag}(s)^{-1}$ 为平滑后的激活，$\operatorname{diag}(s) \cdot W$ 为吸收离群值后的权重。

缩放因子的计算公式为：

$$
s = \left( \frac{A_{\text{scale}}^{\alpha}}{W_{\text{scale}}^{1-\alpha}} \right) \cdot \operatorname{clamp}(\min=10^{-5})
$$

- $X$：激活矩阵
- $W$：权重矩阵
- $s$：逐通道平滑缩放因子
- $A_{\text{scale}}$：激活值每通道的绝对最大值
- $W_{\text{scale}}$：权重每列的绝对最大值
- $\alpha$：平衡参数，控制激活与权重的相对重要性（默认 $0.5$）
- $10^{-5}$：缩放因子的下界，防止数值不稳定

### 3. 关键性质

- **计算等价性**：变换前后 $Y = XW$ 严格不变，不引入额外推理误差。
- **离群值迁移**：激活中的离群值被迁移到权重，权重虽更难量化，但可通过 [SSZ](../ssz/term_ssz.md)、[GPTQ](../gptq/term_gptq.md) 等权重量化算法处理。
- **逐通道粒度**：缩放因子按通道独立计算，能适配不同通道的动态范围差异。
- **参数可调**：通过 $alpha$ 与 `symmetric` 可控制平滑强度与非对称偏移，适配不同模型与量化方案。

---

## 4. 流程示意

> 以下为本算法在 msModelSlim 中的简化流程概览。

```mermaid
flowchart LR
    A[校准数据] --> B[子图发现]
    B --> C[统计激活与权重]
    C --> D[计算平滑缩放]
    D --> E[融合进权重与归一化层]
    E --> F[交付量化]
```

---

## 5. 在 msModelSlim 中的实现

### 1. 实现位置

算法在 `msmodelslim/processor/anti_outlier/smooth_quant/` 目录下实现，通过 `type: "smooth_quant"` 处理器使用。

### 2. 处理流程

- **预处理阶段**：通过模型适配器的 `get_adapter_config_for_subgraph()` 获取 `norm-linear` 子图，为线性模块安装前向钩子，在 `[batch, seq, hidden_dim]` 维度上收集每通道绝对最大值与偏移量。
- **后处理阶段**：基于激活统计与权重信息计算平滑缩放因子，对归一化层做反向缩放、对线性层做正向缩放；非对称模式下同时处理偏移量，随后清理钩子并恢复模型。

### 3. 配置示例

> 以下为最小可用的 YAML 配置片段。各字段的详细含义如下表所示。

```yaml
spec:
  process:
    - type: "smooth_quant"
      alpha: 0.5
      symmetric: True
      include: ["*"]
      exclude: ["*self_attn*"]
```

**字段说明**：

| 字段名 | 作用 | 说明 |
| --- | --- | --- |
| type | 处理器类型标识 | 固定为 `"smooth_quant"`。 |
| alpha | 平衡参数 | 0~1 之间的浮点数，控制激活和权重的相对重要性，默认 `0.5`。 |
| symmetric | 是否对称量化 | 布尔值，`True` 为对称，`False` 为非对称，默认 `True`。 |
| include | 包含的层 | 字符串列表，支持通配符匹配，默认为 `["*"]`（全量）。 |
| exclude | 排除的层 | 字符串列表，支持通配符匹配，默认为空。 |

### 4. 模型适配接口

模型适配需实现 `SmoothQuantInterface` 接口的 `get_adapter_config_for_subgraph()` 方法，返回 `List[AdapterConfig]`（`subgraph_type="norm-linear"` 及 `mapping` 模块映射）。参考实现：`msmodelslim/model/qwen3/model_adapter.py` 中的 `Qwen3ModelAdapter`。

---

## 6. 适用场景与限制

### 1. 适用场景

- 激活值存在显著离群值、直接量化精度损失较大的大语言模型量化场景。
- 作为 W8A8、W4A8 等量化方案的前置离群值抑制步骤，为 [MinMax](../minmax/term_minmax.md) 等激活量化器创造更均匀的数值分布。

### 2. 使用限制

- 模型必须实现 `SmoothQuantInterface` 接口并提供 `norm-linear` 子图映射，非标准结构需谨慎评估。
- 仅支持 `norm-linear` 子图类型，不支持 `ov`、`up-down`、`linear-linear` 等其他子图。
- 目标模块必须存在且具备可写的 `weight`（及可选 `bias`），模块名须与 `named_modules()` 返回的完整路径一致。

---

## 7. 关联流程

- 《[一键量化 (V1)](../../../user_guide/usage_quick_quantization.md)》：默认集成本算法作为离群值抑制前置步骤。
- 《[量化精度调优指南](../../../user_guide/process_quantization_precision_tuning.md)》：精度不达标时可考虑启用本算法。

---

## 8. 关联词条

- [Iterative Smooth](../iterative_smooth/term_iterative_smooth.md)：同类算法，本算法的迭代扩展，支持更多子图类型。
- [Flex Smooth Quant](../flex_smooth_quant/term_flex_smooth_quant.md)：同类算法，通过二阶段网格搜索自动寻找最优 `alpha`/`beta`。
- [AWQ](../awq_smooth/term_awq_smooth.md)：同类算法，基于激活均值识别重要通道并搜索缩放因子。
- [QuaRot](../quarot/term_quarot.md)：对比算法，采用正交旋转而非通道缩放抑制离群值。
- [MinMax](../minmax/term_minmax.md)：应用对象，平滑后的激活更易于 MinMax 量化。

---

## 9. 参考资料

1. Xiao G et al. SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models. ICML 2023. https://arxiv.org/abs/2211.10438
2. 《SmoothQuant 使用指南》([./usage_smooth_quant.md](./usage_smooth_quant.md))
