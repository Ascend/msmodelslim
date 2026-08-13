# AutoRound 低比特量化算法词条

> **词条类别**：量化算法
> **英文名称**：AutoRound
> **首次提出**：Intel, 2023
> **应用领域**：大语言模型量化压缩、低比特量化精度优化
> **msModelSlim 实现**：`msmodelslim/processor/quant/autoround.py`

---

## 1. 概述

AutoRound 是一种基于 SignSGD 的大语言模型低比特权重量化算法。它通过引入可学习的舍入偏移参数，结合 SignSGD 优化器自适应调整各权重的舍入方向，并利用温度调度策略逐步硬化舍入操作，有效降低量化重构误差。其核心特征是：可学习舍入、逐层迭代优化、支持 W4A4 等超低比特量化，常与 [QuaRot](../quarot/term_quarot.md) 等离群值抑制算法配合使用。

---

## 2. 词条介绍

传统量化方法（如四舍五入）在权重量化中并非最优选择，往往会引入较大的量化误差，尤其在低比特（如 4bit 及以下）量化场景中表现更为明显。AutoRound 观察到，通过引入可学习的舍入偏移并结合优化器，可以自适应地为每个权重选择最优的舍入方向，从而显著降低量化重构误差。

---

## 3. 原理

### 1. 核心思想

AutoRound 的核心思想是“把舍入方向变成可学习参数”：不采用简单的四舍五入，而是基于 SignSGD（符号梯度下降）算法，为每个权重学习一个舍入偏移 $V$，自适应决定权重向上或向下舍入，并有针对性地调整缩放因子和零点，从而最小化量化重构误差。

### 2. 数学描述

传统量化中权重 $W$ 的量化公式为：

$$
\hat{W} = s \times \operatorname{clip}(\lfloor W/s + zp \rceil, n, m)
$$

- $\hat{W}$：量化后的权重
- $s$：缩放因子
- $zp$：零点
- $n$、$m$：量化后的上下界

AutoRound 在此基础上引入可学习的舍入偏移 $V$ 与可选的缩放因子调整参数 $\alpha$、$\beta$：

$$
\hat{W} = s \times \operatorname{clip}(\lfloor W/s + zp + V \rceil, n, m)
$$

$$
s = \frac{\max(W) \times \alpha - \min(W) \times \beta}{2^{bit} - 1}
$$

- $V$：控制舍入方向的可学习偏移
- $\alpha$、$\beta$：用于调整缩放因子范围的参数
- $bit$：量化位数

优化过程为逐层迭代：采集浮点前向输出作为基准，初始化并训练缩放因子与舍入偏移，量化-反量化前向得到量化结果，计算重构损失，通过 SignSGD 更新参数，重复直至收敛或达到最大迭代次数。

### 3. 关键性质

- **可学习舍入**：舍入方向由可学习偏移 $V$ 决定，而非固定四舍五入。
- **逐层优化**：对每个 decoder 层独立优化，避免误差跨层累积。
- **超低比特支持**：面向 4bit 及以下的超低比特量化场景。
- **混合量化**：支持对不同层使用不同量化配置（如 W8A8 与 W4A4 混合）。

---

## 4. 流程示意

> 以下为本算法在 msModelSlim 中的简化流程概览。

```mermaid
flowchart LR
    A[浮点前向] --> B[初始化量化参数]
    B --> C[量化重构]
    C --> D[计算重构损失]
    D --> E[SignSGD 更新]
    E --> F[迭代收敛]
    F --> G[应用量化权重]
```

---

## 5. 在 msModelSlim 中的实现

### 1. 实现位置

算法在 `msmodelslim/processor/quant/autoround.py` 中实现，核心类为 `AutoroundQuantProcessor`，通过 `type: "autoround_quant"` 处理器使用。

### 2. 处理流程

- **初始化阶段**：读取量化配置，为每个网络层分配对应的量化配置方案，预分配浮点输出、量化输出与最佳参数。
- **pre_run 阶段**：关闭所有网络层的自动梯度计算，防止训练过程中直接优化权重。
- **preprocess 阶段**（逐层）：采集当前层浮点前向输出作为基准，对线性层包装并注入可训练的缩放因子与舍入偏移参数。
- **process 阶段**（逐层）：配置 SignSGD 优化器，迭代更新缩放因子与舍入偏移，最小化重构误差。
- **postprocess 阶段**（逐层）：应用优化后的量化参数，解除线性层包装，执行量化后的前向传播。
- **post_run 阶段**：清理所有临时属性。

### 3. 配置示例

> 以下为最小可用的 YAML 配置片段。各字段的详细含义如下表所示。

```yaml
spec:
  process:
    - type: "autoround_quant"
      iters: 400
      enable_minmax_tuning: True
      enable_round_tuning: True
      strategies:
        - qconfig: *default_w4a4_dynamic
          include:
            - "*.up_proj"
            - "*.gate_proj"
```

**字段说明**：

| 字段名 | 作用 | 说明 |
| --- | --- | --- |
| type | 处理器类型标识 | 固定为 `"autoround_quant"`。 |
| iters | 优化迭代次数 | 大于 0 的整数，影响优化效果与计算时间，默认 `10`。 |
| enable_minmax_tuning | 最小最大值调优开关 | 布尔值，是否启用最小最大值调优，默认 `True`。 |
| enable_round_tuning | 舍入调优开关 | 布尔值，是否启用舍入调优，默认 `True`。 |
| strategies | 量化策略配置 | 策略列表，支持对不同层使用不同量化配置（如 int4 与 int8 混合量化）。 |

---

## 6. 适用场景与限制

### 1. 适用场景

- 4bit 等超低比特权重量化场景。
- 低比特条件下仍需要保持较高模型精度的场景。

### 2. 使用限制

- 仅适用于 LLM 中的线性层量化。
- 需要足够的校准数据或训练迭代次数来优化参数。
- 包含训练过程，对 NPU 显存有一定要求，仅支持 NPU 显存 ≥64G 的设备。
- 低比特量化极度依赖良好的离群值抑制算法，建议配合 [QuaRot](../quarot/term_quarot.md) 或 [Iterative Smooth](../iterative_smooth/term_iterative_smooth.md) 使用，不建议单独使用。

---

## 7. 关联流程

- 《[一键量化 (V1)](../../../user_guide/usage_quick_quantization.md)》：可集成 AutoRound 作为低比特权重量化步骤。
- 《[量化精度调优指南](../../../user_guide/process_quantization_precision_tuning.md)》：精度不达标时可调整 AutoRound 迭代次数与量化配置。

---

## 8. 关联词条

- [QuaRot](../quarot/term_quarot.md)：配套术语，常在本算法前作为离群值抑制步骤。
- [Adapt Rotation](../adapt_rotation/term_adapt_rotation.md)：配套术语，常与本算法配合用于 W4A4 量化。
- [Iterative Smooth](../iterative_smooth/term_iterative_smooth.md)：配套术语，常在本算法前作为离群值抑制步骤。
- [GPTQ](../gptq/term_gptq.md)：同类算法，同为高精度权重量化优化算法。
- [MinMax](../minmax/term_minmax.md)：对比算法，本算法是 MinMax 的低比特精度优化扩展。

---

## 9. 参考资料

1. Cheng W et al. Optimize Weight Rounding via Signed Gradient Descent for the Quantization of LLMs. 2023. https://arxiv.org/abs/2309.05516
2. 《AutoRound 使用指南》([./usage_autoround.md](./usage_autoround.md))
