# Adapt Rotation 自适应旋转优化算法词条

> **词条类别**：离群值抑制算法
> **英文名称**：Adapt Rotation
> **英文缩写**：AdaptRotation
> **应用领域**：大语言模型量化压缩、低比特量化精度优化
> **msModelSlim 实现**：`msmodelslim/processor/adapt_rotation/`

---

## 1. 概述

Adapt Rotation（自适应旋转优化）是一种用于大语言模型量化的离群值抑制算法，属于 [QuaRot](../quarot/term_quarot.md) 的扩展。它以校准数据驱动的方式，在固定 Hadamard 矩阵的基础上通过迭代优化学习正交旋转矩阵，使变换后的激活值在量化时具有更小的重构误差，从而进一步抑制激活离群值、提升低比特量化精度。其核心特征是：正交变换保证计算等价、数据驱动优化、采用两阶段（Stage1 优化 / Stage2 应用）流程。

---

## 2. 词条介绍

[QuaRot](../quarot/term_quarot.md) 使用固定的 Hadamard 矩阵对权重与激活施加正交旋转以均衡各通道数值范围，但固定的 Hadamard 矩阵未必与特定模型的激活分布最匹配。Adapt Rotation 观察到，若旋转矩阵能针对校准数据迭代优化，则变换后的激活在量化-反量化后的重构误差更小。因此它在 QuaRot 基础上引入数据驱动的旋转优化，为解决固定旋转矩阵与目标模型激活分布不匹配的问题提供了更优选择。

---

## 3. 原理

### 1. 核心思想

Adapt Rotation 的核心思想是“用数据驱动的方式优化正交旋转”：给定初始 Hadamard 矩阵 $H$ 与校准激活数据，通过迭代优化学习一个可优化的正交矩阵 $R$，使变换后的旋转矩阵 $H_{\text{adapted}} = H \cdot R$ 在给定激活数据上的量化-反量化重构误差最小。由于 $R$ 为正交矩阵，变换前后模型计算等价。

### 2. 数学描述

设初始 Hadamard 矩阵为 $H$，学习得到的正交旋转为 $R$，则变换后的旋转矩阵为：

$$
H_{\text{adapted}} = H \cdot R
$$

其中 $R$ 为正交矩阵（满足 $R^T R = I$），通过 Newton-Schulz 迭代对 $A^T B$ 求正交极因子得到，累积旋转记为 $R_{\text{acc}} = R_{\text{acc}} \cdot R_{\text{step}}$。优化目标为最小化变换后激活值经 per-token 对称量化-反量化后的重构损失。

- $H$：初始 Hadamard 矩阵
- $R$：迭代学习得到的正交旋转矩阵
- $H_{\text{adapted}}$：优化后的旋转矩阵，满足正交性，保持计算等价
- $A$、$B$：Newton-Schulz 迭代中用于求解正交极因子的矩阵
- $R_{\text{acc}}$：累积旋转矩阵

### 3. 关键性质

- **计算等价性**：正交旋转不改变矩阵乘法的数学结果，不引入额外推理误差。
- **数据驱动优化**：旋转矩阵针对校准数据迭代优化，而非固定 Hadamard。
- **两阶段流程**：Stage1 优化旋转矩阵，Stage2 将优化结果应用到 QuaRot 流程。
- **适配器依赖**：依赖 `AdaptRotationInterface`（继承 `QuaRotInterface` 并实现 `get_hidden_dim()`）。

---

## 4. 流程示意

> 以下为本算法在 msModelSlim 中的简化流程概览。

```mermaid
flowchart LR
    A[校准数据] --> B[Stage1 收集激活]
    B --> C[Hadamard 优化求正交极因子]
    C --> D[写入 Context 传递旋转矩阵]
    D --> E[Stage2 应用优化旋转]
    E --> F[层融合与旋转]
```

---

## 5. 在 msModelSlim 中的实现

### 1. 实现位置

算法在 `msmodelslim/processor/adapt_rotation/` 目录下实现，核心类包括：`AdaptRotationProcessor`（顶层处理器，按 `stage` 分发）、`AdaptRotationStage1Processor`（收集激活并优化旋转矩阵）、`AdaptRotationStage2Processor`（继承 `QuaRotProcessor`，应用优化后的旋转矩阵）与 `HadamardOptimizer`（迭代优化 Hadamard 旋转以求正交极因子）。

### 2. 处理流程

- **Stage1（prior 阶段）**：从适配器获取 LayerNorm 与 Linear 融合映射，创建初始 Hadamard 旋转矩阵并执行融合；为匹配 `layer_type` 的 Linear 层注册前向钩子收集激活；按 `max_samples` 采样后运行 Hadamard 优化得到优化后的旋转矩阵，写入 Context 供 Stage2 使用。
- **Stage2（主阶段）**：从 Context 读取 Stage1 得到的优化旋转矩阵，覆盖 QuaRot 中对应维度的旋转矩阵（替代默认 Hadamard），复用 QuaRot 的层融合与旋转流程逐层执行。

### 3. 配置示例

> 以下为多阶段 YAML 配置片段，Stage1 置于 `spec.prior`、Stage2 置于 `spec.process`。

```yaml
spec:
  prior:
    - process:
        - type: "adapt_rotation"
          stage: 1
          layer_type: ["up_proj"]
          steps: 20
          quant_dtype: "int4"
          block_size: -1
          max_samples: 2048
      dataset: boolq.jsonl

  process:
    - type: "adapt_rotation"
      stage: 2
      online: False
      block_size: -1
      max_tp_size: 1
```

**字段说明**：

| 字段名 | 作用 | 说明 |
| --- | --- | --- |
| type | 处理器类型标识 | 固定为 `"adapt_rotation"`。 |
| stage | 阶段标识 | Stage1 固定为 `1`，Stage2 固定为 `2`。 |
| steps | 迭代优化步数 | Hadamard 优化最大迭代次数，默认 `20`。 |
| quant_dtype | 量化激活类型 | `"int4"` 或 `"int8"`，应与下游量化的 `act.dtype` 一致。 |
| layer_type | 收集激活的层名子串 | 用于匹配 Linear 层名称，如 `["up_proj"]`。 |
| block_size | 旋转矩阵块大小 | 大于 0 的 2 的幂或 `-1`（表示 hidden_dim，不分块）。 |
| max_samples | 每层最大采样数 | 控制激活采样数量，默认 `2048`。 |
| online | 在线旋转开关 | 仅 Stage2 生效，`True` 时在量化过程动态注入旋转计算。 |
| max_tp_size | 最大张量并行度 | 仅 Stage2 在线旋转时生效，需为 `1` 或正的 2 的幂。 |

### 4. 模型适配接口

模型适配需实现 `AdaptRotationInterface`（继承 `QuaRotInterface`，并额外实现 `get_hidden_dim()`）：

- **Stage1**：生成并优化 `hidden_dim` 维度的旋转矩阵，写入 `ctx["adapt_rotation"].state["adapted_matrix"]`。
- **Stage2**：复用 QuaRot 的融合/旋转流程，从 Context 读取该矩阵覆盖对应维度的默认旋转。
- **在线旋转**：仅在 Stage2 配置 `online: True` 时需额外实现 `LAOSOnlineRotationInterface`。
- **通用适配**：与 `QuaRotInterface` 相关的通用适配步骤参考《[QuaRot 词条](../quarot/term_quarot.md)》。

---

## 6. 适用场景与限制

### 1. 适用场景

- 需要比固定 Hadamard 旋转更优的离群值抑制效果、低比特量化的场景。
- 作为 [AutoRound](../autoround/term_autoround.md) 等低比特量化算法的前置离群值抑制步骤。

### 2. 使用限制

- 模型必须实现 `AdaptRotationInterface`（依赖 `QuaRotInterface` 并实现 `get_hidden_dim()`）。
- Stage1 必须在 ContextManager 下运行，以便将 `adapted_matrix` 传递给 Stage2。
- Stage1 的 `quant_dtype` 应与下游量化（如 `linear_quant`/`autoround_quant`）的激活值类型一致（w4a4 用 `int4`，w8a8 用 `int8`）。
- MoE 模型若 `layer_type` 匹配范围落入专家非共享线性层，激活收集与优化耗时会显著增加。

---

## 7. 关联流程

- 《[一键量化 (V1)](../../../user_guide/usage_quick_quantization.md)》：可集成离群值抑制前置步骤。
- 《[量化精度调优指南](../../../user_guide/process_quantization_precision_tuning.md)》：精度不达标时可考虑启用本算法。

---

## 8. 关联词条

- [QuaRot](../quarot/term_quarot.md)：上位概念，本算法在 QuaRot 基础上优化旋转矩阵。
- [SmoothQuant](../smooth_quant/term_smooth_quant.md)：对比算法，采用通道级缩放而非正交旋转抑制离群值。
- [AutoRound](../autoround/term_autoround.md)：配套术语，常在本算法处理后接 AutoRound 权重量化。
- [MinMax](../minmax/term_minmax.md)：应用对象，旋转后的激活值更易于 MinMax 量化。

---

## 9. 参考资料

1. 《Adapt Rotation 使用指南》([./usage_adapt_rotation.md](./usage_adapt_rotation.md))
