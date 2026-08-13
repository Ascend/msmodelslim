# LAOS 低比特量化方案词条

> **词条类别**：量化算法
> **英文名称**：LAOS
> **英文缩写**：LAOS
> **应用领域**：大语言模型量化压缩、低比特量化精度优化
> **msModelSlim 实现**：通过 `adapt_rotation` 与 `autoround_quant` 处理器组合实现

---

## 1. 概述

LAOS 是一种面向 W4A4 超低比特量化场景的量化方案，核心思想是“旋转矩阵优化 + 基于舍入偏移参数训练的低比特量化”。它先通过 [Adapt Rotation](../adapt_rotation/term_adapt_rotation.md) 在模型上进行旋转矩阵优化（分阶段执行）实现有效的离群值抑制，再通过 [AutoRound](../autoround/term_autoround.md) 进行低比特量化与舍入偏移参数优化，从而提升 W4A4 场景下的精度与稳定性。

---

## 2. 词条介绍

在低比特量化（如 W4A4）场景下，模型精度损失尤为显著，核心难点在于权重和激活值中的极端离群值会显著扭曲量化区间，导致数值表示精度急剧下降，传统方法难以解决。LAOS 将离群值抑制与低比特量化训练相结合，先用数据驱动的方式优化旋转矩阵抑制离群值，再用可学习舍入优化量化参数，从而在 W4A4 下获得更高的精度与稳定性。

---

## 3. 原理

### 1. 核心思想

LAOS 的核心思想是“先抑制离群值、再训练量化参数”的两段式方案：Stage1 使用 [Adapt Rotation](../adapt_rotation/term_adapt_rotation.md) 基于校准数据迭代优化 Hadamard 旋转矩阵，使变换后的激活值更利于量化；Stage2 使用 [AutoRound](../autoround/term_autoround.md) 引入可学习的舍入偏移参数，通过 SignSGD 优化器自适应调整各权重的舍入方向，最小化量化重构误差。

### 2. 数学描述

LAOS 方案由两个阶段构成。第一阶段基于旋转优化：

$$
H_{\text{adapted}} = H \cdot R
$$

- $H$：初始 Hadamard 矩阵
- $R$：数据驱动优化得到的正交旋转矩阵
- $H_{\text{adapted}}$：优化后的旋转矩阵

第二阶段基于可学习舍入的权重量化：

$$
\hat{W} = s \times \operatorname{clip}(\lfloor W/s + zp + V \rceil, n, m)
$$

- $\hat{W}$：量化后的权重
- $s$：缩放因子
- $zp$：零点
- $V$：控制舍入方向的可学习偏移
- $n$、$m$：量化后的上下界

### 3. 关键性质

- **两段式组合**：旋转优化（Adapt Rotation）与舍入训练（AutoRound）协同提升 W4A4 精度。
- **数据驱动旋转**：旋转矩阵基于校准数据迭代优化，比固定 Hadamard 更适配模型分布。
- **可学习舍入**：舍入方向由可学习偏移决定，降低量化重构误差。
- **混合量化支持**：支持对敏感层使用 W8A8、其余层使用 W4A4 的混合策略。

---

## 4. 流程示意

> 以下为本算法在 msModelSlim 中的简化流程概览。

```mermaid
flowchart LR
    A[校准数据] --> B[Stage1 旋转优化]
    B --> C[Stage2 应用旋转]
    C --> D[AutoRound 量化]
    D --> E[混合量化部署]
```

---

## 5. 在 msModelSlim 中的实现

### 1. 实现位置

LAOS 方案通过 `adapt_rotation`（Stage1 + Stage2）与 `autoround_quant` 两个处理器组合实现，配置中需在 `spec.prior` 中配置 Stage1、在 `spec.process` 中配置 Stage2 与 `autoround_quant`。

### 2. 处理流程

- **prior 阶段**：配置 `adapt_rotation` Stage1，收集指定层（如 `up_proj`）的激活并优化旋转矩阵，写入 Context。
- **主阶段**：配置 `adapt_rotation` Stage2 应用优化后的旋转矩阵；随后配置 `autoround_quant` 执行低比特量化与舍入优化；最后配置 `ascendv1_saver` 保存量化权重。

### 3. 配置示例

> 以下为最小可用的 YAML 配置片段。各字段的详细含义如下表所示。

```yaml
spec:
  prior:
    - process:
        - type: "adapt_rotation"
          stage: 1
          steps: 20
          layer_type: ["up_proj"]
  process:
    - type: "adapt_rotation"
      stage: 2
      online: false
      block_size: -1
      max_tp_size: 1
    - type: "autoround_quant"
      iters: 400
      enable_round_tuning: true
      strategies:
        - qconfig: *default_w8a8_dynamic
          include: ["*self_attn*", "*.down_proj"]
        - qconfig: *default_w4a4_dynamic
          include: ["*.up_proj", "*.gate_proj"]
  save:
    - type: "ascendv1_saver"
      part_file_size: 4
  dataset: laos_calib.jsonl
```

**字段说明**：

| 字段名 | 作用 | 说明 |
| --- | --- | --- |
| adapt_rotation（Stage1） | 旋转矩阵优化 | `type: "adapt_rotation"`、`stage: 1`，配置 `steps`（默认 `20`）与 `layer_type`（如 `["up_proj"]`）。 |
| adapt_rotation（Stage2） | 应用优化旋转 | `type: "adapt_rotation"`、`stage: 2`，配置 `online`（默认 `false`）、`block_size`（默认 `-1`）、`max_tp_size`（默认 `1`）。 |
| autoround_quant | 低比特量化 | 配置 `iters`（默认 `400`）、`enable_round_tuning`（默认 `true`）及 `strategies` 混合量化策略。 |
| save | 权重保存 | `type: "ascendv1_saver"`，可配置 `part_file_size`。 |
| dataset | 校准数据集 | 指定校准数据集名称，如 `laos_calib.jsonl`。 |

### 4. 模型适配接口

模型适配要求模型支持 `AdaptRotationInterface`（继承 `QuaRotInterface` 并实现 `get_hidden_dim()`），且线性层为可量化的 `nn.Linear`；当前主要面向 Qwen3 稠密系列。

---

## 6. 适用场景与限制

### 1. 适用场景

- W4A4 等超低比特量化场景，需要保持较高模型精度的场景。
- 激活值存在显著离群值、需要数据驱动旋转抑制的场景。

### 2. 使用限制

- 需要足够的校准数据或训练迭代次数来优化参数，量化时长相对较久。
- 当前主要面向 Qwen3 稠密系列模型（如 Qwen3-8B/14B/32B），不保证可泛化到其他系列模型。
- 算法实现包含训练过程，对 NPU 显存有一定要求，仅支持 NPU 显存 ≥64G 的设备。

---

## 7. 关联流程

- 《[一键量化 (V1)](../../../user_guide/usage_quick_quantization.md)》：可集成 LAOS 作为 W4A4 低比特量化方案。
- 《[量化精度调优指南](../../../user_guide/process_quantization_precision_tuning.md)》：精度不达标时可调整 LAOS 组合配置。

---

## 8. 关联词条

- [Adapt Rotation](../adapt_rotation/term_adapt_rotation.md)：前置术语，LAOS 使用其旋转优化作为离群值抑制步骤。
- [AutoRound](../autoround/term_autoround.md)：配套术语，LAOS 使用其舍入训练完成低比特量化。
- [QuaRot](../quarot/term_quarot.md)：上位概念，Adapt Rotation 基于 QuaRot 框架。
- [DualScale](../dual_scale/term_dual_scale.md)：同类算法，同为面向 Qwen3 稠密系列的 W4A4 量化方案。

---

## 9. 参考资料

1. 《LAOS 使用指南》([./usage_laos.md](./usage_laos.md))
