# DualScale 双尺度量化算法词条

> **词条类别**：量化算法
> **英文名称**：DualScale
> **英文缩写**：DualScale
> **应用领域**：大语言模型量化压缩、低比特量化精度优化
> **msModelSlim 实现**：`msmodelslim/core/quantizer/impl/dualscale.py`

---

## 1. 概述

DualScale 是一种面向 W4A4 低比特量化场景的双尺度量化方案。它通过两级粒度递进的缩放因子，在保持硬件高效性的同时缓解激活异常通道（outlier channels）对量化精度的影响。其核心特征是：外层大块缩放 + 内层 block 量化的两级结构、权重静态量化存储、激活 data-free 伪量化，主要面向 Qwen3 稠密系列模型，作为 [线性量化](../linear_quant/term_linear_quant.md) 的量化方法使用。

---

## 2. 词条介绍

传统的分组量化通常采用单尺度（single-scale）策略：每组 K 维元素共享一个缩放因子。然而，激活值中存在结构化的异常通道——某些通道的值平均比其他通道高出数个数量级，单尺度量化难以同时精准表示这些差异巨大的数值范围，导致精度损失。DualScale 通过两级粒度递进的缩放因子，先按大块计算外层尺度吸收异常通道差异，再在内层 block 完成低比特量化。

---

## 3. 原理

### 1. 核心思想

DualScale 的核心思想是“两级缩放递进吸收分布差异”：外层按 `dual_block_size` 划分大块，计算每个大块的最大绝对值得到外层尺度 $S_{dual}$，将输入除以该尺度；内层再按 `inner_block_size` 划分，执行 mxFP4 量化-反量化。通过先缩放再量化的两级结构，异常通道的差异被外层尺度吸收。

### 2. 数学描述

激活的外层缩放（Dual Scale）：

$$
X_{\text{dualscaled}} = \frac{X}{S_{\text{dual\_x}}}, \quad S_{\text{dual\_x}} = \frac{\max(|X_{\text{block}}|)}{\text{MXFP4\_MAX\_NORMAL}}
$$

内层量化-反量化：

$$
X_{q\_dq\_inner} = \operatorname{mxfp4\_quantize\_dequantize}(X_{\text{dualscaled}}, S_{\text{inner\_x}})
$$

外层反量化：

$$
X_{q\_dq} = X_{q\_dq\_inner} \times S_{\text{dual\_x}}
$$

权重在初始化时完成静态量化存储，前向时进行两级反量化：

$$
W_{q\_dq} = W_{\text{dualscaled\_q\_dq}} \times S_{\text{dual\_w}}
$$

- $X$：激活值
- $W$：权重
- $S_{\text{dual\_x}}$、$S_{\text{dual\_w}}$：激活/权重外层尺度
- $S_{\text{inner\_x}}$、$S_{\text{inner\_w}}$：激活/权重内层尺度
- $\text{MXFP4\_MAX\_NORMAL}$：MXFP4 最大正规数，取 $6.0$

矩阵乘法：

$$
\text{Output} = X_{q\_dq} \cdot W_{q\_dq}^T + \text{bias}
$$

### 3. 关键性质

- **两级缩放**：外层大块缩放 + 内层 block 量化的递进结构。
- **异常通道吸收**：外层尺度吸收异常通道的差异，缓解单尺度量化的精度损失。
- **权重静态存储**：权重在初始化时完成量化存储，前向仅做反量化。
- **激活 data-free**：激活量化器 `is_data_free` 返回 `True`，实际量化在推理阶段由硬件完成。

---

## 4. 流程示意

> 以下为本算法在 msModelSlim 中的简化流程概览。

```mermaid
flowchart LR
    A[外层大块缩放] --> B[内层 block 量化]
    B --> C[外层反量化]
    C --> D[矩阵乘法]
```

---

## 5. 在 msModelSlim 中的实现

### 1. 实现位置

算法在 `msmodelslim/core/quantizer/impl/dualscale.py` 中实现，包括权重量化器 `MXWeightDualScaleMinmax` 与激活量化器 `MXActDualScaleMinmax`，均通过 `QABCRegistry.multi_register` 注册，dispatch_key 为 `(qir.mxfp4_dual_scale_sym, "dualscale")`。

### 2. 处理流程

- **权重量化器**（`MXWeightDualScaleMinmax`）：在权重初始化阶段完成静态量化。`init_weight` 按 `axes` 与 `dual_block_size` 将权重重塑为块状，统计外层块最大值计算外层尺度，权重除以外层尺度后交由内层量化器进行 mxFP4 量化存储。`forward` 完成内层反量化、乘以 $S_{\text{dual\_w}}$ 恢复外层尺度。
- **激活量化器**（`MXActDualScaleMinmax`）：data-free 量化，`forward` 直接返回输入 `x`（实际量化在推理阶段由硬件完成）。

### 3. 配置示例

> 以下为最小可用的 YAML 配置片段。各字段的详细含义如下表所示。

```yaml
spec:
  process:
    - type: "linear_quant"
      qconfig:
        act:
          scope: "dual_scale"
          dtype: "mxfp4"
          symmetric: True
          method: "dualscale"
          ext:
            dual_block_size: 512
        weight:
          scope: "dual_scale"
          dtype: "mxfp4"
          symmetric: True
          method: "dualscale"
          ext:
            dual_block_size: 512
```

**字段说明**：

| 字段名 | 作用 | 说明 |
| --- | --- | --- |
| scope | 量化范围 | 固定为 `"dual_scale"`（双尺度）。 |
| dtype | 量化数据类型 | 固定为 `"mxfp4"`。 |
| symmetric | 是否对称量化 | `true` 为对称，`false` 为非对称。 |
| method | 量化方法 | 固定为 `"dualscale"`。 |
| ext.dual_block_size | 外层大块大小 | 整数，如 `512`。 |

---

## 6. 适用场景与限制

### 1. 适用场景

- W4A4 等超低比特量化场景，需要保持较高模型精度的场景。
- 激活值存在结构化异常通道的模型量化场景。

### 2. 使用限制

- 需要足够的校准数据或训练迭代次数来优化参数，量化时长相对较久。
- 当前主要面向 Qwen3 稠密系列模型（如 Qwen3-8B/14B/32B），不保证可泛化到其他系列模型。
- 算法实现包含训练过程，对 NPU 显存有一定要求，仅支持 NPU 显存 ≥64G 的设备。

---

## 7. 关联流程

- 《[一键量化 (V1)](../../../user_guide/usage_quick_quantization.md)》：可集成 DualScale 作为 W4A4 低比特量化方案。
- 《[量化精度调优指南](../../../user_guide/process_quantization_precision_tuning.md)》：精度不达标时可调整 DualScale 配置。

---

## 8. 关联词条

- [线性量化](../linear_quant/term_linear_quant.md)：应用对象，DualScale 作为线性量化的量化方法（`scope: "dual_scale"`）使用。
- [LAOS](../laos/term_laos.md)：同类算法，同为面向 Qwen3 稠密系列的 W4A4 量化方案。
- [MinMax](../minmax/term_minmax.md)：对比算法，DualScale 通过两级缩放缓解 MinMax 对异常通道的敏感性。

---

## 9. 参考资料

1. 《DualScale 使用指南》([./usage_dual_scale.md](./usage_dual_scale.md))
