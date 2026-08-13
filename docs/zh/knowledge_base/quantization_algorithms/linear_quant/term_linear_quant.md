# 线性量化算法词条

> **词条类别**：量化算法
> **英文名称**：Linear Quant
> **英文缩写**：LQ
> **应用领域**：深度学习模型压缩、大语言模型量化
> **msModelSlim 实现**：`msmodelslim/processor/quant/linear.py`

---

## 1. 概述

线性量化（Linear Quantization）是深度学习模型压缩中最基础且广泛应用的算法类别。它将连续的浮点数值范围映射到离散的数值集合中，通过缩放因子（Scale）与零点（Zero-point）实现浮点与整数的相互转换。在 msModelSlim 中，线性量化通过 `linear_quant` 处理器实现，支持对线性层的权重和激活进行灵活配置，并可按 `method` 接入 [MinMax](../minmax/term_minmax.md)、[Histogram](../histogram_activation_quantization/term_histogram_activation_quantization.md)、[SSZ](../ssz/term_ssz.md)、[GPTQ](../gptq/term_gptq.md) 等量化算法。

---

## 2. 词条介绍

模型推理与部署对显存、带宽与算力有极高要求，浮点权重的存储与计算开销成为瓶颈。线性量化通过将浮点数值映射到低比特整数，显著降低模型体积与计算成本。按量化参数统计与计算时机，可分为静态量化、动态量化以及混合量化（如 [PDMIX](../pdmix/term_pdmix.md)），适配不同的精度与性能需求。

---

## 3. 原理

### 1. 核心思想

线性量化的核心思想是“以线性映射逼近浮点分布”：用缩放因子 $S$ 与零点 $Z$ 将浮点值 $V$ 映射到量化区间 $[Q_{\min}, Q_{\max}]$，量化后的整数再乘以 $S$ 加 $Z$ 即可反量化回浮点。通过合理选择 $S$ 与 $Z$（如 MinMax 统计、直方图截断、迭代搜索等），可以控制量化误差。

### 2. 数学描述

量化公式：

$$
Q = \operatorname{clamp}\left(\operatorname{round}\left(\frac{V}{S}\right) + Z, \; Q_{\min}, Q_{\max}\right)
$$

反量化公式：

$$
V \approx S \times (Q - Z)
$$

- $V$：原始浮点值
- $S$：缩放因子，决定量化步长
- $Z$：零点偏移，用于处理非对称分布
- $Q$：量化后的整数
- $Q_{\min}$、$Q_{\max}$：量化数值范围（如 INT8 的 $[-128, 127]$）

对称量化时 $Z = 0$；非对称量化时 $Z$ 可调整。按统计粒度可分为 `per_tensor`（整个张量共用参数）、`per_channel`（每通道独立参数）、`per_group`（每分组独立参数）等。

### 3. 关键性质

- **基础通用**：适用于权重与激活的量化，是大多数量化方案的基础。
- **粒度可调**：支持 `per_tensor`、`per_channel`、`per_group` 等量化粒度。
- **静态与动态**：支持静态量化（离线固定参数）与动态量化（在线计算参数）。
- **算法可扩展**：通过 `method` 字段可接入 MinMax、Histogram、SSZ、GPTQ 等量化算法。
- **层过滤**：通过 `include`/`exclude` 通配符灵活控制量化层范围。

---

## 4. 流程示意

> 以下为本算法在 msModelSlim 中的简化流程概览。

```mermaid
flowchart LR
    A[校准数据] --> B[统计分布]
    B --> C[计算量化参数]
    C --> D[量化-反量化]
    D --> E[部署量化权重]
```

---

## 5. 在 msModelSlim 中的实现

### 1. 实现位置

算法在 `msmodelslim/processor/quant/linear.py` 中实现，通过 `type: "linear_quant"` 处理器使用，量化器实现位于 `msmodelslim/core/quantizer/`。

### 2. 处理流程

`linear_quant` 处理器对模型中的 `nn.Linear` 模块进行量化：根据 `qconfig` 配置权重与激活的量化方式（`scope`、`dtype`、`symmetric`、`method`），按 `include`/`exclude` 过滤层范围，对目标层执行量化-反量化并生成量化 IR。`qconfig.act.scope` 为 `per_tensor`/`per_token`/`pd_mix` 时分别对应静态/动态/PDMIX 混合量化。

### 3. 配置示例

> 以下为最小可用的 YAML 配置片段。各字段的详细含义如下表所示。

```yaml
spec:
  process:
    - type: "linear_quant"
      qconfig:
        act:
          scope: "per_token"
          dtype: "int8"
          symmetric: false
          method: "minmax"
        weight:
          scope: "per_channel"
          dtype: "int8"
          symmetric: true
          method: "minmax"
      include: ["*"]
      exclude: ["*down_proj*"]
```

**字段说明**：

| 字段名 | 作用 | 说明 |
| --- | --- | --- |
| type | 处理器类型标识 | 固定为 `"linear_quant"`。 |
| qconfig.act.scope | 激活量化范围 | `"per_tensor"`（静态）、`"per_token"`（动态）、`"pd_mix"`（PDMIX 混合）。 |
| qconfig.act.dtype | 激活量化数据类型 | `"int8"`、`"int4"`、`"float"`（16位浮点激活）。 |
| qconfig.act.symmetric | 激活是否对称量化 | `true` 为对称，`false` 为非对称。 |
| qconfig.act.method | 激活量化方法 | `"minmax"` 或 `"histogram"`。 |
| qconfig.weight.scope | 权重量化范围 | `"per_tensor"`、`"per_channel"`、`"per_group"`。 |
| qconfig.weight.dtype | 权重量化数据类型 | `"int8"` 或 `"int4"`。 |
| qconfig.weight.symmetric | 权重是否对称量化 | `true` 为对称，`false` 为非对称。 |
| qconfig.weight.method | 权重量化方法 | `"minmax"`、`"ssz"`、`"gptq"`。 |
| include | 包含的层 | 字符串列表，支持通配符匹配。 |
| exclude | 排除的层 | 字符串列表，支持通配符匹配，优先级高于 `include`。 |

---

## 6. 适用场景与限制

### 1. 适用场景

- 基础量化场景，需要对线性层权重与激活进行量化的模型部署。
- 作为 PDMIX、Histogram、SSZ、GPTQ 等量化算法的宿主处理器。

### 2. 使用限制

- 仅处理 `nn.Linear` 模块，其他自定义模块暂不支持。
- 并非所有配置组合都是有效组合，无效组合会抛出 `UnsupportedError` 异常。
- `include`/`exclude` 未匹配到任何层时工具会告警，需关注匹配结果。

---

## 7. 关联流程

- 《[一键量化 (V1)](../../../user_guide/usage_quick_quantization.md)》：默认集成线性量化作为基础量化步骤。
- 《[量化精度调优指南](../../../user_guide/process_quantization_precision_tuning.md)》：精度不达标时可调整线性量化配置。

---

## 8. 关联词条

- [MinMax](../minmax/term_minmax.md)：应用对象，线性量化最基础的量化方法。
- [Histogram](../histogram_activation_quantization/term_histogram_activation_quantization.md)：应用对象，线性量化的激活值量化方法。
- [SSZ](../ssz/term_ssz.md)：应用对象，线性量化的权重量化方法。
- [GPTQ](../gptq/term_gptq.md)：应用对象，线性量化的高精度权重量化方法。
- [PDMIX](../pdmix/term_pdmix.md)：应用对象，线性量化的激活值混合量化方法。

---

## 9. 参考资料

1. Jacob B et al. Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference. CVPR 2018. https://arxiv.org/abs/1712.05877
2. 《线性量化 使用指南》([./usage_linear_quant.md](./usage_linear_quant.md))
